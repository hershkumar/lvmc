import jax
import jax.numpy as jnp
from jax import random, lax
from functools import partial

EPS = 1e-12


# ── helpers ──────────────────────────────────────────────────────────────────
def normalize(v, axis=-1, eps=EPS):
    return v / (jnp.linalg.norm(v, axis=axis, keepdims=True) + eps)


@jax.jit
def project_to_sphere(x):
    """Project cartesian vectors to S^2 sitewise."""
    return normalize(x, axis=-1)


# ── sampler ───────────────────────────────────────────────────────────────────


class Sampler:
    def __init__(self, psi, shape):
        """
        psi(params, coords) -> scalar amplitude for a single configuration.

        shape:
          - (N, 3): N sites, each a unit vector in R^3
          - or (..., 3): any leading site/batch dims, last dim must be 3

        Targets π(x) ∝ |psi(params, x)|² with MH updates on S².
        """
        self.psi = psi
        self.shape = tuple(shape)
        if self.shape[-1] != 3:
            raise ValueError(
                f"Expected last dim = 3 for cartesian coords, got shape {self.shape}"
            )

    def _mh_step(self, log_psi_fn, carry, rng_noise_log_u):
        """One MH update. carry = (pos, log_pos, acc)."""
        pos, log_pos, acc = carry
        noise, log_u = rng_noise_log_u

        prop = project_to_sphere(pos + noise)
        log_new = log_psi_fn(prop)
        accept = log_u <= 2.0 * (log_new - log_pos)

        pos_out = lax.select(accept, prop, pos)
        log_pos_out = lax.select(accept, log_new, log_pos)
        acc_out = acc + accept.astype(acc.dtype)
        return pos_out, log_pos_out, acc_out

    @partial(jax.jit, static_argnums=(0, 2, 3, 4))
    def run_chain(self, params, Nsweeps, Ntherm, keep, stepsize, pos_initial, seed):
        pos = project_to_sphere(jnp.asarray(pos_initial))
        stepsize = jnp.asarray(stepsize, dtype=pos.dtype)

        key = random.PRNGKey(seed)

        def log_psi_fn(x):
            return jnp.log(jnp.abs(self.psi(params, x)) + 1e-300)

        log_pos = log_psi_fn(pos)
        acc0 = jnp.array(0, dtype=jnp.int32)

        total_steps = Ntherm + Nsweeps * keep
        key_noise, key_u = random.split(key)
        all_noise = stepsize * random.normal(
            key_noise, shape=(total_steps,) + self.shape, dtype=pos.dtype
        )
        all_log_u = jnp.log(
            random.uniform(key_u, shape=(total_steps,), dtype=pos.dtype)
        )

        def therm_step(carry, rng_i):
            return self._mh_step(log_psi_fn, carry, rng_i), None

        (pos, log_pos, acc_therm), _ = lax.scan(
            therm_step,
            (pos, log_pos, acc0),
            xs=(all_noise[:Ntherm], all_log_u[:Ntherm]),
        )

        sample_noise = all_noise[Ntherm:]
        sample_log_u = all_log_u[Ntherm:]

        def sample_step(carry, rng_i):
            carry = self._mh_step(log_psi_fn, carry, rng_i)
            return carry, carry[0]

        (_, _, acc_sample), all_pos = lax.scan(
            sample_step,
            (pos, log_pos, acc0),
            xs=(sample_noise, sample_log_u),
        )

        samples = all_pos.reshape((Nsweeps, keep) + self.shape)[:, -1]

        total_acc = acc_therm + acc_sample
        acc_rate = total_acc / jnp.float32(total_steps)
        return samples, acc_rate

    @partial(jax.jit, static_argnums=(0, 2, 3, 4))
    def run_many_chains(
        self, params, Nsweeps, Ntherm, keep, stepsize, pos_initials, seeds
    ):
        pos_initials = jnp.asarray(pos_initials)
        seeds = jnp.asarray(seeds, dtype=jnp.uint32)

        samples_per_chain, acc_rates = jax.vmap(
            lambda pos0, sd: self.run_chain(
                params, Nsweeps, Ntherm, keep, stepsize, pos0, sd
            ),
            in_axes=(0, 0),
            out_axes=(0, 0),
        )(pos_initials, seeds)

        # (nchains, Nsweeps, *shape) → (nchains*Nsweeps, *shape)
        nchains = pos_initials.shape[0]
        samples = samples_per_chain.reshape((nchains * Nsweeps,) + self.shape)
        acc_rate = acc_rates.mean()
        return samples, acc_rate



@jax.jit
def rodrigues_rotate(v, axis, angle):
    """
    Rotate v about 'axis' by 'angle' (radians) using Rodrigues' formula.
    v:    (..., 3)
    axis: (..., 3) (need not be unit; normalized internally)
    angle: scalar or broadcastable to v[...,0]
    """
    axis = normalize(axis, axis=-1)
    c = jnp.cos(angle)
    s = jnp.sin(angle)

    # Ensure broadcasting works: make c,s have trailing singleton dim
    c = jnp.asarray(c)[(...,) + (None,)]
    s = jnp.asarray(s)[(...,) + (None,)]

    ax_dot_v = jnp.sum(axis * v, axis=-1, keepdims=True)
    ax_x_v = jnp.cross(axis, v)

    return v * c + ax_x_v * s + axis * ax_dot_v * (1.0 - c)


# ── sampler by me ───────────────────────────────────────────────────────────────────
class newSampler:
    def __init__(self, psi, shape):
        """
        psi(params, coords) -> scalar amplitude for a single configuration.
        shape: (N, 3) (or any leading dims, last dim must be 3)

        Targets π(x) ∝ |psi(params, x)|² with MH updates on S² using rotations.
        """
        self.psi = psi
        self.shape = tuple(shape)
        if self.shape[-1] != 3:
            raise ValueError(
                f"Expected last dim = 3 for cartesian coords, got shape {self.shape}"
            )

    def _mh_step(self, log_psi_fn, carry, rng_axis_logu_angle, stepsize):
        """
        One MH update with rotation proposals.
        carry = (pos, log_pos, acc)
        rng_axis_logu_angle = (axis_raw, log_u, angle)
        """
        pos, log_pos, acc = carry
        axis_raw, log_u, angle = rng_axis_logu_angle

        # axis_raw has shape self.shape; normalize to get unit axes per site
        axis = normalize(axis_raw, axis=-1)

        # angle can be scalar, (leading dims...), or full self.shape[:-1]
        # (rodrigues_rotate should broadcast over last dim=3)
        prop = rodrigues_rotate(pos, axis, angle)
        prop = project_to_sphere(prop)

        log_new = log_psi_fn(prop)

        # MH acceptance for target |psi|^2:
        accept = log_u <= 2.0 * (log_new - log_pos)

        pos_out = lax.select(accept, prop, pos)
        log_pos_out = lax.select(accept, log_new, log_pos)
        acc_out = acc + accept.astype(acc.dtype)
        return pos_out, log_pos_out, acc_out

    @partial(jax.jit, static_argnums=(0, 2, 3, 4))
    def run_chain(self, params, Nsweeps, Ntherm, keep, stepsize, pos_initial, seed):
        """
        stepsize: max rotation angle in radians (scalar). Each proposal uses
                  angle ~ Uniform(-stepsize, +stepsize).
        Proposals: rotate each n_x by an independent random angle around an
                   independent random axis per site.
        """
        pos = project_to_sphere(jnp.asarray(pos_initial))
        stepsize = jnp.asarray(stepsize, dtype=pos.dtype)

        key = random.PRNGKey(seed)

        def log_psi_fn(x):
            return jnp.log(jnp.abs(self.psi(params, x)) + 1e-300)

        log_pos = log_psi_fn(pos)
        acc0 = jnp.array(0, dtype=jnp.int32)

        total_steps = Ntherm + Nsweeps * keep

        # keys for axes, MH uniforms, and angles
        key_axis, key_u, key_ang = random.split(key, 3)

        # Random axes: Gaussian -> normalize later
        all_axis_raw = random.normal(
            key_axis, shape=(total_steps,) + self.shape, dtype=pos.dtype
        )

        # log u for MH test
        all_log_u = jnp.log(random.uniform(key_u, shape=(total_steps,), dtype=pos.dtype))

        # Independent random angle per site per step:
        # shape is (total_steps,) + self.shape[:-1]
        # (i.e., one scalar angle for each 3-vector)
        all_angle = random.uniform(
            key_ang,
            shape=(total_steps,) + self.shape[:-1],
            minval=-stepsize,
            maxval=stepsize,
            dtype=pos.dtype,
        )

        def therm_step(carry, rng_i):
            axis_raw_i, log_u_i, ang_i = rng_i
            carry = self._mh_step(log_psi_fn, carry, (axis_raw_i, log_u_i, ang_i), stepsize)
            return carry, None

        (pos, log_pos, acc_therm), _ = lax.scan(
            therm_step,
            (pos, log_pos, acc0),
            xs=(all_axis_raw[:Ntherm], all_log_u[:Ntherm], all_angle[:Ntherm]),
        )

        axis_raw_s = all_axis_raw[Ntherm:]
        log_u_s = all_log_u[Ntherm:]
        ang_s = all_angle[Ntherm:]

        def sample_step(carry, rng_i):
            axis_raw_i, log_u_i, ang_i = rng_i
            carry = self._mh_step(log_psi_fn, carry, (axis_raw_i, log_u_i, ang_i), stepsize)
            return carry, carry[0]

        (_, _, acc_sample), all_pos = lax.scan(
            sample_step,
            (pos, log_pos, acc0),
            xs=(axis_raw_s, log_u_s, ang_s),
        )

        samples = all_pos.reshape((Nsweeps, keep) + self.shape)[:, -1]

        total_acc = acc_therm + acc_sample
        acc_rate = total_acc / jnp.float32(total_steps)
        return samples, acc_rate

        
    @partial(jax.jit, static_argnums=(0, 2, 3, 4))
    def run_many_chains(
        self, params, Nsweeps, Ntherm, keep, stepsize, pos_initials, seeds
    ):
        pos_initials = jnp.asarray(pos_initials)
        seeds = jnp.asarray(seeds, dtype=jnp.uint32)

        samples_per_chain, acc_rates = jax.vmap(
            lambda pos0, sd: self.run_chain(
                params, Nsweeps, Ntherm, keep, stepsize, pos0, sd
            ),
            in_axes=(0, 0),
            out_axes=(0, 0),
        )(pos_initials, seeds)

        # (nchains, Nsweeps, *shape) → (nchains*Nsweeps, *shape)
        nchains = pos_initials.shape[0]
        samples = samples_per_chain.reshape((nchains * Nsweeps,) + self.shape)
        acc_rate = acc_rates.mean()
        return samples, acc_rate
 

