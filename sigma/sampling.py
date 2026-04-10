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
        self.psi = psi
        self.shape = tuple(shape)
        if self.shape[-1] != 3:
            raise ValueError(f"Expected last dim = 3, got {self.shape}")

    def _mh_step_rng(self, log_psi_fn, carry, stepsize):
        """
        carry = (key, pos, log_pos, acc)
        Generates axis/angle/log_u on the fly (no preallocation).
        """
        key, pos, log_pos, acc = carry
        key, k_axis, k_u, k_ang = random.split(key, 4)

        axis_raw = random.normal(k_axis, shape=self.shape, dtype=pos.dtype)
        axis = normalize(axis_raw, axis=-1)

        angle = random.uniform(
            k_ang, shape=self.shape[:-1], minval=-stepsize, maxval=stepsize, dtype=pos.dtype
        )
        log_u = jnp.log(random.uniform(k_u, shape=(), dtype=pos.dtype))

        prop = rodrigues_rotate(pos, axis, angle)
        prop = project_to_sphere(prop)

        log_new = log_psi_fn(prop)
        accept = log_u <= 2.0 * (log_new - log_pos)

        pos = lax.select(accept, prop, pos)
        log_pos = lax.select(accept, log_new, log_pos)
        acc = acc + accept.astype(acc.dtype)

        return (key, pos, log_pos, acc)

    @partial(jax.jit, static_argnums=(0, 2, 3, 4))
    def run_chain(self, params, Nsweeps, Ntherm, keep, stepsize, pos_initial, seed):
        pos = project_to_sphere(jnp.asarray(pos_initial))
        stepsize = jnp.asarray(stepsize, dtype=pos.dtype)

        key = random.PRNGKey(seed)

        def log_psi_fn(x):
            return jnp.log(jnp.abs(self.psi(params, x)) + 1e-300)

        log_pos = log_psi_fn(pos)
        acc0 = jnp.array(0, dtype=jnp.int32)

        # --- thermalization: run Ntherm steps, store nothing ---
        def therm_body(carry, _):
            carry = self._mh_step_rng(log_psi_fn, carry, stepsize)
            return carry, None

        (key, pos, log_pos, acc_therm), _ = lax.scan(
            therm_body, (key, pos, log_pos, acc0), xs=None, length=Ntherm
        )

        # --- sampling: for each sweep, do `keep` MH steps but only record final pos ---
        def one_sweep(carry, _):
            key, pos, log_pos, acc = carry

            def one_keep_step(carry2, _):
                return self._mh_step_rng(log_psi_fn, carry2, stepsize), None

            (key, pos, log_pos, acc), _ = lax.scan(
                one_keep_step, (key, pos, log_pos, acc), xs=None, length=keep
            )
            return (key, pos, log_pos, acc), pos  # record only once per sweep

        (key, pos, log_pos, acc_total), samples = lax.scan(
            one_sweep, (key, pos, log_pos, acc0), xs=None, length=Nsweeps
        )

        total_steps = Ntherm + Nsweeps * keep
        acc_rate = (acc_therm + acc_total) / jnp.float32(total_steps)

        # samples: (Nsweeps, N, 3)
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
 




class Sampler_TI:
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
    def run_many_chains(self, params, Nsweeps, Ntherm, keep, stepsize, pos_initials, seeds):
        pos_initials = jnp.asarray(pos_initials)
        seeds = jnp.asarray(seeds, dtype=jnp.uint32)

        samples_per_chain, acc_rates = jax.vmap(
            lambda pos0, sd: self.run_chain(params, Nsweeps, Ntherm, keep, stepsize, pos0, sd),
            in_axes=(0, 0),
            out_axes=(0, 0),
        )(pos_initials, seeds)

        nchains = pos_initials.shape[0]
        samples = samples_per_chain.reshape((nchains * Nsweeps,) + self.shape)  # (B, ...)

        L = self.shape[0]
        shifts = jnp.arange(1, L)  # exclude shift=0 (original samples)

        # Make (L-1, B, ...) then flatten to ((L-1)*B, ...)
        shifted = jax.vmap(lambda s: jnp.roll(samples, shift=s, axis=1))(shifts)
        shifted = shifted.reshape(((L - 1) * (nchains * Nsweeps),) + self.shape)

        out = jnp.concatenate([samples, shifted], axis=0)
        acc_rate = acc_rates.mean()
        return out, acc_rate



class ClusterSampler:
    def __init__(self, psi, shape):
        self.psi = psi
        self.shape = tuple(shape)
        if self.shape[-1] != 3:
            raise ValueError(f"Expected last dim = 3, got {self.shape}")
        N = self.shape[0]
        # 1D periodic neighbor list — shape (N, 2)
        self.neighbors = jnp.stack(
            [jnp.arange(N, dtype=jnp.int32) - 1,
             (jnp.arange(N, dtype=jnp.int32) + 1) % N],
            axis=-1,
        )
        # fix wrap for site 0
        self.neighbors = self.neighbors.at[0, 0].set(N - 1)

    # ── cluster building via BFS in a while_loop ────────────────────────
    def _build_cluster(self, key, projections, beta_embed, seed_site):
        N = projections.shape[0]
        neighbors = self.neighbors          # (N, 2)
        n_nbr = neighbors.shape[1]

        in_cluster = jnp.zeros(N, dtype=jnp.bool_)
        stack = jnp.zeros(N, dtype=jnp.int32)
        in_cluster = in_cluster.at[seed_site].set(True)
        stack = stack.at[0].set(seed_site)
        pp = jnp.int32(0)                  # process pointer
        ss = jnp.int32(1)                  # stack size

        def cond_fn(state):
            _, _, pp, ss, _ = state
            return pp < ss

        def body_fn(state):
            in_cluster, stack, pp, ss, key = state
            site = stack[pp]
            pp = pp + 1

            def process_neighbor(carry, nb_idx):
                in_cluster, ss, stack, key = carry
                nb = neighbors[site, nb_idx]
                key, subkey = random.split(key)
                p_bond = jnp.maximum(
                    0.0,
                    1.0 - jnp.exp(
                        -2.0 * beta_embed * projections[site] * projections[nb]
                    ),
                )
                u = random.uniform(subkey, dtype=projections.dtype)
                add = (~in_cluster[nb]) & (u < p_bond)
                in_cluster = in_cluster.at[nb].set(in_cluster[nb] | add)
                stack = stack.at[ss].set(jnp.where(add, nb, stack[ss]))
                ss = ss + add.astype(jnp.int32)
                return (in_cluster, ss, stack, key), None

            (in_cluster, ss, stack, key), _ = lax.scan(
                process_neighbor,
                (in_cluster, ss, stack, key),
                jnp.arange(n_nbr),
            )
            return (in_cluster, stack, pp, ss, key)

        in_cluster, _, _, _, key = lax.while_loop(
            cond_fn, body_fn, (in_cluster, stack, pp, ss, key)
        )
        return in_cluster, key

    # ── single cluster step + Metropolis accept/reject on |ψ|² ──────────
    def _cluster_step_rng(self, log_psi_fn, carry, beta_embed):
        key, pos, log_pos, acc = carry
        N = pos.shape[0]

        key, k_r, k_site = random.split(key, 3)

        # random reflection plane
        r = normalize(random.normal(k_r, shape=(3,), dtype=pos.dtype))

        # random seed site
        seed = random.randint(k_site, shape=(), minval=0, maxval=N)

        # projections onto r
        projections = pos @ r                       # (N,)

        # build Wolff cluster
        in_cluster, key = self._build_cluster(key, projections, beta_embed, seed)

        # reflect cluster spins: n_i → n_i − 2(n_i·r)r
        reflected = pos - 2.0 * projections[:, None] * r[None, :]
        prop = jnp.where(in_cluster[:, None], reflected, pos)
        prop = project_to_sphere(prop)

        # Metropolis accept/reject for the residual |ψ|² weight
        key, k_u = random.split(key)
        log_new = log_psi_fn(prop)
        log_u = jnp.log(random.uniform(k_u, shape=(), dtype=pos.dtype))
        accept = log_u <= 2.0 * (log_new - log_pos)

        pos = lax.select(accept, prop, pos)
        log_pos = lax.select(accept, log_new, log_pos)
        acc = acc + accept.astype(acc.dtype)
        return (key, pos, log_pos, acc)

    # ── run_chain: drop-in replacement ──────────────────────────────────
    @partial(jax.jit, static_argnums=(0, 2, 3, 4))
    def run_chain(self, params, Nsweeps, Ntherm, keep, stepsize, pos_initial, seed):
        """
        stepsize is reinterpreted as beta_embed (embedding coupling).
        Larger values → bigger clusters.
        """
        pos = project_to_sphere(jnp.asarray(pos_initial))
        beta_embed = jnp.asarray(stepsize, dtype=pos.dtype)
        key = random.PRNGKey(seed)

        def log_psi_fn(x):
            return jnp.log(jnp.abs(self.psi(params, x)) + 1e-300)

        log_pos = log_psi_fn(pos)
        acc0 = jnp.array(0, dtype=jnp.int32)

        # --- thermalization ---
        def therm_body(carry, _):
            return self._cluster_step_rng(log_psi_fn, carry, beta_embed), None

        (key, pos, log_pos, acc_therm), _ = lax.scan(
            therm_body, (key, pos, log_pos, acc0), xs=None, length=Ntherm
        )

        # --- sampling: `keep` cluster steps per sweep, record once ---
        def one_sweep(carry, _):
            key, pos, log_pos, acc = carry

            def one_keep_step(carry2, _):
                return self._cluster_step_rng(log_psi_fn, carry2, beta_embed), None

            (key, pos, log_pos, acc), _ = lax.scan(
                one_keep_step, (key, pos, log_pos, acc), xs=None, length=keep
            )
            return (key, pos, log_pos, acc), pos

        (key, pos, log_pos, acc_total), samples = lax.scan(
            one_sweep, (key, pos, log_pos, acc0), xs=None, length=Nsweeps
        )

        total_steps = Ntherm + Nsweeps * keep
        acc_rate = (acc_therm + acc_total) / jnp.float32(total_steps)
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
        nchains = pos_initials.shape[0]
        samples = samples_per_chain.reshape((nchains * Nsweeps,) + self.shape)
        acc_rate = acc_rates.mean()
        return samples, acc_rate
