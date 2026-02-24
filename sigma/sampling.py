import jax
import jax.numpy as jnp
from jax import random, lax
from functools import partial

EPS = 1e-12


# ── helpers ──────────────────────────────────────────────────────────────────

def normalize(v, axis=-1, eps=EPS):
    return v / (jnp.linalg.norm(v, axis=axis, keepdims=True) + eps)


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
        # Total number of floats per configuration (used for RNG splitting).
        self._n_sites = 1
        for d in self.shape[:-1]:
            self._n_sites *= d

    
    def _mh_step(self, log_psi_fn, carry, rng_noise_u):
        """One MH update. carry = (pos, log_pos, acc)."""
        pos, log_pos, acc = carry
        noise, u = rng_noise_u

        prop = project_to_sphere(pos + carry[0].__class__(noise))   # type hint trick omitted
        prop = project_to_sphere(pos + noise)
        log_new = log_psi_fn(prop)
        accept = jnp.log(u) <= 2.0 * (log_new - log_pos)

        pos_out     = lax.select(accept, prop, pos)
        log_pos_out = lax.select(accept, log_new, log_pos)
        acc_out     = acc + accept.astype(acc.dtype)
        return (pos_out, log_pos_out, acc_out), pos_out

    @partial(jax.jit, static_argnums=(0, 2, 3, 4))
    def run_chain(self, params, Nsweeps, Ntherm, keep, stepsize, pos_initial, seed):
        pos      = project_to_sphere(jnp.asarray(pos_initial))
        stepsize = jnp.asarray(stepsize)

        key = random.PRNGKey(seed)

        def log_psi_fn(x):
            return jnp.log(jnp.abs(self.psi(params, x)) + 1e-300)

        log_pos = log_psi_fn(pos)
        acc0    = jnp.array(0, dtype=jnp.int32)

        
        total_steps  = Ntherm + Nsweeps * keep
        key, subkey  = random.split(key)
        # shape: (total_steps, *site_shape)  and  (total_steps,)
        all_noise = stepsize * random.normal(subkey, shape=(total_steps,) + self.shape)
        key, subkey2 = random.split(key)
        all_u     = random.uniform(subkey2, shape=(total_steps,))

        def therm_body(carry, rng_i):
            noise_i, u_i = rng_i
            (pos_i, log_pos_i, acc_i), _ = self._mh_step(
                log_psi_fn, (pos_i, log_pos_i, acc_i), (noise_i, u_i)
            )
            # re-pack (lax.scan carry must be flat)
            return (pos_i, log_pos_i, acc_i), None

        
        def therm_step(carry, rng_i):
            pos_i, log_pos_i, acc_i = carry
            noise_i, u_i = rng_i
            prop      = project_to_sphere(pos_i + noise_i)
            log_new   = log_psi_fn(prop)
            accept    = jnp.log(u_i) <= 2.0 * (log_new - log_pos_i)
            pos_o     = lax.select(accept, prop, pos_i)
            log_pos_o = lax.select(accept, log_new, log_pos_i)
            acc_o     = acc_i + accept.astype(acc_i.dtype)
            return (pos_o, log_pos_o, acc_o), None

        (pos, log_pos, acc_therm), _ = lax.scan(
            therm_step,
            (pos, log_pos, acc0),
            xs=(all_noise[:Ntherm], all_u[:Ntherm]),
        )

        sample_noise = all_noise[Ntherm:]   # (Nsweeps*keep, *shape)
        sample_u     = all_u[Ntherm:]       # (Nsweeps*keep,)

        def sample_step(carry, rng_i):
            pos_i, log_pos_i, acc_i = carry
            noise_i, u_i = rng_i
            prop      = project_to_sphere(pos_i + noise_i)
            log_new   = log_psi_fn(prop)
            accept    = jnp.log(u_i) <= 2.0 * (log_new - log_pos_i)
            pos_o     = lax.select(accept, prop, pos_i)
            log_pos_o = lax.select(accept, log_new, log_pos_i)
            acc_o     = acc_i + accept.astype(acc_i.dtype)
            return (pos_o, log_pos_o, acc_o), pos_o

        (pos_f, log_pos_f, acc_sample), all_pos = lax.scan(
            sample_step,
            (pos, log_pos, acc0),
            xs=(sample_noise, sample_u),
        )
        
        samples = all_pos.reshape((Nsweeps, keep) + self.shape)[:, -1]  # (Nsweeps, *shape)

        total_acc = acc_therm + acc_sample
        acc_rate  = total_acc / jnp.float32(total_steps)
        return samples, acc_rate

    
    @partial(jax.jit, static_argnums=(0, 2, 3, 4))
    def run_many_chains(self, params, Nsweeps, Ntherm, keep, stepsize, pos_initials, seeds):
        pos_initials = project_to_sphere(jnp.asarray(pos_initials))
        seeds        = jnp.asarray(seeds, dtype=jnp.uint32)

        samples_per_chain, acc_rates = jax.vmap(
            lambda pos0, sd: self.run_chain(
                params, Nsweeps, Ntherm, keep, stepsize, pos0, sd
            ),
            in_axes=(0, 0),
            out_axes=(0, 0),
        )(pos_initials, seeds)

        # (nchains, Nsweeps, *shape) → (nchains*Nsweeps, *shape)
        nchains = pos_initials.shape[0]
        samples  = samples_per_chain.reshape((nchains * Nsweeps,) + self.shape)
        acc_rate = acc_rates.mean()
        return samples, acc_rate