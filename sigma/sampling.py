import jax
import jax.numpy as jnp
from jax import random, lax
from functools import partial

EPS = 1e-12


def normalize(v, axis=-1, eps=EPS):
    return v / (jnp.linalg.norm(v, axis=axis, keepdims=True) + eps)


def project_to_sphere(x):
    """Project cartesian vectors to S^2 sitewise."""
    return normalize(x, axis=-1)


class Sampler:
    def __init__(self, psi, shape):
        """
        psi(params, coords) -> scalar amplitude for a single configuration.

        shape:
          - (N, 3): N sites, each a unit vector in R^3
          - or (..., 3): any leading site/batch dims, last dim must be 3

        This sampler targets π(x) ∝ |psi(params, x)|^2 with MH updates on S^2.
        """
        self.psi = psi
        self.shape = tuple(shape)
        if self.shape[-1] != 3:
            raise ValueError(f"Expected last dim = 3 for cartesian coords, got shape {self.shape}")

    # Only Nsweeps/Ntherm/keep are static because they determine shapes/loop lengths.
    @partial(jax.jit, static_argnums=(0, 2, 3, 4))
    def run_chain(self, params, Nsweeps, Ntherm, keep, stepsize, pos_initial, seed):
        pos = project_to_sphere(jnp.asarray(pos_initial))
        stepsize = jnp.asarray(stepsize)  # scalar (shared across all sites)

        NS = Nsweeps * keep + Ntherm
        key = random.PRNGKey(seed)

        # Inline log(psi) to avoid nested-jit boundary.
        def log_psi(x):
            return jnp.log(jnp.abs(self.psi(params, x)) + 1e-300)

        log_pos = log_psi(pos)
        acc0 = jnp.array(0, dtype=jnp.int32)

        # -----------------------
        # Thermalization
        # -----------------------
        def therm_step(carry, _):
            pos, log_pos, key, acc = carry
            key, k_noise, k_u = random.split(key, 3)

            noise = random.normal(k_noise, shape=self.shape)  # same shape as pos
            prop = project_to_sphere(pos + stepsize * noise)

            log_new = log_psi(prop)
            log_ratio = 2.0 * (log_new - log_pos)  # |psi|^2 target

            u = random.uniform(k_u, shape=())
            accept = jnp.log(u) <= log_ratio

            pos2 = lax.select(accept, prop, pos)
            log_pos2 = lax.select(accept, log_new, log_pos)
            acc2 = acc + accept.astype(acc.dtype)
            return (pos2, log_pos2, key, acc2), None

        (pos, log_pos, key, acc_therm), _ = lax.scan(
            therm_step,
            (pos, log_pos, key, acc0),
            xs=None,
            length=Ntherm,
        )

        # -----------------------
        # Sampling: batch RNG per sweep, then scan `keep` inner steps
        # -----------------------
        def sweep_step(carry, _):
            pos, log_pos, key, acc = carry
            key, k_noise, k_u = random.split(key, 3)

            noise_batch = random.normal(k_noise, shape=(keep,) + self.shape)
            u_batch = random.uniform(k_u, shape=(keep,))

            def inner(carry2, inputs):
                pos_i, log_pos_i, acc_i = carry2
                noise_i, u_i = inputs

                prop = project_to_sphere(pos_i + stepsize * noise_i)
                log_new = log_psi(prop)
                accept = jnp.log(u_i) <= 2.0 * (log_new - log_pos_i)

                pos_o = lax.select(accept, prop, pos_i)
                log_pos_o = lax.select(accept, log_new, log_pos_i)
                acc_o = acc_i + accept.astype(acc_i.dtype)
                return (pos_o, log_pos_o, acc_o), None

            (pos2, log_pos2, acc2), _ = lax.scan(
                inner,
                (pos, log_pos, acc),
                (noise_batch, u_batch),
            )

            return (pos2, log_pos2, key, acc2), pos2  # store after sweep

        (pos_f, log_pos_f, key_f, acc_sample), samples = lax.scan(
            sweep_step,
            (pos, log_pos, key, acc0),
            xs=None,
            length=Nsweeps,
        )

        total_acc = acc_therm + acc_sample
        acc_rate = total_acc.astype(jnp.float32) / jnp.asarray(NS, dtype=jnp.float32)
        return samples, acc_rate

    @partial(jax.jit, static_argnums=(0, 2, 3, 4))
    def run_many_chains(self, params, Nsweeps, Ntherm, keep, stepsize, pos_initials, seeds):
        pos_initials = project_to_sphere(jnp.asarray(pos_initials))  # (nchains, ...) + (3,)
        seeds = jnp.asarray(seeds)

        chain_vmapped = jax.vmap(
            lambda pos0, sd: self.run_chain(params, Nsweeps, Ntherm, keep, stepsize, pos0, sd),
            in_axes=(0, 0),
            out_axes=(0, 0),
        )

        samples_per_chain, acc_rates = chain_vmapped(pos_initials, seeds)
        samples = samples_per_chain.reshape((-1,) + self.shape)  # (nchains*Nsweeps, ...) + (3,)
        acc_rate = acc_rates.mean()
        return samples, acc_rate