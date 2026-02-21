import jax
import jax.numpy as jnp
from jax import random, lax
from functools import partial

TWOPI = 2.0 * jnp.pi


def wrap_phi(phi):
    return jnp.mod(phi, TWOPI)


def reflect_theta(theta):
    t = jnp.mod(theta, TWOPI)
    return jnp.where(t <= jnp.pi, t, TWOPI - t)


def project_angles(coords):
    theta = reflect_theta(coords[..., 0])
    phi = wrap_phi(coords[..., 1])
    return jnp.stack([theta, phi], axis=-1)


class Sampler:
    def __init__(self, psi, shape):
        self.psi = psi
        self.shape = tuple(shape)
        self.N = self.shape[0]

    # Only Nsweeps/Ntherm/keep are static because they determine shapes/loop lengths.
    @partial(jax.jit, static_argnums=(0, 2, 3, 4))
    def run_chain(self, params, Nsweeps, Ntherm, keep, stepsize, pos_initial, seed):
        pos = project_angles(jnp.asarray(pos_initial))
        stepsize = jnp.asarray(stepsize)

        NS = Nsweeps * keep + Ntherm
        key = random.PRNGKey(seed)

        # Inline log(psi) to avoid nested-jit boundary.
        # Add tiny epsilon so log is well-defined if psi can be 0.
        def log_psi(coords):
            return jnp.log(self.psi(params, coords) + 1e-300)

        log_pos = log_psi(pos)
        acc0 = jnp.array(0, dtype=jnp.int32)

        # -----------------------
        # Thermalization (step-wise)
        # -----------------------
        # Thermalization is typically smaller; step-wise split is usually fine.
        def therm_step(carry, _):
            pos, log_pos, key, acc = carry
            key, k_dx, k_u = random.split(key, 3)

            dx = random.uniform(k_dx, shape=self.shape, minval=-1.0, maxval=1.0) * stepsize
            proposed = project_angles(pos + dx)

            log_new = log_psi(proposed)
            log_ratio = 2.0 * (log_new - log_pos)

            u = random.uniform(k_u, shape=())
            accept = jnp.log(u) <= log_ratio

            pos2 = lax.select(accept, proposed, pos)
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
        # Sampling (batched RNG per sweep)
        # -----------------------
        # For each sweep, generate all `keep` proposals + uniforms at once, then scan.
        def sweep_step(carry, _):
            pos, log_pos, key, acc = carry
            key, k_dx, k_u = random.split(key, 3)

            dx_batch = random.uniform(
                k_dx, shape=(keep,) + self.shape, minval=-1.0, maxval=1.0
            ) * stepsize
            u_batch = random.uniform(k_u, shape=(keep,))

            def inner(carry2, inputs):
                pos_i, log_pos_i, acc_i = carry2
                dx_i, u_i = inputs

                proposed = project_angles(pos_i + dx_i)
                log_new = log_psi(proposed)
                log_ratio = 2.0 * (log_new - log_pos_i)

                accept = jnp.log(u_i) <= 2.0 * (log_new - log_pos_i)

                pos_o = lax.select(accept, proposed, pos_i)
                log_pos_o = lax.select(accept, log_new, log_pos_i)
                acc_o = acc_i + accept.astype(acc_i.dtype)
                return (pos_o, log_pos_o, acc_o), None

            (pos2, log_pos2, acc2), _ = lax.scan(
                inner,
                (pos, log_pos, acc),
                (dx_batch, u_batch),
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
        pos_initials = project_angles(jnp.asarray(pos_initials))
        seeds = jnp.asarray(seeds)

        chain_vmapped = jax.vmap(
            lambda pos0, sd: self.run_chain(params, Nsweeps, Ntherm, keep, stepsize, pos0, sd),
            in_axes=(0, 0),
            out_axes=(0, 0),
        )

        samples_per_chain, acc_rates = chain_vmapped(pos_initials, seeds)
        samples = samples_per_chain.reshape((-1,) + self.shape)
        acc_rate = acc_rates.mean()
        return samples, acc_rate