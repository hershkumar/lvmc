import jax
import jax.numpy as jnp
from jax import random, lax
from functools import partial

class Sampler:
    def __init__(self, psi, shape):
        """
        psi(params, coords) -> scalar amplitude for a single configuration.
        shape: e.g. (N,d)
        """
        self.psi = psi
        self.shape = tuple(shape)
        self.N = self.shape[0]

    @partial(jax.jit, static_argnums=(0,))
    def log_psi(self, coords, params):
        return jnp.log(self.psi(params, coords) + 1e-300)

    @partial(jax.jit, static_argnums=(0, 2, 3, 4, 5))
    def run_chain(self, params, Nsweeps, Ntherm, keep, stepsize, pos_initial, seed):
        """
        Returns:
          samples: (Nsweeps, *self.shape)  e.g. (Nsweeps, N, d)
          acc_rate: scalar
        """
        pos_initial = jnp.asarray(pos_initial)
        NS = Nsweeps * keep + Ntherm

        key = random.PRNGKey(seed)
        key, k1, k2 = random.split(key, 3)

        # Proposals match configuration shape (N,d)
        randoms = random.uniform(
            k1,
            shape=(NS,) + self.shape,
            minval=-stepsize,
            maxval=stepsize,
        )
        limits = random.uniform(k2, shape=(NS,))

        # Store chain: (NS+1, N, d)
        sq = jnp.zeros((NS + 1,) + self.shape, dtype=pos_initial.dtype).at[0].set(pos_initial)
        counter = jnp.array(0, dtype=jnp.int32)

        def one_step(i, vals):
            sq, counter = vals
            dx = randoms[i]   # (N,d)
            old = sq[i]       # (N,d)
            new = old + dx    # (N,d)

            # MH acceptance: exp(2*(log|psi(new)| - log|psi(old)|))
            log_ratio = 2.0 * (self.log_psi(new, params) - self.log_psi(old, params))
            prob = jnp.exp(log_ratio)

            accept = prob >= limits[i]  # scalar bool

            sq = sq.at[i + 1].set(lax.select(accept, new, old))
            counter = counter + accept.astype(counter.dtype)
            return sq, counter

        sq, counter = lax.fori_loop(0, NS, one_step, (sq, counter))

        samples = sq[1 + Ntherm :: keep]  # (Nsweeps, N, d)
        acc_rate = counter.astype(jnp.float32) / jnp.asarray(NS, dtype=jnp.float32)
        return samples, acc_rate

    @partial(jax.jit, static_argnums=(0, 2, 3, 4, 5))
    def run_many_chains(self, params, Nsweeps, Ntherm, keep, stepsize, pos_initials, seeds):
        """
        pos_initials: (nchains, *self.shape)
        seeds: (nchains,)

        Returns:
          samples:  (nchains*Nsweeps, *self.shape)
          acc_rate: scalar mean acceptance over chains
        """
        pos_initials = jnp.asarray(pos_initials)
        seeds = jnp.asarray(seeds)

        chain_vmapped = jax.vmap(
            lambda pos0, sd: self.run_chain(params, Nsweeps, Ntherm, keep, stepsize, pos0, sd),
            in_axes=(0, 0),
            out_axes=(0, 0),
        )

        samples_per_chain, acc_rates = chain_vmapped(pos_initials, seeds)
        # samples_per_chain: (C, Nsweeps, N, d)

        samples = samples_per_chain.reshape((-1,) + self.shape)  # (C*Nsweeps, N, d)
        acc_rate = acc_rates.mean()
        return samples, acc_rate