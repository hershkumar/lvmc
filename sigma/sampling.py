import jax
import jax.numpy as jnp
from jax import random, lax
from functools import partial

TWOPI = 2.0 * jnp.pi

def wrap_phi(phi):
    # Map to [0, 2π)
    return jnp.mod(phi, TWOPI)

def reflect_theta(theta):
    """
    Reflect to [0, π] in a way that preserves symmetry.
    Equivalent to reflecting on boundaries repeatedly.
    """
    # First fold into [0, 2π)
    t = jnp.mod(theta, TWOPI)
    # Then reflect [π, 2π) back into [0, π]
    return jnp.where(t <= jnp.pi, t, TWOPI - t)

def project_angles(coords):
    """
    coords: (..., N, 2) with coords[..., 0]=theta, coords[..., 1]=phi
    returns projected coords in correct domains.
    """
    theta = reflect_theta(coords[..., 0])
    phi   = wrap_phi(coords[..., 1])
    return coords.at[..., 0].set(theta).at[..., 1].set(phi)

class Sampler:
    def __init__(self, psi, shape):
        """
        psi(params, coords) -> scalar amplitude for a single configuration.
        shape: (N, 2), sampling N sites and each site has 2 angles.
        First angle is polar angle theta in [0, pi],
        second is azimuthal angle phi in [0, 2*pi).
        """
        self.psi = psi
        self.shape = tuple(shape)
        self.N = self.shape[0]
        if self.shape[1] != 2:
            raise ValueError(f"Expected shape (N,2) for (theta,phi), got {self.shape}")

    @partial(jax.jit, static_argnums=(0,))
    def log_psi(self, coords, params):
        return jnp.log(self.psi(params, coords) + 1e-300)

    @partial(jax.jit, static_argnums=(0, 2, 3, 4, 5))
    def run_chain(self, params, Nsweeps, Ntherm, keep, stepsize, pos_initial, seed):
        """
        stepsize:
          - scalar: used for both theta and phi
          - or array-like shape (2,): [stepsize_theta, stepsize_phi]

        Returns:
          samples: (Nsweeps, *self.shape)  e.g. (Nsweeps, N, 2)
          acc_rate: scalar
        """
        pos_initial = project_angles(jnp.asarray(pos_initial))
        NS = Nsweeps * keep + Ntherm

        # Allow scalar or (2,) stepsize
        stepsize = jnp.asarray(stepsize)
        stepsize = jnp.where(stepsize.shape == (), jnp.array([stepsize, stepsize]), stepsize)
        stepsize_theta, stepsize_phi = stepsize[0], stepsize[1]

        key = random.PRNGKey(seed)
        key, k1, k2 = random.split(key, 3)

        # Proposals: uniform increments, but separate scales for theta/phi
        # randoms: (NS, N, 2)
        raw = random.uniform(k1, shape=(NS,) + self.shape, minval=-1.0, maxval=1.0)
        dx = raw.at[..., 0].multiply(stepsize_theta).at[..., 1].multiply(stepsize_phi)

        limits = random.uniform(k2, shape=(NS,))

        # Store chain: (NS+1, N, 2)
        sq = jnp.zeros((NS + 1,) + self.shape, dtype=pos_initial.dtype).at[0].set(pos_initial)
        counter = jnp.array(0, dtype=jnp.int32)

        def one_step(i, vals):
            sq, counter = vals
            old = sq[i]           # (N,2)
            proposed = old + dx[i]

            # Enforce domains: theta in [0,π], phi in [0,2π)
            new = project_angles(proposed)

            # MH acceptance: exp(2*(log|psi(new)| - log|psi(old)|))
            log_ratio = 2.0 * (self.log_psi(new, params) - self.log_psi(old, params))
            prob = jnp.exp(log_ratio)

            accept = prob >= limits[i]  # scalar bool

            sq = sq.at[i + 1].set(lax.select(accept, new, old))
            counter = counter + accept.astype(counter.dtype)
            return sq, counter

        sq, counter = lax.fori_loop(0, NS, one_step, (sq, counter))

        samples = sq[1 + Ntherm :: keep]
        acc_rate = counter.astype(jnp.float32) / jnp.asarray(NS, dtype=jnp.float32)
        return samples, acc_rate

    @partial(jax.jit, static_argnums=(0, 2, 3, 4, 5))
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