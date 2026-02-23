from wavefunction import *
from utils import *
from functools import partial
import jax.numpy as jnp
from jax import jit
import jax

@partial(jit, static_argnums=(0))
def spherical_laplacian(model, params, config):
    """
    Returns sum of spherical laplacian per site i.e the angular momentum per site squared 
    """
    N = config.shape[0]
    v_local = jax.vmap(lambda i : local_L2_per_site(model, params, config, i), in_axes=0)

    return jnp.sum(v_local(jnp.arange(N)))


def local_L2_per_site(model, params, config, site_i):
    """
    Returns (L_i^2 psi)/psi at site_i, assuming:
      model.apply(params, config) = A(config) = -log psi(config)
    so psi = exp(-A).
    """
    config = jnp.asarray(config)
    eps=1e-8
    def A_of_site(a_i):
        new_config = config.at[site_i].set(a_i)
        return model.apply(params, new_config)  # A = -logψ (possibly complex)

    a0 = config[site_i]  # (theta, phi)

    g = jax.grad(A_of_site)(a0)        # (A_theta, A_phi)
    H = jax.hessian(A_of_site)(a0)     # [[A_tt, A_tφ],[A_φt, A_φφ]]

    theta = a0[0]
    sin_t = jnp.sin(theta)
    sin_t = jnp.where(jnp.abs(sin_t) < eps, jnp.sign(sin_t) * eps + (sin_t == 0) * eps, sin_t)
    sin2  = sin_t * sin_t
    cot_t = jnp.cos(theta) / sin_t

    A_t, A_p = g[0], g[1]
    A_tt = H[0, 0]
    A_pp = H[1, 1]

    lap_A = A_tt + cot_t * A_t + A_pp / sin2
    gradA_sq = (A_t * A_t) + (A_p * A_p) / sin2  # no complex conjugation

    # (L^2 ψ)/ψ = ΔA - |∇A|^2  (with S^2 metric)
    return lap_A - gradA_sq


def angles_to_unitvec(config):
    """
    config: (L, 2) with columns (theta, phi)
    returns n: (L, 3)
    """
    theta = config[:, 0]
    phi = config[:, 1]
    st = jnp.sin(theta)
    return jnp.stack(
        [st * jnp.cos(phi), st * jnp.sin(phi), jnp.cos(theta)],
        axis=-1,
    )


@partial(jit, static_argnums=(0,1))
def config_energy(model, eta, g, params, config):
    """
    Computes the energy of a single configuration.
    """
    kinetic_term = eta*g**2 * spherical_laplacian(model, params, config)

    n = angles_to_unitvec(config)
    n_next = jnp.roll(n, shift=-1, axis=0)
    nn = jnp.sum(n*n_next, axis=-1)
    potential_term = -(eta/g**2.)*jnp.sum(nn)

    return kinetic_term + potential_term

@partial(jit, static_argnums=(0,1))
def batch_config_energy(model, eta, g, params, configs):
    """
    Computes the energy of many configurations at once.
    """
    raise NotImplementedError("Energy observable not implemented yet.")


@partial(jit, static_argnums=(0,))
def dlogpsi_dparams(model, params, config):
    """
    Returns pytree with same structure as params:
        d/dparams [ log psi(config) ]

    model.apply(params, config) must return A = -log psi.
    """

    def A_of_params(p):
        return model.apply(p, config)   # A = -log psi

    gradA = jax.grad(A_of_params)(params)

    # d logpsi / d params = - dA / d params
    return jax.tree_util.tree_map(lambda x: -x, gradA)

def gradient(model, eta, g, params, configs):
    """
    Computes the gradient of the energy with respect to the parameters.
    """

    local_energy = config_energy(model, eta, g, params, config)
    dlogpsi_dparams = dlogpsi_dparams(model, params, config)

    return [local_energy, dlogpsi_dparams, pytree_mult(local_energy,dlogpsi_dparams)]
