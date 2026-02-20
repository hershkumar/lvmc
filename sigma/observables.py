from wavefunction import *
from functools import partial
import jax.numpy as jnp
from jax import jit

@partial(jit, static_argnums=(0))
def spherical_laplacian(model, params, config):
    """
    Returns sum of spherical laplacian per site i.e the angular momentum per site squared 
    """
    N = config.shape[0]
    v_local = jax.vmap(lambda i : local_L2_per_site(model, params, config, site_i), in_axes=0)

    return jnp.sum(v_local(jnp.arange(N))


def local_L2_per_site(model, params, config, site_i):
    """
    Return L_i^2 psi / psi for a single site i
    """
    eps = 1e-8 
    config = jnp.asarray(config)

    def psi_of_site(a_i):
        new_config = config.at[site_i].set(a_i)
        return model.apply(params, new_config)

    a0 = config[site_i]

    hess_psi = jax.hessian(psi_of_site)(a0)

    psi_val = psi_of_site(a0)

    theta = a0[0]
    sin_t = jnp.sin(theta)
    sin2 = jnp.maximum(sin_t * sin_t, eps)
    cot_t = jnp.cos(theta) / jnp.maximum(sin_t, jnp.sqrt(eps))

    L2_psi = -(
        hess_psi[0, 0]
        + cot_t * grad_psi[0]
        + hess_psi[1, 1] / sin2
    )

    return L2_psi / psi_val


@partial(jit, static_argnums=(0,1))
def config_energy(model, couplings, params, config):
    """
    Computes the energy of a single configuration.
    """
    eta = couplings['eta']
    g = couplings['g']
    kinetic_term = eta*g**2 * spherical_laplacian(model, params, config)

    raise NotImplementedError("Energy observable not implemented yet.")

@partial(jit, static_argnums=(0,1))
def batch_config_energy(model, couplings, params, configs):
    """
    Computes the energy of many configurations at once.
    """
    raise NotImplementedError("Energy observable not implemented yet.")


def gradient(model, couplings, params, configs):
    """
    Computes the gradient of the energy with respect to the parameters.
    """

    raise NotImplementedError("Gradient observable not implemented yet.")
