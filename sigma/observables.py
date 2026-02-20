from wavefunction import *
from functools import partial
import jax.numpy as jnp
from jax import jit


@partial(jit, static_argnums=(0,1))
def config_energy(model, couplings, params, config):
    """
    Computes the energy of a single configuration.
    """

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