from optax import adam
from jax import numpy as jnp
from tqdm import trange

# =============
# Gradient Descent Methods

# We define MC options as a dict that defines:
# num_samples: Number of KEPT samples (not per chain, but total across all chains)
# thermalization: Number of initial samples to discard (per chain)
# skip: number of samples to skip between kept samples (to reduce autocorrelation)
# var: variance of the proposal distribution in the sampler
# nchains: number of parallel MCMC chains to run
# seeds: random seeds for each chain (array of length nchains)
# pos_initials: initial positions for each chain (array of shape (nchains, N, 2))
example_MC_options = {
    "num_samples": 1000,
    "thermalization": 100,
    "skip": 5,
    "var": 5.0,
    "nchains": 16,
    "seeds": jnp.arange(16),
    "pos_initials": None,
}


def train(init_params, model, sampler, MC_options, steps, lr, ):
    """
    Args:
        init_params: Initial parameters of the model.
        model: The neural network model representing the wavefunction.
        sampler: The sampler to generate configurations.
        MC_options: Monte Carlo options for sampling.
        steps: Number of training steps.
        lr: Learning rate for optimization.
    """
    # unpack the MC options
    num_samples = MC_options["num_samples"]
    thermalization = MC_options["thermalization"]
    skip = MC_options["skip"]
    var = MC_options["var"]
    nchains = MC_options["nchains"]
    seeds = MC_options["seeds"]
    pos_initials = MC_options["pos_initials"]

    params = init_params
    # loop over training steps
    for step in trange(steps):

        # call the step function

        # update the progress bar with the current energy


        continue 
    raise NotImplementedError(
        "This is a placeholder for the training loop. We will implement this later."
    )


def step(params, model, sampler, MC_options):
    """
    Performs a single training step: samples configurations, computes the loss, and updates the model parameters.
    """
    raise NotImplementedError(
        "This is a placeholder for the training step. We will implement this later."
    )


# =============
# Stochastic Reconfiguration Methods
# TODO: Implement these if we feel like it
