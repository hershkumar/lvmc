from optax import adam, chain, scale, scale_by_adam
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


def train(init_params, model, eta, g, sampler, MC_options, steps, lr, fig=None):
    """
    Args:
        init_params: Initial parameters of the model.
        model: The neural network model representing the wavefunction.
        sampler: The sampler to generate configurations.
        MC_options: Monte Carlo options for sampling.
        steps: Number of training steps.
        lr: Learning rate for optimization.
        fig: optional argument, a plotly figure widget to update with training progress (energy and uncertainty). If None, no plotting is done.

    Returns:
        params: The optimized parameters after training.
        energies: List of average energies at each training step.
        uncert: List of uncertainties at each training step.
    """

    tx = chain(
        scale_by_adam(),
        scale(-lr),
    )
    opt_state = tx.init(init_params)

    params = init_params

    avg_energies = []
    avg_uncerts = []

    # loop over training steps
    for step_num in trange(steps):

        grads, energy, uncert = step(params, model, eta, g, sampler, MC_options)
        avg_energies.append(energy)
        avg_uncerts.append(uncert)

        if fig is not None:
            fig.data[0].x = step_num
            fig.data[0].y = energy

            fig.data[1].x = step_num
            fig.data[1].y = energy + uncert

            fig.data[2].x = step_num
            fig.data[2].y = energy - uncert
        
        updates, opt_state = tx.update(grads, opt_state)

        params = tx.apply_updates(params, updates)

    return params, avg_energies, avg_uncerts


def step(params, model, eta, g, sampler, MC_options):
    """
    Performs a single training step: samples configurations, computes the loss, and updates the model parameters.
    """
    # sample configurations using the sampler
    samples, _ = sampler.run_many_chains(
        params,
        MC_options["num_samples"] // MC_options["nchains"],
        MC_options["thermalization"],
        MC_options["skip"],
        MC_options["var"],
        MC_options["pos_initials"],
        MC_options["seeds"],
    )

    return dE_dparams(model, eta, g, params, samples)


# =============
# Stochastic Reconfiguration Methods
# TODO: Implement these if we feel like it
