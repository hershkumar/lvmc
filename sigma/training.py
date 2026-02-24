import optax
from jax import numpy as jnp
import numpy as np
from observables import dE_dparams
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


def train(results, model, eta, g, sampler, MC_options, steps, lr, fig=None):
    """
    Args:
        results: tuple of initial params, energies, and uncertainties (e.g. from a previous training run or from a pretraining step). If starting fresh, pass (init_params, [], []) where init_params are the initial parameters of the model.
        model: The neural network model representing the wavefunction.
        sampler: The sampler to generate configurations.
        MC_options: Monte Carlo options for sampling.
        steps: Number of training steps.
        lr: Learning rate for optimization.
        fig: optional argument, a plotly figure widget to update with training progress (energy and uncertainty). If None, no plotting is done.
        chain: If True, the initial samples for the next step are taken from the final samples of the previous step, instead of starting fresh each time. This can improve convergence but may introduce bias if not done carefully.

    Returns:
        params: The optimized parameters after training.
        energies: List of average energies at each training step.
        uncert: List of uncertainties at each training step.
    """

    params = results[0]
    tx = optax.adam(lr)
    opt_state = tx.init(params)
    step_nums = list(np.arange(len(results[1])))  # existing step numbers from results
    avg_energies = results[1]
    avg_uncerts = results[2]

    def push_point(fig, step, e, s):
        step_nums.append(step)
        avg_energies.append(e)
        avg_uncerts.append(s)

        steps_arr = jnp.asarray(step_nums)
        E_arr = jnp.asarray(avg_energies)
        sig_arr = jnp.asarray(avg_uncerts)

        upper = E_arr + sig_arr
        lower = E_arr - sig_arr

        # Batch-update to reduce redraw flicker
        with fig.batch_update():
            fig.data[0].x = steps_arr
            fig.data[0].y = E_arr

            fig.data[1].x = steps_arr
            fig.data[1].y = upper

            fig.data[2].x = steps_arr
            fig.data[2].y = lower


    prev_max_step = step_nums[-1] if step_nums else 0
    pbar = trange(steps, desc="", leave=True)

    if not MC_options["chain"]:
        # loop over training steps
        for step_num in pbar:

            grads, energy, uncert = step(params, model, eta, g, sampler, MC_options)
            # print(energy)
            # print(grads)
            avg_energies.append(energy)
            avg_uncerts.append(uncert)
            # set the label of the tqdm bar to print the current energy
            pbar.set_description(f"Energy = {energy}", refresh=True)

            if fig is not None:
                push_point(fig, prev_max_step + step_num, energy, uncert)

            updates, opt_state = tx.update(grads, opt_state, params)

            params = optax.apply_updates(params, updates)
    else:
        prev_samples = MC_options['pos_initials']
        for step_num in pbar:
            comp, prev_samples = step_chain(params, model, eta, g, sampler, MC_options, prev_samples)
            grads, energy, uncert = comp
            avg_energies.append(energy)
            avg_uncerts.append(uncert)
            pbar.set_description(f"Energy = {energy}", refresh=True)

            if fig is not None:
                push_point(fig, prev_max_step + step_num, energy, uncert)
            updates, opt_state = tx.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
    return params, avg_energies, avg_uncerts


def step_chain(params, model, eta, g, sampler, MC_options, prev_samples):
    nchains = MC_options["nchains"]
    Nsweeps = MC_options["num_samples"] // nchains

    prev_samples = jnp.asarray(prev_samples)
    # accept either (N,3) or (nchains,N,3)
    if prev_samples.shape == sampler.shape:
        prev_samples = jnp.broadcast_to(prev_samples, (nchains,) + sampler.shape)

    samples, _ = sampler.run_many_chains(
        params,
        Nsweeps,
        MC_options["thermalization"],
        MC_options["skip"],
        MC_options["var"],
        prev_samples,
        MC_options["seeds"],
    )

    # samples is chain-major flattened: (nchains*Nsweeps, N, 3)
    last_pos = samples.reshape((nchains, Nsweeps) + sampler.shape)[:, -1]  # (nchains, N, 3)

    return dE_dparams(model, eta, g, params, samples), last_pos

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
