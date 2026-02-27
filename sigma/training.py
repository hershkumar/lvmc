import optax
from jax import numpy as jnp
import numpy as np
from observables import * 
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


def train(results, model, eta, g, sampler, MC_options, steps, lr, fig=None, batch_size=None, clip_threshold=None):
    """
    Args:
        results: tuple of initial params, energies, and uncertainties (e.g. from a previous training run or from a pretraining step). If starting fresh, pass (init_params, [], []) where init_params are the initial parameters of the model.
        model: The neural network model representing the wavefunction.
        sampler: The sampler to generate configurations.
        MC_options: Monte Carlo options for sampling.
        steps: Number of training steps.
        lr: Learning rate for optimization.
        fig: optional argument, a plotly figure widget to update with training progress (energy and uncertainty). If None, no plotting is done.
        batch_size: optional argument, the number of samples to use in each batch for gradient computation. If None, all samples are used in each batch.
        clip_threshold: optional argument, the maximum norm of gradients to be clipped. If None, no clipping is performed.
    Returns:
        params: The optimized parameters after training.
        energies: List of average energies at each training step.
        uncert: List of uncertainties at each training step.
        last_samples: The last batch of samples generated (useful for resuming training with chain continuation).
    """

    params = results[0]
    if clip_threshold is not None:
        tx = optax.chain(
            optax.clip_by_global_norm(clip_threshold),
            optax.adam(lr),
        )
    else:
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

            grads, energy, uncert = step(params, model, eta, g, sampler, MC_options, batch_size)
            # print(energy)
            # print(grads)
            
            # set the label of the tqdm bar to print the current energy
            pbar.set_description(f"Energy Density = {energy/sampler.shape[0]}", refresh=True)

            if fig is not None:
                push_point(fig, prev_max_step + step_num, energy, uncert)
            else:
                avg_energies.append(energy)
                avg_uncerts.append(uncert)

            updates, opt_state = tx.update(grads, opt_state, params)

            params = optax.apply_updates(params, updates)
    else:
        prev_samples = MC_options['pos_initials']
        for step_num in pbar:
            comp, prev_samples = step_chain(params, model, eta, g, sampler, MC_options, prev_samples, batch_size=batch_size)
            grads, energy, uncert = comp
            
            pbar.set_description(f"Energy Density = {energy/sampler.shape[0]}", refresh=True)

            if fig is not None:
                push_point(fig, prev_max_step + step_num, energy, uncert)
            else:
                avg_energies.append(energy)
                avg_uncerts.append(uncert)
            updates, opt_state = tx.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
    return params, avg_energies, avg_uncerts


def step_chain(params, model, eta, g, sampler, MC_options, prev_samples, batch_size=None):
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

    return dE_dparams(model, eta, g, params, samples, batch_size=batch_size), last_pos

def step(params, model, eta, g, sampler, MC_options, batch_size=None):
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

    return dE_dparams(model, eta, g, params, samples, batch_size=batch_size)


# =============
# Stochastic Reconfiguration Methods
# TODO: Implement these if we feel like it


# def sr_step(model, eta, g_coup, params, configs, lr,
#             damping=1e-3, cg_tol=1e-6, cg_maxiter=200, batch_size=None):
#     """
#     SR / natural gradient step:
#       delta = (S + damping*I)^{-1} grad
#       params <- params - lr * delta
#     """
#     grad, logs_centered, E, uncert = sr_stats(model, eta, g_coup, params, configs, batch_size)

#     # Flatten pytrees so we can use jax.scipy.sparse.linalg.cg on vectors
#     grad_flat, unravel = ravel_pytree(grad)

#     def matvec_flat(v_flat):
#         v = unravel(v_flat)
#         Sv = fisher_matvec_from_samples(logs_centered, v)     # covariance * v (since centered)
#         Sv = add_damping(Sv, v, damping=damping)              # (S + damping I) v
#         return ravel_pytree(Sv)[0]

#     # CG solve (SPD if damping > 0)
#     delta_flat, info = jsp.sparse.linalg.cg(
#         A=matvec_flat,
#         b=grad_flat,
#         tol=cg_tol,
#         maxiter=cg_maxiter,
#     )
#     delta = unravel(delta_flat)

#     new_params = jax.tree_util.tree_map(lambda p, d: p - lr * d, params, delta)
#     return new_params, E, uncert, info

def sr_step(model, eta, g_coup, params, configs, lr,
            damping=1e-3, cg_tol=1e-6, cg_maxiter=200,
            clip_threshold=1.0, batch_size=None):
    """
    SR / natural gradient step:
      delta = (S + damping*I)^{-1} grad
      delta <- clip(delta, clip_threshold)   # clip by global norm
      params <- params - lr * delta

    clip_threshold: maximum L2 norm of the update delta. Set to None to disable.
    """
    grad, logs_centered, E, uncert = sr_stats(
        model, eta, g_coup, params, configs, batch_size
    )

    def matvec(v):
        return add_damping(fisher_matvec(logs_centered, v), v, damping)

    delta, info = jax.scipy.sparse.linalg.cg(
        A=matvec,
        b=grad,
        tol=cg_tol,
        maxiter=cg_maxiter,
    )

    # Clip delta by its global L2 norm.
    # If ||delta|| > clip_threshold, rescale so ||delta|| == clip_threshold.
    if clip_threshold is not None:
        delta_norm = jnp.sqrt(pytree_dot(delta, delta))
        scale = jnp.minimum(1.0, clip_threshold / (delta_norm + 1e-12))
        delta = pytree_mult(scale, delta)

    new_params = jax.tree_util.tree_map(lambda p, d: p - lr * d, params, delta)
    return new_params, E, uncert, info

def damping_schedule(step, damping_init=0.1, damping_final=1e-3, decay=0.05):
    """
    Exponential decay from damping_init to damping_final:
      damping(t) = damping_final + (damping_init - damping_final) * exp(-decay * t)
    """
    return damping_final + (damping_init - damping_final) * np.exp(-decay * step)


def sr_train(results, model, eta, g, sampler, MC_options, steps, lr,
             damping_init=0.1, damping_final=1e-3, damping_decay=0.05,
             fig=None, batch_size=None):
    """
    SR training loop with an exponentially decaying damping schedule.

    Damping starts at damping_init and decays toward damping_final
    with rate damping_decay:
        damping(t) = damping_final + (damping_init - damping_final) * exp(-damping_decay * t)
    where t is the global step number (so the schedule is continuous across
    resumed runs).
    """

    params = results[0]
    step_nums = list(np.arange(len(results[1])))
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
        for step_num in pbar:
            global_step = prev_max_step + step_num
            damping = damping_schedule(global_step, damping_init, damping_final, damping_decay)

            samples, _ = sampler.run_many_chains(
                params,
                MC_options["num_samples"] // MC_options["nchains"],
                MC_options["thermalization"],
                MC_options["skip"],
                MC_options["var"],
                MC_options["pos_initials"],
                MC_options["seeds"],
            )
            params, energy, uncert, info = sr_step(
                model, eta, g, params, samples, lr, damping=damping, clip_threshold=1.0
            )
            
            pbar.set_description(
                f"E/N = {energy/sampler.shape[0]:.4f} | damping = {damping:.2e} | cg = {info}",
                refresh=True,
            )

            if fig is not None:
                push_point(fig, global_step, energy, uncert)
            else:
                avg_energies.append(energy)
                avg_uncerts.append(uncert)
    else:
        prev_samples = MC_options['pos_initials']
        for step_num in pbar:
            global_step = prev_max_step + step_num
            damping = damping_schedule(global_step, damping_init, damping_final, damping_decay)

            samples, _ = sampler.run_many_chains(
                params,
                MC_options["num_samples"] // MC_options["nchains"],
                MC_options["thermalization"],
                MC_options["skip"],
                MC_options["var"],
                prev_samples,
                MC_options["seeds"],
            )
            params, energy, uncert, info = sr_step(
                model, eta, g, params, samples, lr, damping=damping,clip_threshold=1.0, batch_size=batch_size
            )
            
            pbar.set_description(
                f"E/N = {energy/sampler.shape[0]:.4f} | damping = {damping:.2e} | cg = {info}",
                refresh=True,
            )

            if fig is not None:
                push_point(fig, global_step, energy, uncert)
            else:
                avg_energies.append(energy)
                avg_uncerts.append(uncert)
            prev_samples = samples.reshape((MC_options["nchains"], -1) + sampler.shape)[:, -1]

    return params, avg_energies, avg_uncerts