#!/usr/bin/env python3
import os
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('XLA_PYTHON_CLIENT_ALLOCATOR', 'platform')
os.environ.setdefault('JAX_ENABLE_X64', 'true')

import argparse
import multiprocessing
os.environ.setdefault('XLA_FLAGS', f'--xla_force_host_platform_device_count={multiprocessing.cpu_count()}')

import time
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import gvar
import tqdm
from scipy.optimize import curve_fit
from scipy.special import kn

import jax
import jax.numpy as jnp
from jax import jit


# -----------------------------
# Correlator fit helpers
# -----------------------------

def corr_pbc_k0(r, A, xi, L, eps=1e-12):
    r = np.asarray(r, dtype=float)
    r1 = np.maximum(r, eps)
    r2 = np.maximum(L - r, eps)
    return A * (kn(0, r1 / xi) + kn(0, r2 / xi))


def fit_xi_windows_correlated(C, C_cov, L, rmin=2, rmax_stop=None, jitter_scale=1e-12):
    C = np.asarray(C, dtype=float)
    C_cov = np.asarray(C_cov, dtype=float)

    if rmax_stop is None:
        rmax_stop = L // 2
    if rmin < 1:
        raise ValueError('rmin must be >= 1')

    rmax_list = np.arange(rmin + 1, rmax_stop + 1)
    xi_list = np.full_like(rmax_list, np.nan, dtype=float)
    xi_err_list = np.full_like(rmax_list, np.nan, dtype=float)
    A_list = np.full_like(rmax_list, np.nan, dtype=float)
    A_err_list = np.full_like(rmax_list, np.nan, dtype=float)
    fit_ok = np.zeros_like(rmax_list, dtype=bool)

    for idx, rmax in enumerate(rmax_list):
        r = np.arange(rmin, rmax + 1, dtype=float)
        y = C[rmin:rmax + 1]
        cov = C_cov[rmin:rmax + 1, rmin:rmax + 1].copy()
        if jitter_scale != 0.0:
            cov.flat[:: cov.shape[0] + 1] += jitter_scale * np.trace(cov) / max(cov.shape[0], 1)

        xi0 = max(1.0, 0.15 * L)
        denom0 = kn(0, rmin / xi0) + kn(0, (L - rmin) / xi0)
        A0 = float(y[0] / (denom0 + 1e-300))
        p0 = (A0, xi0)
        bounds = ([-np.inf, 1e-6], [np.inf, 10.0 * L])

        try:
            popt, pcov = curve_fit(
                lambda rr, A, xi: corr_pbc_k0(rr, A, xi, L),
                r,
                y,
                p0=p0,
                sigma=cov,
                absolute_sigma=True,
                bounds=bounds,
                maxfev=20000,
            )
            A_hat, xi_hat = popt
            A_list[idx] = A_hat
            xi_list[idx] = xi_hat
            A_err_list[idx] = np.sqrt(max(pcov[0, 0], 0.0))
            xi_err_list[idx] = np.sqrt(max(pcov[1, 1], 0.0))
            fit_ok[idx] = True
        except Exception:
            pass

    return rmax_list, xi_list, xi_err_list, A_list, A_err_list, fit_ok


def jackknife_fit_xi_windows(samples, Cr_with_cov_optimized, block_size, rmin=2, rmax_stop=None, jitter_scale=1e-12, progress=True):
    samples = np.asarray(samples)
    Nsamp = samples.shape[0]
    L = samples.shape[1]
    if rmax_stop is None:
        rmax_stop = L // 2

    Nb = Nsamp // block_size
    if Nb < 2:
        raise ValueError('Need at least 2 blocks for jackknife.')
    Ntrim = Nb * block_size
    samples = samples[:Ntrim]

    C_full, C_cov_full, _ = Cr_with_cov_optimized(samples)
    rmax_list, xi_full, xi_fiterr_full, A_full, Aerr_full, fit_ok_full = fit_xi_windows_correlated(
        C_full, C_cov_full, L, rmin=rmin, rmax_stop=rmax_stop, jitter_scale=jitter_scale
    )

    nwin = len(rmax_list)
    xi_jk_samples = np.full((Nb, nwin), np.nan, dtype=float)
    fit_ok_jk = np.zeros((Nb, nwin), dtype=bool)

    iterator = range(Nb)
    if progress:
        iterator = tqdm.tqdm(iterator, total=Nb, desc='Jackknife xi blocks')

    for b in iterator:
        start = b * block_size
        stop = (b + 1) * block_size
        jk_samples = np.concatenate([samples[:start], samples[stop:]], axis=0)
        C_jk, C_cov_jk, _ = Cr_with_cov_optimized(jk_samples)
        _, xi_jk, _, _, _, fit_ok = fit_xi_windows_correlated(
            C_jk, C_cov_jk, L, rmin=rmin, rmax_stop=rmax_stop, jitter_scale=jitter_scale
        )
        xi_jk_samples[b] = xi_jk
        fit_ok_jk[b] = fit_ok

    xi_jk_mean = np.full(nwin, np.nan, dtype=float)
    xi_jk_err = np.full(nwin, np.nan, dtype=float)
    for j in range(nwin):
        vals = xi_jk_samples[:, j]
        mask = np.isfinite(vals)
        if np.sum(mask) >= 2:
            v = vals[mask]
            mean = np.mean(v)
            var = (len(v) - 1) / len(v) * np.sum((v - mean) ** 2)
            xi_jk_mean[j] = mean
            xi_jk_err[j] = np.sqrt(var)

    return {
        'rmax_list': rmax_list,
        'xi_full': xi_full,
        'xi_fiterr_full': xi_fiterr_full,
        'xi_jk_mean': xi_jk_mean,
        'xi_jk_err': xi_jk_err,
        'xi_jk_samples': xi_jk_samples,
        'fit_ok_full': fit_ok_full,
        'fit_ok_jk': fit_ok_jk,
    }


def choose_xi_plateau(results, min_points=3, max_redchi2=2.0):
    rmax_list = np.asarray(results['rmax_list'])
    xi = np.asarray(results['xi_jk_mean'], float)
    err = np.asarray(results['xi_jk_err'], float)
    good = np.isfinite(xi) & np.isfinite(err) & (err > 0)
    n = len(rmax_list)

    best = None
    for i0 in range(n):
        if not good[i0]:
            continue
        for i1 in range(i0 + min_points - 1, n):
            sl = slice(i0, i1 + 1)
            if not np.all(good[sl]):
                break
            x = xi[sl]
            s = err[sl]
            w = 1.0 / (s * s)
            xbar = np.sum(w * x) / np.sum(w)
            chi2 = np.sum(((x - xbar) / s) ** 2)
            dof = len(x) - 1
            redchi2 = chi2 / dof if dof > 0 else np.inf
            if redchi2 <= max_redchi2:
                score = (len(x), -redchi2)
                if best is None or score > best['score']:
                    best = {
                        'score': score,
                        'i0': i0,
                        'i1': i1,
                        'xbar': xbar,
                        'err': np.sqrt(1.0 / np.sum(w)),
                        'redchi2': redchi2,
                    }
    if best is None:
        raise RuntimeError('No acceptable xi plateau found.')
    i0, i1 = best['i0'], best['i1']
    return {
        'i0': i0,
        'i1': i1,
        'rmax_plateau': rmax_list[i0:i1 + 1],
        'xi_plateau': best['xbar'],
        'xi_plateau_err': best['err'],
        'redchi2': best['redchi2'],
    }


# -----------------------------
# Gap helpers
# -----------------------------

def block_jackknife_gap_loo(we1, w, e0, block_size):
    we1 = np.asarray(we1, float)
    w = np.asarray(w, float)
    e0 = np.asarray(e0, float)
    N = len(we1)
    m = int(block_size)
    Nb = N // m
    if Nb < 2:
        raise ValueError('Need at least 2 blocks for gap jackknife.')
    Ntrim = Nb * m
    we1 = we1[:Ntrim].reshape(Nb, m)
    w = w[:Ntrim].reshape(Nb, m)
    e0 = e0[:Ntrim].reshape(Nb, m)

    U_b = we1.sum(axis=1)
    V_b = w.sum(axis=1)
    W_b = e0.sum(axis=1)

    U = U_b.sum()
    V = V_b.sum()
    W = W_b.sum()

    gap_hat = (U / V) - (W / Ntrim)
    gap_loo = (U - U_b) / (V - V_b) - (W - W_b) / (Ntrim - m)
    return gap_hat, gap_loo


def jackknife_xi_gap_product_plateau(results, we1, w, e0, block_size, min_points=3, max_redchi2=2.0):
    plateau = choose_xi_plateau(results, min_points=min_points, max_redchi2=max_redchi2)
    i0, i1 = plateau['i0'], plateau['i1']

    xi_jk_samples = np.asarray(results['xi_jk_samples'], float)
    xi_jk_err = np.asarray(results['xi_jk_err'], float)

    s = xi_jk_err[i0:i1 + 1]
    good_w = np.isfinite(s) & (s > 0)
    if not np.any(good_w):
        raise RuntimeError('No valid xi plateau weights.')
    w_plateau = np.zeros_like(s)
    w_plateau[good_w] = 1.0 / (s[good_w] ** 2)

    xi_hat = plateau['xi_plateau']
    xi_block_vals = xi_jk_samples[:, i0:i1 + 1]
    xi_plateau_loo = np.full(xi_block_vals.shape[0], np.nan, dtype=float)
    for b in range(xi_block_vals.shape[0]):
        row = xi_block_vals[b]
        mask = np.isfinite(row) & (w_plateau > 0)
        if np.any(mask):
            ww = w_plateau[mask]
            xi_plateau_loo[b] = np.sum(ww * row[mask]) / np.sum(ww)

    gap_hat, gap_loo = block_jackknife_gap_loo(we1, w, e0, block_size)
    mask = np.isfinite(xi_plateau_loo) & np.isfinite(gap_loo)
    xi_loo = xi_plateau_loo[mask]
    gap_loo = gap_loo[mask]
    Nb = len(xi_loo)
    if Nb < 2:
        raise RuntimeError('Not enough valid jackknife replicas for xi*gap.')

    product_loo = xi_loo * gap_loo
    xi_bar = xi_loo.mean()
    xi_var = (Nb - 1) / Nb * np.sum((xi_loo - xi_bar) ** 2)
    gap_bar = gap_loo.mean()
    gap_var = (Nb - 1) / Nb * np.sum((gap_loo - gap_bar) ** 2)
    prod_bar = product_loo.mean()
    prod_var = (Nb - 1) / Nb * np.sum((product_loo - prod_bar) ** 2)

    return {
        'plateau': plateau,
        'xi_gvar': gvar.gvar(xi_hat, np.sqrt(xi_var)),
        'gap_gvar': gvar.gvar(gap_hat, np.sqrt(gap_var)),
        'product_gvar': gvar.gvar(xi_hat * gap_hat, np.sqrt(prod_var)),
    }


# -----------------------------
# Energy batching
# -----------------------------

def batched_local_energies(config_energy_opt, model, eta, g, mu, params, samples, num_batches=512, desc='Energy batches'):
    samples = jnp.asarray(samples)
    Ntotal, Nsites, dim = samples.shape
    if Ntotal % num_batches != 0:
        raise ValueError('For this helper, total number of samples must be divisible by num_batches.')
    batched = jnp.reshape(samples, (num_batches, -1, Nsites, dim))
    B = batched.shape[1]
    total = num_batches * B

    batch_fn = jax.jit(
        lambda configs: jax.vmap(lambda c: config_energy_opt(model, eta, g, mu, params, c))(configs)
    )

    out = np.empty((total,), dtype=np.float64)
    idx = 0
    for i in tqdm.tqdm(range(num_batches), desc=desc, leave=True):
        e = batch_fn(batched[i])
        e.block_until_ready()
        e_np = np.asarray(e)
        out[idx:idx + B] = e_np
        idx += B
    return out


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description='Compute xi, gap, and xi*gap from MC samples.')
    parser.add_argument('--repo-dir', type=str, required=True, help='Path to sigma repo containing sampling/wavefunction/observables modules.')
    parser.add_argument('--N', type=int, required=True)
    parser.add_argument('--g2', type=float, required=True)
    parser.add_argument('--eta', type=float, default=1.0)
    parser.add_argument('--mu', type=float, default=0.0)
    parser.add_argument('--hidden-size', type=int, default=40)
    parser.add_argument('--params', type=str, required=True)
    parser.add_argument('--params-excited', type=str, required=True)
    parser.add_argument('--nchains', type=int, default=16)
    parser.add_argument('--sweeps-per-chain', type=int, default=16384)
    parser.add_argument('--ntherm', type=int, default=20000)
    parser.add_argument('--keep', type=int, default=10)
    parser.add_argument('--embed-beta', type=float, default=None, help='Cluster reference coupling. Default 1/g^2.')
    parser.add_argument('--num-batches-energy', type=int, default=512)
    parser.add_argument('--jackknife-block-size', type=int, default=8192)
    parser.add_argument('--rmin', type=int, default=2)
    parser.add_argument('--rmax-stop', type=int, default=None)
    parser.add_argument('--jitter-scale', type=float, default=1e-12)
    parser.add_argument('--plateau-min-points', type=int, default=3)
    parser.add_argument('--plateau-max-redchi2', type=float, default=2.0)
    parser.add_argument('--output-csv', type=str, default='xi_gap_summary.csv')
    args = parser.parse_args()

    repo_dir = str(Path(args.repo_dir).expanduser().resolve())
    sys.path.insert(0, repo_dir)

    from sampling import newSampler  # noqa
    from wavefunction import MLP_SO3, MLP_excited  # noqa
    from observables import config_energy_opt, Cr_with_cov_optimized  # noqa

    N = args.N
    g = np.sqrt(args.g2)
    beta_embed = (1.0 / args.g2) if args.embed_beta is None else args.embed_beta

    x = np.random.normal(size=(N, 3))
    model = MLP_SO3(hidden_sizes=(args.hidden_size,))
    model_excited = MLP_excited(hidden_sizes=(args.hidden_size,))

    # init templates (for shape only)
    _ = model.init(jax.random.PRNGKey(int(time.time())), x)
    _ = model_excited.init(jax.random.PRNGKey(int(time.time()) + 1), x)

    @jit
    def psi(params, config):
        return model.apply(params, config)

    @jit
    def psi_excited(params, config):
        return model_excited.apply(params, config)

    params = np.load(args.params, allow_pickle=True).item()
    params_excited = np.load(args.params_excited, allow_pickle=True).item()

    sampler = newSampler(psi, (N, 3))

    pos_initials = [1 / np.sqrt(3) * jnp.ones((N, 3)) for _ in range(args.nchains)]
    seeds = np.random.randint(1, 1_000_000, size=(args.nchains,))

    samples, acc_rate = sampler.run_many_chains(
        params,
        args.sweeps_per_chain,
        args.ntherm,
        args.keep,
        np.deg2rad(20),
        pos_initials,
        seeds,
    )
    samples = np.asarray(samples)

    # xi from correlator + block-jackknife fit
    xi_results = jackknife_fit_xi_windows(
        samples,
        Cr_with_cov_optimized,
        block_size=args.jackknife_block_size,
        rmin=args.rmin,
        rmax_stop=args.rmax_stop,
        jitter_scale=args.jitter_scale,
        progress=True,
    )

    # gap from already-computed observables
    energies = batched_local_energies(
        config_energy_opt, model, args.eta, g, args.mu, params, samples,
        num_batches=args.num_batches_energy, desc='Ground-state energies'
    )
    excited = batched_local_energies(
        config_energy_opt, model_excited, args.eta, g, args.mu, params_excited, samples,
        num_batches=args.num_batches_energy, desc='Excited-state energies'
    )

    psi_exc_vals = np.asarray(jax.vmap(psi_excited, in_axes=[None, 0])(params_excited, jnp.asarray(samples)))
    psi_vals = np.asarray(jax.vmap(psi, in_axes=[None, 0])(params, jnp.asarray(samples)))
    weights = (psi_exc_vals / psi_vals) ** 2.0
    re_excited = excited * weights

    combined = jackknife_xi_gap_product_plateau(
        xi_results,
        re_excited,
        weights,
        energies,
        block_size=args.jackknife_block_size,
        min_points=args.plateau_min_points,
        max_redchi2=args.plateau_max_redchi2,
    )

    plateau = combined['plateau']
    xi_g = combined['xi_gvar']
    gap_g = combined['gap_gvar']
    prod_g = combined['product_gvar']

    df = pd.DataFrame([{
        'g2': args.g2,
        'N': N,
        'acceptance_rate': float(acc_rate),
        'xi': gvar.mean(xi_g),
        'xi_err': gvar.sdev(xi_g),
        'gap': gvar.mean(gap_g),
        'gap_err': gvar.sdev(gap_g),
        'xi_gap': gvar.mean(prod_g),
        'xi_gap_err': gvar.sdev(prod_g),
        'plateau_rmax_min': int(plateau['rmax_plateau'][0]),
        'plateau_rmax_max': int(plateau['rmax_plateau'][-1]),
        'plateau_redchi2': float(plateau['redchi2']),
        'nsamples': int(samples.shape[0]),
        'jackknife_block_size': int(args.jackknife_block_size),
    }])

    print(df.to_string(index=False))
    out_csv = Path(args.output_csv)
    df.to_csv(out_csv, index=False)
    print(f'\nWrote summary to {out_csv.resolve()}')


if __name__ == '__main__':
    main()
