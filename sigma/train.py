#!/usr/bin/env python3
import os
import csv
import json
import time
import math
import argparse
from pathlib import Path

# Set JAX env vars before importing jax.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
os.environ.setdefault("JAX_ENABLE_X64", "false")

import numpy as np
import jax
import jax.numpy as jnp

from sampling import newSamplerSharded, ClusterSamplerSharded
from wavefunction import GodSlayer3
from training import sr_train_adapt_shard
from observables import batched_energy, Cr_with_cov_optimized


def xi_second_moment_from_corr(C, C_cov=None):
    """Compute the second-moment correlation length from a 1D periodic correlator."""
    C = np.asarray(C, dtype=float)
    L = C.shape[0]
    k_min = 2.0 * np.pi / L

    x = np.arange(L, dtype=float)
    G0 = np.sum(C)
    Gk = np.sum(np.exp(1j * k_min * x) * C).real

    pref = 1.0 / (4.0 * np.sin(k_min / 2.0) ** 2)
    xi2 = pref * (G0 / Gk - 1.0)
    xi2 = max(xi2, 0.0)
    xi = np.sqrt(xi2)

    if C_cov is None:
        return xi, None

    C_cov = np.asarray(C_cov, dtype=float)
    cos_kx = np.cos(k_min * x)
    dxi2_dC = pref * (1.0 / Gk - (G0 / (Gk * Gk)) * cos_kx)

    if xi > 0.0:
        dxi_dC = 0.5 * dxi2_dC / xi
        xi_var = dxi_dC @ C_cov @ dxi_dC
        xi_err = np.sqrt(max(xi_var, 0.0))
    else:
        xi_err = 0.0

    return xi, xi_err


def positive_int(value: str) -> int:
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"Expected a positive integer, got {value}")
    return ivalue


def nonnegative_float(value: str) -> float:
    fvalue = float(value)
    if fvalue < 0:
        raise argparse.ArgumentTypeError(f"Expected a nonnegative float, got {value}")
    return fvalue


def gen_init_positions(nchains: int, N: int, seed: int) -> list[jnp.ndarray]:
    """Generate normalized random initial positions on S^2."""
    positions = []
    for i in range(nchains):
        x = jax.random.normal(jax.random.PRNGKey(seed + i), shape=(N, 3), dtype=jnp.float32)
        x = x / jnp.linalg.norm(x, axis=-1, keepdims=True)
        positions.append(x)
    return positions


def count_flax_params(params) -> int:
    leaves = jax.tree_util.tree_leaves(params)
    return int(sum(x.size for x in leaves))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train the GodSlayer3 wavefunction with sharded SR on the UMD cluster."
    )

    # Physics / model args
    p.add_argument("--N", type=positive_int, required=True, help="Lattice size")
    p.add_argument("--eta", type=float, default=1.0, help="Eta coupling")
    p.add_argument("--g2", type=nonnegative_float, required=True, help="Value of g^2")
    p.add_argument("--mu", type=float, default=0.0, help="Chemical potential")
    p.add_argument("--channels", type=positive_int, required=True, help="Number of channels")
    p.add_argument("--layers", type=positive_int, required=True, help="Number of GodSlayer layers")
    p.add_argument("--neighbors", type=positive_int, required=True, help="Number of neighbors")
    p.add_argument(
        "--mlp",
        type=positive_int,
        nargs="+",
        required=True,
        help="MLP hidden sizes, e.g. --mlp 40 or --mlp 64 64",
    )
    p.add_argument(
        "--residual-scale",
        type=float,
        default=None,
        help="Residual scale. Default: 0.25 / layers, matching the notebook.",
    )

    # Training args
    p.add_argument("--samples", type=positive_int, required=True, help="Number of MC samples per step")
    p.add_argument("--thermal", type=positive_int, required=True, help="Thermalization sweeps")
    p.add_argument("--skip", type=positive_int, required=True, help="Sweeps between kept samples")
    p.add_argument("--lr", type=float, required=True, help="SR learning rate")
    p.add_argument("--nchains", type=positive_int, required=True, help="Number of Markov chains")
    p.add_argument("--steps", type=positive_int, required=True, help="Number of SR steps")
    p.add_argument("--batchsize", type=positive_int, required=True, help="Batch size for sharded energy/gradient")
    p.add_argument(
        "--var",
        type=nonnegative_float,
        default=2.0 * math.pi,
        help="Initial proposal angle in radians",
    )
    p.add_argument(
        "--adapt-rate",
        type=float,
        default=1.0,
        help="Acceptance-rate adaptation strength",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Master seed. Default: current unix time.",
    )

    # Post-training evaluation args
    p.add_argument(
        "--eval-chains",
        type=positive_int,
        default=50,
        help="Number of chains for final evaluation sampling",
    )
    p.add_argument(
        "--eval-samples-total",
        type=positive_int,
        default=2**20,
        help="Total target number of post-training samples before division across chains",
    )
    p.add_argument(
        "--eval-thermal",
        type=positive_int,
        default=50000,
        help="Thermalization sweeps for final evaluation sampling",
    )
    p.add_argument(
        "--eval-skip",
        type=positive_int,
        default=10,
        help="Sweeps between kept samples for final evaluation sampling",
    )
    p.add_argument(
        "--corr-batchsize",
        type=positive_int,
        default=20000,
        help="Batch size for correlator covariance computation",
    )
    p.add_argument(
        "--energy-batchsize",
        type=positive_int,
        default=8,
        help="Batch size for batched final energy computation",
    )

    # IO args
    p.add_argument(
        "--outdir",
        type=Path,
        default=Path("runs"),
        help="Directory where outputs are written",
    )
    p.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Optional run tag for output filenames",
    )
    p.add_argument(
        "--init-params",
        type=Path,
        default=None,
        help="Optional .npy file with initial parameters to continue training from",
    )

    return p


def make_run_name(args: argparse.Namespace) -> str:
    mlp_str = "x".join(str(x) for x in args.mlp)
    base = (
        f"L_{args.N}_g2_{args.g2}"
        f"_c_{args.channels}_layers_{args.layers}"
        f"_neighbors_{args.neighbors}_mlp_{mlp_str}"
        f"_samples_{args.samples}_steps_{args.steps}"
    )
    if args.tag:
        base += f"_{args.tag}"
    return base.replace("/", "-")


def save_summary_csv(path: Path, summary: dict) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["key", "value"])
        for key, value in summary.items():
            writer.writerow([key, value])


def main() -> None:
    args = build_parser().parse_args()

    seed = args.seed if args.seed is not None else int(time.time())
    g = float(np.sqrt(args.g2))
    residual_scale = args.residual_scale if args.residual_scale is not None else 0.25 / args.layers

    jax.config.update("jax_enable_x64", False)

    run_name = make_run_name(args)
    outdir = args.outdir / run_name
    outdir.mkdir(parents=True, exist_ok=True)

    # Store exact CLI invocation for reproducibility.
    with (outdir / "command.txt").open("w") as f:
        f.write("python " + " ".join(os.sys.argv) + "\n")

    x_init = np.random.default_rng(seed).normal(size=(args.N, 3)).astype(np.float32)

    model = GodSlayer3(
        num_layers=args.layers,
        num_channels=args.channels,
        num_neighbors=args.neighbors,
        residual_scale=residual_scale,
        mlp_hidden_sizes=tuple(args.mlp),
    )

    if args.init_params is not None:
        params = np.load(args.init_params, allow_pickle=True).item()
        init_source = str(args.init_params)
    else:
        params = model.init(jax.random.PRNGKey(seed), x_init)
        init_source = "random_init"

    psi = jax.jit(model.apply)
    sampler = newSamplerSharded(psi, (args.N, 3))

    mc_options = {
        "num_samples": args.samples,
        "thermalization": args.thermal,
        "skip": args.skip,
        "var": args.var,
        "nchains": args.nchains,
        "seeds": [seed + i for i in range(args.nchains)],
        "pos_initials": gen_init_positions(args.nchains, args.N, seed + 10_000),
        "chain": True,
    }

    print("=" * 80)
    print("Starting training")
    print(f"Run name          : {run_name}")
    print(f"Devices           : {jax.device_count()}")
    print(f"Seed              : {seed}")
    print(f"N                 : {args.N}")
    print(f"eta               : {args.eta}")
    print(f"g^2               : {args.g2}")
    print(f"g                 : {g}")
    print(f"mu                : {args.mu}")
    print(f"channels          : {args.channels}")
    print(f"layers            : {args.layers}")
    print(f"neighbors         : {args.neighbors}")
    print(f"mlp hidden sizes  : {tuple(args.mlp)}")
    print(f"residual_scale    : {residual_scale}")
    print(f"samples           : {args.samples}")
    print(f"thermalization    : {args.thermal}")
    print(f"skip              : {args.skip}")
    print(f"nchains           : {args.nchains}")
    print(f"steps             : {args.steps}")
    print(f"lr                : {args.lr}")
    print(f"batchsize         : {args.batchsize}")
    print(f"var               : {args.var}")
    print(f"adapt_rate        : {args.adapt_rate}")
    print(f"init source       : {init_source}")
    print(f"parameter count   : {count_flax_params(params)}")
    print("=" * 80, flush=True)

    t0 = time.time()
    results = sr_train_adapt_shard(
        (params, [], []),
        model,
        args.eta,
        g,
        args.mu,
        sampler,
        mc_options,
        args.steps,
        args.lr,
        fig=None,
        adapt_rate=args.adapt_rate,
        batch_size=args.batchsize,
    )
    wall_s = time.time() - t0

    final_params, energies, uncerts = results
    energies = np.asarray(energies)
    uncerts = np.asarray(uncerts)

    np.save(outdir / "final_params.npy", final_params, allow_pickle=True)
    np.savez(
        outdir / "history.npz",
        energies=energies,
        uncerts=uncerts,
    )

    print("Starting post-training evaluation", flush=True)
    eval_sampler = ClusterSamplerSharded(psi, (args.N, 3))
    eval_chains = args.eval_chains
    eval_sweeps_per_chain = max(1, args.eval_samples_total // eval_chains)
    eval_seed_base = seed + 1_000_000
    eval_pos_initials = gen_init_positions(eval_chains, args.N, eval_seed_base)
    eval_seeds = [eval_seed_base + i for i in range(eval_chains)]
    samples, eval_acc_rate = eval_sampler.run_many_chains(
        final_params,
        eval_sweeps_per_chain,
        args.eval_thermal,
        args.eval_skip,
        args.var,
        eval_pos_initials,
        eval_seeds,
    )
    final_energy_eval, final_uncert_eval = batched_energy(
        model, args.eta, g, args.mu, final_params, samples, batch_size=args.energy_batchsize
    )
    C, cov, C_uncerts = Cr_with_cov_optimized(samples, batch_size=args.corr_batchsize)

    samples_np = np.asarray(samples)
    C = np.asarray(C)
    cov = np.asarray(cov)
    C_uncerts = np.asarray(C_uncerts)
    xi, xi_err = xi_second_moment_from_corr(C, cov)

    np.savez(
        outdir / "evaluation.npz",
        samples=samples_np,
        acceptance_rate=np.asarray(eval_acc_rate),
        final_energy=np.asarray(final_energy_eval),
        final_uncert=np.asarray(final_uncert_eval),
    )
    np.savez(
        outdir / "correlators.npz",
        C=C,
        cov=cov,
        C_uncerts=C_uncerts,
        xi=np.asarray(xi),
        xi_err=np.asarray(xi_err),
    )

    config = vars(args).copy()
    config["outdir"] = str(args.outdir)
    config["init_params"] = None if args.init_params is None else str(args.init_params)
    config["g"] = g
    config["residual_scale_resolved"] = residual_scale
    config["seed_resolved"] = seed
    config["parameter_count"] = count_flax_params(final_params)
    config["wall_seconds"] = wall_s
    config["training_final_energy"] = float(energies[-1]) if len(energies) else None
    config["training_final_uncert"] = float(uncerts[-1]) if len(uncerts) else None
    config["evaluation_acceptance_rate"] = float(np.asarray(eval_acc_rate))
    config["evaluation_final_energy"] = float(np.asarray(final_energy_eval))
    config["evaluation_final_uncert"] = float(np.asarray(final_uncert_eval))
    config["xi_second_moment"] = float(np.asarray(xi))
    config["xi_second_moment_err"] = None if xi_err is None else float(np.asarray(xi_err))

    with (outdir / "config.json").open("w") as f:
        json.dump(config, f, indent=2)

    summary = {
        "run_name": run_name,
        "seed": seed,
        "N": args.N,
        "eta": args.eta,
        "g2": args.g2,
        "g": g,
        "mu": args.mu,
        "channels": args.channels,
        "layers": args.layers,
        "neighbors": args.neighbors,
        "mlp": "x".join(str(x) for x in args.mlp),
        "samples": args.samples,
        "thermal": args.thermal,
        "skip": args.skip,
        "lr": args.lr,
        "nchains": args.nchains,
        "steps": args.steps,
        "batchsize": args.batchsize,
        "var": args.var,
        "adapt_rate": args.adapt_rate,
        "parameter_count": count_flax_params(final_params),
        "wall_seconds": wall_s,
        "training_final_energy": float(energies[-1]) if len(energies) else "",
        "training_final_uncert": float(uncerts[-1]) if len(uncerts) else "",
        "evaluation_acceptance_rate": float(np.asarray(eval_acc_rate)),
        "evaluation_final_energy": float(np.asarray(final_energy_eval)),
        "evaluation_final_uncert": float(np.asarray(final_uncert_eval)),
        "xi_second_moment": float(np.asarray(xi)),
        "xi_second_moment_err": "" if xi_err is None else float(np.asarray(xi_err)),
    }
    save_summary_csv(outdir / "summary.csv", summary)

    print("Training complete")
    print(f"Output directory  : {outdir}")
    print(f"Wall time [s]     : {wall_s:.2f}")
    if len(energies):
        print(f"Training final energy      : {energies[-1]}")
        print(f"Training final uncertainty : {uncerts[-1]}")
    print(f"Evaluation acceptance rate : {float(np.asarray(eval_acc_rate))}")
    print(f"Evaluation final energy    : {float(np.asarray(final_energy_eval))}")
    print(f"Evaluation final uncertainty: {float(np.asarray(final_uncert_eval))}")
    print(f"Second-moment xi          : {float(np.asarray(xi))}")
    if xi_err is not None:
        print(f"Second-moment xi error    : {float(np.asarray(xi_err))}")
    print(f"Saved correlators          : {outdir / 'correlators.npz'}")


if __name__ == "__main__":
    main()
