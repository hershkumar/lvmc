#!/usr/bin/env python3
import os

# Set JAX env vars before importing jax.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
os.environ.setdefault("JAX_ENABLE_X64", "false")

import argparse
import json
import time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from sampling import newSamplerSharded, ClusterSamplerSharded
from wavefunction import GodSlayer3Excited
from training import sr_train_adapt_shard
from observables import make_batched_energy_sharded

try:
    from plotly import graph_objects as go
except Exception:
    go = None


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
    pos_initials = [
        jax.random.normal(jax.random.PRNGKey(seed + i), shape=(N, 3), dtype=jnp.float32)
        for i in range(nchains)
    ]
    for i, x in enumerate(pos_initials):
        pos_initials[i] = x / jnp.linalg.norm(x, axis=-1, keepdims=True)
    return pos_initials


def maybe_make_figure(enable: bool):
    if not enable or go is None:
        return None

    fig = go.FigureWidget()
    fig.add_trace(
        go.Scatter(x=[], y=[], mode="lines+markers", name="Energy", line=dict(color="blue"))
    )
    fig.add_trace(
        go.Scatter(
            x=[], y=[], mode="lines", line=dict(width=0), name="Energy + σ", showlegend=False
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[], y=[], mode="lines", fill="tonexty", fillcolor="rgba(0, 0, 255, 0.2)",
            line=dict(width=0), name="Energy ± σ"
        )
    )
    fig.update_layout(
        title="Energy with Uncertainty Band over Training Steps",
        xaxis_title="Training Step",
        yaxis_title="Energy",
    )
    return fig


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Two-stage SR training for GodSlayer3Excited with neighbor count controlled from the CLI."
    )
    p.add_argument("--neighbors", type=positive_int, required=True, help="Number of neighbors")
    p.add_argument("--outdir", type=Path, default=Path("neighbor_scaling_data"), help="Output directory")
    p.add_argument("--seed", type=int, default=None, help="Master seed; defaults to current unix time")

    # Fixed physics/model defaults from the notebook cell, but overridable.
    p.add_argument("--N", type=positive_int, default=40, help="Lattice size")
    p.add_argument("--eta", type=float, default=1.0, help="Eta coupling")
    p.add_argument("--g2", type=nonnegative_float, default=0.67, help="Value of g^2")
    p.add_argument("--mu", type=float, default=0.0, help="Chemical potential")
    p.add_argument("--channels", type=positive_int, default=1, help="Number of channels")
    p.add_argument("--layers", type=positive_int, default=5, help="Number of layers")
    p.add_argument(
        "--mlp", type=positive_int, nargs="+", default=[40], help="MLP hidden sizes"
    )
    p.add_argument(
        "--residual-scale", type=float, default=None,
        help="Residual scale; default is 1 / layers to match your cell"
    )

    # Optional warmup sampling call before training.
    p.add_argument("--warmup-chains", type=positive_int, default=5, help="Chains for pre-training sampler test")
    p.add_argument("--warmup-sweeps", type=positive_int, default=1000, help="Total warmup sweeps before division across chains")
    p.add_argument("--warmup-thermal", type=positive_int, default=10000, help="Thermalization for warmup sampler test")
    p.add_argument("--warmup-skip", type=positive_int, default=5, help="Skip for warmup sampler test")
    p.add_argument("--warmup-var", type=float, default=None, help="Proposal variance for warmup sampler test")

    # Stage A from the notebook.
    p.add_argument("--stage1-samples", type=positive_int, default=5000)
    p.add_argument("--stage1-thermal", type=positive_int, default=10000)
    p.add_argument("--stage1-skip", type=positive_int, default=10)
    p.add_argument("--stage1-var", type=float, default=None, help="Defaults to 1 / g^2")
    p.add_argument("--stage1-nchains", type=positive_int, default=5)
    p.add_argument("--stage1-lr", type=float, default=1e-1)
    p.add_argument("--stage1-steps", type=positive_int, default=500)
    p.add_argument("--stage1-adapt-rate", type=float, default=1.0)
    p.add_argument("--stage1-batchsize", type=positive_int, default=50000)

    # Stage B from the notebook.
    p.add_argument("--stage2-samples", type=positive_int, default=100000)
    p.add_argument("--stage2-thermal", type=positive_int, default=10000)
    p.add_argument("--stage2-skip", type=positive_int, default=10)
    p.add_argument("--stage2-var", type=float, default=0.3195)
    p.add_argument("--stage2-nchains", type=positive_int, default=50)
    p.add_argument("--stage2-lr", type=float, default=1e-2)
    p.add_argument("--stage2-steps", type=positive_int, default=500)
    p.add_argument("--stage2-adapt-rate", type=float, default=0.5)
    p.add_argument("--stage2-batchsize", type=positive_int, default=30000)

    # Final cluster evaluation from the notebook.
    p.add_argument("--eval-chains", type=positive_int, default=50)
    p.add_argument("--eval-total-samples", type=positive_int, default=2**20)
    p.add_argument("--eval-thermal", type=positive_int, default=50000)
    p.add_argument("--eval-skip", type=positive_int, default=10)
    p.add_argument("--eval-var", type=float, default=None, help="Defaults to 1 / g^2")
    p.add_argument("--eval-batchsize", type=positive_int, default=50000)

    p.add_argument("--with-figure", action="store_true", help="Create a Plotly FigureWidget during training")
    return p


def make_run_name(args: argparse.Namespace) -> str:
    mlp_str = "x".join(str(x) for x in args.mlp)
    return (
        f"L_{args.N}_g2_{args.g2}_neighbors_{args.neighbors}"
        f"_layers_{args.layers}_channels_{args.channels}_mlp_{mlp_str}"
    ).replace("/", "-")


def main() -> None:
    args = build_parser().parse_args()
    seed = args.seed if args.seed is not None else int(time.time())

    jax.config.update("jax_enable_x64", False)
    print(jax.device_count(), flush=True)

    devices = jax.local_devices()
    mesh = Mesh(devices, ("data",))
    _configs_sharding = NamedSharding(mesh, P("data", None, None))
    _replicated = NamedSharding(mesh, P())

    g = float(np.sqrt(args.g2))
    residual_scale = args.residual_scale if args.residual_scale is not None else 1.0 / args.layers
    stage1_var = args.stage1_var if args.stage1_var is not None else 1.0 / (g * g)
    eval_var = args.eval_var if args.eval_var is not None else 1.0 / (g * g)
    warmup_var = args.warmup_var if args.warmup_var is not None else stage1_var

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    run_name = make_run_name(args)

    x = np.random.default_rng(seed).normal(size=(args.N, 3)).astype(np.float32)
    model = GodSlayer3Excited(
        num_layers=args.layers,
        num_channels=args.channels,
        num_neighbors=args.neighbors,
        residual_scale=residual_scale,
        mlp_hidden_sizes=tuple(args.mlp),
    )

    params = model.init(jax.random.PRNGKey(seed), x)
    _ = model.apply(params, x)
    psi = jit(model.apply)

    fig = maybe_make_figure(args.with_figure)

    sampler = newSamplerSharded(psi, (args.N, 3))
    warmup_pos_initials = gen_init_positions(args.warmup_chains, args.N, seed + 1_000)
    warmup_seeds = [seed + 2_000 + i for i in range(args.warmup_chains)]
    warmup_sweeps_per_chain = max(1, args.warmup_sweeps // args.warmup_chains)
    warmup_samples, warmup_acc_rate = sampler.run_many_chains(
        params,
        warmup_sweeps_per_chain,
        args.warmup_thermal,
        args.warmup_skip,
        warmup_var,
        warmup_pos_initials,
        warmup_seeds,
    )
    print(f"Warmup samples shape: {np.asarray(warmup_samples).shape}")
    print(f"Warmup acceptance rate: {warmup_acc_rate}", flush=True)

    stage1_mc_options = {
        "num_samples": args.stage1_samples,
        "thermalization": args.stage1_thermal,
        "skip": args.stage1_skip,
        "var": stage1_var,
        "nchains": args.stage1_nchains,
        "seeds": [seed + 10_000 + i for i in range(args.stage1_nchains)],
        "pos_initials": gen_init_positions(args.stage1_nchains, args.N, seed + 20_000),
        "chain": True,
    }

    result_stage1 = sr_train_adapt_shard(
        (params, [], []),
        model,
        args.eta,
        g,
        args.mu,
        sampler,
        stage1_mc_options,
        args.stage1_steps,
        args.stage1_lr,
        fig=fig,
        adapt_rate=args.stage1_adapt_rate,
        batch_size=args.stage1_batchsize,
    )

    stage2_mc_options = {
        "num_samples": args.stage2_samples,
        "thermalization": args.stage2_thermal,
        "skip": args.stage2_skip,
        "var": args.stage2_var,
        "nchains": args.stage2_nchains,
        "seeds": [seed + 30_000 + i for i in range(args.stage2_nchains)],
        "pos_initials": gen_init_positions(args.stage2_nchains, args.N, seed + 40_000),
        "chain": True,
    }

    result_stage2 = sr_train_adapt_shard(
        result_stage1,
        model,
        args.eta,
        g,
        args.mu,
        sampler,
        stage2_mc_options,
        args.stage2_steps,
        args.stage2_lr,
        fig=fig,
        adapt_rate=args.stage2_adapt_rate,
        batch_size=args.stage2_batchsize,
    )

    final_params, stage2_energies, stage2_uncerts = result_stage2
    param_path = outdir / f"L_{args.N}_g2_{args.g2}_neighbors_{args.neighbors}_params.npy"
    np.save(param_path, final_params, allow_pickle=True)

    eval_sampler = ClusterSamplerSharded(psi, (args.N, 3))
    eval_sweeps_per_chain = max(1, args.eval_total_samples // args.eval_chains)
    eval_pos_initials = gen_init_positions(args.eval_chains, args.N, seed + 50_000)
    eval_seeds = [seed + 60_000 + i for i in range(args.eval_chains)]
    eval_samples, eval_acc_rate = eval_sampler.run_many_chains(
        final_params,
        eval_sweeps_per_chain,
        args.eval_thermal,
        args.eval_skip,
        eval_var,
        eval_pos_initials,
        eval_seeds,
    )
    print(f"Acceptance rate: {eval_acc_rate}", flush=True)

    get_e = make_batched_energy_sharded(model, args.eta, g, args.mu, mesh, batch_size=args.eval_batchsize)
    final_energy, final_uncert = get_e(final_params, eval_samples)

    results_path = outdir / f"L_{args.N}_g2_{args.g2}_neighbors_{args.neighbors}_results.npz"
    np.savez(
        results_path,
        stage2_energies=np.asarray(stage2_energies),
        stage2_uncerts=np.asarray(stage2_uncerts),
        eval_acceptance_rate=np.asarray(eval_acc_rate),
        final_energy=np.asarray(final_energy),
        final_uncert=np.asarray(final_uncert),
    )

    config = {
        "seed": seed,
        "N": args.N,
        "eta": args.eta,
        "g2": args.g2,
        "g": g,
        "mu": args.mu,
        "channels": args.channels,
        "layers": args.layers,
        "neighbors": args.neighbors,
        "mlp_hidden_sizes": list(args.mlp),
        "residual_scale": residual_scale,
        "stage1_var": stage1_var,
        "stage2_var": args.stage2_var,
        "eval_var": eval_var,
        "param_path": str(param_path),
        "results_path": str(results_path),
        "final_energy": float(np.asarray(final_energy)),
        "final_uncert": float(np.asarray(final_uncert)),
        "eval_acceptance_rate": float(np.asarray(eval_acc_rate)),
    }
    with (outdir / f"L_{args.N}_g2_{args.g2}_neighbors_{args.neighbors}_config.json").open("w") as f:
        json.dump(config, f, indent=2)

    print(f"Saved parameters to: {param_path}")
    print(f"Saved results to   : {results_path}")
    print(f"Final energy       : {float(np.asarray(final_energy))}")
    print(f"Final uncertainty  : {float(np.asarray(final_uncert))}")


if __name__ == "__main__":
    main()
