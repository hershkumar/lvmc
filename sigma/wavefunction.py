import numpy as np
from jax import numpy as jnp
from jax import jit
from functools import partial
from flax import linen as nn
from typing import Sequence, Callable, Optional

"""
Feedfoward network for outputting psi(U) given field configuration U in spherical basis
"""


class MLP(nn.Module):
    hidden_sizes: Sequence[int] = (128, 128)
    activation: Callable = nn.celu

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = jnp.asarray(x)
        x = x.reshape(*x.shape[:-2], -1)

        for h in self.hidden_sizes:
            x = nn.Dense(h)(x)
            x = self.activation(x)

        y = nn.Dense(1)(x)
        return jnp.squeeze(y, axis=-1)




class TI_CNN(nn.Module):
    """1D translation-invariant CNN with 2 features per lattice site (periodic BCs).

    Expected input shape: (..., L, 2)
    Output shape: (...)   (scalar per leading batch-like index)
    """
    conv_features: Sequence[int] = (64, 64, 64)
    mlp_hidden_sizes: Sequence[int] = (128, 128)
    activation: Callable = nn.celu
    pool: str = "mean"  # "mean" | "sum" | "max"
    kernel_size: int = 3

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = jnp.asarray(x)

        if x.ndim < 2:
            raise ValueError(f"Expected shape (..., L, 2), got {x.shape}")
        if x.shape[-1] != 2:
            raise ValueError(f"Expected last axis to be 2, got {x.shape[-1]}")

        # 1D conv stack over the lattice axis (the axis right before channels).
        for f in self.conv_features:
            x = nn.Conv(
                features=f,
                kernel_size=(self.kernel_size,),
                padding="CIRCULAR",  # periodic lattice
                use_bias=True,
            )(x)
            x = self.activation(x)

        # Global pooling over the lattice axis -> translation invariance.
        lattice_axis = x.ndim - 2  # the L axis
        if self.pool == "mean":
            x = jnp.mean(x, axis=lattice_axis)
        elif self.pool == "sum":
            x = jnp.sum(x, axis=lattice_axis)
        elif self.pool == "max":
            x = jnp.max(x, axis=lattice_axis)
        else:
            raise ValueError(f"Unknown pool='{self.pool}' (use 'mean', 'sum', or 'max').")

        # MLP head.
        for h in self.mlp_hidden_sizes:
            x = nn.Dense(h)(x)
            x = self.activation(x)

        y = nn.Dense(1)(x)
        return jnp.squeeze(y, axis=-1)


"""
Helper functions to convert field configurations between spherical and cartesian
"""


def spherical_to_cartesian(angles: jnp.ndarray) -> jnp.ndarray:
    angles = jnp.asarray(angles)

    theta = angles[:, 0]
    phi = angles[:, 1]

    sin_theta = jnp.sin(theta)

    x = sin_theta * jnp.cos(phi)
    y = sin_theta * jnp.sin(phi)
    z = jnp.cos(theta)

    return jnp.stack([x, y, z], axis=-1)


def cartesian_to_spherical(vectors: jnp.ndarray) -> jnp.ndarray:
    vectors = jnp.asarray(vectors)

    x = vectors[:, 0]
    y = vectors[:, 1]
    z = vectors[:, 2]

    r = jnp.sqrt(x**2 + y**2 + z**2)
    r = jnp.where(r == 0.0, 1.0, r)  # avoid divide-by-zero

    theta = jnp.arccos(jnp.clip(z / r, -1.0, 1.0))
    phi = jnp.arctan2(y, x)

    return jnp.stack([theta, phi], axis=-1)
