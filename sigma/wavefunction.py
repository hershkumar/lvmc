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


class MLP_TI(nn.Module):
    """
    Site-translation invariant network for inputs x with shape (..., N, 3).
    Translation invariance is enforced by taking the magnitude of the DFT along
    the *site axis* (axis=-2), which removes the phase that encodes the origin/shift.
    """

    hidden_sizes: Sequence[int] = (128, 128)
    activation: Callable = nn.celu
    eps: float = (
        0.0  # optional: add a tiny eps inside sqrt for numerical stability if you want
    )

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = jnp.asarray(x)
        if x.shape[-1] != 3:
            raise ValueError(
                f"Expected last dim = 3 for Cartesian vectors; got {x.shape}."
            )

        # x: (..., N, 3)
        # FFT over the site axis (axis=-2), separately for each channel.
        Xk = jnp.fft.fft(x, axis=-2)  # (..., N, 3), complex
        feat = jnp.abs(Xk)  # (..., N, 3), real, shift-invariant

        # Optional: if you prefer a smooth magnitude (rarely needed)
        if self.eps != 0.0:
            feat = jnp.sqrt((Xk.real**2 + Xk.imag**2) + self.eps)

        # Flatten frequency-and-channel features into a single vector for the MLP.
        feat = feat.reshape(*feat.shape[:-2], -1)  # (..., N*3)

        # Standard MLP head
        h = feat
        for width in self.hidden_sizes:
            h = nn.Dense(width)(h)
            h = self.activation(h)

        y = nn.Dense(1)(h)
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
