import numpy as np
from jax import numpy as jnp
from jax import jit
from functools import partial
from flax import linen as nn
import jax
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


class CNN1D_TI(nn.Module):
    """
    Optimized translation-invariant 1D CNN for x (..., N, 3) with PBC (circular conv)
    + global pooling + MLP head.

    Optimizations vs your version:
      - Avoids explicit jnp.concatenate padding each layer (memory/copy heavy).
        Uses "gather with modulo" to implement circular padding (PBC) without materializing.
      - Uses LayerNorm over channel dim only (stable, cheaper) and optional residual blocks.
      - Keeps a configurable head via head_sizes.

    Notes:
      - Still translation-equivariant (ring) before pooling; pooling makes it invariant.
      - Requires odd kernel sizes for symmetric receptive field.
    """

    channels: Sequence[int] = (64, 64, 64)
    kernel_sizes: Sequence[int] = (5, 5, 5)
    activation: Callable = nn.celu
    use_layer_norm: bool = False
    use_residual: bool = True
    head_sizes: Sequence[int] = (128, 128)

    @staticmethod
    def _circular_pad_gather(h: jnp.ndarray, pad: int) -> jnp.ndarray:
        """
        Circular padding along site axis (-2) by indexing with modulo, without concat.
        h: (..., N, C)
        returns: (..., N + 2*pad, C)
        """
        if pad == 0:
            return h
        N = h.shape[-2]
        idx = (jnp.arange(-pad, N + pad) % N).astype(jnp.int32)  # (N+2pad,)
        return jnp.take(h, idx, axis=-2)

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = jnp.asarray(x)
        if x.shape[-1] != 3:
            raise ValueError(f"Expected last dim = 3; got {x.shape}.")
        if len(self.channels) != len(self.kernel_sizes):
            raise ValueError("channels and kernel_sizes must have the same length.")

        h = x  # (..., N, 3)

        for li, (c, k) in enumerate(zip(self.channels, self.kernel_sizes)):
            if k % 2 == 0:
                raise ValueError("Use odd kernel sizes for symmetric circular padding.")
            pad = k // 2

            h_in = h

            # circular padding without materializing concat
            h_pad = self._circular_pad_gather(h, pad)  # (..., N+2pad, Cin)

            # conv with VALID (now equivalent to circular conv)
            h = nn.Conv(
                features=c,
                kernel_size=(k,),
                strides=(1,),
                padding="VALID",
                use_bias=not self.use_layer_norm,  # bias often redundant with norm
                name=f"conv_{li}",
            )(h_pad)

            if self.use_layer_norm:
                # LayerNorm over channels (last dim) per site; stable for varying N
                h = nn.LayerNorm(name=f"ln_{li}")(h)

            h = self.activation(h)

            # Optional residual connection (with projection if channels change)
            if self.use_residual:
                if h_in.shape[-1] != c:
                    h_in_proj = nn.Dense(c, use_bias=False, name=f"res_proj_{li}")(h_in)
                else:
                    h_in_proj = h_in
                h = h + h_in_proj

        # translation-invariant pooling
        h = jnp.mean(h, axis=-2)  # (..., channels[-1])

        # MLP head (configurable)
        for i, width in enumerate(self.head_sizes):
            h = nn.Dense(width, name=f"head_dense_{i}")(h)
            h = self.activation(h)

        y = nn.Dense(1, name="out")(h)
        return jnp.squeeze(y, axis=-1)


def o3_local_invariants(x: jnp.ndarray):
    """
    x: (..., N, 3) unit (or approximately unit) vectors
    Returns:
      dots:   (..., N)   where dots[i] = n_i · n_{i+1}   (PBC)
      triples:(..., N)   where triples[i] = (n_i × n_{i+1}) · n_{i+2} (PBC)
    """
    x = jnp.asarray(x)
    if x.shape[-1] != 3:
        raise ValueError(f"Expected last dim = 3; got {x.shape}.")

    n1 = x
    n2 = jnp.roll(x, shift=-1, axis=-2)  # i+1
    n3 = jnp.roll(x, shift=-2, axis=-2)  # i+2

    dots = jnp.sum(n1 * n2, axis=-1)  # (..., N)
    triples = jnp.sum(jnp.cross(n1, n2) * n3, axis=-1)  # (..., N)

    return dots, triples


class MLP_O3_TI(nn.Module):
    """
    O(3)- and site-translation-invariant network for inputs x with shape (..., N, 3).

    Pipeline:
      1) Map x -> two length-N scalar sequences:
         a_i = n_i · n_{i+1}
         b_i = (n_i × n_{i+1}) · n_{i+2}
         (both with PBCs)
      2) Take DFT of each sequence along site axis, then take magnitude to remove shift phase.
      3) Concatenate the two magnitude spectra into a single vector (size 2N).
      4) Feed into a standard MLP to output a scalar.

    Note: b_i is a pseudoscalar (SO(3)-invariant, flips under reflections). If you need full
    O(3) invariance (including reflections), replace b_i by |b_i| or b_i^2.


    Btw this doesn't actually work, it doesn't account for correlations in the fields for points far away i.e n_i and n_i+r 
    for bigger r even though the actual fields are correlated
    """

    hidden_sizes: Sequence[int] = (128, 128)
    activation: Callable = nn.celu
    eps: float = 0.0
    enforce_full_O3: bool = True  # if True, uses b_i -> b_i^2 to remove parity sign

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = jnp.asarray(x)
        if x.shape[-1] != 3:
            raise ValueError(f"Expected last dim = 3 for Cartesian vectors; got {x.shape}.")

        # Build O(3) local invariants (with PBCs)
        a, b = o3_local_invariants(x)  # both (..., N)

        # If you truly want O(3) (incl. reflections), remove pseudoscalar sign:
        if self.enforce_full_O3:
            b = b * b  # or jnp.abs(b)

        # FFT over the site axis (axis=-1 because a,b are (..., N))
        Ak = jnp.fft.fft(a, axis=-1)  # (..., N), complex
        Bk = jnp.fft.fft(b, axis=-1)  # (..., N), complex

        # Magnitude -> translation (shift) invariant
        if self.eps == 0.0:
            feat_a = jnp.abs(Ak)  # (..., N)
            feat_b = jnp.abs(Bk)  # (..., N)
        else:
            feat_a = jnp.sqrt(Ak.real**2 + Ak.imag**2 + self.eps)
            feat_b = jnp.sqrt(Bk.real**2 + Bk.imag**2 + self.eps)

        # Concatenate into (..., 2N)
        feat = jnp.concatenate([feat_a, feat_b], axis=-1)

        # Standard MLP head
        h = feat
        for width in self.hidden_sizes:
            h = nn.Dense(width)(h)
            h = self.activation(h)

        y = nn.Dense(1)(h)
        return jnp.squeeze(y, axis=-1)


def corr_all_r(x):
    # x: (..., N, 3)
    N = x.shape[-2]
    def corr_r(r):
        xr = jnp.roll(x, shift=-r, axis=-2)
        return jnp.mean(jnp.sum(x * xr, axis=-1), axis=-1)  # (...) mean over sites
    return jax.vmap(corr_r)(jnp.arange(N)).swapaxes(0, -1)  # (..., N)


class MLP_TI_Gram(nn.Module):
    """
    O(3)-invariant (via Gram) and translation-invariant (via cyclic-diagonal averaging)
    MLP for inputs x (..., N, 3).

    Pipeline:
      1) (optional) normalize x to unit vectors
      2) compute Gram G_ij = n_i · n_j
      3) compute translation-invariant features c[r] = (1/N) sum_i G[i,i+r]
      4) (optional) take |rFFT(c)| (not necessary; c is already TI)
      5) standard MLP -> scalar
    """
    hidden_sizes: Sequence[int] = (128, 128)
    activation: Callable = nn.celu
    eps_norm: float = 1e-8


    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = jnp.asarray(x)
        if x.shape[-1] != 3:
            raise ValueError(f"Expected last dim = 3; got {x.shape}.")

        c = corr_all_r(x) # (..., N)

        feat = c                               # (..., N)

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
