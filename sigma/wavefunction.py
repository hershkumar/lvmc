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




def gram_matrix(x: jnp.ndarray) -> jnp.ndarray:
    """
    x: (..., N, 3)
    returns G: (..., N, N) with G[..., i, j] = n_i · n_j
    """
    return jnp.einsum("...id,...jd->...ij", x, x)


def translation_invariant_gram_features(G: jnp.ndarray) -> jnp.ndarray:
    """
    Make features invariant under cyclic shifts i -> i+s (PBC) by averaging
    the cyclic diagonals of the Gram matrix:

      c[r] = (1/N) sum_i G[i, (i+r) mod N],   r=0..N-1

    G: (..., N, N)
    returns c: (..., N)
    """
    N = G.shape[-1]

    def diag_mean(r):
        # shift columns so that diagonal picks out (i, i+r)
        Gs = jnp.roll(G, shift=-r, axis=-1)  # (..., N, N)
        d = jnp.diagonal(Gs, axis1=-2, axis2=-1)  # (..., N)
        return jnp.mean(d, axis=-1)  # (...)

    c = jax.vmap(diag_mean)(jnp.arange(N))  # (N, ...)
    return jnp.moveaxis(c, 0, -1)          # (..., N)


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

    # optional: if you want the same “FFT magnitude” style as MLP_TI
    use_fft: bool = False
    eps_fft: float = 0.0

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = jnp.asarray(x)
        if x.shape[-1] != 3:
            raise ValueError(f"Expected last dim = 3; got {x.shape}.")

        G = gram_matrix(x)                         # (..., N, N)
        c = translation_invariant_gram_features(G) # (..., N)

        if self.use_fft:
            Ck = jnp.fft.rfft(c, axis=-1)          # (..., N//2+1) complex
            feat = Ck.real * Ck.real + Ck.imag * Ck.imag
            if self.eps_fft != 0.0:
                feat = feat + self.eps_fft
        else:
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
