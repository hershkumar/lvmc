import numpy as np
from jax import numpy as jnp
from jax import jit
from functools import partial
from flax import linen as nn
import jax
from typing import Sequence, Callable, Optional
from dataclasses import field


"""
Feedfoward network for outputting psi(n(x)) given field configuration n(x) in cartesian coordinates with constraint n(x).n(x)=1
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
        return jnp.exp(-jnp.squeeze(y, axis=-1))


class MLP_SO3(nn.Module):
    hidden_sizes: Sequence[int] = (128, 128)
    activation: Callable = nn.celu

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = jnp.asarray(x)
        x = pairwise_dot_products(x)
        x = x.reshape(*x.shape[:-2], -1)

        for h in self.hidden_sizes:
            x = nn.Dense(h)(x)
            x = self.activation(x)

        y = nn.Dense(1)(x)
        return jnp.exp(-jnp.squeeze(y, axis=-1))


class MLP_excited(nn.Module):
    hidden_sizes: Sequence[int] = (128, 128)
    activation: Callable = nn.celu

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = jnp.asarray(x)
        n = x                                    # (..., N, 3)

        g = pairwise_dot_products(x)             # (..., N, N)
        h = g.reshape(*g.shape[:-2], -1)         # (..., N*N)

        for w in self.hidden_sizes:
            h = nn.Dense(w)(h)
            h = self.activation(h)

        y = nn.Dense(1)(h)
        psi0_like = jnp.exp(-jnp.squeeze(y, axis=-1))   # scalar envelope

        Oz = jnp.sum(n, axis=-2)[..., 2]                # sum over sites, take z-component
        return Oz * psi0_like

def pairwise_dot_products(n, idx=False):
    """
    Compute all distinct pairwise dot products n[i]·n[j] for 0<=i<j<L.

    Args:
      n: array of shape (..., L, D)

    Returns:
      dots: array of shape (..., L*(L-1)//2) in the order of jnp.triu_indices(L, k=1)
      idx:  tuple (i_idx, j_idx) so dots[..., t] = (n[..., i_idx[t], :] * n[..., j_idx[t], :]).sum(-1)
    """
    n = jnp.asarray(n)
    L = n.shape[-2]
    i_idx, j_idx = jnp.triu_indices(L, k=1)

    ni = n[..., i_idx, :]  # (..., M, D)
    nj = n[..., j_idx, :]  # (..., M, D)
    dots = jnp.sum(ni * nj, axis=-1)  # (..., M)
    return (dots, (i_idx, j_idx)) if idx else dots


def rotx(a):
    ca, sa = jnp.cos(a), jnp.sin(a)
    z = jnp.zeros_like(a)
    o = jnp.ones_like(a)
    return jnp.stack([
        jnp.stack([ o,  z,  z], axis=-1),
        jnp.stack([ z, ca, -sa], axis=-1),
        jnp.stack([ z, sa,  ca], axis=-1),
    ], axis=-2)   # (..., 3, 3)

def roty(a):
    ca, sa = jnp.cos(a), jnp.sin(a)
    z = jnp.zeros_like(a)
    o = jnp.ones_like(a)
    return jnp.stack([
        jnp.stack([ ca,  z, sa], axis=-1),
        jnp.stack([  z,  o,  z], axis=-1),
        jnp.stack([-sa,  z, ca], axis=-1),
    ], axis=-2)   # (..., 3, 3)

def rotz(a):
    ca, sa = jnp.cos(a), jnp.sin(a)
    z = jnp.zeros_like(a)
    o = jnp.ones_like(a)
    return jnp.stack([
        jnp.stack([ca, -sa,  z], axis=-1),
        jnp.stack([sa,  ca,  z], axis=-1),
        jnp.stack([ z,   z,  o], axis=-1),
    ], axis=-2)   # (..., 3, 3)


def gram_matrix(n: jnp.ndarray) -> jnp.ndarray:
    """
    n: (..., L, 3)
    returns: (..., L, L) with G_ij = n_i · n_j
    """
    return jnp.einsum("...id,...jd->...ij", n, n)


def remove_diagonal(g: jnp.ndarray) -> jnp.ndarray:
    """
    Zero the diagonal of the Gram matrix.
    g: (..., L, L)
    """
    L = g.shape[-1]
    eye = jnp.eye(L, dtype=g.dtype)
    return g * (1.0 - eye)


def cyclic_diagonal_sums(x: jnp.ndarray) -> jnp.ndarray:
    """
    Read out translation-invariant features by separation:
      c_r = sum_i x_{i, i+r mod L}

    x: (..., C, L, L)
    returns: (..., C, L)
    """
    L = x.shape[-1]
    feats = []
    for r in range(L):
        xr = jnp.roll(x, shift=-r, axis=-1)
        diag = jnp.diagonal(xr, axis1=-2, axis2=-1)  # (..., C, L)
        feats.append(jnp.sum(diag, axis=-1))          # (..., C)
    return jnp.stack(feats, axis=-1)                  # (..., C, L)


class GodSlayer(nn.Module):
    """
    O(3)- and translation-invariant wavefunction on a set of L 3-vectors.

    Architecture:
      1) Compute Gram matrix G_ij = n_i · n_j
      2) Remove diagonal G_ii
      3) For each channel and each layer:
           y = W_self * x + sum_d W_nbr[d] * neighbor_shifts(x)
           x <- x + act(y)     (residual)
           x <- symmetrize(x)
      4) Read out cyclic diagonal sums by separation r
      5) Flatten and pass through a small MLP head
      6) Return psi = exp(-head)

    This keeps parameter count small while building in:
      - O(3) invariance via Gram matrix
      - translation invariance via PBC shifts and separation-based readout
    """
    n_layers: int
    n_neighbors: int = 1
    n_channels: int = 1
    hidden_sizes: Sequence[int] = (64,)
    activation: Callable = nn.celu
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, n: jnp.ndarray) -> jnp.ndarray:
        n = jnp.asarray(n, dtype=self.dtype)
        if n.shape[-1] != 3:
            raise ValueError(f"Expected input shape (..., L, 3), got {n.shape}")

        g = gram_matrix(n)              # (..., L, L)
        g = remove_diagonal(g)          # (..., L, L)
        L = g.shape[-1]

        w_self = self.param(
            "w_self",
            lambda key, shape: jnp.ones(shape, dtype=self.dtype),
            (self.n_channels, self.n_layers),
        )
        w_nbr = self.param(
            "w_nbr",
            lambda key, shape: 0.1 * jnp.ones(shape, dtype=self.dtype),
            (self.n_channels, self.n_layers, self.n_neighbors),
        )

        # (..., C, L, L)
        x = jnp.broadcast_to(
            g[..., None, :, :],
            g.shape[:-2] + (self.n_channels,) + g.shape[-2:],
        )

        for ell in range(self.n_layers):
            y = w_self[:, ell][None, :, None, None] * x

            for d in range(1, self.n_neighbors + 1):
                s = (
                    jnp.roll(x, shift=+d, axis=-1) +
                    jnp.roll(x, shift=-d, axis=-1) +
                    jnp.roll(x, shift=+d, axis=-2) +
                    jnp.roll(x, shift=-d, axis=-2)
                )
                y = y + w_nbr[:, ell, d - 1][None, :, None, None] * s

            # residual update
            x = x + self.activation(y)

            # explicit symmetrization
            x = 0.5 * (x + jnp.swapaxes(x, -1, -2))

            # keep diagonal removed
            eye = jnp.eye(L, dtype=x.dtype)
            x = x * (1.0 - eye)[None, None, :, :]

        # (..., C, L)
        sep_features = cyclic_diagonal_sums(x)

        # flatten channels and separations: (..., C*L)
        h = sep_features.reshape(*sep_features.shape[:-2], -1)

        for width in self.hidden_sizes:
            h = nn.Dense(width)(h)
            h = self.activation(h)

        y = nn.Dense(1)(h)
        y = jnp.squeeze(y, axis=-1)

        return jnp.exp(-y)


class GodSlayer2(nn.Module):
    num_layers: int
    num_neighbors: int
    num_channels: int = 1
    activation: Callable = nn.celu
    final_activation: Optional[Callable] = None
    use_bias: bool = False
    param_dtype: jnp.dtype = jnp.float32

    @staticmethod
    def gram_matrix(n: jnp.ndarray) -> jnp.ndarray:
        return jnp.einsum("...ia,...ja->...ij", n, n)

    def setup(self) -> None:
        self.weights = self.param(
            "weights",
            nn.initializers.lecun_normal(),
            (self.num_layers, self.num_channels, self.num_neighbors + 1),
            self.param_dtype,
        )
        if self.use_bias:
            self.bias = self.param(
                "bias",
                nn.initializers.zeros,
                (self.num_layers, self.num_channels),
                self.param_dtype,
            )

    def _all_neighbor_sums(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Compute neighbor sums for all distances 1..N_N in one batched operation.

        Instead of looping over d and calling roll 4 times each, we:
          1. Stack all shift amounts into a single array
          2. vmap a single pair of rolls over that array
          3. Sum i and j shifts together

        Parameters
        ----------
        x : (..., C, L, L)

        Returns
        -------
        out : (N_N, ..., C, L, L)
            out[d-1] = neighbor sum at distance d
        """
        shifts = jnp.arange(1, self.num_neighbors + 1)  # (N_N,)

        def single_distance_sum(d):
            return (
                jnp.roll(x, d,  axis=-1) +
                jnp.roll(x, -d, axis=-1) +
                jnp.roll(x, d,  axis=-2) +
                jnp.roll(x, -d, axis=-2)
            )

        # vmap over distances — compiles as a single batched gather rather than N_N separate ops
        return jax.vmap(single_distance_sum)(shifts)  # (N_N, ..., C, L, L)

    def __call__(self, n: jnp.ndarray) -> jnp.ndarray:
        if n.shape[-1] != 3:
            raise ValueError(f"Expected (..., L, 3), got {n.shape}")

        L = n.shape[-2]

        # Gram matrix, diagonal zeroed: (..., L, L)
        g = self.gram_matrix(n)
        g = g * (1.0 - jnp.eye(L, dtype=g.dtype))
        g = (g + g.mT) * 0.5 # symmetrize to let XLA do better
        # Expand to channels without broadcast copy: (..., C, L, L)
        # Tile via einsum — gives XLA a clean contraction to fuse downstream
        x = jnp.repeat(g[..., None, :, :], self.num_channels, axis=-3)

        # Python loop — let XLA unroll and fuse across all layers
        for layer in range(self.num_layers):
            w = self.weights[layer]          # (C, N_N+1)
            w_self   = w[:, 0]               # (C,)
            w_neigh  = w[:, 1:]              # (C, N_N)

            # Onsite term
            y = x * w_self[..., None, None]  # (..., C, L, L)

            if self.num_neighbors > 0:
                # All neighbor sums at once: (N_N, ..., C, L, L)
                ns = self._all_neighbor_sums(x)

                # Contract over neighbor distances in one einsum:
                #   w_neigh: (C, N_N)  ->  broadcast as (N_N, ..., C, 1, 1)
                # This fuses the weighted sum over d into a single matmul-like op
                y = y + jnp.einsum("d...cij,cd->...cij", ns, w_neigh)

            if self.use_bias:
                y = y + self.bias[layer][:, None, None]

            x = self.activation(y)

        out = jnp.sum(x, axis=(-1, -2, -3)) / (L * L)

        if self.final_activation is not None:
            out = self.final_activation(out)

        return jnp.exp(-out)


def canonicalize_so3_fast(n, eps=1e-15):
    """
    Same constraints as your canonicalize_so3, but avoids atan2/cos/sin composition.
      1) Rotate so n_plus -> +z
      2) Use residual z-rotation to make (rotated q)_x = 0
    n: (..., L, 3)
    returns: n_std (..., L, 3)
    """
    n = jnp.asarray(n)
    L = n.shape[-2]

    n_plus = jnp.sum(n, axis=-2)  # (...,3)
    signs = jnp.where((jnp.arange(L) % 2) == 0, 1.0, -1.0)
    q = jnp.sum(n * signs[:, None], axis=0)

    x, y, z = n_plus[..., 0], n_plus[..., 1], n_plus[..., 2]

    # r_xy = sqrt(x^2+y^2), nrm = sqrt(x^2+y^2+z^2)
    r_xy = jnp.sqrt(x*x + y*y)
    nrm  = jnp.sqrt(x*x + y*y + z*z)

    # For phi = atan2(y,x):  cos(phi)=x/r_xy, sin(phi)=y/r_xy
    cphi = x / r_xy
    sphi = y / r_xy

    # R1 = Rz(-phi): cos(-phi)=cphi, sin(-phi)=-sphi
    # [[ cphi,  sphi, 0],
    #  [-sphi,  cphi, 0],
    #  [   0,     0,  1]]
    z0 = jnp.zeros_like(cphi)
    o1 = jnp.ones_like(cphi)
    R1 = jnp.stack([
        jnp.stack([ cphi,  sphi, z0], axis=-1),
        jnp.stack([-sphi,  cphi, z0], axis=-1),
        jnp.stack([  z0,    z0,  o1], axis=-1),
    ], axis=-2)  # (...,3,3)

    # For theta = atan2(r_xy, z): cos(theta)=z/nrm, sin(theta)=r_xy/nrm
    cth = z / nrm
    sth = r_xy / nrm

    # R2 = Ry(-theta): sin(-theta)=-sth
    # [[ cth, 0, -sth],
    #  [  0, 1,   0 ],
    #  [ sth, 0,  cth]]
    z0 = jnp.zeros_like(cth)
    o1 = jnp.ones_like(cth)
    R2 = jnp.stack([
        jnp.stack([ cth, z0, -sth], axis=-1),
        jnp.stack([  z0, o1,   z0], axis=-1),
        jnp.stack([ sth, z0,  cth], axis=-1),
    ], axis=-2)  # (...,3,3)

    # q2 = R2 * (R1 * q)
    q1 = jnp.einsum("...ij,...j->...i", R1, q)
    q2 = jnp.einsum("...ij,...j->...i", R2, q1)
    a, b = q2[..., 0], q2[..., 1]

    rab = jnp.sqrt(a*a + b*b + eps*eps)

    # alpha = atan2(a,b): cos(alpha)=b/rab, sin(alpha)=a/rab
    ca = b / rab
    sa = a / rab

    # R3 = Rz(alpha):
    # [[ ca, -sa, 0],
    #  [ sa,  ca, 0],
    #  [  0,   0, 1]]
    z0 = jnp.zeros_like(ca)
    o1 = jnp.ones_like(ca)
    R3 = jnp.stack([
        jnp.stack([ ca, -sa, z0], axis=-1),
        jnp.stack([ sa,  ca, z0], axis=-1),
        jnp.stack([ z0,  z0, o1], axis=-1),
    ], axis=-2)  # (...,3,3)

    R = jnp.einsum("...ij,...jk->...ik", R3, jnp.einsum("...ij,...jk->...ik", R2, R1))
    #R = jnp.einsum("...ij,...jk->...ik", R2,R1)
    n_std = jnp.einsum("...ij,...Lj->...Li", R, n)

    # Optional: if you still want the exact same degeneracy behavior, keep it.
    # Note: any hard where introduces non-smoothness at the threshold.
    #deg1 = (jnp.linalg.norm(n_plus, axis=-1) < eps)
    #deg2 = (a*a + b*b) < (eps*eps)
    #deg = deg1 | deg2
    #n_std = jnp.where(deg[..., None, None], n, n_std)

    return n_std



def translation_invariant_features_real(x, eps=1e-8):
    """
    Real-valued fields -> real-valued translation-invariant features using the paper procedure.

    Inputs
    ------
    x : array
        Either shape (..., L)   (single real signal)
        or     shape (..., L, C) (C channels, e.g. C=3 components of O(3) field)

    Returns
    -------
    feats : real array
        For each channel independently, computes FFT y_l, then y~_l, then outputs only real numbers:
          [Re(y~_0), Re(y~_1), Re(y~_2..y~_{L-1}), Im(y~_2..y~_{L-1})]
        Shapes:
          - input (..., L)     -> output (..., 2L-2)
          - input (..., L, C)  -> output (..., C, 2L-2)
    """
    x = jnp.asarray(x)
    if x.ndim >= 2 and x.shape[-1] != x.shape[-2] and x.shape[-1] <= 64:
        # Treat as (..., L, C) channels in last axis
        # FFT along L axis = -2
        y = jnp.fft.fft(x, axis=-2)
        y_tilde = _translation_invariant_tilde_y(y, eps=eps, axis=-2)

        re = jnp.real(y_tilde)
        im = jnp.imag(y_tilde)

        # Build real-only feature vector along L axis
        feat = jnp.concatenate([re[..., 0:2, :], re[..., 2:, :], im[..., 2:, :]], axis=-2)  # (..., 2L-2, C)
        feat = jnp.swapaxes(feat, -1, -2)  # (..., C, 2L-2)
        return feat
    else:
        # Treat as (..., L)
        y = jnp.fft.fft(x, axis=-1)
        y_tilde = _translation_invariant_tilde_y(y, eps=eps, axis=-1)

        re = jnp.real(y_tilde)
        im = jnp.imag(y_tilde)

        feat = jnp.concatenate([re[..., 0:2], re[..., 2:], im[..., 2:]], axis=-1)  # (..., 2L-2)
        return feat



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
