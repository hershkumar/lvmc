import numpy as np
from jax import numpy as jnp
from jax import jit
from functools import partial
from flax import linen as nn
import jax
from typing import Sequence, Callable, Optional
from dataclasses import field
import copy
from typing import Any
from flax.core import freeze, unfreeze
from jax import lax

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


def remove_diagonal(g: jnp.ndarray, eye: jnp.ndarray) -> jnp.ndarray:
    """
    Zero the diagonal of the Gram matrix.
    g: (..., L, L)
    """
    L = g.shape[-1]
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
        L = g.shape[-1]
        eye = jnp.eye(L, dtype=g.dtype)
        g = remove_diagonal(g, eye)     # (..., L, L)
        

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


# ============================================================
# Activation
# ============================================================
def _celu(x: jnp.ndarray, alpha: float = 1.0) -> jnp.ndarray:
    xa = x / alpha
    neg = alpha * jnp.expm1(jnp.minimum(xa, 0.0))
    return jnp.maximum(x, 0.0) + neg


def _celu_prime(x: jnp.ndarray, alpha: float = 1.0) -> jnp.ndarray:
    xa = x / alpha
    return jnp.where(x > 0, jnp.ones_like(x), jnp.exp(jnp.minimum(xa, 0.0)))

def _build_kernel(w: jnp.ndarray, num_neighbors: int) -> jnp.ndarray:
    """
    w: (C, Nn+1)  — col 0 is w_self, cols 1: are w_neigh
    returns: (C, 1, K, K) depthwise conv kernel
    """
    C  = w.shape[0]
    Nn = num_neighbors
    K  = 2 * Nn + 1
    mid = Nn  # center index

    # Start from zeros, scatter in the weights
    k = jnp.zeros((C, K, K), dtype=w.dtype)

    # Self weight at center
    k = k.at[:, mid, mid].set(w[:, 0])

    # Neighbor weights — symmetric cross pattern
    for d in range(1, Nn + 1):
        v = w[:, d]                          # (C,)
        k = k.at[:, mid,     mid + d].set(v) # right
        k = k.at[:, mid,     mid - d].set(v) # left
        k = k.at[:, mid + d, mid    ].set(v) # down
        k = k.at[:, mid - d, mid    ].set(v) # up

    return k[:, None, :, :]  # (C, 1, K, K) for depthwise


def _apply_layer(
    x: jnp.ndarray,       # (..., C, L, L)
    w: jnp.ndarray,       # (C, Nn+1)
    num_neighbors: int,
) -> jnp.ndarray:
    C = x.shape[-3]
    L = x.shape[-1]
    batch_shape = x.shape[:-3]

    kernel = _build_kernel(w, num_neighbors)   # (C, 1, K, K)
    x_flat = x.reshape(-1, C, L, L)            # (B, C, L, L)

    p = num_neighbors
    x_pad = jnp.pad(
        x_flat,
        ((0, 0), (0, 0), (p, p), (p, p)),
        mode="wrap",
    )

    y = lax.conv_general_dilated(
        x_pad,
        kernel,
        window_strides=(1, 1),
        padding="VALID",
        feature_group_count=C,
        dimension_numbers=("NCHW", "OIHW", "NCHW"),
    )

    return y.reshape(batch_shape + (C, L, L))



# ============================================================
# Fused residual stack with one custom VJP
# ============================================================

def make_godslayer2_fused_stack(num_neighbors: int):

    def fused_stack(x0, weights, residual_scale):
        s = jnp.asarray(residual_scale, dtype=x0.dtype)

        def layer_forward(x_in, w):
            y     = _apply_layer(x_in, w, num_neighbors)
            x_out = x_in + s * _celu(y)
            return x_out, None

        x_final, _ = lax.scan(layer_forward, x0, weights)
        return x_final

    return fused_stack

# ============================================================
# Module
# ============================================================

class GodSlayer2(nn.Module):
    num_layers:     int            = 1
    num_neighbors:  int            = 1
    num_channels:   int            = 1
    activation:     Callable       = nn.celu
    param_dtype:    jnp.dtype      = jnp.float32
    residual_scale: float          = 1.0

    @staticmethod
    def gram_matrix(n: jnp.ndarray) -> jnp.ndarray:
        return jnp.einsum("...ia,...ja->...ij", n, n)

    @staticmethod
    def godslayer_laplacian_init(num_layers, num_neighbors, self_val=-0.15, neigh_val=0.04, noise=0.01):
        def init(key, shape, dtype=jnp.float32):
            L, C, K = shape
            depth = jnp.sqrt(float(num_layers))
            neigh = jnp.sqrt(float(max(1, num_neighbors)))
            w = jnp.zeros(shape, dtype=dtype)
            w = w.at[:, :, 0].set(self_val / depth)
            if num_neighbors > 0:
                w = w.at[:, :, 1:].set(neigh_val / (depth * neigh))
            if noise > 0:
                w = w + noise * jax.random.normal(key, shape, dtype)
            return w
        return init

    def setup(self) -> None:
        self.weights = self.param(
            "weights",
            self.godslayer_laplacian_init(self.num_layers, self.num_neighbors),
            (self.num_layers, self.num_channels, self.num_neighbors + 1),
            self.param_dtype,
        )
        # Lift closure creation out of __call__ so it isn't re-traced each forward pass.
        self._fused_stack = make_godslayer2_fused_stack(self.num_neighbors)

    def __call__(self, n: jnp.ndarray) -> jnp.ndarray:
        if n.shape[-1] != 3:
            raise ValueError(f"Expected (..., L, 3), got {n.shape}")

        L     = n.shape[-2]
        dtype = self.param_dtype

        g   = self.gram_matrix(n).astype(dtype)
        eye = jnp.eye(L, dtype=dtype)
        g   = g * (1.0 - eye)

        x0 = jnp.broadcast_to(
            g[..., None, :, :],
            g.shape[:-2] + (self.num_channels, L, L),
        )

        s = jnp.asarray(self.residual_scale, dtype=dtype)
        x = self._fused_stack(x0, self.weights, s)

        out = jnp.mean(x, axis=(-1, -2, -3))
        return jnp.exp(-out)





def transfer_learn(
    params: Any,
    new_num_layers: int,
    new_num_channels: int,
    *,
    new_layer_weight_scale: float = 0.0,
    freeze_output: bool = True,
):
    """
    Expand parameters for the residual GodSlayer2 model to a larger number of
    layers and/or channels.

    Assumes the parameter tree contains:
        params["params"]["weights"]

    with shape:
        (old_num_layers, old_num_channels, num_neighbors + 1)

    Behavior
    --------
    1. Existing weights are copied exactly.
    2. If new_num_channels > old_num_channels:
         new channels are created by repeating old channels cyclically.
    3. If new_num_layers > old_num_layers:
         new layers are appended with very small / zero weights so that,
         in the residual architecture,
             x_out = x_in + residual_scale * activation(y),
         the added layers initially do almost nothing.

    Parameters
    ----------
    params
        Flax params tree. Can be a FrozenDict or nested dict.
    new_num_layers
        Target number of layers. Must be >= old number of layers.
    new_num_channels
        Target number of channels. Must be >= old number of channels.
    new_layer_weight_scale
        Value used to initialize all weights in newly added layers.
        For the residual architecture, 0.0 is the natural choice if you want
        the added layers to start near "doing nothing".
    freeze_output
        If True and the input was FrozenDict-like, return a FrozenDict.

    Returns
    -------
    new_params
        Expanded parameter tree.
    """
    # Make a mutable copy
    was_frozen = hasattr(params, "unfreeze")
    p = unfreeze(params) if was_frozen else copy.deepcopy(params)

    try:
        old_w = p["params"]["weights"]
    except KeyError as e:
        raise KeyError(
            "Could not find params['params']['weights'] in the provided parameter tree."
        ) from e

    if old_w.ndim != 3:
        raise ValueError(
            f"Expected params['params']['weights'] to have shape "
            f"(num_layers, num_channels, num_neighbors + 1), got {old_w.shape}"
        )

    old_num_layers, old_num_channels, width = old_w.shape

    if new_num_layers < old_num_layers:
        raise ValueError(
            f"new_num_layers={new_num_layers} must be >= old_num_layers={old_num_layers}"
        )
    if new_num_channels < old_num_channels:
        raise ValueError(
            f"new_num_channels={new_num_channels} must be >= old_num_channels={old_num_channels}"
        )

    dtype = old_w.dtype

    # Allocate new weight tensor
    new_w = jnp.zeros((new_num_layers, new_num_channels, width), dtype=dtype)

    # Copy overlapping block exactly
    new_w = new_w.at[:old_num_layers, :old_num_channels, :].set(old_w)

    # Expand channels in existing layers by repeating old channels cyclically
    if new_num_channels > old_num_channels:
        extra_channel_idx = jnp.arange(old_num_channels, new_num_channels)
        src_channel_idx = extra_channel_idx % old_num_channels
        copied_channels = old_w[:, src_channel_idx, :]  # (old_layers, extra_channels, width)
        new_w = new_w.at[:old_num_layers, old_num_channels:, :].set(copied_channels)

    # Add new layers with near-zero weights so residual blocks start near identity
    if new_num_layers > old_num_layers:
        n_new_layers = new_num_layers - old_num_layers

        added_layers = jnp.full(
            (n_new_layers, new_num_channels, width),
            jnp.asarray(new_layer_weight_scale, dtype=dtype),
            dtype=dtype,
        )

        new_w = new_w.at[old_num_layers:, :, :].set(added_layers)

    p["params"]["weights"] = new_w

    if freeze_output and was_frozen:
        return freeze(p)
    return p







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
