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

import functools

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



#### More optimized

def _celu(x: jnp.ndarray, alpha: float = 1.0) -> jnp.ndarray:
    xa = x / alpha
    neg = alpha * jnp.expm1(jnp.minimum(xa, 0.0))
    return jnp.maximum(x, 0.0) + neg


def _celu_prime(x: jnp.ndarray, alpha: float = 1.0) -> jnp.ndarray:
    xa = x / alpha
    return jnp.where(x > 0.0, 1.0, jnp.exp(jnp.minimum(xa, 0.0)))

    
def _stencil_apply(
    x: jnp.ndarray,      # (..., L, L, C)
    w: jnp.ndarray,      # (C, Nn+1)
    num_neighbors: int,
) -> jnp.ndarray:
    # center
    y = x * w[:, 0]

    # neighbors
    for d in range(1, num_neighbors + 1):
        wd = w[:, d]
        nbr = (
            jnp.roll(x, +d, axis=-2) +  # rows
            jnp.roll(x, -d, axis=-2) +
            jnp.roll(x, +d, axis=-3) +  # cols
            jnp.roll(x, -d, axis=-3)
        )
        y = y + nbr * wd

    return y


def make_godslayer2_layer(num_neighbors: int):
    @jax.custom_vjp
    def layer(x, w, residual_scale):
        y = _stencil_apply(x, w, num_neighbors)
        return x + residual_scale * _celu(y)

    def layer_fwd(x, w, residual_scale):
        y = _stencil_apply(x, w, num_neighbors)
        out = x + residual_scale * _celu(y)
        # save x, y, w for backward
        return out, (x, y, w, residual_scale)

    def layer_bwd(res, g_out):
        x, y, w, residual_scale = res

        delta = g_out * residual_scale * _celu_prime(y)

        # dL/dx = g_out + A(w)^T delta
        # operator is symmetric under periodic rolls, so A^T = A
        gx = g_out + _stencil_apply(delta, w, num_neighbors)

        # dL/dw
        grads = []

        # center weight
        gw0 = jnp.sum(delta * x, axis=tuple(range(delta.ndim - 1)))
        grads.append(gw0)

        # neighbor weights
        for d in range(1, num_neighbors + 1):
            nbr = (
                jnp.roll(x, +d, axis=-2) +
                jnp.roll(x, -d, axis=-2) +
                jnp.roll(x, +d, axis=-3) +
                jnp.roll(x, -d, axis=-3)
            )
            gwd = jnp.sum(delta * nbr, axis=tuple(range(delta.ndim - 1)))
            grads.append(gwd)

        gw = jnp.stack(grads, axis=-1)  # (C, Nn+1)

        # residual_scale gradient
        gs = jnp.sum(g_out * _celu(y))

        return gx, gw, gs

    layer.defvjp(layer_fwd, layer_bwd)
    return layer


def make_godslayer2_fused_stack_nhwc_fast(num_neighbors: int):
    layer = make_godslayer2_layer(num_neighbors)

    def fused_stack(x0, weights, residual_scale):
        def body(x, w):
            x = layer(x, w, residual_scale)
            return x, None

        x_final, _ = lax.scan(body, x0, weights)
        return x_final

    return fused_stack


class GodSlayer2(nn.Module):
    num_layers: int = 1
    num_neighbors: int = 1
    num_channels: int = 1
    param_dtype: jnp.dtype = jnp.float32
    residual_scale: float = 1.0

    @staticmethod
    def gram_matrix(n: jnp.ndarray) -> jnp.ndarray:
        return jnp.einsum("...ia,...ja->...ij", n, n)

    @staticmethod
    def godslayer_laplacian_init(
        num_layers, num_neighbors,
        self_val=-0.15, neigh_val=0.04, noise=0.05
    ):
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

    def setup(self):
        self.weights = self.param(
            "weights",
            self.godslayer_laplacian_init(self.num_layers, self.num_neighbors),
            (self.num_layers, self.num_channels, self.num_neighbors + 1),
            self.param_dtype,
        )
        self._fused_stack = make_godslayer2_fused_stack_nhwc_fast(self.num_neighbors)

    def __call__(self, n: jnp.ndarray) -> jnp.ndarray:
        if n.shape[-1] != 3:
            raise ValueError(f"Expected (..., L, 3), got {n.shape}")

        L = n.shape[-2]
        dtype = self.param_dtype

        g = self.gram_matrix(n).astype(dtype)
        g = g * (1.0 - jnp.eye(L, dtype=dtype))

        x0 = jnp.broadcast_to(
            g[..., :, :, None],
            g.shape[:-2] + (L, L, self.num_channels),
        )

        x = self._fused_stack(
            x0,
            self.weights,
            jnp.asarray(self.residual_scale, dtype=dtype),
        )

        out = jnp.mean(x, axis=(-1, -2, -3))
        return jnp.exp(-out)


class GodSlayer2_5(nn.Module):
    """
    Same stencil stack as GodSlayer2, but with a separation-resolved readout.

    After the fused stack, compute

        c_r = mean_i mean_c x_{i, i+r, c}

    and then

        out = sum_r sep_weights[r] * c_r

    This keeps translation invariance but does not collapse all separations equally.
    """
    num_layers: int = 1
    num_neighbors: int = 1
    num_channels: int = 1
    param_dtype: jnp.dtype = jnp.float32
    residual_scale: float = 1.0
    sep_weight_init: str = "mean"   # "mean" or "zero"
    symmetric_sep_weights: bool = False

    @staticmethod
    def gram_matrix(n: jnp.ndarray) -> jnp.ndarray:
        return jnp.einsum("...ia,...ja->...ij", n, n)

    @staticmethod
    def godslayer_laplacian_init(
        num_layers, num_neighbors,
        self_val=-0.15, neigh_val=0.04, noise=0.05
    ):
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

    @staticmethod
    def separation_readout_fast(h: jnp.ndarray) -> jnp.ndarray:
        """
        h: (..., L, L, C)

        returns:
            c: (..., L, C)

        c[..., r, c] = mean_i h[..., i, i+r mod L, c]
        """
        L = h.shape[-3]
        i = jnp.arange(L)
        r = jnp.arange(L)
        j = (i[:, None] + r[None, :]) % L

        vals = h[..., i[:, None], j, :]    # (..., L_i, L_r, C)
        return jnp.mean(vals, axis=-3)     # (..., L_r, C)

    @staticmethod
    def symmetrize_sep_weights(w: jnp.ndarray) -> jnp.ndarray:
        """
        Enforce w[r] = w[-r mod L].
        """
        return 0.5 * (w + jnp.roll(w[::-1], 1))

    def setup(self):
        self.weights = self.param(
            "weights",
            self.godslayer_laplacian_init(self.num_layers, self.num_neighbors),
            (self.num_layers, self.num_channels, self.num_neighbors + 1),
            self.param_dtype,
        )

        self._fused_stack = make_godslayer2_fused_stack_nhwc_fast(
            self.num_neighbors
        )

    @nn.compact
    def __call__(self, n: jnp.ndarray) -> jnp.ndarray:
        if n.shape[-1] != 3:
            raise ValueError(f"Expected (..., L, 3), got {n.shape}")

        L = n.shape[-2]
        dtype = self.param_dtype

        g = self.gram_matrix(n).astype(dtype)
        g = g * (1.0 - jnp.eye(L, dtype=dtype))

        x0 = jnp.broadcast_to(
            g[..., :, :, None],
            g.shape[:-2] + (L, L, self.num_channels),
        )

        x = self._fused_stack(
            x0,
            self.weights,
            jnp.asarray(self.residual_scale, dtype=dtype),
        )

        c = self.separation_readout_fast(x)     # (..., L, C)

        def sep_init(key, shape, dtype=jnp.float32):
            if self.sep_weight_init == "mean":
                # Starts close to GS2's mean readout.
                return jnp.ones(shape, dtype=dtype) / shape[0]
            elif self.sep_weight_init == "zero":
                return jnp.zeros(shape, dtype=dtype)
            else:
                raise ValueError("sep_weight_init must be 'mean' or 'zero'.")

        sep_weights = self.param(
            "sep_weights",
            sep_init,
            (L,),
            self.param_dtype,
        )

        if self.symmetric_sep_weights:
            sep_weights = self.symmetrize_sep_weights(sep_weights)

        # First average over channels, then weighted sum over separations.
        # c_mean: (..., L)
        c_mean = jnp.mean(c, axis=-1)

        # out: (...)
        out = jnp.einsum("...r,r->...", c_mean, sep_weights)

        return jnp.exp(-out)



class GodSlayer2_5_Fast(nn.Module):
    """
    Faster version of GodSlayer2_5.

    Main changes:
      1. Zeroes Gram diagonal with indexed update instead of multiplying by eye(L).
      2. Fuses channel mean + separation readout.
      3. Avoids constructing c = (..., L, C).
      4. Avoids gathering (..., L, L, C) during readout.
         Instead gathers from the channel-averaged matrix (..., L, L).
    """
    num_layers: int = 1
    num_neighbors: int = 1
    num_channels: int = 1
    param_dtype: jnp.dtype = jnp.float32
    residual_scale: float = 1.0
    sep_weight_init: str = "mean"
    symmetric_sep_weights: bool = False

    @staticmethod
    def gram_matrix(n: jnp.ndarray) -> jnp.ndarray:
        # For (..., L, 3), this returns (..., L, L).
        return jnp.einsum("...ia,...ja->...ij", n, n)

    @staticmethod
    def zero_diagonal(g: jnp.ndarray) -> jnp.ndarray:
        """
        Faster and less memory-hungry than

            g * (1 - eye(L))

        because it avoids constructing/broadcasting a full L x L mask.
        """
        L = g.shape[-1]
        idx = jnp.arange(L)
        return g.at[..., idx, idx].set(0.0)

    @staticmethod
    def godslayer_laplacian_init(
        num_layers,
        num_neighbors,
        self_val=-0.15,
        neigh_val=0.04,
        noise=0.05,
    ):
        def init(key, shape, dtype=jnp.float32):
            L_depth, C, K = shape

            depth = jnp.sqrt(jnp.asarray(float(num_layers), dtype=dtype))
            neigh = jnp.sqrt(jnp.asarray(float(max(1, num_neighbors)), dtype=dtype))

            w = jnp.zeros(shape, dtype=dtype)
            w = w.at[:, :, 0].set(jnp.asarray(self_val, dtype=dtype) / depth)

            if num_neighbors > 0:
                w = w.at[:, :, 1:].set(
                    jnp.asarray(neigh_val, dtype=dtype) / (depth * neigh)
                )

            if noise > 0:
                w = w + jnp.asarray(noise, dtype=dtype) * jax.random.normal(
                    key, shape, dtype
                )

            return w

        return init

    @staticmethod
    def symmetrize_sep_weights(w: jnp.ndarray) -> jnp.ndarray:
        """
        Enforce w[r] = w[-r mod L].
        """
        return 0.5 * (w + jnp.roll(w[::-1], 1))

    @staticmethod
    def weighted_separation_readout_fast(
        h: jnp.ndarray,
        sep_weights: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Computes

            out = sum_r sep_weights[r] * mean_i mean_c h[..., i, i+r, c]

        but avoids explicitly forming

            c[..., r, c] = mean_i h[..., i, i+r, c]

        and avoids gathering the full (..., L, L, C) tensor.

        h shape:
            (..., L, L, C)

        sep_weights shape:
            (L,)

        returns:
            (...,)
        """
        L = h.shape[-3]

        # First average over channels.
        # This reduces readout memory by a factor of C.
        h_mean = jnp.mean(h, axis=-1)  # (..., L, L)

        i = jnp.arange(L, dtype=jnp.int32)
        r = jnp.arange(L, dtype=jnp.int32)

        # Flat index for h_mean[..., i, i+r mod L].
        # idx shape: (L_i, L_r)
        idx = i[:, None] * L + ((i[:, None] + r[None, :]) % L)

        h_flat = h_mean.reshape(h_mean.shape[:-2] + (L * L,))

        # vals shape: (..., L_i, L_r)
        vals = jnp.take(h_flat, idx, axis=-1)

        # weighted_i shape: (..., L_i)
        weighted_i = jnp.einsum("...ir,r->...i", vals, sep_weights)

        # Average over i.
        return jnp.mean(weighted_i, axis=-1)

    def setup(self):
        self.weights = self.param(
            "weights",
            self.godslayer_laplacian_init(
                self.num_layers,
                self.num_neighbors,
            ),
            (self.num_layers, self.num_channels, self.num_neighbors + 1),
            self.param_dtype,
        )

        self._fused_stack = make_godslayer2_fused_stack_nhwc_fast(
            self.num_neighbors
        )

    @nn.compact
    def __call__(self, n: jnp.ndarray) -> jnp.ndarray:
        if n.shape[-1] != 3:
            raise ValueError(f"Expected (..., L, 3), got {n.shape}")

        L = n.shape[-2]
        dtype = self.param_dtype

        g = self.gram_matrix(n).astype(dtype)
        g = self.zero_diagonal(g)

        # Broadcast is lazy initially, but the fused stack will eventually
        # materialize an (..., L, L, C) working array.
        x0 = jnp.broadcast_to(
            g[..., :, :, None],
            g.shape[:-2] + (L, L, self.num_channels),
        )

        x = self._fused_stack(
            x0,
            self.weights,
            jnp.asarray(self.residual_scale, dtype=dtype),
        )

        def sep_init(key, shape, dtype=jnp.float32):
            if self.sep_weight_init == "mean":
                return jnp.ones(shape, dtype=dtype) / jnp.asarray(shape[0], dtype=dtype)
            elif self.sep_weight_init == "zero":
                return jnp.zeros(shape, dtype=dtype)
            else:
                raise ValueError("sep_weight_init must be 'mean' or 'zero'.")

        sep_weights = self.param(
            "sep_weights",
            sep_init,
            (L,),
            self.param_dtype,
        )

        if self.symmetric_sep_weights:
            sep_weights = self.symmetrize_sep_weights(sep_weights)

        out = self.weighted_separation_readout_fast(x, sep_weights)

        return jnp.exp(-out)



def transfer_learn(
    params: Any,
    new_num_layers: int,
    new_num_channels: int,
    *,
    new_layer_weight_scale: float = 0.0,
    channel_init: str = "zero",   # "zero", "repeat", "repeat_scale"
    return_frozen: bool = True,
):
    """
    Expand parameters for a residual CNN/GodSlayer-style model.

    Expected tree:
        params["params"]["weights"]  # shape (L, C, K)

    Optional:
        params["params"]["bias"]     # shape (L, C)

    Parameters
    ----------
    params
        Flax params tree, frozen or mutable.
    new_num_layers
        Target number of layers, must be >= old number.
    new_num_channels
        Target number of channels, must be >= old number.
    new_layer_weight_scale
        Fill value for newly added layers.
    channel_init
        How to initialize newly added channels in existing layers:
          - "zero": keep old function most closely
          - "repeat": copy old channels cyclically
          - "repeat_scale": repeat old channels and scale by old_C / new_C
    return_frozen
        If True and the input was frozen-like, return a FrozenDict.

    Returns
    -------
    new_params
        Expanded parameter tree.
    """
    was_frozen = type(params).__name__ == "FrozenDict"
    p = unfreeze(params) if was_frozen else copy.deepcopy(params)

    if "params" not in p or "weights" not in p["params"]:
        raise KeyError("Could not find params['params']['weights'].")

    old_w = p["params"]["weights"]
    if old_w.ndim != 3:
        raise ValueError(
            "Expected params['params']['weights'] to have shape "
            f"(num_layers, num_channels, width), got {old_w.shape}"
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
    if channel_init not in {"zero", "repeat", "repeat_scale"}:
        raise ValueError(
            "channel_init must be one of {'zero', 'repeat', 'repeat_scale'}"
        )

    dtype = old_w.dtype

    # ---- weights ----
    new_w = jnp.zeros((new_num_layers, new_num_channels, width), dtype=dtype)

    # Copy old block exactly
    new_w = new_w.at[:old_num_layers, :old_num_channels, :].set(old_w)

    # Expand channels for existing layers
    if new_num_channels > old_num_channels:
        n_extra = new_num_channels - old_num_channels

        if channel_init in {"repeat", "repeat_scale"}:
            extra_channel_idx = jnp.arange(old_num_channels, new_num_channels)
            src_channel_idx = extra_channel_idx % old_num_channels
            copied_channels = old_w[:, src_channel_idx, :]  # (old_L, n_extra, width)

            if channel_init == "repeat_scale":
                copied_channels = copied_channels * (
                    old_num_channels / new_num_channels
                )

                # Also scale the original copied block so total channel sum stays closer
                scaled_old = old_w * (old_num_channels / new_num_channels)
                new_w = new_w.at[:old_num_layers, :old_num_channels, :].set(scaled_old)

            new_w = new_w.at[:old_num_layers, old_num_channels:, :].set(copied_channels)

        elif channel_init == "zero":
            # Leave added channels at zero
            pass

    # Add new layers near identity / near no-op
    if new_num_layers > old_num_layers:
        n_new_layers = new_num_layers - old_num_layers
        added_layers = jnp.full(
            (n_new_layers, new_num_channels, width),
            fill_value=jnp.array(new_layer_weight_scale, dtype=dtype),
            dtype=dtype,
        )
        new_w = new_w.at[old_num_layers:, :, :].set(added_layers)

    p["params"]["weights"] = new_w

    # ---- optional bias ----
    if "bias" in p["params"]:
        old_b = p["params"]["bias"]
        if old_b.ndim != 2:
            raise ValueError(
                "Expected params['params']['bias'] to have shape "
                f"(num_layers, num_channels), got {old_b.shape}"
            )

        if old_b.shape != (old_num_layers, old_num_channels):
            raise ValueError(
                "Bias shape does not match weights shape: "
                f"bias={old_b.shape}, expected {(old_num_layers, old_num_channels)}"
            )

        new_b = jnp.zeros((new_num_layers, new_num_channels), dtype=old_b.dtype)
        new_b = new_b.at[:old_num_layers, :old_num_channels].set(old_b)

        if new_num_channels > old_num_channels:
            if channel_init in {"repeat", "repeat_scale"}:
                extra_channel_idx = jnp.arange(old_num_channels, new_num_channels)
                src_channel_idx = extra_channel_idx % old_num_channels
                copied_bias = old_b[:, src_channel_idx]

                if channel_init == "repeat_scale":
                    copied_bias = copied_bias * (old_num_channels / new_num_channels)
                    scaled_old_b = old_b * (old_num_channels / new_num_channels)
                    new_b = new_b.at[:old_num_layers, :old_num_channels].set(scaled_old_b)

                new_b = new_b.at[:old_num_layers, old_num_channels:].set(copied_bias)

        if new_num_layers > old_num_layers:
            # New-layer bias remains zero, which is usually safest
            pass

        p["params"]["bias"] = new_b

    if return_frozen and was_frozen:
        return freeze(p)
    return p


class GodSlayer3(nn.Module):
    num_layers: int = 1
    num_neighbors: int = 1
    num_channels: int = 1
    mlp_hidden_sizes: Sequence[int] = (64, 64)
    mlp_activation: Callable = nn.celu
    param_dtype: jnp.dtype = jnp.float32
    residual_scale: float = 1.0

    @staticmethod
    def gram_matrix(n: jnp.ndarray) -> jnp.ndarray:
        return jnp.einsum("...ia,...ja->...ij", n, n)

    @staticmethod
    def godslayer_laplacian_init(
        num_layers, num_neighbors,
        self_val=-0.15, neigh_val=0.04, noise=0.05
    ):
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

    @staticmethod
    def separation_readout_fast(h: jnp.ndarray) -> jnp.ndarray:
        L = h.shape[-3]
        i = jnp.arange(L)
        r = jnp.arange(L)
        j = (i[:, None] + r[None, :]) % L
        vals = h[..., i[:, None], j, :]     # (..., L, L, C)
        return jnp.mean(vals, axis=-3)      # (..., L, C)

    def setup(self):
        self.weights = self.param(
            "weights",
            self.godslayer_laplacian_init(self.num_layers, self.num_neighbors),
            (self.num_layers, self.num_channels, self.num_neighbors + 1),
            self.param_dtype,
        )
        self._fused_stack = make_godslayer2_fused_stack_nhwc_fast(self.num_neighbors)

        self.mlp_layers = [
            nn.Dense(width, param_dtype=self.param_dtype)
            for width in self.mlp_hidden_sizes
        ]
        self.out_layer = nn.Dense(1, param_dtype=self.param_dtype)

    @nn.compact
    def __call__(self, n: jnp.ndarray) -> jnp.ndarray:
        if n.shape[-1] != 3:
            raise ValueError(f"Expected (..., L, 3), got {n.shape}")

        L = n.shape[-2]
        dtype = self.param_dtype

        g = self.gram_matrix(n).astype(dtype)
        g = g * (1.0 - jnp.eye(L, dtype=dtype))

        x0 = jnp.broadcast_to(
            g[..., :, :, None],
            g.shape[:-2] + (L, L, self.num_channels),
        )

        h = self._fused_stack(
            x0,
            self.weights,
            jnp.asarray(self.residual_scale, dtype=dtype),
        )

        c = self.separation_readout_fast(h)
        feat = c.reshape(*c.shape[:-2], -1)

        y = feat
        for dense in self.mlp_layers:
            y = dense(y)
            y = self.mlp_activation(y)

        out = jnp.squeeze(self.out_layer(y), axis=-1)
        return jnp.exp(-out)






class GodSlayer3Excited(nn.Module):
    num_layers: int = 1
    num_neighbors: int = 1
    num_channels: int = 1
    mlp_hidden_sizes: Sequence[int] = (64, 64)
    mlp_activation: Callable = nn.celu
    param_dtype: jnp.dtype = jnp.float32
    residual_scale: float = 1.0

    @staticmethod
    def gram_matrix(n: jnp.ndarray) -> jnp.ndarray:
        return jnp.einsum("...ia,...ja->...ij", n, n)

    @staticmethod
    def godslayer_laplacian_init(
        num_layers, num_neighbors,
        self_val=-0.15, neigh_val=0.04, noise=0.05
    ):
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

    @staticmethod
    def separation_readout_fast(h: jnp.ndarray) -> jnp.ndarray:
        L = h.shape[-3]
        i = jnp.arange(L)
        r = jnp.arange(L)
        j = (i[:, None] + r[None, :]) % L
        vals = h[..., i[:, None], j, :]     # (..., L, L, C)
        return jnp.mean(vals, axis=-3)      # (..., L, C)

    def setup(self):
        self.weights = self.param(
            "weights",
            self.godslayer_laplacian_init(self.num_layers, self.num_neighbors),
            (self.num_layers, self.num_channels, self.num_neighbors + 1),
            self.param_dtype,
        )
        self._fused_stack = make_godslayer2_fused_stack_nhwc_fast(self.num_neighbors)

        self.mlp_layers = [
            nn.Dense(width, param_dtype=self.param_dtype)
            for width in self.mlp_hidden_sizes
        ]
        self.out_layer = nn.Dense(1, param_dtype=self.param_dtype)

    @nn.compact
    def __call__(self, n: jnp.ndarray) -> jnp.ndarray:
        if n.shape[-1] != 3:
            raise ValueError(f"Expected (..., L, 3), got {n.shape}")

        L = n.shape[-2]
        dtype = self.param_dtype
        
        g = self.gram_matrix(n).astype(dtype)
        g = g * (1.0 - jnp.eye(L, dtype=dtype))

        x0 = jnp.broadcast_to(
            g[..., :, :, None],
            g.shape[:-2] + (L, L, self.num_channels),
        )

        h = self._fused_stack(
            x0,
            self.weights,
            jnp.asarray(self.residual_scale, dtype=dtype),
        )

        c = self.separation_readout_fast(h)
        feat = c.reshape(*c.shape[:-2], -1)

        y = feat
        for dense in self.mlp_layers:
            y = dense(y)
            y = self.mlp_activation(y)

        out = jnp.squeeze(self.out_layer(y), axis=-1)
        Oz = jnp.sum(n, axis=-2)[..., 2]
        return jnp.exp(-out) * Oz


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





#### CNN implementation
Array = jax.Array


def normalize_rotors(x: Array, eps: float = 1e-12) -> Array:
    norm = jnp.linalg.norm(x, axis=-1, keepdims=True)
    return x / jnp.maximum(norm, eps)


def full_gram_matrix(x: Array) -> Array:
    """
    x shape:
        (..., N, 3)

    returns:
        (..., N, N)
    """
    return jnp.matmul(x, jnp.swapaxes(x, -1, -2))


def gram_to_obc_separation_features(
    G: Array,
    r_max: int,
    *,
    include_mask: bool = True,
    include_edge_features: bool = True,
) -> Array:
    """
    Vectorized OBC separation features from a full Gram matrix.

    G shape:
        (..., N, N)

    returns:
        (..., N, C)

    with
        X[..., i, r-1] = G[..., i, i+r]
    for valid OBC pairs i+r < N.
    """
    if G.shape[-1] != G.shape[-2]:
        raise ValueError(f"G must have shape (..., N, N), got {G.shape}")

    N = G.shape[-1]
    dtype = G.dtype

    if r_max < 1:
        raise ValueError("r_max must be >= 1")
    if r_max >= N:
        raise ValueError(f"Need r_max <= N-1. Got r_max={r_max}, N={N}")

    lead_shape = G.shape[:-2]
    nlead = len(lead_shape)

    i = jnp.arange(N, dtype=jnp.int32)[:, None]                 # (N, 1)
    r = jnp.arange(1, r_max + 1, dtype=jnp.int32)[None, :]      # (1, r_max)
    j = i + r                                                   # (N, r_max)

    valid = j < N
    j_safe = jnp.where(valid, j, 0)

    # Broadcast gather indices to (..., N, r_max)
    idx = j_safe.reshape((1,) * nlead + (N, r_max))
    idx = jnp.broadcast_to(idx, lead_shape + (N, r_max))

    sep = jnp.take_along_axis(G, idx, axis=-1)

    valid_b = valid.reshape((1,) * nlead + (N, r_max))
    valid_b = jnp.broadcast_to(valid_b, lead_shape + (N, r_max))

    sep = jnp.where(valid_b, sep, jnp.zeros((), dtype=dtype))

    features = [sep]

    if include_mask:
        features.append(valid_b.astype(dtype))

    if include_edge_features:
        if N == 1:
            pos = jnp.zeros((N,), dtype=dtype)
        else:
            pos = jnp.arange(N, dtype=dtype) / jnp.asarray(N - 1, dtype=dtype)

        left = pos
        right = 1.0 - pos
        center_dist = jnp.abs(2.0 * pos - 1.0)

        edge = jnp.stack([left, right, center_dist], axis=-1)  # (N, 3)
        edge = edge.reshape((1,) * nlead + (N, 3))
        edge = jnp.broadcast_to(edge, lead_shape + (N, 3))

        features.append(edge)

    return jnp.concatenate(features, axis=-1)


class FastMLP(nn.Module):
    hidden_dims: Sequence[int]
    out_dim: int
    activation: Callable[[Array], Array] = nn.celu
    use_bias: bool = True

    def setup(self):
        self.hidden = [
            nn.Dense(width, use_bias=self.use_bias)
            for width in self.hidden_dims
        ]
        self.out = nn.Dense(self.out_dim, use_bias=self.use_bias)

    def __call__(self, x: Array) -> Array:
        for layer in self.hidden:
            x = self.activation(layer(x))
        return self.out(x)


class FastResidualConv1DBlock(nn.Module):
    channels: int
    kernel_size: int = 3
    activation: Callable[[Array], Array] = nn.celu
    use_layernorm: bool = False
    residual_scale: float = 1.0
    use_bias: bool = True

    def setup(self):
        if self.use_layernorm:
            self.norm = nn.LayerNorm(axis=-1)

        self.conv1 = nn.Conv(
            features=self.channels,
            kernel_size=(self.kernel_size,),
            padding="SAME",
            use_bias=self.use_bias,
        )
        self.conv2 = nn.Conv(
            features=self.channels,
            kernel_size=(self.kernel_size,),
            padding="SAME",
            use_bias=self.use_bias,
        )

    def __call__(self, h: Array) -> Array:
        x = h
        if self.use_layernorm:
            x = self.norm(x)

        x = self.activation(self.conv1(x))
        x = self.conv2(x)

        return h + self.residual_scale * x


class CNN(nn.Module):
    """
    Faster OBC Gram-CNN.

    Compatible replacement for your CNN-style ansatz.

    Input:
        x shape (..., N, 3)

    Output:
        logpsi if return_log=True
        psi if return_log=False
    """

    r_max: int

    channels: int = 64
    num_layers: int = 2
    kernel_size: int = 3
    mlp_hidden_dims: Sequence[int] = (20,)

    output_scale: float = 0.05
    activation: Callable[[Array], Array] = nn.celu

    include_mask: bool = True
    include_edge_features: bool = True
    normalize_input: bool = False

    use_layernorm: bool = False
    use_bias: bool = True

    reflection_symmetrize: bool = False
    return_log: bool = False

    def setup(self):
        self.input_projection = nn.Dense(
            self.channels,
            use_bias=self.use_bias,
        )

        residual_scale = 1.0 / jnp.sqrt(float(max(self.num_layers, 1)))

        self.blocks = [
            FastResidualConv1DBlock(
                channels=self.channels,
                kernel_size=self.kernel_size,
                activation=self.activation,
                use_layernorm=self.use_layernorm,
                residual_scale=residual_scale,
                use_bias=self.use_bias,
            )
            for _ in range(self.num_layers)
        ]

        self.site_readout = FastMLP(
            hidden_dims=self.mlp_hidden_dims,
            out_dim=1,
            activation=self.activation,
            use_bias=self.use_bias,
        )

    def _forward_features(self, features: Array) -> Array:
        h = self.activation(self.input_projection(features))

        for block in self.blocks:
            h = block(h)

        site_logits = self.site_readout(h)
        site_logits = jnp.squeeze(site_logits, axis=-1)

        logpsi = jnp.sum(site_logits, axis=-1)
        return self.output_scale * logpsi

    def _features_from_gram(self, G: Array) -> Array:
        return gram_to_obc_separation_features(
            G,
            self.r_max,
            include_mask=self.include_mask,
            include_edge_features=self.include_edge_features,
        )

    def __call__(self, x: Array) -> Array:
        if self.normalize_input:
            x = normalize_rotors(x)

        # Compute the full Gram matrix once.
        G = full_gram_matrix(x)

        features = self._features_from_gram(G)
        
        logpsi = self._forward_features(features)

        return jnp.exp(jnp.clip(logpsi, -30.0, 30.0))
