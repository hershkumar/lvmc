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




##### CNN_OPT

Array = jax.Array

def normalize_rotors(x: Array, eps: float = 1e-12) -> Array:
    norm = jnp.linalg.norm(x, axis=-1, keepdims=True)
    return x / jnp.maximum(norm, eps)

def extract_obc_separation_features(
    x: Array,
    r_max: int,
    *,
    include_mask: bool = True,
    include_edge_features: bool = True,
) -> Array:
    """
    Directly extracts OBC separation features in O(N * r_max) time.
    Bypasses the memory-heavy O(N^2) full Gram matrix calculation entirely.
    """
    N = x.shape[-2]
    dtype = x.dtype
    lead_shape = x.shape[:-2]

    if r_max < 1:
        raise ValueError("r_max must be >= 1")
    if r_max >= N:
        raise ValueError(f"Need r_max <= N-1. Got r_max={r_max}, N={N}")

    # 1. Efficient Shifted Dot-Products
    seps = []
    for r in range(1, r_max + 1):
        # Dot product of x[i] and x[i+r]
        # x_base: (..., 0:N-r, 3), x_shifted: (..., r:N, 3)
        dot = jnp.sum(x[..., :-r, :] * x[..., r:, :], axis=-1)
        
        # Pad with r zeros at the end to maintain sequence length N
        pad_width = [(0, 0)] * len(lead_shape) + [(0, r)]
        dot_padded = jnp.pad(dot, pad_width)
        seps.append(dot_padded)

    # Shape: (..., N, r_max)
    sep = jnp.stack(seps, axis=-1)
    features = [sep]

    # 2. Mask Generation
    if include_mask:
        # The mask corresponds exactly to the unpadded valid indices (i + r < N)
        i = jnp.arange(N, dtype=jnp.int32)[:, None]
        r_vec = jnp.arange(1, r_max + 1, dtype=jnp.int32)[None, :]
        mask = (i + r_vec < N).astype(dtype)
        
        mask_b = jnp.broadcast_to(mask, lead_shape + (N, r_max))
        features.append(mask_b)

    # 3. Edge Features 
    if include_edge_features:
        if N == 1:
            pos = jnp.zeros((N,), dtype=dtype)
        else:
            pos = jnp.linspace(0.0, 1.0, N, dtype=dtype)

        left = pos
        right = 1.0 - pos
        center_dist = jnp.abs(2.0 * pos - 1.0)

        edge = jnp.stack([left, right, center_dist], axis=-1)  # (N, 3)
        edge_b = jnp.broadcast_to(edge, lead_shape + (N, 3))
        features.append(edge_b)

    return jnp.concatenate(features, axis=-1)


class FastMLP(nn.Module):
    hidden_dims: Sequence[int]
    out_dim: int
    activation: Callable[[Array], Array] = nn.celu
    use_bias: bool = True

    @nn.compact
    def __call__(self, x: Array) -> Array:
        for width in self.hidden_dims:
            x = nn.Dense(width, use_bias=self.use_bias)(x)
            x = self.activation(x)
        return nn.Dense(self.out_dim, use_bias=self.use_bias)(x)


class FastResidualConv1DBlock(nn.Module):
    channels: int
    kernel_size: int = 3
    activation: Callable[[Array], Array] = nn.celu
    use_layernorm: bool = False
    residual_scale: float = 1.0
    use_bias: bool = True

    @nn.compact
    def __call__(self, h: Array) -> Array:
        x = h
        if self.use_layernorm:
            x = nn.LayerNorm(axis=-1)(x)

        x = nn.Conv(
            features=self.channels, 
            kernel_size=(self.kernel_size,), 
            padding="SAME", 
            use_bias=self.use_bias
        )(x)
        x = self.activation(x)
        
        x = nn.Conv(
            features=self.channels, 
            kernel_size=(self.kernel_size,), 
            padding="SAME", 
            use_bias=self.use_bias
        )(x)

        return h + self.residual_scale * x


class CNN(nn.Module):
    """
    Faster OBC Gram-CNN.
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
        self.input_projection = nn.Dense(self.channels, use_bias=self.use_bias)
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

    def __call__(self, x: Array) -> Array:
        if self.normalize_input:
            x = normalize_rotors(x)

        # Replaced the O(N^2) Gram matrix with O(N*r_max) feature extractor
        features = extract_obc_separation_features(
            x,
            self.r_max,
            include_mask=self.include_mask,
            include_edge_features=self.include_edge_features,
        )
        
        logpsi = self._forward_features(features)

        # BUG FIX: Actually adhere to the return_log flag!
        if self.return_log:
            return logpsi
            
        return jnp.exp(jnp.clip(logpsi, -30.0, 30.0))





##### GS4

# ============================================================
# Basic utilities
# ============================================================

def normalize_rotors(x: Array, eps: float = 1e-12) -> Array:
    norm = jnp.linalg.norm(x, axis=-1, keepdims=True)
    return x / jnp.maximum(norm, eps)


def gram_matrix(n: Array) -> Array:
    """
    n:
        (..., L, 3)

    returns:
        (..., L, L), G_ij = n_i · n_j
    """
    return jnp.einsum("...ia,...ja->...ij", n, n)


def zero_diagonal_matrix(g: Array) -> Array:
    """
    g:
        (..., L, L)

    returns:
        g with g[..., i, i] = 0
    """
    L = g.shape[-1]
    idx = jnp.arange(L)
    return g.at[..., idx, idx].set(0.0)


def zero_diagonal_tensor(h: Array) -> Array:
    """
    h:
        (..., L, L, C)

    returns:
        h with h[..., i, i, :] = 0
    """
    L = h.shape[-3]
    idx = jnp.arange(L)
    return h.at[..., idx, idx, :].set(0.0)


def _celu(x: Array, alpha: float = 1.0) -> Array:
    xa = x / alpha
    neg = alpha * jnp.expm1(jnp.minimum(xa, 0.0))
    return jnp.maximum(x, 0.0) + neg


def _celu_prime(x: Array, alpha: float = 1.0) -> Array:
    xa = x / alpha
    return jnp.where(x > 0.0, 1.0, jnp.exp(jnp.minimum(xa, 0.0)))


# ============================================================
# PBC raw separation features
# ============================================================

def extract_pbc_raw_separation_features(
    n: Array,
    r_max: int,
    *,
    reflection_mode: str = "none",
) -> Array:
    """
    Direct PBC analogue of the optimized CNN feature extractor.

    n:
        (..., L, 3)

    returns:
        (..., L, F)

    For reflection_mode="none":

        F = r_max
        X[..., i, r-1] = n_i · n_{i+r mod L}

    For reflection_mode="average":

        F = r_max
        X[..., i, r-1] =
            0.5 * [n_i · n_{i+r} + n_i · n_{i-r}]

    For reflection_mode="both":

        F = 2 * r_max
        includes both forward and backward separations as separate channels.
    """
    if n.shape[-1] != 3:
        raise ValueError(f"Expected n shape (..., L, 3), got {n.shape}")

    L = n.shape[-2]

    if r_max < 0:
        raise ValueError("r_max must be >= 0")
    if r_max == 0:
        return jnp.zeros(n.shape[:-1] + (0,), dtype=n.dtype)
    if r_max >= L:
        raise ValueError(f"Need r_max <= L-1 for PBC. Got r_max={r_max}, L={L}.")

    if reflection_mode not in {"none", "average", "both"}:
        raise ValueError("reflection_mode must be one of {'none', 'average', 'both'}.")

    feats = []

    for r in range(1, r_max + 1):
        n_fwd = jnp.roll(n, shift=-r, axis=-2)
        dot_fwd = jnp.sum(n * n_fwd, axis=-1)

        if reflection_mode == "none":
            feats.append(dot_fwd)

        else:
            n_bwd = jnp.roll(n, shift=+r, axis=-2)
            dot_bwd = jnp.sum(n * n_bwd, axis=-1)

            if reflection_mode == "average":
                feats.append(0.5 * (dot_fwd + dot_bwd))
            elif reflection_mode == "both":
                feats.append(dot_fwd)
                feats.append(dot_bwd)

    return jnp.stack(feats, axis=-1)


# ============================================================
# GodSlayer 2D PBC stencil stack
# ============================================================

def _depthwise_pbc_stencil_apply(
    x: Array,
    w: Array,
    num_neighbors: int,
) -> Array:
    """
    PBC depthwise row/column stencil.

    x:
        (..., L, L, C)

    w:
        (C, num_neighbors + 1)

    returns:
        (..., L, L, C)
    """
    y = x * w[:, 0]

    for d in range(1, num_neighbors + 1):
        wd = w[:, d]

        nbr = (
            jnp.roll(x, shift=+d, axis=-3) +
            jnp.roll(x, shift=-d, axis=-3) +
            jnp.roll(x, shift=+d, axis=-2) +
            jnp.roll(x, shift=-d, axis=-2)
        )

        y = y + nbr * wd

    return y

# ============================================================
# SR-safe GodSlayer 2D PBC stencil stack
# ============================================================

def make_depthwise_godslayer_layer(num_neighbors: int):
    """
    SR-safe version.

    No custom_vjp. This is required because SR commonly uses jvp/linearize
    through the model, and JAX cannot apply forward-mode autodiff to a
    custom_vjp function.
    """

    def layer(x: Array, w: Array, residual_scale: Array) -> Array:
        y = _depthwise_pbc_stencil_apply(x, w, num_neighbors)
        return x + residual_scale * _celu(y)

    return layer


def make_fused_godslayer_cnn_stack(
    num_neighbors: int,
    *,
    symmetrize_each_layer: bool = True,
    zero_diagonal_each_layer: bool = True,
):
    """
    SR-safe fused GodSlayer stack.

    Each layer does:

        x <- x + residual_scale * CELU(depthwise_PBC_stencil(x))
        x <- x + mix_scale      * CELU(pointwise_channel_mix(x))

    This version is fully autodiff-compatible for Adam, reverse-mode grad,
    and SR/JVP-based Fisher matvecs.
    """
    depthwise_layer = make_depthwise_godslayer_layer(num_neighbors)

    def fused_stack(
        x0: Array,
        depthwise_weights: Array,
        mix_weights: Array,
        mix_bias: Array,
        residual_scale: Array,
        mix_scale: Array,
    ) -> Array:
        def body(x, inputs):
            wd, wm, bm = inputs

            # Depthwise GodSlayer stencil update.
            x = depthwise_layer(x, wd, residual_scale)

            # Pointwise channel mixing.
            z = jnp.einsum("...c,cd->...d", x, wm) + bm
            x = x + mix_scale * _celu(z)

            if symmetrize_each_layer:
                x = 0.5 * (x + jnp.swapaxes(x, -3, -2))

            if zero_diagonal_each_layer:
                x = zero_diagonal_tensor(x)

            return x, None

        x_final, _ = lax.scan(
            body,
            x0,
            (depthwise_weights, mix_weights, mix_bias),
        )

        return x_final

    return fused_stack


# ============================================================
# Extract PBC site/separation bands from evolved Gram tensor
# ============================================================
def extract_pbc_matrix_band_features(
    h: Array,
    r_max: int,
    *,
    reflection_mode: str = "none",
) -> Array:
    """
    h:
        (L, L, C) or (..., L, L, C)

    returns:
        (L, F) or (..., L, F)
    """
    if h.ndim < 3:
        raise ValueError(f"Expected h shape (L, L, C) or (..., L, L, C), got {h.shape}")

    L = h.shape[-3]

    if h.shape[-2] != L:
        raise ValueError(f"Expected square matrix axes in h, got {h.shape}")

    if r_max < 0:
        raise ValueError("r_max must be >= 0")

    if r_max == 0:
        return jnp.zeros(h.shape[:-3] + (L, 0), dtype=h.dtype)

    if r_max >= L:
        raise ValueError(f"Need r_max <= L-1 for PBC. Got r_max={r_max}, L={L}.")

    if reflection_mode not in {"none", "average", "both"}:
        raise ValueError("reflection_mode must be one of {'none', 'average', 'both'}.")

    i = jnp.arange(L, dtype=jnp.int32)
    bands = []

    for r in range(1, r_max + 1):
        jp = (i + r) % L
        vals_fwd = h[..., i, jp, :]  # (L, C) or (..., L, C)

        if reflection_mode == "none":
            bands.append(vals_fwd)

        else:
            jm = (i - r) % L
            vals_bwd = h[..., i, jm, :]

            if reflection_mode == "average":
                bands.append(0.5 * (vals_fwd + vals_bwd))
            elif reflection_mode == "both":
                bands.append(vals_fwd)
                bands.append(vals_bwd)

    return jnp.concatenate(bands, axis=-1)


def broadcast_global_separation_features(
    h: Array,
    r_max: int,
    *,
    reflection_mode: str = "average",
) -> Array:
    """
    Optional global separation summary.

    h:
        (..., L, L, C)

    returns:
        (..., L, F)

    It computes separation-band means over sites and broadcasts them back to
    every site. This gives the site-CNN access to global separation statistics
    without breaking translation invariance.
    """
    site_feats = extract_pbc_matrix_band_features(
        h,
        r_max,
        reflection_mode=reflection_mode,
    )  # (..., L, F)

    global_feats = jnp.mean(site_feats, axis=-2, keepdims=True)  # (..., 1, F)
    return jnp.broadcast_to(global_feats, site_feats.shape)


# ============================================================
# Circular Conv1D site readout
# ============================================================

class FastMLP(nn.Module):
    hidden_dims: Sequence[int]
    out_dim: int
    activation: Callable[[Array], Array] = nn.celu
    use_bias: bool = True
    param_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: Array) -> Array:
        for width in self.hidden_dims:
            x = nn.Dense(
                width,
                use_bias=self.use_bias,
                param_dtype=self.param_dtype,
            )(x)
            x = self.activation(x)

        return nn.Dense(
            self.out_dim,
            use_bias=self.use_bias,
            param_dtype=self.param_dtype,
        )(x)


def circular_pad_1d(x: Array, radius: int) -> Array:
    """
    Circularly pads the site axis.

    x:
        (..., L, C)

    returns:
        (..., L + 2 * radius, C)
    """
    if radius == 0:
        return x

    pad_width = [(0, 0)] * x.ndim
    pad_width[-2] = (radius, radius)
    return jnp.pad(x, pad_width, mode="wrap")


class CircularResidualConv1DBlock(nn.Module):
    channels: int
    kernel_size: int = 3
    activation: Callable[[Array], Array] = nn.celu
    use_layernorm: bool = False
    residual_scale: float = 1.0
    use_bias: bool = True
    param_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, h: Array) -> Array:
        if self.kernel_size < 1:
            raise ValueError("kernel_size must be >= 1")
        if self.kernel_size % 2 != 1:
            raise ValueError("Use odd kernel_size for circular SAME padding.")

        radius = self.kernel_size // 2

        x = h

        if self.use_layernorm:
            x = nn.LayerNorm(axis=-1)(x)

        x = circular_pad_1d(x, radius)
        x = nn.Conv(
            features=self.channels,
            kernel_size=(self.kernel_size,),
            padding="VALID",
            use_bias=self.use_bias,
            param_dtype=self.param_dtype,
        )(x)
        x = self.activation(x)

        x = circular_pad_1d(x, radius)
        x = nn.Conv(
            features=self.channels,
            kernel_size=(self.kernel_size,),
            padding="VALID",
            use_bias=self.use_bias,
            param_dtype=self.param_dtype,
        )(x)

        return h + self.residual_scale * x


class PBCCNNReadout(nn.Module):
    channels: int = 64
    num_layers: int = 2
    kernel_size: int = 3
    mlp_hidden_dims: Sequence[int] = (32,)

    output_scale: float = 0.05
    activation: Callable[[Array], Array] = nn.celu

    use_layernorm: bool = False
    use_bias: bool = True
    param_dtype: jnp.dtype = jnp.float32

    readout: str = "sum"
    return_log: bool = False
    clip_log: float = 30.0

    @nn.compact
    def __call__(self, features: Array) -> Array:
        if features.ndim < 2:
            raise ValueError(
                f"Expected features shape (L, F) or (..., L, F), got {features.shape}"
            )

        if self.readout not in {"sum", "mean"}:
            raise ValueError("readout must be 'sum' or 'mean'.")

        h = nn.Dense(
            self.channels,
            use_bias=self.use_bias,
            param_dtype=self.param_dtype,
        )(features)
        h = self.activation(h)

        residual_scale = 1.0 / jnp.sqrt(jnp.asarray(max(self.num_layers, 1), h.dtype))

        for _ in range(self.num_layers):
            h = CircularResidualConv1DBlock(
                channels=self.channels,
                kernel_size=self.kernel_size,
                activation=self.activation,
                use_layernorm=self.use_layernorm,
                residual_scale=residual_scale,
                use_bias=self.use_bias,
                param_dtype=self.param_dtype,
            )(h)

        site_logits = FastMLP(
            hidden_dims=self.mlp_hidden_dims,
            out_dim=1,
            activation=self.activation,
            use_bias=self.use_bias,
            param_dtype=self.param_dtype,
        )(h)

        site_logits = jnp.squeeze(site_logits, axis=-1)

        if self.readout == "sum":
            logpsi = jnp.sum(site_logits, axis=-1)
        else:
            logpsi = jnp.mean(site_logits, axis=-1)

        logpsi = self.output_scale * logpsi

        if self.return_log:
            return logpsi

        return jnp.exp(jnp.clip(logpsi, -self.clip_log, self.clip_log))
# ============================================================
# Initializers
# ============================================================

def godslayer_laplacian_init(
    num_layers: int,
    num_neighbors: int,
    *,
    self_val: float = -0.15,
    neigh_val: float = 0.04,
    noise: float = 0.02,
):
    """
    Initializes the depthwise 2D Gram stencil close to a weak Laplacian-like
    local smoothing/diffusion operator.
    """
    def init(key, shape, dtype=jnp.float32):
        # shape = (num_layers, channels, num_neighbors + 1)
        _, _, _ = shape

        depth = jnp.sqrt(jnp.asarray(float(max(num_layers, 1)), dtype=dtype))
        neigh = jnp.sqrt(jnp.asarray(float(max(num_neighbors, 1)), dtype=dtype))

        w = jnp.zeros(shape, dtype=dtype)

        w = w.at[:, :, 0].set(jnp.asarray(self_val, dtype=dtype) / depth)

        if num_neighbors > 0:
            w = w.at[:, :, 1:].set(
                jnp.asarray(neigh_val, dtype=dtype) / (depth * neigh)
            )

        if noise > 0:
            w = w + jnp.asarray(noise, dtype=dtype) * jax.random.normal(
                key,
                shape,
                dtype,
            )

        return w

    return init


def small_channel_mix_init(scale: float = 0.02):
    """
    Small near-identity channel mixer.

    The residual form is:

        x <- x + mix_scale * CELU(x @ W + b)

    so W can safely start small.
    """
    def init(key, shape, dtype=jnp.float32):
        return jnp.asarray(scale, dtype=dtype) * jax.random.normal(
            key,
            shape,
            dtype,
        )

    return init


def input_gain_init(noise: float = 0.02):
    def init(key, shape, dtype=jnp.float32):
        return jnp.ones(shape, dtype=dtype) + jnp.asarray(noise, dtype=dtype) * jax.random.normal(
            key,
            shape,
            dtype,
        )
    return init


# ============================================================
# Full ansatz
# ============================================================

class GodSlayer4(nn.Module):
    """
    Optimized PBC GodSlayer-CNN ansatz.

    This is meant to be the "full" production model.

    It combines:

      - raw PBC shifted dot-product features:
            n_i · n_{i+r}

      - full-Gram GodSlayer 2D stencil processing:
            h_ijc <- stencil/mix residual stack

      - site/separation extraction:
            h_i,i+r,c

      - circular Conv1D site processing:
            h_i -> CNN(h)_i

      - extensive invariant readout:
            logpsi = output_scale * sum_i site_logit_i

    Hyperparameter guidance:

      For speed:
          raw_r_max   = 8 to 20
          stack_r_max = 8 to 20
          stack_channels = 4 to 16
          cnn_channels = 32 to 128

      For maximal finite-L expressivity:
          raw_r_max   = L - 1
          stack_r_max = L - 1

    reflection_mode:
        "none":
            keeps oriented positive separations only.

        "average":
            enforces reflection/parity symmetry by averaging +r and -r.
            Good default for ground states.

        "both":
            includes both directions separately.
            Use if you want translation invariance but not forced reflection symmetry.
    """

    # Raw direct PBC separation features.
    raw_r_max: int = 10
    use_raw_features: bool = True

    # GodSlayer full-Gram stack.
    stack_r_max: int = 10
    stack_layers: int = 2
    stack_neighbors: int = 1
    stack_channels: int = 8
    stack_residual_scale: float = 1.0
    stack_mix_scale: float = 0.5

    # Feature options.
    use_stack_features: bool = True
    use_global_stack_features: bool = False
    reflection_mode: str = "average"

    # Site CNN readout.
    cnn_channels: int = 64
    cnn_layers: int = 2
    cnn_kernel_size: int = 3
    mlp_hidden_dims: Sequence[int] = (32,)

    # Output.
    output_scale: float = 0.05
    readout: str = "sum"       # "sum" for extensive logpsi, "mean" for size-normalized logpsi
    return_log: bool = False
    clip_log: float = 30.0

    # Numerics.
    activation: Callable[[Array], Array] = nn.celu
    param_dtype: jnp.dtype = jnp.float32
    normalize_input: bool = False
    use_layernorm: bool = False
    use_bias: bool = True

    # GodSlayer stack controls.
    symmetrize_each_layer: bool = True
    zero_diagonal_each_layer: bool = True

    # Initializer scales.
    laplacian_self_val: float = -0.15
    laplacian_neigh_val: float = 0.04
    laplacian_noise: float = 0.02
    input_gain_noise: float = 0.02
    channel_mix_init_scale: float = 0.02

    def setup(self):
        if not self.use_raw_features and not self.use_stack_features:
            raise ValueError("At least one of use_raw_features/use_stack_features must be True.")

        if self.reflection_mode not in {"none", "average", "both"}:
            raise ValueError("reflection_mode must be one of {'none', 'average', 'both'}.")

        if self.stack_layers < 0:
            raise ValueError("stack_layers must be >= 0")

        if self.stack_channels < 1:
            raise ValueError("stack_channels must be >= 1")

        self.input_gain = self.param(
            "input_gain",
            input_gain_init(self.input_gain_noise),
            (self.stack_channels,),
            self.param_dtype,
        )

        self.input_bias = self.param(
            "input_bias",
            nn.initializers.zeros,
            (self.stack_channels,),
            self.param_dtype,
        )

        self.depthwise_weights = self.param(
            "depthwise_weights",
            godslayer_laplacian_init(
                self.stack_layers,
                self.stack_neighbors,
                self_val=self.laplacian_self_val,
                neigh_val=self.laplacian_neigh_val,
                noise=self.laplacian_noise,
            ),
            (self.stack_layers, self.stack_channels, self.stack_neighbors + 1),
            self.param_dtype,
        )

        self.mix_weights = self.param(
            "mix_weights",
            small_channel_mix_init(self.channel_mix_init_scale),
            (self.stack_layers, self.stack_channels, self.stack_channels),
            self.param_dtype,
        )

        self.mix_bias = self.param(
            "mix_bias",
            nn.initializers.zeros,
            (self.stack_layers, self.stack_channels),
            self.param_dtype,
        )

        self._fused_stack = make_fused_godslayer_cnn_stack(
            self.stack_neighbors,
            symmetrize_each_layer=self.symmetrize_each_layer,
            zero_diagonal_each_layer=self.zero_diagonal_each_layer,
        )

        self.readout_module = PBCCNNReadout(
            channels=self.cnn_channels,
            num_layers=self.cnn_layers,
            kernel_size=self.cnn_kernel_size,
            mlp_hidden_dims=self.mlp_hidden_dims,
            output_scale=self.output_scale,
            activation=self.activation,
            use_layernorm=self.use_layernorm,
            use_bias=self.use_bias,
            param_dtype=self.param_dtype,
            readout=self.readout,
            return_log=self.return_log,
            clip_log=self.clip_log,
        )

    def _initial_gram_channels(self, n: Array) -> Array:
        """
        Builds initial h_ijc from G_ij using a learnable scalar-to-channel
        projection:

            h_ijc = G_ij * gain_c + bias_c

        Then zeros the diagonal.
        """
        dtype = self.param_dtype

        g = gram_matrix(n).astype(dtype)
        g = zero_diagonal_matrix(g)

        h = g[..., :, :, None] * self.input_gain + self.input_bias
        h = zero_diagonal_tensor(h)

        return h

    def _run_stack(self, h0: Array) -> Array:
        if self.stack_layers == 0:
            return h0

        return self._fused_stack(
            h0,
            self.depthwise_weights,
            self.mix_weights,
            self.mix_bias,
            jnp.asarray(self.stack_residual_scale, dtype=self.param_dtype),
            jnp.asarray(self.stack_mix_scale, dtype=self.param_dtype),
        )

    def _make_site_features(self, n: Array, h: Array) -> Array:
        features = []

        if self.use_raw_features:
            raw = extract_pbc_raw_separation_features(
                n.astype(self.param_dtype),
                self.raw_r_max,
                reflection_mode=self.reflection_mode,
            )
            features.append(raw)

        if self.use_stack_features:
            stack_feats = extract_pbc_matrix_band_features(
                h,
                self.stack_r_max,
                reflection_mode=self.reflection_mode,
            )
            features.append(stack_feats)

        if self.use_global_stack_features:
            global_feats = broadcast_global_separation_features(
                h,
                self.stack_r_max,
                reflection_mode=self.reflection_mode,
            )
            features.append(global_feats)

        return jnp.concatenate(features, axis=-1)

    def __call__(self, n: Array) -> Array:
        if n.shape[-1] != 3:
            raise ValueError(f"Expected input shape (..., L, 3), got {n.shape}")

        if self.normalize_input:
            n = normalize_rotors(n)

        L = n.shape[-2]

        if self.raw_r_max >= L:
            raise ValueError(f"Need raw_r_max <= L-1. Got raw_r_max={self.raw_r_max}, L={L}.")

        if self.stack_r_max >= L:
            raise ValueError(f"Need stack_r_max <= L-1. Got stack_r_max={self.stack_r_max}, L={L}.")

        h0 = self._initial_gram_channels(n)
        h = self._run_stack(h0)

        site_features = self._make_site_features(n, h)

        return self.readout_module(site_features)


# ============================================================
# Optional excited-state wrapper
# ============================================================

class GodSlayer4Excited(GodSlayer4):
    """
    Excited-state variant of GodSlayer4.

    psi_excited(n) = O(n) * psi_even(n)

    Default:
        O(n) = sum_i n_i^z
    """

    excited_component: int = 2
    excited_staggered: bool = False

    @nn.compact
    def __call__(self, n: Array) -> Array:
        if self.return_log:
            raise ValueError(
                "GodSlayer4Excited should use return_log=False because "
                "O(n) * psi(n) is signed."
            )

        psi_even = super().__call__(n)

        comp = n[..., self.excited_component]

        if self.excited_staggered:
            L = n.shape[-2]
            signs = (-1.0) ** jnp.arange(L, dtype=n.dtype)
            Oz = jnp.sum(signs * comp, axis=-1)
        else:
            Oz = jnp.sum(comp, axis=-1)

        return Oz * psi_even