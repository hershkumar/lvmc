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
        return jnp.squeeze(jnp.squeeze(y, axis=-1))


class MLP_TI(nn.Module):
    """
    Translation invariant MLP via gauge fixing of Fourier transform features
    """
    hidden_sizes: Sequence[int] = (128, 128)
    activation: Callable = nn.celu
    eps: float = 1e-8
    
    @nn.compact
    def __call__(self, n: jnp.ndarray) -> jnp.ndarray:
        n = jnp.asarray(n)
        feats = translation_invariant_features_real(n, eps=self.eps)
        mlp = MLP(hidden_sizes=self.hidden_sizes, activation=self.activation)
        out = mlp(feats)
        return out



class MLP_SO3(nn.Module):
    """
    Translation + full O(3) invariant scalar via:
      SO(3) canonicalize -> translation-invariant Fourier features -> shared MLP
      then symmetrize over one fixed reflection.
    """
    hidden_sizes: Sequence[int] = (128, 128)
    activation: Callable = nn.celu
    eps: float = 1e-8

    # Avoid mutable default by constructing inside __call__ (or use default_factory).
    @nn.compact
    def __call__(self, n: jnp.ndarray) -> jnp.ndarray:
        n = jnp.asarray(n)

        n_std = canonicalize_so3_fast(n, eps=self.eps)
        feats = n_std
        mlp = MLP(hidden_sizes=self.hidden_sizes, activation=self.activation)
        out = mlp(feats)
        return out


class MLP_O3_TI(nn.Module):
    """
    Translation + full O(3) invariant scalar via:
      SO(3) canonicalize -> translation-invariant Fourier features -> shared MLP
      then symmetrize over one fixed reflection.
    """
    hidden_sizes: Sequence[int] = (128, 128)
    activation: Callable = nn.celu
    eps: float = 1e-8

    # Avoid mutable default by constructing inside __call__ (or use default_factory).
    @nn.compact
    def __call__(self, n: jnp.ndarray) -> jnp.ndarray:
        n = jnp.asarray(n)

        Px = jnp.diag(jnp.array([-1.0,  1.0,  1.0], dtype=n.dtype))
        Py = jnp.diag(jnp.array([ 1.0, -1.0,  1.0], dtype=n.dtype))
        Pz = jnp.diag(jnp.array([ 1.0,  1.0, -1.0], dtype=n.dtype))

        n_std = canonicalize_so3(n, eps=self.eps)
        feats = translation_invariant_features_real(n_std, eps=self.eps)

        mlp = MLP(hidden_sizes=self.hidden_sizes, activation=self.activation)
        out0 = mlp(feats)

        def eval_ref(P):
            n_ref = jnp.einsum("ij,...Lj->...Li", P, n_std)
            feats_ref = translation_invariant_features_real(n_ref, eps=self.eps)
            return mlp(feats_ref)

        out = (out0 + eval_ref(Px) + eval_ref(Py) + eval_ref(Pz)) * 0.25
        return out


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


def canonicalize_so3(n, eps=1e-8, return_R=False):
    """
    Translation-commuting SO(3) canonicalization using two covariant, translation-invariant vectors:
      n_plus = sum_i n_i
      q      = sum_i n_i x n_{i+1}   (periodic)

    Fix frame by:
      1) rotate so n_plus -> +z
      2) use residual z-rotation to make (rotated q)_x = 0

    n: (..., L, 3)
    Returns:
      n_std: (..., L, 3)
      R:     (..., 3, 3)   (if return_R)
      angles: (phi, theta, alpha) (if return_R)
    """
    n = jnp.asarray(n)
    L = n.shape[-2]

    # n_plus (translation-invariant, rotation-covariant)
    n_plus = jnp.sum(n, axis=-2)  # (..., 3)

    # q = sum_i n_i x n_{i+1} with periodic boundary (also translation-invariant, rotation-covariant)
    n_next = jnp.roll(n, shift=-1, axis=-2)   # (..., L, 3)
    q = jnp.sum(jnp.cross(n, n_next), axis=-2)  # (..., 3)

    # Step 1: rotate about z to put n_plus into xz-plane
    phi = jnp.arctan2(n_plus[..., 1], n_plus[..., 0])
    R1 = rotz(-phi)

    nplus1 = jnp.einsum("...ij,...j->...i", R1, n_plus)
    r = jnp.sqrt(nplus1[..., 0]**2 + nplus1[..., 1]**2 + eps)

    # Step 2: rotate about y to send n_plus -> +z
    theta = jnp.arctan2(r, nplus1[..., 2])
    R2 = roty(-theta)

    # Rotate q into the same intermediate frame
    q2 = jnp.einsum("...ij,...j->...i", R2, jnp.einsum("...ij,...j->...i", R1, q))
    a, b = q2[..., 0], q2[..., 1]

    # Step 3: residual z-rotation to make x component of q vanish
    # After Rz(alpha): x' = a cosα - b sinα, choose alpha = atan2(a, b)
    alpha = jnp.arctan2(a, b)
    R3 = rotz(alpha)

    R = jnp.einsum("...ij,...jk->...ik", R3, jnp.einsum("...ij,...jk->...ik", R2, R1))
    n_std = jnp.einsum("...ij,...Lj->...Li", R, n)

    # Degeneracies: if n_plus ~ 0 or q transverse component ~ 0, frame is ambiguous
    deg1 = jnp.linalg.norm(n_plus, axis=-1) < eps
    deg2 = (a*a + b*b) < (eps*eps)
    deg = deg1 | deg2

    I = jnp.eye(3, dtype=n.dtype)
    R = jnp.where(deg[..., None, None], I, R)
    n_std = jnp.where(deg[..., None, None], n, n_std)

    return n_std


def canonicalize_so3_fast(n, eps=1e-8):
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
    r_xy = jnp.sqrt(x*x + y*y + eps)
    nrm  = jnp.sqrt(x*x + y*y + z*z + eps)

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

    rab = jnp.sqrt(a*a + b*b + eps)

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
    n_std = jnp.einsum("...ij,...Lj->...Li", R, n)

    # Optional: if you still want the exact same degeneracy behavior, keep it.
    # Note: any hard where introduces non-smoothness at the threshold.
    #deg1 = (jnp.linalg.norm(n_plus, axis=-1) < eps)
    #deg2 = (a*a + b*b) < (eps*eps)
    #deg = deg1 | deg2
    #n_std = jnp.where(deg[..., None, None], n, n_std)

    return n_std


def canonicalize_so3_xaxis(n, eps=1e-6, return_R=False):
    n = jnp.asarray(n)
    n_next = jnp.roll(n, shift=-1, axis=-2)
    n_plus = jnp.sum(n, axis=-2)           # (..., 3)
    q      = jnp.sum(jnp.cross(n, n_next), axis=-2)  # (..., 3)

    # Step 1: rotate n_plus -> +x  (use atan2 only on normalized xy)
    nxy_norm = _safe_norm(n_plus[..., :2], eps=eps)
    phi = jnp.arctan2(n_plus[..., 1], n_plus[..., 0] + eps)
    R1  = rotz(-phi)
    nplus1 = jnp.einsum("...ij,...j->...i", R1, n_plus)

    r     = _safe_norm(nplus1[..., :2], eps=eps)  # guaranteed > 0
    theta = jnp.arctan2(nplus1[..., 2], r)
    R2    = roty(theta)

    # Step 2: residual rotation to kill q_z
    R12 = jnp.einsum("...ij,...jk->...ik", R2, R1)
    q2  = jnp.einsum("...ij,...j->...i", R12, q)

    # Guard: blend to identity when q2_yz is tiny
    q2_yz_norm = _safe_norm(q2[..., 1:], eps=eps)  # smooth, > 0
    alpha = jnp.arctan2(-q2[..., 2], q2[..., 1] + eps)
    R3    = rotx(alpha)

    R     = jnp.einsum("...ij,...jk->...ik", R3, R12)
    n_std = jnp.einsum("...ij,...Lj->...Li", R, n)
    return n_std

def _safe_norm(v, eps=1e-8):
    return jnp.sqrt(jnp.sum(v * v, axis=-1, keepdims=True) + eps)

def _normalize(v, eps=1e-8):
    return v / _safe_norm(v, eps)

@partial(jit, static_argnums=(2))
def canonicalize_so3_smooth(n, eps=1e-8, k_shifts=(1, 2, 3)):
    """
    Smooth, translation-commuting SO(3) canonicalization (no atan2, no hard branches).

    Builds an orthonormal frame (e1,e2,e3) from translation-invariant, rotation-covariant vectors:
      v0 = n_plus = sum_i n_i
      qk = sum_i n_i x n_{i+k}   (periodic, for several k)
      q_eff = weighted sum of qk (weights depend smoothly on ||qk||)

    Frame:
      e3 = normalize(v0)
      q_perp = q_eff - (q_eff·e3)e3
      e1 = normalize(q_perp)
      e2 = normalize(e3 x e1)

    Rotation matrix R has rows [e1^T; e2^T; e3^T] so that for any site vector u:
      u_std = R u
    and the standardized field is n_std[..., i, :] = R @ n[..., i, :].

    Properties:
      - Translation commuting: uses only sums and periodic rolls.
      - SO(3) gauge-fixing is smooth; degeneracies are handled by eps-regularized normalization
        (still ill-conditioned if both n_plus and all qk_perp are ~0, but no discontinuities).

    Args:
      n: (..., L, 3)
      eps: small positive for smooth normalization
      k_shifts: tuple of neighbor offsets used to build q_eff
      return_R: if True, returns (n_std, R)

    Returns:
      n_std: (..., L, 3)
      R: (..., 3, 3) if return_R
    """
    n = jnp.asarray(n)
    L = n.shape[-2]

    # v0 = sum_i n_i (translation-invariant, covariant)
    v0 = jnp.sum(n, axis=-2)  # (..., 3)
    e3 = _normalize(v0, eps)  # (..., 3)

    # Build multiple chirality-like covariant vectors qk = sum_i n_i x n_{i+k}
    qs = []
    q_norm2 = []
    for k in k_shifts:
        n_k = jnp.roll(n, shift=-int(k), axis=-2)
        qk = jnp.sum(jnp.cross(n, n_k), axis=-2)  # (..., 3)
        qs.append(qk)
        q_norm2.append(jnp.sum(qk * qk, axis=-1, keepdims=True))  # (..., 1)

    qs = jnp.stack(qs, axis=-2)         # (..., K, 3)
    q_norm2 = jnp.stack(q_norm2, axis=-2)  # (..., K, 1)

    # Smooth weights: emphasize the largest-norm qk but without argmax discontinuity
    # w_k = norm^2 / (sum norm^2 + eps)
    w = q_norm2 / (jnp.sum(q_norm2, axis=-2, keepdims=True) + eps)  # (..., K, 1)
    q_eff = jnp.sum(w * qs, axis=-2)  # (..., 3)

    # Project q_eff perpendicular to e3 and normalize to get e1
    dot_qe3 = jnp.sum(q_eff * e3, axis=-1, keepdims=True)  # (..., 1)
    q_perp = q_eff - dot_qe3 * e3
    e1 = _normalize(q_perp, eps)

    # e2 completes right-handed frame
    e2 = _normalize(jnp.cross(e3, e1), eps)

    # Assemble rotation matrix with rows [e1; e2; e3]
    R = jnp.stack([e1, e2, e3], axis=-2)  # (..., 3, 3)

    # Apply to all sites
    n_std = jnp.einsum("...ij,...Lj->...Li", R, n)
    return n_std


def _translation_invariant_tilde_y(y, eps=1e-8, axis=-1):
    """
    y: complex FFT coefficients along `axis`, length L.
    Implements  y~_l = y_l * (|y_1|/y_1)^l  in a numerically-stable way.
    """
    L = y.shape[axis]
    y1 = jnp.take(y, 1, axis=axis)

    # Stable version of |y1|/y1 = exp(-i arg(y1)):
    # u = conj(y1)/(|y1|+eps)  ~ (|y1|/y1)
    u = jnp.conj(y1) / (jnp.abs(y1) + eps)

    l = jnp.arange(L, dtype=jnp.result_type(jnp.real(y1), 1.0))  # (L,)

    # Broadcast u**l along the FFT axis
    # Create a shape like (1,1,...,L,...,1) for l and align to `axis`.
    shape = [1] * y.ndim
    shape[axis] = L
    l = l.reshape(shape)

    # Expand u to broadcast along axis
    u = jnp.expand_dims(u, axis=axis)

    return y * (u ** l)

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
