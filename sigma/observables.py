from wavefunction import *
from utils import *
from functools import partial
import jax.numpy as jnp
from jax import jit
import jax
from jax import lax


def cartesian_to_angles(n, eps=1e-8):
    """
    n: (L,3) unit vectors (approximately)
    returns angles: (L,2) = (theta, phi)
    """
    n = jnp.asarray(n)
    x, y, z = n[:, 0], n[:, 1], n[:, 2]

    zc = jnp.clip(z, -1.0 + eps, 1.0 - eps)
    theta = jnp.arccos(zc)

    phi = jnp.arctan2(y, x)  # in (-pi, pi]
    # Map to [0, 2pi) if you prefer:
    # phi = jnp.mod(phi, 2.0*jnp.pi)

    return jnp.stack([theta, phi], axis=-1)


def local_L2_per_site_cartesian(model, params, config_xyz, site_i, eps=1e-8):
    """
    Computes (L_i^2 psi)/psi at site_i, where:
      - config_xyz is (L,3) of unit vectors n_i
      - model.apply(params, config_xyz) returns A = model = -log psi
      - psi = exp(-A)

    Implementation:
      - convert config_xyz -> angles (theta, phi)
      - compute Δ_{S^2} A and |∇A|^2 in angular coordinates
      - return (L^2 psi)/psi = ΔA - |∇A|^2
    """
    config_xyz = jnp.asarray(config_xyz)
    ang = cartesian_to_angles(config_xyz, eps=eps)  # (L,2)

    def A_of_site(ang_i):
        ang2 = ang.at[site_i].set(ang_i)
        xyz2 = angles_to_unitvec(ang2)  # re-embed onto S^2
        return model.apply(params, xyz2)

    a0 = ang[site_i]  # (theta, phi)

    g = jax.grad(A_of_site)(a0)  # (A_theta, A_phi)
    H = jax.hessian(A_of_site)(a0)  # [[A_tt, ...],[..., A_pp]]

    theta = a0[0]
    sin_t = jnp.sin(theta)
    sin_t = jnp.where(
        jnp.abs(sin_t) < eps, jnp.sign(sin_t) * eps + (sin_t == 0) * eps, sin_t
    )
    sin2 = jnp.maximum(sin_t * sin_t, eps)
    cot_t = jnp.cos(theta) * jax.lax.rsqrt(sin2)

    A_t, A_p = g[0], g[1]
    A_tt = H[0, 0]
    A_pp = H[1, 1]

    lap_A = A_tt + cot_t * A_t + A_pp / sin2
    gradA_sq = (A_t * A_t) + (A_p * A_p) / sin2  # no complex conjugation

    return lap_A - gradA_sq


@partial(jit, static_argnums=(0,))
def spherical_laplacian_cartesian(model, params, config_xyz):
    """
    Returns sum_i (L_i^2 psi)/psi over sites i, with config in Cartesian.
    """
    L = config_xyz.shape[0]
    vals = jax.vmap(
        lambda i: local_L2_per_site_cartesian(model, params, config_xyz, i)
    )(jnp.arange(L))
    return jnp.sum(vals)


def angles_to_unitvec(config_ang):
    """
    config_ang: (L,2) with columns (theta, phi)
    returns n: (L,3)
    """
    theta = config_ang[:, 0]
    phi = config_ang[:, 1]
    st = jnp.sin(theta)
    return jnp.stack([st * jnp.cos(phi), st * jnp.sin(phi), jnp.cos(theta)], axis=-1)


@partial(jit, static_argnums=(0,))
def config_energy(model, eta, g, params, config_xyz):
    """
    Local energy for config in Cartesian (L,3), PBCs on the neighbor dot-product term.

    Assumes Hamiltonian:
      kinetic:   eta * g^2 * sum_i (L_i^2 psi)/psi
      potential: -(eta/g^2) * sum_i n_i · n_{i+1}   (PBC)
    """
    kinetic_term = (
        0.5 * eta * (g**2) * spherical_laplacian_cartesian(model, params, config_xyz)
    )

    n = jnp.asarray(config_xyz)
    n_next = jnp.roll(n, shift=-1, axis=0)  # PBC
    nn = jnp.sum(n * n_next, axis=-1)
    potential_term = -(eta / (g**2)) * jnp.sum(nn)

    return kinetic_term + potential_term


@partial(jit, static_argnums=(0,))
def dlogpsi_dparams(model, params, config):
    """
    Returns pytree with same structure as params:
        d/dparams [ log psi(config) ]

    model.apply(params, config) must return A = -log psi.
    """

    def A_of_params(p):
        return model.apply(p, config)  # A = -log psi

    gradA = jax.grad(A_of_params)(params)

    # d logpsi / d params = - dA / d params
    return jax.tree_util.tree_map(lambda x: -x, gradA)


def local_terms(model, eta, g, params, config):
    """
    Computes the gradient of the energy with respect to the parameters.
    """

    local_energy = config_energy(model, eta, g, params, config)
    lg = dlogpsi_dparams(model, params, config)

    return [local_energy, lg, pytree_mult(local_energy, lg)]


def dE_dparams(model, eta, g, params, configs):
    energies, logs, mults = jax.vmap(
        lambda config: local_terms(model, eta, g, params, config), in_axes=0
    )(configs)
    energy = jnp.mean(energies)
    uncert = jnp.std(energies) / jnp.sqrt(energies.shape[0])

    # grad = 2 * mean(mults) - 2 * mean(energies) * mean(logs)
    grad = pytree_add(
        pytree_mult(2, pytree_mean(mults)), pytree_mult(-2 * energy, pytree_mean(logs))
    )

    return grad, energy, uncert


# ============================================================================
# Optimized observables implementation (toggleable)
# Keep original functions above intact. To revert to originals, comment out the
# alias assignments in the "ACTIVE IMPLEMENTATION" section below.
def _lap_s2_all_sites_cartesian(model, params, xyz):
    """
    xyz: (L,3) assumed ~unit vectors
    returns sum_i [(L_i^2 psi)/psi] with L_i^2 = -Δ_{S^2,i}
    where psi = model.apply(params, xyz).

    Uses the Cartesian Laplace–Beltrami identity for a function f on S^2
    extended off-sphere:
        Δ_{S^2} f = tr(P H) - 2 n·∇f
    with P = I - n n^T, H the 3x3 Hessian wrt that site's Cartesian coords,
    and ∇f the 3-gradient wrt that site's coords.

    Then (L^2 f) = -Δ_{S^2} f, so (L^2 psi)/psi = -(Δ_{S^2} psi)/psi.
    """
    xyz = jnp.asarray(xyz)
    L = xyz.shape[0]

    def psi_of_xyz(xyz_full):
        return model.apply(params, xyz_full)  # psi directly (scalar, possibly complex)

    psi_val = psi_of_xyz(xyz)

    # Cartesian gradient wrt all xyz: (L,3)
    g_all = jax.jacrev(psi_of_xyz)(xyz)

    # Hessian-vector product helper: returns (L,3)
    def hvp(v):
        return jax.jvp(jax.grad(psi_of_xyz), (xyz,), (v,))[1]

    # Full Hessian in flattened coords (3L x 3L)
    basis = jnp.eye(3 * L, dtype=xyz.dtype).reshape(3 * L, L, 3)
    H_rows = jax.vmap(hvp)(basis)           # (3L, L, 3)
    H_full = H_rows.reshape(3 * L, 3 * L)   # (3L, 3L)

    def per_site(i):
        n = xyz[i]              # (3,)
        g = g_all[i]            # (3,)
        P = jnp.eye(3, dtype=xyz.dtype) - jnp.outer(n, n)  # tangent projector

        idx = jnp.arange(3) + 3 * i
        Hii = H_full[jnp.ix_(idx, idx)]     # (3,3)

        # Δ_{S^2} psi at site i:
        lap_psi = jnp.sum(P * Hii) - 2.0 * jnp.dot(n, g)

        # (L_i^2 psi)/psi = -(Δ_{S^2} psi)/psi
        return -lap_psi / (psi_val + 1e-300)

    return jnp.sum(jax.vmap(per_site)(jnp.arange(L)))


@partial(jax.jit, static_argnums=(0,))
def spherical_laplacian_cartesian_opt(model, params, config_xyz):
    config_xyz = jnp.asarray(config_xyz)
    config_xyz = config_xyz / (jnp.linalg.norm(config_xyz, axis=-1, keepdims=True) + 1e-12)
    return _lap_s2_all_sites_cartesian(model, params, config_xyz)


@partial(jit, static_argnums=(0,))
def config_energy_opt(model, eta, g, params, config_xyz):
    """
    Assumes Hamiltonian kinetic uses sum_i L_i^2, with local kinetic estimator:
        sum_i (L_i^2 psi)/psi
    """
    kinetic_term = eta * (g ** 2) * spherical_laplacian_cartesian_opt(model, params, config_xyz)

    n = jnp.asarray(config_xyz)
    n_next = jnp.roll(n, shift=-1, axis=0)
    nn = jnp.sum(n * n_next, axis=-1)
    potential_term = -(eta / (g ** 2)) * jnp.sum(nn)

    return 0.5 * kinetic_term + potential_term


@partial(jit, static_argnums=(0,))
def dlogpsi_dparams_opt(model, params, config):
    """
    model.apply returns psi(config) directly.
    Returns d/dparams [log psi] = (1/psi) dpsi/dparams.
    """
    psi_val = model.apply(params, config)

    gradpsi = jax.grad(lambda p: model.apply(p, config))(params)
    invpsi = 1.0 / (psi_val + 1e-300)

    return jax.tree_util.tree_map(lambda x: x * invpsi, gradpsi)


@partial(jit, static_argnums=(0,))
def local_terms_opt(model, eta, g, params, config):
    local_energy = config_energy_opt(model, eta, g, params, config)
    lg = dlogpsi_dparams_opt(model, params, config)
    return local_energy, lg


def _batched_vmap(f, configs, batch_size):
    """
    Apply f over configs in chunks of batch_size using lax.map over full
    batches, handling remainders manually. Falls back to plain vmap if
    batch_size is None or >= ncfg.
    """
    ncfg = configs.shape[0]
    if batch_size is None or batch_size >= ncfg:
        return jax.vmap(f)(configs)

    n_full = (ncfg // batch_size) * batch_size
    remainder = ncfg - n_full

    full_configs = configs[:n_full].reshape(ncfg // batch_size, batch_size, *configs.shape[1:])
    batched_f = jax.vmap(f)
    full_results = jax.lax.map(batched_f, full_configs)
    full_results = jax.tree_util.tree_map(
        lambda x: x.reshape(n_full, *x.shape[2:]), full_results
    )

    if remainder == 0:
        return full_results

    rem_results = jax.vmap(f)(configs[n_full:])
    return jax.tree_util.tree_map(
        lambda a, b: jnp.concatenate([a, b], axis=0), full_results, rem_results
    )


@partial(jit, static_argnums=(0, 5))
def dE_dparams_opt(model, eta, g, params, configs, batch_size=None):
    """
    Compute energy gradient over a batch of configurations.

    Assumes configs are drawn from |psi|^2 and the standard VMC gradient:
        ∂E = 2( <E_loc O> - <E_loc><O> ),  O = ∂ log psi / ∂params
    """
    local_fn = lambda config: local_terms_opt(model, eta, g, params, config)
    energies, logs = _batched_vmap(local_fn, configs, batch_size)

    ncfg = energies.shape[0]
    energy = jnp.mean(energies)
    var = jnp.maximum(
        jnp.mean(energies * energies) - energy * energy,
        jnp.array(0.0, dtype=energies.dtype),
    )
    uncert = jnp.sqrt(var / ncfg)

    weighted_logs = jax.tree_util.tree_map(
        lambda x: x * energies.reshape((ncfg,) + (1,) * (x.ndim - 1)),
        logs,
    )
    mean_logs = pytree_mean(logs)
    mean_weighted_logs = pytree_mean(weighted_logs)

    grad = pytree_add(
        pytree_mult(2, mean_weighted_logs),
        pytree_mult(-2 * energy, mean_logs)
    )
    return grad, energy, uncert




# ACTIVE IMPLEMENTATION (comment these lines to use originals above)
spherical_laplacian_cartesian = spherical_laplacian_cartesian_opt
config_energy = config_energy_opt
dlogpsi_dparams = dlogpsi_dparams_opt
dE_dparams = dE_dparams_opt

@jit
def nx_n0(configs):

    nxs = nx(configs)

    def nx_n0_single(config):
        n0 = config[0]
        return jnp.sum(config * n0[None, :], axis=-1)

    C = jnp.mean(jax.vmap(nx_n0_single)(configs), axis=0)
    # compute the uncertainty of C[r] as stddev/sqrt(Ncfg)
    uncerts = jnp.std(jax.vmap(nx_n0_single)(configs), axis=0) / jnp.sqrt(configs.shape[0])

    for x in range(C.shape[0]):
        prod = jnp.dot(nxs[x], nxs[0])
        C = C.at[x].set(C[x] - prod)   
    
    return C, uncerts

# we want to compute the correlation function that takes into account the fact that the system is translationally invariant, so we average over all pairs of sites separated by distance r. This should give us a better estimate of the correlation function and reduce noise.
@jit
def Cr(configs):
    nxs = nx(configs)

    def Cr_single(config, r):
        return jnp.mean(jnp.sum(config * jnp.roll(config, shift=-r, axis=0), axis=-1))

    L = configs.shape[1]
    C = jnp.array([jnp.mean(jax.vmap(lambda config: Cr_single(config, r))(configs)) for r in range(L)])
    C_uncerts = jnp.array([jnp.std(jax.vmap(lambda config: Cr_single(config, r))(configs)) / jnp.sqrt(configs.shape[0]) for r in range(L)])
    return C, C_uncerts



@jit
def nx(configs):
    return jnp.mean(configs, axis=0)








### Testing SR implementation

from jax.flatten_util import ravel_pytree



# ---------------------------------------------------------------------------
# Fisher matrix-vector product
# ---------------------------------------------------------------------------

def fisher_matvec(logs_centered, v):
    """
    Compute the Fisher (quantum geometric tensor) matrix-vector product:

        [S v]_k  =  (1/N) sum_i  O_i^k  (O_i . v)

    where O_i are the *centred* log-derivative samples (pytrees with a
    leading sample axis of size N).

    Args:
        logs_centered: pytree; each leaf has shape (N, *param_shape).
        v:             pytree; each leaf has shape (*param_shape).

    Returns:
        pytree with the same structure/shape as v.
    """
    # s_i = O_i . v — one scalar per sample                        (N,)
    s = jax.vmap(lambda oi: pytree_dot(oi, v))(logs_centered)

    # [Sv]_k = mean_i( s_i * O_i^k )
    def _weighted_mean(leaf):   # leaf: (N, *param_shape)
        return jnp.einsum("i,i...->...", s, leaf) / s.shape[0]

    return jax.tree_util.tree_map(_weighted_mean, logs_centered)


# ---------------------------------------------------------------------------
# Damping
# ---------------------------------------------------------------------------

def add_damping(Sv, v, damping: float = 1e-3):
    """Return  (S + damping * I) v  given S*v and v."""
    return pytree_add(Sv, pytree_mult(damping, v))


# ---------------------------------------------------------------------------
# SR statistics
# ---------------------------------------------------------------------------

def clip_local_energies(energies, n_mad=5.0):
    """
    Clip local energies to  [median - n_mad * MAD, median + n_mad * MAD]
    where MAD = median absolute deviation.

    This is applied only to the gradient/Fisher estimates — the reported
    energy uses the raw (unclipped) mean so you can still see spikes in
    the monitoring output and know when something is wrong.

    n_mad=5 is a standard choice; tighten to 3 if spikes persist,
    loosen to 10 if you're worried about biasing the gradient.
    """
    median  = jnp.median(energies)
    mad     = jnp.median(jnp.abs(energies - median))
    return jnp.clip(energies, median - n_mad * mad, median + n_mad * mad)


@partial(jit, static_argnums=(0, 5))
def sr_stats(model, eta, g_coup, params, configs, batch_size=None):
    """
    Compute energy, gradient, and centred log-derivatives for SR.

    Returns:
        grad:           pytree  SR gradient 2 * Cov(E, O).
        logs_centered:  pytree  centred log-derivative samples (N, ...).
        E:              scalar  mean local energy (unclipped, for monitoring).
        uncert:         scalar  standard error of the mean energy (unclipped).
    """
    local_fn = lambda config: local_terms_opt(model, eta, g_coup, params, config)
    energies, logs = _batched_vmap(local_fn, configs, batch_size)

    N = energies.shape[0]
    E      = jnp.mean(energies)
    uncert = jnp.sqrt(jnp.maximum(jnp.var(energies), 0.0) / N)

    mean_logs     = pytree_mean(logs)
    logs_centered = jax.tree_util.tree_map(lambda x, m: x - m, logs, mean_logs)

    # Clip energies for gradient/Fisher only — E and uncert above are raw.
    energies_clipped = clip_local_energies(energies)
    e_centered       = energies_clipped - jnp.mean(energies_clipped)

    def _grad_leaf(leaf):
        return 2.0 * jnp.einsum("i,i...->...", e_centered, leaf) / N

    grad = jax.tree_util.tree_map(_grad_leaf, logs_centered)

    return grad, logs_centered, E, uncert


# ---------------------------------------------------------------------------
# SR update (linear solve)
# ---------------------------------------------------------------------------

def sr_update(grad, logs_centered, damping: float = 1e-3, maxiter: int = 200):
    """
    Solve the SR linear system  (S + damping * I) delta = grad  via
    conjugate gradients, staying entirely in pytree space (never forms
    the full N_params x N_params matrix).

    Args:
        grad:           pytree  the energy gradient.
        logs_centered:  pytree  centred log-derivative samples.
        damping:        regularisation strength (default 1e-3).
        maxiter:        CG iteration limit.

    Returns:
        delta:  pytree  the SR parameter update.
        info:   int     0 if CG converged, >0 otherwise.
    """
    def matvec(v):
        return add_damping(fisher_matvec(logs_centered, v), v, damping)

    delta, info = jax.scipy.sparse.linalg.cg(matvec, grad, maxiter=maxiter)
    return delta, info
