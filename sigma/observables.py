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
        eta * (g**2) * spherical_laplacian_cartesian(model, params, config_xyz)
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

def _spherical_lap_all_sites(model, params, ang, eps=1e-8):
    """
    Compute spherical Laplacian for all sites in one pass.
    Instead of vmap over sites with closure+grad, use jacrev/jacfwd
    over the full angle array at once.
    """
    def A_of_angles(ang_full):
        xyz = angles_to_unitvec(ang_full)
        return model.apply(params, xyz)

    # Gradient: shape (L, 2)
    g_all = jax.jacrev(A_of_angles)(ang)
    # Hessian diagonal entries only — use forward-over-reverse for efficiency
    # We only need H[i,j,i,j] (diagonal blocks), but jax.hessian gives full (L,2,L,2).
    # Instead compute per-site Hessian diagonals using vmap over jvp.
    L = ang.shape[0]

    # Compute diagonal of Hessian: d²A/d(ang[i,k])² for each (i,k)
    # Using forward-over-reverse: one jvp per basis vector
    def hvp(v):
        # v: tangent vector, same shape as ang (L, 2)
        return jax.jvp(jax.grad(A_of_angles), (ang,), (v,))[1]

    # We need diag of Hessian: entries (i,0,i,0) and (i,1,i,1)
    # Build basis vectors for each of the 2L scalar params
    basis = jnp.eye(L * 2).reshape(L * 2, L, 2)
    # vmap hvp over all basis vectors
    H_diag_flat = jax.vmap(hvp)(basis)  # (L*2, L, 2)
    # H_diag_flat[k, :, :] = row k of Hessian; we want diagonal: entry k = H[k,k]
    H_diag = H_diag_flat.reshape(L * 2, L * 2)  # full Hessian
    # Extract diagonal, reshape to (L, 2)
    H_diag = jnp.diag(H_diag).reshape(L, 2)

    thetas = ang[:, 0]
    sin_t = jnp.sin(thetas)
    sin_t = jnp.where(jnp.abs(sin_t) < eps, jnp.sign(sin_t) * eps + (sin_t == 0.0) * eps, sin_t)
    sin2 = sin_t ** 2
    cot_t = jnp.cos(thetas) / sin_t

    A_t = g_all[:, 0]
    A_p = g_all[:, 1]
    A_tt = H_diag[:, 0]
    A_pp = H_diag[:, 1]

    lap_A = A_tt + cot_t * A_t + A_pp / sin2
    gradA_sq = A_t ** 2 + A_p ** 2 / sin2
    return jnp.sum(lap_A - gradA_sq)


@partial(jit, static_argnums=(0,))
def spherical_laplacian_cartesian_opt(model, params, config_xyz):
    config_xyz = jnp.asarray(config_xyz)
    ang = cartesian_to_angles(config_xyz)
    return _spherical_lap_all_sites(model, params, ang)


@partial(jit, static_argnums=(0,))
def config_energy_opt(model, eta, g, params, config_xyz):
    kinetic_term = eta * (g ** 2) * spherical_laplacian_cartesian_opt(model, params, config_xyz)
    n = jnp.asarray(config_xyz)
    n_next = jnp.roll(n, shift=-1, axis=0)
    nn = jnp.sum(n * n_next, axis=-1)
    potential_term = -(eta / (g ** 2)) * jnp.sum(nn)
    return 0.5 * kinetic_term + potential_term


@partial(jit, static_argnums=(0,))
def dlogpsi_dparams_opt(model, params, config):
    gradA = jax.grad(lambda p: model.apply(p, config))(params)
    return jax.tree_util.tree_map(lambda x: -x, gradA)


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

    # Process full batches via lax.map (constant memory per batch)
    full_configs = configs[:n_full].reshape(ncfg // batch_size, batch_size, *configs.shape[1:])
    batched_f = jax.vmap(f)
    full_results = jax.lax.map(batched_f, full_configs)
    # Flatten leading two dims
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

    Args:
        batch_size: if provided, processes configs in chunks of this size to
                    limit peak memory. Useful when ncfg is large. Must be a
                    compile-time static integer (hence static_argnums).
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
    for x in range(C.shape[0]):
        prod = jnp.dot(nxs[x], nxs[0])
        C = C.at[x].set(C[x] - prod)   
    
    return C

@jit
def nx(configs):
    return jnp.mean(configs, axis=0)