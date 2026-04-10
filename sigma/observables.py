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
# Optimized observables implementation (active below)
# Keep original functions above intact. To revert to originals, comment out the
# alias assignments in the "ACTIVE IMPLEMENTATION" section below.
def _scalar_tiny(x):
    x = jnp.asarray(x)
    real_dtype = jnp.real(x).dtype
    return jnp.asarray(jnp.finfo(real_dtype).tiny, dtype=x.dtype)


def _normalize_cartesian_sites(xyz, eps=1e-12):
    xyz = jnp.asarray(xyz)
    norm_sq = jnp.sum(xyz * xyz, axis=-1, keepdims=True)
    inv_norm = lax.rsqrt(jnp.maximum(norm_sq, jnp.asarray(eps, dtype=norm_sq.dtype)))
    return xyz * inv_norm


def _orthonormal_tangent_basis(xyz):
    xyz = jnp.asarray(xyz)
    z_axis = jnp.array([0.0, 0.0, 1.0], dtype=xyz.dtype)
    x_axis = jnp.array([1.0, 0.0, 0.0], dtype=xyz.dtype)
    helper = jnp.where(jnp.abs(xyz[:, 2:3]) > 0.9, x_axis, z_axis)

    t1 = _normalize_cartesian_sites(jnp.cross(helper, xyz))
    t2 = _normalize_cartesian_sites(jnp.cross(xyz, t1))
    return t1, t2


def _tree_weighted_sum(tree, weights):
    return jax.tree_util.tree_map(
        lambda leaf: jnp.tensordot(weights, leaf, axes=((0,), (0,))),
        tree,
    )


def _reshape_into_padded_batches(configs, batch_size):
    ncfg = configs.shape[0]
    n_batches = (ncfg + batch_size - 1) // batch_size
    pad = n_batches * batch_size - ncfg

    pad_width = ((0, pad),) + ((0, 0),) * (configs.ndim - 1)
    padded_configs = jnp.pad(configs, pad_width)
    batch_shape = (n_batches, batch_size) + configs.shape[1:]
    batched_configs = padded_configs.reshape(batch_shape)

    mask = jnp.pad(jnp.ones((ncfg,), dtype=configs.dtype), (0, pad))
    batched_mask = mask.reshape(n_batches, batch_size)
    return batched_configs, batched_mask


def _lap_s2_all_sites_tangent_linearized(model, params, xyz):
    """
    Optimized: O(L) memory (no (L,L,3) direction tensors, no (L,L,3) HVP storage).

    Computes:
      sum_i (L_i^2 psi)/psi = - (sum_i Δ_{S^2,i} psi) / psi
    using tangent-space HVPs at each site.

    For each site i and tangent basis e1,e2:
      Δ_{S^2,i} psi = e1^T H_i e1 + e2^T H_i e2 - 2 n_i · (∇_i psi)
    """
    xyz = jnp.asarray(xyz)
    L = xyz.shape[0]

    def psi_of_xyz(xyz_full):
        return model.apply(params, xyz_full)

    # One linearization at xyz: gives psi_val and grad_xyz, and a fast JVP closure.
    (psi_val, grad_xyz), jvp_fn = jax.linearize(jax.value_and_grad(psi_of_xyz), xyz)

    t1, t2 = _orthonormal_tangent_basis(xyz)  # (L,3), (L,3)

    # Radial correction summed over sites: 2 * sum_i n_i · grad_i psi
    radial_sum = 2.0 * jnp.sum(jnp.sum(xyz * grad_xyz, axis=-1))

    # Accumulate projected trace sum_i [e1^T H_i e1 + e2^T H_i e2] without storing hvps.
    def body(i, acc):
        # Build a direction v that is zero everywhere except at site i.
        v = jnp.zeros_like(xyz).at[i].set(t1[i])
        hvp1 = jvp_fn(v)[1]            # (L,3)
        acc = acc + jnp.dot(t1[i], hvp1[i])

        v = jnp.zeros_like(xyz).at[i].set(t2[i])
        hvp2 = jvp_fn(v)[1]            # (L,3)
        acc = acc + jnp.dot(t2[i], hvp2[i])

        return acc

    proj_trace_sum = lax.fori_loop(
        0, L, body, jnp.array(0.0, dtype=jnp.result_type(xyz, psi_val))
    )

    # Sum_i Δ_i psi
    lap_sum = proj_trace_sum - radial_sum

    inv_psi = 1.0 / (psi_val + _scalar_tiny(psi_val))
    # Return sum_i (L_i^2 psi)/psi = -(sum_i Δ_i psi)/psi
    return -lap_sum * inv_psi


@partial(jax.jit, static_argnums=(0,))
def spherical_laplacian_cartesian_fast(model, params, config_xyz):
    config_xyz = _normalize_cartesian_sites(config_xyz)
    return _lap_s2_all_sites_tangent_linearized(model, params, config_xyz)


@partial(jax.jit, static_argnums=(0,))
def isospin_charge(model, params, config_xyz):
    """
    Computes total Q_z psi/psi for a configuration of unit vectors on S^2.

    Assumptions:
      - config_xyz: (L,3) unit vectors n_i = (x_i,y_i,z_i)
      - model.apply(params, config_xyz) returns psi(config_xyz) (scalar, possibly complex)
      - Q_z = sum_i L_{z,i}
      - L_{z} = -i (x ∂_y - y ∂_x) acting on functions of n

    Implementation:
      - Extend psi off the sphere by normalizing the varied site vector.
      - For each site i, compute (L_{z,i} psi)/psi and sum.
    """
    xyz = jnp.asarray(config_xyz)
    L = xyz.shape[0]

    def psi_of_xyz(xyz_full):
        return model.apply(params, xyz_full)

    psi_val = psi_of_xyz(xyz) # avoid division by 0

    def per_site(i):
        n0 = xyz[i]

        # off-sphere extension via normalization
        def psi_ext(m):
            mhat = m / (jnp.linalg.norm(m))
            cfg2 = xyz.at[i].set(mhat)
            return psi_of_xyz(cfg2)

        grad = jax.grad(psi_ext)(n0)  # (∂_x psi, ∂_y psi, ∂_z psi)

        x, y = n0[0], n0[1]
        dpsi_dx, dpsi_dy = grad[0], grad[1]

        # Lz psi = -i (x ∂_y - y ∂_x) psi
        Lz_psi = -1j * (x * dpsi_dy - y * dpsi_dx)

        return Lz_psi / psi_val

    return jnp.imag(jnp.sum(jax.vmap(per_site)(jnp.arange(L))))


@partial(jit, static_argnums=(0,))
def config_energy_fast(model, eta, g, mu, params, config_xyz):
    kinetic_term = 0.5 * eta * (g**2) * spherical_laplacian_cartesian_fast(
        model, params, config_xyz
    )

    n = _normalize_cartesian_sites(config_xyz)
    n_next = jnp.roll(n, shift=-1, axis=0)
    nn = jnp.sum(n * n_next, axis=-1)
    potential_term = -(eta / (g**2)) * jnp.sum(nn)
    chemical_term = -mu*isospin_charge(model, params, config_xyz)


    return kinetic_term + potential_term + chemical_term


@partial(jit, static_argnums=(0,))
def dlogpsi_dparams_fast(model, params, config):
    """
    model.apply returns psi(config) directly.
    Returns d/dparams [log psi] = (1 / psi) dpsi/dparams.
    """

    def psi_of_params(p):
        return model.apply(p, config)

    psi_val, gradpsi = jax.value_and_grad(psi_of_params)(params)
    invpsi = 1.0 / (psi_val + _scalar_tiny(psi_val))
    return jax.tree_util.tree_map(lambda x: x * invpsi, gradpsi)


@partial(jit, static_argnums=(0,))
def local_terms_fast(model, eta, g, mu, params, config):
    local_energy = config_energy_fast(model, eta, g, mu, params, config)
    log_grad = dlogpsi_dparams_fast(model, params, config)
    return local_energy, log_grad


spherical_laplacian_cartesian_opt = spherical_laplacian_cartesian_fast
config_energy_opt = config_energy_fast
dlogpsi_dparams_opt = dlogpsi_dparams_fast
local_terms_opt = local_terms_fast


def _batched_vmap(f, configs, batch_size):
    ncfg = configs.shape[0]
    if batch_size is None or batch_size >= ncfg:
        return jax.vmap(f)(configs)

    n_full = (ncfg // batch_size) * batch_size
    remainder = ncfg - n_full

    full_configs = configs[:n_full].reshape(
        ncfg // batch_size, batch_size, *configs.shape[1:]
    )

    # vmap inside lax.map is fine, but outputs are stacked — reshape correctly
    batched_f = jax.vmap(f)

    # Use scan instead of lax.map to avoid upfront full allocation
    def scan_fn(carry, batch):
        return carry, batched_f(batch)

    _, full_results = jax.lax.scan(scan_fn, None, full_configs)

    full_results = jax.tree_util.tree_map(
        lambda x: x.reshape(n_full, *x.shape[2:]), full_results
    )

    if remainder == 0:
        return full_results

    rem_results = jax.vmap(f)(configs[n_full:])
    return jax.tree_util.tree_map(
        lambda a, b: jnp.concatenate([a, b], axis=0), full_results, rem_results
    )


@partial(jit, static_argnums=(0, 6))
def dE_dparams_opt(model, eta, g, mu, params, configs, batch_size=None):
    """
    Compute energy gradient over a batch of configurations.

    Assumes configs are drawn from |psi|^2 and the standard VMC gradient:
        ∂E = 2( <E_loc O> - <E_loc><O> ),  O = ∂ log psi / ∂params
    """
    ncfg = configs.shape[0]
    local_fn = jax.vmap(lambda config: local_terms_opt(model, eta, g, mu, params, config))

    if batch_size is None or batch_size >= ncfg:
        energies, logs = local_fn(configs)
        count = jnp.asarray(ncfg, dtype=energies.dtype)
        inv_count = 1.0 / count
        energy = jnp.sum(energies) * inv_count
        var = jnp.maximum(
            jnp.sum(energies * energies) * inv_count - energy * energy,
            jnp.array(0.0, dtype=energies.dtype),
        )
        uncert = jnp.sqrt(var * inv_count)
        mean_logs = pytree_mult(inv_count, _tree_weighted_sum(logs, jnp.ones_like(energies)))
        mean_weighted_logs = pytree_mult(inv_count, _tree_weighted_sum(logs, energies))
        grad = pytree_add(
            pytree_mult(2.0, mean_weighted_logs),
            pytree_mult(-2.0 * energy, mean_logs),
        )
        return grad, energy, uncert

    effective_batch_size = min(batch_size, ncfg)
    batched_configs, batched_mask = _reshape_into_padded_batches(
        configs, effective_batch_size
    )

    zero_scalar = jnp.array(0.0, dtype=configs.dtype)
    sum_logs0 = pytree_zeros_like(params)
    sum_e_logs0 = pytree_zeros_like(params)

    def scan_fn(acc, batch):
        sum_e, sum_e2, sum_logs, sum_e_logs = acc
        batch_configs, mask = batch
        energies, logs = local_fn(batch_configs)
        weighted_energies = energies * mask

        sum_e = sum_e + jnp.sum(weighted_energies)
        sum_e2 = sum_e2 + jnp.sum(weighted_energies * energies)
        sum_logs = pytree_add(sum_logs, _tree_weighted_sum(logs, mask))
        sum_e_logs = pytree_add(
            sum_e_logs, _tree_weighted_sum(logs, weighted_energies)
        )
        return (sum_e, sum_e2, sum_logs, sum_e_logs), None

    (sum_e, sum_e2, sum_logs, sum_e_logs), _ = jax.lax.scan(
        scan_fn,
        (zero_scalar, zero_scalar, sum_logs0, sum_e_logs0),
        (batched_configs, batched_mask),
    )

    count = jnp.asarray(ncfg, dtype=sum_e.dtype)
    inv_count = 1.0 / count
    energy = sum_e * inv_count
    var = jnp.maximum(
        sum_e2 * inv_count - energy * energy,
        jnp.array(0.0, dtype=sum_e.dtype),
    )
    uncert = jnp.sqrt(var * inv_count)

    mean_logs = pytree_mult(inv_count, sum_logs)
    mean_weighted_logs = pytree_mult(inv_count, sum_e_logs)
    grad = pytree_add(
        pytree_mult(2.0, mean_weighted_logs),
        pytree_mult(-2.0 * energy, mean_logs),
    )
    return grad, energy, uncert



@partial(jit, static_argnums=(0, 6))
def batched_energy(model, eta, g, mu, params, configs, batch_size=None):
    """
    Compute energy over a batch of configurations.
    """
    ncfg = configs.shape[0]
    local_fn = jax.vmap(lambda config: config_energy_opt(model, eta, g, mu, params, config))

    if batch_size is None or batch_size >= ncfg:
        energies = local_fn(configs)
        count = jnp.asarray(ncfg, dtype=energies.dtype)
        inv_count = 1.0 / count
        energy = jnp.sum(energies) * inv_count
        var = jnp.maximum(
            jnp.sum(energies * energies) * inv_count - energy * energy,
            jnp.array(0.0, dtype=energies.dtype),
        )
        uncert = jnp.sqrt(var * inv_count)
        return energy, uncert

    effective_batch_size = min(batch_size, ncfg)
    batched_configs, batched_mask = _reshape_into_padded_batches(
        configs, effective_batch_size
    )

    def scan_fn(acc, batch):
        sum_e, sum_e2 = acc
        batch_configs, mask = batch
        energies = local_fn(batch_configs)

        masked_energies = jnp.where(mask, energies, 0.0)
        masked_e2 = jnp.where(mask, energies * energies, 0.0)

        sum_e = sum_e + jnp.sum(masked_energies)
        sum_e2 = sum_e2 + jnp.sum(masked_e2)
        return (sum_e, sum_e2), None

    zero_scalar = jnp.array(0.0, dtype=configs.dtype)
    (sum_e, sum_e2), _ = jax.lax.scan(
        scan_fn,
        (zero_scalar, zero_scalar),
        (batched_configs, batched_mask),
    )

    count = jnp.asarray(ncfg, dtype=sum_e.dtype)
    inv_count = 1.0 / count
    energy = sum_e * inv_count
    var = jnp.maximum(
        sum_e2 * inv_count - energy * energy,
        jnp.array(0.0, dtype=sum_e.dtype),
    )
    uncert = jnp.sqrt(var * inv_count)

    return energy, uncert



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
def _corr_all_r_fft(configs):
    """
    configs: (N, L, d)
    returns: (N, L)
    """
    fk = jnp.fft.rfft(configs, axis=1)                  # (N, Lf, d)
    power = jnp.sum(jnp.abs(fk) ** 2, axis=-1)         # (N, Lf)
    corr = jnp.fft.irfft(power, n=configs.shape[1], axis=1) / configs.shape[1]
    return corr


@partial(jit, static_argnums=(1,))
def Cr_with_cov_optimized(configs, batch_size=1024):
    """
    Returns
    -------
    C : (L,)
        Mean correlator
    cov : (L, L)
        Covariance matrix of the mean correlator
    uncerts : (L,)
        sqrt(diag(cov))
    """
    N, L, d = configs.shape
    if N < 2:
        raise ValueError("Need at least two configurations to estimate covariance.")

    # Use a stable accumulation dtype
    acc_dtype = jnp.float64 if jnp.issubdtype(configs.dtype, jnp.floating) else jnp.float64

    nbatch = (N + batch_size - 1) // batch_size
    padded_N = nbatch * batch_size
    pad = padded_N - N

    configs_pad = jnp.pad(configs, ((0, pad), (0, 0), (0, 0)))
    mask = (jnp.arange(padded_N) < N).reshape(nbatch, batch_size)

    batched_configs = configs_pad.reshape(nbatch, batch_size, L, d)

    def scan_fn(carry, xs):
        sum_corr, sum_outer = carry
        batch_configs, batch_mask = xs

        corr_batch = _corr_all_r_fft(batch_configs).astype(acc_dtype)   # (B, L)

        w = batch_mask.astype(acc_dtype)
        corr_batch = corr_batch * w[:, None]

        sum_corr = sum_corr + jnp.sum(corr_batch, axis=0)               # (L,)
        sum_outer = sum_outer + corr_batch.T @ corr_batch               # (L, L)

        return (sum_corr, sum_outer), None

    init = (
        jnp.zeros((L,), dtype=acc_dtype),
        jnp.zeros((L, L), dtype=acc_dtype),
    )

    (sum_corr, sum_outer), _ = lax.scan(scan_fn, init, (batched_configs, mask))

    Nf = jnp.asarray(N, dtype=acc_dtype)
    C = sum_corr / Nf

    centered_sum_outer = sum_outer - Nf * jnp.outer(C, C)
    denom = Nf * (Nf - jnp.asarray(1, dtype=acc_dtype))
    cov = centered_sum_outer / denom

    cov = 0.5 * (cov + cov.T)
    uncerts = jnp.sqrt(jnp.clip(jnp.diag(cov), a_min=0.0))

    return C, cov, uncerts

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


@partial(jit, static_argnums=(0, 6))
def sr_stats(model, eta, g_coup, mu, params, configs, batch_size=None):
    """
    Compute energy, gradient, and centred log-derivatives for SR.

    Returns:
        grad:           pytree  SR gradient 2 * Cov(E, O).
        logs_centered:  pytree  centred log-derivative samples (N, ...).
        E:              scalar  mean local energy (unclipped, for monitoring).
        uncert:         scalar  standard error of the mean energy (unclipped).
    """
    local_fn = lambda config: local_terms_opt(model, eta, g_coup, mu, params, config)
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
