using LinearAlgebra
using Printf
using Statistics
using TensorKit
using MPSKit
using WignerSymbols
using JLD2


if length(ARGS) < 3
    println("Usage: julia imps_o3_one_g2.jl g2 lmax D")
    println("Example: julia imps_o3_one_g2.jl 1.0 3 64")
    exit(1)
end

const g2 = parse(Float64, ARGS[1])
const lmax = parse(Int, ARGS[2])
const D = parse(Int, ARGS[3])

tol = 1e-6
maxiter = 500
verbosity = 3

filename = "data/g2_$(g2)_lmax_$(lmax)_D_$(D).jld2"


# ============================================================
# Rotor basis and local O(3) operators
# ============================================================

function rotor_basis(lmax::Int)
    states = Tuple{Int, Int}[]
    index = Dict{Tuple{Int, Int}, Int}()

    k = 1
    for l in 0:lmax
        for m in -l:l
            push!(states, (l, m))
            index[(l, m)] = k
            k += 1
        end
    end

    return states, index
end


function n_spherical_matrix(lmax::Int, q::Int)
    @assert q in (-1, 0, 1)

    states, _ = rotor_basis(lmax)
    d = length(states)

    Tq = zeros(ComplexF64, d, d)

    for (a, (l, m)) in enumerate(states)
        for (b, (lp, mp)) in enumerate(states)

            # Gaunt/Wigner selection rule:
            #   -m + q + mp = 0
            # so mp = m - q.
            if mp != m - q
                continue
            end

            val =
                (-1.0)^m *
                sqrt((2l + 1) * (2lp + 1)) *
                wigner3j(l, 1, lp, 0, 0, 0) *
                wigner3j(l, 1, lp, -m, q, mp)

            Tq[a, b] = ComplexF64(val)
        end
    end

    return Tq
end


function rotor_operators(lmax::Int)
    states, _ = rotor_basis(lmax)
    d = length(states)

    L2 = zeros(ComplexF64, d, d)

    for (a, (l, m)) in enumerate(states)
        L2[a, a] = l * (l + 1)
    end

    Tm = n_spherical_matrix(lmax, -1)
    T0 = n_spherical_matrix(lmax, 0)
    Tp = n_spherical_matrix(lmax, 1)

    # Spherical-to-Cartesian convention:
    # n_{+1} = -(nx + i ny)/sqrt(2)
    # n_0    = nz
    # n_{-1} =  (nx - i ny)/sqrt(2)
    nx = (Tm - Tp) / sqrt(2)
    ny = im * (Tm + Tp) / sqrt(2)
    nz = T0

    # Hermitian cleanup.
    L2 = 0.5 * (L2 + L2')
    nx = 0.5 * (nx + nx')
    ny = 0.5 * (ny + ny')
    nz = 0.5 * (nz + nz')

    V = ComplexSpace(d)

    return (
        V = V,
        d = d,
        L2 = TensorMap(L2, V, V),
        nx = TensorMap(nx, V, V),
        ny = TensorMap(ny, V, V),
        nz = TensorMap(nz, V, V),
    )
end


function o3_nlsm_infinite_hamiltonian(; g2::Float64, lmax::Int)
    ops = rotor_operators(lmax)

    lattice = PeriodicVector([ops.V])

    ndotn =
        ops.nx ⊗ ops.nx +
        ops.ny ⊗ ops.ny +
        ops.nz ⊗ ops.nz

    h1 = (g2 / 2) * ops.L2
    h2 = -(1 / g2) * ndotn

    H = InfiniteMPOHamiltonian(
        lattice,
        (1,) => h1,
        (1, 2) => h2,
    )

    return H, ops
end


function random_infinite_mps(ops, D::Int)
    return InfiniteMPS(
        [ops.V],
        [ComplexSpace(D)],
    )
end

function safe_real(x; tol_imag = 1e-8)
    if abs(imag(x)) > tol_imag
        @warn "Non-negligible imaginary part" imag_part = imag(x)
    end
    return real(x)
end


function safe_energy_density(x)
    if x isa Number
        return real(x)
    else
        xs = collect(x)
        return sum(real.(xs)) / length(xs)
    end
end


function solve_infinite_ground_state(; g2::Float64, lmax::Int, D::Int)
    H, ops = o3_nlsm_infinite_hamiltonian(; g2 = g2, lmax = lmax)

    ψ0 = random_infinite_mps(ops, D)

    alg = VUMPS(
        ;
        tol = tol,
        maxiter = maxiter,
        verbosity = verbosity,
    )

    t_solve = @elapsed begin
        ψ, envs, err = find_groundstate(ψ0, H, alg)
    end

    e_raw = expectation_value(ψ, H, envs)
    e = safe_energy_density(e_raw)

    return (
        psi = ψ,
        envs = envs,
        H = H,
        ops = ops,
        d = ops.d,
        energy_density = e,
        raw_energy = e_raw,
        err = err,
        solve_runtime_sec = t_solve,
    )
end



# ============================================================
# Infinite-MPS finite-window correlator
#
# C(r) = <n_0 ⋅ n_r>_connected
#
# xi_2nd(Lbox) =
#   sqrt(S(0)/S(kmin) - 1) / (2 sin(kmin/2)),
#   kmin = 2π/Lbox.
#
# This uses finite-window operator terms:
#
#   expectation_value(ψ, (1, 1+r) => op2, envs)
#
# If your MPSKit version rejects this syntax for InfiniteMPS,
# the correlator measurement needs to be rewritten using explicit
# transfer-matrix contractions.
# ============================================================

function try_expectation(ψ, envs, opterm)
    try
        return expectation_value(ψ, opterm, envs)
    catch
        return expectation_value(ψ, opterm)
    end
end


function one_site_expectation_inf(ψ, envs, op)
    return safe_real(try_expectation(ψ, envs, (1,) => op))
end


function two_site_expectation_inf(ψ, envs, r::Int, op2)
    @assert r >= 1
    return safe_real(try_expectation(ψ, envs, (1, 1 + r) => op2))
end


function finite_window_correlator_inf(ψ, envs, ops; Lbox::Int)
    @assert Lbox >= 4

    ndotn =
        ops.nx ⊗ ops.nx +
        ops.ny ⊗ ops.ny +
        ops.nz ⊗ ops.nz

    n2op = ops.nx * ops.nx + ops.ny * ops.ny + ops.nz * ops.nz

    mx = one_site_expectation_inf(ψ, envs, ops.nx)
    my = one_site_expectation_inf(ψ, envs, ops.ny)
    mz = one_site_expectation_inf(ψ, envs, ops.nz)

    disconnected = mx * mx + my * my + mz * mz

    C = zeros(Float64, Lbox)

    C[1] = one_site_expectation_inf(ψ, envs, n2op) - disconnected

    for r in 1:(Lbox - 1)
        C[r + 1] = two_site_expectation_inf(ψ, envs, r, ndotn) - disconnected
    end

    return C
end


function structure_factor_from_C(C::AbstractVector, k::Real)
    val = 0.0
    for r in 0:(length(C) - 1)
        val += C[r + 1] * cos(k * r)
    end
    return val
end


function xi_second_moment_finite_window(C::AbstractVector)
    Lbox = length(C)
    kmin = 2π / Lbox

    S0 = structure_factor_from_C(C, 0.0)
    Sk = structure_factor_from_C(C, kmin)

    ratio = S0 / Sk - 1.0

    if !isfinite(ratio) || ratio < 0
        @warn "Invalid second-moment ratio" Lbox = Lbox S0 = S0 Sk = Sk ratio = ratio
        return NaN, S0, Sk
    end

    xi = sqrt(ratio) / (2 * sin(kmin / 2))

    return xi, S0, Sk
end


function compute_xi_for_box(ψ, envs, ops; Lbox::Int)
    t_xi = @elapsed begin
        C = finite_window_correlator_inf(ψ, envs, ops; Lbox = Lbox)
        xi, S0, Sk = xi_second_moment_finite_window(C)
    end

    return (
        Lbox = Lbox,
        xi = xi,
        S0 = S0,
        Skmin = Sk,
        runtime_sec = t_xi,
    )
end

res = solve_infinite_ground_state(; g2, lmax, D)

@save filename res