using LinearAlgebra
using Printf
using Statistics
using TensorKit
using MPSKit
using WignerSymbols

# ============================================================
# Command-line arguments
#
# Usage:
#
#   julia imps_o3_one_g2.jl g2 lmax D
#
# Example:
#
#   julia imps_o3_one_g2.jl 1.0 3 64
# ============================================================

if length(ARGS) < 3
    println("Usage: julia imps_o3_one_g2.jl g2 lmax D")
    println("Example: julia imps_o3_one_g2.jl 1.0 3 64")
    exit(1)
end

const g2 = parse(Float64, ARGS[1])
const lmax = parse(Int, ARGS[2])
const D = parse(Int, ARGS[3])

# ============================================================
# User-adjustable solver settings
# ============================================================

const maxiter = 100
const tol = 1e-8
const verbosity = 3

# Correlation-length box scan settings.
const Lbox_start = 16
const Lbox_factor = 1.5
const Lbox_max = 1024

# Stop when |xi(L₂)-xi(L₁)| / max(|xi(L₂)|, eps) < xi_rel_tol.
const xi_rel_tol = 1e-3

# Also require an absolute tolerance, useful when xi is small.
const xi_abs_tol = 1e-4

# If xi is a sizable fraction of the box, keep increasing box size even
# if consecutive estimates look similar.
const max_xi_over_Lbox = 0.20

const out_prefix = "o3_imps_g2_$(g2)_lmax_$(lmax)_D_$(D)"
const out_csv = out_prefix * "_xi_scan.csv"
const out_summary = out_prefix * "_summary.txt"


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


# ============================================================
# Infinite O(3) NLSM Hamiltonian
#
# Convention:
#
# h = (g² / 2) L_i² - (1 / g²) n_i ⋅ n_{i+1}
#
# If comparing to:
#
# h = (g²/2)L² + (1/g²)(1 - n⋅n),
#
# shift energy density by +1/g².
# ============================================================

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


# ============================================================
# iMPS / VUMPS helpers
# ============================================================

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


function converged_xi(prev_xi, curr_xi, Lbox)
    if !isfinite(curr_xi)
        return false
    end

    if prev_xi === nothing || !isfinite(prev_xi)
        return false
    end

    absdiff = abs(curr_xi - prev_xi)
    reldiff = absdiff / max(abs(curr_xi), eps(Float64))

    # Do not accept convergence if the box is too small compared to xi.
    # This avoids false convergence from finite-window saturation.
    box_large_enough = curr_xi / Lbox < max_xi_over_Lbox

    return box_large_enough && (absdiff < xi_abs_tol || reldiff < xi_rel_tol)
end


function adaptive_correlation_length(ψ, envs, ops)
    rows = NamedTuple[]

    Lbox = Lbox_start
    prev_xi = nothing
    converged = false

    println()
    println("Adaptive finite-window second-moment ξ scan")
    println(
        rpad("Lbox", 10),
        rpad("xi", 22),
        rpad("S0", 22),
        rpad("Skmin", 22),
        rpad("rel_change", 18),
        "time_sec"
    )
    println("-"^110)

    while Lbox <= Lbox_max
        xr = compute_xi_for_box(ψ, envs, ops; Lbox = Lbox)

        rel_change =
            if prev_xi === nothing || !isfinite(prev_xi) || !isfinite(xr.xi)
                NaN
            else
                abs(xr.xi - prev_xi) / max(abs(xr.xi), eps(Float64))
            end

        @printf(
            "%-10d %-22.12e %-22.12e %-22.12e %-18.8e %.3f\n",
            xr.Lbox,
            xr.xi,
            xr.S0,
            xr.Skmin,
            rel_change,
            xr.runtime_sec,
        )

        push!(
            rows,
            (
                Lbox = xr.Lbox,
                xi = xr.xi,
                S0 = xr.S0,
                Skmin = xr.Skmin,
                rel_change = rel_change,
                runtime_sec = xr.runtime_sec,
            ),
        )

        if converged_xi(prev_xi, xr.xi, Lbox)
            converged = true
            break
        end

        prev_xi = xr.xi
        Lbox *= Lbox_factor
    end

    final = rows[end]

    return (
        xi = final.xi,
        Lbox = final.Lbox,
        S0 = final.S0,
        Skmin = final.Skmin,
        rel_change = final.rel_change,
        converged = converged,
        rows = rows,
    )
end


# ============================================================
# Output helpers
# ============================================================

function save_xi_scan_csv(rows, filename::String)
    open(filename, "w") do io
        println(io, "g2,beta,lmax,d,D,Lbox,xi,S0,Skmin,rel_change,runtime_sec")

        for r in rows
            println(
                io,
                join(
                    [
                        g2,
                        1 / g2,
                        lmax,
                        (lmax + 1)^2,
                        D,
                        r.Lbox,
                        r.xi,
                        r.S0,
                        r.Skmin,
                        r.rel_change,
                        r.runtime_sec,
                    ],
                    ",",
                ),
            )
        end
    end
end


function save_summary_txt(filename::String; ground, xi_result)
    open(filename, "w") do io
        println(io, "O(3) NLSM infinite-MPS result")
        println(io, "Hamiltonian convention:")
        println(io, "h = (g²/2)L² - (1/g²)n_i⋅n_{i+1}")
        println(io)
        println(io, "g2 = ", g2)
        println(io, "beta = ", 1 / g2)
        println(io, "lmax = ", lmax)
        println(io, "local dimension d = ", ground.d)
        println(io, "bond dimension D = ", D)
        println(io)
        println(io, "energy_density = ", ground.energy_density)
        println(io, "vumps_err = ", ground.err)
        println(io, "vumps_runtime_sec = ", ground.solve_runtime_sec)
        println(io)
        println(io, "xi_2nd = ", xi_result.xi)
        println(io, "xi_Lbox = ", xi_result.Lbox)
        println(io, "xi_converged = ", xi_result.converged)
        println(io, "xi_rel_change = ", xi_result.rel_change)
        println(io, "S0 = ", xi_result.S0)
        println(io, "Skmin = ", xi_result.Skmin)
        println(io)
        println(io, "Energy-density convention conversion:")
        println(io, "If using h = (g²/2)L² + (1/g²)(1 - n⋅n),")
        println(io, "then shifted_energy_density = energy_density + 1/g² = ",
                ground.energy_density + 1 / g2)
    end
end


# ============================================================
# Main
# ============================================================

println()
println("O(3) NLSM infinite-MPS single-point run")
println("g²    = ", g2)
println("beta  = ", 1 / g2)
println("lmax  = ", lmax)
println("D     = ", D)
println("d     = ", (lmax + 1)^2)
println()

ground = solve_infinite_ground_state(
    ;
    g2 = g2,
    lmax = lmax,
    D = D,
)

println()
println("Ground state result")
println("energy density = ", ground.energy_density)
println("VUMPS err      = ", ground.err)
println("solve time     = ", ground.solve_runtime_sec, " sec")

xi_result = adaptive_correlation_length(
    ground.psi,
    ground.envs,
    ground.ops,
)

println()
println("Final correlation-length estimate")
println("xi_2nd      = ", xi_result.xi)
println("Lbox        = ", xi_result.Lbox)
println("converged   = ", xi_result.converged)
println("rel_change  = ", xi_result.rel_change)
println("S0          = ", xi_result.S0)
println("Skmin       = ", xi_result.Skmin)

save_xi_scan_csv(xi_result.rows, out_csv)
println("Saved xi scan CSV to: ", out_csv)

save_summary_txt(out_summary; ground = ground, xi_result = xi_result)
println("Saved summary to: ", out_summary)
