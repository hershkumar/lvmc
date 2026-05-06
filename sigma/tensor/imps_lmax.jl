using LinearAlgebra
using Printf
using Statistics
using TensorKit
using MPSKit
using WignerSymbols
using Plots

# ============================================================
# User parameters
# ============================================================

const g2 = .75
const beta = 1 / g2

const D = 160
const lmaxs = collect(1:10)

const Lbox = 100

# Based on your run, tol = 1e-6 is more realistic than 1e-8.
const maxiter = 40
const tol = 1e-6
const verbosity = 1

const out_csv = "o3_imps_lmax_scan_D300_g2_$(g2).csv"
const out_energy_png = "o3_imps_energy_vs_lmax_D300_g2_$(g2).png"
const out_xi_png = "o3_imps_xi_vs_lmax_D300_g2_$(g2).png"
const out_combined_png = "o3_imps_lmax_convergence_D300_g2_$(g2).png"

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

            # Selection rule:
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
        lmax = lmax,
        d = ops.d,
        D = D,
        energy_density = e,
        raw_energy = e_raw,
        err = err,
        solve_runtime_sec = t_solve,
    )
end

# ============================================================
# Finite-window second-moment correlation length
#
# C(r) = <n_0 ⋅ n_r>_connected
#
# ξ₂nd(Lbox) =
#   sqrt(S(0)/S(kmin) - 1) / (2 sin(kmin/2)),
#   kmin = 2π/Lbox.
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


function compute_xi_Lbox(ψ, envs, ops; Lbox::Int)
    t_xi = @elapsed begin
        C = finite_window_correlator_inf(ψ, envs, ops; Lbox = Lbox)
        xi, S0, Sk = xi_second_moment_finite_window(C)
    end

    return (
        xi = xi,
        S0 = S0,
        Skmin = Sk,
        xi_runtime_sec = t_xi,
    )
end

# ============================================================
# Scan lmax
# ============================================================

function scan_lmax_D_fixed(; g2::Float64, D::Int, lmaxs, Lbox::Int)
    rows = NamedTuple[]

    println()
    println("O(3) NLSM iMPS truncation scan")
    println("Hamiltonian convention:")
    println("h = (g²/2)L² - (1/g²)nᵢ⋅nᵢ₊₁")
    println()
    println("g²      = ", g2)
    println("β       = ", 1 / g2)
    println("D       = ", D)
    println("Lbox    = ", Lbox)
    println("lmaxs   = ", lmaxs)
    println()

    println(
        rpad("lmax", 8),
        rpad("d", 8),
        rpad("e∞", 24),
        rpad("ξ₂nd", 18),
        rpad("S0", 18),
        rpad("Skmin", 18),
        rpad("err", 14),
        rpad("solve_s", 12),
        "xi_s"
    )
    println("-"^140)

    for lmax in lmaxs
        ground = solve_infinite_ground_state(
            ;
            g2 = g2,
            lmax = lmax,
            D = D,
        )

        xi_res = try
            compute_xi_Lbox(
                ground.psi,
                ground.envs,
                ground.ops;
                Lbox = Lbox,
            )
        catch err
            @warn "Correlation-length computation failed" lmax = lmax exception = (err, catch_backtrace())
            (
                xi = NaN,
                S0 = NaN,
                Skmin = NaN,
                xi_runtime_sec = NaN,
            )
        end

        row = (
            g2 = g2,
            beta = 1 / g2,
            lmax = lmax,
            d = ground.d,
            D = D,
            Lbox = Lbox,
            energy_density = ground.energy_density,
            shifted_energy_density = ground.energy_density + 1 / g2,
            xi = xi_res.xi,
            S0 = xi_res.S0,
            Skmin = xi_res.Skmin,
            vumps_err = ground.err,
            solve_runtime_sec = ground.solve_runtime_sec,
            xi_runtime_sec = xi_res.xi_runtime_sec,
        )

        push!(rows, row)

        @printf(
            "%-8d %-8d %-24.16f %-18.8e %-18.8e %-18.8e %-14.4e %-12.3f %.3f\n",
            row.lmax,
            row.d,
            row.energy_density,
            row.xi,
            row.S0,
            row.Skmin,
            row.vumps_err,
            row.solve_runtime_sec,
            row.xi_runtime_sec,
        )

        flush(stdout)
        GC.gc()
    end

    return rows
end

# ============================================================
# Save and plot
# ============================================================

function save_rows_csv(rows, filename::String)
    open(filename, "w") do io
        println(
            io,
            "g2,beta,lmax,d,D,Lbox,energy_density,shifted_energy_density,xi,S0,Skmin,vumps_err,solve_runtime_sec,xi_runtime_sec"
        )

        for r in rows
            println(
                io,
                join(
                    [
                        r.g2,
                        r.beta,
                        r.lmax,
                        r.d,
                        r.D,
                        r.Lbox,
                        r.energy_density,
                        r.shifted_energy_density,
                        r.xi,
                        r.S0,
                        r.Skmin,
                        r.vumps_err,
                        r.solve_runtime_sec,
                        r.xi_runtime_sec,
                    ],
                    ",",
                ),
            )
        end
    end
end


function plot_energy_convergence(rows)
    rs = sort(rows; by = r -> r.lmax)

    lvals = [r.lmax for r in rs]
    evals = [r.energy_density for r in rs]

    p = plot(
        lvals,
        evals;
        marker = :circle,
        linewidth = 2,
        xlabel = "Rotor truncation lmax",
        ylabel = "Energy density e∞",
        title = "O(3) iMPS energy convergence, D = $D, g² = $g2",
        legend = false,
        grid = true,
    )

    return p
end


function plot_xi_convergence(rows)
    rs = sort(rows; by = r -> r.lmax)

    lvals = [r.lmax for r in rs]
    xis = [r.xi for r in rs]

    p = plot(
        lvals,
        xis;
        marker = :circle,
        linewidth = 2,
        xlabel = "Rotor truncation lmax",
        ylabel = "ξ₂nd, Lbox = $Lbox",
        title = "O(3) iMPS correlation-length convergence, D = $D, g² = $g2",
        legend = false,
        grid = true,
    )

    return p
end


function plot_combined(rows)
    p1 = plot_energy_convergence(rows)
    p2 = plot_xi_convergence(rows)

    return plot(p1, p2; layout = (1, 2), size = (1100, 420))
end

# ============================================================
# Main
# ============================================================

rows = scan_lmax_D_fixed(
    ;
    g2 = g2,
    D = D,
    lmaxs = lmaxs,
    Lbox = Lbox,
)

save_rows_csv(rows, out_csv)
println("Saved CSV to: ", out_csv)

fig_e = plot_energy_convergence(rows)
display(fig_e)
savefig(fig_e, out_energy_png)
println("Saved energy plot to: ", out_energy_png)

fig_xi = plot_xi_convergence(rows)
display(fig_xi)
savefig(fig_xi, out_xi_png)
println("Saved xi plot to: ", out_xi_png)

fig_combined = plot_combined(rows)
display(fig_combined)
savefig(fig_combined, out_combined_png)
println("Saved combined plot to: ", out_combined_png)
