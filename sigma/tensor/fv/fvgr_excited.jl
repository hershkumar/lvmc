using ITensors
using ITensorMPS
using LinearAlgebra
using Printf
using Statistics
using WignerSymbols
using JLD2
using Plots
using Dates
using CUDA


# ============================================================
# Usage:
#
#   julia fvgor_excited_pbc.jl g2 N D lmax [--open] [--excited] [--outdir data] [--tol 1e-8] [--maxiter 100] [--verbosity 3] [--penalty 100.0] [--device gpu|cpu|auto] [--gpu-id 0] [--allow-scalar false] [--unified-memory false]
#
# Example:
#
#   julia fvgor_excited_pbc.jl 1.0 40 400 4 --device gpu --gpu-id 0 --unified-memory true
#
# Hamiltonian:
#
#   H = sum_i (g²/2) L_i² - (1/g²) sum_i n_i · n_{i+1 mod N}
#
# Real-basis variant:
#   This file uses a real tesseral-harmonic basis obtained by a unitary
#   change of basis from the complex |l,m> spherical basis. The local
#   matrices are transformed as O_real = S' * O_complex * S, so the
#   Hamiltonian is unitarily equivalent to fvgo.jl but can use real MPS
#   tensors and real MPO tensors.
#
# By default periodic boundary conditions are used, including the wraparound bond n_N · n_1.
# Use --open to run the open-boundary Hamiltonian instead.
#
# Output format intentionally matches the provided MPSKit script:
#
#   data/g2_<g2>_N_<N>_lmax_<lmax>_D_<D>_ground_state.jld2
#   data/g2_<g2>_N_<N>_lmax_<lmax>_D_<D>_correlator.csv
#   data/g2_<g2>_N_<N>_lmax_<lmax>_D_<D>_first_excited_state.jld2
#   data/g2_<g2>_N_<N>_lmax_<lmax>_D_<D>_correlator_logplot.png
#
# GPU notes:
#
#   This uses the current ITensors.jl CUDA extension path: load CUDA.jl,
#   build dense CPU ITensors, then move the MPO/MPS to the NVIDIA GPU with cu(...).
#   CUDA acceleration helps most when D and the local Hilbert dimension are large enough
#   that contractions/factorizations dominate overhead. For small N/D/lmax, CPU can be faster.
#
#   With --unified-memory true, arrays uploaded with cu(...) are allocated as
#   CUDA.UnifiedMemory. This can oversubscribe VRAM by paging through host memory,
#   but it can be much slower and does not reduce the size of individual DMRG contractions.
#
#   Add CUDA to your project if needed:
#
#       julia --project -e 'using Pkg; Pkg.add("CUDA")'
#
#   Useful environment variables on clusters:
#
#       export JULIA_CUDA_USE_BINARYBUILDER=true
#       export CUDA_VISIBLE_DEVICES=0
#
# ============================================================


# ============================================================
# Argument parsing
# ============================================================

function get_kwarg(args, name::String, default)
    idx = findfirst(==(name), args)
    if idx === nothing
        return default
    end
    if idx == length(args)
        error("Flag $name requires a value.")
    end
    return args[idx + 1]
end

function has_flag(args, name::String)
    return any(==(name), args)
end

function parse_bool_string(x)
    s = lowercase(String(x))
    if s in ("1", "true", "yes", "y", "on")
        return true
    elseif s in ("0", "false", "no", "n", "off")
        return false
    else
        error("Could not parse boolean value: $x")
    end
end

if length(ARGS) < 4
    println("Usage: julia fvgor_excited_pbc.jl g2 N D lmax [--open] [--excited] [--outdir data] [--tol 1e-8] [--maxiter 100] [--verbosity 3] [--penalty 100.0] [--device gpu|cpu|auto] [--gpu-id 0] [--allow-scalar false] [--unified-memory false]")
    println("Example: julia fvgor_excited_pbc.jl 1.0 40 400 4 --device gpu --gpu-id 0")
    exit(1)
end

const g2 = parse(Float64, ARGS[1])
const N = parse(Int, ARGS[2])
const D = parse(Int, ARGS[3])
const lmax = parse(Int, ARGS[4])
# Periodic boundary conditions are now the default.
# This includes the wraparound N--1 bond in the Hamiltonian.
# Use --open to explicitly request the OBC Hamiltonian.
# --pbc is accepted as a no-op compatibility flag.
const open_boundary = has_flag(ARGS, "--open")
const periodic_boundary = !open_boundary
const run_excited = has_flag(ARGS, "--excited")
const boundary = open_boundary ? "open" : "pbc"

const outdir = String(get_kwarg(ARGS, "--outdir", "data"))
const tol = parse(Float64, get_kwarg(ARGS, "--tol", "1e-5"))
const maxiter = parse(Int, get_kwarg(ARGS, "--maxiter", "12"))
const verbosity = parse(Int, get_kwarg(ARGS, "--verbosity", "1"))
const penalty_weight = parse(Float64, get_kwarg(ARGS, "--penalty", "100.0"))

const device_arg_raw = String(get_kwarg(ARGS, "--device", has_flag(ARGS, "--cpu") ? "cpu" : (has_flag(ARGS, "--gpu") ? "gpu" : "gpu")))
const device_arg = lowercase(device_arg_raw)
const gpu_id = parse(Int, get_kwarg(ARGS, "--gpu-id", "0"))
const allow_scalar = parse_bool_string(get_kwarg(ARGS, "--allow-scalar", "false"))
const use_unified_memory = parse_bool_string(get_kwarg(ARGS, "--unified-memory", "false"))

if has_flag(ARGS, "--open") && has_flag(ARGS, "--pbc")
    error("Use either --open or --pbc, not both. Periodic boundary conditions are the default; --pbc is optional.")
end

if !(device_arg in ("gpu", "cpu", "auto"))
    error("--device must be one of: gpu, cpu, auto")
end

if g2 <= 0
    error("g² must be positive.")
end
if N < 4
    error("N must be at least 4 for the finite-chain calculation.")
end
if D < 1
    error("D must be positive.")
end
if lmax < 0
    error("lmax must be nonnegative.")
end
if penalty_weight <= 0
    error("--penalty must be positive for excited-state DMRG.")
end

function init_cuda_backend(device_arg::String, gpu_id::Int, allow_scalar::Bool)
    if device_arg == "cpu"
        return false
    end

    if !CUDA.functional()
        if device_arg == "auto"
            @warn "CUDA is not functional; falling back to CPU."
            return false
        else
            error("CUDA is not functional. Use --device cpu to force a CPU run, or fix the CUDA installation.")
        end
    end

    devs = CUDA.devices()
    if gpu_id < 0 || gpu_id >= length(devs)
        error("--gpu-id=$gpu_id is invalid; CUDA reports $(length(devs)) visible device(s).")
    end

    CUDA.device!(gpu_id)
    CUDA.allowscalar(allow_scalar)

    # Trigger CUDA context creation early, so failures occur before DMRG starts.
    CUDA.synchronize()
    return true
end

const USE_CUDA = init_cuda_backend(device_arg, gpu_id, allow_scalar)

function to_device(x)
    if !USE_CUDA
        return x
    end
    # CUDA.jl unified memory path. This keeps the same `cu` Adapt-based upload
    # behavior as the original script, but requests CUDA.UnifiedMemory storage.
    # Note: `cu` may convert Float64/ComplexF64 to Float32/ComplexF32 by design.
    return cu(x; unified = use_unified_memory)
end

to_host(x) = USE_CUDA ? ITensors.cpu(x) : x
sync_device() = USE_CUDA ? CUDA.synchronize() : nothing


# Force tensors used for post-processing to ordinary CPU storage. This avoids
# ITensors/NDTensors contraction failures caused by mixing CUDA.DeviceMemory,
# CUDA.UnifiedMemory, CPU Vector storage, and ComplexF32/ComplexF64 operators.
function force_cpu_itensor(A::ITensor)
    try
        return ITensors.cpu(A)
    catch
        return A
    end
end

function force_cpu_mps(ψ::MPS)
    ψc = copy(ψ)
    @inbounds for n in 1:length(ψc)
        ψc[n] = force_cpu_itensor(ψc[n])
    end
    return ψc
end

function force_cpu_mpo(H::MPO)
    Hc = copy(H)
    @inbounds for n in 1:length(Hc)
        Hc[n] = force_cpu_itensor(Hc[n])
    end
    return Hc
end

function scalar_type_of_mps(ψ::MPS)
    try
        T = eltype(ψ[1])
        return T <: Number ? T : Float64
    catch
        return Float64
    end
end

function print_cuda_memory_status(label::String)
    if USE_CUDA
        println("\nCUDA memory status: ", label)
        CUDA.memory_status()
        flush(stdout)
    end
end

function host_real_array(x)
    y = real.(x)
    return y isa CUDA.AbstractGPUArray ? Array(y) : collect(y)
end

mkpath(outdir)

const basis_tag = "realbasis"
const tag = @sprintf("g2_%g_N_%d_lmax_%d_D_%d_%s_%s", g2, N, lmax, D, boundary, basis_tag)
const gs_file = joinpath(outdir, tag * "_ground_state.jld2")
const corr_file = joinpath(outdir, tag * "_correlator.csv")
const ex_file = joinpath(outdir, tag * "_first_excited_state.jld2")
const plot_file = joinpath(outdir, tag * "_correlator_logplot.png")

println("g²        = ", g2)
println("N         = ", N)
println("D         = ", D)
println("lmax      = ", lmax)
println("basis     = real tesseral")
println("boundary  = ", boundary)
println("excited   = ", run_excited ? "enabled" : "disabled")
println("tol       = ", tol)
println("maxiter   = ", maxiter)
println("verbosity = ", verbosity)
println("penalty   = ", penalty_weight)
println("outdir    = ", outdir)
println("threads   = ", Threads.nthreads())
println("device    = ", USE_CUDA ? "cuda" : "cpu")
if USE_CUDA
    println("gpu id    = ", gpu_id)
    println("gpu name  = ", CUDA.name(CUDA.device()))
    println("allowscalar = ", allow_scalar)
    println("unified memory = ", use_unified_memory)
end
flush(stdout)


# ============================================================
# ITensor custom rotor site type: real tesseral rotor basis
# ============================================================

function ITensors.space(::SiteType"RotorO3R"; lmax::Int)
    return (lmax + 1)^2
end

function infer_lmax_from_dim(d::Int)
    lm = round(Int, sqrt(d) - 1)
    if (lm + 1)^2 != d
        error("RotorO3R local dimension d=$d is not of form (lmax+1)^2.")
    end
    return lm
end


# ============================================================
# Rotor basis and local O(3) operators in a real tesseral basis
#
# Construction:
#   1. Build the same complex spherical-basis matrices as fvgo.jl.
#   2. Build a block-diagonal unitary S whose columns are real tesseral
#      combinations expressed in the complex |l,m> basis.
#   3. Transform every local operator as O_real = S' * O_complex * S.
#
# This is a unitary change of basis, so the Hamiltonian is the same operator:
#   H = sum_i (g²/2) L_i² - (1/g²) sum_i (Nx_i Nx_j + Ny_i Ny_j + Nz_i Nz_j).
# ============================================================

function rotor_basis(lmax::Int)
    states = Tuple{Int,Int}[]
    index = Dict{Tuple{Int,Int},Int}()

    sizehint!(states, (lmax + 1)^2)

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

    @inbounds for (a, (l, m)) in enumerate(states)
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

function real_tesseral_transform(lmax::Int)
    states, index = rotor_basis(lmax)
    d = length(states)
    S = zeros(ComplexF64, d, d)

    col = 1
    for l in 0:lmax
        # m = 0 real spherical harmonic basis vector.
        S[index[(l, 0)], col] = 1.0 + 0.0im
        col += 1

        for m in 1:l
            # Real cosine-like tesseral vector:
            #   |l,m,c> = ( (-1)^m |l,m> + |l,-m> ) / sqrt(2)
            S[index[(l,  m)], col] = (-1.0)^m / sqrt(2)
            S[index[(l, -m)], col] = 1.0 / sqrt(2)
            col += 1

            # Real sine-like tesseral vector:
            #   |l,m,s> = ( (-1)^m |l,m> - |l,-m> ) / (i sqrt(2))
            S[index[(l,  m)], col] = (-1.0)^m / (im * sqrt(2))
            S[index[(l, -m)], col] = -1.0 / (im * sqrt(2))
            col += 1
        end
    end

    # Check unitarity to catch basis-ordering mistakes.
    err = norm(S' * S - Matrix{ComplexF64}(I, d, d))
    if err > 1e-12
        error("Real tesseral transform is not unitary: ||S' S - I|| = $err")
    end

    return S
end

function realify_matrix(M::AbstractMatrix{<:Complex}; name::String = "operator", tol::Float64 = 1e-10)
    max_imag = maximum(abs.(imag.(M)))
    if max_imag > tol
        @warn "Transformed real-basis matrix has residual imaginary part" name = name max_imag = max_imag
    end
    return Matrix{Float64}(real.(M))
end

const ROTOR_OP_CACHE = Dict{Int,Dict{String,Matrix{Float64}}}()

function build_rotor_matrices(lmax::Int)
    states, _ = rotor_basis(lmax)
    d = length(states)

    L2_c = zeros(ComplexF64, d, d)

    @inbounds for (a, (l, m)) in enumerate(states)
        L2_c[a, a] = l * (l + 1)
    end

    Tm = n_spherical_matrix(lmax, -1)
    T0 = n_spherical_matrix(lmax, 0)
    Tp = n_spherical_matrix(lmax, 1)

    # Same spherical-to-Cartesian convention as fvgo.jl:
    #   n_{+1} = -(nx + i ny)/sqrt(2)
    #   n_0    = nz
    #   n_{-1} =  (nx - i ny)/sqrt(2)
    nx_c = (Tm - Tp) / sqrt(2)
    ny_c = im * (Tm + Tp) / sqrt(2)
    nz_c = T0

    # Hermitian cleanup in the original complex basis.
    L2_c = 0.5 * (L2_c + L2_c')
    nx_c = 0.5 * (nx_c + nx_c')
    ny_c = 0.5 * (ny_c + ny_c')
    nz_c = 0.5 * (nz_c + nz_c')

    S = real_tesseral_transform(lmax)

    L2 = realify_matrix(S' * L2_c * S; name = "L2")
    nx = realify_matrix(S' * nx_c * S; name = "Nx")
    ny = realify_matrix(S' * ny_c * S; name = "Ny")
    nz = realify_matrix(S' * nz_c * S; name = "Nz")

    # Symmetrize after dropping roundoff imaginary parts.
    L2 = 0.5 * (L2 + transpose(L2))
    nx = 0.5 * (nx + transpose(nx))
    ny = 0.5 * (ny + transpose(ny))
    nz = 0.5 * (nz + transpose(nz))

    n2 = nx * nx + ny * ny + nz * nz
    Id = Matrix{Float64}(I, d, d)

    return Dict(
        "L2" => L2,
        "Nx" => nx,
        "Ny" => ny,
        "Nz" => nz,
        "N2" => n2,
        "Id" => Id,
    )
end

function rotor_ops(lmax::Int)
    return get!(ROTOR_OP_CACHE, lmax) do
        build_rotor_matrices(lmax)
    end
end

function get_rotor_op(lmax::Int, opname::String)
    return rotor_ops(lmax)[opname]
end

function rotor_op_itensor(opname::String, s::Index)
    lm = infer_lmax_from_dim(dim(s))
    mat = get_rotor_op(lm, opname)

    # ITensor operator index convention:
    #   row/output index = prime(s)
    #   col/input index  = dag(s)
    return ITensor(mat, prime(s), dag(s))
end

function ITensors.op(::OpName"L2", ::SiteType"RotorO3R", s::Index)
    return rotor_op_itensor("L2", s)
end

function ITensors.op(::OpName"Nx", ::SiteType"RotorO3R", s::Index)
    return rotor_op_itensor("Nx", s)
end

function ITensors.op(::OpName"Ny", ::SiteType"RotorO3R", s::Index)
    return rotor_op_itensor("Ny", s)
end

function ITensors.op(::OpName"Nz", ::SiteType"RotorO3R", s::Index)
    return rotor_op_itensor("Nz", s)
end

function ITensors.op(::OpName"N2", ::SiteType"RotorO3R", s::Index)
    return rotor_op_itensor("N2", s)
end

function ITensors.op(::OpName"Id", ::SiteType"RotorO3R", s::Index)
    return rotor_op_itensor("Id", s)
end

function ITensors.state(::StateName"L0M0", ::SiteType"RotorO3R", s::Index)
    return 1
end


# ============================================================
# Numerical safety helpers
# ============================================================

function safe_real(x; tol_imag = 1e-8)
    if abs(imag(x)) > tol_imag
        @warn "Non-negligible imaginary part" imag_part = imag(x)
    end
    return real(x)
end

function safe_energy(x)
    if x isa Number
        return safe_real(x)
    end
    xs = collect(x)
    return sum(real.(xs))
end


# ============================================================
# Sweep schedules
# ============================================================

function maxdim_schedule(D::Int)
    base = Int[20, 40, 80, 120, 200, 300, 400, 600, 800, 1200, 1600, 2000, 3000, 4000]
    sched = Int[]

    for x in base
        if x < D
            push!(sched, x)
        end
    end

    push!(sched, D)
    return unique(sched)
end

function noise_schedule(nsweeps::Int)
    noise = zeros(Float64, nsweeps)

    for sw in 1:nsweeps
        if sw <= max(1, nsweeps ÷ 4)
            noise[sw] = 1e-6
        elseif sw <= max(2, nsweeps ÷ 2)
            noise[sw] = 1e-8
        else
            noise[sw] = 0.0
        end
    end

    return noise
end

function dmrg_kwargs(D::Int)
    maxdims = maxdim_schedule(D)
    nsweeps_eff = max(maxiter, length(maxdims))

    if length(maxdims) < nsweeps_eff
        append!(maxdims, fill(D, nsweeps_eff - length(maxdims)))
    end

    cutoffs = fill(tol, nsweeps_eff)
    noises = noise_schedule(nsweeps_eff)

    return (
        nsweeps = nsweeps_eff,
        maxdim = maxdims,
        cutoff = cutoffs,
        noise = noises,
        outputlevel = verbosity,
    )
end


# ============================================================
# Finite Hamiltonian: PBC by default, OBC with --open
# ============================================================

function o3_nlsm_opsum(; g2::Float64, N::Int, open_boundary::Bool = false)
    os = OpSum()

    c_kin = g2 / 2
    c_int = -1 / g2

    @inbounds for i in 1:N
        os += c_kin, "L2", i
    end

    # Nearest-neighbor chain bonds.
    @inbounds for i in 1:(N - 1)
        os += c_int, "Nx", i, "Nx", i + 1
        os += c_int, "Ny", i, "Ny", i + 1
        os += c_int, "Nz", i, "Nz", i + 1
    end

    if !open_boundary
        # PBC wraparound. Use ordered finite sites (1, N).
        # Since n_i · n_j is symmetric, this is equivalent to the N--1 bond.
        os += c_int, "Nx", 1, "Nx", N
        os += c_int, "Ny", 1, "Ny", N
        os += c_int, "Nz", 1, "Nz", N
    end

    return os
end

function o3_nlsm_finite_hamiltonian(; g2::Float64, N::Int, lmax::Int, open_boundary::Bool = false)
    sites = siteinds("RotorO3R", N; lmax = lmax)
    os = o3_nlsm_opsum(; g2 = g2, N = N, open_boundary = open_boundary)

    # Let ITensor infer scalar type, since custom ops return ITensors.
    H = MPO(os, sites)

    ops = rotor_ops(lmax)

    return H, sites, ops
end

function random_finite_mps(sites, D::Int)
    return random_mps(Float64, sites; linkdims = min(D, max(1, min(10, D))))
end


# ============================================================
# Optional warm start from a previously saved MPS
#
# Search `outdir` for a saved ground-state JLD2 file with the same
# g², N, lmax, and boundary. Prefer the largest D <= target D; if none
# exists, use the smallest D > target D. If no compatible file is found,
# the script falls back to a random MPS exactly as before.
# ============================================================

function parse_ground_state_filename(path::AbstractString)
    name = basename(path)
    m = match(r"^g2_([^_]+)_N_(\d+)_lmax_(\d+)_D_(\d+)_(open|pbc)_realbasis_ground_state\.jld2$", name)
    if m === nothing
        return nothing
    end

    return (
        g2 = parse(Float64, m.captures[1]),
        N = parse(Int, m.captures[2]),
        lmax = parse(Int, m.captures[3]),
        D = parse(Int, m.captures[4]),
        boundary = String(m.captures[5]),
        path = String(path),
    )
end

function saved_psi_from_object(x)
    if hasproperty(x, :psi)
        return getproperty(x, :psi)
    elseif x isa AbstractDict && haskey(x, "psi")
        return x["psi"]
    elseif x isa AbstractDict && haskey(x, :psi)
        return x[:psi]
    else
        return nothing
    end
end

function mps_maxlinkdim_safe(ψ::MPS)
    try
        return maxlinkdim(ψ)
    catch
        if length(ψ) <= 1
            return 1
        end
        vals = Int[]
        for b in 1:(length(ψ) - 1)
            lb = commonind(ψ[b], ψ[b + 1])
            if lb !== nothing
                push!(vals, dim(lb))
            end
        end
        return isempty(vals) ? 1 : maximum(vals)
    end
end

function compatible_starting_mps(ψ::MPS; N::Int, lmax::Int)
    if length(ψ) != N
        return false
    end

    expected_dim = (lmax + 1)^2
    for i in 1:N
        s = siteind(ψ, i)
        if dim(s) != expected_dim
            return false
        end
    end

    return true
end

function find_starting_mps_file(; outdir::String, g2::Float64, N::Int, lmax::Int, D::Int, boundary::String, current_file::String)
    if !isdir(outdir)
        return nothing
    end

    candidates = []
    for path in readdir(outdir; join = true)
        if abspath(path) == abspath(current_file)
            continue
        end
        meta = parse_ground_state_filename(path)
        if meta === nothing
            continue
        end
        if isapprox(meta.g2, g2; rtol = 1e-12, atol = 1e-14) &&
           meta.N == N &&
           meta.lmax == lmax &&
           meta.boundary == boundary
            push!(candidates, meta)
        end
    end

    if isempty(candidates)
        return nothing
    end

    leq = filter(x -> x.D <= D, candidates)
    if !isempty(leq)
        return sort(leq; by = x -> x.D, rev = true)[1]
    end

    return sort(candidates; by = x -> x.D)[1]
end

function load_starting_mps(; outdir::String, g2::Float64, N::Int, lmax::Int, D::Int, boundary::String, current_file::String)
    meta = find_starting_mps_file(; outdir = outdir, g2 = g2, N = N, lmax = lmax, D = D, boundary = boundary, current_file = current_file)
    if meta === nothing
        println("No compatible saved ground-state MPS found in: ", outdir)
        println("Using random initial MPS.")
        flush(stdout)
        return nothing
    end

    println("Found candidate starting MPS:")
    println("  file      = ", meta.path)
    println("  saved D   = ", meta.D)
    println("  target D  = ", D)
    flush(stdout)

    try
        gs_saved = JLD2.load(meta.path, "gs")
        ψ = saved_psi_from_object(gs_saved)
        if ψ === nothing
            @warn "Candidate file does not contain a `gs.psi` field; ignoring it." file = meta.path
            return nothing
        end

        ψ = force_cpu_mps(ψ)

        if !compatible_starting_mps(ψ; N = N, lmax = lmax)
            @warn "Candidate MPS has incompatible site structure; ignoring it." file = meta.path
            return nothing
        end

        println("Using saved MPS as DMRG initial state.")
        println("  start maxlinkdim = ", mps_maxlinkdim_safe(ψ))
        flush(stdout)
        return (psi = ψ, path = meta.path, D = meta.D)
    catch err
        @warn "Could not load candidate starting MPS; falling back to random initial MPS." file = meta.path exception = (err, catch_backtrace())
        return nothing
    end
end


# ============================================================
# Ground and first excited states
# ============================================================

function solve_ground_state(; g2::Float64, N::Int, D::Int, lmax::Int)
    start = load_starting_mps(;
        outdir = outdir,
        g2 = g2,
        N = N,
        lmax = lmax,
        D = D,
        boundary = boundary,
        current_file = gs_file,
    )

    if start === nothing
        H_cpu, sites, ops = o3_nlsm_finite_hamiltonian(; g2 = g2, N = N, lmax = lmax, open_boundary = open_boundary)
        ψ0_cpu = random_finite_mps(sites, D)
        starting_mps_file = nothing
        starting_mps_D = nothing
    else
        ψ0_cpu = start.psi
        sites = [siteind(ψ0_cpu, i) for i in 1:N]
        os = o3_nlsm_opsum(; g2 = g2, N = N, open_boundary = open_boundary)
        H_cpu = MPO(os, sites)
        ops = rotor_ops(lmax)
        starting_mps_file = start.path
        starting_mps_D = start.D
    end

    H = to_device(H_cpu)
    ψ0 = to_device(ψ0_cpu)
    sync_device()

    kwargs = dmrg_kwargs(D)

    ψ = nothing
    E = NaN

    t = @elapsed begin
        E, ψ = dmrg(
            H,
            ψ0;
            nsweeps = kwargs.nsweeps,
            maxdim = kwargs.maxdim,
            cutoff = kwargs.cutoff,
            noise = kwargs.noise,
            outputlevel = kwargs.outputlevel,
	    write_when_maxdim_exceeds=1
        )
        sync_device()
    end

    E = safe_energy(E)

    return (
        psi = ψ,
        envs = nothing,
        H = H,
        sites = sites,
        ops = ops,
        energy = E,
        energy_density = E / N,
        err = nothing,
        runtime_sec = t,
        method = USE_CUDA ? "ITensorMPS.dmrg CUDA" : "ITensorMPS.dmrg",
        starting_mps_file = starting_mps_file,
        starting_mps_D = starting_mps_D,
        maxdim_schedule = kwargs.maxdim,
        cutoff_schedule = kwargs.cutoff,
        noise_schedule = kwargs.noise,
    )
end

function solve_first_excited_state(H, sites, ψ_ground; N::Int, D::Int)
    # Defensive check: ITensorMPS.dmrg(H, [ψ_ground], ψ0; ...) requires the
    # penalty state and trial state to share exact site Index IDs, not just the
    # same tags and dimensions.
    @inbounds for i in 1:N
        if siteind(ψ_ground, i) != sites[i]
            error("Excited-state DMRG site-index mismatch at site $i: " *
                  "ψ_ground has $(siteind(ψ_ground, i)), but sites[$i] is $(sites[i]). " *
                  "Build H and ψ0 from siteind(ψ_ground, i), not from a fresh siteinds(...) call.")
        end
    end

    ψ0 = to_device(random_finite_mps(sites, D))
    sync_device()
    kwargs = dmrg_kwargs(D)

    ψ_ex = nothing
    E_ex = NaN

    t = @elapsed begin
        # ITensorMPS excited-state DMRG:
        # pass previous states as penalty states immediately after H.
        E_ex, ψ_ex = dmrg(
            H,
            [ψ_ground],
            ψ0;
            nsweeps = kwargs.nsweeps,
            maxdim = kwargs.maxdim,
            cutoff = kwargs.cutoff,
            noise = kwargs.noise,
            outputlevel = kwargs.outputlevel,
            weight = penalty_weight,
        )
        sync_device()
    end

    E_ex = safe_energy(E_ex)
    overlap = abs(inner(ψ_ground, ψ_ex))

    return (
        psi = ψ_ex,
        envs = nothing,
        energy = E_ex,
        energy_density = E_ex / N,
        excitation_output = nothing,
        runtime_sec = t,
        method = USE_CUDA ? "ITensorMPS.dmrg CUDA penalty-state excited-state DMRG" : "ITensorMPS.dmrg penalty-state excited-state DMRG",
        penalty_weight = penalty_weight,
        overlap_with_ground = overlap,
        maxdim_schedule = kwargs.maxdim,
        cutoff_schedule = kwargs.cutoff,
        noise_schedule = kwargs.noise,
    )
end


# ============================================================
# Correlator C(r)
#
# C(r) = (1/N) sum_i <n_i · n_{i+r}> - |<n>|²
#
# For r = 0, use <n_i²>.
# For r > 0, average over all wrapped PBC pairs.
# ============================================================


function rotor_op_itensor_typed(opname::String, s::Index, ::Type{T}; use_gpu::Bool = false) where {T<:Number}
    lm = infer_lmax_from_dim(dim(s))
    mat = T.(get_rotor_op(lm, opname))
    O = ITensor(mat, prime(s), dag(s))

    # For GPU correlators, the local operator must live on the same backend as
    # the MPS tensors. These operators are cached by the optimized correlator,
    # so this upload happens O(N), not O(N^2).
    return (use_gpu && USE_CUDA) ? to_device(O) : O
end

function rotor_op_product_itensor_typed(op1::String, op2::String, s::Index, ::Type{T}; use_gpu::Bool = false) where {T<:Number}
    lm = infer_lmax_from_dim(dim(s))
    mat = T.(get_rotor_op(lm, op1) * get_rotor_op(lm, op2))
    O = ITensor(mat, prime(s), dag(s))
    return (use_gpu && USE_CUDA) ? to_device(O) : O
end

function prime_inds(A::ITensor, inds_to_prime)
    B = A
    for ind in inds_to_prime
        if ind !== nothing && hasind(B, ind)
            B = prime(B, ind)
        end
    end
    return B
end

function left_linkind(ψ::MPS, i::Int)
    return i > 1 ? commonind(ψ[i - 1], ψ[i]) : nothing
end

function right_linkind(ψ::MPS, i::Int)
    return i < length(ψ) ? commonind(ψ[i], ψ[i + 1]) : nothing
end

function scalar_real_itensor(T::ITensor; tol_imag = 1e-6, warn_imag::Bool = false)
    # The contraction itself can run on the GPU. Extracting the final scalar is
    # done from a CPU scalar ITensor to avoid CUDA scalar-indexing errors when
    # CUDA.allowscalar(false). In the real-basis script, imaginary parts should
    # only arise from roundoff or GPU backend promotion.
    USE_CUDA && sync_device()
    Tc = USE_CUDA ? force_cpu_itensor(T) : T
    x = scalar(Tc)
    if warn_imag && abs(imag(x)) > tol_imag
        @warn "Non-negligible imaginary part in scalar ITensor" imag_part = imag(x)
    end
    return real(x)
end

function build_op_cache(ψ::MPS, opnames; use_gpu::Bool)
    T = scalar_type_of_mps(ψ)
    cache = Dict{Tuple{String,Int},ITensor}()

    @inbounds for i in 1:length(ψ)
        s = siteind(ψ, i)
        for op in opnames
            cache[(String(op), i)] = rotor_op_itensor_typed(String(op), s, T; use_gpu = use_gpu)
        end
    end

    return cache
end

function build_op_product_cache(ψ::MPS, opnames; use_gpu::Bool)
    T = scalar_type_of_mps(ψ)
    cache = Dict{Tuple{String,Int},ITensor}()

    @inbounds for i in 1:length(ψ)
        s = siteind(ψ, i)
        for op in opnames
            opname = String(op)
            cache[(opname, i)] = rotor_op_product_itensor_typed(opname, opname, s, T; use_gpu = use_gpu)
        end
    end

    return cache
end

function local_one_site_expectation_centered(A::ITensor, O::ITensor, s::Index)
    return scalar_real_itensor(dag(prime(A, s)) * O * A)
end

function one_site_expectations_cached!(ψ::MPS, op_cache, opname::String)
    Nloc = length(ψ)
    vals = zeros(Float64, Nloc)

    @inbounds for i in 1:Nloc
        orthogonalize!(ψ, i)
        s = siteind(ψ, i)
        vals[i] = local_one_site_expectation_centered(ψ[i], op_cache[(opname, i)], s)
    end

    return vals
end

function left_operator_transfer(ψ::MPS, i::Int, O::ITensor)
    s = siteind(ψ, i)
    A = ψ[i]
    r = right_linkind(ψ, i)

    # Prime site so bra site contracts through O. Prime only the outgoing right
    # link so the transfer state can propagate to larger site index.
    Bra = dag(prime_inds(prime(A, s), (r,)))
    return Bra * O * A
end

function middle_identity_transfer(ψ::MPS, i::Int)
    A = ψ[i]
    l = left_linkind(ψ, i)
    r = right_linkind(ψ, i)

    # Site contracts locally; both links are primed so the transfer state moves
    # from the left bond to the right bond.
    Bra = dag(prime_inds(A, (l, r)))
    return Bra * A
end

function right_operator_transfer(ψ::MPS, i::Int, O::ITensor)
    s = siteind(ψ, i)
    A = ψ[i]
    l = left_linkind(ψ, i)

    # Prime site so bra site contracts through O. Prime only the incoming left
    # link so the incoming transfer closes. The right link, if present, contracts
    # locally against the ket right link.
    Bra = dag(prime_inds(prime(A, s), (l,)))
    return Bra * O * A
end

function diagonal_product_expectation_centered(ψ::MPS, i::Int, O2::ITensor)
    s = siteind(ψ, i)
    A = ψ[i]
    return local_one_site_expectation_centered(A, O2, s)
end

function two_site_expectation_manual!(ψ::MPS, i::Int, j::Int, op1::String, op2::String; use_gpu::Bool = USE_CUDA)
    # Retained for debugging and PBC fallback. The optimized OBC correlator below
    # does not call this in its inner loop.
    use_gpu = use_gpu && USE_CUDA

    if i == j
        T = scalar_type_of_mps(ψ)
        orthogonalize!(ψ, i)
        O12 = rotor_op_product_itensor_typed(op1, op2, siteind(ψ, i), T; use_gpu = use_gpu)
        return diagonal_product_expectation_centered(ψ, i, O12)
    end

    if i > j
        return two_site_expectation_manual!(ψ, j, i, op2, op1; use_gpu = use_gpu)
    end

    T = scalar_type_of_mps(ψ)
    orthogonalize!(ψ, i)

    Oi = rotor_op_itensor_typed(op1, siteind(ψ, i), T; use_gpu = use_gpu)
    Oj = rotor_op_itensor_typed(op2, siteind(ψ, j), T; use_gpu = use_gpu)

    E = left_operator_transfer(ψ, i, Oi)
    @inbounds for k in (i + 1):(j - 1)
        E *= middle_identity_transfer(ψ, k)
    end
    E *= right_operator_transfer(ψ, j, Oj)

    return scalar_real_itensor(E)
end

function correlation_matrix_safe(ψ::MPS, sites, op1::String, op2::String; use_gpu::Bool = USE_CUDA)
    # Compatibility fallback. It is correct but not fast. The OBC path below no
    # longer builds Cxx/Cyy/Czz matrices.
    use_gpu = use_gpu && USE_CUDA
    Nloc = length(ψ)
    C = zeros(Float64, Nloc, Nloc)
    ψw = use_gpu ? copy(ψ) : force_cpu_mps(ψ)

    @inbounds for i in 1:Nloc
        C[i, i] = two_site_expectation_manual!(ψw, i, i, op1, op2; use_gpu = use_gpu)
        for j in (i + 1):Nloc
            Cij = two_site_expectation_manual!(ψw, i, j, op1, op2; use_gpu = use_gpu)
            C[i, j] = Cij
            if op1 == op2
                C[j, i] = Cij
            else
                C[j, i] = two_site_expectation_manual!(ψw, j, i, op1, op2; use_gpu = use_gpu)
            end
        end
    end

    USE_CUDA && sync_device()
    return C
end

function connected_obc_correlator_sweep(ψ::MPS, sites, ops; N::Int, use_gpu::Bool = USE_CUDA)
    use_gpu = use_gpu && USE_CUDA
    ψw = use_gpu ? copy(ψ) : force_cpu_mps(ψ)

    opnames = ("Nx", "Ny", "Nz")

    println("  building cached local operators on ", use_gpu ? "cuda" : "cpu", "...")
    flush(stdout)
    op_cache = build_op_cache(ψw, opnames; use_gpu = use_gpu)
    op2_cache = build_op_product_cache(ψw, opnames; use_gpu = use_gpu)

    # One-point functions for the connected subtraction. This is only O(N).
    println("  computing one-point functions...")
    flush(stdout)
    mxs = one_site_expectations_cached!(ψw, op_cache, "Nx")
    mys = one_site_expectations_cached!(ψw, op_cache, "Ny")
    mzs = one_site_expectations_cached!(ψw, op_cache, "Nz")

    mx = mean(mxs)
    my = mean(mys)
    mz = mean(mzs)
    disconnected = mx^2 + my^2 + mz^2

    sums = zeros(Float64, N)
    counts = zeros(Int, N)
    @inbounds begin
        counts[1] = N
        for r in 1:(N - 1)
            counts[r + 1] = N - r
        end
    end

    # Sweep-style OBC accumulation:
    #   for fixed i and operator component a, build <... nᵃ_i ...| once,
    #   propagate the transfer tensor rightward, and accumulate directly into
    #   C(r). This avoids full Cxx/Cyy/Czz matrices and avoids repeated operator
    #   uploads. It also removes orthogonalize! from the inner j-loop.
    for op in opnames
        println("  sweeping ", op, "...")
        flush(stdout)

        @inbounds for i in 1:N
            orthogonalize!(ψw, i)

            # r = 0 contribution: <(nᵃ_i)^2>
            sums[1] += diagonal_product_expectation_centered(ψw, i, op2_cache[(op, i)])

            if i == N
                continue
            end

            E = left_operator_transfer(ψw, i, op_cache[(op, i)])

            for j in (i + 1):N
                # Accumulate <nᵃ_i nᵃ_j> before turning site j into an identity
                # transfer for the next separation.
                val = scalar_real_itensor(E * right_operator_transfer(ψw, j, op_cache[(op, j)]))
                sums[j - i + 1] += val

                if j < N
                    E *= middle_identity_transfer(ψw, j)
                end
            end
        end
    end

    C = zeros(Float64, N)
    @inbounds for r in 1:N
        C[r] = sums[r] / counts[r] - disconnected
    end

    USE_CUDA && sync_device()
    return C, (mx = mx, my = my, mz = mz, disconnected = disconnected, counts = counts, correlator_algorithm = "sweep_obc_cached_operators")
end

function connected_pbc_correlator(ψ::MPS, sites, ops; N::Int, use_gpu::Bool = USE_CUDA)
    # PBC correlator path. This is the default when using the PBC Hamiltonian.
    # It currently uses the compatibility matrix path rather than the optimized
    # OBC sweep path.
    use_gpu = use_gpu && USE_CUDA
    @warn "PBC correlator is using the compatibility matrix path, not the optimized OBC sweep path."

    ψw = use_gpu ? copy(ψ) : force_cpu_mps(ψ)
    opnames = ("Nx", "Ny", "Nz")
    op_cache = build_op_cache(ψw, opnames; use_gpu = use_gpu)

    mxs = one_site_expectations_cached!(ψw, op_cache, "Nx")
    mys = one_site_expectations_cached!(ψw, op_cache, "Ny")
    mzs = one_site_expectations_cached!(ψw, op_cache, "Nz")

    mx = mean(mxs)
    my = mean(mys)
    mz = mean(mzs)
    disconnected = mx^2 + my^2 + mz^2

    println("  computing Cxx with fallback matrix path...")
    flush(stdout)
    Cxx = correlation_matrix_safe(ψ, sites, "Nx", "Nx"; use_gpu = use_gpu)

    println("  computing Cyy with fallback matrix path...")
    flush(stdout)
    Cyy = correlation_matrix_safe(ψ, sites, "Ny", "Ny"; use_gpu = use_gpu)

    println("  computing Czz with fallback matrix path...")
    flush(stdout)
    Czz = correlation_matrix_safe(ψ, sites, "Nz", "Nz"; use_gpu = use_gpu)

    Cdot = Cxx + Cyy + Czz
    C = zeros(Float64, N)

    @inbounds begin
        C[1] = mean(Cdot[i, i] for i in 1:N) - disconnected
        for r in 1:(N - 1)
            C[r + 1] = mean(Cdot[i, mod1(i + r, N)] for i in 1:N) - disconnected
        end
    end

    return C, (mx = mx, my = my, mz = mz, disconnected = disconnected, correlator_algorithm = "pbc_matrix_fallback")
end

function connected_obc_correlator(ψ::MPS, sites, ops; N::Int, use_gpu::Bool = USE_CUDA)
    return connected_obc_correlator_sweep(ψ, sites, ops; N = N, use_gpu = use_gpu)
end

function connected_correlator(ψ::MPS, sites, ops; N::Int, open_boundary::Bool = false, use_gpu::Bool = USE_CUDA)
    use_gpu = use_gpu && USE_CUDA
    if open_boundary
        return connected_obc_correlator(ψ, sites, ops; N = N, use_gpu = use_gpu)
    else
        return connected_pbc_correlator(ψ, sites, ops; N = N, use_gpu = use_gpu)
    end
end

function write_correlator_csv(filename::String, rs, C; corr_meta = nothing, corr_runtime_sec = NaN)
    open(filename, "w") do io
        println(io, "# Connected $(open_boundary ? "OBC" : "PBC") correlator")
        println(io, "# g2,$g2")
        println(io, "# N,$N")
        println(io, "# D,$D")
        println(io, "# lmax,$lmax")
        println(io, "# boundary,$boundary")
        println(io, "# corr_runtime_sec,$corr_runtime_sec")
        println(io, "# device,$(USE_CUDA ? :cuda : :cpu)")
        println(io, "# gpu_id,$gpu_id")
        println(io, "# unified_memory,$use_unified_memory")

        if corr_meta !== nothing
            println(io, "# mx,$(corr_meta.mx)")
            println(io, "# my,$(corr_meta.my)")
            println(io, "# mz,$(corr_meta.mz)")
            println(io, "# disconnected,$(corr_meta.disconnected)")
            if hasproperty(corr_meta, :counts)
                println(io, "# counts_are_obc_pair_counts,true")
            end
        end

        println(io, "r,C")

        @inbounds for k in eachindex(rs, C)
            println(io, "$(rs[k]),$(C[k])")
        end
    end

    return filename
end

function save_correlator_plot(rs, C; filename::String)
    eps_floor = 1e-300
    y = max.(abs.(C), eps_floor)

    p = plot(
        rs,
        y;
        yscale = :log10,
        marker = :circle,
        linewidth = 2,
        xlabel = "r",
        ylabel = "|C(r)|",
        title = "Connected $(open_boundary ? "OBC" : "PBC") correlator, g²=$g2, N=$N, lmax=$lmax, D=$D",
        label = "|C(r)|",
        grid = true,
    )

    savefig(p, filename)
    return p
end


# ============================================================
# Operator diagnostics
# ============================================================

function print_operator_checks(lmax::Int)
    ops = rotor_ops(lmax)
    Nx = ops["Nx"]
    Ny = ops["Ny"]
    Nz = ops["Nz"]
    L2 = ops["L2"]
    Id = ops["Id"]

    n2 = Nx * Nx + Ny * Ny + Nz * Nz

    println("Operator checks:")
    @printf("  ||Nx - Nx†|| = %.3e\n", norm(Nx - Nx'))
    @printf("  ||Ny - Ny†|| = %.3e\n", norm(Ny - Ny'))
    @printf("  ||Nz - Nz†|| = %.3e\n", norm(Nz - Nz'))
    @printf("  ||L2 - L2†|| = %.3e\n", norm(L2 - L2'))
    @printf("  ||n² - I||   = %.3e\n", norm(n2 - Id))
    println("  Note: ||n² - I|| is nonzero at finite lmax because of rotor truncation.")
    flush(stdout)
end


# ============================================================
# Main
# ============================================================

println("\n", "="^80)
println("Building rotor operators")
flush(stdout)
print_operator_checks(lmax)

println("\n", "="^80)
println("Ground-state DMRG")
flush(stdout)

gs = solve_ground_state(; g2 = g2, N = N, D = D, lmax = lmax)

println("Ground-state energy        = ", gs.energy)
println("Ground-state energy / site = ", gs.energy_density)
println("Ground-state runtime sec   = ", gs.runtime_sec)
flush(stdout)
print_cuda_memory_status("after ground-state DMRG")

# JLD2 files are kept CPU-resident for portability.
#
# Important GPU-memory cleanup policy:
#   After DMRG finishes, `gs.psi` and `gs.H` live on the GPU when USE_CUDA=true.
#   Calling GC.gc()/CUDA.reclaim() is not enough while those objects remain
#   reachable. Therefore we first copy the result to CPU, save it, drop every
#   reference to the GPU DMRG result, reclaim the CUDA pool, and then re-upload
#   only the CPU ground-state MPS needed by the correlator.
#
# This makes the correlator start with a clean CUDA pool instead of competing
# with the DMRG MPO/MPS and discarded DMRG work buffers for VRAM.
gs_runtime = gs
gs_cpu = USE_CUDA ? merge(gs_runtime, (psi = force_cpu_mps(gs_runtime.psi), H = force_cpu_mpo(gs_runtime.H))) : gs_runtime

# Preserve the original JLD2 variable name `gs` for compatibility with existing loaders.
gs = gs_cpu
@save gs_file gs g2 N D lmax tol maxiter verbosity USE_CUDA device_arg gpu_id use_unified_memory open_boundary boundary run_excited basis_tag
println("Saved ground state to: ", gs_file)
flush(stdout)

# Drop GPU-resident ground-state DMRG objects before correlator allocation.
# Keep only CPU-resident `gs_cpu`.
gs_runtime = nothing
gs = gs_cpu

if USE_CUDA
    println("Cleaning GPU memory after ground-state DMRG and before correlator...")
    flush(stdout)

    # Re-select the same CUDA device used for DMRG before cleanup/re-upload.
    # In Slurm with one visible GPU, this should still be local device id 0.
    CUDA.device!(gpu_id)
    sync_device()
    GC.gc(true)
    CUDA.reclaim()
    sync_device()
    print_cuda_memory_status("after dropping GPU DMRG objects before correlator")
end

println("\n", "="^80)
println("Computing connected $(open_boundary ? "OBC" : "PBC") correlator")
println("correlator device = ", USE_CUDA ? "cuda" : "cpu")
if USE_CUDA
    CUDA.device!(gpu_id)
    println("correlator gpu id = ", gpu_id)
    println("correlator gpu name = ", CUDA.name(CUDA.device()))
end
flush(stdout)

# Re-upload only what the correlator needs, on the same GPU as DMRG.
# Do not keep the DMRG MPO on GPU during the correlator.
corr_psi = USE_CUDA ? to_device(gs_cpu.psi) : gs_cpu.psi
corr_sites = gs_cpu.sites
if USE_CUDA
    sync_device()
    print_cuda_memory_status("after uploading correlator MPS only")
end

C = nothing
corr_meta = nothing
corr_runtime_sec = @elapsed begin
    C, corr_meta = connected_correlator(corr_psi, corr_sites, gs_cpu.ops; N = N, open_boundary = open_boundary, use_gpu = USE_CUDA)
end

# Drop correlator GPU MPS before plotting and optional excited-state work.
corr_psi = nothing
if USE_CUDA
    sync_device()
    GC.gc(true)
    CUDA.reclaim()
    sync_device()
    print_cuda_memory_status("after correlator cleanup")
end

rs = collect(0:(N - 1))
write_correlator_csv(corr_file, rs, C; corr_meta = corr_meta, corr_runtime_sec = corr_runtime_sec)

println("Saved correlator CSV to: ", corr_file)
println("Correlator runtime sec = ", corr_runtime_sec)
flush(stdout)

save_correlator_plot(rs, C; filename = plot_file)
println("Saved correlator plot to: ", plot_file)
flush(stdout)

gap = NaN
ex = nothing

if run_excited
    println("\n", "="^80)
    println("First-excited-state DMRG")
    flush(stdout)

    # The ground-state GPU objects were intentionally freed before the correlator.
    # Rebuild and re-upload the Hamiltonian and CPU ground-state MPS only if the
    # excited-state calculation is requested.
    if USE_CUDA
        CUDA.device!(gpu_id)
        sync_device()
    end
    # IMPORTANT: excited-state DMRG penalty states must share the exact same
    # ITensor site Index IDs as both H and the initial excited-state trial MPS.
    # Do not call o3_nlsm_finite_hamiltonian(...) here, since that creates fresh
    # siteinds with identical tags/dimensions but different internal Index ids.
    sites_ex = [siteind(gs_cpu.psi, i) for i in 1:N]
    os_ex = o3_nlsm_opsum(; g2 = g2, N = N, open_boundary = open_boundary)
    H_ex_cpu = MPO(os_ex, sites_ex)

    H_ex = to_device(H_ex_cpu)
    ψ_ground_ex = to_device(gs_cpu.psi)
    sync_device()
    print_cuda_memory_status("after re-upload for excited-state DMRG")

    ex = solve_first_excited_state(H_ex, sites_ex, ψ_ground_ex; N = N, D = D)

    H_ex = nothing
    ψ_ground_ex = nothing
    H_ex_cpu = nothing
    if USE_CUDA
        sync_device()
        GC.gc(true)
        CUDA.reclaim()
    end

    gap = ex.energy - gs.energy

    println("Excited-state energy        = ", ex.energy)
    println("Excited-state energy / site = ", ex.energy_density)
    println("Gap E1 - E0                 = ", gap)
    println("Overlap |<E0|E1>|           = ", ex.overlap_with_ground)
    println("Excited-state runtime sec   = ", ex.runtime_sec)
    flush(stdout)

    ex_runtime = ex
    ex = USE_CUDA ? merge(ex_runtime, (psi = to_host(ex_runtime.psi),)) : ex_runtime
    @save ex_file ex gap g2 N D lmax tol maxiter verbosity penalty_weight gs_file USE_CUDA device_arg gpu_id use_unified_memory open_boundary boundary run_excited basis_tag
    ex = ex_runtime
    println("Saved first excited state to: ", ex_file)
else
    println("\n", "="^80)
    println("Skipping first-excited-state DMRG. Pass --excited to enable it.")
    flush(stdout)
end

println("\n", "="^80)
println("Summary")
@printf("E0       = %.16e\n", gs.energy)
@printf("E0 / N   = %.16e\n", gs.energy_density)
if run_excited && ex !== nothing
    @printf("E1       = %.16e\n", ex.energy)
    @printf("gap      = %.16e\n", gap)
    @printf("E1 / N   = %.16e\n", ex.energy_density)
else
    println("E1       = not computed")
    println("gap      = not computed")
end
println("boundary = ", boundary)
println("excited  = ", run_excited ? "enabled" : "disabled")
println("ground state file  = ", gs_file)
if run_excited
    println("excited state file = ", ex_file)
end
println("correlator CSV     = ", corr_file)
println("plot file          = ", plot_file)
flush(stdout)
