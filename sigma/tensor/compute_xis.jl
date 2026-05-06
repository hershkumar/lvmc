using LinearAlgebra
using Printf
using Statistics
using TensorKit
using MPSKit
using WignerSymbols
using JLD2

# ============================================================
# Command-line arguments
#
# Usage:
#
#   julia compute_xis_for_directory_threaded.jl DATA_DIR [Lboxes_csv] [--recompute]
#
# Examples:
#
#   julia compute_xis_for_directory_threaded.jl data
#   julia compute_xis_for_directory_threaded.jl data 10,20,40,80,160
#   julia compute_xis_for_directory_threaded.jl data 10,20,40,80,160 --recompute
#   julia compute_xis_for_directory_threaded.jl data --recompute
# ============================================================

if length(ARGS) < 1
    println("Usage: julia compute_xis_for_directory_threaded.jl DATA_DIR [Lboxes_csv] [--recompute]")
    println("Example: julia compute_xis_for_directory_threaded.jl data")
    println("Example: julia compute_xis_for_directory_threaded.jl data 10,20,40,80,160")
    println("Example: julia compute_xis_for_directory_threaded.jl data 10,20,40,80,160 --recompute")
    println("Example: julia compute_xis_for_directory_threaded.jl data --recompute")
    exit(1)
end

const recompute = "--recompute" in ARGS

# Positional args are all non-flag args.
const positional_args = [arg for arg in ARGS if !startswith(arg, "--")]

if length(positional_args) < 1
    error("DATA_DIR is required.")
end

const data_dir = positional_args[1]

const Lboxes =
    if length(positional_args) >= 2
        parse.(Int, split(positional_args[2], ","))
    else
        [10, 20, 40, 80, 160]
    end

if !isdir(data_dir)
    error("Directory does not exist: $data_dir")
end

# Avoid nested BLAS oversubscription.
BLAS.set_num_threads(1)

println("Data directory: ", data_dir)
println("Lboxes: ", Lboxes)
println("recompute existing xis: ", recompute)
println("Julia threads: ", Threads.nthreads())
println("BLAS threads: ", BLAS.get_num_threads())
flush(stdout)

# ============================================================
# Helpers
# ============================================================

function safe_real(x; tol_imag = 1e-8)
    if abs(imag(x)) > tol_imag
        @warn "Non-negligible imaginary part" imag_part = imag(x)
    end
    return real(x)
end


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


function compute_xi_scan(ψ, envs, ops, Lboxes; prefix = "")
    rows = NamedTuple[]

    for Lbox in Lboxes
        row = try
            compute_xi_for_box(ψ, envs, ops; Lbox = Lbox)
        catch err
            @warn "xi computation failed" prefix = prefix Lbox = Lbox exception = (err, catch_backtrace())
            (
                Lbox = Lbox,
                xi = NaN,
                S0 = NaN,
                Skmin = NaN,
                runtime_sec = NaN,
            )
        end

        push!(rows, row)

        @printf(
            "%s Lbox = %-8d xi = %-18.10e S0 = %-18.10e Sk = %-18.10e time = %.3f sec\n",
            prefix,
            row.Lbox,
            row.xi,
            row.S0,
            row.Skmin,
            row.runtime_sec,
        )

        flush(stdout)
        GC.gc()
    end

    return rows
end


function rows_to_xis_dict(xi_rows)
    xis = Dict{Int, NamedTuple}()

    for row in xi_rows
        xis[row.Lbox] = (
            xi = row.xi,
            S0 = row.S0,
            Skmin = row.Skmin,
            runtime_sec = row.runtime_sec,
        )
    end

    return xis
end


function augment_res_with_xis(res, xis::Dict{Int, NamedTuple})
    Lvals = sort(collect(keys(xis)))
    Lmax = maximum(Lvals)

    return merge(
        res,
        (
            xis = xis,
            xi_Lboxes = Lvals,
            xi_latest = xis[Lmax].xi,
            xi_latest_Lbox = Lmax,
            xi_latest_S0 = xis[Lmax].S0,
            xi_latest_Skmin = xis[Lmax].Skmin,
        ),
    )
end


function existing_xis_Lboxes(res)
    if !hasproperty(res, :xis)
        return Int[]
    end

    try
        return sort(collect(keys(res.xis)))
    catch
        return Int[]
    end
end


function process_file(path::String; Lboxes, recompute::Bool = false)
    tid = Threads.threadid()
    prefix = "[thread $tid]"

    println()
    println(prefix, " ", "="^80)
    println(prefix, " Processing: ", path)
    flush(stdout)

    data = try
        JLD2.load(path)
    catch err
        @warn "Could not load file; skipping" path = path exception = (err, catch_backtrace())
        return (
            path = path,
            status = "load_failed",
            message = sprint(showerror, err),
        )
    end

    if !haskey(data, "res")
        @warn "File does not contain top-level key `res`; skipping" path = path keys = collect(keys(data))
        return (
            path = path,
            status = "missing_res",
            message = "No top-level res key",
        )
    end

    res = data["res"]

    if hasproperty(res, :xis) && !recompute
        existing_Lboxes = existing_xis_Lboxes(res)

        println(prefix, " Existing res.xis found; skipping. Existing Lboxes = ", existing_Lboxes)

        return (
            path = path,
            status = "already_has_xis",
            message = "Skipped existing xis",
        )
    end

    for field in [:psi, :envs, :ops]
        if !hasproperty(res, field)
            @warn "res is missing required field; skipping" path = path field = field
            return (
                path = path,
                status = "missing_field",
                message = "Missing field $field",
            )
        end
    end

    ψ = res.psi
    envs = res.envs
    ops = res.ops

    if recompute && hasproperty(res, :xis)
        println(prefix, " Recomputing xi scan; existing res.xis will be overwritten.")
    else
        println(prefix, " Computing xi scan...")
    end

    flush(stdout)

    xi_rows = compute_xi_scan(ψ, envs, ops, Lboxes; prefix = prefix)
    xis = rows_to_xis_dict(xi_rows)

    res_augmented = augment_res_with_xis(res, xis)

    backup_path = path * ".backup_before_xis"

    try
        if !isfile(backup_path)
            cp(path, backup_path; force = false)
            println(prefix, " Backup written to: ", backup_path)
        else
            println(prefix, " Backup already exists: ", backup_path)
        end

        # Each thread writes a distinct file, so this is safe as long as
        # the input file list has no duplicates.
        jldsave(path; res = res_augmented)

        println(prefix, " Updated file with res.xis: ", path)
        println(prefix, " Stored Lboxes: ", sort(collect(keys(xis))))
        println(prefix, " Latest xi: ", res_augmented.xi_latest)
        flush(stdout)

        return (
            path = path,
            status = recompute ? "recomputed" : "ok",
            message = recompute ? "Recomputed" : "Updated",
        )
    catch err
        @warn "Failed while saving augmented file" path = path exception = (err, catch_backtrace())
        return (
            path = path,
            status = "save_failed",
            message = sprint(showerror, err),
        )
    end
end


# ============================================================
# Main
# ============================================================

files = sort(filter(f -> endswith(f, ".jld2"), readdir(data_dir; join = true)))

# Do not process backups.
files = filter(f -> !occursin(".backup", f), files)

# Remove duplicates defensively.
files = unique(files)

println("Found $(length(files)) .jld2 files.")
flush(stdout)

statuses = Vector{NamedTuple}(undef, length(files))

Threads.@threads for i in eachindex(files)
    file = files[i]

    st = process_file(
        file;
        Lboxes = Lboxes,
        recompute = recompute,
    )

    statuses[i] = st

    GC.gc()
end

println()
println("="^100)
println("Summary")
println(rpad("status", 20), "count")
println("-"^40)

for status in sort(unique(s.status for s in statuses))
    count = sum(s.status == status for s in statuses)
    println(rpad(status, 20), count)
end

# Save status log serially after threaded work is done.
status_csv = joinpath(data_dir, "xi_augmentation_status.csv")

open(status_csv, "w") do io
    println(io, "path,status,message")

    for s in statuses
        path_str = replace(s.path, "\"" => "\"\"")
        msg_str = replace(s.message, "\"" => "\"\"")
        println(io, "\"$path_str\",$(s.status),\"$msg_str\"")
    end
end

println("Saved status log to: ", status_csv)