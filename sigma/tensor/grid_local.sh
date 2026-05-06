set -euo pipefail

# Usage:
#   ./run_grid_local.sh 1.0
#
# Runs:
#   julia --project=julia_env run_one.jl <g2> <lmax> <D>
#
# serially for all lmax and D values below.

g2="${1:-1.0}"

lmaxs=(2 3 4 5)
Ds=(100 200 300 400)

export JULIA_NUM_THREADS=4
export JULIA_NUM_GC_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OMP_NUM_THREADS=4
export JULIA_PKG_PRECOMPILE_AUTO=1

mkdir -p logs

echo "Running local serial grid"
echo "g2    = $g2"
echo "lmaxs = ${lmaxs[*]}"
echo "Ds    = ${Ds[*]}"
echo

for lmax in "${lmaxs[@]}"; do
  for D in "${Ds[@]}"; do
    log="logs/run_g2_${g2}_lmax_${lmax}_D_${D}.log"

    echo "============================================================"
    echo "Starting g2=$g2 lmax=$lmax D=$D"
    echo "Log: $log"
    echo "Time: $(date)"
    echo "============================================================"

    julia run_one.jl "$g2" "$lmax" "$D" 2>&1 | tee "$log"

    echo "Finished g2=$g2 lmax=$lmax D=$D at $(date)"
    echo
  done
done

echo "All runs finished."
