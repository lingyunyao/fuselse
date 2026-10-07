#!/bin/bash -l
#SBATCH --partition=gpu-h100-80g
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:30:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=ncu_h100
#SBATCH --output=ncu_h100_output.log

# Use ncu via absolute path; do NOT load cuda/12.6.2 module here, since
# loading it overrides the python env's nvcc with one that conflicts
# with the host gcc-14, breaking cpp_extension JIT compilation.
NCU=$(command -v ncu || echo ncu)

cd "$(dirname "$0")"

$NCU --replay-mode application \
    --target-processes all \
    --csv \
    --nvtx \
    --kernel-name regex:"block_logsumexp_fp_kernel|block_logsumexp_float_kernel|reduce_kernel|vectorized_elementwise_kernel|elementwise_kernel" \
    --metrics dram__bytes.sum,gpu__time_duration.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active \
    --log-file ncu_h100.csv \
    python3 -u profile_kernels_ncu.py
