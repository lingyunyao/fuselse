#!/bin/bash -l
#SBATCH --partition=gpu-v100-32g
#SBATCH --gres=gpu:v100:1
#SBATCH --time=00:15:00
#SBATCH --mem=6G
#SBATCH --cpus-per-task=1
#SBATCH --job-name=ncu_v100
#SBATCH --output=ncu_v100_output.log

# Use --replay-mode application: re-runs the whole program once per metric
# set, instead of replaying each individual kernel.  Slower but works under
# more restrictive hardware-counter permissions (V100/A100 nodes here).
NCU=$(command -v ncu || echo ncu)

cd "$(dirname "$0")"

$NCU --replay-mode application \
    --profile-from-start no \
    --target-processes all \
    --csv \
    --nvtx \
    --kernel-name regex:"block_logsumexp_fp_kernel|block_logsumexp_float_kernel|reduce_kernel|vectorized_elementwise_kernel|elementwise_kernel" \
    --metrics dram__bytes.sum,gpu__time_duration.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active \
    --log-file ncu_v100.csv \
    python3 -u profile_kernels_ncu.py
