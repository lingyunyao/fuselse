#!/bin/bash -l
#SBATCH --partition=gpu-v100-32g
#SBATCH --gres=gpu:v100:1
#SBATCH --constraint=skl
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=micro_bench
#SBATCH --output=outputs/micro_v100_output.log

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

# IntLSE CPU kernel lives in ../intlse/.  Build if not present.
INTLSE_CPU=../../intlse/cpu
if [ ! -f "$INTLSE_CPU/lse_kernel_simd.so" ]; then
    bash "$INTLSE_CPU/build.sh"
fi

cd "$(dirname "$0")"
mkdir -p outputs
python3 -u run_microbenchmark.py
