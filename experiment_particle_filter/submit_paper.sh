#!/bin/bash -l
#SBATCH --partition=gpu-v100-32g
#SBATCH --gres=gpu:v100:1
#SBATCH --constraint=skl
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --job-name=paper_pf
#SBATCH --output=outputs/paper_output.log


export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# IntLSE kernel lives in ../intlse/. Rebuild if .so is missing.
INTLSE_CPU=../intlse/cpu
if [ ! -f "$INTLSE_CPU/lse_kernel_simd.so" ]; then
    bash "$INTLSE_CPU/build.sh"
fi

cd "$(dirname "$0")"
mkdir -p outputs
python3 -u run_paper_experiment.py
