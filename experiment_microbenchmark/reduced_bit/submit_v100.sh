#!/bin/bash -l
#SBATCH --partition=gpu-v100-32g
#SBATCH --gres=gpu:v100:1
#SBATCH --constraint=skl
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=fp16_bench
#SBATCH --output=outputs/run.log

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

cd "$(dirname "$0")"
mkdir -p outputs
python3 -u run_bench.py
