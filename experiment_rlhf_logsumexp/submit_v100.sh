#!/bin/bash -l
#SBATCH --partition=gpu-v100-32g
#SBATCH --gres=gpu:v100:1
#SBATCH --time=00:45:00
#SBATCH --mem=32G
#SBATCH --job-name=rlhf_v100
#SBATCH --output=outputs/rlhf_v100_output.log

cd "$(dirname "$0")"
mkdir -p outputs

# The compute node's Python may differ from the login node's. Print
# the version and path so the log records which interpreter ran.
python3 --version
python3 -c "import sys; print('exec:', sys.executable)"

# TRL is only installed in the Python 3.9 user-site on the login node;
# if the compute node uses 3.11/3.12 it won't see it. Install it into
# whatever USER_SITE this Python points at (no-op if already present).
# Version is pinned exactly so the measurement is reproducible - the
# paper numbers were collected against trl 1.0.0.
python3 -m pip install --user --quiet 'trl==1.0.0' 2>&1 | tail -5 || true
python3 -c "import trl; print('trl', trl.__version__, 'at', trl.__file__)"

export OMP_NUM_THREADS=8
python3 -u run_rlhf.py
