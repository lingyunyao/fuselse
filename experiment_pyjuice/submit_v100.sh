#!/bin/bash -l
#SBATCH --partition=gpu-v100-32g
#SBATCH --gres=gpu:v100:1
#SBATCH --constraint=skl
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --job-name=pyjuice_q16
#SBATCH --output=outputs/pyjuice_q16_output.log

cd "$(dirname "$0")"
mkdir -p outputs

python3 --version
python3 -c "import sys; print('exec:', sys.executable)"

# pyjuice is not on PyPI; clone from upstream into ./pyjuice/ first:
#   git clone https://github.com/Tractables/pyjuice.git
# This script then installs it in editable mode (no-op if already installed).
python3 -m pip install --user --quiet -e pyjuice/ 2>&1 | tail -5 || true
python3 -c "import pyjuice; print('pyjuice at', pyjuice.__file__)"

export OMP_NUM_THREADS=8
python3 -u run_pyjuice.py
