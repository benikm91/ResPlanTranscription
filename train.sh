#!/bin/bash
#SBATCH --time=4:00:00
#SBATCH --job-name=ResPlan
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --account=cai_cv
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=72G
#SBATCH --output=/cluster/home/%u/.logs/slurm/%j/ResPlan_%j.out
#SBATCH --error=/cluster/home/%u/.logs/slurm/%j/ResPlan_%j.err

VENV_BASE_DIR=/raid/persistent_scratch/$(whoami)/venvs
ENV_NAME=ResPlan-py3.13.2

VENV_PATH="$VENV_BASE_DIR/$ENV_NAME"

module load python/3.13.2
module load slurm
unset PIP_TARGET
unset PYTHONPATH

if [ -d "$VENV_PATH" ]; then
    echo "Virtual environment ($ENV_NAME) found. Activating..."
    source "$VENV_PATH/bin/activate"
    python -m pip install --quiet -r requirements.txt
else
    echo "Virtual environment ($ENV_NAME) not found. Creating..."
    python3 -m venv $VENV_PATH
    source "$VENV_PATH/bin/activate"
    python -m pip install --quiet -r requirements.txt
fi

python -u train.py "$@"