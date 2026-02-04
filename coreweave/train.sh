#!/bin/bash
#SBATCH --job-name=inf_training_real
#SBATCH --partition=all
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --output=./logs/%x_%j.out
#SBATCH --error=./logs/%x_%j.err

#SBATCH --container-image=gp201/meh:26.01-py3
#SBATCH --container-mounts=/path/to/folder:/workspace
#SBATCH --container-workdir=/workspace

set -euo pipefail

export WANDB_API_KEY="***************"
export MAIN_PROCESS_PORT=$((29500 + (${SLURM_ARRAY_TASK_ID:-0} % 1000)))

accelerate launch --main_process_port $MAIN_PROCESS_PORT --config_file ./default_config.yaml pretraining.py --config_file pretrain_real_org.yaml
