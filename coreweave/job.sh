#!/bin/bash
#SBATCH --job-name=inf_training_real
#SBATCH --partition=all
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --output=./logs/%x_%A_%a.out
#SBATCH --error=./logs/%x_%A_%a.err
#SBATCH --array=1-10%5

#SBATCH --container-image=gp201/meh:latest
#SBATCH --container-mounts=/path/to/folder:/workspace
#SBATCH --container-workdir=/workspace

set -euo pipefail

export WANDB_API_KEY="***************"
export MAIN_PROCESS_PORT=$((29500 + (${SLURM_ARRAY_TASK_ID:-0} % 1000)))

wandb agent --count 1 <SWEEP_ID>
