#!/bin/bash

# --- job ---
#SBATCH --job-name=test_sweep
#SBATCH --partition=all

# --- sweep ---
#SBATCH --array=1-10%5

# --- resources ---
#SBATCH --gpus=8
#SBATCH --cpus-per-task=128
#SBATCH --mem=128G

# --- logs ---
#SBATCH --output=/mnt/home/sburbach/logs/%x_%j.out
#SBATCH --error=/mnt/home/sburbach/logs/%x_%j.err

# --- container ---
#SBATCH --container-name=deeplearning_v2026-04-16
#SBATCH --container-image=brineylab/deeplearning:v2026-04-16
#SBATCH --container-mounts=/mnt/home/sburbach:/mnt/home/sburbach,/mnt/data:/mnt/data,/tmp:/tmp
#SBATCH --container-workdir=/mnt/home/sburbach
#SBATCH --container-env=HOME=/mnt/home/sburbach
#SBATCH --no-container-mount-home

set -euo pipefail

# source env file
source /mnt/home/sburbach/.env

# cd into the directory the job was submitted from
cd "$SLURM_SUBMIT_DIR"

# unique port per array task to avoid collisions across concurrent agents
export MAIN_PROCESS_PORT=$((29500 + (${SLURM_ARRAY_TASK_ID:-0} % 1000)))

wandb agent --count 1 <SWEEP_ID>
