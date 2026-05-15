#!/bin/bash

# --- job ---
#SBATCH --job-name=test_coreweave
#SBATCH --partition=all

# --- resources ---
#SBATCH --gpus=8
#SBATCH --cpus-per-task=128
#SBATCH --mem=128G

# --- logs ---
#SBATCH --output=/mnt/home/sburbach/logs/%x_%A_%a.out
#SBATCH --error=/mnt/home/sburbach/logs/%x_%A_%a.err

# --- container ---
#SBATCH --container-image=/mnt/data/containers/deeplearning_v2026-04-16.sqsh
#SBATCH --container-mounts=/mnt/home/sburbach:/mnt/home/sburbach,/mnt/data:/mnt/data,/tmp:/tmp
#SBATCH --container-workdir=/mnt/home/sburbach
#SBATCH --container-env=HOME=/mnt/home/sburbach
#SBATCH --no-container-mount-home

set -euo pipefail

# source env file
source /mnt/home/sburbach/.env

# cd into the directory the job was submitted from
cd "$SLURM_SUBMIT_DIR"

# unique port to avoid collisions if multiple jobs run concurrently
export MAIN_PROCESS_PORT=$((29500 + (${SLURM_ARRAY_TASK_ID:-0} % 1000)))

accelerate launch --main_process_port $MAIN_PROCESS_PORT --config_file ./accelerate_config.yaml pretraining.py --config_file ./train_config.yaml
