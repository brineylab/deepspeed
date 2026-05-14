#!/bin/bash
#SBATCH --job-name=test_coreweave
#SBATCH --partition=all
#SBATCH --gpus=8
#SBATCH --cpus-per-task=128
#SBATCH --mem=128G
#SBATCH --output=/mnt/home/sburbach/logs/%x_%j.out
#SBATCH --error=/mnt/home/sburbach/logs/%x_%j.err

#SBATCH --container-name=deeplearning_v2026-04-16
#SBATCH --container-image=brineylab/deeplearning:v2026-04-16
#SBATCH --container-mounts=/mnt/home/sburbach:/mnt/home/sburbach,/mnt/data:/mnt/data
#SBATCH --container-workdir=/mnt/home/sburbach
#SBATCH --container-env=HOME=/mnt/home/sburbach
#SBATCH --no-container-mount-home

set -euo pipefail

# cd into the directory the job was submitted from
cd "$SLURM_SUBMIT_DIR"

export MAIN_PROCESS_PORT=$((29500 + (${SLURM_ARRAY_TASK_ID:-0} % 1000)))

accelerate launch --main_process_port $MAIN_PROCESS_PORT --config_file ./accelerate_config.yaml pretraining.py --config_file ./train_config.yaml
