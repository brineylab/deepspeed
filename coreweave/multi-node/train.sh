#!/bin/bash

# --- job ---
#SBATCH --job-name=test_multinode
#SBATCH --partition=all

# --- nodes & resources ---
# 2 nodes, with one task per node. 8 gpus, 128 cpus, and all of the memory per node. 
# Also requires a whole-node lock, meaning the node won't recieve any other jobs.
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --mem=0
#SBATCH --exclusive

# --- logs ---
#SBATCH --output=/mnt/home/sburbach/logs/%x_%j.out
#SBATCH --error=/mnt/home/sburbach/logs/%x_%j.err

# Note: no container is used for the initial job. Containers are specified below,
# so that the training job runs inside a container.

set -euo pipefail

# source env file
source /mnt/home/sburbach/.env

# distributed rendezvous (sbatch body runs on the rank-0 node)
export MASTER_ADDR=$(hostname --ip-address)
export MASTER_PORT=$((10000 + SLURM_JOB_ID % 50000))
export NCCL_DEBUG=INFO
export OMP_NUM_THREADS=1

srun --nodes=$SLURM_NNODES --ntasks-per-node=1 \
  --export=ALL \
  --container-image=/mnt/data/containers/deeplearning_v2026-04-16.sqsh \
  --container-mounts=/mnt/home/sburbach:/mnt/home/sburbach,/mnt/data:/mnt/data,/tmp:/tmp \
  --container-workdir="$SLURM_SUBMIT_DIR" \
  --container-env=HOME=/mnt/home/sburbach \
  --no-container-mount-home \
  accelerate launch \
    --config_file ./accelerate_config.yaml \
    --num_machines $SLURM_NNODES \
    --num_processes $((SLURM_NNODES * 8)) \
    --machine_rank $SLURM_PROCID \
    --main_process_ip "$MASTER_ADDR" \
    --main_process_port $MASTER_PORT \
    pretraining.py --config_file ./train_config.yaml
