#!/bin/bash

# setup
accelerate config 
wandb login
export WANDB_JOB_TYPE=""

# run scripts
accelerate launch robertaconfig-train.py --train_config train-config_BALM-paired.yaml

# end of scripts
echo "Scripts ran!"