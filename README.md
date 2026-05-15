# Example Training Sripts

Scripts and configs for distributed model training using HuggingFace Accelerate and DeepSpeed.

## Repo structure

```
training-scripts/
├── accelerate/         # general introduction to accelerate for local training
│   ├── esm/
│   └── roberta/
└── coreweave/          # training on CoreWeave SUNK clusters
    ├── multi-node/
    ├── single-run/
    └── wandb-sweep/
```

## accelerate/

Training scripts for local clusters using HuggingFace Accelerate and Deepspeed.

## coreweave/

Job submission scripts for CoreWeave's SUNK clusters. Covers single training runs, wandb sweeps, and multi-node training. 
