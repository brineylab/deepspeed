# Example Training Sripts

Scripts and configs for distributed model training using HuggingFace Accelerate and DeepSpeed.

## Repo structure

```
training-scripts/
├── accelerate/         # local multi-GPU training (Scripps clusters)
│   ├── esm/
│   └── roberta/
└── coreweave/          # multi-node training on CoreWeave Slurm clusters
    ├── single-run/
    └── wandb-sweep/
```

## accelerate/

Training scripts for local clusters using HuggingFace Accelerate and Deepspeed.

## coreweave/

Job submission scripts for CoreWeave's Slurm clusters. Covers single runs, WandB hyperparameter sweeps, and multi-node training. 
