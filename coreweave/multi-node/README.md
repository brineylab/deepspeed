## Multi-Node Training on SUNK

Scales the [`single-run/`](../single-run) workflow across **2 nodes × 8 GPUs = 16 GPUs** via `accelerate launch`.

Based on the CoreWeave SUNK training tutorial: <https://docs.coreweave.com/products/sunk/tutorials/train-on-sunk>.

### Submit

```bash
sbatch train.sh
```

To run on more nodes, change `#SBATCH --nodes` in [`train.sh`](./train.sh). The accelerate `--num_machines` and `--num_processes` flags read from `$SLURM_NNODES`, so they scale automatically.

### What's different from `single-run/`

#### More `#SBATCH` flags

Multi-node replaces the simple `--gpus=8` with a few new flags:

```diff
+ #SBATCH --nodes=2
+ #SBATCH --ntasks-per-node=1
  #SBATCH --gres=gpu:8
+ #SBATCH --exclusive
```

`--ntasks-per-node=1` tells Slurm to start one process per node — and that one process is `accelerate launch`, which then takes over and spawns the 8 per-GPU workers locally. `--exclusive` reserves the entire node for this job (no co-tenants), which avoids contention on the GPUs and network.

#### Workers need to find each other

When training spans 2 nodes, every GPU process has to know the IP and port of the "main" process (rank 0) so they can connect to it. We compute those in `train.sh` before launching:

```bash
export MASTER_ADDR=$(hostname --ip-address)
export MASTER_PORT=$((10000 + SLURM_JOB_ID % 50000))
```

The sbatch body runs on the first allocated node — which is also the node that ends up running rank 0 — so `hostname` returns the right address. The port is derived from the job ID, so two multi-node jobs running at the same time pick different ports and don't collide.

#### Container directives move from `#SBATCH` to `srun`

In `single-run/`, `#SBATCH --container-*` wraps the whole script inside the container. We can't do that here, because the script itself needs to call `srun` — and `srun` is a Slurm tool that doesn't exist inside the image. So the container directives move to the `srun` line instead, and only the training step runs inside the container.

One side-effect: we set `--container-workdir="$SLURM_SUBMIT_DIR"` so accelerate launches from this folder. Otherwise it would start in the image's default working directory, and the relative paths in `train_config.yaml` (like `./output/`) wouldn't resolve to the right place.

#### `.env` works the same as before

The sbatch body sources `/mnt/home/sburbach/.env` just like single-run does. Slurm's `--export=ALL` (its default) passes the resulting environment to `srun`, and pyxis carries it into the container — so `WANDB_API_KEY`, `TOKENIZERS_PARALLELISM`, and your cache paths all reach the Python process with no extra plumbing. To add a new var, drop it in `.env` (or `export` it in the sbatch body) and you're done.

### Troubleshooting

- **Job hangs at NCCL init.** Workers can't reach `MASTER_ADDR`. Add `export NCCL_SOCKET_IFNAME=eth0` (or whatever `ip a` shows on the compute nodes) before the `srun`.
- **`SLURM_PROCID` is 0 on every node.** The `--ntasks-per-node=1` flag got dropped from the `srun` line. With multiple tasks per node, `SLURM_PROCID` is the global task rank, not the node rank — which breaks the `--machine_rank=$SLURM_PROCID` mapping.
- **Only 8 GPUs visible in W&B.** Accelerate fell back to the YAML's `num_processes`. The CLI override didn't take effect — confirm the `srun` line is intact.
