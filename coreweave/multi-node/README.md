## Multi-Node Training on SUNK

Scales the [`single-run/`](../single-run) workflow across **2 nodes × 8 GPUs = 16 GPUs** via `accelerate launch`.

Based on:
- CoreWeave SUNK training tutorial: <https://docs.coreweave.com/products/sunk/tutorials/train-on-sunk>
- HuggingFace's multi-node SLURM example: <https://github.com/huggingface/accelerate/blob/main/examples/slurm/submit_multinode.sh>

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

#### Tokenization runs once, not 16 times

Every rank executes `pretraining.py` top-to-bottom, so without protection all 16 ranks would simultaneously call `dataset.map()` and race on the HuggingFace datasets cache. The script wraps the tokenization call in `training_args.main_process_first(local=False, ...)`, which makes rank 0 do the work first and the other 15 ranks block until the cache is populated.

The `local=False` is the multi-node-specific tweak: with a shared NFS cache directory across nodes, only **global** rank 0 needs to tokenize. The default (`local=True`) would have one rank-0 per node tokenize redundantly — fine on single-node, wasteful here.

### Verify it's actually multi-node

The fastest sanity check is the W&B run config: a working multi-node job produces **one** W&B run with `world_size=16`. If you see **two parallel runs** each with `world_size=8`, the nodes formed independent worlds and never actually communicated. You can also grep the slurm log:

```bash
grep -E 'Num (machines|processes)' /mnt/home/sburbach/logs/test_multinode_*.out
```

Working: `Num machines: 2  Num processes: 16` (logged once). Broken: `Num machines: 1  Num processes: 8` logged twice (once per node).

### Troubleshooting

- **Two W&B runs / `world_size=8` instead of 16.** The nodes formed independent worlds instead of joining one. Check that `accelerate_config.yaml` has `rdzv_backend: c10d` (not `static`) — with static rendezvous, you'd need an explicit per-node `--machine_rank`, which requires deferring `$SLURM_PROCID` expansion via a `bash -c` wrapper. c10d sidesteps that whole issue by auto-assigning ranks dynamically.
- **Job hangs at NCCL init.** Workers can't reach `MASTER_ADDR`. Add `export NCCL_SOCKET_IFNAME=eth0` (or whatever `ip a` shows on the compute nodes) before the `srun`.
- **Only 8 GPUs visible in W&B (single run).** Accelerate fell back to the YAML's `num_processes` — the CLI override didn't take effect. Confirm the `srun` line is intact.
