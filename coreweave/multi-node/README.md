## Multi-Node Training on SUNK

This folder scales the model training as in [`single-run/`](../single-run) across **2 nodes × 8 GPUs = 16 GPUs**, launched via `accelerate launch`.

Based on the CoreWeave SUNK training tutorial: <https://docs.coreweave.com/products/sunk/tutorials/train-on-sunk>.

### Job Submission

```bash
sbatch train.sh
```
- To scale to more nodes, change `#SBATCH --nodes` in [`train.sh`](./train.sh) and the matching `num_machines` / `num_processes` in [`accelerate_config.yaml`](./accelerate_config.yaml).

### Key changes vs. `single-run/`

#### 1. SBATCH resource directives

Multi-node adds `--nodes`, `--ntasks-per-node=1`, and `--exclusive` on top of the existing `--gres=gpu:8`. Note that `--gres=gpu:N` specifies GPUs **per node**, not total across the job — so with `--nodes=2`, the job allocates 16 GPUs. The single task-per-node hands control to `accelerate launch`, which spawns the 8 per-GPU processes locally.

```diff
+ #SBATCH --nodes=2
+ #SBATCH --ntasks-per-node=1
  #SBATCH --gres=gpu:8
+ #SBATCH --exclusive
```

`--exclusive` is required by the SUNK tutorial so the job owns whole nodes (avoids NCCL contention with other tenants).

#### 2. Rendezvous (`MASTER_ADDR` / `MASTER_PORT`)

Workers on the second node have to know the IP of the head node, and Slurm only picks the allocation at runtime, so this can't go in any YAML. `train.sh` resolves it before launch:

```bash
export MASTER_ADDR=$(hostname --ip-address)
export MASTER_PORT=$((10000 + SLURM_JOB_ID % 50000))
```

Why this works: the sbatch script body runs on the first allocated node (the "batch host"), and with `--ntasks-per-node=1` that same node receives `SLURM_PROCID=0` in the main `srun` — i.e. it's the `machine_rank=0` node, exactly what we want `MASTER_ADDR` to point at.

Notes:
- `hostname --ip-address` (not `$(hostname)`) — the literal hostname isn't always resolvable across nodes on SUNK; the IP always is.
- `MASTER_PORT` is derived from `SLURM_JOB_ID` so concurrent multi-node jobs don't collide on the same port.
- We deliberately avoid `scontrol show hostnames` here — it's a Slurm host binary and isn't installed inside the `brineylab/deeplearning` container.

#### 3. Launcher: `srun` wraps `accelerate launch`

Single-node calls `accelerate launch` directly. Multi-node has to launch one `accelerate` process **per node**, so we wrap it in `srun`. The Slurm-derived values (`machine_rank`, `main_process_ip`, `main_process_port`) are passed as CLI flags, which override anything in the YAML:

```bash
srun --nodes=$SLURM_NNODES --ntasks-per-node=1 \
  accelerate launch \
    --config_file ./accelerate_config.yaml \
    --num_machines $SLURM_NNODES \
    --num_processes $((SLURM_NNODES * 8)) \
    --machine_rank $SLURM_PROCID \
    --main_process_ip "$MASTER_ADDR" \
    --main_process_port $MASTER_PORT \
    pretraining.py --config_file ./train_config.yaml
```

Because `--ntasks-per-node=1`, `$SLURM_PROCID` equals the node index (0 on the head, 1 on the worker) — exactly what Accelerate expects for `--machine_rank`.

#### 4. `accelerate_config.yaml`

Only the machine/process counts change. `rdzv_backend: static` is kept (it pairs with the explicit `--main_process_ip` / `--main_process_port`).

```diff
- num_machines: 1
- num_processes: 4
+ num_machines: 2
+ num_processes: 16
```

#### 5. `pretraining.py` and `train_config.yaml`

No code changes — HF `Trainer` under Accelerate works the same in multi-node as single-node. The only diff in `train_config.yaml` is `wandb_group: "multinode_testing"` (vs. `"initial_testing"`) so the W&B runs sort separately.

### Troubleshooting

- **Job hangs at NCCL init** — the worker can't reach `MASTER_ADDR`. Add `export NCCL_SOCKET_IFNAME=eth0` (or whatever `ip a` shows on the compute nodes) to `train.sh` before the `srun`.
- **`SLURM_PROCID` is 0 on both nodes** — check that `--ntasks-per-node=1` survived into the `srun` line; with multiple tasks per node, `SLURM_PROCID` is the global task rank, not the node rank.
- **Only 8 GPUs visible in W&B** — `--num_processes` on the CLI wasn't picked up, so Accelerate fell back to the YAML's value. Confirm the `srun` line is intact.
