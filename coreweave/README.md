## CoreWeave SUNK Training

Job submission scripts and configs for training on CoreWeave's SUNK (Slurm-on-Kubernetes) clusters. Organized by run type:

- [`single-run/`](./single-run) — single-node training (1 × 8 GPUs)
- [`wandb-sweep/`](./wandb-sweep) — W&B hyperparameter sweeps as Slurm array jobs
- [`multi-node/`](./multi-node) — multi-node distributed training across 2+ nodes (WIP - do not use, this is not working!!)
- `.env` — shared env vars (HuggingFace, W&B, NVIDIA cache paths, `WANDB_API_KEY`)

### First-time login & setup

After your first login to the head node, set up micromamba:

> **Note:** the head nodes also have miniconda installed — **please don't use it.** Scripps doesn't allow conda for licensing / data reasons. Use micromamba instead.

```bash
micromamba shell init
source ~/.bashrc
```

Activate the base environment and install python:

```bash
micromamba activate base
micromamba install python
```

Install `wandb` (needed on the head node to create sweeps). In general, **avoid installing large packages on the head node** — training dependencies live inside the container image.

```bash
pip install wandb
```

### Storage

Every training container mounts three storage destinations:

| Path | Scope | Persistence |  
| --- | --- | --- |  
| `/mnt/home/<user>` | per-user, shared across all nodes (DFS) | persistent |  
| `/mnt/data` | shared across all nodes (DFS) | persistent |  
| `/tmp` | local to each compute node | **ephemeral** - data needs to be transfered at the end of each job |  

These get wired into every training container via the `--container-mounts` flag on `srun` (or `#SBATCH --container-mounts` for single-node jobs):

```bash
--container-mounts=/mnt/home/<user>:/mnt/home/<user>,/mnt/data:/mnt/data,/tmp:/tmp
```

The [`.env`](./.env) file deliberately points all cache paths (`HF_HOME`, `WANDB_*`, `CUDA_CACHE_PATH`, `XDG_CACHE_HOME`, etc.) to `/tmp/*` so they hit local SSD rather than round-tripping through the DFS — important for tokenization throughput.


### Data transfer (object storage)

CoreWeave provides S3-compatible object storage. Two buckets are provisioned for the lab:

| Bucket name | Storage limit | Location |
| --- | --- | --- |
| `brineylab-eu` | 273 TiB | EU-SOUTH-04A (B200 cluster) |
| `brineylab-us-east` | 100 TiB | US-EAST-13A (RTX6000 cluster) |

Use `s5cmd` (CoreWeave's fork) for transfers.

#### Install s5cmd

```bash
wget https://github.com/coreweave/s5cmd/releases/download/v2.3.0-acb67716/s5cmd_2.3.0-acb67716_linux_amd64.deb
sudo apt install ./s5cmd_2.3.0-acb67716_linux_amd64.deb
```

#### Configure credentials

Generate access keys in the CoreWeave console (Administration -> Object storage access keys -> Create key), then export:

```bash
export AWS_ACCESS_KEY_ID=your_access_key_id
export AWS_SECRET_ACCESS_KEY=your_secret_access_key
```

#### Pick the right endpoint

Which endpoint to use depends on **where you're running `s5cmd` from**:

| Running from | Endpoint | When to use |
| --- | --- | --- |
| Inside CoreWeave regions (Slurm/compute nodes) | `http://cwlota.com` | Internal, high-speed. Use for all transfers from Slurm/compute nodes. |
| Our servers (external) | `https://cwobject.com` | External endpoint. Use when pushing/pulling from our local storage. |

#### Examples

List files in object storage:

```bash
s5cmd --endpoint-url https://cwobject.com ls s3://brineylab-eu
```

Copy from our servers → object storage (external, so use `cwobject`):

```bash
s5cmd --endpoint-url https://cwobject.com cp ./test-training/ s3://brineylab-eu/username/test-training/
```

Copy from a CoreWeave node → object storage (internal, so use `cwlota`):

```bash
s5cmd --endpoint-url http://cwlota.com cp ./test-training/ s3://brineylab-eu/username/test-training/
```
