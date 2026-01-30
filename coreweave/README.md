## Running Hyperparameter Sweeps with Weights & Biases

### Steps

#### 1. Initialize the W&B Sweep

Create a new sweep in your W&B project:

```bash
wandb sweep --project project_id sweep.yaml
```

This command will output a sweep ID that looks like: `username/project_id/sweep_id`

#### 2. Configure the Jobs Script

Open [`job.sh`](./job.sh) and replace the placeholder `<SWEEP_ID>` with the sweep ID from step 1:

```diff
- wandb agent <SWEEP_ID>

+ wandb agent username/deepspeed_coreweave/abc123xyz
```

Modify the mount path to point to the folder where your code is located.

```diff
- #SBATCH --container-mounts=/path/to/folder:/workspace

+ #SBATCH --container-mounts=/mnt/home/user/code_folder:/workspace
```

Increase or decrease the number jobs to run by modifying the `--array` parameter:

```bash
#SBATCH --array=0-3%4
                | | |
                | | └─ Number of concurrent jobs
                | └─ Total number of jobs
                └─ Start index
```

```diff
- #SBATCH --array=0-3%4
+ #SBATCH --array=0-7%4
```

#### 3. Launch Sweep Agents

Start the sweep agents by running:

```bash
bash jobs.sh
```
