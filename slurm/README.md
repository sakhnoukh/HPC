# Slurm Batch Scripts

Job submission scripts for running experiments on HPC clusters.

## Prerequisites

Before submitting any jobs, you **must** edit each `.sbatch` file to:

1. **Set your account:** Replace `YOUR_ACCOUNT` with your Slurm account
2. **Set partition:** Replace `gpu` with your cluster's GPU partition name
3. **Set network interface:** Replace `ib0` with your cluster's network interface (find with `ip addr show`)

## Available Scripts

### 1. `ddp_baseline.sbatch`
**Purpose:** Baseline training on single node with 4 GPUs  
**Usage:**
```bash
sbatch slurm/ddp_baseline.sbatch
```
**Configuration:**
- Nodes: 1
- GPUs: 4
- Time: 1 hour
- Epochs: 10

### 2. `ddp_strong_scaling.sbatch`
**Purpose:** Strong scaling experiment (fixed global batch size)  
**Usage:**
```bash
# Edit --nodes and --ntasks-per-node for different GPU counts
sbatch slurm/ddp_strong_scaling.sbatch
```
**Configuration:**
- Global batch: 512 (fixed)
- Batch per GPU: 512 / num_GPUs
- Test: 1, 2, 4, 8 GPUs

### 3. `ddp_weak_scaling.sbatch`
**Purpose:** Weak scaling experiment (fixed per-GPU batch size)  
**Usage:**
```bash
# Edit --nodes and --ntasks-per-node for different GPU counts
sbatch slurm/ddp_weak_scaling.sbatch
```
**Configuration:**
- Batch per GPU: 128 (fixed)
- Global batch: 128 × num_GPUs
- Test: 1, 2, 4, 8 GPUs

### 4. `ddp_sensitivity.sbatch`
**Purpose:** Hyperparameter sensitivity sweep  
**Usage:**
```bash
sbatch slurm/ddp_sensitivity.sbatch
```
**Configuration:**
- Job array: 6 tasks
- Tests: batch_size ∈ {64, 128, 256} × precision ∈ {fp32, bf16}

### 5. `profile_gpu_nsys.sbatch`
**Purpose:** Profile with Nsight Systems (timeline)  
**Usage:**
```bash
sbatch slurm/profile_gpu_nsys.sbatch
```
**Output:** `results/logs/nsys_JOBID.nsys-rep`  
**View with:** Nsight Systems GUI

### 6. `profile_gpu_ncu.sbatch`
**Purpose:** Profile with Nsight Compute (kernel details)  
**Usage:**
```bash
sbatch slurm/profile_gpu_ncu.sbatch
```
**Output:** `results/logs/ncu_JOBID.ncu-rep`  
**View with:** Nsight Compute GUI

## Common Parameters to Modify

### Resource Allocation
```bash
#SBATCH --nodes=N              # Number of nodes
#SBATCH --ntasks-per-node=N    # Tasks per node (= GPUs per node)
#SBATCH --gpus-per-node=N      # GPUs per node
#SBATCH --cpus-per-task=N      # CPU cores per task
#SBATCH --mem=XG               # Memory per node
#SBATCH --time=HH:MM:SS        # Wall time
```

### Training Hyperparameters
In the script body:
```bash
EPOCHS=10          # Number of epochs
BATCH_SIZE=128     # Batch size per GPU
LR=0.1             # Learning rate
```

## Monitoring Jobs

### Check queue
```bash
squeue -u $USER
```

### Watch output
```bash
tail -f results/logs/baseline_JOBID.out
```

### Cancel job
```bash
scancel JOBID
```

### View job details
```bash
scontrol show job JOBID
sacct -j JOBID --format=JobID,Elapsed,State,ExitCode
```

## Output Files

Each job produces:
- **stdout:** `results/logs/EXPERIMENT_JOBID.out`
- **stderr:** `results/logs/EXPERIMENT_JOBID.err`
- **CSV metrics:** `results/csv/EXPERIMENT_JOBID.csv`
- **Job stats:** `results/logs/sacct_JOBID.txt`

## Troubleshooting

### Job pending forever
```bash
# Check reason
squeue -u $USER --start

# Common issues:
# - Wrong partition
# - Wrong account
# - Resources not available
```

### Job fails immediately
```bash
# Check error log
cat results/logs/EXPERIMENT_JOBID.err

# Common issues:
# - Module not found → check env/load_modules.sh
# - NCCL error → check NCCL_SOCKET_IFNAME
# - Out of memory → reduce batch size
```

### NCCL timeout
```bash
# Set debug mode in script
export NCCL_DEBUG=INFO

# Check network interface
ip addr show

# Try different interface
export NCCL_SOCKET_IFNAME=eth0  # or ib1, mlx5_0
```

## Tips

1. **Test on 1 GPU first** before scaling up
2. **Use short runs** for initial testing (1-2 epochs)
3. **Check logs** immediately after submission
4. **Save job IDs** for tracking
5. **Use job arrays** for parameter sweeps

## Example Workflow

```bash
# 1. Test environment
sbatch slurm/ddp_baseline.sbatch

# 2. Wait for completion
watch -n 5 'squeue -u $USER'

# 3. Check results
cat results/csv/baseline_*.csv

# 4. Run scaling experiments
for nodes in 1 2; do
    # Edit ddp_strong_scaling.sbatch: --nodes=$nodes
    sbatch slurm/ddp_strong_scaling.sbatch
done

# 5. Generate plots
python src/plots/make_all.py
```
