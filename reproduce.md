# Reproduction Guide

Step-by-step instructions to reproduce all experiments and results.

## Prerequisites

### Required Access
- HPC cluster with GPU nodes (minimum: 1 node with 4 GPUs)
- Slurm workload manager
- CUDA-capable GPUs (tested on A100/V100)

### Software Requirements
- Python 3.9+
- PyTorch 2.0+
- torchvision
- CUDA 11.8+ or 12.x
- Slurm

---

## Step 1: Environment Setup

### Clone Repository
```bash
git clone https://github.com/sakhnoukh/HPC.git
cd HPC
```

### Choose Environment Strategy

**Option A: Apptainer (Recommended)**
```bash
# Build container
apptainer build hpc_pytorch.sif env/project.def

# Test container
apptainer exec --nv hpc_pytorch.sif python --version
```

**Option B: Modules**
```bash
# Edit env/load_modules.sh with your cluster's module names
vi env/load_modules.sh

# Load modules
source env/load_modules.sh

# Install Python packages
pip install -r env/requirements.txt
```

### Verify Environment
```bash
python test_setup.py
```

---

## Step 2: Data Preparation

### Download CIFAR-10
```bash
python data/fetch_cifar10.py --data-dir ./data
```

Expected output:
- Dataset location: `./data/cifar-10-batches-py/`
- Size: ~170 MB
- Files: 50,000 training images + 10,000 test images

---

## Step 3: Configure Slurm Scripts

Edit ALL `.sbatch` files in `slurm/` directory:

1. **Account and Partition**
   ```bash
   #SBATCH --account=YOUR_ACCOUNT    # Replace with your account
   #SBATCH --partition=gpu            # Replace with your GPU partition
   ```

2. **Network Interface** (line ~40 in each script)
   ```bash
   export NCCL_SOCKET_IFNAME=ib0  # Change to your interface (ib0, eth0, etc.)
   ```
   
   Find your interface:
   ```bash
   # On cluster node
   ip addr show
   # or
   ifconfig
   ```

3. **Container Path** (if using Apptainer)
   Uncomment Apptainer execution lines and comment out direct execution.

---

## Step 4: Baseline Experiment (Single Node, 4 GPUs)

### Submit Job
```bash
sbatch slurm/ddp_baseline.sbatch
```

### Monitor Job
```bash
# Check queue
squeue -u $USER

# Watch output (replace JOBID)
tail -f results/logs/baseline_JOBID.out

# Check job details
scontrol show job JOBID
```

### Expected Results
- Runtime: ~10-15 minutes (10 epochs)
- CSV output: `results/csv/baseline_JOBID_4gpu_JOBID.csv`
- Validation accuracy: ~70-75% after 10 epochs
- Throughput: ~15,000-20,000 images/sec (depends on GPU model)

---

## Step 5: Strong Scaling Experiments

Test configurations: 1, 2, 4, 8 GPUs (1-2 nodes)

### 1 GPU Baseline
```bash
# Edit slurm/ddp_baseline.sbatch
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1

sbatch slurm/ddp_baseline.sbatch
```

### 2 GPUs
```bash
# Edit slurm/ddp_baseline.sbatch
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=2

sbatch slurm/ddp_baseline.sbatch
```

### 4 GPUs (Single Node)
```bash
# Use default ddp_baseline.sbatch
sbatch slurm/ddp_baseline.sbatch
```

### 8 GPUs (2 Nodes)
```bash
# Edit slurm/ddp_strong_scaling.sbatch
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4

sbatch slurm/ddp_strong_scaling.sbatch
```

### Expected Strong Scaling Efficiency
| GPUs | Target Efficiency | Expected Throughput |
|------|-------------------|---------------------|
| 1    | 100% (baseline)   | ~5,000 img/s        |
| 2    | ≥90%              | ~9,000 img/s        |
| 4    | ≥80%              | ~16,000 img/s       |
| 8    | ≥70%              | ~28,000 img/s       |

---

## Step 6: Weak Scaling Experiments

Test configurations: 1, 2, 4, 8 GPUs with fixed per-GPU batch size (128).

### Run Weak Scaling
```bash
# For each configuration (1, 2, 4, 8 GPUs)
# Edit --nodes and --ntasks-per-node in slurm/ddp_weak_scaling.sbatch
sbatch slurm/ddp_weak_scaling.sbatch
```

### Expected Weak Scaling
- Time per epoch should remain relatively constant (±10%)
- Efficiency ≥90% at 8 GPUs

---

## Step 7: Sensitivity Analysis

Test different batch sizes and precision settings.

### Submit Job Array
```bash
sbatch slurm/ddp_sensitivity.sbatch
```

This submits 6 jobs testing combinations of:
- Batch sizes: 64, 128, 256
- Precision: FP32, BF16

### Monitor Array Jobs
```bash
squeue -u $USER
```

Expected behavior:
- Smaller batch sizes: Lower throughput, potentially better accuracy
- BF16: Higher throughput (~1.5-2x), similar accuracy

---

## Step 8: Profiling

### Nsight Systems (Timeline)
```bash
sbatch slurm/profile_gpu_nsys.sbatch
```

Output: `results/logs/nsys_JOBID.nsys-rep`

Download and open in Nsight Systems GUI to view:
- CUDA kernel timeline
- NCCL communication
- CPU-GPU synchronization
- Memory transfers

### Nsight Compute (Kernel Details)
```bash
sbatch slurm/profile_gpu_ncu.sbatch
```

Output: `results/logs/ncu_JOBID.ncu-rep`

Download and open in Nsight Compute GUI to analyze:
- Kernel efficiency
- Memory bandwidth utilization
- Occupancy
- Roofline analysis

---

## Step 9: Generate Plots

After collecting all results:

```bash
python src/plots/make_all.py --csv-dir results/csv --output-dir results/plots
```

Generated plots:
- `throughput_vs_gpus.png` - Scaling performance
- `strong_scaling_efficiency.png` - Strong scaling analysis
- `weak_scaling.png` - Weak scaling analysis
- `accuracy_curves.png` - Training convergence
- `sensitivity_batch_size.png` - Hyperparameter sensitivity
- `summary_statistics.csv` - Numerical results table

---

## Step 10: Verify Results

### Check CSV Files
```bash
ls -lh results/csv/
head results/csv/*.csv
```

### Verify Metrics
```bash
# Final validation accuracy should be ~70-75%
grep "val_acc" results/csv/*.csv | tail -n 20

# Check throughput scaling
python -c "import pandas as pd; df = pd.read_csv('results/csv/baseline_*.csv'); print(df[['gpus', 'images_per_sec']].groupby('gpus').mean())"
```

### Expected Outputs

**CSV Columns:**
```
timestamp, jobid, commit, epoch, epochs, world_size, gpus, batch_per_gpu, 
global_batch, train_loss, train_acc, val_loss, val_acc, epoch_time_s, 
images_per_sec
```

**Key Metrics:**
- Train accuracy: 75-80% after 10 epochs
- Validation accuracy: 70-75% after 10 epochs
- Strong scaling efficiency (8 GPUs): ≥70%
- Weak scaling time variance: ≤10%

---

## Troubleshooting

### Job Fails Immediately
```bash
# Check Slurm output
cat results/logs/*.err

# Common issues:
# 1. Wrong partition/account
# 2. NCCL_SOCKET_IFNAME incorrect
# 3. Module not found
```

### Out of Memory Error
```bash
# Reduce batch size in .sbatch file
BATCH_SIZE=64  # or 32
```

### NCCL Timeout
```bash
# Check network interface
export NCCL_DEBUG=INFO
# Look for "NET/" lines in output

# Try different interface
export NCCL_SOCKET_IFNAME=eth0  # or ib1, mlx5_0, etc.
```

### Wrong GPU Count
```bash
# Verify in job output
python -c "import torch; print(torch.cuda.device_count())"

# Check Slurm allocation
echo $SLURM_GPUS_PER_NODE
```

---

## Expected Runtime

| Experiment | GPUs | Epochs | Time | Priority |
|------------|------|--------|------|----------|
| Baseline | 1 | 10 | ~30 min | High |
| Baseline | 4 | 10 | ~10 min | High |
| Strong Scaling | 8 | 10 | ~6 min | High |
| Weak Scaling | 1-8 | 10 | ~30 min | High |
| Sensitivity | 4 | 10 | ~60 min | Medium |
| Profiling (Nsys) | 4 | 2 | ~5 min | Medium |
| Profiling (Ncu) | 1 | 1 | ~30 min | Low |

**Total compute time:** ~3-4 GPU-node-hours

---

## Validation Checklist

- [ ] All CSV files generated
- [ ] Strong scaling efficiency ≥70% at 8 GPUs
- [ ] Weak scaling time variance ≤10%
- [ ] Validation accuracy 70-75%
- [ ] All plots generated successfully
- [ ] Nsight profiles collected
- [ ] Summary statistics table created

---

## Support

**Issues:** https://github.com/sakhnoukh/HPC/issues  
**Documentation:** See `docs/SYSTEM.md` for cluster-specific details

---

**Last Updated:** 2025-10-30  
**Tested On:** [YOUR_CLUSTER_NAME]  
**GPU Model:** [e.g., NVIDIA A100]
