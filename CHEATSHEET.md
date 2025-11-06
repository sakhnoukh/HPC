# HPC CIFAR-10 Quick Reference

Essential commands and workflows for the HPC project.

## 🚀 Quick Start

### First Time Setup
```bash
# Clone repository
git clone https://github.com/sakhnoukh/HPC.git
cd HPC

# Setup environment
source env/load_modules.sh  # OR build container
pip install -r env/requirements.txt

# Download data
python data/fetch_cifar10.py --data-dir ./data

# Test setup
python test_setup.py
```

### Configure for Your Cluster
```bash
# Edit ALL .sbatch files:
# 1. Set account: #SBATCH --account=YOUR_ACCOUNT
# 2. Set partition: #SBATCH --partition=gpu
# 3. Set network: export NCCL_SOCKET_IFNAME=ib0

# Find your network interface:
ip addr show | grep -E "^[0-9]+:"
```

---

## 📝 Common Workflows

### Run Experiments
```bash
# Interactive submission
chmod +x scripts/run_all_experiments.sh
./scripts/run_all_experiments.sh

# Manual submission
sbatch slurm/ddp_baseline.sbatch
sbatch slurm/ddp_strong_scaling.sbatch
sbatch slurm/ddp_weak_scaling.sbatch
```

### Monitor Jobs
```bash
# Quick status
squeue -u $USER

# Detailed monitoring
chmod +x scripts/monitor_jobs.sh
./scripts/monitor_jobs.sh

# Watch live (updates every 5 seconds)
watch -n 5 ./scripts/monitor_jobs.sh

# View output
tail -f results/logs/baseline_JOBID.out
```

### Analyze Results
```bash
# Quick analysis
python scripts/quick_analysis.py

# Generate all plots
python src/plots/make_all.py

# View specific CSV
head results/csv/*.csv
```

---

## 🔧 Training Commands

### Local Testing
```bash
# Single GPU
python src/train.py --epochs 2 --batch-size 128 --data ./data

# Multi-GPU (4 GPUs)
torchrun --nproc_per_node=4 src/train.py --epochs 2 --batch-size 128 --data ./data

# With mixed precision
python src/train.py --precision bf16 --epochs 5 --batch-size 128
```

### Cluster Jobs
```bash
# Baseline (1 node, 4 GPUs)
sbatch slurm/ddp_baseline.sbatch

# Strong scaling (2 nodes, 8 GPUs)
sbatch slurm/ddp_strong_scaling.sbatch

# Sensitivity sweep (6 configs)
sbatch slurm/ddp_sensitivity.sbatch

# Profiling
sbatch slurm/profile_gpu_nsys.sbatch
sbatch slurm/profile_gpu_ncu.sbatch
```

---

## 📊 Slurm Commands

### Job Submission
```bash
sbatch script.sbatch                    # Submit job
sbatch --dependency=afterok:123 job.sh  # Submit after job 123
```

### Job Management
```bash
squeue -u $USER                         # My jobs
squeue -u $USER --start                 # Estimated start time
scancel JOBID                           # Cancel job
scancel -u $USER                        # Cancel all my jobs
scontrol show job JOBID                 # Job details
scontrol update JobId=JOBID TimeLimit=02:00:00  # Extend time
```

### Job History
```bash
sacct -u $USER                          # Recent jobs
sacct -j JOBID                          # Specific job
sacct -j JOBID --format=JobID,Elapsed,State,MaxRSS,AllocTRES
sacct -S 2025-11-01                     # Jobs since date
```

### Resource Info
```bash
sinfo                                   # Cluster status
sinfo -p gpu                            # GPU partition
scontrol show partition gpu             # Partition details
scontrol show node nodename             # Node details
```

---

## 📈 Data Analysis

### Quick Stats
```bash
# CSV summary
python -c "import pandas as pd; df=pd.read_csv('results/csv/baseline_*.csv'); print(df.describe())"

# Latest results
tail -n 5 results/csv/*.csv

# Validation accuracy
grep val_acc results/csv/*.csv | tail -n 10
```

### Plotting
```bash
# All plots
python src/plots/make_all.py

# Custom plot directory
python src/plots/make_all.py --csv-dir results/csv --output-dir plots/

# View plots
ls -lh results/plots/
```

### Benchmarking
```bash
# Run benchmark
python scripts/benchmark.py

# Different batch sizes
python scripts/benchmark.py --batch-size 256 --iterations 200
```

---

## 🔍 Debugging

### Check Job Status
```bash
# Why is my job pending?
squeue -u $USER --start

# Job errors
cat results/logs/JOBID.err

# Job output (last 50 lines)
tail -n 50 results/logs/JOBID.out
```

### Common Issues

**Out of Memory**
```bash
# Reduce batch size in .sbatch file or CLI
--batch-size 64  # instead of 128
```

**NCCL Timeout**
```bash
# Check network interface
ip addr show

# Try different interface in .sbatch
export NCCL_SOCKET_IFNAME=eth0  # or ib1, mlx5_0
export NCCL_DEBUG=INFO          # More verbose logging
```

**Module Not Found**
```bash
# Check module availability
module avail pytorch
module avail cuda

# Update env/load_modules.sh with correct versions
```

**Job Fails Immediately**
```bash
# Check Slurm configuration
scontrol show job JOBID

# Verify account and partition
sacctmgr show user $USER
sinfo -p gpu
```

---

## 🎯 Performance Tips

### Optimize Throughput
```bash
# Use mixed precision
--precision bf16  # For A100/H100
--precision fp16  # For V100

# Increase batch size
--batch-size 256  # If memory allows

# More dataloader workers
--num-workers 8   # Match CPU cores per task
```

### Reduce Queue Time
```bash
# Request shorter walltime
#SBATCH --time=00:30:00

# Use development partition (if available)
#SBATCH --partition=gpu-dev

# Run during off-peak hours
# Usually nights and weekends
```

---

## 📂 File Locations

### Important Files
```
results/csv/              # Training metrics (CSV)
results/plots/            # Generated figures (PNG/SVG)
results/logs/             # Slurm output and profiling
slurm/                    # Job scripts
src/                      # Source code
scripts/                  # Helper scripts
```

### Output Naming
```
results/csv/baseline_4gpu_JOBID.csv
results/logs/baseline_JOBID.out
results/logs/nsys_JOBID.nsys-rep
```

---

## 🧪 Testing Before Production

```bash
# 1. Test environment
python test_setup.py

# 2. Benchmark performance
python scripts/benchmark.py

# 3. Quick training test (1 epoch, CPU)
python src/train.py --epochs 1 --batch-size 32

# 4. GPU test (if available locally)
python src/train.py --epochs 1 --batch-size 128 --data ./data

# 5. Submit minimal cluster job
# Edit slurm/ddp_baseline.sbatch: --epochs=1
sbatch slurm/ddp_baseline.sbatch

# 6. Check output
tail -f results/logs/baseline_*.out
```

---

## 🔄 Git Workflow

```bash
# Pull latest changes
git pull origin main

# Check status
git status

# Add new results (be selective!)
git add src/ slurm/ scripts/

# Commit
git commit -m "Description of changes"

# Push
git push origin main

# Create release tag
git tag -a v1.0 -m "Final submission"
git push origin v1.0
```

---

## 📞 Getting Help

### Cluster Support
```bash
# Check documentation
module help pytorch
man sbatch

# Cluster status page
# (Check your cluster's documentation)
```

### Project Issues
- **GitHub Issues:** https://github.com/sakhnoukh/HPC/issues
- **Documentation:** See README.md, reproduce.md, ROADMAP.md

---

## ✅ Pre-Submission Checklist

- [ ] Data downloaded: `ls data/cifar-10-batches-py/`
- [ ] Environment working: `python test_setup.py`
- [ ] Slurm scripts configured (account, partition, network)
- [ ] Baseline job tested: `sbatch slurm/ddp_baseline.sbatch`
- [ ] Results collected: `ls results/csv/*.csv`
- [ ] Plots generated: `python src/plots/make_all.py`
- [ ] Profiling completed: Nsight reports in `results/logs/`
- [ ] Code committed: `git status`

---

## 🎓 Expected Results

### Target Metrics
- **Strong scaling efficiency @ 8 GPUs:** ≥70%
- **Weak scaling time variance:** ≤10%
- **Validation accuracy:** 70-75% (10 epochs)
- **Throughput improvement (BF16 vs FP32):** ~1.5-2x

### Typical Runtimes
- 1 GPU, 10 epochs: ~30 minutes
- 4 GPUs, 10 epochs: ~10 minutes
- 8 GPUs, 10 epochs: ~6 minutes
- Nsight Systems: ~5 minutes
- Nsight Compute: ~30 minutes (slow due to overhead)

---

**Last Updated:** 2025-11-06  
**For detailed guides, see:** `reproduce.md` and `ROADMAP.md`
