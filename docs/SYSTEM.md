# System Configuration

> **Note:** This file should be filled out with your specific HPC cluster details.  
> Run commands on your cluster to gather accurate information.

## HPC Cluster Specifications

**Cluster Name:** Magic Castle HPC  
**Institution:** IE University  
**Access Date:** November 2025

---

## Compute Nodes

### GPU Nodes
- **Number of Nodes:** 2 GPU nodes available
- **GPUs per Node:** 1x NVIDIA Tesla T4
- **GPU Memory:** 16GB per GPU
- **GPU Architecture:** Turing (T4)

### CPU Specifications
- **Processors:** Intel Xeon Platinum 8473C
- **Total Cores per Node:** 4 cores per GPU node
- **RAM per Node:** 56 GB

### Interconnect
- **Type:** Ethernet
- **Topology:** Standard Ethernet network
- **Interface Name:** eth0 (for NCCL_SOCKET_IFNAME)

---

## Software Stack

### Operating System
- **OS:** Rocky Linux 9 (el9)
- **Kernel:** Linux (exact version TBD)

### Workload Manager
- **Slurm Version:** Slurm 23.x
- **Partitions Available:** gpu-node, cpubase_bycore_b1, node

### CUDA Ecosystem
- **CUDA Version:** 12.4 (from NVIDIA Driver)
- **cuDNN Version:** 8.9+ (from NGC container)
- **NCCL Version:** 2.14.3
- **NVIDIA Driver:** 550.144.06

### Compilers
- **GCC:** [e.g., 11.2.0]
- **Intel Compiler:** [e.g., 2023.0] (if available)

### MPI
- **OpenMPI:** [e.g., 4.1.5]
- **MPICH:** [e.g., 4.0] (if available)

### Python & Deep Learning
- **Python Version:** 3.10.11 (from Apptainer container)
- **PyTorch Version:** 2.0.1
- **torchvision Version:** 0.15.2

---

## Storage

### Home Directory
- **Path:** [e.g., /home/username]
- **Quota:** [e.g., 50 GB]
- **Backup:** [e.g., Daily snapshots]

### Scratch/Temporary Storage
- **Path:** [e.g., /scratch/username or $TMPDIR]
- **Quota:** [e.g., 10 TB or unlimited]
- **Filesystem:** [e.g., Lustre, BeeGFS, GPFS]
- **Retention:** [e.g., Files deleted after 30 days]
- **Performance:** [e.g., High-speed parallel filesystem]

### Project/Persistent Storage
- **Path:** [e.g., /project/groupname]
- **Quota:** [e.g., 5 TB]
- **Shared:** [e.g., Yes, across team members]

---

## Job Submission Details

### Slurm Configuration
```bash
# Account and partition info (fill in your values)
ACCOUNT="your_account"
PARTITION="gpu"
QOS="normal"
```

### Resource Limits
- **Max Walltime:** [e.g., 24 hours for normal QoS, 72 hours for long QoS]
- **Max Nodes per Job:** [e.g., 8 nodes]
- **Max GPUs per Job:** [e.g., 32 GPUs]
- **Priority Queue:** [e.g., Yes, for short jobs < 1 hour]

### Example sbatch Header
```bash
#!/bin/bash
#SBATCH --account=your_account
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --job-name=cifar10-ddp
#SBATCH --output=results/logs/job_%j.out
#SBATCH --error=results/logs/job_%j.err
```

---

## Environment Setup

### Chosen Strategy
**Selected:** [X] Apptainer  [ ] Modules  [ ] Conda

### Setup Commands
```bash
# Fill in based on your choice

# Option A: Apptainer
apptainer build hpc_pytorch.sif env/project.def

# Option B: Modules
source env/load_modules.sh

# Option C: Conda (if used)
# conda env create -f env/environment.yml
# conda activate hpc-cifar10
```

---

## Network Configuration

### NCCL Settings
```bash
# Network interface for NCCL (check with `ifconfig` or `ip a`)
export NCCL_SOCKET_IFNAME=ib0  # Change to your interface

# NCCL debug level (INFO for debugging, WARN for production)
export NCCL_DEBUG=WARN

# Optional: Force NCCL to use specific protocol
# export NCCL_IB_DISABLE=0  # Enable InfiniBand
# export NCCL_NET_GDR_LEVEL=3  # GPU Direct RDMA
```

### Checking Network
```bash
# List network interfaces
ip addr show

# Check InfiniBand status (if applicable)
ibstatus

# Test GPU-to-GPU communication
nvidia-smi topo -m
```

---

## Verification Commands

Run these on your cluster to gather system information:

### GPU Info
```bash
nvidia-smi
nvidia-smi topo -m
```

### Module Info
```bash
module avail
module list
```

### CUDA/PyTorch Test
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"
python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}')"
```

### Slurm Info
```bash
sinfo
squeue -u $USER
sacct -u $USER
```

---

## Tested Configurations

| Date | Config | Nodes | GPUs | Batch/GPU | Status | Notes |
|------|--------|-------|------|-----------|--------|-------|
| 2025-11-17 | 1-GPU FP32 | 1 | 1 | 128 | ✅ | Baseline: 4,460 img/s |
| 2025-11-17 | 1-GPU FP16 | 1 | 1 | 128 | ✅ | Optimized: 7,600 img/s (1.7x speedup) |
| 2025-11-17 | 2-GPU Multi-Node | 2 | 2 | 128 | ✅ | Scaling: 5,000 img/s (limited by network) |

---

## Known Issues / Workarounds

### Issue 1: [Description]
**Solution:** [Workaround]

### Issue 2: [Description]
**Solution:** [Workaround]

---

## Performance Baseline

### Single GPU FP32 (Baseline)
- **Throughput:** 4,460 images/sec
- **Time per Epoch:** 11 seconds
- **Final Accuracy:** 77.42%

### Single GPU FP16 (Optimized)
- **Throughput:** 7,600 images/sec
- **Time per Epoch:** 6.6 seconds
- **Speedup:** 1.7x over FP32
- **Final Accuracy:** 78.7%

### 2 GPUs (2 Nodes, Multi-Node)
- **Throughput:** 5,000 images/sec (global)
- **Time per Epoch:** 20 seconds
- **Scaling Efficiency:** 56% (limited by inter-node communication)
- **Final Accuracy:** 74.08%

---

## Contact & Support

**Cluster Support:** [e.g., hpc-support@university.edu]  
**Documentation:** [e.g., https://hpc.university.edu/docs]  
**Status Page:** [e.g., https://status.hpc.university.edu]

---

**Last Updated:** 2025-11-17  
**Completed:** All experiments finished
