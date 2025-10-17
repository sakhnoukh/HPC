# System Configuration

> **Note:** This file should be filled out with your specific HPC cluster details.  
> Run commands on your cluster to gather accurate information.

## HPC Cluster Specifications

**Cluster Name:** [YOUR_CLUSTER_NAME]  
**Institution:** [YOUR_UNIVERSITY]  
**Access Date:** October 2025

---

## Compute Nodes

### GPU Nodes
- **Number of Nodes:** [e.g., 8 GPU nodes available]
- **GPUs per Node:** [e.g., 4x NVIDIA A100-SXM4-40GB]
- **GPU Memory:** [e.g., 40GB per GPU]
- **GPU Architecture:** [e.g., Ampere (A100), Volta (V100), Hopper (H100)]

### CPU Specifications
- **Processors:** [e.g., 2x AMD EPYC 7742 (64 cores per socket)]
- **Total Cores per Node:** [e.g., 128 cores]
- **RAM per Node:** [e.g., 512 GB DDR4]

### Interconnect
- **Type:** [e.g., InfiniBand HDR 200Gb/s, Ethernet 100GbE]
- **Topology:** [e.g., Fat-tree, Dragonfly]
- **Interface Name:** [e.g., ib0, mlx5_0] (for NCCL_SOCKET_IFNAME)

---

## Software Stack

### Operating System
- **OS:** [e.g., Rocky Linux 8.8, Ubuntu 22.04]
- **Kernel:** [e.g., 4.18.0-477.el8.x86_64]

### Workload Manager
- **Slurm Version:** [e.g., 23.02.6]
- **Partitions Available:** [e.g., gpu, gpu-dev, gpu-long]

### CUDA Ecosystem
- **CUDA Version:** [e.g., 12.1.1]
- **cuDNN Version:** [e.g., 8.9.2]
- **NCCL Version:** [e.g., 2.18.3]
- **NVIDIA Driver:** [e.g., 535.104.05]

### Compilers
- **GCC:** [e.g., 11.2.0]
- **Intel Compiler:** [e.g., 2023.0] (if available)

### MPI
- **OpenMPI:** [e.g., 4.1.5]
- **MPICH:** [e.g., 4.0] (if available)

### Python & Deep Learning
- **Python Version:** [e.g., 3.10.12]
- **PyTorch Version:** [e.g., 2.0.1]
- **torchvision Version:** [e.g., 0.15.2]

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
**Selected:** [ ] Apptainer  [ ] Modules  [ ] Conda

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
| 2025-10-17 | Local | 1 | 1 | 128 | ✅ | Baseline code working |
| TBD | Single Node | 1 | 4 | 128 | ⏳ | Testing DDP |
| TBD | Multi-Node | 2 | 8 | 128 | ⏳ | Strong scaling |
| TBD | Multi-Node | 3 | 12 | 128 | ⏳ | Weak scaling |

---

## Known Issues / Workarounds

### Issue 1: [Description]
**Solution:** [Workaround]

### Issue 2: [Description]
**Solution:** [Workaround]

---

## Performance Baseline

### Single GPU (Baseline)
- **Throughput:** [e.g., 5,000 images/sec]
- **Time per Epoch:** [e.g., 10 seconds]
- **GPU Utilization:** [e.g., 85%]

### 4 GPUs (Single Node)
- **Throughput:** [TBD]
- **Strong Scaling Efficiency:** [TBD]

### 8 GPUs (2 Nodes)
- **Throughput:** [TBD]
- **Strong Scaling Efficiency:** [TBD]

---

## Contact & Support

**Cluster Support:** [e.g., hpc-support@university.edu]  
**Documentation:** [e.g., https://hpc.university.edu/docs]  
**Status Page:** [e.g., https://status.hpc.university.edu]

---

**Last Updated:** 2025-10-17  
**To Be Completed:** After first cluster access
