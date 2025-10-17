#!/bin/bash
# Load required modules for HPC cluster
# IMPORTANT: Adjust module names/versions to match your specific cluster
# Run `module avail` on your cluster to see available modules

module purge

# Compiler (choose one based on your cluster)
module load gcc/11.2.0
# module load intel/2023.0

# CUDA (adjust version)
module load cuda/12.1
# module load cuda/11.8

# MPI (if needed for multi-node)
module load openmpi/4.1.5
# module load mpich/4.0

# Python
module load python/3.10
# module load anaconda3/2023.03

# PyTorch (if available as module)
# module load pytorch/2.0.1
# If not available, install via pip below

# List loaded modules
module list

# Set environment variables
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME=ib0  # Adjust for your cluster (ib0, eth0, etc.)

# Create/activate virtual environment (optional, recommended)
# Uncomment and adjust path as needed
# if [ ! -d "./venv" ]; then
#     echo "Creating virtual environment..."
#     python -m venv venv
#     source venv/bin/activate
#     pip install --upgrade pip
#     pip install torch torchvision matplotlib seaborn pandas tqdm
# else
#     source venv/bin/activate
# fi

echo "Environment loaded successfully!"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "Number of GPUs: $(python -c 'import torch; print(torch.cuda.device_count())')"
