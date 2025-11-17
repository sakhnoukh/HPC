#!/bin/bash
#SBATCH --job-name=test-pytorch
#SBATCH --partition=gpu-node
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=results/logs/test_%j.out
#SBATCH --error=results/logs/test_%j.err

echo "=========================================="
echo "Test Job - PyTorch + GPU Check"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "=========================================="

# Set environment
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NCCL_SOCKET_IFNAME=eth0

# Container path
CONTAINER=./env/hpc_pytorch.sif

echo ""
echo "Testing container..."
apptainer exec --nvccli $CONTAINER python3 --version
apptainer exec --nvccli $CONTAINER python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
apptainer exec --nvccli $CONTAINER python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
apptainer exec --nvccli $CONTAINER python3 -c "import torch; print(f'CUDA version: {torch.version.cuda}')"
apptainer exec --nvccli $CONTAINER python3 -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"
apptainer exec --nvccli $CONTAINER python3 -c "import torch; print(f'GPU name: {torch.cuda.get_device_name(0)}')"

echo ""
echo "Running test_setup.py..."
apptainer exec --nvccli $CONTAINER python3 test_setup.py

echo ""
echo "=========================================="
echo "Test completed!"
echo "=========================================="
