#!/bin/bash
#SBATCH --job-name=test-2gpu
#SBATCH --partition=gpu-node
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=results/logs/test_2gpu_%j.out
#SBATCH --error=results/logs/test_2gpu_%j.err

echo "=========================================="
echo "Test 2-GPU DDP Training"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NODELIST"
echo "Tasks: $SLURM_NTASKS"
echo "GPUs: $SLURM_GPUS"
echo "=========================================="

# Set environment
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth0
export CONTAINER=./env/hpc_pytorch.sif

echo ""
echo "Testing GPU visibility on each node:"
srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 \
    apptainer exec --nvccli $CONTAINER nvidia-smi --query-gpu=name,memory.total --format=csv

echo ""
echo "Testing PyTorch CUDA on each node:"
srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 \
    apptainer exec --nvccli $CONTAINER \
    python3 -c "import torch; print(f'Node {torch.cuda.current_device()}: CUDA={torch.cuda.is_available()}, GPUs={torch.cuda.device_count()}')"

echo ""
echo "=========================================="
echo "Starting 2-GPU Training (2 epochs test)"
echo "=========================================="

# Training parameters
DATA_DIR="./data"
EPOCHS=2
BATCH_SIZE=128
LR=0.1
RESULTS_DIR="./results/csv"
EXP_NAME="test_2gpu_${SLURM_JOB_ID}"

# Run DDP training across 2 nodes
srun apptainer exec --nvccli $CONTAINER \
    python -m torch.distributed.run \
    --nproc_per_node=1 \
    --nnodes=2 \
    --node_rank=$SLURM_NODEID \
    --master_addr=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1) \
    --master_port=29500 \
    src/train.py \
    --data $DATA_DIR \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --lr $LR \
    --results-dir $RESULTS_DIR \
    --exp-name $EXP_NAME

echo ""
echo "=========================================="
echo "Test completed at: $(date)"
echo "Results: $RESULTS_DIR/${EXP_NAME}_2gpu_${SLURM_JOB_ID}.csv"
echo "=========================================="
