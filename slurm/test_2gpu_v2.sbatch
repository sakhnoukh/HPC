#!/bin/bash
#SBATCH --job-name=test-2gpu-v2
#SBATCH --partition=gpu-node
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=results/logs/test_2gpu_v2_%j.out
#SBATCH --error=results/logs/test_2gpu_v2_%j.err

echo "=========================================="
echo "Test 2-GPU DDP Training (v2 - Simplified)"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NODELIST"
echo "=========================================="

# Set environment
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth0
export CONTAINER=./env/hpc_pytorch.sif

# Get master node address (must be done before srun)
MASTER_NODE=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_ADDR=$MASTER_NODE
export MASTER_PORT=29500

echo "Master node: $MASTER_ADDR:$MASTER_PORT"
echo "Node list: $SLURM_JOB_NODELIST"
echo ""

# Training parameters
DATA_DIR="./data"
EPOCHS=2
BATCH_SIZE=128
LR=0.1
RESULTS_DIR="./results/csv"
EXP_NAME="test_2gpu_v2_${SLURM_JOB_ID}"

echo "Starting training..."
echo ""

# Run with srun using wrapper script that maps Slurm env vars to DDP vars
# Export MASTER_ADDR so it's available in the container
srun --export=ALL --ntasks=2 --ntasks-per-node=1 \
    apptainer exec --nvccli \
    --env MASTER_ADDR=$MASTER_ADDR \
    --env MASTER_PORT=$MASTER_PORT \
    $CONTAINER \
    bash src/train_slurm_wrapper.sh \
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
