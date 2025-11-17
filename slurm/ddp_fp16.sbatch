#!/bin/bash
#SBATCH --job-name=cifar-fp16
#SBATCH --partition=gpu-node
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=results/logs/fp16_%j.out
#SBATCH --error=results/logs/fp16_%j.err

echo "=========================================="
echo "CIFAR-10 ResNet-18 with FP16 Mixed Precision"
echo "Job ID: $SLURM_JOB_ID"
echo "=========================================="

# Set environment
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME=eth0
export CONTAINER=./env/hpc_pytorch.sif

# Get master node address (for DDP compatibility)
MASTER_NODE=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_ADDR=$MASTER_NODE
export MASTER_PORT=29500

# Training parameters
DATA_DIR="./data"
EPOCHS=10
BATCH_SIZE=128
LR=0.1
RESULTS_DIR="./results/csv"
EXP_NAME="fp16_${SLURM_JOB_ID}"

echo "Training Configuration:"
echo "  Precision: FP16 (Mixed Precision)"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "=========================================="
echo ""

# Run with FP16
srun --export=ALL --ntasks=1 \
    apptainer exec --nvccli \
    --env MASTER_ADDR=$MASTER_ADDR \
    --env MASTER_PORT=$MASTER_PORT \
    $CONTAINER \
    bash src/train_slurm_wrapper.sh \
    --data $DATA_DIR \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --lr $LR \
    --precision fp16 \
    --results-dir $RESULTS_DIR \
    --exp-name $EXP_NAME

echo ""
echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="
