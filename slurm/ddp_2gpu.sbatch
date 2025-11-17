#!/bin/bash
#SBATCH --job-name=cifar-2gpu
#SBATCH --partition=gpu-node
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=results/logs/baseline_2gpu_%j.out
#SBATCH --error=results/logs/baseline_2gpu_%j.err

echo "=========================================="
echo "CIFAR-10 ResNet-18 Baseline (2 GPUs, 2 Nodes)"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NODELIST"
echo "Tasks: $SLURM_NTASKS"
echo "=========================================="

# Set environment
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME=eth0
export CONTAINER=./env/hpc_pytorch.sif

# Get master node address
MASTER_NODE=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_ADDR=$MASTER_NODE
export MASTER_PORT=29500

echo "Master node: $MASTER_ADDR:$MASTER_PORT"
echo ""

# Training parameters
DATA_DIR="./data"
EPOCHS=10
BATCH_SIZE=128  # Per GPU
LR=0.1
RESULTS_DIR="./results/csv"
EXP_NAME="baseline_${SLURM_JOB_ID}"

echo "Training Configuration:"
echo "  Epochs: $EPOCHS"
echo "  Batch size per GPU: $BATCH_SIZE"
echo "  Global batch size: $((BATCH_SIZE * SLURM_NTASKS))"
echo "  Learning rate: $LR"
echo "  Results: $RESULTS_DIR/${EXP_NAME}_2gpu_${SLURM_JOB_ID}.csv"
echo "=========================================="
echo ""

# Run DDP training
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
echo "Job completed at: $(date)"
echo "Results: $RESULTS_DIR/${EXP_NAME}_2gpu_${SLURM_JOB_ID}.csv"
echo "=========================================="

# Save job statistics
sacct -j $SLURM_JOB_ID --format=JobID,Elapsed,AveCPU,AveRSS,MaxRSS,AllocTRES%40,State \
    > results/logs/sacct_${SLURM_JOB_ID}.txt 2>&1 || echo "Note: sacct not available during job"
