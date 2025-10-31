#!/bin/bash
#SBATCH --job-name=cifar-weak-scaling
#SBATCH --account=YOUR_ACCOUNT          # CHANGE THIS
#SBATCH --partition=gpu                  # CHANGE THIS if needed
#SBATCH --qos=normal
#
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#
#SBATCH --time=02:00:00
#SBATCH --output=results/logs/weak_%j.out
#SBATCH --error=results/logs/weak_%j.err

# ============================================================================
# CIFAR-10 ResNet-18 Weak Scaling Experiment
#
# Weak scaling: Fixed per-GPU batch size, global batch increases with GPUs
# Test configurations: 1, 2, 4, 8 GPUs (1-2 nodes)
# ============================================================================

echo "=========================================="
echo "Weak Scaling Experiment"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "Total GPUs: $((SLURM_JOB_NUM_NODES * SLURM_GPUS_PER_NODE))"
echo "=========================================="

# Load environment
source env/load_modules.sh

# Set environment variables
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME=ib0  # CHANGE THIS

# Fixed per-GPU batch size for weak scaling
BATCH_PER_GPU=128
TOTAL_GPUS=$((SLURM_JOB_NUM_NODES * SLURM_GPUS_PER_NODE))
GLOBAL_BATCH=$((BATCH_PER_GPU * TOTAL_GPUS))

echo ""
echo "Weak Scaling Configuration:"
echo "  Batch per GPU (fixed): $BATCH_PER_GPU"
echo "  Total GPUs: $TOTAL_GPUS"
echo "  Global batch size: $GLOBAL_BATCH"
echo "=========================================="
echo ""

# Training parameters
DATA_DIR="./data"
EPOCHS=10
LR=0.1
RESULTS_DIR="./results/csv"
EXP_NAME="weak_${TOTAL_GPUS}gpu_${SLURM_JOB_ID}"

# Run training
srun python -m torch.distributed.run \
    --nproc_per_node=$SLURM_GPUS_PER_NODE \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --node_rank=$SLURM_NODEID \
    --master_addr=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1) \
    --master_port=29500 \
    src/train.py \
    --data $DATA_DIR \
    --epochs $EPOCHS \
    --batch-size $BATCH_PER_GPU \
    --lr $LR \
    --results-dir $RESULTS_DIR \
    --exp-name $EXP_NAME

echo ""
echo "=========================================="
echo "Weak scaling run completed"
echo "Results: $RESULTS_DIR/${EXP_NAME}_${TOTAL_GPUS}gpu_${SLURM_JOB_ID}.csv"
echo "=========================================="

# Save job statistics
sacct -j $SLURM_JOB_ID --format=JobID,Elapsed,AveCPU,MaxRSS,AllocTRES%40,State \
    > results/logs/sacct_weak_${SLURM_JOB_ID}.txt
