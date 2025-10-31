#!/bin/bash
#SBATCH --job-name=cifar-strong-scaling
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
#SBATCH --output=results/logs/strong_%j.out
#SBATCH --error=results/logs/strong_%j.err

# ============================================================================
# CIFAR-10 ResNet-18 Strong Scaling Experiment
# 
# Strong scaling: Fixed global batch size, distributed across more GPUs
# Test configurations: 1, 2, 4, 8 GPUs (1-2 nodes)
# ============================================================================

echo "=========================================="
echo "Strong Scaling Experiment"
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

# Fixed global batch size for strong scaling
GLOBAL_BATCH=512
TOTAL_GPUS=$((SLURM_JOB_NUM_NODES * SLURM_GPUS_PER_NODE))
BATCH_PER_GPU=$((GLOBAL_BATCH / TOTAL_GPUS))

echo ""
echo "Strong Scaling Configuration:"
echo "  Global batch size (fixed): $GLOBAL_BATCH"
echo "  Total GPUs: $TOTAL_GPUS"
echo "  Batch per GPU: $BATCH_PER_GPU"
echo "=========================================="
echo ""

# Training parameters
DATA_DIR="./data"
EPOCHS=10
LR=0.1
RESULTS_DIR="./results/csv"
EXP_NAME="strong_${TOTAL_GPUS}gpu_${SLURM_JOB_ID}"

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
echo "Strong scaling run completed"
echo "Results: $RESULTS_DIR/${EXP_NAME}_${TOTAL_GPUS}gpu_${SLURM_JOB_ID}.csv"
echo "=========================================="

# Save job statistics
sacct -j $SLURM_JOB_ID --format=JobID,Elapsed,AveCPU,MaxRSS,AllocTRES%40,State \
    > results/logs/sacct_strong_${SLURM_JOB_ID}.txt
