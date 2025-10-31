#!/bin/bash
#SBATCH --job-name=cifar-sensitivity
#SBATCH --account=YOUR_ACCOUNT          # CHANGE THIS
#SBATCH --partition=gpu                  # CHANGE THIS if needed
#SBATCH --qos=normal
#
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#
#SBATCH --time=01:00:00
#SBATCH --array=0-5                      # 6 configurations
#SBATCH --output=results/logs/sensitivity_%A_%a.out
#SBATCH --error=results/logs/sensitivity_%A_%a.err

# ============================================================================
# CIFAR-10 ResNet-18 Sensitivity Sweep
#
# Test different hyperparameter combinations:
# - Batch sizes: 64, 128, 256
# - Precision: fp32, bf16
# ============================================================================

echo "=========================================="
echo "Sensitivity Sweep Experiment"
echo "Job Array ID: $SLURM_ARRAY_JOB_ID"
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "=========================================="

# Load environment
source env/load_modules.sh

# Set environment variables
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME=ib0  # CHANGE THIS

# Define configurations for each array task
# Format: batch_size,precision
declare -a CONFIGS=(
    "64,fp32"
    "64,bf16"
    "128,fp32"
    "128,bf16"
    "256,fp32"
    "256,bf16"
)

# Get configuration for this task
CONFIG=${CONFIGS[$SLURM_ARRAY_TASK_ID]}
IFS=',' read -r BATCH_SIZE PRECISION <<< "$CONFIG"

echo ""
echo "Configuration $SLURM_ARRAY_TASK_ID:"
echo "  Batch size per GPU: $BATCH_SIZE"
echo "  Precision: $PRECISION"
echo "=========================================="
echo ""

# Training parameters
DATA_DIR="./data"
EPOCHS=10
LR=0.1
RESULTS_DIR="./results/csv"
EXP_NAME="sensitivity_bs${BATCH_SIZE}_${PRECISION}_${SLURM_ARRAY_JOB_ID}"

# Build command with precision flag
if [ "$PRECISION" == "bf16" ]; then
    PRECISION_FLAG="--precision bf16"
elif [ "$PRECISION" == "fp16" ]; then
    PRECISION_FLAG="--precision fp16"
else
    PRECISION_FLAG=""
fi

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
    --batch-size $BATCH_SIZE \
    --lr $LR \
    $PRECISION_FLAG \
    --results-dir $RESULTS_DIR \
    --exp-name $EXP_NAME

echo ""
echo "=========================================="
echo "Sensitivity task $SLURM_ARRAY_TASK_ID completed"
echo "Results: $RESULTS_DIR/${EXP_NAME}.csv"
echo "=========================================="

# Save job statistics
sacct -j ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} \
    --format=JobID,Elapsed,AveCPU,MaxRSS,AllocTRES%40,State \
    > results/logs/sacct_sensitivity_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.txt
