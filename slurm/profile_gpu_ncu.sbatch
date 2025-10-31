#!/bin/bash
#SBATCH --job-name=cifar-profile-ncu
#SBATCH --account=YOUR_ACCOUNT          # CHANGE THIS
#SBATCH --partition=gpu                  # CHANGE THIS if needed
#SBATCH --qos=normal
#
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1            # Single GPU for NCU
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#
#SBATCH --time=01:00:00
#SBATCH --output=results/logs/ncu_%j.out
#SBATCH --error=results/logs/ncu_%j.err

# ============================================================================
# CIFAR-10 ResNet-18 Profiling with Nsight Compute
#
# Captures detailed kernel metrics:
# - FLOPs achieved
# - Memory bandwidth
# - Occupancy
# - Warp stalls
#
# NOTE: NCU adds significant overhead, run on single GPU with limited batches
# ============================================================================

echo "=========================================="
echo "Nsight Compute Profiling"
echo "Job ID: $SLURM_JOB_ID"
echo "WARNING: This will be slow due to profiling overhead"
echo "=========================================="

# Load environment
source env/load_modules.sh

# Set environment variables
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Training parameters (very short run for kernel profiling)
DATA_DIR="./data"
EPOCHS=1  # Just 1 epoch
BATCH_SIZE=64  # Smaller batch for faster profiling
LR=0.1
RESULTS_DIR="./results/csv"
EXP_NAME="profile_ncu_${SLURM_JOB_ID}"
NCU_OUTPUT="results/logs/ncu_${SLURM_JOB_ID}"

echo ""
echo "Profiling Configuration:"
echo "  Output: ${NCU_OUTPUT}.ncu-rep"
echo "  Epochs: $EPOCHS (minimal for kernel profiling)"
echo "  Batch size: $BATCH_SIZE (reduced for faster profiling)"
echo "=========================================="
echo ""

# Run with Nsight Compute profiling
# Profile key kernels only to reduce overhead
ncu \
    --set full \
    --target-processes all \
    --kernel-name regex:".*gemm.*|.*conv.*" \
    --launch-skip 10 \
    --launch-count 5 \
    --export ${NCU_OUTPUT} \
    --force-overwrite \
    python src/train.py \
        --data $DATA_DIR \
        --epochs $EPOCHS \
        --batch-size $BATCH_SIZE \
        --lr $LR \
        --results-dir $RESULTS_DIR \
        --exp-name $EXP_NAME \
        --print-freq 1

echo ""
echo "=========================================="
echo "Kernel profiling completed"
echo "Nsight Compute report: ${NCU_OUTPUT}.ncu-rep"
echo ""
echo "To view the report:"
echo "  1. Download ${NCU_OUTPUT}.ncu-rep to your local machine"
echo "  2. Open with Nsight Compute GUI"
echo "  3. Or generate text report: ncu --import ${NCU_OUTPUT}.ncu-rep"
echo ""
echo "Key metrics to look for:"
echo "  - SM efficiency (target: >80%)"
echo "  - Memory throughput"
echo "  - Achieved vs peak FLOPs"
echo "  - Warp stall reasons"
echo "=========================================="

# Save job statistics
sacct -j $SLURM_JOB_ID --format=JobID,Elapsed,AveCPU,MaxRSS,AllocTRES%40,State \
    > results/logs/sacct_ncu_${SLURM_JOB_ID}.txt
