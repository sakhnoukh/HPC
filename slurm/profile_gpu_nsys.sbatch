#!/bin/bash
#SBATCH --job-name=cifar-profile-nsys
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
#SBATCH --time=00:30:00
#SBATCH --output=results/logs/nsys_%j.out
#SBATCH --error=results/logs/nsys_%j.err

# ============================================================================
# CIFAR-10 ResNet-18 Profiling with Nsight Systems
#
# Captures timeline of:
# - CUDA kernels
# - NCCL communication
# - CPU activity
# - Memory transfers
# ============================================================================

echo "=========================================="
echo "Nsight Systems Profiling"
echo "Job ID: $SLURM_JOB_ID"
echo "=========================================="

# Load environment
source env/load_modules.sh

# Set environment variables
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NCCL_DEBUG=INFO  # More verbose for profiling
export NCCL_SOCKET_IFNAME=ib0  # CHANGE THIS

# Training parameters (short run for profiling)
DATA_DIR="./data"
EPOCHS=2  # Just a few epochs for profiling
BATCH_SIZE=128
LR=0.1
RESULTS_DIR="./results/csv"
EXP_NAME="profile_nsys_${SLURM_JOB_ID}"
NSYS_OUTPUT="results/logs/nsys_${SLURM_JOB_ID}"

echo ""
echo "Profiling Configuration:"
echo "  Output: ${NSYS_OUTPUT}.nsys-rep"
echo "  Epochs: $EPOCHS (short for profiling)"
echo "=========================================="
echo ""

# Run with Nsight Systems profiling
nsys profile \
    --output=${NSYS_OUTPUT} \
    --trace=cuda,nvtx,osrt,cudnn,cublas \
    --cuda-memory-usage=true \
    --force-overwrite=true \
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
        --results-dir $RESULTS_DIR \
        --exp-name $EXP_NAME

echo ""
echo "=========================================="
echo "Profiling completed"
echo "Nsight Systems report: ${NSYS_OUTPUT}.nsys-rep"
echo ""
echo "To view the report:"
echo "  1. Download ${NSYS_OUTPUT}.nsys-rep to your local machine"
echo "  2. Open with Nsight Systems GUI"
echo "  3. Or generate text report: nsys stats ${NSYS_OUTPUT}.nsys-rep"
echo "=========================================="

# Generate text statistics
if command -v nsys &> /dev/null; then
    nsys stats ${NSYS_OUTPUT}.nsys-rep > ${NSYS_OUTPUT}_stats.txt
    echo "Text statistics: ${NSYS_OUTPUT}_stats.txt"
fi

# Save job statistics
sacct -j $SLURM_JOB_ID --format=JobID,Elapsed,AveCPU,MaxRSS,AllocTRES%40,State \
    > results/logs/sacct_nsys_${SLURM_JOB_ID}.txt
