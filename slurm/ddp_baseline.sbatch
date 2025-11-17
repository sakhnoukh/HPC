#!/bin/bash
#SBATCH --job-name=cifar-ddp-baseline
#SBATCH --partition=gpu-node
#
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#
#SBATCH --time=01:00:00
#SBATCH --output=results/logs/baseline_%j.out
#SBATCH --error=results/logs/baseline_%j.err

# ============================================================================
# CIFAR-10 ResNet-18 Baseline Training (Single Node, 4 GPUs)
# ============================================================================

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "Tasks: $SLURM_NTASKS"
echo "GPUs per node: $SLURM_GPUS_PER_NODE"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "=========================================="

# Load environment (CHOOSE ONE METHOD)
# ============================================================================

# Option A: Apptainer (SELECTED)
export CONTAINER=./env/hpc_pytorch.sif

# Option B: Modules
# source env/load_modules.sh

# Option C: Conda
# module load anaconda3
# source activate hpc-cifar10

# ============================================================================

# Set environment variables
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME=eth0

# Print environment info
echo ""
echo "Environment:"
apptainer exec --nvccli $CONTAINER python --version
apptainer exec --nvccli $CONTAINER python -c "import torch; print(f'PyTorch: {torch.__version__}')"
apptainer exec --nvccli $CONTAINER python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
apptainer exec --nvccli $CONTAINER python -c "import torch; print(f'GPUs visible: {torch.cuda.device_count()}')"
echo ""

# Training parameters
DATA_DIR="./data"
EPOCHS=10
BATCH_SIZE=128  # Per GPU
LR=0.1
RESULTS_DIR="./results/csv"
EXP_NAME="baseline_${SLURM_JOB_ID}"

echo "Training Configuration:"
echo "  Data: $DATA_DIR"
echo "  Epochs: $EPOCHS"
echo "  Batch size per GPU: $BATCH_SIZE"
echo "  Global batch size: $((BATCH_SIZE * SLURM_NTASKS))"
echo "  Learning rate: $LR"
echo "  Results: $RESULTS_DIR/$EXP_NAME"
echo "=========================================="
echo ""

# Run training
# ============================================================================

# With Apptainer
srun apptainer exec --nvccli $CONTAINER \
    python -m torch.distributed.run \
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

# ============================================================================

echo ""
echo "=========================================="
echo "Job completed at: $(date)"
echo "=========================================="

# Save job statistics
sacct -j $SLURM_JOB_ID --format=JobID,Elapsed,AveCPU,AveRSS,MaxRSS,AllocTRES%40,State \
    > results/logs/sacct_${SLURM_JOB_ID}.txt

echo "Job statistics saved to: results/logs/sacct_${SLURM_JOB_ID}.txt"
