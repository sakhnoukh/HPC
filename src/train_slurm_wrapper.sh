#!/bin/bash
# Wrapper to run train.py with Slurm environment variables mapped to PyTorch DDP variables

# Map Slurm variables to PyTorch DDP variables
export RANK=$SLURM_PROCID
export WORLD_SIZE=$SLURM_NTASKS
export LOCAL_RANK=$SLURM_LOCALID
# MASTER_ADDR and MASTER_PORT should already be set by sbatch script
export MASTER_PORT=${MASTER_PORT:-29500}

echo "DDP Environment (Rank $RANK/$WORLD_SIZE):"
echo "  RANK=$RANK"
echo "  WORLD_SIZE=$WORLD_SIZE"
echo "  LOCAL_RANK=$LOCAL_RANK"
echo "  MASTER_ADDR=$MASTER_ADDR"
echo "  MASTER_PORT=$MASTER_PORT"
echo ""

# Run the training script with all passed arguments
python src/train.py "$@"
