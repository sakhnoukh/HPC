#!/bin/bash
# Run all scaling experiments
# This script submits all jobs for baseline, strong scaling, and weak scaling

set -e  # Exit on error

echo "=============================================="
echo "HPC CIFAR-10 Experiment Suite"
echo "=============================================="
echo ""

# Check if we're on a Slurm system
if ! command -v sbatch &> /dev/null; then
    echo "ERROR: sbatch not found. Are you on a Slurm cluster?"
    exit 1
fi

# Configuration
JOBS_FILE="submitted_jobs.txt"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Create job tracking file
echo "Experiment run started at: $(date)" > $JOBS_FILE
echo "========================================" >> $JOBS_FILE
echo "" >> $JOBS_FILE

# Function to submit and track job
submit_job() {
    local script=$1
    local description=$2
    
    echo "Submitting: $description"
    echo "  Script: $script"
    
    if [ ! -f "$script" ]; then
        echo "  ERROR: Script not found: $script"
        return 1
    fi
    
    # Submit job and capture ID
    job_output=$(sbatch "$script" 2>&1)
    
    if [ $? -eq 0 ]; then
        # Extract job ID
        job_id=$(echo "$job_output" | grep -oP "Submitted batch job \K\d+")
        echo "  ✓ Submitted: Job ID $job_id"
        echo "$description: $job_id" >> $JOBS_FILE
    else
        echo "  ✗ Failed: $job_output"
        echo "$description: FAILED" >> $JOBS_FILE
    fi
    
    echo ""
}

# Ask user what to run
echo "Select experiments to run:"
echo "  1) Baseline only (1 node, 4 GPUs)"
echo "  2) Baseline + Strong scaling (1, 2, 4, 8 GPUs)"
echo "  3) Baseline + Weak scaling (1, 2, 4, 8 GPUs)"
echo "  4) All experiments (baseline + strong + weak + sensitivity)"
echo "  5) Profiling only (Nsight Systems + Compute)"
echo "  6) Custom selection"
echo ""
read -p "Enter choice [1-6]: " choice

case $choice in
    1)
        echo "Running: Baseline only"
        echo ""
        submit_job "slurm/ddp_baseline.sbatch" "Baseline (1 node, 4 GPUs)"
        ;;
    
    2)
        echo "Running: Baseline + Strong scaling"
        echo ""
        submit_job "slurm/ddp_baseline.sbatch" "Baseline (1 node, 4 GPUs)"
        sleep 2
        submit_job "slurm/ddp_strong_scaling.sbatch" "Strong scaling (2 nodes, 8 GPUs)"
        ;;
    
    3)
        echo "Running: Baseline + Weak scaling"
        echo ""
        submit_job "slurm/ddp_baseline.sbatch" "Baseline (1 node, 4 GPUs)"
        sleep 2
        submit_job "slurm/ddp_weak_scaling.sbatch" "Weak scaling (2 nodes, 8 GPUs)"
        ;;
    
    4)
        echo "Running: All experiments"
        echo ""
        submit_job "slurm/ddp_baseline.sbatch" "Baseline (1 node, 4 GPUs)"
        sleep 2
        submit_job "slurm/ddp_strong_scaling.sbatch" "Strong scaling (2 nodes)"
        sleep 2
        submit_job "slurm/ddp_weak_scaling.sbatch" "Weak scaling (2 nodes)"
        sleep 2
        submit_job "slurm/ddp_sensitivity.sbatch" "Sensitivity analysis (job array)"
        ;;
    
    5)
        echo "Running: Profiling only"
        echo ""
        submit_job "slurm/profile_gpu_nsys.sbatch" "Nsight Systems profiling"
        sleep 2
        submit_job "slurm/profile_gpu_ncu.sbatch" "Nsight Compute profiling"
        ;;
    
    6)
        echo "Custom selection:"
        read -p "Run baseline? [y/n]: " run_baseline
        read -p "Run strong scaling? [y/n]: " run_strong
        read -p "Run weak scaling? [y/n]: " run_weak
        read -p "Run sensitivity? [y/n]: " run_sens
        read -p "Run profiling? [y/n]: " run_prof
        
        echo ""
        [ "$run_baseline" = "y" ] && submit_job "slurm/ddp_baseline.sbatch" "Baseline"
        [ "$run_strong" = "y" ] && submit_job "slurm/ddp_strong_scaling.sbatch" "Strong scaling"
        [ "$run_weak" = "y" ] && submit_job "slurm/ddp_weak_scaling.sbatch" "Weak scaling"
        [ "$run_sens" = "y" ] && submit_job "slurm/ddp_sensitivity.sbatch" "Sensitivity"
        [ "$run_prof" = "y" ] && {
            submit_job "slurm/profile_gpu_nsys.sbatch" "Nsight Systems"
            submit_job "slurm/profile_gpu_ncu.sbatch" "Nsight Compute"
        }
        ;;
    
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac

echo "=============================================="
echo "Submission complete!"
echo ""
echo "Job IDs saved to: $JOBS_FILE"
echo ""
echo "Monitor jobs with:"
echo "  squeue -u \$USER"
echo "  ./scripts/monitor_jobs.sh"
echo ""
echo "View results:"
echo "  tail -f results/logs/*.out"
echo "=============================================="
