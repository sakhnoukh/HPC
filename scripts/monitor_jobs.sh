#!/bin/bash
# Monitor running and completed jobs with detailed status

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "=============================================="
echo "Job Monitoring Dashboard"
echo "=============================================="
echo ""

# Check for submitted jobs file
if [ -f "submitted_jobs.txt" ]; then
    echo "Previously submitted jobs:"
    cat submitted_jobs.txt | grep -v "^=" | grep -v "^$" | grep -v "started at"
    echo ""
fi

# Current queue status
echo "Current Queue Status:"
echo "----------------------------------------------"
squeue -u $USER -o "%.18i %.9P %.30j %.8u %.2t %.10M %.6D %R" 2>/dev/null || {
    echo "No Slurm queue access or no jobs running"
}
echo ""

# Recent jobs (last 24 hours)
echo "Recent Jobs (Last 24 hours):"
echo "----------------------------------------------"
sacct -u $USER --starttime $(date -d '24 hours ago' +%Y-%m-%d) \
      --format=JobID%15,JobName%25,State%12,Elapsed%12,TotalCPU%12 2>/dev/null || {
    echo "No sacct access or no recent jobs"
}
echo ""

# Job statistics summary
echo "Job Summary:"
echo "----------------------------------------------"

running=$(squeue -u $USER -h -t R 2>/dev/null | wc -l)
pending=$(squeue -u $USER -h -t PD 2>/dev/null | wc -l)
completed=$(sacct -u $USER --starttime $(date -d '7 days ago' +%Y-%m-%d) -s COMPLETED -n 2>/dev/null | wc -l)
failed=$(sacct -u $USER --starttime $(date -d '7 days ago' +%Y-%m-%d) -s FAILED -n 2>/dev/null | wc -l)

echo -e "${GREEN}Running:   $running${NC}"
echo -e "${YELLOW}Pending:   $pending${NC}"
echo -e "${GREEN}Completed: $completed (last 7 days)${NC}"
echo -e "${RED}Failed:    $failed (last 7 days)${NC}"
echo ""

# Check for output files
echo "Recent Output Files:"
echo "----------------------------------------------"
if [ -d "results/logs" ]; then
    ls -lht results/logs/*.out 2>/dev/null | head -n 5 | awk '{print $9, "("$6, $7, $8")"}'
else
    echo "No logs directory found"
fi
echo ""

# Check for CSV results
echo "Recent Result Files:"
echo "----------------------------------------------"
if [ -d "results/csv" ]; then
    csv_count=$(ls results/csv/*.csv 2>/dev/null | wc -l)
    echo "Total CSV files: $csv_count"
    if [ $csv_count -gt 0 ]; then
        echo "Latest:"
        ls -lht results/csv/*.csv 2>/dev/null | head -n 3 | awk '{print $9, "("$6, $7, $8")"}'
    fi
else
    echo "No CSV directory found"
fi
echo ""

# Quick analysis if CSV files exist
if [ -d "results/csv" ] && [ $(ls results/csv/*.csv 2>/dev/null | wc -l) -gt 0 ]; then
    echo "Quick Results Preview:"
    echo "----------------------------------------------"
    python3 - << 'EOF' 2>/dev/null || echo "Python analysis unavailable"
import glob
import pandas as pd

csv_files = glob.glob('results/csv/*.csv')
if csv_files:
    try:
        df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
        summary = df.groupby('gpus').agg({
            'val_acc': ['mean', 'max'],
            'images_per_sec': ['mean', 'max']
        }).round(2)
        print("\nBy GPU count:")
        print(summary.to_string())
    except Exception as e:
        print(f"Could not analyze CSV files: {e}")
EOF
    echo ""
fi

echo "=============================================="
echo "Commands:"
echo "  Watch live:     watch -n 5 ./scripts/monitor_jobs.sh"
echo "  Cancel job:     scancel <JOBID>"
echo "  Job details:    scontrol show job <JOBID>"
echo "  View output:    tail -f results/logs/<JOBID>.out"
echo "  Generate plots: python src/plots/make_all.py"
echo "=============================================="
