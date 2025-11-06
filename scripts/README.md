# Helper Scripts

Utility scripts for experiment management, monitoring, and analysis.

## Available Scripts

### 1. `run_all_experiments.sh`
**Purpose:** Interactive script to submit all experiments  
**Usage:**
```bash
chmod +x scripts/run_all_experiments.sh
./scripts/run_all_experiments.sh
```

**Features:**
- Interactive menu for selecting experiments
- Automatic job tracking
- Saves job IDs to `submitted_jobs.txt`

**Options:**
1. Baseline only (1 node, 4 GPUs)
2. Baseline + Strong scaling
3. Baseline + Weak scaling
4. All experiments (baseline + strong + weak + sensitivity)
5. Profiling only
6. Custom selection

### 2. `monitor_jobs.sh`
**Purpose:** Real-time job monitoring dashboard  
**Usage:**
```bash
chmod +x scripts/monitor_jobs.sh
./scripts/monitor_jobs.sh

# Watch mode (updates every 5 seconds)
watch -n 5 ./scripts/monitor_jobs.sh
```

**Shows:**
- Current queue status
- Recent jobs (last 24 hours)
- Job summary (running/pending/completed/failed)
- Recent output and result files
- Quick results preview

### 3. `quick_analysis.py`
**Purpose:** Fast analysis of experimental results  
**Usage:**
```bash
python scripts/quick_analysis.py
python scripts/quick_analysis.py --csv-dir results/csv
```

**Outputs:**
- Summary statistics by GPU count
- Strong and weak scaling efficiency
- Convergence analysis
- Target verification (≥70% @ 8 GPUs, ≥70% accuracy)

### 4. `benchmark.py`
**Purpose:** Benchmark single-GPU performance  
**Usage:**
```bash
python scripts/benchmark.py
python scripts/benchmark.py --batch-size 256 --iterations 200
```

**Benchmarks:**
- FP32 forward/backward pass timing
- BF16 performance (if supported)
- FP16 performance with AMP
- DataLoader throughput
- Speedup calculations

**Use Cases:**
- Establish baseline before scaling experiments
- Compare performance across different GPUs
- Verify mixed precision speedup
- Identify optimal precision mode

## Typical Workflow

### Initial Setup
```bash
# 1. Make scripts executable
chmod +x scripts/*.sh

# 2. Run benchmark to establish baseline
python scripts/benchmark.py

# 3. Test environment
python test_setup.py
```

### Running Experiments
```bash
# 1. Submit experiments
./scripts/run_all_experiments.sh
# Choose option (e.g., 4 for all experiments)

# 2. Monitor progress
watch -n 5 ./scripts/monitor_jobs.sh

# 3. Quick analysis of results
python scripts/quick_analysis.py
```

### After Experiments
```bash
# 1. Generate plots
python src/plots/make_all.py

# 2. Review results
ls -lh results/plots/
ls -lh results/csv/
```

## Output Files

### `submitted_jobs.txt`
Created by `run_all_experiments.sh`. Contains:
- Timestamp of submission
- Job descriptions and IDs
- Useful for tracking which jobs were submitted

**Example:**
```
Experiment run started at: Fri Nov  6 20:00:00 CET 2025
========================================

Baseline (1 node, 4 GPUs): 12345
Strong scaling (2 nodes): 12346
Weak scaling (2 nodes): 12347
```

## Requirements

### Shell Scripts
- Bash
- Slurm commands (`squeue`, `sacct`, `sbatch`, `scontrol`)
- Standard Unix utilities (`grep`, `awk`, `sed`)

### Python Scripts
```bash
pip install pandas numpy matplotlib seaborn torch torchvision
```

## Tips & Best Practices

### Monitoring
- Use `watch -n 5 ./scripts/monitor_jobs.sh` for live updates
- Check `submitted_jobs.txt` to track job IDs
- Tail log files for detailed progress: `tail -f results/logs/*.out`

### Analysis
- Run `quick_analysis.py` after each major experiment
- Generate plots regularly to track progress
- Keep CSV files for reproducibility

### Benchmarking
- Run benchmark on each new GPU model
- Compare FP32 vs BF16/FP16 speedup
- Use results to choose optimal precision

### Debugging
- Check `monitor_jobs.sh` for failed jobs
- Review `.err` files in `results/logs/`
- Verify environment with `test_setup.py`

## Troubleshooting

### Script Permission Denied
```bash
chmod +x scripts/*.sh
```

### Python Module Not Found
```bash
pip install -r env/requirements.txt
# OR
source env/load_modules.sh
```

### No Jobs Showing in Monitor
```bash
# Check if you're on Slurm system
which squeue

# Check if jobs are running
squeue -u $USER
```

### Quick Analysis Fails
```bash
# Verify CSV files exist
ls results/csv/*.csv

# Check CSV format
head results/csv/*.csv
```

## Advanced Usage

### Custom Benchmark Parameters
```bash
# Larger batch size
python scripts/benchmark.py --batch-size 512 --iterations 50

# Different data path
python scripts/benchmark.py --data /scratch/datasets/cifar10
```

### Selective Job Submission
```bash
# Edit run_all_experiments.sh to customize
# Option 6 allows custom selection of experiments
./scripts/run_all_experiments.sh
# Choose 6, then select which experiments to run
```

### Automated Analysis
```bash
# Run analysis in a loop
while true; do
    python scripts/quick_analysis.py
    sleep 300  # Every 5 minutes
done
```

## Integration with Other Tools

### With Plotting
```bash
# After quick analysis, generate plots
python scripts/quick_analysis.py && python src/plots/make_all.py
```

### With Git
```bash
# Track submission history
git add submitted_jobs.txt
git commit -m "Submitted scaling experiments"
```

### With Notebooks
```python
# In Jupyter notebook
import pandas as pd
import subprocess

# Run quick analysis
subprocess.run(['python', 'scripts/quick_analysis.py'])

# Load results
df = pd.concat([pd.read_csv(f) for f in glob.glob('results/csv/*.csv')])
```

---

**For more information, see:**
- `CHEATSHEET.md` - Quick reference for common commands
- `reproduce.md` - Complete reproduction guide
- `README.md` - Project overview
