# Plotting Utilities

Scripts for generating figures from CSV results.

## Usage

```bash
# Generate all plots from CSV files
python src/plots/make_all.py --csv-dir results/csv --output-dir results/plots
```

## Generated Plots

### 1. Throughput vs GPUs
**File:** `throughput_vs_gpus.png`  
Shows training throughput (images/sec) as a function of GPU count, with ideal scaling reference line.

### 2. Strong Scaling Efficiency
**File:** `strong_scaling_efficiency.png`  
Efficiency percentage: E_s(N) = T(1) / (N × T(N)) × 100%  
Target: ≥70% at 8 GPUs.

### 3. Weak Scaling
**File:** `weak_scaling.png`  
Two panels:
- Time per epoch vs GPU count (should remain constant)
- Weak scaling efficiency: E_w(N) = T(1) / T(N) × 100%

### 4. Accuracy Curves
**File:** `accuracy_curves.png`  
Training and validation accuracy over epochs for different GPU counts.

### 5. Sensitivity Analysis
**File:** `sensitivity_batch_size.png`  
Throughput vs batch size for different GPU configurations.

### 6. Summary Statistics
**File:** `summary_statistics.csv`  
Table with mean and standard deviation of key metrics grouped by GPU count.

## Output Formats

- **PNG**: High-resolution (300 DPI) for papers
- **SVG**: Vector format for presentations and further editing

## Requirements

```bash
pip install matplotlib seaborn pandas
```

## Customization

Edit `make_all.py` to:
- Change plot style (line 15-17)
- Modify color schemes
- Add new plot types
- Adjust figure dimensions
