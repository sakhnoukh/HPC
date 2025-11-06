#!/usr/bin/env python3
"""
Quick analysis of experimental results.

Usage:
    python scripts/quick_analysis.py
    python scripts/quick_analysis.py --csv-dir results/csv
"""
import argparse
import glob
import os
import sys

import pandas as pd


def load_results(csv_dir):
    """Load all CSV files."""
    csv_files = glob.glob(os.path.join(csv_dir, "*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {csv_dir}")
        return None
    
    print(f"Loading {len(csv_files)} CSV files...")
    dfs = []
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except Exception as e:
            print(f"Warning: Could not load {f}: {e}")
    
    if not dfs:
        return None
    
    combined = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(combined)} rows\n")
    return combined


def print_summary(df):
    """Print summary statistics."""
    print("=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    
    # Group by GPU count
    final_epochs = df.groupby(['jobid', 'gpus']).last().reset_index()
    
    summary = final_epochs.groupby('gpus').agg({
        'train_acc': ['mean', 'std', 'max'],
        'val_acc': ['mean', 'std', 'max'],
        'images_per_sec': ['mean', 'std', 'max'],
        'epoch_time_s': ['mean', 'std', 'min']
    }).round(2)
    
    print("\nBy GPU Count:")
    print(summary)
    print()


def analyze_scaling(df):
    """Analyze strong and weak scaling."""
    print("=" * 70)
    print("SCALING ANALYSIS")
    print("=" * 70)
    
    # Get final epoch for each run
    final_epochs = df.groupby(['jobid', 'gpus']).last().reset_index()
    
    # Group by GPU count
    grouped = final_epochs.groupby('gpus').agg({
        'epoch_time_s': 'mean',
        'images_per_sec': 'mean'
    }).sort_index()
    
    if 1 not in grouped.index:
        print("No 1-GPU baseline found. Cannot compute scaling efficiency.\n")
        return
    
    baseline_time = grouped.loc[1, 'epoch_time_s']
    baseline_throughput = grouped.loc[1, 'images_per_sec']
    
    print("\nStrong Scaling Efficiency:")
    print("-" * 70)
    print(f"{'GPUs':<8} {'Time(s)':<12} {'Speedup':<12} {'Efficiency(%)':<15}")
    print("-" * 70)
    
    for gpus in sorted(grouped.index):
        time = grouped.loc[gpus, 'epoch_time_s']
        speedup = baseline_time / time
        efficiency = (speedup / gpus) * 100
        
        print(f"{gpus:<8} {time:<12.2f} {speedup:<12.2f} {efficiency:<15.1f}")
    
    print()
    print("\nThroughput Scaling:")
    print("-" * 70)
    print(f"{'GPUs':<8} {'Throughput':<15} {'Ideal':<15} {'Efficiency(%)':<15}")
    print("-" * 70)
    
    for gpus in sorted(grouped.index):
        throughput = grouped.loc[gpus, 'images_per_sec']
        ideal = baseline_throughput * gpus
        efficiency = (throughput / ideal) * 100
        
        print(f"{gpus:<8} {throughput:<15.1f} {ideal:<15.1f} {efficiency:<15.1f}")
    
    print()


def analyze_convergence(df):
    """Analyze training convergence."""
    print("=" * 70)
    print("CONVERGENCE ANALYSIS")
    print("=" * 70)
    
    # Group by GPU count and epoch
    convergence = df.groupby(['gpus', 'epoch']).agg({
        'train_acc': 'mean',
        'val_acc': 'mean',
        'train_loss': 'mean'
    }).reset_index()
    
    print("\nFinal Epoch Accuracy by GPU Count:")
    print("-" * 70)
    
    for gpus in sorted(df['gpus'].unique()):
        gpu_data = convergence[convergence['gpus'] == gpus]
        if len(gpu_data) > 0:
            final = gpu_data.iloc[-1]
            print(f"{gpus} GPU(s): Train={final['train_acc']:.2f}%, Val={final['val_acc']:.2f}%")
    
    print()


def check_targets(df):
    """Check if targets are met."""
    print("=" * 70)
    print("TARGET VERIFICATION")
    print("=" * 70)
    
    final_epochs = df.groupby(['jobid', 'gpus']).last().reset_index()
    grouped = final_epochs.groupby('gpus').agg({
        'epoch_time_s': 'mean',
        'images_per_sec': 'mean',
        'val_acc': 'mean'
    }).sort_index()
    
    if 1 not in grouped.index:
        print("No 1-GPU baseline found.\n")
        return
    
    baseline_time = grouped.loc[1, 'epoch_time_s']
    
    print("\nTargets:")
    print("-" * 70)
    
    # Strong scaling efficiency target
    if 8 in grouped.index:
        time_8gpu = grouped.loc[8, 'epoch_time_s']
        efficiency_8gpu = (baseline_time / (8 * time_8gpu)) * 100
        target_met = "✓" if efficiency_8gpu >= 70 else "✗"
        print(f"{target_met} Strong scaling @ 8 GPUs: {efficiency_8gpu:.1f}% (target: ≥70%)")
    else:
        print("- Strong scaling @ 8 GPUs: Not tested yet")
    
    # Accuracy target
    if len(grouped) > 0:
        final_acc = grouped['val_acc'].iloc[-1]
        target_met = "✓" if final_acc >= 70 else "✗"
        print(f"{target_met} Validation accuracy: {final_acc:.2f}% (target: ≥70%)")
    
    print()


def main():
    parser = argparse.ArgumentParser(description='Quick analysis of results')
    parser.add_argument('--csv-dir', type=str, default='results/csv',
                        help='Directory containing CSV files')
    args = parser.parse_args()
    
    # Load data
    df = load_results(args.csv_dir)
    
    if df is None:
        print("\nNo data to analyze. Run some experiments first!")
        print("  sbatch slurm/ddp_baseline.sbatch")
        return 1
    
    # Run analyses
    try:
        print_summary(df)
        analyze_scaling(df)
        analyze_convergence(df)
        check_targets(df)
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("=" * 70)
    print("Analysis complete!")
    print("\nTo generate plots:")
    print("  python src/plots/make_all.py")
    print("=" * 70)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
