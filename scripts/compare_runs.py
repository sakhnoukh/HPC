#!/usr/bin/env python3
"""
Compare multiple experimental runs.

Usage:
    python scripts/compare_runs.py run1.csv run2.csv
    python scripts/compare_runs.py --dir results/csv --pattern "baseline_*"
"""
import argparse
import glob
import os
import sys

import pandas as pd


def load_run(filepath):
    """Load a single run from CSV."""
    try:
        df = pd.read_csv(filepath)
        run_name = os.path.basename(filepath).replace('.csv', '')
        return run_name, df
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None, None


def compare_runs(runs):
    """Compare multiple runs."""
    if not runs:
        print("No runs to compare")
        return
    
    print("=" * 80)
    print("RUN COMPARISON")
    print("=" * 80)
    print()
    
    # Summary table
    print("Summary:")
    print("-" * 80)
    print(f"{'Run':<30} {'GPUs':<8} {'Final Acc':<12} {'Throughput':<15} {'Time/Epoch':<12}")
    print("-" * 80)
    
    for name, df in runs:
        if df is None or len(df) == 0:
            continue
        
        # Get final epoch
        final = df.iloc[-1]
        gpus = final.get('gpus', final.get('world_size', '?'))
        val_acc = final.get('val_acc', 0)
        throughput = final.get('images_per_sec', 0)
        epoch_time = final.get('epoch_time_s', 0)
        
        print(f"{name:<30} {gpus:<8} {val_acc:<12.2f} {throughput:<15.0f} {epoch_time:<12.2f}")
    
    print()
    
    # Convergence comparison
    print("Convergence Comparison:")
    print("-" * 80)
    
    # Find common epochs
    all_epochs = set()
    for name, df in runs:
        if df is not None:
            all_epochs.update(df['epoch'].unique())
    
    common_epochs = sorted(all_epochs)
    if len(common_epochs) > 0:
        # Show every Nth epoch to keep output manageable
        step = max(1, len(common_epochs) // 10)
        display_epochs = common_epochs[::step]
        
        print(f"{'Epoch':<10}", end='')
        for name, _ in runs:
            print(f"{name:<20}", end='')
        print()
        print("-" * 80)
        
        for epoch in display_epochs:
            print(f"{epoch:<10}", end='')
            for name, df in runs:
                if df is not None:
                    epoch_data = df[df['epoch'] == epoch]
                    if len(epoch_data) > 0:
                        val_acc = epoch_data.iloc[0].get('val_acc', 0)
                        print(f"{val_acc:<20.2f}", end='')
                    else:
                        print(f"{'N/A':<20}", end='')
                else:
                    print(f"{'N/A':<20}", end='')
            print()
    
    print()
    
    # Performance comparison
    print("Performance Metrics:")
    print("-" * 80)
    
    metrics = ['train_acc', 'val_acc', 'images_per_sec', 'epoch_time_s']
    
    for metric in metrics:
        print(f"\n{metric}:")
        for name, df in runs:
            if df is None or metric not in df.columns:
                continue
            
            final_value = df.iloc[-1][metric]
            mean_value = df[metric].mean()
            max_value = df[metric].max()
            
            print(f"  {name:<28} final={final_value:<10.2f} mean={mean_value:<10.2f} max={max_value:<10.2f}")
    
    print()
    
    # Scaling comparison (if different GPU counts)
    gpu_counts = []
    for name, df in runs:
        if df is not None and 'gpus' in df.columns:
            gpus = df.iloc[-1]['gpus']
            gpu_counts.append((name, gpus, df.iloc[-1]['images_per_sec']))
    
    if len(set([g for _, g, _ in gpu_counts])) > 1:
        print("Scaling Analysis:")
        print("-" * 80)
        
        # Sort by GPU count
        gpu_counts.sort(key=lambda x: x[1])
        
        if len(gpu_counts) >= 2 and gpu_counts[0][1] == 1:
            baseline_throughput = gpu_counts[0][2]
            print(f"{'GPUs':<10} {'Run':<30} {'Throughput':<15} {'Speedup':<10} {'Efficiency':<12}")
            print("-" * 80)
            
            for name, gpus, throughput in gpu_counts:
                speedup = throughput / baseline_throughput
                efficiency = (speedup / gpus) * 100
                print(f"{gpus:<10} {name:<30} {throughput:<15.0f} {speedup:<10.2f} {efficiency:<12.1f}%")
        else:
            print(f"{'GPUs':<10} {'Run':<30} {'Throughput':<15}")
            print("-" * 80)
            for name, gpus, throughput in gpu_counts:
                print(f"{gpus:<10} {name:<30} {throughput:<15.0f}")
        
        print()


def main():
    parser = argparse.ArgumentParser(description='Compare experimental runs')
    parser.add_argument('files', nargs='*', help='CSV files to compare')
    parser.add_argument('--dir', type=str, help='Directory containing CSV files')
    parser.add_argument('--pattern', type=str, help='Glob pattern for files (e.g., "baseline_*")')
    args = parser.parse_args()
    
    # Collect files to compare
    files_to_compare = []
    
    if args.files:
        files_to_compare.extend(args.files)
    
    if args.dir and args.pattern:
        pattern_path = os.path.join(args.dir, args.pattern + '.csv')
        matched_files = glob.glob(pattern_path)
        files_to_compare.extend(matched_files)
    elif args.dir:
        csv_files = glob.glob(os.path.join(args.dir, '*.csv'))
        files_to_compare.extend(csv_files)
    
    if not files_to_compare:
        print("No files to compare. Usage:")
        print("  python scripts/compare_runs.py file1.csv file2.csv")
        print("  python scripts/compare_runs.py --dir results/csv")
        print("  python scripts/compare_runs.py --dir results/csv --pattern 'baseline_*'")
        return 1
    
    # Load all runs
    runs = []
    for filepath in files_to_compare:
        name, df = load_run(filepath)
        if name is not None:
            runs.append((name, df))
    
    if not runs:
        print("No valid runs loaded")
        return 1
    
    print(f"Comparing {len(runs)} runs:\n")
    for name, _ in runs:
        print(f"  - {name}")
    print()
    
    # Compare
    compare_runs(runs)
    
    print("=" * 80)
    print("Comparison complete!")
    print("=" * 80)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
