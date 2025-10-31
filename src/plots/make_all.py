#!/usr/bin/env python3
"""
Generate all plots from CSV results.

Usage:
    python src/plots/make_all.py --csv-dir results/csv --output-dir results/plots
"""
import argparse
import glob
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12


def load_all_csvs(csv_dir):
    """Load all CSV files and combine into a single DataFrame."""
    csv_files = glob.glob(os.path.join(csv_dir, "*.csv"))
    
    if not csv_files:
        print(f"Warning: No CSV files found in {csv_dir}")
        return None
    
    print(f"Loading {len(csv_files)} CSV files...")
    dfs = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            dfs.append(df)
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")
    
    if not dfs:
        return None
    
    combined = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(combined)} rows from {len(csv_files)} files")
    return combined


def plot_throughput_vs_gpus(df, output_dir):
    """Plot throughput (images/sec) vs number of GPUs."""
    print("Generating throughput vs GPUs plot...")
    
    # Get final epoch for each run
    df_final = df.groupby(['jobid', 'gpus']).last().reset_index()
    
    # Group by GPU count and compute statistics
    stats = df_final.groupby('gpus')['images_per_sec'].agg(['mean', 'std', 'count'])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot with error bars if multiple runs
    if stats['count'].max() > 1:
        ax.errorbar(stats.index, stats['mean'], yerr=stats['std'], 
                    marker='o', markersize=8, linewidth=2, capsize=5,
                    label='Throughput')
    else:
        ax.plot(stats.index, stats['mean'], marker='o', markersize=8, 
                linewidth=2, label='Throughput')
    
    # Ideal scaling line (from 1 GPU)
    if 1 in stats.index:
        baseline = stats.loc[1, 'mean']
        ideal_gpus = range(1, int(stats.index.max()) + 1)
        ideal_throughput = [baseline * g for g in ideal_gpus]
        ax.plot(ideal_gpus, ideal_throughput, '--', linewidth=2, 
                alpha=0.7, label='Ideal scaling')
    
    ax.set_xlabel('Number of GPUs')
    ax.set_ylabel('Throughput (images/sec)')
    ax.set_title('Training Throughput vs GPU Count')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Set integer x-axis
    ax.set_xticks(stats.index)
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'throughput_vs_gpus.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.savefig(output_file.replace('.png', '.svg'), bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()


def plot_strong_scaling_efficiency(df, output_dir):
    """Plot strong scaling efficiency."""
    print("Generating strong scaling efficiency plot...")
    
    # Get final epoch for each run
    df_final = df.groupby(['jobid', 'gpus']).last().reset_index()
    
    # Group by GPU count
    stats = df_final.groupby('gpus')['epoch_time_s'].agg(['mean', 'std', 'count'])
    
    if 1 not in stats.index:
        print("  Warning: No 1-GPU baseline found, skipping strong scaling plot")
        return
    
    # Calculate efficiency: E_s(N) = T(1) / (N * T(N))
    baseline_time = stats.loc[1, 'mean']
    stats['efficiency'] = (baseline_time / (stats.index * stats['mean'])) * 100
    stats['efficiency_std'] = (baseline_time / (stats.index * stats['mean']**2)) * stats['std'] * 100
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot efficiency
    if stats['count'].max() > 1:
        ax.errorbar(stats.index, stats['efficiency'], yerr=stats['efficiency_std'],
                    marker='o', markersize=8, linewidth=2, capsize=5,
                    label='Strong scaling efficiency')
    else:
        ax.plot(stats.index, stats['efficiency'], marker='o', markersize=8,
                linewidth=2, label='Strong scaling efficiency')
    
    # Reference lines
    ax.axhline(y=100, color='green', linestyle='--', linewidth=2, 
               alpha=0.7, label='Ideal (100%)')
    ax.axhline(y=70, color='orange', linestyle='--', linewidth=2,
               alpha=0.7, label='Target (70%)')
    
    ax.set_xlabel('Number of GPUs')
    ax.set_ylabel('Scaling Efficiency (%)')
    ax.set_title('Strong Scaling Efficiency')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 110)
    ax.set_xticks(stats.index)
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'strong_scaling_efficiency.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.savefig(output_file.replace('.png', '.svg'), bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()


def plot_weak_scaling(df, output_dir):
    """Plot weak scaling (time per epoch vs GPUs)."""
    print("Generating weak scaling plot...")
    
    # Get final epoch for each run
    df_final = df.groupby(['jobid', 'gpus']).last().reset_index()
    
    # Group by GPU count
    stats = df_final.groupby('gpus')['epoch_time_s'].agg(['mean', 'std', 'count'])
    
    if 1 not in stats.index:
        print("  Warning: No 1-GPU baseline found, skipping weak scaling plot")
        return
    
    # Calculate efficiency: E_w(N) = T(1) / T(N)
    baseline_time = stats.loc[1, 'mean']
    stats['efficiency'] = (baseline_time / stats['mean']) * 100
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Time per epoch
    if stats['count'].max() > 1:
        ax1.errorbar(stats.index, stats['mean'], yerr=stats['std'],
                     marker='o', markersize=8, linewidth=2, capsize=5)
    else:
        ax1.plot(stats.index, stats['mean'], marker='o', markersize=8, linewidth=2)
    
    ax1.axhline(y=baseline_time, color='green', linestyle='--', linewidth=2,
                alpha=0.7, label='1-GPU baseline')
    ax1.axhline(y=baseline_time * 1.1, color='orange', linestyle='--', linewidth=2,
                alpha=0.7, label='+10% threshold')
    
    ax1.set_xlabel('Number of GPUs')
    ax1.set_ylabel('Time per Epoch (seconds)')
    ax1.set_title('Weak Scaling: Time per Epoch')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(stats.index)
    
    # Plot 2: Weak scaling efficiency
    ax2.plot(stats.index, stats['efficiency'], marker='o', markersize=8, linewidth=2)
    ax2.axhline(y=100, color='green', linestyle='--', linewidth=2,
                alpha=0.7, label='Ideal (100%)')
    ax2.axhline(y=90, color='orange', linestyle='--', linewidth=2,
                alpha=0.7, label='Target (90%)')
    
    ax2.set_xlabel('Number of GPUs')
    ax2.set_ylabel('Scaling Efficiency (%)')
    ax2.set_title('Weak Scaling Efficiency')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 110)
    ax2.set_xticks(stats.index)
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'weak_scaling.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.savefig(output_file.replace('.png', '.svg'), bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()


def plot_accuracy_curves(df, output_dir):
    """Plot training and validation accuracy curves."""
    print("Generating accuracy curves...")
    
    # Group by GPU count and epoch
    grouped = df.groupby(['gpus', 'epoch']).agg({
        'train_acc': ['mean', 'std'],
        'val_acc': ['mean', 'std']
    }).reset_index()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot for each GPU configuration
    for gpus in sorted(df['gpus'].unique()):
        data = grouped[grouped['gpus'] == gpus]
        
        # Training accuracy
        ax1.plot(data['epoch'], data['train_acc']['mean'], 
                marker='o', label=f'{gpus} GPU(s)', linewidth=2)
        
        # Validation accuracy
        ax2.plot(data['epoch'], data['val_acc']['mean'],
                marker='o', label=f'{gpus} GPU(s)', linewidth=2)
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Accuracy (%)')
    ax1.set_title('Training Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Validation Accuracy (%)')
    ax2.set_title('Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'accuracy_curves.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.savefig(output_file.replace('.png', '.svg'), bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()


def plot_sensitivity_analysis(df, output_dir):
    """Plot sensitivity analysis if multiple batch sizes tested."""
    print("Generating sensitivity analysis plot...")
    
    if 'batch_per_gpu' not in df.columns:
        print("  No batch_per_gpu column found, skipping sensitivity plot")
        return
    
    # Get final epoch for each run
    df_final = df.groupby(['jobid', 'batch_per_gpu', 'gpus']).last().reset_index()
    
    # Group by batch size and GPU count
    stats = df_final.groupby(['batch_per_gpu', 'gpus'])['images_per_sec'].mean().reset_index()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for gpus in sorted(stats['gpus'].unique()):
        data = stats[stats['gpus'] == gpus]
        ax.plot(data['batch_per_gpu'], data['images_per_sec'],
                marker='o', markersize=8, linewidth=2, label=f'{gpus} GPU(s)')
    
    ax.set_xlabel('Batch Size per GPU')
    ax.set_ylabel('Throughput (images/sec)')
    ax.set_title('Throughput vs Batch Size')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'sensitivity_batch_size.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.savefig(output_file.replace('.png', '.svg'), bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()


def generate_summary_table(df, output_dir):
    """Generate summary statistics table."""
    print("Generating summary table...")
    
    # Get final epoch for each run
    df_final = df.groupby(['jobid', 'gpus']).last().reset_index()
    
    # Summary by GPU count
    summary = df_final.groupby('gpus').agg({
        'train_acc': ['mean', 'std'],
        'val_acc': ['mean', 'std'],
        'images_per_sec': ['mean', 'std'],
        'epoch_time_s': ['mean', 'std']
    }).round(2)
    
    # Save as CSV
    output_file = os.path.join(output_dir, 'summary_statistics.csv')
    summary.to_csv(output_file)
    print(f"  Saved: {output_file}")
    
    # Also print to console
    print("\nSummary Statistics:")
    print(summary)
    print()


def main():
    parser = argparse.ArgumentParser(description='Generate plots from CSV results')
    parser.add_argument('--csv-dir', type=str, default='results/csv',
                        help='Directory containing CSV files')
    parser.add_argument('--output-dir', type=str, default='results/plots',
                        help='Directory to save plots')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print("CIFAR-10 Results Plotting")
    print("=" * 60)
    print(f"CSV directory: {args.csv_dir}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 60)
    print()
    
    # Load all CSVs
    df = load_all_csvs(args.csv_dir)
    
    if df is None or len(df) == 0:
        print("No data to plot. Exiting.")
        return
    
    print()
    print("Available columns:", df.columns.tolist())
    print()
    
    # Generate all plots
    try:
        plot_throughput_vs_gpus(df, args.output_dir)
        plot_strong_scaling_efficiency(df, args.output_dir)
        plot_weak_scaling(df, args.output_dir)
        plot_accuracy_curves(df, args.output_dir)
        plot_sensitivity_analysis(df, args.output_dir)
        generate_summary_table(df, args.output_dir)
    except Exception as e:
        print(f"Error generating plots: {e}")
        import traceback
        traceback.print_exc()
    
    print()
    print("=" * 60)
    print("✓ All plots generated successfully!")
    print(f"  Output directory: {args.output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
