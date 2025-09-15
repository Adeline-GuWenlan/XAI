#!/usr/bin/env python3
"""
Batch generation of datasets with different lambda values.
Runs generate_stabilizer_dataset.py with lambda values from 1 to 35 (step 5).
Creates detailed reports and distribution plots for each lambda.
"""

import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

def run_dataset_generation(lam, num_samples, qubits, output_dir):
    """
    Run generate_stabilizer_dataset.py with specified parameters.
    """
    script_path = "/Users/guwenlan/Desktop/XAI/Utils/generate_stabilizer_dataset.py"
    cmd = [
        "python", script_path,
        "--num", str(num_samples),
        "--qubits", str(qubits), 
        "--lam", str(lam),
        "--output", output_dir,
        "--seed", "42"  # Fixed seed for reproducibility
    ]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error running dataset generation for lambda={lam}:")
        print(result.stderr)
        return False
    
    print(result.stdout)
    return True

def analyze_and_plot(lam, output_dir, qubits, num_samples):
    """
    Load the generated dataset, analyze distribution, and create plots.
    """
    # Load data
    labels_file = os.path.join(output_dir, f"{qubits}q_{num_samples}_labels.npy")
    states_file = os.path.join(output_dir, f"{qubits}q_{num_samples}_states.npy")
    
    if not os.path.exists(labels_file):
        print(f"Warning: Labels file not found: {labels_file}")
        return
    
    labels = np.load(labels_file)
    states = np.load(states_file)
    
    # Basic statistics
    stats = {
        'lambda': lam,
        'total_samples': len(labels),
        'min_label': np.min(labels),
        'max_label': np.max(labels),
        'mean_label': np.mean(labels),
        'std_label': np.std(labels),
        'median_label': np.median(labels),
        'q25_label': np.percentile(labels, 25),
        'q75_label': np.percentile(labels, 75),
        'below_1': np.sum(labels < 1.0),
        'above_equal_1': np.sum(labels >= 1.0),
        'below_1_percent': np.sum(labels < 1.0) / len(labels) * 100,
        'above_equal_1_percent': np.sum(labels >= 1.0) / len(labels) * 100
    }
    
    # Write detailed report
    report_file = os.path.join(output_dir, "analysis_report.txt")
    with open(report_file, 'w') as f:
        f.write(f"Dataset Analysis Report\n")
        f.write(f"=====================\n\n")
        f.write(f"Parameters:\n")
        f.write(f"  Lambda (λ): {stats['lambda']}\n")
        f.write(f"  Qubits: {qubits}\n")
        f.write(f"  Samples: {stats['total_samples']}\n")
        f.write(f"  Seed: 42\n\n")
        
        f.write(f"Label Distribution Statistics:\n")
        f.write(f"  Range: [{stats['min_label']:.6f}, {stats['max_label']:.6f}]\n")
        f.write(f"  Mean: {stats['mean_label']:.6f}\n")
        f.write(f"  Std: {stats['std_label']:.6f}\n")
        f.write(f"  Median: {stats['median_label']:.6f}\n")
        f.write(f"  Q25: {stats['q25_label']:.6f}\n")
        f.write(f"  Q75: {stats['q75_label']:.6f}\n\n")
        
        f.write(f"Magic vs Non-Magic Classification:\n")
        f.write(f"  Non-Magic (SN < 1.0): {stats['below_1']} ({stats['below_1_percent']:.2f}%)\n")
        f.write(f"  Magic (SN >= 1.0): {stats['above_equal_1']} ({stats['above_equal_1_percent']:.2f}%)\n\n")
        
        f.write(f"State Properties:\n")
        f.write(f"  States shape: {states.shape}\n")
        f.write(f"  States dtype: {states.dtype}\n")
        
        # Additional analysis
        f.write(f"\nAdditional Statistics:\n")
        f.write(f"  Labels > 2.0: {np.sum(labels > 2.0)} ({np.sum(labels > 2.0)/len(labels)*100:.2f}%)\n")
        f.write(f"  Labels > 3.0: {np.sum(labels > 3.0)} ({np.sum(labels > 3.0)/len(labels)*100:.2f}%)\n")
        
        # Bins analysis
        hist, bin_edges = np.histogram(labels, bins=20)
        f.write(f"\nHistogram Analysis (20 bins):\n")
        for i in range(len(hist)):
            f.write(f"  [{bin_edges[i]:.3f}, {bin_edges[i+1]:.3f}): {hist[i]} samples\n")
    
    # Create distribution plot - only histogram
    plt.figure(figsize=(8, 6))
    
    plt.hist(labels, bins=50, alpha=0.7, edgecolor='black')
    plt.axvline(x=1.0, color='red', linestyle='--', linewidth=2, label='Magic threshold (SN=1)')
    plt.xlabel('Stabilizer Norm')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of SN (λ={lam})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plot_file = os.path.join(output_dir, "distribution_analysis.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Analysis complete for λ={lam}")
    print(f"  Report saved: {report_file}")
    print(f"  Plot saved: {plot_file}")
    
    return stats

def main():
    parser = argparse.ArgumentParser(description='Batch generate datasets with different lambda values')
    parser.add_argument('--num', type=int, default=1000, help='Number of samples per dataset')
    parser.add_argument('--qubits', type=int, default=2, help='Number of qubits')
    parser.add_argument('--start_lam', type=float, default=0.1, help='Starting lambda value')
    parser.add_argument('--end_lam', type=float, default=2.0, help='Ending lambda value')
    parser.add_argument('--step_lam', type=float, default=0.1, help='Lambda step size')
    
    args = parser.parse_args()
    
    # Create main results directory
    base_dir = "/Users/guwenlan/Desktop/XAI/Utils"
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # Generate lambda values
    lambda_values = np.arange(args.start_lam, args.end_lam + args.step_lam, args.step_lam)
    
    print(f"Running batch generation for lambda values: {lambda_values}")
    print(f"Samples per dataset: {args.num}")
    print(f"Qubits: {args.qubits}")
    print(f"Results directory: {results_dir}")
    
    all_stats = []
    
    for lam in lambda_values:
        print(f"\n{'='*50}")
        print(f"Processing λ = {lam}")
        print(f"{'='*50}")
        
        # Create subdirectory for this lambda
        lam_dir = os.path.join(results_dir, f"lambda_{lam}")
        os.makedirs(lam_dir, exist_ok=True)
        
        # Run dataset generation
        success = run_dataset_generation(lam, args.num, args.qubits, lam_dir)
        
        if success:
            # Analyze and plot
            stats = analyze_and_plot(lam, lam_dir, args.qubits, args.num)
            all_stats.append(stats)
        else:
            print(f"Skipping analysis for λ={lam} due to generation failure")
    
    # Create summary report
    if all_stats:
        summary_file = os.path.join(results_dir, "summary_report.txt")
        with open(summary_file, 'w') as f:
            f.write("Summary Report: Lambda Variation Study\n")
            f.write("=====================================\n\n")
            f.write(f"Parameters:\n")
            f.write(f"  Samples per dataset: {args.num}\n")
            f.write(f"  Qubits: {args.qubits}\n")
            f.write(f"  Lambda range: {args.start_lam} to {args.end_lam} (step {args.step_lam})\n")
            f.write(f"  Total datasets: {len(all_stats)}\n\n")
            
            f.write("Lambda\tMean_SN\tStd_SN\tMagic_%\tNon-Magic_%\n")
            for stats in all_stats:
                f.write(f"{stats['lambda']:.1f}\t{stats['mean_label']:.4f}\t{stats['std_label']:.4f}\t")
                f.write(f"{stats['above_equal_1_percent']:.2f}\t{stats['below_1_percent']:.2f}\n")
        
        print(f"\nSummary report saved: {summary_file}")
        
        # Create summary plot
        create_summary_plot(all_stats, results_dir)
    
    print(f"\nBatch processing complete!")
    print(f"Results saved in: {results_dir}")

def create_summary_plot(all_stats, results_dir):
    """Create summary plots showing trends across lambda values."""
    lambdas = [s['lambda'] for s in all_stats]
    means = [s['mean_label'] for s in all_stats]
    stds = [s['std_label'] for s in all_stats]
    magic_percents = [s['above_equal_1_percent'] for s in all_stats]
    
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Standard deviation vs lambda
    plt.subplot(1, 3, 1)
    plt.plot(lambdas, stds, 'ro-', linewidth=2, markersize=8)
    plt.xlabel('Lambda (λ)')
    plt.ylabel('Std Stabilizer Norm')
    plt.title('Standard Deviation vs Lambda')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Min/Mean/Max vs lambda
    plt.subplot(1, 3, 2)
    mins = [s['min_label'] for s in all_stats]
    maxs = [s['max_label'] for s in all_stats]
    plt.plot(lambdas, mins, 'go-', label='Min', linewidth=2, markersize=6)
    plt.plot(lambdas, maxs, 'ro-', label='Max', linewidth=2, markersize=6)
    plt.plot(lambdas, means, 'bo-', label='Mean', linewidth=2, markersize=6)
    plt.xlabel('Lambda (λ)')
    plt.ylabel('Stabilizer Norm')
    plt.title('Min/Mean/Max vs Lambda')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Magic vs non-magic percentage
    plt.subplot(1, 3, 3)
    non_magic_percents = [100 - mp for mp in magic_percents]
    plt.plot(lambdas, magic_percents, 'ro-', label='Magic (%)', linewidth=2, markersize=6)
    plt.plot(lambdas, non_magic_percents, 'bo-', label='Non-Magic (%)', linewidth=2, markersize=6)
    plt.axhline(y=50, color='black', linestyle='--', alpha=0.7, label='50% threshold')
    plt.xlabel('Lambda (λ)')
    plt.ylabel('Percentage')
    plt.title('Magic vs Non-Magic Percentage')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    summary_plot_file = os.path.join(results_dir, "summary_analysis.png")
    plt.savefig(summary_plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Summary plot saved: {summary_plot_file}")

if __name__ == "__main__":
    main()