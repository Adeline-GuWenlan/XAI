import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import os

# Load the labels
labels = np.load('Utils/Data/3q_100000_labels.npy')

print("Label Distribution Analysis")
print("=" * 40)

# Basic statistics
print(f"Total samples: {len(labels)}")
print(f"Min value: {np.min(labels):.6f}")
print(f"Max value: {np.max(labels):.6f}")
print(f"Mean: {np.mean(labels):.6f}")
print(f"Median: {np.median(labels):.6f}")
print(f"Standard deviation: {np.std(labels):.6f}")
print(f"Variance: {np.var(labels):.6f}")

print("\nPercentiles:")
percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
for p in percentiles:
    print(f"{p}th percentile: {np.percentile(labels, p):.6f}")

print("\nDistribution shape:")
print(f"Skewness: {stats.skew(labels):.6f}")
print(f"Kurtosis: {stats.kurtosis(labels):.6f}")

# Check for unique values
unique_labels = np.unique(labels)
print(f"\nNumber of unique values: {len(unique_labels)}")

if len(unique_labels) < 20:
    print("Unique values:", unique_labels)
else:
    print("First 20 unique values:", unique_labels[:20])
    print("Last 20 unique values:", unique_labels[-20:])

# Binned analysis
print("\nBinned analysis (10 bins):")
hist, bin_edges = np.histogram(labels, bins=10)
for i in range(len(hist)):
    print(f"Bin [{bin_edges[i]:.3f}, {bin_edges[i+1]:.3f}]: {hist[i]} samples ({hist[i]/len(labels)*100:.1f}%)")

# Test for common distributions
print("\nDistribution tests:")

# Test for normal distribution
stat, p_value = stats.normaltest(labels)
print(f"Normality test (D'Agostino): statistic={stat:.4f}, p-value={p_value:.6f}")

# Test for exponential distribution
stat, p_value = stats.kstest(labels, 'expon', args=(np.min(labels), np.mean(labels) - np.min(labels)))
print(f"Exponential test: statistic={stat:.4f}, p-value={p_value:.6f}")

# Test for uniform distribution
stat, p_value = stats.kstest(labels, 'uniform', args=(np.min(labels), np.max(labels) - np.min(labels)))
print(f"Uniform test: statistic={stat:.4f}, p-value={p_value:.6f}")

print("\nSample of labels:")
print(labels[:20])