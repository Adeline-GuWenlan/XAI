import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import os

# Load the labels
decoded_tokens_dir = "/Users/guwenlan/Desktop/XAI/MixMLP/Decoded_Tokens/"
labels = np.load(os.path.join(decoded_tokens_dir, "2q_100000_states_labels.npy"))

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Label Distribution Analysis', fontsize=16)

# Histogram
axes[0,0].hist(labels, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
axes[0,0].set_title('Histogram of Labels')
axes[0,0].set_xlabel('Label Value')
axes[0,0].set_ylabel('Frequency')
axes[0,0].grid(True, alpha=0.3)

# Box plot
axes[0,1].boxplot(labels)
axes[0,1].set_title('Box Plot of Labels')
axes[0,1].set_ylabel('Label Value')
axes[0,1].grid(True, alpha=0.3)

# Q-Q plot against normal distribution
stats.probplot(labels, dist="norm", plot=axes[1,0])
axes[1,0].set_title('Q-Q Plot vs Normal Distribution')
axes[1,0].grid(True, alpha=0.3)

# Cumulative distribution
sorted_labels = np.sort(labels)
y = np.arange(1, len(sorted_labels) + 1) / len(sorted_labels)
axes[1,1].plot(sorted_labels, y, 'b-', linewidth=2)
axes[1,1].set_title('Cumulative Distribution Function')
axes[1,1].set_xlabel('Label Value')
axes[1,1].set_ylabel('Cumulative Probability')
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(decoded_tokens_dir, 'label_distribution_analysis.png'), dpi=300, bbox_inches='tight')
print("Distribution plots saved as 'label_distribution_analysis.png'")

# Additional analysis
print("\nDetailed Distribution Analysis:")
print("=" * 50)

# Check if it might be a mixture distribution
# Look for potential bimodality
hist, bin_edges = np.histogram(labels, bins=50)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# Find peaks in histogram
from scipy.signal import find_peaks
peaks, _ = find_peaks(hist, height=np.max(hist)*0.1)
print(f"Potential modes (peaks) at values: {bin_centers[peaks]}")
print(f"Number of potential modes: {len(peaks)}")

# Check concentration around specific values
print(f"\nConcentration analysis:")
print(f"Values < 0.5: {np.sum(labels < 0.5)} samples ({np.sum(labels < 0.5)/len(labels)*100:.1f}%)")
print(f"Values 0.5-1.0: {np.sum((labels >= 0.5) & (labels < 1.0))} samples ({np.sum((labels >= 0.5) & (labels < 1.0))/len(labels)*100:.1f}%)")
print(f"Values 1.0-1.5: {np.sum((labels >= 1.0) & (labels < 1.5))} samples ({np.sum((labels >= 1.0) & (labels < 1.5))/len(labels)*100:.1f}%)")
print(f"Values >= 1.5: {np.sum(labels >= 1.5)} samples ({np.sum(labels >= 1.5)/len(labels)*100:.1f}%)")

plt.show()