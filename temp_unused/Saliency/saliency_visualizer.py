import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

class SaliencyVisualizer:
    def __init__(self, output_dir='saliency_results'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        plt.style.use('default')
        sns.set_palette("husl")
    
    def plot_saliency_heatmap(self, saliency_map, title, filename, 
                             quantum_labels=None, vmax=None):
        """Create heatmap with quantum state labels"""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Create heatmap
        im = ax.imshow(saliency_map, cmap='hot', aspect='equal', 
                      vmin=0, vmax=vmax)
        
        # Add quantum state labels if provided
        if quantum_labels:
            ax.set_xticks(range(len(quantum_labels)))
            ax.set_yticks(range(len(quantum_labels)))
            ax.set_xticklabels(quantum_labels)
            ax.set_yticklabels(quantum_labels)
        
        plt.colorbar(im, ax=ax, label='Saliency Magnitude')
        ax.set_title(title)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f"{filename}.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def compare_methods(self, results, sample_idx=0, quantum_labels=None):
        """Compare different saliency methods side by side"""
        methods = list(results.keys())
        n_methods = len(methods)
        
        fig, axes = plt.subplots(1, n_methods, figsize=(5*n_methods, 4))
        if n_methods == 1:
            axes = [axes]
        
        # Find global vmax for consistent scaling
        all_maps = [results[method]['sample_results'][sample_idx]['saliency_map'] 
                   for method in methods]
        vmax = np.max([np.max(m) for m in all_maps])
        
        for i, method in enumerate(methods):
            saliency_map = results[method]['sample_results'][sample_idx]['saliency_map']
            true_magic = results[method]['sample_results'][sample_idx]['true_magic']
            pred_magic = results[method]['sample_results'][sample_idx]['predicted_magic']
            
            im = axes[i].imshow(saliency_map, cmap='hot', aspect='equal', 
                              vmin=0, vmax=vmax)
            axes[i].set_title(f'{method}\nTrue: {true_magic:.3f}, Pred: {pred_magic:.3f}')
            
            if quantum_labels:
                axes[i].set_xticks(range(len(quantum_labels)))
                axes[i].set_yticks(range(len(quantum_labels)))
                axes[i].set_xticklabels(quantum_labels)
                axes[i].set_yticklabels(quantum_labels)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f"method_comparison_sample_{sample_idx}.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_aggregated_patterns(self, aggregated_results):
        """Plot aggregated saliency patterns"""
        methods = list(aggregated_results.keys())
        
        for method in methods:
            data = aggregated_results[method]
            
            # Plot mean saliency
            self.plot_saliency_heatmap(
                data['mean_saliency'], 
                f'{method} - Mean Saliency',
                f'{method}_mean_saliency'
            )
            
            # Plot high vs low magic comparison
            if 'high_magic_avg' in data and 'low_magic_avg' in data:
                fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
                
                vmax = max(np.max(data['high_magic_avg']), np.max(data['low_magic_avg']))
                
                im1 = ax1.imshow(data['high_magic_avg'], cmap='hot', vmin=0, vmax=vmax)
                ax1.set_title('High Magic States (>0.1)')
                
                im2 = ax2.imshow(data['low_magic_avg'], cmap='hot', vmin=0, vmax=vmax)
                ax2.set_title('Low Magic States (≤0.1)')
                
                # Difference map
                diff = data['high_magic_avg'] - data['low_magic_avg']
                im3 = ax3.imshow(diff, cmap='RdBu', vmin=-np.max(np.abs(diff)), vmax=np.max(np.abs(diff)))
                ax3.set_title('Difference (High - Low)')
                
                plt.colorbar(im1, ax=ax1)
                plt.colorbar(im2, ax=ax2)
                plt.colorbar(im3, ax=ax3)
                
                plt.tight_layout()
                plt.savefig(self.output_dir / f'{method}_magic_comparison.png', 
                           dpi=300, bbox_inches='tight')
                plt.close()