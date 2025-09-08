import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap

class AttentionVisualizer:
    """Visualizer for attention patterns from quantum magic prediction transformer"""
    
    def __init__(self, output_dir: str, figsize_scale: float = 1.0):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.figsize_scale = figsize_scale
        
        # Set up style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create custom colormap for attention
        self.attention_cmap = LinearSegmentedColormap.from_list(
            'attention', ['white', 'lightblue', 'darkblue', 'red']
        )
        
    def generate_quantum_labels(self, n_qubits: int) -> List[str]:
        """Generate |00⟩, |01⟩, etc. labels for matrix positions"""
        labels = []
        matrix_dim = 2 ** n_qubits
        for i in range(matrix_dim):
            for j in range(matrix_dim):
                i_binary = format(i, f'0{n_qubits}b')
                j_binary = format(j, f'0{n_qubits}b')
                labels.append(f'⟨{i_binary}|ρ|{j_binary}⟩')
        return labels
        
    def plot_cls_attention_evolution(self, attention_analysis: Dict, sample_idx: int, 
                                   quantum_labels: List[str] = None) -> None:
        """
        Plot how CLS token attention evolves through transformer layers
        
        Args:
            attention_analysis: Output from AttentionAnalyzer.analyze_cls_attention_patterns
            sample_idx: Sample index for title
            quantum_labels: Labels for matrix elements
        """
        cls_patterns = attention_analysis['cls_patterns']
        num_layers = len(cls_patterns)
        matrix_dim = attention_analysis['matrix_dim']
        
        fig, axes = plt.subplots(2, num_layers, figsize=(4*num_layers*self.figsize_scale, 8*self.figsize_scale))
        if num_layers == 1:
            axes = axes.reshape(2, 1)
            
        fig.suptitle(f'CLS Attention Evolution - Sample {sample_idx}\n'
                    f'Predicted Magic: {attention_analysis["prediction"]:.4f}', 
                    fontsize=16, y=0.95)
        
        # Plot attention heatmaps
        for layer_idx, pattern in enumerate(cls_patterns):
            # Average attention across heads
            avg_attention = pattern['avg_cls_to_matrix']
            
            # Top row: Heatmap
            im1 = axes[0, layer_idx].imshow(avg_attention, cmap=self.attention_cmap, 
                                          vmin=0, vmax=avg_attention.max())
            axes[0, layer_idx].set_title(f'Layer {layer_idx}\nEntropy: {pattern["attention_entropy"]:.3f}')
            
            # Add quantum state labels if provided
            if quantum_labels and len(quantum_labels) == matrix_dim * matrix_dim:
                # Create grid labels
                row_labels = [f'|{format(i, f"0{int(np.log2(matrix_dim))}b")}⟩' for i in range(matrix_dim)]
                col_labels = [f'⟨{format(i, f"0{int(np.log2(matrix_dim))}b")}|' for i in range(matrix_dim)]
                axes[0, layer_idx].set_xticks(range(matrix_dim))
                axes[0, layer_idx].set_yticks(range(matrix_dim))
                axes[0, layer_idx].set_xticklabels(col_labels, rotation=45)
                axes[0, layer_idx].set_yticklabels(row_labels)
            
            # Add colorbar
            plt.colorbar(im1, ax=axes[0, layer_idx], fraction=0.046, pad=0.04)
            
            # Mark maximum attention position
            max_pos = pattern['max_attended_positions']
            rect = patches.Rectangle((max_pos[1]-0.5, max_pos[0]-0.5), 1, 1, 
                                   linewidth=3, edgecolor='red', facecolor='none')
            axes[0, layer_idx].add_patch(rect)
            
            # Bottom row: Flattened attention pattern
            flattened_attn = avg_attention.flatten()
            axes[1, layer_idx].bar(range(len(flattened_attn)), flattened_attn, 
                                  color='skyblue', alpha=0.7)
            axes[1, layer_idx].set_title(f'Flattened Attention')
            axes[1, layer_idx].set_xlabel('Matrix Element Index')
            axes[1, layer_idx].set_ylabel('Attention Weight')
            
            # Highlight top-3 attended positions
            top_indices = np.argsort(flattened_attn)[-3:]
            for idx in top_indices:
                axes[1, layer_idx].bar(idx, flattened_attn[idx], color='red', alpha=0.8)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'cls_attention_evolution_sample_{sample_idx}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_attention_rollout(self, rollout_matrix: np.ndarray, sample_idx: int,
                             quantum_labels: List[str] = None) -> None:
        """
        Visualize attention rollout - cumulative attention flow
        
        Args:
            rollout_matrix: Output from AttentionAnalyzer.compute_attention_rollout
            sample_idx: Sample index for title
            quantum_labels: Labels for matrix elements
        """
        matrix_dim = int(np.sqrt(rollout_matrix.shape[0] - 1))
        
        fig, axes = plt.subplots(1, 2, figsize=(12*self.figsize_scale, 5*self.figsize_scale))
        
        # Full rollout matrix
        im1 = axes[0].imshow(rollout_matrix, cmap='Blues')
        axes[0].set_title(f'Complete Attention Rollout - Sample {sample_idx}')
        axes[0].set_xlabel('To Token')
        axes[0].set_ylabel('From Token')
        plt.colorbar(im1, ax=axes[0])
        
        # CLS rollout (how CLS aggregates from all positions)
        cls_rollout = rollout_matrix[0, 1:]  # CLS row, skip CLS self-attention
        cls_rollout_matrix = cls_rollout.reshape(matrix_dim, matrix_dim)
        
        im2 = axes[1].imshow(cls_rollout_matrix, cmap=self.attention_cmap)
        axes[1].set_title('CLS Token Information Aggregation')
        
        if quantum_labels:
            n_qubits = int(np.log2(matrix_dim))
            row_labels = [f'|{format(i, f"0{n_qubits}b")}⟩' for i in range(matrix_dim)]
            col_labels = [f'⟨{format(i, f"0{n_qubits}b")}|' for i in range(matrix_dim)]
            axes[1].set_xticks(range(matrix_dim))
            axes[1].set_yticks(range(matrix_dim))
            axes[1].set_xticklabels(col_labels, rotation=45)
            axes[1].set_yticklabels(row_labels)
        
        plt.colorbar(im2, ax=axes[1])
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'attention_rollout_sample_{sample_idx}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_head_specialization(self, head_analysis: Dict, layer_idx: int = 0) -> None:
        """
        Visualize how different attention heads specialize
        
        Args:
            head_analysis: Output from AttentionAnalyzer._aggregate_head_specialization
            layer_idx: Which layer to visualize
        """
        if layer_idx not in head_analysis:
            print(f"Layer {layer_idx} not found in head analysis")
            return
            
        layer_heads = head_analysis[layer_idx]
        num_heads = len(layer_heads)
        matrix_size = int(np.sqrt(len(layer_heads[0]['mean_pattern'])))
        
        fig, axes = plt.subplots(2, num_heads, figsize=(3*num_heads*self.figsize_scale, 6*self.figsize_scale))
        if num_heads == 1:
            axes = axes.reshape(2, 1)
            
        fig.suptitle(f'Attention Head Specialization - Layer {layer_idx}', fontsize=16)
        
        for head_idx in range(num_heads):
            head_data = layer_heads[head_idx]
            mean_pattern = head_data['mean_pattern'].reshape(matrix_size, matrix_size)
            std_pattern = head_data['std_pattern'].reshape(matrix_size, matrix_size)
            
            # Top row: Mean attention pattern
            im1 = axes[0, head_idx].imshow(mean_pattern, cmap=self.attention_cmap)
            axes[0, head_idx].set_title(f'Head {head_idx}\nConsistency: {head_data["consistency"]:.3f}')
            plt.colorbar(im1, ax=axes[0, head_idx], fraction=0.046, pad=0.04)
            
            # Bottom row: Variability (standard deviation)
            im2 = axes[1, head_idx].imshow(std_pattern, cmap='Reds')
            axes[1, head_idx].set_title(f'Variability')
            plt.colorbar(im2, ax=axes[1, head_idx], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'head_specialization_layer_{layer_idx}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_layer_evolution_statistics(self, layer_evolution: List[Dict]) -> None:
        """
        Plot how attention statistics evolve through layers
        
        Args:
            layer_evolution: Output from AttentionAnalyzer._analyze_layer_evolution
        """
        layers = [stat['layer'] for stat in layer_evolution]
        entropies = [stat['mean_entropy'] for stat in layer_evolution]
        entropy_stds = [stat['std_entropy'] for stat in layer_evolution]
        sparsities = [stat['mean_sparsity'] for stat in layer_evolution]
        sparsity_stds = [stat['std_sparsity'] for stat in layer_evolution]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12*self.figsize_scale, 5*self.figsize_scale))
        
        # Entropy evolution
        ax1.errorbar(layers, entropies, yerr=entropy_stds, marker='o', linewidth=2, 
                    markersize=8, capsize=5)
        ax1.set_xlabel('Layer')
        ax1.set_ylabel('Attention Entropy')
        ax1.set_title('Attention Entropy Through Layers')
        ax1.grid(True, alpha=0.3)
        
        # Sparsity evolution  
        ax2.errorbar(layers, sparsities, yerr=sparsity_stds, marker='s', linewidth=2,
                    markersize=8, capsize=5, color='red')
        ax2.set_xlabel('Layer')
        ax2.set_ylabel('Attention Sparsity')
        ax2.set_title('Attention Sparsity Through Layers')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'layer_evolution_statistics.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_individual_sample_analysis(self, sample_data: Dict, quantum_labels: List[str] = None) -> None:
        """
        Comprehensive analysis for individual sample
        
        Args:
            sample_data: Single sample from AttentionAnalyzer.analyze_dataset
            quantum_labels: Labels for matrix elements
        """
        sample_idx = sample_data['sample_idx']
        true_magic = sample_data['true_magic']
        pred_magic = sample_data['predicted_magic']
        
        # Reconstruct attention data for cls analysis
        attention_weights = sample_data['attention_weights']
        
        # Create temporary format for cls analysis
        temp_data = {
            'prediction': np.array([pred_magic]),
            'attention_weights': [
                {
                    'layer': layer_data['layer'],
                    'cls_attention': layer_data['cls_attention'],
                    'avg_head_weights': layer_data['avg_head_weights']
                }
                for layer_data in attention_weights
            ]
        }
        
        # Analyze CLS patterns - create the analysis directly without dummy analyzer
        cls_analysis = self._analyze_cls_patterns_direct(temp_data, sample_idx=0)
        
        # Compute attention rollout
        rollout = self._compute_rollout_direct(temp_data, sample_idx=0)
        
        # Generate visualizations
        self.plot_cls_attention_evolution(cls_analysis, sample_idx, quantum_labels)
        self.plot_attention_rollout(rollout, sample_idx, quantum_labels)
        
        # Additional individual analysis plot
        self._plot_sample_summary(sample_data, quantum_labels)
        
    def _plot_sample_summary(self, sample_data: Dict, quantum_labels: List[str] = None) -> None:
        """Plot summary for individual sample"""
        sample_idx = sample_data['sample_idx']
        true_magic = sample_data['true_magic']
        pred_magic = sample_data['predicted_magic']
        error = abs(pred_magic - true_magic)
        
        num_layers = len(sample_data['attention_weights'])
        
        fig, axes = plt.subplots(1, num_layers + 1, figsize=((num_layers + 1) * 4*self.figsize_scale, 4*self.figsize_scale))
        
        # Plot attention focus for each layer
        for layer_idx in range(num_layers):
            # Get average CLS attention for this layer
            cls_attn = sample_data['attention_weights'][layer_idx]['avg_head_weights'][0, 0, 1:]
            matrix_dim = int(np.sqrt(len(cls_attn)))
            cls_matrix = cls_attn.reshape(matrix_dim, matrix_dim)
            
            im = axes[layer_idx].imshow(cls_matrix, cmap=self.attention_cmap)
            axes[layer_idx].set_title(f'Layer {layer_idx}')
            plt.colorbar(im, ax=axes[layer_idx], fraction=0.046, pad=0.04)
        
        # Summary statistics
        axes[-1].axis('off')
        summary_text = f"""Sample {sample_idx} Summary:
        
True Magic: {true_magic:.4f}
Predicted: {pred_magic:.4f}
Error: {error:.4f}

Attention Statistics:
"""
        
        for layer_idx in range(num_layers):
            cls_attn = sample_data['attention_weights'][layer_idx]['avg_head_weights'][0, 0, 1:]
            entropy = -np.sum(cls_attn * np.log(cls_attn + 1e-10))
            sparsity = (cls_attn < 0.01).mean()
            max_attn = np.max(cls_attn)
            
            summary_text += f"Layer {layer_idx}: Entropy={entropy:.3f}, Sparsity={sparsity:.3f}, Max={max_attn:.3f}\n"
        
        axes[-1].text(0.1, 0.9, summary_text, transform=axes[-1].transAxes, 
                     fontsize=12, verticalalignment='top', fontfamily='monospace')
        
        plt.suptitle(f'Individual Sample Analysis - Sample {sample_idx}', fontsize=16)
        plt.tight_layout()
        plt.savefig(self.output_dir / f'individual_analysis_sample_{sample_idx}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
    def create_attention_dashboard(self, analysis_results: Dict, quantum_labels: List[str] = None) -> None:
        """
        Create comprehensive dashboard with all attention visualizations
        
        Args:
            analysis_results: Complete output from AttentionAnalyzer.analyze_dataset
            quantum_labels: Labels for matrix elements
        """
        print("Creating attention analysis dashboard...")
        
        # Plot layer evolution statistics
        self.plot_layer_evolution_statistics(analysis_results['layer_evolution'])
        
        # Plot head specialization for each layer
        for layer_idx in range(len(analysis_results['layer_evolution'])):
            self.plot_head_specialization(analysis_results['head_specialization'], layer_idx)
        
        # Analyze individual samples (first 20 for detailed analysis)
        max_individual = min(20, len(analysis_results['sample_data']))
        print(f"Creating individual sample analyses for {max_individual} samples...")
        
        for i in range(max_individual):
            sample_data = analysis_results['sample_data'][i]
            self.plot_individual_sample_analysis(sample_data, quantum_labels)
        
        print(f"Dashboard complete! Results saved to {self.output_dir}/")
        
        # Create summary report
        self._create_summary_report(analysis_results)
        
    def _create_summary_report(self, analysis_results: Dict) -> None:
        """Create text summary report"""
        summary_stats = analysis_results['summary_stats']
        
        report = f"""
ATTENTION ANALYSIS SUMMARY REPORT
================================

Dataset Overview:
- Total samples analyzed: {summary_stats['total_samples']}
- Average prediction error: {summary_stats['avg_prediction_error']:.4f}
- True magic value range: [{summary_stats['true_magic_range'][0]:.4f}, {summary_stats['true_magic_range'][1]:.4f}]
- Predicted magic range: [{summary_stats['predicted_magic_range'][0]:.4f}, {summary_stats['predicted_magic_range'][1]:.4f}]

Layer Evolution Analysis:
"""
        
        for layer_stat in analysis_results['layer_evolution']:
            report += f"""
Layer {layer_stat['layer']}:
  - Mean attention entropy: {layer_stat['mean_entropy']:.3f} ± {layer_stat['std_entropy']:.3f}
  - Mean attention sparsity: {layer_stat['mean_sparsity']:.3f} ± {layer_stat['std_sparsity']:.3f}
"""
        
        report += """
Files Generated:
- layer_evolution_statistics.png: How attention patterns evolve through layers
- head_specialization_layer_*.png: Specialization patterns for each attention head
- individual_analysis_sample_*.png: Detailed analysis for first 20 samples
- cls_attention_evolution_sample_*.png: CLS attention evolution for each sample
- attention_rollout_sample_*.png: Information flow analysis for each sample

Interpretation Guide:
- Entropy: Higher values indicate more distributed attention (less focused)
- Sparsity: Higher values indicate more sparse attention (more focused)
- Head Specialization: Different heads may focus on different matrix regions
- Rollout: Shows cumulative information flow to CLS token
"""
        
        with open(self.output_dir / 'attention_analysis_report.txt', 'w') as f:
            f.write(report)
            
        print(f"Summary report saved to {self.output_dir}/attention_analysis_report.txt")
    
    def _analyze_cls_patterns_direct(self, attention_data, sample_idx=0):
        """Direct CLS pattern analysis without external dependency"""
        cls_patterns = []
        matrix_dim = int(np.sqrt(attention_data['attention_weights'][0]['cls_attention'].shape[-1] - 1))
        
        for layer_data in attention_data['attention_weights']:
            # Get CLS attention for this sample [num_heads, seq_len]
            cls_attn = layer_data['cls_attention'][sample_idx]
            
            # Skip CLS token itself (position 0), focus on matrix elements
            matrix_attn = cls_attn[:, 1:]  # [num_heads, matrix_elements]
            
            # Reshape to matrix form for each head
            matrix_attn_reshaped = matrix_attn.reshape(cls_attn.shape[0], matrix_dim, matrix_dim)
            
            layer_analysis = {
                'layer': layer_data['layer'],
                'cls_to_matrix': matrix_attn_reshaped,  # [num_heads, D, D]
                'avg_cls_to_matrix': matrix_attn_reshaped.mean(axis=0),  # [D, D]
                'max_attended_positions': np.unravel_index(
                    np.argmax(matrix_attn_reshaped.mean(axis=0)), 
                    (matrix_dim, matrix_dim)
                ),
                'attention_entropy': -np.sum(
                    matrix_attn * np.log(matrix_attn + 1e-10), axis=1
                ).mean(),  # Average entropy across heads
            }
            cls_patterns.append(layer_analysis)
            
        return {
            'sample_idx': sample_idx,
            'prediction': attention_data['prediction'][sample_idx],
            'matrix_dim': matrix_dim,
            'cls_patterns': cls_patterns
        }
    
    def _compute_rollout_direct(self, attention_data, sample_idx=0):
        """Direct attention rollout computation"""
        rollout = None
        
        for layer_data in attention_data['attention_weights']:
            # Average across heads for this layer
            layer_attn = layer_data['avg_head_weights'][sample_idx]  # [seq_len, seq_len]
            
            if rollout is None:
                rollout = layer_attn
            else:
                # Matrix multiply to track cumulative attention flow
                rollout = np.matmul(rollout, layer_attn)
                
        return rollout