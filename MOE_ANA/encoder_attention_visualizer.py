import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap

class EncoderAttentionVisualizer:
    """Visualizer for encoder attention patterns from quantum magic prediction transformer"""
    
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
        Plot how CLS token attention evolves through transformer encoder layers
        
        Args:
            attention_analysis: Output from EncoderAttentionAnalyzer.analyze_cls_attention_patterns
            sample_idx: Sample index for title
            quantum_labels: Labels for matrix elements
        """
        cls_patterns = attention_analysis['cls_patterns']
        num_layers = len(cls_patterns)
        matrix_dim = attention_analysis['matrix_dim']
        n_qubits = int(np.log2(matrix_dim))
        
        fig, axes = plt.subplots(2, num_layers, figsize=(4*num_layers*self.figsize_scale, 8*self.figsize_scale))
        if num_layers == 1:
            axes = axes.reshape(2, 1)
            
        fig.suptitle(f'CLS Attention Evolution Through Encoder Layers - Sample {sample_idx}\n'
                    f'Predicted Magic: {attention_analysis["prediction"]:.4f}', 
                    fontsize=16, y=0.95)
        
        # Plot attention heatmaps
        for layer_idx, pattern in enumerate(cls_patterns):
            # Average attention across heads
            avg_attention = pattern['avg_cls_to_matrix']
            
            # Top row: Heatmap
            im1 = axes[0, layer_idx].imshow(avg_attention, cmap=self.attention_cmap, 
                                          vmin=0, vmax=avg_attention.max())
            axes[0, layer_idx].set_title(f'Layer {layer_idx}\nEntropy: {pattern["attention_entropy"]:.3f}\n'
                                       f'Sparsity: {pattern["attention_sparsity"]:.3f}')
            
            # Add quantum state labels
            row_labels = [f'|{format(i, f"0{n_qubits}b")}⟩' for i in range(matrix_dim)]
            col_labels = [f'⟨{format(i, f"0{n_qubits}b")}|' for i in range(matrix_dim)]
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
            axes[1, layer_idx].set_title(f'Max Attn: {pattern["max_attention_value"]:.4f}')
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
        
    def plot_head_specialization_by_layer(self, head_analysis: Dict, layer_idx: int = 0) -> None:
        """
        Visualize how different attention heads specialize within a specific layer
        
        Args:
            head_analysis: Output from EncoderAttentionAnalyzer.aggregate_head_specialization
            layer_idx: Which layer to visualize
        """
        if layer_idx not in head_analysis:
            print(f"Layer {layer_idx} not found in head analysis")
            return
            
        layer_heads = head_analysis[layer_idx]
        num_heads = len(layer_heads)
        matrix_size = int(np.sqrt(len(layer_heads[0]['mean_pattern'])))
        n_qubits = int(np.log2(matrix_size))
        
        fig, axes = plt.subplots(3, num_heads, figsize=(4*num_heads*self.figsize_scale, 12*self.figsize_scale))
        if num_heads == 1:
            axes = axes.reshape(3, 1)
            
        fig.suptitle(f'Attention Head Specialization - Layer {layer_idx}', fontsize=16)
        
        for head_idx in range(num_heads):
            head_data = layer_heads[head_idx]
            mean_pattern = head_data['mean_pattern'].reshape(matrix_size, matrix_size)
            std_pattern = head_data['std_pattern'].reshape(matrix_size, matrix_size)
            
            # Top row: Mean attention pattern
            im1 = axes[0, head_idx].imshow(mean_pattern, cmap=self.attention_cmap)
            axes[0, head_idx].set_title(f'Head {head_idx}\nConsistency: {head_data["consistency"]:.3f}\n'
                                      f'Entropy: {head_data["mean_entropy"]:.3f}')
            
            # Add quantum state labels
            row_labels = [f'|{format(i, f"0{n_qubits}b")}⟩' for i in range(matrix_size)]
            col_labels = [f'⟨{format(i, f"0{n_qubits}b")}|' for i in range(matrix_size)]
            axes[0, head_idx].set_xticks(range(matrix_size))
            axes[0, head_idx].set_yticks(range(matrix_size))
            axes[0, head_idx].set_xticklabels(col_labels, rotation=45)
            axes[0, head_idx].set_yticklabels(row_labels)
            
            plt.colorbar(im1, ax=axes[0, head_idx], fraction=0.046, pad=0.04)
            
            # Middle row: Variability (standard deviation)
            im2 = axes[1, head_idx].imshow(std_pattern, cmap='Reds')
            axes[1, head_idx].set_title(f'Variability\nSparsity: {head_data["mean_sparsity"]:.3f}')
            axes[1, head_idx].set_xticks(range(matrix_size))
            axes[1, head_idx].set_yticks(range(matrix_size))
            axes[1, head_idx].set_xticklabels(col_labels, rotation=45)
            axes[1, head_idx].set_yticklabels(row_labels)
            plt.colorbar(im2, ax=axes[1, head_idx], fraction=0.046, pad=0.04)
            
            # Bottom row: Statistics
            axes[2, head_idx].axis('off')
            stats_text = f"""Head {head_idx} Statistics:
            
Mean Entropy: {head_data['mean_entropy']:.3f}
Std Entropy: {head_data['std_entropy']:.3f}

Mean Sparsity: {head_data['mean_sparsity']:.3f}
Std Sparsity: {head_data['std_sparsity']:.3f}

Most Common Max Pos: {head_data['most_common_max_position']}
Position Consistency: {head_data['position_consistency']:.3f}

Pattern Consistency: {head_data['consistency']:.3f}"""
            
            axes[2, head_idx].text(0.1, 0.9, stats_text, transform=axes[2, head_idx].transAxes,
                                 fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'head_specialization_layer_{layer_idx}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_layer_evolution_statistics(self, layer_evolution: List[Dict]) -> None:
        """
        Plot how attention statistics evolve through encoder layers
        
        Args:
            layer_evolution: Output from EncoderAttentionAnalyzer.analyze_layer_evolution
        """
        layers = [stat['layer'] for stat in layer_evolution]
        entropies = [stat['mean_entropy'] for stat in layer_evolution]
        entropy_stds = [stat['std_entropy'] for stat in layer_evolution]
        sparsities = [stat['mean_sparsity'] for stat in layer_evolution]
        sparsity_stds = [stat['std_sparsity'] for stat in layer_evolution]
        max_attentions = [stat['mean_max_attention'] for stat in layer_evolution]
        max_attention_stds = [stat['std_max_attention'] for stat in layer_evolution]
        concentrations = [stat['mean_concentration'] for stat in layer_evolution]
        concentration_stds = [stat['std_concentration'] for stat in layer_evolution]
        
        fig, axes = plt.subplots(2, 2, figsize=(12*self.figsize_scale, 10*self.figsize_scale))
        
        # Entropy evolution
        axes[0, 0].errorbar(layers, entropies, yerr=entropy_stds, marker='o', linewidth=2, 
                          markersize=8, capsize=5)
        axes[0, 0].set_xlabel('Encoder Layer')
        axes[0, 0].set_ylabel('Attention Entropy')
        axes[0, 0].set_title('Attention Entropy Through Encoder Layers')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_xticks(layers)
        
        # Sparsity evolution  
        axes[0, 1].errorbar(layers, sparsities, yerr=sparsity_stds, marker='s', linewidth=2,
                          markersize=8, capsize=5, color='red')
        axes[0, 1].set_xlabel('Encoder Layer')
        axes[0, 1].set_ylabel('Attention Sparsity')
        axes[0, 1].set_title('Attention Sparsity Through Encoder Layers')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_xticks(layers)
        
        # Maximum attention evolution
        axes[1, 0].errorbar(layers, max_attentions, yerr=max_attention_stds, marker='^', linewidth=2,
                          markersize=8, capsize=5, color='green')
        axes[1, 0].set_xlabel('Encoder Layer')
        axes[1, 0].set_ylabel('Maximum Attention Value')
        axes[1, 0].set_title('Peak Attention Through Encoder Layers')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_xticks(layers)
        
        # Concentration evolution
        axes[1, 1].errorbar(layers, concentrations, yerr=concentration_stds, marker='d', linewidth=2,
                          markersize=8, capsize=5, color='purple')
        axes[1, 1].set_xlabel('Encoder Layer')
        axes[1, 1].set_ylabel('Attention Concentration')
        axes[1, 1].set_title('Attention Concentration Through Encoder Layers')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_xticks(layers)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'encoder_layer_evolution_statistics.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_attention_rollout(self, rollout_matrix: np.ndarray, sample_idx: int) -> None:
        """
        Visualize attention rollout through encoder layers
        
        Args:
            rollout_matrix: Output from EncoderAttentionAnalyzer.compute_attention_rollout
            sample_idx: Sample index for title
        """
        matrix_dim = int(np.sqrt(rollout_matrix.shape[0] - 1))
        n_qubits = int(np.log2(matrix_dim))
        
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
        
        # Add quantum state labels
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
        
    def plot_head_comparison_across_layers(self, head_analysis: Dict) -> None:
        """Compare the same head index across different layers"""
        num_layers = len(head_analysis)
        num_heads = len(head_analysis[0]) if head_analysis else 0
        
        if num_layers == 0 or num_heads == 0:
            return
        
        for head_idx in range(num_heads):
            fig, axes = plt.subplots(2, num_layers, figsize=(4*num_layers*self.figsize_scale, 8*self.figsize_scale))
            if num_layers == 1:
                axes = axes.reshape(2, 1)
                
            fig.suptitle(f'Head {head_idx} Across All Encoder Layers', fontsize=16)
            
            for layer_idx in range(num_layers):
                if layer_idx in head_analysis and head_idx < len(head_analysis[layer_idx]):
                    head_data = head_analysis[layer_idx][head_idx]
                    matrix_size = int(np.sqrt(len(head_data['mean_pattern'])))
                    mean_pattern = head_data['mean_pattern'].reshape(matrix_size, matrix_size)
                    
                    # Top row: Mean attention pattern
                    im1 = axes[0, layer_idx].imshow(mean_pattern, cmap=self.attention_cmap)
                    axes[0, layer_idx].set_title(f'Layer {layer_idx}\nEntropy: {head_data["mean_entropy"]:.3f}')
                    plt.colorbar(im1, ax=axes[0, layer_idx], fraction=0.046, pad=0.04)
                    
                    # Bottom row: Entropy and sparsity comparison
                    metrics = ['Entropy', 'Sparsity', 'Consistency']
                    values = [
                        head_data['mean_entropy'],
                        head_data['mean_sparsity'], 
                        head_data['consistency']
                    ]
                    colors = ['skyblue', 'lightcoral', 'lightgreen']
                    
                    bars = axes[1, layer_idx].bar(metrics, values, color=colors)
                    axes[1, layer_idx].set_title(f'Layer {layer_idx} Metrics')
                    axes[1, layer_idx].set_ylabel('Value')
                    axes[1, layer_idx].tick_params(axis='x', rotation=45)
                    
                    # Add value labels on bars
                    for bar, value in zip(bars, values):
                        axes[1, layer_idx].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                                              f'{value:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / f'head_{head_idx}_across_layers.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
    def create_encoder_attention_dashboard(self, analysis_results: Dict, quantum_labels: List[str] = None) -> None:
        """
        Create comprehensive encoder attention dashboard
        
        Args:
            analysis_results: Complete output from EncoderAttentionAnalyzer.analyze_dataset_encoder_attention
            quantum_labels: Labels for matrix elements
        """
        print("Creating encoder attention analysis dashboard...")
        
        # Plot layer evolution statistics
        self.plot_layer_evolution_statistics(analysis_results['layer_evolution'])
        
        # Plot head specialization for each layer
        num_layers = len(analysis_results['layer_evolution'])
        for layer_idx in range(num_layers):
            self.plot_head_specialization_by_layer(analysis_results['head_specialization'], layer_idx)
        
        # Plot head comparison across layers
        self.plot_head_comparison_across_layers(analysis_results['head_specialization'])
        
        # Analyze individual samples (first 10 for detailed analysis)
        max_individual = min(10, len(analysis_results['sample_data']))
        print(f"Creating individual sample analyses for {max_individual} samples...")
        
        for i in range(max_individual):
            sample_data = analysis_results['sample_data'][i]
            
            # Reconstruct data for CLS analysis
            temp_attention_data = {
                'prediction': np.array([sample_data['predicted_magic']]),
                'attention_weights': sample_data['attention_weights']
            }
            
            # Analyze CLS patterns directly
            cls_analysis = self._analyze_cls_patterns_direct(temp_attention_data, sample_data['sample_idx'])
            
            # Create visualizations
            self.plot_cls_attention_evolution(cls_analysis, sample_data['sample_idx'], quantum_labels)
            
            # Compute and plot attention rollout
            rollout = self._compute_rollout_direct(temp_attention_data, 0)
            self.plot_attention_rollout(rollout, sample_data['sample_idx'])
        
        print(f"Encoder attention dashboard complete! Results saved to {self.output_dir}/")
        
        # Create summary report
        self._create_encoder_summary_report(analysis_results)
        
    def _create_encoder_summary_report(self, analysis_results: Dict) -> None:
        """Create comprehensive encoder attention analysis report"""
        summary_stats = analysis_results['summary_stats']
        
        report = f"""
ENCODER ATTENTION ANALYSIS SUMMARY REPORT
========================================

Dataset Overview:
- Total samples analyzed: {summary_stats['total_samples']}
- Average prediction error: {summary_stats['avg_prediction_error']:.4f}
- True magic value range: [{summary_stats['true_magic_range'][0]:.4f}, {summary_stats['true_magic_range'][1]:.4f}]
- Predicted magic range: [{summary_stats['predicted_magic_range'][0]:.4f}, {summary_stats['predicted_magic_range'][1]:.4f}]

Encoder Layer Evolution Analysis:
"""
        
        for layer_stat in analysis_results['layer_evolution']:
            report += f"""
Layer {layer_stat['layer']}:
  - Mean attention entropy: {layer_stat['mean_entropy']:.3f} ± {layer_stat['std_entropy']:.3f}
  - Mean attention sparsity: {layer_stat['mean_sparsity']:.3f} ± {layer_stat['std_sparsity']:.3f}
  - Mean max attention: {layer_stat['mean_max_attention']:.3f} ± {layer_stat['std_max_attention']:.3f}
  - Mean concentration: {layer_stat['mean_concentration']:.3f} ± {layer_stat['std_concentration']:.3f}
"""
        
        # Head specialization summary
        head_analysis = analysis_results['head_specialization']
        num_layers = len(head_analysis)
        num_heads = len(head_analysis[0]) if head_analysis else 0
        
        report += f"""
Head Specialization Analysis:
- Number of encoder layers: {num_layers}
- Number of attention heads per layer: {num_heads}

Head Consistency Across Layers:"""
        
        for layer_idx in range(num_layers):
            if layer_idx in head_analysis:
                layer_heads = head_analysis[layer_idx]
                avg_consistency = np.mean([head['consistency'] for head in layer_heads.values()])
                avg_entropy = np.mean([head['mean_entropy'] for head in layer_heads.values()])
                report += f"""
  Layer {layer_idx}: Avg Consistency = {avg_consistency:.3f}, Avg Entropy = {avg_entropy:.3f}"""
        
        report += """

Files Generated:
- encoder_layer_evolution_statistics.png: How attention patterns evolve through encoder layers
- head_specialization_layer_*.png: Specialization patterns for each attention head per layer
- head_*_across_layers.png: Comparison of same head across different layers
- cls_attention_evolution_sample_*.png: CLS attention evolution for individual samples
- attention_rollout_sample_*.png: Information flow analysis for individual samples

Interpretation Guide:
- Entropy: Higher values indicate more distributed attention (less focused)
- Sparsity: Higher values indicate more sparse attention (more focused)
- Concentration: Higher values indicate attention is more concentrated on fewer elements
- Head Consistency: Higher values indicate heads have consistent attention patterns across samples
- Layer Evolution: Shows how attention patterns change as information flows through encoder layers
"""
        
        with open(self.output_dir / 'encoder_attention_analysis_report.txt', 'w') as f:
            f.write(report)
            
        print(f"Encoder attention analysis report saved to {self.output_dir}/encoder_attention_analysis_report.txt")
    
    def _analyze_cls_patterns_direct(self, attention_data, sample_idx):
        """Direct CLS pattern analysis without external dependency"""
        cls_patterns = []
        matrix_dim = int(np.sqrt(attention_data['attention_weights'][0]['cls_attention'].shape[-1] - 1))
        
        for layer_data in attention_data['attention_weights']:
            # Get CLS attention for this sample [num_heads, seq_len]
            cls_attn = layer_data['cls_attention'][0]
            
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
                'attention_sparsity': (matrix_attn < 0.01).mean(),
                'max_attention_value': matrix_attn_reshaped.mean(axis=0).max(),
            }
            cls_patterns.append(layer_analysis)
            
        return {
            'sample_idx': sample_idx,
            'prediction': attention_data['prediction'][0],
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