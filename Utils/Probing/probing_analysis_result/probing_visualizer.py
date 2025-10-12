import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path

class ProbingVisualizer:
    """Visualizer for layer-wise probing results"""
    
    def __init__(self, output_dir: str, figsize_scale: float = 1.0):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.figsize_scale = figsize_scale
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
    def plot_emergence_progression(self, emergence_analysis: Dict) -> None:
        """
        Plot how magic prediction capability emerges through layers
        
        Args:
            emergence_analysis: Output from LayerWiseProber.analyze_information_emergence
        """
        progression = emergence_analysis['performance_progression']
        
        layer_names = [p['layer'] for p in progression]
        r2_scores = [p['r2_score'] for p in progression]
        mse_scores = [p['mse'] for p in progression]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14*self.figsize_scale, 6*self.figsize_scale))
        
        # R² progression
        ax1.plot(range(len(layer_names)), r2_scores, 'o-', linewidth=3, markersize=8, color='blue')
        ax1.set_xlabel('Layer')
        ax1.set_ylabel('R² Score (Magic Prediction Capability)')
        ax1.set_title('Information Emergence Through Transformer Layers')
        ax1.set_xticks(range(len(layer_names)))
        ax1.set_xticklabels(layer_names, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # Highlight emergence layer if found
        if emergence_analysis['emergence_layer']:
            emergence_idx = layer_names.index(emergence_analysis['emergence_layer'])
            ax1.axvline(x=emergence_idx, color='red', linestyle='--', alpha=0.7, linewidth=2)
            ax1.text(emergence_idx, 0.95, f"Emergence\nΔR²={emergence_analysis['emergence_improvement']:.3f}", 
                    ha='center', va='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        # MSE progression (log scale)
        ax2.semilogy(range(len(layer_names)), mse_scores, 's-', linewidth=3, markersize=8, color='red')
        ax2.set_xlabel('Layer')
        ax2.set_ylabel('Mean Squared Error (log scale)')
        ax2.set_title('Prediction Error Reduction Through Layers')
        ax2.set_xticks(range(len(layer_names)))
        ax2.set_xticklabels(layer_names, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        # Highlight best layer
        best_layer_idx = layer_names.index(emergence_analysis['max_r2_layer'])
        ax2.axvline(x=best_layer_idx, color='green', linestyle='--', alpha=0.7, linewidth=2)
        ax2.text(best_layer_idx, min(mse_scores)*2, f"Best Layer\nR²={emergence_analysis['max_r2_score']:.3f}", 
                ha='center', va='bottom', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'layer_emergence_progression.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_representation_similarity(self, similarity_analysis: Dict) -> None:
        """
        Plot similarity between layer representations
        
        Args:
            similarity_analysis: Output from LayerWiseProber.compute_representation_similarity
        """
        similarity_matrix = similarity_analysis['similarity_matrix']
        layer_names = similarity_analysis['layer_names']
        
        fig, ax = plt.subplots(1, 1, figsize=(10*self.figsize_scale, 8*self.figsize_scale))
        
        # Create heatmap
        im = ax.imshow(similarity_matrix, cmap='RdYlBu_r', vmin=-1, vmax=1)
        
        # Add labels
        ax.set_xticks(range(len(layer_names)))
        ax.set_yticks(range(len(layer_names)))
        ax.set_xticklabels(layer_names, rotation=45, ha='right')
        ax.set_yticklabels(layer_names)
        
        # Add text annotations
        for i in range(len(layer_names)):
            for j in range(len(layer_names)):
                text = ax.text(j, i, f'{similarity_matrix[i, j]:.2f}',
                             ha="center", va="center", color="black" if abs(similarity_matrix[i, j]) < 0.5 else "white")
        
        ax.set_title('Layer Representation Similarity\n(Based on Probe Coefficient Patterns)')
        plt.colorbar(im, ax=ax, label='Cosine Similarity')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'layer_similarity_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_probe_coefficients(self, probe_results: Dict, layer_name: str, 
                               quantum_labels: List[str] = None) -> None:
        """
        Visualize probe coefficients to see what each layer focuses on
        
        Args:
            probe_results: Output from LayerWiseProber.probe_layer_representations
            layer_name: Which layer to visualize
            quantum_labels: Labels for quantum states
        """
        if layer_name not in probe_results:
            print(f"Layer {layer_name} not found in probe results")
            return
            
        probe = probe_results[layer_name]['probe']
        coefficients = probe.coef_
        r2_score = probe_results[layer_name]['r2_score']
        
        # Determine if we can reshape to matrix form
        matrix_dim = int(np.sqrt(len(coefficients))) if len(coefficients) == 4 or len(coefficients) == 16 else None
        
        fig, axes = plt.subplots(1, 2, figsize=(12*self.figsize_scale, 5*self.figsize_scale))
        
        # Linear coefficients
        axes[0].bar(range(len(coefficients)), coefficients, alpha=0.7)
        axes[0].set_xlabel('Feature Dimension')
        axes[0].set_ylabel('Probe Coefficient')
        axes[0].set_title(f'{layer_name} Probe Coefficients\nR² = {r2_score:.4f}')
        axes[0].grid(True, alpha=0.3)
        
        # Matrix form if applicable
        if matrix_dim and matrix_dim in [2, 4]:
            coeff_matrix = coefficients[:matrix_dim**2].reshape(matrix_dim, matrix_dim)
            im = axes[1].imshow(coeff_matrix, cmap='RdBu_r', vmin=-np.abs(coefficients).max(), 
                              vmax=np.abs(coefficients).max())
            axes[1].set_title(f'Coefficient Matrix Pattern')
            
            if quantum_labels and len(quantum_labels) >= matrix_dim**2:
                # Add quantum labels
                row_labels = [f'|{format(i, f"0{int(np.log2(matrix_dim))}b")}⟩' for i in range(matrix_dim)]
                col_labels = [f'⟨{format(i, f"0{int(np.log2(matrix_dim))}b")}|' for i in range(matrix_dim)]
                axes[1].set_xticks(range(matrix_dim))
                axes[1].set_yticks(range(matrix_dim))
                axes[1].set_xticklabels(col_labels)
                axes[1].set_yticklabels(row_labels)
            
            plt.colorbar(im, ax=axes[1])
        else:
            # Just show reshaped view if not square
            if len(coefficients) >= 64:
                # Show as heatmap
                coeff_2d = coefficients[:64].reshape(8, 8)
                im = axes[1].imshow(coeff_2d, cmap='RdBu_r')
                axes[1].set_title('Coefficient Heatmap (First 64 dims)')
                plt.colorbar(im, ax=axes[1])
            else:
                axes[1].axis('off')
                axes[1].text(0.5, 0.5, f'Layer dimension: {len(coefficients)}\nNot suitable for matrix view', 
                           ha='center', va='center', transform=axes[1].transAxes)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'probe_coefficients_{layer_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def create_probing_dashboard(self, probe_results: Dict, emergence_analysis: Dict,
                               similarity_analysis: Dict, quantum_labels: List[str] = None) -> None:
        """
        Create comprehensive probing analysis dashboard
        
        Args:
            probe_results: Results from layer probing
            emergence_analysis: Information emergence analysis
            similarity_analysis: Layer similarity analysis
            quantum_labels: Quantum state labels
        """
        print("Creating layer-wise probing dashboard...")
        
        # Main emergence plot
        self.plot_emergence_progression(emergence_analysis)
        
        # Similarity analysis
        self.plot_representation_similarity(similarity_analysis)
        
        # Individual layer coefficient plots
        layer_names = probe_results['metadata']['layer_names']
        print(f"Creating coefficient visualizations for {len(layer_names)} layers...")
        
        for layer_name in layer_names:
            if layer_name in probe_results:
                self.plot_probe_coefficients(probe_results, layer_name, quantum_labels)
        
        # Summary comparison plot
        self._plot_layer_comparison_summary(probe_results)
        
        # Create text report
        self._create_probing_report(probe_results, emergence_analysis)
        
        print(f"Probing dashboard complete! Results saved to {self.output_dir}/")
        
    def _plot_layer_comparison_summary(self, probe_results: Dict) -> None:
        """Create summary comparison of all layer probing results"""
        layer_names = probe_results['metadata']['layer_names']
        
        performance_data = []
        for layer_name in layer_names:
            if layer_name in probe_results:
                performance_data.append({
                    'layer': layer_name,
                    'r2': probe_results[layer_name]['r2_score'],
                    'mse': probe_results[layer_name]['mse']
                })
        
        # Add model final performance
        if 'model_final' in probe_results:
            performance_data.append({
                'layer': 'model_final',
                'r2': probe_results['model_final']['r2_score'],
                'mse': probe_results['model_final']['mse']
            })
        
        # Sort by R² score
        performance_data.sort(key=lambda x: x['r2'], reverse=True)
        
        fig, ax = plt.subplots(1, 1, figsize=(12*self.figsize_scale, 6*self.figsize_scale))
        
        layers = [p['layer'] for p in performance_data]
        r2_scores = [p['r2'] for p in performance_data]
        colors = ['gold' if layer == 'model_final' else 'skyblue' for layer in layers]
        
        bars = ax.bar(range(len(layers)), r2_scores, color=colors, alpha=0.8)
        ax.set_xlabel('Layer')
        ax.set_ylabel('R² Score')
        ax.set_title('Magic Prediction Capability by Layer\n(Higher = Better Magic Detection)')
        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels(layers, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, score in zip(bars, r2_scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Add legend
        ax.legend(['Intermediate Layers', 'Final Model'], loc='upper left')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'layer_performance_ranking.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def _create_probing_report(self, probe_results: Dict, emergence_analysis: Dict) -> None:
        """Create detailed text report of probing results"""
        
        report = """
LAYER-WISE PROBING ANALYSIS REPORT
=================================

This analysis reveals when and how magic prediction capability emerges through the transformer layers.

Key Findings:
"""
        
        # Emergence analysis
        if emergence_analysis['emergence_layer']:
            report += f"""
🚀 INFORMATION EMERGENCE:
   - Primary emergence layer: {emergence_analysis['emergence_layer']}
   - R² improvement at emergence: +{emergence_analysis['emergence_improvement']:.4f}
   - Best performing layer: {emergence_analysis['max_r2_layer']} (R² = {emergence_analysis['max_r2_score']:.4f})
"""
        
        # Layer performance details
        report += f"""
📊 LAYER PERFORMANCE BREAKDOWN:
"""
        
        for layer_data in emergence_analysis['performance_progression']:
            report += f"""
{layer_data['layer']}:
   - R² score: {layer_data['r2_score']:.4f}
   - MSE: {layer_data['mse']:.4f}  
   - Representation dimension: {layer_data['representation_dim']}
"""
        
        # Model comparison
        if 'model_final' in probe_results:
            model_r2 = probe_results['model_final']['r2_score']
            best_probe_r2 = emergence_analysis['max_r2_score']
            
            report += f"""
🎯 PROBE vs MODEL COMPARISON:
   - Final model R²: {model_r2:.4f}
   - Best probe R²: {best_probe_r2:.4f}
   - Probe efficiency: {(best_probe_r2/model_r2)*100:.1f}% of model performance
"""
            
            if best_probe_r2 > 0.8 * model_r2:
                report += "   → Linear probes capture most of the model's magic detection capability!\n"
            elif best_probe_r2 > 0.5 * model_r2:
                report += "   → Linear probes capture substantial magic detection patterns.\n"
            else:
                report += "   → Model uses complex non-linear patterns for magic detection.\n"
        
        report += f"""
📈 INTERPRETATION:
   - Higher R² scores indicate better linear separability of magic values
   - Large jumps in R² show where critical information processing occurs
   - The emergence layer reveals when the model "understands" quantum magic
   - Probe coefficients show which input features each layer prioritizes

Files Generated:
   - layer_emergence_progression.png: Shows R² and MSE evolution through layers
   - layer_similarity_matrix.png: Similarity between layer representations  
   - layer_performance_ranking.png: Ranking of layers by magic detection capability
   - probe_coefficients_*.png: Detailed coefficient patterns for each layer

💡 Next Steps:
   - Focus on the emergence layer for deeper analysis
   - Examine probe coefficients to understand what patterns the model learned
   - Compare attention patterns with probe coefficients for consistency
"""
        
        # Save report
        with open(self.output_dir / 'layer_probing_report.txt', 'w') as f:
            f.write(report)
            
        print(f"Detailed probing report saved to {self.output_dir}/layer_probing_report.txt")