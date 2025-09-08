import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap

class MoEVisualizer:
    """Visualizer for MoE quantum magic prediction model analysis"""
    
    def __init__(self, output_dir: str, figsize_scale: float = 1.0):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.figsize_scale = figsize_scale
        
        # Set up style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create custom colormaps
        self.attention_cmap = LinearSegmentedColormap.from_list(
            'attention', ['white', 'lightblue', 'darkblue', 'red']
        )
        self.expert_cmap = LinearSegmentedColormap.from_list(
            'expert', ['white', 'lightgreen', 'darkgreen', 'purple']
        )
    
    def plot_moe_expert_analysis(self, sample_analysis: Dict) -> None:
        """Plot comprehensive MoE expert analysis for a sample"""
        sample_idx = sample_analysis['sample_idx']
        true_magic = sample_analysis['true_magic']
        pred_magic = sample_analysis['predicted_magic']
        expert_data = sample_analysis['expert_analysis']
        
        if expert_data['type'] != 'moe':
            print(f"Sample {sample_idx} does not use MoE architecture")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(15*self.figsize_scale, 10*self.figsize_scale))
        fig.suptitle(f'MoE Analysis - Sample {sample_idx}\n'
                    f'True Magic: {true_magic:.4f}, Predicted: {pred_magic:.4f}', 
                    fontsize=16)
        
        # 1. Expert gating weights
        expert_weights = expert_data['sample_gating_weights']
        expert_outputs = expert_data['sample_expert_outputs']
        num_experts = len(expert_weights)
        
        colors = plt.cm.Set3(np.linspace(0, 1, num_experts))
        
        axes[0, 0].bar(range(num_experts), expert_weights, color=colors, alpha=0.7)
        axes[0, 0].set_title('Expert Gating Weights')
        axes[0, 0].set_xlabel('Expert Index')
        axes[0, 0].set_ylabel('Gating Weight')
        axes[0, 0].set_xticks(range(num_experts))
        
        # Highlight dominant expert
        dominant_expert = expert_data['dominant_expert']
        axes[0, 0].bar(dominant_expert, expert_weights[dominant_expert], 
                      color='red', alpha=0.8, label=f'Dominant (Expert {dominant_expert})')
        axes[0, 0].legend()
        
        # 2. Expert individual outputs
        axes[0, 1].bar(range(num_experts), expert_outputs, color=colors, alpha=0.7)
        axes[0, 1].axhline(y=pred_magic, color='red', linestyle='--', 
                          label=f'Final Prediction: {pred_magic:.4f}')
        axes[0, 1].axhline(y=true_magic, color='green', linestyle='--', 
                          label=f'True Value: {true_magic:.4f}')
        axes[0, 1].set_title('Expert Individual Outputs')
        axes[0, 1].set_xlabel('Expert Index')
        axes[0, 1].set_ylabel('Expert Output')
        axes[0, 1].set_xticks(range(num_experts))
        axes[0, 1].legend()
        
        # 3. Expert contribution (weight * output)
        contributions = expert_weights * expert_outputs
        axes[0, 2].bar(range(num_experts), contributions, color=colors, alpha=0.7)
        axes[0, 2].set_title('Expert Contributions to Final Prediction')
        axes[0, 2].set_xlabel('Expert Index')
        axes[0, 2].set_ylabel('Weighted Contribution')
        axes[0, 2].set_xticks(range(num_experts))
        
        # 4. Attention patterns for this sample
        if 'attention_analysis' in sample_analysis:
            self._plot_attention_summary(axes[1, 0], sample_analysis['attention_analysis'])
        else:
            axes[1, 0].text(0.5, 0.5, 'No Attention Analysis', ha='center', va='center')
            axes[1, 0].set_title('Attention Analysis')
        
        # 5. Expert agreement analysis
        expert_agreement = expert_data['expert_agreement']
        weighted_pred = expert_data['weighted_prediction']
        
        axes[1, 1].text(0.1, 0.8, f'Expert Agreement (std): {expert_agreement:.4f}', 
                       transform=axes[1, 1].transAxes, fontsize=12)
        axes[1, 1].text(0.1, 0.7, f'Weighted Prediction: {weighted_pred:.4f}', 
                       transform=axes[1, 1].transAxes, fontsize=12)
        axes[1, 1].text(0.1, 0.6, f'Dominant Expert: {dominant_expert}', 
                       transform=axes[1, 1].transAxes, fontsize=12)
        axes[1, 1].text(0.1, 0.5, f'Dominant Weight: {expert_data["dominant_weight"]:.4f}', 
                       transform=axes[1, 1].transAxes, fontsize=12)
        
        agreement_level = "High" if expert_agreement < 0.1 else "Medium" if expert_agreement < 0.5 else "Low"
        axes[1, 1].text(0.1, 0.4, f'Agreement Level: {agreement_level}', 
                       transform=axes[1, 1].transAxes, fontsize=12, 
                       color='green' if agreement_level == 'High' else 'orange' if agreement_level == 'Medium' else 'red')
        
        axes[1, 1].set_title('Expert Agreement Analysis')
        axes[1, 1].axis('off')
        
        # 6. Prediction error breakdown
        total_error = abs(pred_magic - true_magic)
        expert_errors = np.abs(expert_outputs - true_magic)
        
        axes[1, 2].bar(range(num_experts), expert_errors, color=colors, alpha=0.7, 
                      label='Individual Expert Errors')
        axes[1, 2].axhline(y=total_error, color='red', linestyle='--', 
                          label=f'Final Model Error: {total_error:.4f}')
        axes[1, 2].set_title('Expert Error Analysis')
        axes[1, 2].set_xlabel('Expert Index')
        axes[1, 2].set_ylabel('Absolute Error')
        axes[1, 2].set_xticks(range(num_experts))
        axes[1, 2].legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'moe_analysis_sample_{sample_idx}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_attention_summary(self, ax, attention_analysis: Dict) -> None:
        """Plot summary of attention patterns"""
        layer_patterns = attention_analysis['layer_patterns']
        num_layers = len(layer_patterns)
        
        # Plot attention entropy evolution
        entropies = [pattern['attention_entropy'] for pattern in layer_patterns]
        ax.plot(range(num_layers), entropies, 'o-', linewidth=2, markersize=8)
        ax.set_title('Attention Entropy Evolution')
        ax.set_xlabel('Layer')
        ax.set_ylabel('Attention Entropy')
        ax.grid(True, alpha=0.3)
        
        # Highlight trend
        if len(entropies) > 1:
            trend = 'increasing' if entropies[-1] > entropies[0] else 'decreasing'
            ax.text(0.7, 0.9, f'Trend: {trend}', transform=ax.transAxes, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    
    def plot_expert_usage_distribution(self, analysis_results: Dict) -> None:
        """Plot overall expert usage distribution across dataset"""
        expert_stats = analysis_results.get('expert_statistics', {})
        
        if 'avg_expert_usage' not in expert_stats:
            print("No expert usage data available")
            return
        
        avg_usage = expert_stats['avg_expert_usage']
        usage_std = expert_stats['expert_usage_std']
        num_experts = len(avg_usage)
        
        fig, axes = plt.subplots(1, 3, figsize=(15*self.figsize_scale, 5*self.figsize_scale))
        
        # 1. Average expert usage
        colors = plt.cm.Set3(np.linspace(0, 1, num_experts))
        bars = axes[0].bar(range(num_experts), avg_usage, yerr=usage_std, 
                          color=colors, alpha=0.7, capsize=5)
        axes[0].set_title('Average Expert Usage Distribution')
        axes[0].set_xlabel('Expert Index')
        axes[0].set_ylabel('Average Gating Weight')
        axes[0].set_xticks(range(num_experts))
        
        # Add usage percentages
        for i, (usage, bar) in enumerate(zip(avg_usage, bars)):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + usage_std[i] + 0.01,
                        f'{usage:.1%}', ha='center', va='bottom')
        
        # 2. Expert diversity metrics
        if 'expert_diversity' in expert_stats:
            diversity = expert_stats['expert_diversity']
            metrics = ['Mean Entropy', 'Normalized Entropy', 'Balance Score']
            values = [
                diversity['mean_gating_entropy'],
                diversity['normalized_entropy'],
                1.0 - np.std(avg_usage) / np.mean(avg_usage) if np.mean(avg_usage) > 0 else 0
            ]
            
            axes[1].bar(metrics, values, color=['skyblue', 'lightgreen', 'lightcoral'])
            axes[1].set_title('Expert Diversity Metrics')
            axes[1].set_ylabel('Score')
            axes[1].tick_params(axis='x', rotation=45)
            
            # Add value labels
            for i, (metric, value) in enumerate(zip(metrics, values)):
                axes[1].text(i, value + 0.02, f'{value:.3f}', ha='center', va='bottom')
        
        # 3. Expert dominance analysis
        summary = analysis_results.get('summary', {})
        if 'expert_stats' in summary:
            expert_stats_summary = summary['expert_stats']
            dominant_counts = expert_stats_summary['dominant_expert_distribution']
            
            axes[2].pie(dominant_counts, labels=[f'Expert {i}' for i in range(len(dominant_counts))],
                       autopct='%1.1f%%', colors=colors)
            axes[2].set_title('Expert Dominance Distribution\n(Samples where expert has highest weight)')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'expert_usage_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_attention_moe_correlation(self, analysis_results: Dict) -> None:
        """Plot correlation between attention patterns and expert usage"""
        sample_results = analysis_results.get('sample_results', [])
        
        if not sample_results:
            return
        
        # Extract data for correlation analysis
        attention_entropies = []
        dominant_experts = []
        expert_agreements = []
        prediction_errors = []
        
        for result in sample_results:
            if 'attention_analysis' in result and 'expert_analysis' in result:
                if result['expert_analysis']['type'] == 'moe':
                    attention_entropies.append(
                        result['attention_analysis']['attention_evolution']['final_entropy']
                    )
                    dominant_experts.append(result['expert_analysis']['dominant_expert'])
                    expert_agreements.append(result['expert_analysis']['expert_agreement'])
                    prediction_errors.append(
                        abs(result['predicted_magic'] - result['true_magic'])
                    )
        
        if not attention_entropies:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12*self.figsize_scale, 10*self.figsize_scale))
        
        # 1. Attention entropy vs Expert agreement
        axes[0, 0].scatter(attention_entropies, expert_agreements, alpha=0.6, s=30)
        axes[0, 0].set_xlabel('Final Attention Entropy')
        axes[0, 0].set_ylabel('Expert Agreement (lower = more agreement)')
        axes[0, 0].set_title('Attention Entropy vs Expert Agreement')
        
        # Add correlation coefficient
        if len(attention_entropies) > 1:
            corr = np.corrcoef(attention_entropies, expert_agreements)[0, 1]
            axes[0, 0].text(0.05, 0.95, f'Corr: {corr:.3f}', transform=axes[0, 0].transAxes,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="white"))
        
        # 2. Attention entropy by dominant expert
        unique_experts = sorted(set(dominant_experts))
        entropy_by_expert = [
            [ent for ent, exp in zip(attention_entropies, dominant_experts) if exp == expert]
            for expert in unique_experts
        ]
        
        axes[0, 1].boxplot(entropy_by_expert, labels=[f'Expert {i}' for i in unique_experts])
        axes[0, 1].set_title('Attention Entropy Distribution by Dominant Expert')
        axes[0, 1].set_xlabel('Dominant Expert')
        axes[0, 1].set_ylabel('Final Attention Entropy')
        
        # 3. Expert agreement vs Prediction error
        axes[1, 0].scatter(expert_agreements, prediction_errors, alpha=0.6, s=30)
        axes[1, 0].set_xlabel('Expert Agreement (lower = more agreement)')
        axes[1, 0].set_ylabel('Prediction Error')
        axes[1, 0].set_title('Expert Agreement vs Prediction Accuracy')
        
        if len(expert_agreements) > 1:
            corr = np.corrcoef(expert_agreements, prediction_errors)[0, 1]
            axes[1, 0].text(0.05, 0.95, f'Corr: {corr:.3f}', transform=axes[1, 0].transAxes,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="white"))
        
        # 4. Prediction error by dominant expert
        error_by_expert = [
            [err for err, exp in zip(prediction_errors, dominant_experts) if exp == expert]
            for expert in unique_experts
        ]
        
        axes[1, 1].boxplot(error_by_expert, labels=[f'Expert {i}' for i in unique_experts])
        axes[1, 1].set_title('Prediction Error Distribution by Dominant Expert')
        axes[1, 1].set_xlabel('Dominant Expert')
        axes[1, 1].set_ylabel('Prediction Error')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'attention_moe_correlation.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_moe_analysis_dashboard(self, analysis_results: Dict) -> None:
        """Create comprehensive MoE analysis dashboard"""
        print("Creating MoE analysis dashboard...")
        
        # Plot overall expert usage
        self.plot_expert_usage_distribution(analysis_results)
        
        # Plot attention-MoE correlations
        self.plot_attention_moe_correlation(analysis_results)
        
        # Create individual sample analyses (first 10)
        sample_results = analysis_results.get('sample_results', [])
        max_individual = min(10, len(sample_results))
        
        print(f"Creating individual MoE analyses for {max_individual} samples...")
        for i in range(max_individual):
            self.plot_moe_expert_analysis(sample_results[i])
        
        # Create summary report
        self._create_moe_summary_report(analysis_results)
        
        print(f"MoE analysis dashboard complete! Results saved to {self.output_dir}/")
    
    def _create_moe_summary_report(self, analysis_results: Dict) -> None:
        """Create comprehensive MoE analysis report"""
        summary = analysis_results.get('summary', {})
        expert_stats = analysis_results.get('expert_statistics', {})
        
        report = f"""
MoE QUANTUM MAGIC PREDICTION MODEL ANALYSIS
==========================================

Dataset Overview:
- Total samples analyzed: {summary.get('total_samples', 0)}
- Average prediction error (MAE): {summary.get('prediction_stats', {}).get('mae', 0):.4f}
- Prediction range: {summary.get('prediction_stats', {}).get('prediction_range', [0, 0])}
- True magic range: {summary.get('prediction_stats', {}).get('true_range', [0, 0])}

Expert Usage Analysis:
"""
        
        if 'avg_expert_usage' in expert_stats:
            avg_usage = expert_stats['avg_expert_usage']
            for i, usage in enumerate(avg_usage):
                report += f"- Expert {i}: {usage:.1%} average usage\n"
            
            # Expert diversity
            if 'expert_diversity' in expert_stats:
                diversity = expert_stats['expert_diversity']
                report += f"""
Expert Diversity Metrics:
- Mean gating entropy: {diversity['mean_gating_entropy']:.3f}
- Normalized entropy: {diversity['normalized_entropy']:.3f} (max = 1.0 for uniform usage)
- Expert balance score: {1.0 - np.std(avg_usage) / np.mean(avg_usage) if np.mean(avg_usage) > 0 else 0:.3f}
"""
        
        if 'expert_stats' in summary:
            expert_summary = summary['expert_stats']
            report += f"""
Expert Performance Analysis:
- Mean expert agreement: {expert_summary.get('mean_expert_agreement', 0):.4f} (lower = more agreement)
- Expert balance in dominance: {expert_summary.get('expert_balance', 0):.3f}
- Dominant expert distribution: {expert_summary.get('dominant_expert_distribution', [])}
"""
        
        if 'attention_stats' in summary:
            attention_summary = summary['attention_stats']
            report += f"""
Attention Pattern Analysis:
- Mean final layer attention entropy: {attention_summary.get('mean_final_entropy', 0):.3f}
- Attention entropy std: {attention_summary.get('std_final_entropy', 0):.3f}
"""
        
        report += """
Key Findings:
1. Expert Usage: Check if experts are being utilized equally or if there's specialization
2. Expert Agreement: Lower values indicate experts agree more on predictions
3. Attention Patterns: Higher entropy indicates more distributed attention across matrix elements
4. Model Behavior: Correlation between attention patterns and expert selection

Files Generated:
- expert_usage_analysis.png: Overall expert usage and diversity metrics
- attention_moe_correlation.png: Correlation between attention and expert behavior
- moe_analysis_sample_*.png: Detailed analysis for individual samples

Interpretation Guide:
- Balanced expert usage (close to uniform) suggests good load balancing
- High expert agreement with low prediction error indicates consistent, accurate experts
- Attention entropy correlations help understand if different attention patterns trigger different experts
- Expert dominance patterns reveal if certain experts specialize in specific types of quantum states
"""
        
        with open(self.output_dir / 'moe_analysis_report.txt', 'w') as f:
            f.write(report)
        
        print(f"MoE analysis report saved to {self.output_dir}/moe_analysis_report.txt")