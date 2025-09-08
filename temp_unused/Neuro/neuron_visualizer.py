import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path

class NeuronVisualizer:
    """Visualizer for neuron specialization analysis"""
    
    def __init__(self, output_dir: str, figsize_scale: float = 1.0):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.figsize_scale = figsize_scale
        
    def plot_neuron_specialization_overview(self, specialization_results: Dict) -> None:
        """
        Plot overview of neuron specialization across MLP layers
        """
        layer_names = [name for name in specialization_results.keys() if name != 'metadata']
        
        # Collect statistics
        dead_neurons = []
        high_magic_neurons = []
        total_neurons = []
        layer_sparsity = []
        
        for layer_name in layer_names:
            layer_data = specialization_results[layer_name]
            dead_neurons.append(layer_data['dead_neurons'])
            high_magic_neurons.append(layer_data['high_magic_neurons'])
            total_neurons.append(layer_data['num_neurons'])
            layer_sparsity.append(layer_data['layer_sparsity'])
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14*self.figsize_scale, 10*self.figsize_scale))
        
        # Dead neurons
        ax1.bar(range(len(layer_names)), dead_neurons, alpha=0.7, color='red')
        ax1.set_xlabel('MLP Layer')
        ax1.set_ylabel('Number of Dead Neurons')
        ax1.set_title('Dead Neuron Detection')
        ax1.set_xticks(range(len(layer_names)))
        ax1.set_xticklabels(layer_names, rotation=45)
        
        # Magic detector neurons
        ax2.bar(range(len(layer_names)), high_magic_neurons, alpha=0.7, color='blue')
        ax2.set_xlabel('MLP Layer')
        ax2.set_ylabel('Number of Magic Detector Neurons')
        ax2.set_title('Magic Detector Neurons (|correlation| > 0.5)')
        ax2.set_xticks(range(len(layer_names)))
        ax2.set_xticklabels(layer_names, rotation=45)
        
        # Layer sparsity
        ax3.bar(range(len(layer_names)), layer_sparsity, alpha=0.7, color='green')
        ax3.set_xlabel('MLP Layer')
        ax3.set_ylabel('Activation Sparsity')
        ax3.set_title('Layer Activation Sparsity')
        ax3.set_xticks(range(len(layer_names)))
        ax3.set_xticklabels(layer_names, rotation=45)
        
        # Neuron utilization
        active_neurons = [total - dead for total, dead in zip(total_neurons, dead_neurons)]
        utilization = [active/total for active, total in zip(active_neurons, total_neurons)]
        
        ax4.bar(range(len(layer_names)), utilization, alpha=0.7, color='orange')
        ax4.set_xlabel('MLP Layer')
        ax4.set_ylabel('Neuron Utilization Rate')
        ax4.set_title('Active Neuron Utilization')
        ax4.set_xticks(range(len(layer_names)))
        ax4.set_xticklabels(layer_names, rotation=45)
        ax4.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'neuron_specialization_overview.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_magic_detector_neurons(self, magic_detectors: Dict, layer_name: str) -> None:
        """
        Plot detailed analysis of magic detector neurons for a specific layer
        """
        if layer_name not in magic_detectors:
            return
            
        layer_data = magic_detectors[layer_name]
        detectors = layer_data['detectors']
        
        if not detectors:
            print(f"No magic detectors found in {layer_name}")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(12*self.figsize_scale, 10*self.figsize_scale))
        
        # Detector correlations
        correlations = [d['correlation'] for d in detectors]
        detector_types = [d['detector_type'] for d in detectors]
        colors = ['red' if t == 'high_magic' else 'blue' for t in detector_types]
        
        axes[0, 0].bar(range(len(correlations)), correlations, color=colors, alpha=0.7)
        axes[0, 0].set_xlabel('Detector Neuron')
        axes[0, 0].set_ylabel('Correlation with Magic')
        axes[0, 0].set_title(f'{layer_name}: Magic Detector Correlations')
        axes[0, 0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Selectivity scores
        selectivities = [d['selectivity'] for d in detectors]
        axes[0, 1].bar(range(len(selectivities)), selectivities, color=colors, alpha=0.7)
        axes[0, 1].set_xlabel('Detector Neuron')
        axes[0, 1].set_ylabel('High Magic Selectivity')
        axes[0, 1].set_title('Selectivity for High Magic Values')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Activation statistics
        mean_acts = [d['activation_stats']['mean'] for d in detectors]
        max_acts = [d['activation_stats']['max'] for d in detectors]
        
        x_pos = np.arange(len(detectors))
        width = 0.35
        
        axes[1, 0].bar(x_pos - width/2, mean_acts, width, label='Mean', alpha=0.7)
        axes[1, 0].bar(x_pos + width/2, max_acts, width, label='Max', alpha=0.7)
        axes[1, 0].set_xlabel('Detector Neuron')
        axes[1, 0].set_ylabel('Activation Level')
        axes[1, 0].set_title('Activation Statistics')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Detector type distribution
        high_magic_count = layer_data['high_magic_detectors']
        low_magic_count = layer_data['low_magic_detectors']
        
        axes[1, 1].pie([high_magic_count, low_magic_count], 
                      labels=['High Magic Detectors', 'Low Magic Detectors'],
                      colors=['red', 'blue'], autopct='%1.1f%%', alpha=0.7)
        axes[1, 1].set_title('Detector Type Distribution')
        
        plt.suptitle(f'{layer_name}: Magic Detector Neuron Analysis', fontsize=16)
        plt.tight_layout()
        plt.savefig(self.output_dir / f'magic_detectors_{layer_name}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def create_neuron_dashboard(self, specialization_results: Dict, magic_detectors: Dict) -> None:
        """Create comprehensive neuron analysis dashboard"""
        print("Creating neuron specialization dashboard...")
        
        # Overview plot
        self.plot_neuron_specialization_overview(specialization_results)
        
        # Individual layer detector analysis
        layer_names = [name for name in magic_detectors.keys() if isinstance(magic_detectors[name], dict)]
        
        for layer_name in layer_names:
            if magic_detectors[layer_name]['num_detectors'] > 0:
                self.plot_magic_detector_neurons(magic_detectors, layer_name)
        
        # Create summary report
        self._create_neuron_report(specialization_results, magic_detectors)
        
        print(f"Neuron dashboard complete! Results saved to {self.output_dir}/")
        
    def _create_neuron_report(self, specialization_results: Dict, magic_detectors: Dict) -> None:
        """Create detailed neuron analysis report"""
        
        report = """
NEURON SPECIALIZATION ANALYSIS REPORT
====================================

This analysis reveals how individual neurons in the MLP head specialize for quantum magic detection.

"""
        
        # Overall statistics
        layer_names = [name for name in specialization_results.keys() if name != 'metadata']
        total_neurons = sum(specialization_results[name]['num_neurons'] for name in layer_names)
        total_dead = sum(specialization_results[name]['dead_neurons'] for name in layer_names)
        total_detectors = sum(magic_detectors[name]['num_detectors'] for name in layer_names 
                            if name in magic_detectors and isinstance(magic_detectors[name], dict))
        
        report += f"""
📊 OVERALL STATISTICS:
   - Total neurons analyzed: {total_neurons}
   - Dead neurons: {total_dead} ({(total_dead/total_neurons)*100:.1f}%)
   - Magic detector neurons: {total_detectors} ({(total_detectors/total_neurons)*100:.1f}%)
   - Active utilization: {((total_neurons-total_dead)/total_neurons)*100:.1f}%

"""
        
        # Layer-by-layer breakdown
        report += "🔍 LAYER-BY-LAYER BREAKDOWN:\n"
        
        for layer_name in layer_names:
            layer_data = specialization_results[layer_name]
            detector_data = magic_detectors.get(layer_name, {'num_detectors': 0, 'high_magic_detectors': 0, 'low_magic_detectors': 0})
            
            report += f"""
{layer_name}:
   - Total neurons: {layer_data['num_neurons']}
   - Dead neurons: {layer_data['dead_neurons']}
   - Magic detectors: {detector_data['num_detectors']} (High: {detector_data.get('high_magic_detectors', 0)}, Low: {detector_data.get('low_magic_detectors', 0)})
   - Layer sparsity: {layer_data['layer_sparsity']:.3f}
   - Utilization: {((layer_data['num_neurons'] - layer_data['dead_neurons'])/layer_data['num_neurons'])*100:.1f}%
"""
        
        report += """
🎯 INTERPRETATION:
   - Dead neurons indicate model efficiency (unused capacity)
   - Magic detector neurons show specialized quantum magic detection
   - High correlation neurons focus on specific magic value ranges
   - Sparsity patterns reveal selective activation strategies

📈 INSIGHTS:
   - If many neurons are dead, the model may be over-parameterized
   - Specialized detectors suggest the model learned meaningful quantum patterns
   - Layer-wise detector distribution shows information processing hierarchy

Files Generated:
   - neuron_specialization_overview.png: Overall neuron statistics
   - magic_detectors_*.png: Detailed detector analysis per layer
"""
        
        with open(self.output_dir / 'neuron_analysis_report.txt', 'w') as f:
            f.write(report)
            
        print(f"Neuron analysis report saved to {self.output_dir}/neuron_analysis_report.txt")