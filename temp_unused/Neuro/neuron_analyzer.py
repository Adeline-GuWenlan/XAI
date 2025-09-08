import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class NeuronAnalyzer:
    """
    Analyzer for understanding neuron specialization in the MLP head
    """
    
    def __init__(self, model: nn.Module, device: torch.device, config: Dict = None):
        self.model = model
        self.device = device
        self.config = config or {}
        self.model.eval()
        
        # Store neuron activations
        self.neuron_activations = {}
        self.hooks = []
        
    def register_mlp_hooks(self):
        """Register hooks to capture MLP neuron activations"""
        self.neuron_activations = {}
        
        def create_activation_hook(layer_name):
            def hook(module, input, output):
                # Store activations
                self.neuron_activations[layer_name] = output.detach().cpu().numpy()
            return hook
        
        # Hook MLP layers based on your attention_enhanced architecture
        mlp_head = self.model.mlp_head
        
        if hasattr(mlp_head, 'mlp_head'):  # attention_enhanced type
            # Feature extractor
            if hasattr(mlp_head.mlp_head, 'feature_extractor'):
                feat_hook = mlp_head.mlp_head.feature_extractor.register_forward_hook(
                    create_activation_hook('feature_extractor')
                )
                self.hooks.append(feat_hook)
            
            # Post-attention layers
            if hasattr(mlp_head.mlp_head, 'post_attention'):
                # Hook individual layers in the sequential
                for i, layer in enumerate(mlp_head.mlp_head.post_attention):
                    if isinstance(layer, nn.Linear):
                        hook = layer.register_forward_hook(
                            create_activation_hook(f'post_attention_linear_{i}')
                        )
                        self.hooks.append(hook)
                        
    def remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        
    def extract_neuron_activations(self, rho_real: torch.Tensor, rho_imag: torch.Tensor) -> Dict:
        """
        Extract neuron activations from MLP layers
        
        Args:
            rho_real, rho_imag: Input density matrices [B, D, D]
            
        Returns:
            Dict with neuron activations for each MLP layer
        """
        self.register_mlp_hooks()
        
        try:
            # Forward pass to capture activations
            with torch.no_grad():
                predictions = self.model(rho_real, rho_imag)
            
            # Process activations
            processed_activations = {}
            for layer_name, activations in self.neuron_activations.items():
                processed_activations[layer_name] = {
                    'activations': activations,
                    'shape': activations.shape,
                    'mean_activation': np.mean(activations, axis=0),
                    'std_activation': np.std(activations, axis=0),
                    'sparsity': (activations == 0).mean(axis=0),  # Dead neuron detection
                    'max_activation': np.max(activations, axis=0)
                }
            
            processed_activations['predictions'] = predictions.cpu().numpy()
            
        finally:
            self.remove_hooks()
            
        return processed_activations
        
    def analyze_neuron_specialization(self, dataloader, max_samples: int = 150000) -> Dict:
        """
        Analyze neuron specialization patterns across the dataset
        
        Args:
            dataloader: DataLoader with quantum density matrices
            max_samples: Maximum samples to analyze
            
        Returns:
            Dict with neuron specialization analysis
        """
        print(f"Analyzing neuron specialization for up to {max_samples} samples...")
        
        all_activations = {}
        all_true_magic = []
        all_predictions = []
        sample_count = 0
        
        # Collect activations across dataset
        for batch_idx, (rho_real, rho_imag, true_magic) in enumerate(dataloader):
            if sample_count >= max_samples:
                break
                
            # Move to device
            rho_real = rho_real.to(self.device)
            rho_imag = rho_imag.to(self.device)
            
            # Limit batch size if needed
            batch_size = min(rho_real.shape[0], max_samples - sample_count)
            if batch_size < rho_real.shape[0]:
                rho_real = rho_real[:batch_size]
                rho_imag = rho_imag[:batch_size]
                true_magic = true_magic[:batch_size]
            
            # Extract activations
            batch_activations = self.extract_neuron_activations(rho_real, rho_imag)
            
            # Accumulate activations
            for layer_name, layer_data in batch_activations.items():
                if layer_name == 'predictions':
                    continue
                    
                if layer_name not in all_activations:
                    all_activations[layer_name] = []
                all_activations[layer_name].append(layer_data['activations'])
            
            all_true_magic.append(true_magic.numpy())
            all_predictions.append(batch_activations['predictions'])
            
            sample_count += batch_size
            if batch_idx % 20 == 0:
                print(f"  Processed {sample_count} samples...")
        
        # Concatenate all activations
        for layer_name in all_activations:
            all_activations[layer_name] = np.vstack(all_activations[layer_name])
            
        all_true_magic = np.concatenate(all_true_magic)
        all_predictions = np.vstack(all_predictions).flatten()
        
        print(f"Collected activations from {len(all_activations)} MLP layers")
        
        # Analyze neuron specialization
        specialization_results = {}
        
        for layer_name, activations in all_activations.items():
            print(f"  Analyzing {layer_name}...")
            
            num_neurons = activations.shape[1]
            
            # Analyze individual neurons
            neuron_analysis = []
            for neuron_idx in range(num_neurons):
                neuron_acts = activations[:, neuron_idx]
                
                # Compute correlation with magic values
                correlation = np.corrcoef(neuron_acts, all_true_magic)[0, 1]
                if np.isnan(correlation):
                    correlation = 0.0
                
                # Analyze activation patterns
                analysis = {
                    'neuron_idx': neuron_idx,
                    'correlation_with_magic': correlation,
                    'mean_activation': np.mean(neuron_acts),
                    'std_activation': np.std(neuron_acts),
                    'sparsity': (neuron_acts == 0).mean(),
                    'max_activation': np.max(neuron_acts),
                    'activation_range': np.ptp(neuron_acts),  # Peak-to-peak
                    'is_dead': np.all(neuron_acts == 0),
                    'high_magic_selectivity': self._compute_selectivity(neuron_acts, all_true_magic, threshold=0.3)
                }
                
                neuron_analysis.append(analysis)
            
            specialization_results[layer_name] = {
                'layer_name': layer_name,
                'num_neurons': num_neurons,
                'neuron_analysis': neuron_analysis,
                'activations': activations,
                'dead_neurons': sum(1 for n in neuron_analysis if n['is_dead']),
                'high_magic_neurons': sum(1 for n in neuron_analysis if abs(n['correlation_with_magic']) > 0.5),
                'layer_sparsity': (activations == 0).mean()
            }
        
        specialization_results['metadata'] = {
            'total_samples': sample_count,
            'true_magic': all_true_magic,
            'predictions': all_predictions
        }
        
        return specialization_results
        
    def _compute_selectivity(self, neuron_activations: np.ndarray, magic_values: np.ndarray, 
                           threshold: float = 0.3) -> float:
        """
        Compute how selectively a neuron responds to high magic values
        
        Args:
            neuron_activations: Activations for this neuron [samples]
            magic_values: True magic values [samples]  
            threshold: Magic value threshold for "high magic"
            
        Returns:
            Selectivity score (higher = more selective for high magic)
        """
        high_magic_mask = magic_values > threshold
        low_magic_mask = magic_values <= threshold
        
        if not np.any(high_magic_mask) or not np.any(low_magic_mask):
            return 0.0
        
        high_magic_activation = np.mean(neuron_activations[high_magic_mask])
        low_magic_activation = np.mean(neuron_activations[low_magic_mask])
        
        # Selectivity as ratio (avoiding division by zero)
        if low_magic_activation == 0:
            return float('inf') if high_magic_activation > 0 else 0.0
        
        selectivity = high_magic_activation / (low_magic_activation + 1e-8)
        return selectivity
        
    def find_magic_detector_neurons(self, specialization_results: Dict, 
                                  correlation_threshold: float = 0.5) -> Dict:
        """
        Identify neurons that act as magic value detectors
        
        Args:
            specialization_results: Output from analyze_neuron_specialization
            correlation_threshold: Minimum correlation to be considered a detector
            
        Returns:
            Dict with magic detector neurons
        """
        magic_detectors = {}
        
        for layer_name, layer_data in specialization_results.items():
            if layer_name == 'metadata':
                continue
                
            detectors = []
            for neuron in layer_data['neuron_analysis']:
                abs_corr = abs(neuron['correlation_with_magic'])
                if abs_corr > correlation_threshold:
                    detector_info = {
                        'neuron_idx': neuron['neuron_idx'],
                        'correlation': neuron['correlation_with_magic'],
                        'detector_type': 'high_magic' if neuron['correlation_with_magic'] > 0 else 'low_magic',
                        'selectivity': neuron['high_magic_selectivity'],
                        'activation_stats': {
                            'mean': neuron['mean_activation'],
                            'std': neuron['std_activation'],
                            'max': neuron['max_activation']
                        }
                    }
                    detectors.append(detector_info)
            
            # Sort by absolute correlation
            detectors.sort(key=lambda x: abs(x['correlation']), reverse=True)
            
            magic_detectors[layer_name] = {
                'detectors': detectors,
                'num_detectors': len(detectors),
                'high_magic_detectors': len([d for d in detectors if d['detector_type'] == 'high_magic']),
                'low_magic_detectors': len([d for d in detectors if d['detector_type'] == 'low_magic'])
            }
        
        return magic_detectors