import torch
import torch.nn as nn
import numpy as np
import sys
import os
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add paths for model architecture
sys.path.append('/Users/guwenlan/Desktop/XAI')
sys.path.append('/Users/guwenlan/Desktop/XAI/Model_Archi')

from Model_Archi.updated_training_model import CompleteQuantumMagicPredictor
from Model_Archi.quantum_dataset import QuantumDataset

class EncoderAttentionAnalyzer:
    """Specialized analyzer for encoder attention patterns in quantum magic prediction model"""
    
    def __init__(self, model: nn.Module, device: torch.device, config: Dict = None):
        self.model = model
        self.device = device
        self.config = config or {}
        self.model.eval()
        
    def extract_attention_weights(self, rho_real: torch.Tensor, rho_imag: torch.Tensor) -> Dict:
        """
        Extract attention weights from all transformer encoder layers
        
        Args:
            rho_real, rho_imag: Input density matrix components [B, D, D]
            
        Returns:
            Dict containing predictions and attention weights for all layers
        """
        with torch.no_grad():
            # Get predictions and attention weights
            prediction, attention_weights = self.model(rho_real, rho_imag, return_attention=True)
            
            # Convert to numpy for easier handling
            result = {
                'prediction': prediction.cpu().numpy(),
                'attention_weights': []
            }
            
            # Process attention weights from each layer
            for layer_idx, attn in enumerate(attention_weights):
                # attn shape: [B, num_heads, seq_len, seq_len]
                layer_attn = {
                    'layer': layer_idx,
                    'weights': attn.cpu().numpy(),  # [B, num_heads, seq_len, seq_len]
                    'shape': attn.shape,
                    'cls_attention': attn[:, :, 0, :].cpu().numpy(),  # CLS token attention [B, num_heads, seq_len]
                    'avg_head_weights': attn.mean(dim=1).cpu().numpy(),  # Average across heads [B, seq_len, seq_len]
                }
                result['attention_weights'].append(layer_attn)
                
        return result
        
    def analyze_cls_attention_patterns(self, attention_data: Dict, sample_idx: int = 0) -> Dict:
        """
        Analyze CLS token attention patterns across layers
        
        Args:
            attention_data: Output from extract_attention_weights
            sample_idx: Which sample to analyze
            
        Returns:
            Dict with CLS attention analysis
        """
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
                'attention_sparsity': (matrix_attn < 0.01).mean(),  # How sparse the attention is
                'max_attention_value': matrix_attn_reshaped.mean(axis=0).max(),
            }
            cls_patterns.append(layer_analysis)
            
        return {
            'sample_idx': sample_idx,
            'prediction': attention_data['prediction'][sample_idx],
            'matrix_dim': matrix_dim,
            'cls_patterns': cls_patterns
        }
        
    def analyze_head_specialization(self, attention_data: Dict) -> Dict:
        """
        Analyze if different attention heads specialize in different patterns across layers
        
        Args:
            attention_data: Output from extract_attention_weights
            
        Returns:
            Dict with head specialization analysis for each layer
        """
        num_layers = len(attention_data['attention_weights'])
        num_heads = attention_data['attention_weights'][0]['weights'].shape[1]
        batch_size = attention_data['attention_weights'][0]['weights'].shape[0]
        
        layer_head_analysis = {}
        
        for layer_idx, layer_data in enumerate(attention_data['attention_weights']):
            layer_heads = []
            
            for head_idx in range(num_heads):
                # Get attention patterns for this head across all samples
                head_attn = layer_data['weights'][:, head_idx, :, :]  # [B, seq_len, seq_len]
                
                # Focus on CLS attention patterns
                cls_attn = head_attn[:, 0, 1:]  # [B, matrix_elements] - CLS to matrix elements
                
                # Compute head statistics
                head_stats = {
                    'layer': layer_idx,
                    'head': head_idx,
                    'mean_cls_attention': cls_attn.mean(axis=0),  # Average CLS attention pattern
                    'std_cls_attention': cls_attn.std(axis=0),    # Variance in CLS attention
                    'attention_sparsity': (cls_attn < 0.01).mean(),  # How sparse the attention is
                    'max_attention_position': np.argmax(cls_attn.mean(axis=0)),
                    'attention_concentration': np.max(cls_attn.mean(axis=0)),  # Peak attention value
                    'attention_entropy': np.mean([-np.sum(sample_attn * np.log(sample_attn + 1e-10)) 
                                                 for sample_attn in cls_attn]),
                }
                layer_heads.append(head_stats)
                
            layer_head_analysis[layer_idx] = layer_heads
            
        return {
            'num_layers': num_layers,
            'num_heads': num_heads,
            'layer_head_specialization': layer_head_analysis
        }
        
    def aggregate_head_specialization(self, all_data: List[Dict]) -> Dict:
        """Aggregate head specialization analysis across all samples"""
        if not all_data:
            return {}
            
        num_layers = len(all_data[0]['attention_weights'])
        num_heads = all_data[0]['attention_weights'][0]['weights'].shape[1]
        
        # Collect CLS attention patterns for each head in each layer
        layer_head_patterns = {}
        
        for layer_idx in range(num_layers):
            layer_head_patterns[layer_idx] = {}
            
            for head_idx in range(num_heads):
                patterns = []
                entropies = []
                sparsities = []
                max_positions = []
                
                for sample in all_data:
                    # Get CLS attention for this head [1, seq_len]
                    cls_attn = sample['attention_weights'][layer_idx]['cls_attention'][0, head_idx, 1:]
                    patterns.append(cls_attn)
                    
                    # Compute entropy
                    entropy = -np.sum(cls_attn * np.log(cls_attn + 1e-10))
                    entropies.append(entropy)
                    
                    # Compute sparsity
                    sparsity = (cls_attn < 0.01).mean()
                    sparsities.append(sparsity)
                    
                    # Max attention position
                    max_positions.append(np.argmax(cls_attn))
                
                layer_head_patterns[layer_idx][head_idx] = {
                    'mean_pattern': np.mean(patterns, axis=0),
                    'std_pattern': np.std(patterns, axis=0),
                    'consistency': 1.0 - np.mean(np.std(patterns, axis=0)),  # How consistent across samples
                    'mean_entropy': np.mean(entropies),
                    'std_entropy': np.std(entropies),
                    'mean_sparsity': np.mean(sparsities),
                    'std_sparsity': np.std(sparsities),
                    'most_common_max_position': np.bincount(max_positions).argmax(),
                    'position_consistency': np.bincount(max_positions).max() / len(max_positions),
                }
                
        return layer_head_patterns
        
    def analyze_layer_evolution(self, all_data: List[Dict]) -> Dict:
        """Analyze how attention patterns evolve through encoder layers"""
        if not all_data:
            return {}
            
        num_layers = len(all_data[0]['attention_weights'])
        
        layer_stats = []
        for layer_idx in range(num_layers):
            # Collect attention statistics for this layer across all samples
            entropies = []
            sparsities = []
            max_attentions = []
            attention_concentrations = []
            
            for sample in all_data:
                # Average attention weights across heads [seq_len, seq_len]
                avg_attn = sample['attention_weights'][layer_idx]['avg_head_weights'][0]
                
                # Compute entropy of CLS attention
                cls_attn = avg_attn[0, 1:]  # CLS to matrix elements
                entropy = -np.sum(cls_attn * np.log(cls_attn + 1e-10))
                entropies.append(entropy)
                
                # Compute sparsity
                sparsity = (cls_attn < 0.01).mean()
                sparsities.append(sparsity)
                
                # Max attention value
                max_attentions.append(np.max(cls_attn))
                
                # Attention concentration (how concentrated the attention is)
                concentration = np.sum(cls_attn ** 2) / (np.sum(cls_attn) ** 2)
                attention_concentrations.append(concentration)
                
            layer_stats.append({
                'layer': layer_idx,
                'mean_entropy': np.mean(entropies),
                'std_entropy': np.std(entropies),
                'mean_sparsity': np.mean(sparsities),
                'std_sparsity': np.std(sparsities),
                'mean_max_attention': np.mean(max_attentions),
                'std_max_attention': np.std(max_attentions),
                'mean_concentration': np.mean(attention_concentrations),
                'std_concentration': np.std(attention_concentrations),
            })
            
        return layer_stats
        
    def compute_attention_rollout(self, attention_data: Dict, sample_idx: int = 0) -> np.ndarray:
        """
        Compute attention rollout - how information flows from input tokens to CLS through layers
        
        Args:
            attention_data: Output from extract_attention_weights
            sample_idx: Which sample to analyze
            
        Returns:
            Attention rollout matrix [seq_len, seq_len]
        """
        # Start with identity matrix
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
        
    def analyze_dataset_encoder_attention(self, dataloader, max_samples: int = 1000) -> Dict:
        """
        Analyze encoder attention patterns across a dataset
        
        Args:
            dataloader: DataLoader with quantum density matrices
            max_samples: Maximum number of samples to analyze
            
        Returns:
            Dict with comprehensive encoder attention analysis
        """
        print(f"Analyzing encoder attention patterns for up to {max_samples} samples...")
        
        all_attention_data = []
        sample_count = 0
        
        for batch_idx, (rho_real, rho_imag, true_magic) in enumerate(dataloader):
            if sample_count >= max_samples:
                break
                
            # Move to device
            rho_real = rho_real.to(self.device)
            rho_imag = rho_imag.to(self.device)
            
            # Extract attention weights
            attention_data = self.extract_attention_weights(rho_real, rho_imag)
            
            # Store with true labels
            batch_size = rho_real.shape[0]
            for i in range(min(batch_size, max_samples - sample_count)):
                sample_data = {
                    'sample_idx': sample_count + i,
                    'true_magic': true_magic[i].item(),
                    'predicted_magic': attention_data['prediction'][i].item(),
                    'attention_weights': [
                        {
                            'layer': layer_data['layer'],
                            'weights': layer_data['weights'][i:i+1],  # Keep single sample
                            'cls_attention': layer_data['cls_attention'][i:i+1],
                            'avg_head_weights': layer_data['avg_head_weights'][i:i+1],
                        }
                        for layer_data in attention_data['attention_weights']
                    ]
                }
                all_attention_data.append(sample_data)
                
            sample_count += batch_size
            if batch_idx % 10 == 0:
                print(f"Processed {sample_count} samples...")
                
        print(f"Completed encoder attention analysis for {len(all_attention_data)} samples")
        
        # Compute aggregate statistics
        head_specialization = self.aggregate_head_specialization(all_attention_data)
        layer_evolution = self.analyze_layer_evolution(all_attention_data)
        
        # Analyze head specialization across dataset
        full_head_analysis = self.analyze_head_specialization({
            'attention_weights': [
                {
                    'layer': layer_idx,
                    'weights': np.concatenate([
                        sample['attention_weights'][layer_idx]['weights']
                        for sample in all_attention_data
                    ], axis=0)
                }
                for layer_idx in range(len(all_attention_data[0]['attention_weights']))
            ]
        })
        
        return {
            'sample_data': all_attention_data,
            'head_specialization': head_specialization,
            'layer_evolution': layer_evolution,
            'full_head_analysis': full_head_analysis,
            'summary_stats': {
                'total_samples': len(all_attention_data),
                'avg_prediction_error': np.mean([
                    abs(s['predicted_magic'] - s['true_magic']) for s in all_attention_data
                ]),
                'true_magic_range': [
                    min(s['true_magic'] for s in all_attention_data),
                    max(s['true_magic'] for s in all_attention_data)
                ],
                'predicted_magic_range': [
                    min(s['predicted_magic'] for s in all_attention_data),
                    max(s['predicted_magic'] for s in all_attention_data)
                ]
            }
        }