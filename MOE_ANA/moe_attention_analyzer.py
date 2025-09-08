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

class MoEAttentionAnalyzer:
    """Specialized analyzer for MoE quantum magic prediction model"""
    
    def __init__(self, model: nn.Module, device: torch.device, config: Dict = None):
        self.model = model
        self.device = device
        self.config = config or {}
        self.model.eval()
        
    def extract_attention_and_expert_weights(self, rho_real: torch.Tensor, rho_imag: torch.Tensor) -> Dict:
        """
        Extract both attention weights and expert gating information from MoE model
        
        Args:
            rho_real, rho_imag: Input density matrix components [B, D, D]
            
        Returns:
            Dict containing predictions, attention weights, and expert analysis
        """
        with torch.no_grad():
            # Forward pass through the model to get intermediate features
            matrix_tokens = self.model.embedding(rho_real, rho_imag)
            B = matrix_tokens.shape[0]
            
            # Add CLS token if using CLS pooling
            if self.model.pooling_type == "cls" and hasattr(self.model, 'cls_token'):
                cls_tokens = self.model.cls_token.expand(B, -1, -1)
                tokens = torch.cat([cls_tokens, matrix_tokens], dim=1)
            else:
                tokens = matrix_tokens
            
            # Forward through transformer layers with attention extraction
            attention_weights = []
            for layer in self.model.encoder_layers:
                tokens, attn_weights = layer(tokens, return_attention=True)
                attention_weights.append(attn_weights.cpu().numpy())
            
            # Extract features before MLP
            if self.model.pooling_type == "cls":
                global_features = tokens[:, 0, :]
            else:
                global_features = self.model.physics_pooling(tokens)
            
            # Analyze MoE expert behavior
            expert_analysis = self._analyze_moe_experts(global_features)
            
            # Get final prediction
            prediction = self.model.mlp_head(global_features)
            
            result = {
                'prediction': prediction.cpu().numpy(),
                'attention_weights': attention_weights,
                'expert_analysis': expert_analysis,
                'global_features': global_features.cpu().numpy(),
                'matrix_tokens': matrix_tokens.cpu().numpy(),
            }
            
            return result
    
    def _analyze_moe_experts(self, global_features: torch.Tensor) -> Dict:
        """Analyze MoE expert gating and specialization"""
        if self.model.mlp_head.mlp_type != "mixture_of_experts":
            return {"type": "not_moe"}
        
        # Extract gating weights
        x = self.model.mlp_head.input_norm(global_features)
        gates = self.model.mlp_head.mlp_head['gating_network'](x)  # [B, num_experts]
        
        # Get individual expert outputs
        expert_outputs = []
        for expert in self.model.mlp_head.mlp_head['experts']:
            expert_outputs.append(expert(x))
        expert_outputs = torch.stack(expert_outputs, dim=-1)  # [B, 1, num_experts]
        
        # Compute expert statistics
        expert_analysis = {
            'type': 'moe',
            'gating_weights': gates.cpu().numpy(),  # [B, num_experts]
            'expert_outputs': expert_outputs.squeeze(1).cpu().numpy(),  # [B, num_experts]
            'expert_usage': gates.mean(dim=0).cpu().numpy(),  # Average usage per expert
            'gating_entropy': self._compute_gating_entropy(gates),
            'expert_specialization': self._compute_expert_specialization(gates, expert_outputs.squeeze(1)),
        }
        
        return expert_analysis
    
    def _compute_gating_entropy(self, gates: torch.Tensor) -> float:
        """Compute entropy of gating distribution to measure expert diversity"""
        # Average gating distribution across batch
        avg_gates = gates.mean(dim=0)  # [num_experts]
        entropy = -torch.sum(avg_gates * torch.log(avg_gates + 1e-10))
        return entropy.item()
    
    def _compute_expert_specialization(self, gates: torch.Tensor, expert_outputs: torch.Tensor) -> Dict:
        """Analyze how experts specialize based on their outputs and usage patterns"""
        num_experts = gates.shape[1]
        B = gates.shape[0]
        
        specialization = {}
        
        for expert_idx in range(num_experts):
            expert_gates = gates[:, expert_idx]  # [B]
            expert_preds = expert_outputs[:, expert_idx]  # [B]
            
            # Find samples where this expert is dominant
            dominant_mask = expert_gates > 0.4  # Threshold for dominance
            
            if dominant_mask.sum() > 0:
                dominant_predictions = expert_preds[dominant_mask]
                specialization[f'expert_{expert_idx}'] = {
                    'dominance_frequency': dominant_mask.float().mean().item(),
                    'avg_prediction_when_dominant': dominant_predictions.mean().item(),
                    'prediction_std_when_dominant': dominant_predictions.std().item(),
                    'avg_gating_weight': expert_gates.mean().item(),
                    'gating_weight_std': expert_gates.std().item(),
                }
            else:
                specialization[f'expert_{expert_idx}'] = {
                    'dominance_frequency': 0.0,
                    'avg_prediction_when_dominant': expert_preds.mean().item(),
                    'prediction_std_when_dominant': expert_preds.std().item(),
                    'avg_gating_weight': expert_gates.mean().item(),
                    'gating_weight_std': expert_gates.std().item(),
                }
        
        return specialization
    
    def analyze_sample_with_moe(self, rho_real: torch.Tensor, rho_imag: torch.Tensor, 
                               true_magic: float, sample_idx: int = 0) -> Dict:
        """Comprehensive analysis of a single sample including MoE behavior"""
        analysis = self.extract_attention_and_expert_weights(rho_real, rho_imag)
        
        # Focus on single sample
        result = {
            'sample_idx': sample_idx,
            'true_magic': true_magic,
            'predicted_magic': analysis['prediction'][sample_idx].item(),
            'attention_analysis': self._analyze_attention_patterns(analysis['attention_weights'], sample_idx),
            'expert_analysis': self._analyze_sample_expert_behavior(analysis['expert_analysis'], sample_idx),
        }
        
        return result
    
    def _analyze_attention_patterns(self, attention_weights: List[np.ndarray], sample_idx: int) -> Dict:
        """Analyze attention patterns for a specific sample"""
        num_layers = len(attention_weights)
        matrix_dim = int(np.sqrt(attention_weights[0].shape[-1] - 1))  # Subtract 1 for CLS token
        
        layer_patterns = []
        for layer_idx, attn in enumerate(attention_weights):
            # Get CLS attention for this sample [num_heads, seq_len]
            cls_attn = attn[sample_idx, :, 0, :]  # CLS token attention
            
            # Focus on attention to matrix elements (skip CLS position)
            matrix_attn = cls_attn[:, 1:]  # [num_heads, matrix_elements]
            
            # Reshape to matrix form
            matrix_attn_reshaped = matrix_attn.reshape(cls_attn.shape[0], matrix_dim, matrix_dim)
            
            layer_analysis = {
                'layer': layer_idx,
                'cls_to_matrix': matrix_attn_reshaped,
                'avg_cls_to_matrix': matrix_attn_reshaped.mean(axis=0),
                'attention_entropy': -np.sum(matrix_attn * np.log(matrix_attn + 1e-10), axis=1).mean(),
                'max_attention_positions': np.unravel_index(
                    np.argmax(matrix_attn_reshaped.mean(axis=0)), 
                    (matrix_dim, matrix_dim)
                ),
            }
            layer_patterns.append(layer_analysis)
        
        return {
            'matrix_dim': matrix_dim,
            'layer_patterns': layer_patterns,
            'attention_evolution': self._compute_attention_evolution(layer_patterns)
        }
    
    def _analyze_sample_expert_behavior(self, expert_analysis: Dict, sample_idx: int) -> Dict:
        """Analyze expert behavior for a specific sample"""
        if expert_analysis['type'] != 'moe':
            return expert_analysis
        
        sample_gates = expert_analysis['gating_weights'][sample_idx]  # [num_experts]
        sample_expert_outputs = expert_analysis['expert_outputs'][sample_idx]  # [num_experts]
        
        # Find dominant expert
        dominant_expert = np.argmax(sample_gates)
        
        return {
            'type': 'moe',
            'sample_gating_weights': sample_gates,
            'sample_expert_outputs': sample_expert_outputs,
            'dominant_expert': dominant_expert,
            'dominant_weight': sample_gates[dominant_expert],
            'expert_agreement': np.std(sample_expert_outputs),  # Low std = high agreement
            'weighted_prediction': np.sum(sample_gates * sample_expert_outputs),
        }
    
    def _compute_attention_evolution(self, layer_patterns: List[Dict]) -> Dict:
        """Compute how attention patterns evolve through layers"""
        entropies = [pattern['attention_entropy'] for pattern in layer_patterns]
        
        # Compute attention focus change
        focus_change = []
        for i in range(1, len(layer_patterns)):
            prev_max = np.max(layer_patterns[i-1]['avg_cls_to_matrix'])
            curr_max = np.max(layer_patterns[i]['avg_cls_to_matrix'])
            focus_change.append(curr_max - prev_max)
        
        return {
            'entropy_evolution': entropies,
            'focus_intensification': focus_change,
            'final_entropy': entropies[-1] if entropies else 0,
            'entropy_trend': 'increasing' if len(entropies) > 1 and entropies[-1] > entropies[0] else 'decreasing'
        }
    
    def analyze_dataset_with_moe(self, dataloader, max_samples: int = 1000) -> Dict:
        """Analyze MoE behavior across dataset"""
        print(f"Analyzing MoE attention patterns for up to {max_samples} samples...")
        
        all_results = []
        expert_statistics = {
            'expert_usage_distribution': [],
            'gating_entropies': [],
            'expert_specialization_scores': [],
        }
        
        sample_count = 0
        
        for batch_idx, (rho_real, rho_imag, true_magic) in enumerate(dataloader):
            if sample_count >= max_samples:
                break
                
            # Move to device
            rho_real = rho_real.to(self.device)
            rho_imag = rho_imag.to(self.device)
            
            # Extract analysis
            batch_analysis = self.extract_attention_and_expert_weights(rho_real, rho_imag)
            
            # Store batch results
            batch_size = rho_real.shape[0]
            for i in range(min(batch_size, max_samples - sample_count)):
                # Extract single sample data
                single_rho_real = rho_real[i:i+1]  # Keep batch dimension
                single_rho_imag = rho_imag[i:i+1]  # Keep batch dimension
                single_true_magic = true_magic[i].item()
                
                # Get analysis for this single sample
                single_analysis = self.extract_attention_and_expert_weights(single_rho_real, single_rho_imag)
                
                # Create sample result manually to avoid indexing issues
                sample_result = {
                    'sample_idx': sample_count + i,
                    'true_magic': single_true_magic,
                    'predicted_magic': single_analysis['prediction'][0].item(),  # Always index 0 for single sample
                    'attention_analysis': self._analyze_attention_patterns(single_analysis['attention_weights'], 0),
                    'expert_analysis': self._analyze_sample_expert_behavior(single_analysis['expert_analysis'], 0),
                }
                all_results.append(sample_result)
                
                # Collect expert statistics
                if single_analysis['expert_analysis']['type'] == 'moe':
                    expert_statistics['expert_usage_distribution'].append(
                        single_analysis['expert_analysis']['gating_weights'][0]  # Always index 0 for single sample
                    )
                    
            sample_count += batch_size
            
            if batch_idx % 10 == 0:
                print(f"Processed {sample_count} samples...")
        
        # Aggregate statistics
        if expert_statistics['expert_usage_distribution']:
            usage_array = np.array(expert_statistics['expert_usage_distribution'])
            expert_statistics['avg_expert_usage'] = usage_array.mean(axis=0)
            expert_statistics['expert_usage_std'] = usage_array.std(axis=0)
            
            # Compute expert diversity metrics
            expert_statistics['expert_diversity'] = self._compute_expert_diversity(usage_array)
        
        return {
            'sample_results': all_results,
            'expert_statistics': expert_statistics,
            'summary': self._create_summary_statistics(all_results),
        }
    
    def _compute_expert_diversity(self, usage_array: np.ndarray) -> Dict:
        """Compute diversity metrics for expert usage"""
        num_experts = usage_array.shape[1]
        
        # Compute entropy for each sample
        sample_entropies = []
        for i in range(usage_array.shape[0]):
            gates = usage_array[i]
            entropy = -np.sum(gates * np.log(gates + 1e-10))
            sample_entropies.append(entropy)
        
        return {
            'mean_gating_entropy': np.mean(sample_entropies),
            'std_gating_entropy': np.std(sample_entropies),
            'max_possible_entropy': np.log(num_experts),
            'normalized_entropy': np.mean(sample_entropies) / np.log(num_experts),
        }
    
    def _create_summary_statistics(self, all_results: List[Dict]) -> Dict:
        """Create summary statistics from all results"""
        if not all_results:
            return {}
        
        predictions = [r['predicted_magic'] for r in all_results]
        true_values = [r['true_magic'] for r in all_results]
        errors = [abs(p - t) for p, t in zip(predictions, true_values)]
        
        # Attention statistics
        final_entropies = []
        for result in all_results:
            if 'attention_analysis' in result:
                final_entropies.append(result['attention_analysis']['attention_evolution']['final_entropy'])
        
        # Expert statistics (if MoE)
        expert_agreements = []
        dominant_expert_counts = [0, 0, 0]  # Assuming 3 experts
        
        for result in all_results:
            if 'expert_analysis' in result and result['expert_analysis']['type'] == 'moe':
                expert_agreements.append(result['expert_analysis']['expert_agreement'])
                dominant_expert = result['expert_analysis']['dominant_expert']
                if 0 <= dominant_expert < len(dominant_expert_counts):
                    dominant_expert_counts[dominant_expert] += 1
        
        summary = {
            'total_samples': len(all_results),
            'prediction_stats': {
                'mean_error': np.mean(errors),
                'std_error': np.std(errors),
                'mae': np.mean(errors),
                'prediction_range': [min(predictions), max(predictions)],
                'true_range': [min(true_values), max(true_values)],
            },
            'attention_stats': {
                'mean_final_entropy': np.mean(final_entropies) if final_entropies else 0,
                'std_final_entropy': np.std(final_entropies) if final_entropies else 0,
            },
        }
        
        if expert_agreements:
            summary['expert_stats'] = {
                'mean_expert_agreement': np.mean(expert_agreements),
                'std_expert_agreement': np.std(expert_agreements),
                'dominant_expert_distribution': dominant_expert_counts,
                'expert_balance': min(dominant_expert_counts) / max(dominant_expert_counts) if max(dominant_expert_counts) > 0 else 0,
            }
        
        return summary