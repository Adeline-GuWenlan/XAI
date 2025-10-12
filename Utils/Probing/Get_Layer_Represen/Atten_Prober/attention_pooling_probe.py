import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional

class AttentionPoolingProbe:
    """
    Captures attention-weighted pooling outputs and subsequent MLP layer representations.
    Specifically designed to extract vectors from:
    1. Attention pooling: (matrix_tokens * attn_weights).sum(dim=1) 
    2. MLP layers that follow the pooling mechanism
    """
    
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.cache: Dict[str, np.ndarray] = {}
        self._hooks = []
        self.model.eval()
        
        # Check if model uses attention pooling
        self.has_attention_pooling = (
            hasattr(model, 'pooling_type') and 
            model.pooling_type == "attention" and
            hasattr(model, 'physics_pooling')
        )
        
    @staticmethod
    def _to_np(t: torch.Tensor) -> np.ndarray:
        """Convert tensor to numpy array"""
        return t.detach().cpu().numpy()
    
    def _capture_attention_weighted_output(self):
        """
        Hook into PhysicsAwarePooling to capture the attention-weighted sum:
        (matrix_tokens * attn_weights).sum(dim=1)
        """
        if not self.has_attention_pooling:
            print("Warning: Model does not use attention pooling")
            return
            
        pooling_module = self.model.physics_pooling
        
        def attention_pooling_hook(module, inputs, output):
            """Capture the output of attention pooling mechanism"""
            if isinstance(output, torch.Tensor):
                self.cache["attention_pooling_output"] = self._to_np(output)
                
        self._hooks.append(pooling_module.register_forward_hook(attention_pooling_hook))
    
    def _capture_mlp_representations(self):
        """
        Hook into MLP layers that process the attention-pooled features
        """
        mlp_head = self.model.mlp_head
        
        # Hook into MLP input (the pooled features)
        def mlp_input_hook(module, inputs, output):
            if isinstance(inputs, tuple) and len(inputs) > 0:
                input_tensor = inputs[0]
                if isinstance(input_tensor, torch.Tensor):
                    self.cache["mlp_input"] = self._to_np(input_tensor)
                    
        self._hooks.append(mlp_head.register_forward_pre_hook(mlp_input_hook))
        
        # Hook into intermediate MLP layers
        if hasattr(mlp_head, 'mlp_head'):
            # For attention_enhanced MLP type
            mlp_container = mlp_head.mlp_head
            
            # Feature extractor layers
            if hasattr(mlp_container, 'feature_extractor'):
                for i, layer in enumerate(mlp_container.feature_extractor):
                    if isinstance(layer, nn.Linear):
                        def make_mlp_hook(layer_name):
                            def mlp_layer_hook(module, inputs, output):
                                if isinstance(output, torch.Tensor):
                                    self.cache[f"mlp_{layer_name}"] = self._to_np(output)
                            return mlp_layer_hook
                        
                        self._hooks.append(
                            layer.register_forward_hook(make_mlp_hook(f"feature_extractor_{i}"))
                        )
            
            # Attention mechanism output (if exists)
            if hasattr(mlp_container, 'attention'):
                def attention_mlp_hook(module, inputs, output):
                    # MultiheadAttention returns (output, weights)
                    attn_output = output[0] if isinstance(output, tuple) else output
                    if isinstance(attn_output, torch.Tensor):
                        # Handle different tensor shapes
                        if attn_output.dim() == 3:
                            # If (seq_len, batch, embed_dim) or (batch, seq_len, embed_dim)
                            if attn_output.shape[0] == 1:  # (1, batch, embed_dim)
                                attn_output = attn_output.squeeze(0)
                            elif attn_output.shape[1] == 1:  # (batch, 1, embed_dim)
                                attn_output = attn_output.squeeze(1)
                        self.cache["mlp_attention_output"] = self._to_np(attn_output)
                        
                self._hooks.append(
                    mlp_container.attention.register_forward_hook(attention_mlp_hook)
                )
            
            # Post-attention layers
            if hasattr(mlp_container, 'post_attention'):
                for i, layer in enumerate(mlp_container.post_attention):
                    if isinstance(layer, nn.Linear):
                        def make_post_attn_hook(layer_idx):
                            def post_attn_hook(module, inputs, output):
                                if isinstance(output, torch.Tensor):
                                    self.cache[f"mlp_post_attention_{layer_idx}"] = self._to_np(output)
                            return post_attn_hook
                        
                        self._hooks.append(
                            layer.register_forward_hook(make_post_attn_hook(i))
                        )
    
    def register_hooks(self):
        """Register all hooks for capturing attention pooling and MLP representations"""
        self.remove_hooks()
        self.cache = {}
        
        # Capture attention pooling output
        self._capture_attention_weighted_output()
        
        # Capture MLP layer representations
        self._capture_mlp_representations()
    
    def remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self._hooks:
            try:
                hook.remove()
            except Exception:
                pass
        self._hooks = []
    
    @torch.no_grad()
    def extract_representations(self, rho_real: torch.Tensor, rho_imag: torch.Tensor) -> Dict[str, np.ndarray]:
        """
        Run forward pass and extract attention pooling + MLP representations
        
        Args:
            rho_real: Real part of density matrix [B, D, D]
            rho_imag: Imaginary part of density matrix [B, D, D]
            
        Returns:
            Dictionary with captured representations:
            - 'attention_pooling_output': Output after attention-weighted pooling
            - 'mlp_input': Input to MLP head (should be same as attention_pooling_output)
            - 'mlp_feature_extractor_X': Outputs from feature extractor layers
            - 'mlp_attention_output': Output from MLP attention mechanism
            - 'mlp_post_attention_X': Outputs from post-attention MLP layers
        """
        self.register_hooks()
        try:
            # Forward pass triggers all hooks
            _ = self.model(rho_real, rho_imag)
            
            # Verify we captured the expected representations
            if self.has_attention_pooling and "attention_pooling_output" not in self.cache:
                print("Warning: Failed to capture attention pooling output")
            
            return dict(self.cache)
        finally:
            self.remove_hooks()
    
    def get_available_representations(self) -> List[str]:
        """Return list of representation names that can be captured"""
        representations = []
        
        if self.has_attention_pooling:
            representations.append("attention_pooling_output")
        
        representations.extend([
            "mlp_input",
            "mlp_feature_extractor_*",
            "mlp_attention_output",
            "mlp_post_attention_*"
        ])
        
        return representations
    
    def save_representations_csv(self, representations: Dict[str, np.ndarray], output_dir: str, prefix: str = ""):
        """
        Save captured representations to CSV files
        
        Args:
            representations: Dictionary from extract_representations()
            output_dir: Directory to save CSV files
            prefix: Optional prefix for filenames
        """
        import os
        import pandas as pd
        
        os.makedirs(output_dir, exist_ok=True)
        
        for name, data in representations.items():
            if data.ndim == 2:  # [batch_size, feature_dim]
                df = pd.DataFrame(data)
                filename = f"{prefix}_{name}.csv" if prefix else f"{name}.csv"
                filepath = os.path.join(output_dir, filename)
                df.to_csv(filepath, index=False)
                print(f"Saved {name} shape {data.shape} to {filepath}")
            else:
                print(f"Warning: Skipping {name} with unexpected shape {data.shape}")