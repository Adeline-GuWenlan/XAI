"""
Enhanced BERT-style Transformer - Minimal changes from existing codebase

Key improvements:
- Configurable number of encoder layers
- Configurable loss functions (MSE, SmoothL1) 
- FFN automatically 4x d_model
- Uses existing embedding and components where possible
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# Import existing components
sys.path.append('..')
from transformer_embedding import StructuredQuantumEmbedding
from mlp import QuantumMagicMLPv2

class EnhancedTransformerEncoder(nn.Module):
    """Enhanced transformer encoder with configurable layers"""
    
    def __init__(self, d_model, nhead, matrix_dim, n_qubits, num_layers=6, 
                 dropout=0.1, use_physics_mask=False, use_cls_token=True):
        super().__init__()
        
        self.num_layers = num_layers
        
        # Create multiple encoder layers
        self.layers = nn.ModuleList([
            TransformerLayer(d_model, nhead, dropout) 
            for _ in range(num_layers)
        ])
        
        # Final layer norm (BERT-style)
        self.final_norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        # Pass through all layers
        for layer in self.layers:
            x = layer(x)
        
        # Final normalization
        x = self.final_norm(x)
        return x

class TransformerLayer(nn.Module):
    """Single transformer layer with pre-norm architecture"""
    
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        
        # Multi-head attention
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        
        # Feed-forward network (4x d_model as per BERT)
        d_ff = 4 * d_model
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Pre-norm self-attention with residual
        residual = x
        x = self.norm1(x)
        attn_out, _ = self.self_attn(x, x, x)
        x = self.dropout(attn_out) + residual
        
        # Pre-norm FFN with residual
        residual = x
        x = self.norm2(x)
        ffn_out = self.ffn(x)
        x = ffn_out + residual
        
        return x

class EnhancedQuantumPredictor(nn.Module):
    """Enhanced version of CompleteQuantumMagicPredictor with configurable layers and loss"""
    
    def __init__(self, matrix_dim=None, n_qubits=3, d_model=512, num_layers=6,
                 pooling_type="cls", mlp_type="mixture_of_experts", 
                 nhead=8, use_cls_token=True, loss_type="mse", dropout=0.1):
        super().__init__()
        
        # Auto-calculate matrix_dim from n_qubits if not provided
        if matrix_dim is None:
            matrix_dim = 2 ** n_qubits
        
        self.matrix_dim = matrix_dim
        self.n_qubits = n_qubits
        self.d_model = d_model
        self.num_layers = num_layers
        self.pooling_type = pooling_type
        self.nhead = nhead
        self.use_cls_token = use_cls_token
        self.loss_type = loss_type
        
        # Force consistency: if pooling_type is "cls", we need CLS token
        if pooling_type == "cls":
            self.use_cls_token = True
        
        # Use existing embedding
        self.embedding = StructuredQuantumEmbedding(n_qubits, d_model)
        
        # CLS token (only if needed)
        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        
        # Enhanced encoder with configurable layers
        self.encoder = EnhancedTransformerEncoder(
            d_model, nhead, matrix_dim, n_qubits, 
            num_layers=num_layers, dropout=dropout,
            use_physics_mask=False, use_cls_token=self.use_cls_token
        )
        
        # Use existing MLP head
        self.mlp_head = QuantumMagicMLPv2(d_model=d_model, mlp_type=mlp_type, n_qubits=n_qubits)
        
        # Loss function
        if loss_type.lower() == "mse":
            self.loss_fn = nn.MSELoss()
        elif loss_type.lower() == "smoothl1":
            self.loss_fn = nn.SmoothL1Loss()
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
        
        print(f"🚀 Enhanced Model Config:")
        print(f"   Layers: {num_layers}, d_model: {d_model}, heads: {nhead}")
        print(f"   Pooling: {pooling_type}, MLP: {mlp_type}, Loss: {loss_type}")
        print(f"   FFN dimension: {4 * d_model} (4x d_model)")
    
    def forward(self, rho_real, rho_imag):
        # Embedding: [B, D, D] → [B, D², d_model] (use existing)
        tokens = self.embedding(rho_real, rho_imag)
        
        # Add CLS token if using CLS pooling
        if self.use_cls_token:
            batch_size = tokens.size(0)
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            tokens = torch.cat([cls_tokens, tokens], dim=1)  # [B, D²+1, d_model]
        
        # Enhanced transformer processing
        tokens = self.encoder(tokens)
        
        # Pooling: Choose between CLS token or mean pooling
        if self.pooling_type == "cls":
            # Use CLS token (position 0)
            if not self.use_cls_token:
                raise ValueError("Cannot use CLS pooling without CLS token!")
            global_features = tokens[:, 0, :]
        else:
            # Use mean pooling (skip CLS token if present)
            if self.use_cls_token:
                global_features = tokens[:, 1:, :].mean(dim=1)  # Skip CLS token
            else:
                global_features = tokens.mean(dim=1)
        
        # MLP prediction: [B, d_model] → [B, 1] (use existing)
        magic_prediction = self.mlp_head(global_features)
        
        return magic_prediction
    
    def compute_loss(self, predictions, targets):
        """Compute loss using configured loss function"""
        return self.loss_fn(predictions.squeeze(), targets.squeeze())
    
    def get_config(self):
        """Return model configuration"""
        return {
            'n_qubits': self.n_qubits,
            'd_model': self.d_model,
            'num_layers': self.num_layers,
            'nhead': self.nhead,
            'pooling_type': self.pooling_type,
            'mlp_type': 'mixture_of_experts',  # Hardcoded for now
            'use_cls_token': self.use_cls_token,
            'loss_type': self.loss_type
        }