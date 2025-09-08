"""
Enhanced BERT-style Transformer for Quantum Magic Prediction

Features:
- Configurable number of encoder layers
- Configurable FFN dimensions (default 4x d_model)
- Multiple loss functions (MSE, SmoothL1)
- Self-contained implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Literal

class MultiHeadAttention(nn.Module):
    """Standard multi-head attention mechanism"""
    
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % nhead == 0, f"d_model ({d_model}) must be divisible by nhead ({nhead})"
        
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead
        
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False) 
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
            mask: [batch_size, seq_len, seq_len] or [seq_len, seq_len]
        Returns:
            [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.shape
        
        # Linear transformations and reshape for multi-head attention
        Q = self.w_q(x).view(batch_size, seq_len, self.nhead, self.d_k).transpose(1, 2)  # [B, H, L, d_k]
        K = self.w_k(x).view(batch_size, seq_len, self.nhead, self.d_k).transpose(1, 2)  # [B, H, L, d_k]
        V = self.w_v(x).view(batch_size, seq_len, self.nhead, self.d_k).transpose(1, 2)  # [B, H, L, d_k]
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # [B, H, L, L]
        
        # Apply mask if provided
        if mask is not None:
            if mask.dim() == 2:  # [L, L] -> [B, H, L, L]
                mask = mask.unsqueeze(0).unsqueeze(0).expand(batch_size, self.nhead, -1, -1)
            elif mask.dim() == 3:  # [B, L, L] -> [B, H, L, L]
                mask = mask.unsqueeze(1).expand(-1, self.nhead, -1, -1)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Compute attention weights and apply to values
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        out = torch.matmul(attn_weights, V)  # [B, H, L, d_k]
        
        # Concatenate heads and apply output projection
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.w_o(out)

class FeedForwardNetwork(nn.Module):
    """Feed-forward network with configurable hidden dimension"""
    
    def __init__(self, d_model: int, d_ff: Optional[int] = None, dropout: float = 0.1, activation: str = "gelu"):
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model  # Standard BERT ratio
            
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
        if activation.lower() == "gelu":
            self.activation = nn.GELU()
        elif activation.lower() == "relu":
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(self.activation(self.linear1(x))))

class TransformerEncoderLayer(nn.Module):
    """Single transformer encoder layer with pre-norm architecture"""
    
    def __init__(self, d_model: int, nhead: int, d_ff: Optional[int] = None, 
                 dropout: float = 0.1, activation: str = "gelu"):
        super().__init__()
        
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.ffn = FeedForwardNetwork(d_model, d_ff, dropout, activation)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-norm self-attention with residual connection
        residual = x
        x = self.norm1(x)
        x = self.self_attn(x, mask)
        x = self.dropout(x) + residual
        
        # Pre-norm FFN with residual connection  
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = self.dropout(x) + residual
        
        return x

class QuantumEmbedding(nn.Module):
    """Enhanced quantum embedding with better position encoding"""
    
    def __init__(self, n_qubits: int, d_model: int):
        super().__init__()
        self.n_qubits = n_qubits
        self.d_model = d_model
        self.matrix_dim = 2 ** n_qubits
        self.seq_len = self.matrix_dim ** 2
        
        # Value embedding (real and imaginary components)
        self.value_embedding = nn.Linear(2, d_model - 2)  # Leave 2 dims for position
        
        # Position embedding for matrix indices
        self.position_embedding = nn.Embedding(self.seq_len, 2)
        
        # Initialize position embeddings with actual matrix coordinates
        self._initialize_position_embeddings()
        
    def _initialize_position_embeddings(self):
        """Initialize position embeddings with matrix coordinate information"""
        pos_data = torch.zeros(self.seq_len, 2)
        for idx in range(self.seq_len):
            i = idx // self.matrix_dim
            j = idx % self.matrix_dim
            pos_data[idx, 0] = i / (self.matrix_dim - 1)  # Normalized row
            pos_data[idx, 1] = j / (self.matrix_dim - 1)  # Normalized col
        
        with torch.no_grad():
            self.position_embedding.weight.copy_(pos_data)
    
    def forward(self, rho_real: torch.Tensor, rho_imag: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rho_real: [batch_size, matrix_dim, matrix_dim] 
            rho_imag: [batch_size, matrix_dim, matrix_dim]
        Returns:
            [batch_size, seq_len, d_model]
        """
        batch_size = rho_real.shape[0]
        
        # Flatten matrices to sequences
        real_flat = rho_real.view(batch_size, -1, 1)  # [B, seq_len, 1]
        imag_flat = rho_imag.view(batch_size, -1, 1)  # [B, seq_len, 1]
        values = torch.cat([real_flat, imag_flat], dim=-1)  # [B, seq_len, 2]
        
        # Value embedding
        value_emb = self.value_embedding(values)  # [B, seq_len, d_model-2]
        
        # Position embedding
        positions = torch.arange(self.seq_len, device=rho_real.device)
        pos_emb = self.position_embedding(positions).unsqueeze(0).expand(batch_size, -1, -1)  # [B, seq_len, 2]
        
        # Combine value and position embeddings
        embeddings = torch.cat([value_emb, pos_emb], dim=-1)  # [B, seq_len, d_model]
        
        return embeddings

class EnhancedQuantumTransformer(nn.Module):
    """
    Enhanced BERT-style Transformer for Quantum Magic Prediction
    
    Args:
        n_qubits: Number of qubits (determines matrix size)
        d_model: Model dimension 
        num_layers: Number of transformer encoder layers
        nhead: Number of attention heads
        d_ff: FFN hidden dimension (default: 4 * d_model)
        dropout: Dropout rate
        use_cls_token: Whether to use CLS token for classification
        loss_type: Loss function type ('mse' or 'smoothl1')
    """
    
    def __init__(
        self,
        n_qubits: int = 3,
        d_model: int = 512, 
        num_layers: int = 6,
        nhead: int = 8,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
        use_cls_token: bool = True,
        loss_type: Literal['mse', 'smoothl1'] = 'mse'
    ):
        super().__init__()
        
        self.n_qubits = n_qubits
        self.d_model = d_model
        self.num_layers = num_layers
        self.nhead = nhead
        self.use_cls_token = use_cls_token
        self.loss_type = loss_type
        
        if d_ff is None:
            d_ff = 4 * d_model
        
        # Embedding layer
        self.embedding = QuantumEmbedding(n_qubits, d_model)
        
        # CLS token (if used)
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        
        # Transformer encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(d_model)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(), 
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, 1)
        )
        
        # Loss function
        if loss_type == 'mse':
            self.loss_fn = nn.MSELoss()
        elif loss_type == 'smoothl1':
            self.loss_fn = nn.SmoothL1Loss()
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
        
        self.apply(self._init_weights)
        
        print(f"🚀 Enhanced Quantum Transformer initialized:")
        print(f"   Qubits: {n_qubits}, d_model: {d_model}, layers: {num_layers}")
        print(f"   Heads: {nhead}, d_ff: {d_ff}, dropout: {dropout}")
        print(f"   CLS token: {use_cls_token}, loss: {loss_type}")
    
    def _init_weights(self, module):
        """Initialize weights following BERT-style initialization"""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(self, rho_real: torch.Tensor, rho_imag: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rho_real: [batch_size, matrix_dim, matrix_dim]
            rho_imag: [batch_size, matrix_dim, matrix_dim] 
        Returns:
            [batch_size, 1] - Magic monotone predictions
        """
        # Get embeddings
        x = self.embedding(rho_real, rho_imag)  # [B, seq_len, d_model]
        
        # Add CLS token if used
        if self.use_cls_token:
            batch_size = x.shape[0]
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)  # [B, seq_len+1, d_model]
        
        # Pass through transformer layers
        for layer in self.encoder_layers:
            x = layer(x)
        
        # Final normalization
        x = self.final_norm(x)
        
        # Extract features for classification
        if self.use_cls_token:
            # Use CLS token representation
            features = x[:, 0, :]  # [B, d_model]
        else:
            # Use mean pooling
            features = x.mean(dim=1)  # [B, d_model]
        
        # Predict magic value
        prediction = self.classifier(features)  # [B, 1]
        
        return prediction
    
    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute loss based on configured loss type"""
        return self.loss_fn(predictions.squeeze(), targets.squeeze())
    
    def get_config(self) -> dict:
        """Return model configuration"""
        return {
            'n_qubits': self.n_qubits,
            'd_model': self.d_model, 
            'num_layers': self.num_layers,
            'nhead': self.nhead,
            'use_cls_token': self.use_cls_token,
            'loss_type': self.loss_type
        }