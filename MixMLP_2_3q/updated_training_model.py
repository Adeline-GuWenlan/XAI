import torch
import torch.nn as nn
from transformer_embedding import StructuredQuantumEmbedding
from transformer_encoder import PhysicsInformedTransformerEncoder
import torch.nn.functional as F
from mlp import QuantumMagicMLPv2

class CompleteQuantumMagicPredictor(nn.Module):
    """Complete pipeline: embedding → transformer → physics pooling → MLP"""
    def __init__(self, matrix_dim=None, n_qubits=2, d_model=64,
                 pooling_type="cls", mlp_type="standard",
                 mask_threshold=1, nhead=8, use_cls_token=True):
        super().__init__()
        # Auto-calculate matrix_dim from n_qubits if not provided
        if matrix_dim is None:
            matrix_dim = 2 ** n_qubits
        
        self.matrix_dim = matrix_dim
        self.n_qubits = n_qubits
        self.d_model = d_model
        self.pooling_type = pooling_type
        self.mask_threshold = mask_threshold
        self.nhead = nhead
        self.use_cls_token = use_cls_token
        
        # Force consistency: if pooling_type is "cls", we need CLS token
        if pooling_type == "cls":
            self.use_cls_token = True
        
        # Components
        self.embedding = StructuredQuantumEmbedding(n_qubits, d_model)
        
        # CLS token (only if needed)
        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        self.encoder_layers = nn.ModuleList([
            PhysicsInformedTransformerEncoder(
                d_model, nhead=nhead, matrix_dim=matrix_dim, n_qubits=n_qubits,
                use_cls_token=self.use_cls_token,
                mask_threshold=mask_threshold
            ) for _ in range(4)
        ])

        # Physics pooling (only if not using CLS)
        if pooling_type != "cls":
            self.physics_pooling = PhysicsAwarePooling(d_model, matrix_dim, pooling_type)
        else:
            self.physics_pooling = None
        
        self.mlp_head = QuantumMagicMLPv2(d_model=d_model, mlp_type=mlp_type, n_qubits=n_qubits)
        
        print(f"🔧 Model Config:")
        print(f"   Pooling: {pooling_type}, Use CLS: {self.use_cls_token}")
        print(f"   MLP: {mlp_type}")
    
    def forward(self, rho_real, rho_imag):
        # Embedding: [B, D, D] → [B, D², d_model]
        tokens = self.embedding(rho_real, rho_imag)
        
        # Add CLS token if using CLS pooling
        if self.use_cls_token:
            batch_size = tokens.size(0)
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            tokens = torch.cat([cls_tokens, tokens], dim=1)  # [B, D²+1, d_model]
        
        # Debug: print token shapes
        # print(f"🔍 Tokens shape after embedding+CLS: {tokens.shape}")
        
        # Transformer processing
        for layer in self.encoder_layers:
            tokens = layer(tokens)
        
        # Pooling: Choose between CLS token or physics-aware pooling
        if self.pooling_type == "cls":
            # Use CLS token (position 0)
            if not self.use_cls_token:
                raise ValueError("Cannot use CLS pooling without CLS token!")
            global_features = tokens[:, 0, :]
        else:
            # Use physics-aware pooling (handles CLS token internally)
            global_features = self.physics_pooling(tokens)
        
        # MLP prediction: [B, d_model] → [B, 1]
        magic_prediction = self.mlp_head(global_features)
        
        return magic_prediction


class PhysicsAwarePooling(nn.Module):
    """Physics-aware pooling for quantum density matrices"""
    
    def __init__(self, d_model: int, matrix_dim: int, pooling_type: str = "structured"):
        super().__init__()
        self.d_model = d_model
        self.matrix_dim = matrix_dim
        self.pooling_type = pooling_type
        
        if pooling_type == "attention":
            self.attention_weights = nn.Linear(d_model, 1)
            
        elif pooling_type == "structured":
            # Separate processing for diagonal vs off-diagonal
            self.diag_pooling = nn.Linear(d_model, d_model // 2)
            self.offdiag_pooling = nn.Linear(d_model, d_model // 2)
            self.combine = nn.Linear(d_model, d_model)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, seq_len, d_model] - may or may not include CLS token
        Returns:
            [B, d_model] - aggregated features
        """
        B, seq_len, d_model = x.shape
        D = self.matrix_dim
        
        # Handle both cases: with and without CLS token
        if seq_len == D * D + 1:
            # Has CLS token - extract matrix tokens (skip CLS at position 0)
            matrix_tokens = x[:, 1:, :]  # [B, D², d_model]
        elif seq_len == D * D:
            # No CLS token - all tokens are matrix tokens
            matrix_tokens = x  # [B, D², d_model]
        else:
            # Unexpected sequence length
            print(f"⚠️ Unexpected sequence length: {seq_len}, expected {D*D} or {D*D+1}")
            # Try to use all tokens except the first one if it might be CLS
            matrix_tokens = x[:, 1:, :] if seq_len > D * D else x
        
        if self.pooling_type == "mean":
            return matrix_tokens.mean(dim=1)
            
        elif self.pooling_type == "attention":
            # Learnable attention weights
            attn_weights = F.softmax(self.attention_weights(matrix_tokens), dim=1)
            return (matrix_tokens * attn_weights).sum(dim=1)
            
        elif self.pooling_type == "structured":
            # Physics-aware: separate diagonal from off-diagonal
            actual_matrix_tokens = matrix_tokens.size(1)
            expected_tokens = D * D
            
            if actual_matrix_tokens != expected_tokens:
                print(f"⚠️ Matrix token count mismatch: got {actual_matrix_tokens}, expected {expected_tokens}")
                # Use available tokens
                available_D = int(actual_matrix_tokens ** 0.5)
                D = available_D
            
            diag_indices = torch.arange(D, device=x.device) * (D + 1)
            
            all_indices = torch.arange(actual_matrix_tokens, device=x.device)
            diag_mask = torch.isin(all_indices, diag_indices)
            offdiag_mask = ~diag_mask
            
            # Separate elements
            diag_elements = matrix_tokens[:, diag_mask, :]     # [B, D, d_model]
            offdiag_elements = matrix_tokens[:, offdiag_mask, :] # [B, D²-D, d_model]
            
            # Process separately then combine
            diag_pooled = self.diag_pooling(diag_elements.mean(dim=1))
            offdiag_pooled = self.offdiag_pooling(offdiag_elements.mean(dim=1))
            
            combined = torch.cat([diag_pooled, offdiag_pooled], dim=1)
            return self.combine(combined)
        
        else:
            raise ValueError(f"Unknown pooling type: {self.pooling_type}")