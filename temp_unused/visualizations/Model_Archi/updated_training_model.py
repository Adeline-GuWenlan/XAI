import torch
import torch.nn as nn
import torch.nn.functional as F
from Model_Archi.transformer_embedding import StructuredQuantumEmbedding
from Model_Archi.transformer_encoder import PhysicsInformedTransformerEncoder
from Model_Archi.mlp import QuantumMagicMLPv2

# CLS 默认开启，在Encoder，但是可以不用，只要传pooling tyoe 就可以换。 
# 默认CLS。没有修复不使用CLS因为前面从embedding - forward完就加了CLS的那一层。
# 这是错误的。修好了。8.3 // 7.31 Encoder 中的定义。保持一致。use_physics_mask=False
"""        self.encoder_layers = nn.ModuleList([
            PhysicsInformedTransformerEncoder(
                d_model, nhead=8, matrix_dim=matrix_dim, n_qubits=n_qubits,
                use_physics_mask=use_physics_mask（之前是=use_physics_attention）
            ) for _ in range(4)
        ])"""

class CompleteQuantumMagicPredictor(nn.Module):
    """Complete pipeline: embedding → transformer → physics pooling → MLP"""
    def __init__(self, matrix_dim=4, n_qubits=2, d_model=64,
                 pooling_type="cls", mlp_type="standard", use_physics_mask=False, mask_threshold=1, nhead=8, use_cls_token=True):
        super().__init__()
        self.matrix_dim = matrix_dim
        self.n_qubits = n_qubits
        self.d_model = d_model
        self.use_physics_mask = use_physics_mask
        self.pooling_type = pooling_type
        self.mask_threshold = mask_threshold
        self.nhead = nhead
        self.use_cls_token = use_cls_token
        
        # CLS token as a learnable parameter (to match saved weights)
        if use_cls_token or pooling_type == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        
        # Components
        self.embedding = StructuredQuantumEmbedding(n_qubits, d_model)
        self.encoder_layers = nn.ModuleList([
            PhysicsInformedTransformerEncoder(
                d_model, nhead=nhead, matrix_dim=matrix_dim, n_qubits=n_qubits,
                use_physics_mask=use_physics_mask, use_cls_token=use_cls_token,
                mask_threshold=mask_threshold
            ) for _ in range(4)
        ])

        if pooling_type != "cls":
            self.physics_pooling = PhysicsAwarePooling(d_model, matrix_dim, pooling_type)
        else:
            self.physics_pooling = None
        
        self.mlp_head = QuantumMagicMLPv2(d_model=d_model, mlp_type=mlp_type, n_qubits=n_qubits)
        #?1 不知到为什么在这里能看出来attention，也没看出来为什么有用
        print(f"🔧 Model: In Encoder: Msk={use_physics_mask}, and Msk Threshold={mask_threshold}\
            AND In Transformation: Pooling={pooling_type}\
            AND MLP={mlp_type}")
    
    def forward(self, rho_real, rho_imag, return_attention=False):
        # Embedding: [B, D, D] → [B, D², d_model] (matrix tokens only)
        matrix_tokens = self.embedding(rho_real, rho_imag)
        B = matrix_tokens.shape[0]
        
        # Add CLS token if using CLS pooling
        if self.pooling_type == "cls" and hasattr(self, 'cls_token'):
            # Expand CLS token for batch
            cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, d_model]
            tokens = torch.cat([cls_tokens, matrix_tokens], dim=1)  # [B, D²+1, d_model]
        else:
            tokens = matrix_tokens
        
        # Transformer processing
        attention_weights = []
        for layer in self.encoder_layers:
            if return_attention:
                tokens, attn_weights = layer(tokens, return_attention=True)
                attention_weights.append(attn_weights)
            else:
                tokens = layer(tokens)
        
        # Pooling: Choose between CLS token or physics-aware pooling
        if self.pooling_type == "cls":
            # Use CLS token (position 0)
            global_features = tokens[:, 0, :]
        else:
            # Use physics-aware pooling
            global_features = self.physics_pooling(tokens)
        
        # MLP prediction: [B, d_model] → [B, 1]
        magic_prediction = self.mlp_head(global_features)
        
        if return_attention:
            return magic_prediction, attention_weights
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
            x: [B, D²+1, d_model] - includes CLS token at position 0
        Returns:
            [B, d_model] - aggregated features
        """
        B, seq_len, d_model = x.shape
        D = self.matrix_dim
        
        # Extract matrix tokens (skip CLS at position 0)
        matrix_tokens = x[:, 1:, :]  # [B, D², d_model]
        
        if self.pooling_type == "mean":
            return matrix_tokens.mean(dim=1)
            
        elif self.pooling_type == "attention":
            # Learnable attention weights
            attn_weights = F.softmax(self.attention_weights(matrix_tokens), dim=1)
            return (matrix_tokens * attn_weights).sum(dim=1)
            
        elif self.pooling_type == "structured":
            # Physics-aware: separate diagonal from off-diagonal
            D = self.matrix_dim
            diag_indices = torch.arange(D, device=x.device) * (D + 1)
            
            all_indices = torch.arange(D * D, device=x.device)
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