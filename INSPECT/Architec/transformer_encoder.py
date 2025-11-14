import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class StructuredAttention(nn.Module):
    def __init__(self, d_model, nhead, matrix_dim, n_qubits, 
                 use_cls_token=True):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.matrix_dim = matrix_dim
        self.n_qubits = n_qubits
        self.use_cls_token = use_cls_token        
        # Standard attention components
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)


    #  we coudld use selfattention here
    def forward(self, x):
        B, N, D = x.shape
        
        # Compute Q, K, V
        Q = self.q_linear(x).view(B, N, self.nhead, D // self.nhead).transpose(1, 2)
        K = self.k_linear(x).view(B, N, self.nhead, D // self.nhead).transpose(1, 2)
        V = self.v_linear(x).view(B, N, self.nhead, D // self.nhead).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(D // self.nhead)
         
        # Only apply CLS mask if needed (but no physics mask)
        if self.use_cls_token:
            attention_mask = self._create_attention_mask(N, device=scores.device)
            # Since the mask is all ones, this doesn't actually restrict anything
            # But keeping for consistency with CLS token handling
            expanded_mask = attention_mask.unsqueeze(0).unsqueeze(0).expand(B, self.nhead, -1, -1)
            scores = scores.masked_fill(expanded_mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, V)
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        
        return self.out_linear(out)


class PhysicsInformedTransformerEncoder(nn.Module):
    """Transformer encoder layer with physics mask safely disabled"""
    def __init__(self, d_model, nhead, matrix_dim, n_qubits, dim_feedforward=256, use_cls_token=True):
        super().__init__()
        
        # Force disable physics mask regardless of input parameter
        self.structured_attention = StructuredAttention(
            d_model, nhead, matrix_dim, n_qubits,
            use_cls_token=use_cls_token
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Feedforward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, d_model),
        )
        
    def forward(self, x):
        # Attention with residual connection
        attn_out = self.structured_attention(x)
        x = self.norm1(x + attn_out)
        
        # FFN with residual connection
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x