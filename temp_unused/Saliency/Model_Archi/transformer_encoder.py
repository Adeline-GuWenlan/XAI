# 尝试
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
# 7.31 让use_physics_mask=True  之前改了False但是好像不对。
# 我发现是哪里不对了：这个是Encoder里面的mask，不能改，效果是开了更好；但是之前use_physics_Attention=False是因为1. args没有输对，条件是、
# storing=True，不管输入要不要都会是True；现在改了args定义+switch on and off 作为对照实验。

class StructuredAttention(nn.Module):
    def __init__(self, d_model, nhead, matrix_dim, n_qubits, use_physics_mask=False, 
                 use_cls_token=True, mask_threshold=1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.matrix_dim = matrix_dim
        self.n_qubits = n_qubits
        self.use_physics_mask = use_physics_mask
        self.use_cls_token = use_cls_token  # 独立的CLS token控制
        self.mask_threshold = mask_threshold
        
        # Standard attention components
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        
        # Create masks but don't register as buffers to avoid loading issues
        # The masks will be created dynamically during forward pass
        self.physics_mask = None
        self.cls_mask = None
    
    def _create_physics_mask(self):
        """只处理物理结构的注意力限制，不涉及CLS token"""
        D = self.matrix_dim
        mask = torch.zeros(D * D, D * D)
        
        for i in range(D * D):
            for j in range(D * D):
                row_i, col_i = i // D, i % D
                row_j, col_j = j // D, j % D
                
                # Convert to binary strings and compare bit by bit
                row_i_bits = format(row_i, f'0{self.n_qubits}b')
                row_j_bits = format(row_j, f'0{self.n_qubits}b')
                col_i_bits = format(col_i, f'0{self.n_qubits}b')
                col_j_bits = format(col_j, f'0{self.n_qubits}b')
                
                shared_bits = 0
                compare=0
                while compare < D:
                    if row_i_bits[compare:compare+2] == row_j_bits[compare:compare+2]:
                        shared_bits += 1
                    if col_i_bits[compare:compare+2] == col_j_bits[compare:compare+2]:
                        shared_bits += 1
                    compare += 2
                if shared_bits >= self.mask_threshold:
                    mask[i, j] = 1.0
                    
        return mask
    
    def _create_cls_mask(self, seq_len):
        """创建CLS token的特殊处理mask"""
        mask = torch.zeros(seq_len, seq_len)
        
        # CLS token (position 0) 与所有token相互连接
        mask[0, :] = 1.0  # CLS attends to all
        mask[:, 0] = 1.0  # All attend to CLS
        
        return mask
    
    def _combine_masks(self, seq_len, device):
        """组合不同的mask"""
        # 开始时允许所有连接
        combined_mask = torch.ones(seq_len, seq_len, device=device)
        
        # Create physics mask dynamically if needed
        if self.use_physics_mask:
            if self.physics_mask is None:
                self.physics_mask = self._create_physics_mask().to(device)
            
            if self.use_cls_token:
                # 有CLS token：matrix elements在positions 1 to D²
                combined_mask[1:, 1:] = self.physics_mask
            else:
                # 无CLS token：matrix elements占据所有positions
                combined_mask = self.physics_mask
        
        # Create CLS mask dynamically if needed
        if self.use_cls_token:
            # Always create fresh mask with correct sequence length
            cls_mask = self._create_cls_mask(seq_len).to(device)
            
            # CLS token的连接优先级最高（覆盖physics mask的限制）
            cls_positions = (cls_mask == 1.0)
            combined_mask[cls_positions] = 1.0
            
        return combined_mask
    
    def forward(self, x, return_attention=False):
        B, N, D = x.shape
        # 形状就是 [B, N, D]，N是D²+1（如果有CLS token）
        
        # Compute Q, K, V
        Q = self.q_linear(x).view(B, N, self.nhead, D // self.nhead).transpose(1, 2)

        K = self.k_linear(x).view(B, N, self.nhead, D // self.nhead).transpose(1, 2)

        V = self.v_linear(x).view(B, N, self.nhead, D // self.nhead).transpose(1, 2)
        
        # Compute attention weights
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(D // self.nhead)
        
        # 组合所有mask
        if self.use_physics_mask or self.use_cls_token:
            combined_mask = self._combine_masks(N, device=scores.device)
            expanded_mask = combined_mask.unsqueeze(0).unsqueeze(0).expand(B, self.nhead, -1, -1)
            scores = scores.masked_fill(expanded_mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, V)
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        
        output = self.out_linear(out)
        
        if return_attention:
            return output, attn_weights
        return output


class PhysicsInformedTransformerEncoder(nn.Module):
    """Custom transformer encoder layer with optional quantum structure awareness"""
    def __init__(self, d_model, nhead, matrix_dim, n_qubits, dim_feedforward=512, 
                 dropout=0.05, use_physics_mask=False, mask_threshold=1, use_cls_token=True):
        super().__init__()
        
        self.structured_attention = StructuredAttention(
            d_model, nhead, matrix_dim, n_qubits, 
            use_physics_mask=use_physics_mask, 
            mask_threshold=mask_threshold,
            use_cls_token=use_cls_token
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Feedforward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        
    def forward(self, x, return_attention=False):
        # Attention with residual connection
        if return_attention:
            attn_out, attn_weights = self.structured_attention(x, return_attention=True)
        else:
            attn_out = self.structured_attention(x)
            attn_weights = None
            
        x = self.norm1(x + attn_out)
        # 不是完全update，保留原来的知识，增加学到的位点的权重  
        
        # FFN with residual connection
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        if return_attention:
            return x, attn_weights
        return x
