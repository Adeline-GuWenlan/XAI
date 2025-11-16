"""
Model 5: Encoder-MLP Architecture
A flexible model combining transformer encoder with MLP head.
Supports CLS token or weighted pooling methods.
Input: Pauli vectors of shape [B, N] where N is auto-detected
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class StructuredAttention(nn.Module):
    """Multi-head attention without physics mask, with optional CLS token support"""
    def __init__(self, d_model, nhead, use_cls_token=True):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.use_cls_token = use_cls_token

        # Standard attention components
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)

    def forward(self, x, return_attention=False):
        B, N, D = x.shape

        # Compute Q, K, V
        Q = self.q_linear(x).view(B, N, self.nhead, D // self.nhead).transpose(1, 2)
        K = self.k_linear(x).view(B, N, self.nhead, D // self.nhead).transpose(1, 2)
        V = self.v_linear(x).view(B, N, self.nhead, D // self.nhead).transpose(1, 2)

        # Compute attention weights
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(D // self.nhead)
        attn_weights = F.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, V)
        out = out.transpose(1, 2).contiguous().view(B, N, D)

        output = self.out_linear(out)

        if return_attention:
            return output, attn_weights
        return output


class TransformerEncoderLayer(nn.Module):
    """Transformer encoder layer without physics mask or dropout"""
    def __init__(self, d_model, nhead, dim_feedforward=512, use_cls_token=True):
        super().__init__()

        self.structured_attention = StructuredAttention(
            d_model, nhead, use_cls_token=use_cls_token
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Feedforward network (no dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )

    def forward(self, x, return_attention=False):
        # Attention with residual connection
        if return_attention:
            attn_out, attn_weights = self.structured_attention(x, return_attention=True)
        else:
            attn_out = self.structured_attention(x)
            attn_weights = None

        x = self.norm1(x + attn_out)

        # FFN with residual connection
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        if return_attention:
            return x, attn_weights
        return x


class PoolingLayer(nn.Module):
    """Flexible pooling layer supporting CLS token extraction or weighted aggregation"""
    def __init__(self, d_model, method='CLS'):
        """
        Args:
            d_model: Dimension of the model
            method: Pooling method - 'CLS' or 'weight'
                - 'CLS': Extract CLS token at position 0
                - 'weight': Weighted aggregation of all tokens
        """
        super().__init__()
        self.method = method.upper()
        self.d_model = d_model

        if self.method == 'WEIGHT':
            # Learnable weights for weighted aggregation
            self.weight_network = nn.Sequential(
                nn.Linear(d_model, 1),
                nn.Softmax(dim=1)
            )

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [B, N, D]
               where N includes CLS token at position 0 if method='CLS'

        Returns:
            Pooled tensor of shape [B, D]
        """
        if self.method == 'CLS':
            # Extract CLS token at position 0
            return x[:, 0, :]  # [B, D]

        elif self.method == 'WEIGHT':
            # Compute attention weights for each token
            weights = self.weight_network(x)  # [B, N, 1]

            # Weighted sum across sequence dimension
            pooled = torch.sum(x * weights, dim=1)  # [B, D]

            return pooled

        else:
            raise ValueError(f"Unknown pooling method: {self.method}. Use 'CLS' or 'weight'.")


class MLPHead(nn.Module):
    """
    MLP head with ReLU activations
    Edit the structure directly in this class as needed
    """
    def __init__(self, input_dim):
        super().__init__()

        # Edit this structure as needed - default matches train_model4.py
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),      # MLP layer 1
            nn.ReLU(),                      # ReLU activation 1
            nn.Linear(128, 256),            # MLP layer 2
            nn.ReLU(),                      # ReLU activation 2
            nn.Linear(256, 128),            # MLP layer 3
            nn.ReLU(),                      # ReLU activation 3
            nn.Linear(128, 1)               # Output layer
        )

    def forward(self, x):
        return self.model(x)

#NOTE: for the TODO below, use as much package as possible, ie avoding creating your own functions or class. 
#NOTE: However, do create a seperate .md doc(under"/Users/guwenlan/Desktop/XAI/Pauli_Input/Model_5Encoder_MLP") 
#NOTE: that specify the parameter of each class or method u called and listed all the paramters, whether default or modified, in the script. 
#TODO: create a new regression head below, parallel to MLPReLUActivation, and instead use a voting mechanism that add three MLPs(of different depth) outputs with different weight. reference the _build_asymmetric_ensemble in '/Users/guwenlan/Desktop/XAI/Model_Archi/mlp.py'

#TODO: create a new regression head below, with a Lasso inplementation with selfdefined penalty lambda

#TODO: create a new regression head below, with a Gradient boosting decision tree and a regression head for precise regrression result. explicitly accept all possible restriction on growing a tree like depth, num of lead node, num sample to be assigned,, and allow user to self define them in the script. 
#NOTE: note that no need to expose the tuning parameters, ie the depth of tree, Lasso penalty to the actual "/Users/guwenlan/Desktop/XAI/Pauli_Input/Model_5Encoder_MLP/train_encoder_mlp.py" script to avoid redundancy. assume user would change inplace.
#NOTE: However update the /Users/guwenlan/Desktop/XAI/Pauli_Input/Model_5Encoder_MLP/train_encoder_mlp.py to also load all parameter of the actual regreassion head chosen by the user for replicability. 
#NOTE: finally,

class EncoderMLPModel(nn.Module):
    """
    Complete Encoder-MLP model architecture for Pauli vectors

    Architecture:
        Input [B, N] Pauli vectors (N is auto-detected)
        → Reshape to [B, N, 1] (N tokens, 1 feature each)
        → Project to [B, N, d_model]
        → Optional CLS token → [B, N+1, d_model] or [B, N, d_model]
        → Transformer encoder layers
        → Pooling (CLS extraction or weighted aggregation) → [B, d_model]
        → MLP head → [B, 1]
    """
    def __init__(
        self,
        d_model=64,
        nhead=4,
        num_encoder_layers=2,
        dim_feedforward=512,
        pooling_method='CLS'
    ):
        """
        Args:
            d_model: Dimension of transformer model
            nhead: Number of attention heads
            num_encoder_layers: Number of encoder layers
            dim_feedforward: Dimension of feedforward network
            pooling_method: 'CLS' or 'weight'
        """
        super().__init__()

        self.d_model = d_model
        self.pooling_method = pooling_method.upper()
        self.use_cls_token = (self.pooling_method == 'CLS')

        # Input projection: each Pauli coefficient (1 feature) → d_model
        self.input_projection = nn.Linear(1, d_model)

        # CLS token (learnable) - only if using CLS pooling method
        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                use_cls_token=self.use_cls_token
            )
            for _ in range(num_encoder_layers)
        ])

        # Pooling layer
        self.pooling = PoolingLayer(d_model=d_model, method=pooling_method)

        # MLP head - edit structure in MLPHead class as needed
        self.mlp_head = MLPHead(input_dim=d_model)

    def forward(self, x, return_attention=False):
        """
        Args:
            x: Input tensor of shape [B, N] - Pauli vectors (N auto-detected)
            return_attention: Whether to return attention weights

        Returns:
            output: Predictions of shape [B, 1]
            attention_weights: (optional) List of attention weights from each layer
        """
        B, N = x.shape

        # Reshape Pauli vector [B, N] to [B, N, 1]
        # Each of the N Pauli coefficients becomes a token with 1 feature
        x = x.unsqueeze(-1)  # [B, N, 1]

        # Project each token from 1 feature to d_model dimensions
        x = self.input_projection(x)  # [B, N, d_model]

        # Prepend CLS token if using CLS pooling
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, d_model]
            x = torch.cat([cls_tokens, x], dim=1)  # [B, N+1, d_model]

        # Pass through encoder layers
        attention_weights = []
        for encoder_layer in self.encoder_layers:
            if return_attention:
                x, attn_weights = encoder_layer(x, return_attention=True)
                attention_weights.append(attn_weights)
            else:
                x = encoder_layer(x)

        # Pooling
        pooled = self.pooling(x)  # [B, d_model]

        # MLP head
        output = self.mlp_head(pooled)  # [B, 1]

        if return_attention:
            return output, attention_weights
        return output

    def get_pooled_representation(self, x):
        """
        Get the pooled representation before the MLP head
        Useful for visualization or transfer learning

        Args:
            x: Input tensor of shape [B, N] - Pauli vectors (N auto-detected)

        Returns:
            Pooled representation of shape [B, d_model]
        """
        B, N = x.shape

        # Reshape and project
        x = x.unsqueeze(-1)  # [B, N, 1]
        x = self.input_projection(x)  # [B, N, d_model]

        # Prepend CLS token if needed
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)

        # Pass through encoder
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)

        # Pool
        pooled = self.pooling(x)

        return pooled


def create_encoder_mlp_model(
    pooling_method='CLS',
    d_model=64,
    nhead=4,
    num_encoder_layers=2,
    dim_feedforward=512
):
    """
    Factory function to create EncoderMLPModel with sensible defaults

    Example usage:
        # CLS token pooling
        model = create_encoder_mlp_model(pooling_method='CLS')

        # Weighted aggregation pooling
        model = create_encoder_mlp_model(pooling_method='weight')
    """
    return EncoderMLPModel(
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        dim_feedforward=dim_feedforward,
        pooling_method=pooling_method
    )


if __name__ == "__main__":
    # Test the model with both pooling methods
    print("Testing EncoderMLPModel with Pauli vector input...")

    # Test dimensions - Pauli vectors
    batch_size = 4
    pauli_dim = 64  # 64 Pauli coefficients

    # Create dummy input - Pauli vectors
    x = torch.randn(batch_size, pauli_dim)

    print("\n1. Testing CLS token pooling:")
    model_cls = create_encoder_mlp_model(
        pooling_method='CLS',
        d_model=64,
        nhead=4,
        num_encoder_layers=2
    )
    print(f"Model parameters: {sum(p.numel() for p in model_cls.parameters()):,}")

    output_cls = model_cls(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output_cls.shape}")
    print(f"Output: {output_cls[:2]}")

    # Test with attention weights
    output_cls, attn_weights = model_cls(x, return_attention=True)
    print(f"Number of attention weight tensors: {len(attn_weights)}")
    print(f"Attention weights shape (first layer): {attn_weights[0].shape}")

    print("\n2. Testing weighted aggregation pooling:")
    model_weight = create_encoder_mlp_model(
        pooling_method='weight',
        d_model=64,
        nhead=4,
        num_encoder_layers=2
    )
    print(f"Model parameters: {sum(p.numel() for p in model_weight.parameters()):,}")

    output_weight = model_weight(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output_weight.shape}")
    print(f"Output: {output_weight[:2]}")

    # Test pooled representation extraction
    pooled_repr = model_weight.get_pooled_representation(x)
    print(f"\nPooled representation shape: {pooled_repr.shape}")

    print("\nAll tests passed!")
    print("\nNote: Edit the MLP structure directly in the MLPHead class as needed.")
