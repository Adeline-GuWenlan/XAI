import torch
import torch.nn as nn
import torch.nn.functional as F
# for 4 qubits: dmodel = 2048
class QuantumMagicMLPv2(nn.Module):
    """
    Enhanced MLP with quantum-aware components and attention mechanisms
    """
    def __init__(self, d_model=64, mlp_type="standard", n_qubits=2):
        super().__init__()
        self.d_model = d_model
        self.mlp_type = mlp_type
        self.n_qubits = n_qubits

        # Input processing
        self.input_norm = nn.LayerNorm(d_model)


        if mlp_type == "attention_enhanced":
            self.mlp_head = self._build_attention_mlp(d_model)
        elif mlp_type == "mixture_of_experts":
            self.mlp_head = self._build_moe_mlp(d_model)
        elif mlp_type == "asymmetric_ensemble":
            self.mlp_head = self._build_asymmetric_ensemble(d_model)
            self.ensemble_weights = nn.Parameter(torch.ones(3) / 3)
        elif mlp_type == "asymmetric_ensemble_v2":
            # High-dimensional 5-pathway ensemble
            self.mlp_head = self._build_asymmetric_ensemble_v2(d_model)
            self.ensemble_weights = nn.Parameter(torch.ones(5) / 5)
        else:  # "standard"
            self.mlp_head = self._build_standard_mlp(d_model)
    
    
    def _build_attention_mlp(self, d_model):
        """MLP with self-attention on intermediate features"""
        return nn.ModuleDict({
            'feature_extractor': nn.Sequential(
                nn.Linear(d_model, 256),
                nn.GELU(),
                nn.LayerNorm(256)
            ),
            'attention': nn.MultiheadAttention(
                embed_dim=256, num_heads=8, batch_first=True
                ),
            'post_attention': nn.Sequential(
                nn.LayerNorm(256),
                nn.Linear(256, 128),
                nn.GELU(),
                nn.LayerNorm(128),
                nn.Linear(128, 64),
                nn.GELU(),
                nn.Linear(64, 32),
                nn.GELU(),
                nn.Linear(32, 1)
            )
        })
    
    def _build_moe_mlp(self, d_model):
        """Mixture of Experts MLP for different magic detection strategies"""
        # set the num of experts / after gating dim dedicated for 4 qubits case
        # in proportion with qubits = 3, experts = 3, d_model = 512, after gating = 64
        num_experts = 4
        return nn.ModuleDict({
            'gating_network': nn.Sequential(
                nn.Linear(d_model, 512),
                nn.GELU(),
                nn.Linear(512, num_experts),
                nn.Softmax(dim=-1)
            ),
            'experts': nn.ModuleList([
                nn.Sequential(
                    nn.Linear(d_model, 512),
                    nn.GELU(),
                    nn.LayerNorm(512),
                    nn.Linear(512, 128),
                    nn.GELU(),
                    nn.LayerNorm(128),
                    nn.Linear(128, 32),
                    nn.GELU(),

                    nn.Linear(32, 1)
                ) for _ in range(num_experts)
            ])
        })
    
    def _build_asymmetric_ensemble(self, d_model):
        """Asymmetric ensemble with different pathway depths"""
        return nn.ModuleDict({
            # Deep pathway for complex patterns
            'deep_pathway': nn.Sequential(
                nn.Linear(d_model, 256),
                nn.GELU(),
                nn.LayerNorm(256),
                nn.Linear(256, 128),
                nn.GELU(),
                nn.LayerNorm(128),
                nn.Linear(128, 64),
                nn.GELU(),
                nn.Linear(64, 32),
                nn.GELU(),
                nn.Linear(32, 1)
            ),
            # Shallow pathway for direct patterns
            'shallow_pathway': nn.Sequential(
                nn.Linear(d_model, 64),
                nn.GELU(),
                nn.Linear(64, 1)
            ),
            # Medium pathway
            'medium_pathway': nn.Sequential(
                nn.Linear(d_model, 128),
                nn.GELU(),
                nn.LayerNorm(128),
                nn.Linear(128, 32),
                nn.GELU(),
                nn.Linear(32, 1)
            )
        })

    def _build_asymmetric_ensemble_v2(self, d_model):
        """
        Asymmetric ensemble for high-dimensional inputs (d_model >= 1024)
        Designed for d_model=2048 with 5 pathways reducing to (1024, 512, 256, 128, 64)
        """
        return nn.ModuleDict({
            # Pathway 1: 2048 -> 1024 -> 512 -> 256 -> 128 -> 64 -> 32 -> 1 (deepest)
            'pathway_1024': nn.Sequential(
                nn.Linear(d_model, 1024),
                nn.GELU(),
                nn.LayerNorm(1024),

                nn.Linear(1024, 512),
                nn.GELU(),
                nn.LayerNorm(512),

                nn.Linear(512, 256),
                nn.GELU(),
                nn.LayerNorm(256),

                nn.Linear(256, 128),
                nn.GELU(),
                nn.LayerNorm(128),

                nn.Linear(128, 64),
                nn.GELU(),
                nn.Linear(64, 32),
                nn.GELU(),
                nn.Linear(32, 1)
            ),

            # Pathway 2: 2048 -> 512 -> 256 -> 128 -> 64 -> 1 (deep)
            'pathway_512': nn.Sequential(
                nn.Linear(d_model, 512),
                nn.GELU(),
                nn.LayerNorm(512),

                nn.Linear(512, 256),
                nn.GELU(),
                nn.LayerNorm(256),

                nn.Linear(256, 128),
                nn.GELU(),
                nn.LayerNorm(128),


                nn.Linear(128, 64),
                nn.GELU(),
                nn.Linear(64, 1)
            ),

            # Pathway 3: 2048 -> 256 -> 128 -> 64 -> 1 (medium)
            'pathway_256': nn.Sequential(
                nn.Linear(d_model, 256),
                nn.GELU(),
                nn.LayerNorm(256),


                nn.Linear(256, 128),
                nn.GELU(),
                nn.LayerNorm(128),


                nn.Linear(128, 64),
                nn.GELU(),
                nn.Linear(64, 1)
            ),

            # Pathway 4: 2048 -> 128 -> 64 -> 1 (shallow)
            'pathway_128': nn.Sequential(
                nn.Linear(d_model, 128),
                nn.GELU(),
                nn.LayerNorm(128),


                nn.Linear(128, 64),
                nn.GELU(),
                nn.Linear(64, 1)
            ),

            # Pathway 5: 2048 -> 64 -> 1 (shallowest)
            'pathway_64': nn.Sequential(
                nn.Linear(d_model, 64),
                nn.GELU(),
                nn.Linear(64, 1)
            )
        })
    
    def _build_standard_mlp(self, d_model):
        """Standard progressive MLP (your current approach)"""
        return nn.Sequential(
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.LayerNorm(64),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, transformer_features):
        """Forward pass with different MLP architectures"""
        x = self.input_norm(transformer_features)

        if self.mlp_type == "attention_enhanced":
            return self._forward_attention(x)
        elif self.mlp_type == "mixture_of_experts":
            return self._forward_moe(x)
        elif self.mlp_type == "asymmetric_ensemble":
            return self._forward_asymmetric(x)
        elif self.mlp_type == "asymmetric_ensemble_v2":
            return self._forward_asymmetric_v2(x)
        else:  # standard
            return self.mlp_head(x)

    def _forward_attention(self, x):
        """Attention-enhanced forward pass"""
        # Extract features
        features = self.mlp_head['feature_extractor'](x)
        
        # Self-attention on features (treat as sequence of length 1)
        features_seq = features.unsqueeze(1)  # [B, 1, 256]
        attended_features, _ = self.mlp_head['attention'](features_seq, features_seq, features_seq)
        attended_features = attended_features.squeeze(1)  # [B, 256]
        
        # Residual connection
        features = features + attended_features
        
        # Final prediction
        return self.mlp_head['post_attention'](features)
    
    def _forward_moe(self, x):
        """Mixture of Experts forward pass"""
        # Compute gating weights
        gates = self.mlp_head['gating_network'](x)  # [B, num_experts]
        
        # Get expert outputs
        expert_outputs = []
        for expert in self.mlp_head['experts']:
            expert_outputs.append(expert(x))
        expert_outputs = torch.stack(expert_outputs, dim=-1)  # [B, 1, num_experts]
        
        # Weighted combination
        gates = gates.unsqueeze(1)  # [B, 1, num_experts]
        output = torch.sum(expert_outputs * gates, dim=-1)  # [B, 1]
        
        return output
    
    def _forward_asymmetric(self, x):
        """Asymmetric ensemble forward pass"""
        # Get outputs from different pathways
        deep_out = self.mlp_head['deep_pathway'](x)
        shallow_out = self.mlp_head['shallow_pathway'](x)
        medium_out = self.mlp_head['medium_pathway'](x)

        # Learnable ensemble weights
        weights = F.softmax(self.ensemble_weights, dim=0)

        # Weighted combination
        output = (weights[0] * deep_out +
                 weights[1] * shallow_out +
                 weights[2] * medium_out)

        return output

    def _forward_asymmetric_v2(self, x):
        """
        Asymmetric ensemble forward pass for 5-pathway architecture
        For high-dimensional inputs (d_model >= 1024)
        """
        # Get outputs from all 5 pathways
        out_1024 = self.mlp_head['pathway_1024'](x)  # Deepest
        out_512 = self.mlp_head['pathway_512'](x)    # Deep
        out_256 = self.mlp_head['pathway_256'](x)    # Medium
        out_128 = self.mlp_head['pathway_128'](x)    # Shallow
        out_64 = self.mlp_head['pathway_64'](x)      # Shallowest

        # Learnable ensemble weights (5 pathways)
        weights = F.softmax(self.ensemble_weights, dim=0)

        # Weighted combination
        output = (weights[0] * out_1024 +
                 weights[1] * out_512 +
                 weights[2] * out_256 +
                 weights[3] * out_128 +
                 weights[4] * out_64)

        return output
