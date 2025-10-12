import torch
import torch.nn as nn

class QuantumSizeExperts(nn.Module):
    """Expert projectors for different qubit system sizes"""
    def __init__(self, value_dim):
        super().__init__()
        self.value_dim = value_dim
        
        # System size-specific projectors
        self.expert_2q = nn.Linear(2, value_dim)  # 2-qubit: 4x4 matrix
        self.expert_3q = nn.Linear(2, value_dim)  # 3-qubit: 8x8 matrix  
        self.expert_4q = nn.Linear(2, value_dim)  # 4-qubit: 16x16 matrix
        self.expert_5q = nn.Linear(2, value_dim)  # 5-qubit: 32x32 matrix
        
        # Initialize with similar weights for consistency
        self._init_expert_weights()
        
    def _init_expert_weights(self):
        """Initialize all experts with similar weights to start"""
        reference_weight = self.expert_2q.weight.data.clone()
        reference_bias = self.expert_2q.bias.data.clone()
        
        # Copy to other experts with small variations
        for expert in [self.expert_3q, self.expert_4q, self.expert_5q]:
            expert.weight.data.copy_(reference_weight + torch.randn_like(reference_weight) * 0.01)
            expert.bias.data.copy_(reference_bias + torch.randn_like(reference_bias) * 0.01)
    
    def forward(self, complex_values, n_qubits):
        """Apply appropriate expert based on system size"""
        if n_qubits == 2:
            return self.expert_2q(complex_values)
        elif n_qubits == 3:
            return self.expert_3q(complex_values)
        elif n_qubits == 4:
            return self.expert_4q(complex_values)
        elif n_qubits == 5:
            return self.expert_5q(complex_values)
        else:
            raise ValueError(f"Unsupported qubit count: {n_qubits}")

class StructuredQuantumEmbedding(nn.Module):
    def __init__(self, n_qubits: int, d_model: int):
        super().__init__()
        self.n_qubits = n_qubits
        self.d_model = d_model
        
        # Split d_model into two semantic blocks
        self.value_dim = d_model - 2  # e.g., 62 for d_model=64
        self.position_dim = 2         # Always [pos_i, pos_j]
        
        # Use expert projectors for different system sizes
        self.quantum_experts = QuantumSizeExperts(self.value_dim)
        
        # NO CLS token at embedding level - we'll add it later if needed
        
    def forward(self, matrix_real, matrix_imag):
        """
        Output: [B, D², d_model] where d_model = [value_features, pos_i, pos_j]
        """
        batch_size, matrix_size, _ = matrix_real.shape
        
        # Process quantum values through expert projection
        matrix_values = torch.stack([matrix_real, matrix_imag], dim=-1)
        matrix_values = matrix_values.reshape(batch_size, matrix_size * matrix_size, 2)
        value_features = self.quantum_experts(matrix_values, self.n_qubits)  # [B, D², value_dim]
        
        # Create raw position coordinates
        positions = []
        for i in range(matrix_size):
            for j in range(matrix_size):
                positions.append([float(i), float(j)])
        
        position_features = torch.tensor(positions, dtype=torch.float32, device=matrix_real.device)
        position_features = position_features.unsqueeze(0).expand(batch_size, -1, -1)  # [B, D², 2]
        
        # Structured concatenation: [learned_values | raw_positions]
        structured_tokens = torch.cat([
            value_features,    # [B, D², value_dim] - semantic quantum features
            position_features  # [B, D², 2] - raw spatial coordinates
        ], dim=-1)  # [B, D², d_model]
        
        return structured_tokens
