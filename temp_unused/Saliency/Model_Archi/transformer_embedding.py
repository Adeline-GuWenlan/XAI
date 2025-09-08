import torch
import torch.nn as nn

class StructuredQuantumEmbedding(nn.Module):
    def __init__(self, n_qubits: int, d_model: int):
        super().__init__()
        self.n_qubits = n_qubits
        self.d_model = d_model
        
        # Split d_model into two semantic blocks
        self.value_dim = d_model - 2  # e.g., 62 for d_model=64
        self.position_dim = 2         # Always [pos_i, pos_j]
        
        # Only project the quantum values
        self.value_projector = nn.Linear(2, self.value_dim)  # Complex → learned features
        
        # NO CLS token at embedding level - we'll add it later if needed
        
    def forward(self, matrix_real, matrix_imag):
        """
        Output: [B, D², d_model] where d_model = [value_features, pos_i, pos_j]
        """
        batch_size, matrix_size, _ = matrix_real.shape
        
        # Process quantum values through learned projection
        matrix_values = torch.stack([matrix_real, matrix_imag], dim=-1)
        matrix_values = matrix_values.reshape(batch_size, matrix_size * matrix_size, 2)
        value_features = self.value_projector(matrix_values)  # [B, D², value_dim]
        
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
