# Mixed-Size Quantum Training Architecture

## Overview
This architecture enables training a single transformer model on quantum density matrices of different sizes (2-5 qubits) by embedding them into a universal feature space with consistent token representations.

## Core Concept: Universal Embedding Space

### Problem Statement
Traditional quantum magic prediction requires separate models for each qubit system size:
- 2-qubit: 4×4 matrices → 16 matrix elements
- 3-qubit: 8×8 matrices → 64 matrix elements  
- 4-qubit: 16×16 matrices → 256 matrix elements
- 5-qubit: 32×32 matrices → 1024 matrix elements

### Solution: Size-Agnostic Token Embedding
Map all quantum systems to the same **64-dimensional token space** regardless of matrix size, enabling mixed-size training and systematic extrapolation.

## Architecture Components

### 1. Expert Projector System (`QuantumSizeExperts`)

```python
class QuantumSizeExperts(nn.Module):
    def __init__(self, value_dim: int = 62):
        super().__init__()
        # System size-specific projectors
        self.expert_2q = nn.Linear(2, value_dim)  # 2-qubit specialist
        self.expert_3q = nn.Linear(2, value_dim)  # 3-qubit specialist  
        self.expert_4q = nn.Linear(2, value_dim)  # 4-qubit specialist
        self.expert_5q = nn.Linear(2, value_dim)  # 5-qubit specialist
```

**Key Features:**
- Each expert specializes in one qubit system size
- All experts project to the same **62D feature space**
- Initialized with similar weights + small random variations for consistency
- Only one expert activates per forward pass based on `n_qubits`

**Input/Output:**
- Input: `[B, D², 2]` complex matrix values (real, imaginary)
- Output: `[B, D², 62]` learned quantum features
- D varies by system: D=4 (2q), D=8 (3q), D=16 (4q), D=32 (5q)

### 2. Structured Quantum Embedding (`StructuredQuantumEmbedding`)

```python
# Token structure: [learned_quantum_features | raw_position_coordinates]
d_model = 64 = 62 (semantic features) + 2 (position i,j)
```

**Architecture Flow:**
1. **Matrix Flattening**: `[B, D, D]` → `[B, D², 2]` complex values
2. **Expert Projection**: `[B, D², 2]` → `[B, D², 62]` via size-specific expert
3. **Position Encoding**: Add raw `(i,j)` coordinates as last 2 dimensions
4. **Token Assembly**: `[B, D², 64]` structured tokens

**Token Semantics:**
- **Dimensions 0-61**: Learned quantum features from expert projectors
- **Dimensions 62-63**: Raw spatial coordinates `[pos_i, pos_j]`
- This separation enables the transformer to learn quantum physics vs. spatial relationships

### 3. Scale-Aware Transformer Processing

**Sequence Length Scaling:**
- 2-qubit: 16 tokens (4² matrix elements)
- 3-qubit: 64 tokens (8² matrix elements) 
- 4-qubit: 256 tokens (16² matrix elements)
- 5-qubit: 1024 tokens (32² matrix elements)

**Consistent Processing:**
- All tokens have identical 64D embedding regardless of system size
- Same transformer encoder layers process all qubit systems
- Physics-aware attention masks can be applied uniformly

### 4. Universal Pooling Strategy

**CLS Token Approach:**
```python
if self.use_cls_token:
    cls_tokens = self.cls_token.expand(batch_size, -1, -1)
    tokens = torch.cat([cls_tokens, tokens], dim=1)  # [B, D²+1, d_model]
    
# After transformer processing
global_features = tokens[:, 0, :]  # Extract CLS token
```

**Physics-Aware Pooling Alternative:**
- Separate diagonal vs off-diagonal matrix elements
- Preserves quantum mechanical structure across different system sizes
- Handles variable sequence lengths automatically

## Training Strategy

### Mixed-Size Training Protocol

1. **Batch Composition:**
   ```python
   # Example mixed batch
   batch = [
       (rho_2q_real, rho_2q_imag, magic_2q, n_qubits=2),
       (rho_3q_real, rho_3q_imag, magic_3q, n_qubits=3),
       (rho_4q_real, rho_4q_imag, magic_4q, n_qubits=4),
       # ...
   ]
   ```

2. **Dynamic Expert Selection:**
   - Model automatically routes to correct expert based on `n_qubits`
   - No manual switching or separate model loading required

3. **Consistent Loss Computation:**
   - Same MSE loss function across all system sizes
   - Magic values normalized to similar ranges for stable training

### Scalability Benefits

**Memory Efficiency:**
- Single model handles all qubit systems
- Expert projectors add minimal parameters (2×62×4 = 496 extra params)
- Shared transformer weights across all system sizes

**Training Efficiency:**
- Learn universal quantum representations
- Transfer learning between different qubit systems
- Smaller systems help regularize larger ones

## Expected Behavior

### Token Sequence Examples

**2-Qubit System (4×4 matrix):**
```
Input:  [B, 4, 4] complex matrices
Tokens: [B, 16, 64] where each token = [62D_features, i_coord, j_coord]
CLS:    [B, 17, 64] with CLS token at position 0
```

**3-Qubit System (8×8 matrix):**
```
Input:  [B, 8, 8] complex matrices  
Tokens: [B, 64, 64] where each token = [62D_features, i_coord, j_coord]
CLS:    [B, 65, 64] with CLS token at position 0
```

### Position Encoding Scaling
- 2-qubit: `i,j ∈ [0,3]` (4×4 matrix)
- 3-qubit: `i,j ∈ [0,7]` (8×8 matrix)
- 4-qubit: `i,j ∈ [0,15]` (16×16 matrix)
- Raw coordinates allow transformer to learn size-invariant spatial patterns

## Implementation Status

### Current Capabilities (`test_3qubit_pipeline.py`)
- ✅ Expert projectors working for 2-5 qubit systems
- ✅ Embedding scales from 16→64 tokens (2q→3q)
- ✅ 64D embedding space handles 3-qubit complexity
- ✅ Position encoding scales correctly to 8×8 matrices
- ✅ Complete model forward pass and gradient flow working

### Next Steps
1. **3-Qubit Validation**: Full training run to verify R² > 0.99 achievable
2. **Mixed Training**: Simultaneous training on 2+3 qubit data
3. **Scaling Analysis**: Determine optimal embedding dimensions for 4-5 qubit systems
4. **Extrapolation Testing**: Train on 2-4 qubits, test on 5-6 qubit systems

## Technical Advantages

1. **Universal Representation**: Same 64D space for all qubit systems enables transfer learning
2. **Scalable Architecture**: Linear parameter growth vs. exponential data growth  
3. **Physics Preservation**: Structured embedding maintains quantum mechanical relationships
4. **Inference Efficiency**: Single model deployment for all system sizes
5. **Systematic Extrapolation**: Foundation for 6+ qubit predictions beyond training data

This architecture transforms quantum magic prediction from a collection of size-specific models into a unified, scalable system capable of handling variable-size quantum systems through intelligent embedding design.