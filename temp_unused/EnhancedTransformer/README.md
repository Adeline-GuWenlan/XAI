# Enhanced Transformer for Quantum Magic Prediction

This folder contains an enhanced BERT-style transformer implementation with configurable layers and loss functions, designed to work with the existing MixMLP codebase with minimal changes.

## Key Features

### 🚀 Enhanced Architecture
- **Configurable Layers**: Set number of transformer encoder layers (1-12+)
- **Auto FFN Sizing**: FFN automatically set to 4×d_model (BERT standard)
- **Multiple Loss Functions**: MSE (default) or SmoothL1Loss
- **Pre-norm Architecture**: Layer norm before attention/FFN (more stable)
- **Existing Components**: Reuses existing embedding and MLP components

### 📊 Model Configurations
- **num_layers**: Number of transformer layers (default: 6)
- **d_model**: Model dimension (512, 1024, 2048, etc.)
- **nhead**: Attention heads (4, 8, 16, etc.)
- **loss_type**: 'mse' or 'smoothl1'
- **pooling_type**: 'cls' (recommended)
- **mlp_type**: 'mixture_of_experts' (recommended)

## Files

### Core Implementation
- `enhanced_model.py`: Enhanced transformer model that extends existing components
- `train_enhanced.py`: Training script with minimal changes from original

### Removed Files
- `enhanced_transformer.py`: Full implementation (too many changes)
- `train_enhanced_simple.py`: Broken import dependencies

## Usage

### Basic Training
```bash
# From EnhancedTransformer directory
python train_enhanced.py --exp_name test_enhanced --num_qubits 3 --d_model 512 --num_layers 8

# With SmoothL1 loss
python train_enhanced.py --exp_name smooth_test --num_qubits 3 --d_model 1024 --num_layers 12 --loss_type smoothl1

# Quick test with dummy data
python train_enhanced.py --exp_name dummy_test --dummy_test --num_layers 4 --d_model 256 --epochs 5
```

### Advanced Configurations
```bash
# Deep model with 12 layers
python train_enhanced.py --exp_name deep_model --num_layers 12 --d_model 1024 --nhead 16 --epochs 100

# Large model with SmoothL1 loss
python train_enhanced.py --exp_name large_smooth --num_layers 8 --d_model 2048 --nhead 32 --loss_type smoothl1

# Fast experiment
python train_enhanced.py --exp_name quick --num_layers 4 --d_model 512 --epochs 20 --batch_size 1024
```

## Parameters

### Model Architecture
- `--num_layers`: Number of transformer layers (default: 6)
- `--d_model`: Model dimension (default: 512)
- `--nhead`: Number of attention heads (default: 8)
- `--loss_type`: Loss function - 'mse' or 'smoothl1' (default: 'mse')

### Training
- `--epochs`: Training epochs (default: 100)
- `--lr`: Learning rate (default: 1e-4)
- `--batch_size`: Batch size (default: 512)

### Data
- `--num_qubits`: Number of qubits (default: 3)
- `--data_folder`: Path to decoded tokens (default: '../Decoded_Tokens')
- `--labels_path`: Path to labels (default: '../Label/..._3_qubits_...')

## Key Improvements

### 1. **Configurable Depth**
```python
# Before: Fixed 4 layers
self.encoder_layers = nn.ModuleList([...] for _ in range(4))

# Now: Configurable layers
self.encoder = EnhancedTransformerEncoder(..., num_layers=args.num_layers)
```

### 2. **BERT-style Architecture**  
```python
# Pre-norm transformer layers
def forward(self, x):
    residual = x
    x = self.norm1(x)  # Norm before attention
    x = self.self_attn(x) + residual
    
    residual = x  
    x = self.norm2(x)  # Norm before FFN
    x = self.ffn(x) + residual
```

### 3. **Flexible Loss Functions**
```python
# MSE (default) or SmoothL1
if loss_type == "mse":
    self.loss_fn = nn.MSELoss()
elif loss_type == "smoothl1":
    self.loss_fn = nn.SmoothL1Loss()
```

### 4. **Auto FFN Sizing**
```python
# Always 4x d_model (BERT standard)
d_ff = 4 * d_model
self.ffn = nn.Sequential(
    nn.Linear(d_model, d_ff),
    nn.GELU(),
    nn.Linear(d_ff, d_model)
)
```

## Integration

### Minimal Changes Required
1. **No changes** to existing data loading
2. **No changes** to existing embedding/MLP components  
3. **Same interface** as original CompleteQuantumMagicPredictor
4. **Same training loop** structure as train_single.py

### Standalone Operation
- Can be copied/moved independently from main MixMLP folder
- All dependencies clearly imported from parent directory
- Self-contained experiment tracking and model saving

## Recommended Experiments

### Layer Depth Study
```bash
# Compare different depths
python train_enhanced.py --exp_name depth_4 --num_layers 4 --d_model 512
python train_enhanced.py --exp_name depth_6 --num_layers 6 --d_model 512  
python train_enhanced.py --exp_name depth_8 --num_layers 8 --d_model 512
python train_enhanced.py --exp_name depth_12 --num_layers 12 --d_model 512
```

### Loss Function Comparison
```bash
# MSE vs SmoothL1
python train_enhanced.py --exp_name mse_loss --num_layers 8 --loss_type mse
python train_enhanced.py --exp_name smooth_loss --num_layers 8 --loss_type smoothl1
```

### Model Scaling
```bash
# Scale model size and depth together
python train_enhanced.py --exp_name small_deep --num_layers 12 --d_model 512 --nhead 8
python train_enhanced.py --exp_name large_shallow --num_layers 4 --d_model 2048 --nhead 32
```

## Expected Performance

### Advantages of Enhanced Model
- **Better gradient flow** with pre-norm architecture
- **More expressive** with configurable depth
- **Robust training** with SmoothL1 loss option
- **BERT-proven architecture** for sequence modeling

### When to Use Enhanced vs Original
- **Enhanced**: Complex quantum systems, need deeper models, want stable training
- **Original**: Simple experiments, resource-constrained, baseline comparisons

## Output Structure

All outputs follow same structure as original:
```
experiments/
├── {exp_name}_{timestamp}/
│   ├── {exp_name}_{timestamp}.csv          # Training log
│   ├── {exp_name}_{timestamp}_config.json  # Model & training config
│   └── {exp_name}_{timestamp}_final_metrics.json  # Final test results
└── models/
    └── best_{exp_name}_{timestamp}.pth     # Best model weights
```