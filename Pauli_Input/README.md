# AbsValue Models - Stabilizer Norm Prediction

This directory contains four different models for predicting stabilizer norm from 2-qubit Pauli vectors.

## Quick Start

To train any model, navigate to its directory and run the training script:

```bash
cd Model1_LinearRegressor
python train_model1.py
```

## Directory Structure

```
AbsValue/
├── README.md                          # This file
├── TODO.md                            # Original task list
├── MODELS_COMPARISON.md               # Comprehensive performance comparison
│
├── Model1_LinearRegressor/
│   ├── train_model1.py                # Linear regression training script
│   └── results/
│       ├── best_model.pt              # Best checkpoint (MSE: 0.102481)
│       ├── final_model.pt             # Final model
│       ├── training_curves.png        # Loss visualization
│       ├── training_log.json          # Training details
│       ├── model_parameters.json      # Architecture summary
│       └── summary.txt                # Human-readable summary
│
├── Model2_SymmetricActivation/
│   ├── train_model2.py                # Tanh + ReLU training script
│   └── results/
│       ├── best_model.pt              # Best checkpoint (MSE: 0.017339)
│       └── ... (same structure as Model1)
│
├── Model3_AbsValueActivation/
│   ├── train_model3.py                # Abs + ReLU training script
│   └── results/
│       ├── best_model.pt              # Best checkpoint (MSE: 0.010633)
│       └── ... (same structure as Model1)
│
└── Model4_ReLUActivation/
    ├── train_model4.py                # ReLU + ReLU training script
    └── results/
        ├── best_model.pt              # Best checkpoint (MSE: 0.009034) ⭐ BEST
        └── ... (same structure as Model1)
```

## Performance Summary

| Model | Activation Functions | Best Val MSE | Improvement over Linear |
|-------|---------------------|--------------|------------------------|
| Model 1 | None (Linear) | 0.102481 | - (baseline) |
| Model 2 | Tanh + ReLU | 0.017339 | 83.1% better |
| Model 3 | Abs + ReLU | 0.010633 | 89.6% better |
| **Model 4** | **ReLU + ReLU** | **0.009034** | **91.2% better** ⭐ |

## Model Details

### Model 1: Linear Regressor
- **Architecture**: Simple linear regression (16 → 1)
- **Parameters**: 17
- **Use case**: Baseline comparison

### Model 2: MLP with Symmetric Activation
- **Architecture**: Linear → Tanh → Linear → ReLU → Dropout → Linear
- **Parameters**: 5,313
- **Key feature**: Tanh provides symmetric activation

### Model 3: MLP with Absolute Value Activation
- **Architecture**: Linear → |x| → Linear → ReLU → Dropout → Linear
- **Parameters**: 5,313
- **Key feature**: Custom absolute value activation preserves magnitude

### Model 4: MLP with ReLU Activations (BEST)
- **Architecture**: Linear → ReLU → Linear → ReLU → Dropout → Linear
- **Parameters**: 5,313
- **Key feature**: Standard proven architecture

## Data

- **Source**: `/Users/guwenlan/Desktop/XAI/INSPECT/Data`
- **Input**: `2q_1000samples_poisson_lam1.1_20251111_163332_pauli_vectors.npy`
- **Labels**: `2q_1000samples_poisson_lam1.1_20251111_163332_sn_labels.npy`
- **Samples**: 1000 (800 train, 200 validation)

## Hardware

All models trained on **Apple Silicon MPS GPU** for accelerated performance.

## Key Findings

1. **Non-linearity is essential**: All MLP models significantly outperform linear baseline
2. **ReLU works best**: Standard ReLU activations achieved the best performance
3. **Absolute value shows promise**: Only 15% worse than ReLU, may be useful for interpretability
4. **Model capacity helps**: 5,313 parameters vs 17 parameters made a huge difference

## Recommendations

- **Best accuracy**: Use Model 4 (ReLU + ReLU)
- **Interpretability**: Consider Model 3 (Abs + ReLU)
- **Quick baseline**: Model 1 for reference

## Files in Each Model's results/

- `best_model.pt`: Best model checkpoint based on validation loss
- `final_model.pt`: Model state after all training epochs
- `training_curves.png`: Training and validation loss curves
- `training_log.json`: Complete training configuration and results
- `model_parameters.json`: Model architecture and parameter count
- `summary.txt`: Human-readable training summary

## How to Load a Trained Model

```python
import torch
from train_model4 import MLPReLUActivation  # or other model class

# Load the best model
model = MLPReLUActivation(input_dim=16)
model.load_state_dict(torch.load('Model4_ReLUActivation/results/best_model.pt'))
model.eval()

# Use for inference
predictions = model(your_input_data)
```

## Citation

Data generated using Poisson sampling (λ=1.1) for 2-qubit systems.
Training timestamp: 2025-11-11
