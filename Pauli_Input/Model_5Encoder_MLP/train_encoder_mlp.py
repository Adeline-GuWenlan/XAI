"""
Model 5: Encoder-MLP Training Script
Training script for Encoder-MLP architecture with flexible pooling methods
Input: Pauli vectors of shape [B, 64]
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import os
import json
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr

from encoder_mlp_model import create_encoder_mlp_model, LassoMLPHead, GBDTMLPHead

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)


class QuantumDataset(Dataset):
    """Dataset for Pauli vectors and SN labels"""
    def __init__(self, X_path, Y_path):
        self.X = torch.from_numpy(np.load(X_path)).float()
        # Reshape Y to [n, 1] to match model output shape [batch_size, 1]
        self.Y = torch.from_numpy(np.load(Y_path)).float().unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0

    for X_batch, Y_batch in dataloader:
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)

        # Forward pass
        predictions = model(X_batch)
        base_loss = criterion(predictions, Y_batch)

        # Add L1 penalty for Lasso regularization
        if isinstance(model.mlp_head, LassoMLPHead):
            loss = base_loss + model.mlp_head.l1_penalty()
        else:
            loss = base_loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def evaluate(model, dataloader, criterion, device):
    """Evaluate the model with comprehensive metrics"""
    model.eval()
    total_loss = 0
    num_batches = 0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for X_batch, Y_batch in dataloader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)

            predictions = model(X_batch)
            loss = criterion(predictions, Y_batch)

            total_loss += loss.item()
            num_batches += 1

            # Collect predictions and targets for correlation metrics
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(Y_batch.cpu().numpy())

    # Concatenate all predictions and targets
    all_predictions = np.concatenate(all_predictions, axis=0).flatten()
    all_targets = np.concatenate(all_targets, axis=0).flatten()

    # Compute metrics
    mse = total_loss / num_batches

    # R-squared
    ss_res = np.sum((all_targets - all_predictions) ** 2)
    ss_tot = np.sum((all_targets - np.mean(all_targets)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

    # Spearman correlation
    spearman_corr, spearman_pval = spearmanr(all_predictions, all_targets)

    # Pearson correlation
    pearson_corr, pearson_pval = pearsonr(all_predictions, all_targets)

    metrics = {
        'mse': mse,
        'r_squared': r_squared,
        'spearman': spearman_corr,
        'spearman_pval': spearman_pval,
        'pearson': pearson_corr,
        'pearson_pval': pearson_pval
    }

    return metrics


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train Encoder-MLP model')
    parser.add_argument('--pooling', type=str, default='CLS', choices=['CLS', 'weight'],
                        help='Pooling method: CLS or weight')
    parser.add_argument('--d_model', type=int, default=64,
                        help='Dimension of transformer model')
    parser.add_argument('--nhead', type=int, default=4,
                        help='Number of attention heads')
    parser.add_argument('--num_encoder_layers', type=int, default=2,
                        help='Number of encoder layers')
    parser.add_argument('--dim_feedforward', type=int, default=512,
                        help='Dimension of feedforward network')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=30,
                        help='Number of training epochs')
    parser.add_argument('--train_split', type=float, default=0.95,
                        help='Train/validation split ratio')
    parser.add_argument('--head_type', type=str, default='mlp',
                        choices=['mlp', 'ensemble', 'lasso', 'gbdt'],
                        help='Regression head type')

    args = parser.parse_args()

    # Create timestamped results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    RESULTS_DIR = f"./results_{args.pooling.lower()}_{args.head_type}_{timestamp}"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"Results will be saved to: {RESULTS_DIR}")

    # Data paths
    DATA_DIR = "/Users/guwenlan/Desktop/XAI/Full_Ori/3q"
    X_PATH = os.path.join(DATA_DIR, "3q_100000samples_uniform_20251114_182049_pauli_vectors.npy")
    Y_PATH = os.path.join(DATA_DIR, "3q_100000samples_uniform_20251114_182049_sn_labels.npy")

    # Device - prioritize MPS (Apple Silicon GPU), then CUDA, then CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using device: MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using device: CUDA GPU")
    else:
        device = torch.device("cpu")
        print("Using device: CPU")

    # Load dataset
    print("Loading dataset...")
    dataset = QuantumDataset(X_PATH, Y_PATH)

    # Split into train and validation
    train_size = int(args.train_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    print(f"Train samples: {train_size}, Validation samples: {val_size}")
    print(f"Input shape: {dataset.X.shape}")

    # Configure head parameters based on head_type
    # Users can modify these parameters directly in encoder_mlp_model.py
    # or change them here for specific experiments
    head_params = {}
    if args.head_type == 'ensemble':
        # EnsembleMLPHead parameters - modify as needed
        head_params = {
            'deep_weight': 1.0,
            'shallow_weight': 1.0,
            'medium_weight': 1.0
        }
    elif args.head_type == 'lasso':
        # LassoMLPHead parameters - modify as needed
        head_params = {
            'lasso_lambda': 0.01
        }
    elif args.head_type == 'gbdt':
        # GBDTMLPHead parameters - modify as needed
        head_params = {
            'n_estimators': 100,
            'max_depth': 3,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_leaf_nodes': None,
            'learning_rate': 0.1,
            'subsample': 1.0,
            'use_mlp_refinement': True
        }

    # Create model
    print(f"\nCreating Encoder-MLP model with {args.pooling.upper()} pooling and {args.head_type.upper()} head...")
    model = create_encoder_mlp_model(
        pooling_method=args.pooling,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        dim_feedforward=args.dim_feedforward,
        head_type=args.head_type,
        head_params=head_params
    ).to(device)

    # Print model architecture
    print("\nModel Architecture:")
    print(model)

    # Count parameters by component
    encoder_params = sum(p.numel() for layer in model.encoder_layers for p in layer.parameters())
    pooling_params = sum(p.numel() for p in model.pooling.parameters())
    head_params_count = sum(p.numel() for p in model.mlp_head.parameters())
    other_params = sum(p.numel() for p in model.input_projection.parameters())
    if hasattr(model, 'cls_token') and model.cls_token is not None:
        other_params += model.cls_token.numel()
    total_params = sum(p.numel() for p in model.parameters())

    print(f"\nParameter breakdown:")
    print(f"  Input projection + CLS: {other_params:,}")
    print(f"  Encoder layers: {encoder_params:,}")
    print(f"  Pooling layer: {pooling_params:,}")
    print(f"  Regression head ({args.head_type}): {head_params_count:,}")
    print(f"  Total parameters: {total_params:,}")

    # Special handling for GBDT head: fit GBDT on pooled representations
    if args.head_type == 'gbdt':
        print("\nFitting GBDT on training data pooled representations...")
        model.eval()
        all_pooled = []
        all_targets = []

        with torch.no_grad():
            for X_batch, Y_batch in train_loader:
                X_batch = X_batch.to(device)
                pooled = model.get_pooled_representation(X_batch)
                all_pooled.append(pooled.cpu())
                all_targets.append(Y_batch)

        all_pooled = torch.cat(all_pooled, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        # Fit GBDT
        model.mlp_head.fit_gbdt(all_pooled, all_targets)
        print(f"GBDT fitted with {len(all_pooled)} samples")

        # Print feature importance if available
        feature_importance = model.mlp_head.get_feature_importance()
        if feature_importance is not None:
            print(f"GBDT feature importance (top 5): {feature_importance[:5]}")

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Training loop
    print("\nStarting training...")
    print("=" * 80)

    # Open epoch log file
    epoch_log_path = os.path.join(RESULTS_DIR, "epoch_log.txt")
    with open(epoch_log_path, 'w') as epoch_log:
        epoch_log.write("Epoch\tTrain_Loss\tVal_MSE\tVal_R2\tVal_Spearman\tVal_Pearson\n")

        train_losses = []
        val_metrics_history = {
            'mse': [],
            'r_squared': [],
            'spearman': [],
            'pearson': []
        }
        best_val_loss = float('inf')

        for epoch in range(args.num_epochs):
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
            val_metrics = evaluate(model, val_loader, criterion, device)

            train_losses.append(train_loss)
            val_metrics_history['mse'].append(val_metrics['mse'])
            val_metrics_history['r_squared'].append(val_metrics['r_squared'])
            val_metrics_history['spearman'].append(val_metrics['spearman'])
            val_metrics_history['pearson'].append(val_metrics['pearson'])

            # Log each epoch to file
            epoch_log.write(f"{epoch+1}\t{train_loss:.6f}\t{val_metrics['mse']:.6f}\t"
                          f"{val_metrics['r_squared']:.6f}\t{val_metrics['spearman']:.6f}\t"
                          f"{val_metrics['pearson']:.6f}\n")
            epoch_log.flush()

            # Print to console
            print(f"Epoch {epoch+1:3d}/{args.num_epochs}: Train Loss = {train_loss:.6f}, "
                  f"Val MSE = {val_metrics['mse']:.6f}, R² = {val_metrics['r_squared']:.4f}, "
                  f"Spearman = {val_metrics['spearman']:.4f}, Pearson = {val_metrics['pearson']:.4f}")

            # Save best model
            if val_metrics['mse'] < best_val_loss:
                best_val_loss = val_metrics['mse']
                torch.save(model.state_dict(), os.path.join(RESULTS_DIR, "best_model.pt"))

    print("=" * 80)
    print(f"\nTraining completed! Best validation loss: {best_val_loss:.6f}")
    print(f"Epoch log saved to: {epoch_log_path}")

    # Plot training curves
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Plot 1: MSE Loss
    axes[0, 0].plot(train_losses, label='Train MSE', color='blue')
    axes[0, 0].plot(val_metrics_history['mse'], label='Val MSE', color='red')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('MSE Loss')
    axes[0, 0].set_title('MSE Loss Curves')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Plot 2: R-squared
    axes[0, 1].plot(val_metrics_history['r_squared'], label='R²', color='green')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('R² Score')
    axes[0, 1].set_title('R² Score Over Training')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    axes[0, 1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[0, 1].axhline(y=1, color='gray', linestyle='--', alpha=0.5)

    # Plot 3: Spearman Correlation
    axes[1, 0].plot(val_metrics_history['spearman'], label='Spearman ρ', color='purple')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Spearman Correlation')
    axes[1, 0].set_title('Spearman Rank Correlation Over Training')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    axes[1, 0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[1, 0].axhline(y=1, color='gray', linestyle='--', alpha=0.5)

    # Plot 4: Pearson Correlation
    axes[1, 1].plot(val_metrics_history['pearson'], label='Pearson r', color='orange')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Pearson Correlation')
    axes[1, 1].set_title('Pearson Correlation Over Training')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    axes[1, 1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[1, 1].axhline(y=1, color='gray', linestyle='--', alpha=0.5)

    plt.suptitle(f'Model 5: Encoder-MLP ({args.pooling.upper()} pooling, {args.head_type.upper()} head) - Training Metrics',
                 fontsize=16, y=0.995)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "training_curves.png"), dpi=150, bbox_inches='tight')
    print(f"Training curves saved to {os.path.join(RESULTS_DIR, 'training_curves.png')}")

    # Save training log
    log = {
        "model": f"Encoder-MLP with {args.pooling.upper()} pooling and {args.head_type.upper()} head",
        "timestamp": timestamp,
        "device": str(device),
        "hyperparameters": {
            "pooling_method": args.pooling,
            "d_model": args.d_model,
            "nhead": args.nhead,
            "num_encoder_layers": args.num_encoder_layers,
            "dim_feedforward": args.dim_feedforward,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "num_epochs": args.num_epochs,
            "train_split": args.train_split,
            "head_type": args.head_type,
            "head_params": head_params
        },
        "data": {
            "X_path": X_PATH,
            "Y_path": Y_PATH,
            "input_shape": list(dataset.X.shape),
            "train_samples": train_size,
            "val_samples": val_size
        },
        "results": {
            "final_train_mse": float(train_losses[-1]),
            "final_val_mse": float(val_metrics_history['mse'][-1]),
            "final_val_r_squared": float(val_metrics_history['r_squared'][-1]),
            "final_val_spearman": float(val_metrics_history['spearman'][-1]),
            "final_val_pearson": float(val_metrics_history['pearson'][-1]),
            "best_val_mse": float(best_val_loss)
        }
    }

    with open(os.path.join(RESULTS_DIR, "training_log.json"), 'w') as f:
        json.dump(log, f, indent=4)
    print(f"Training log saved to {os.path.join(RESULTS_DIR, 'training_log.json')}")

    # Generate regression head structure description
    def get_head_structure_description(head_type, head_params, d_model):
        if head_type == 'mlp':
            return f"{d_model} -> 128 -> 256 -> 128 -> 1"
        elif head_type == 'ensemble':
            return {
                "type": "Ensemble with 3 pathways (deep, shallow, medium)",
                "deep_pathway": f"{d_model} -> 256 -> 128 -> 64 -> 32 -> 1 (5 layers)",
                "shallow_pathway": f"{d_model} -> 64 -> 1 (2 layers)",
                "medium_pathway": f"{d_model} -> 128 -> 32 -> 1 (3 layers)",
                "weights": f"learnable ensemble weights (initial: deep={head_params.get('deep_weight', 1.0)}, "
                          f"shallow={head_params.get('shallow_weight', 1.0)}, "
                          f"medium={head_params.get('medium_weight', 1.0)})"
            }
        elif head_type == 'lasso':
            return {
                "type": "MLP with L1 regularization",
                "structure": f"{d_model} -> 128 -> 256 -> 128 -> 1",
                "lasso_lambda": head_params.get('lasso_lambda', 0.01)
            }
        elif head_type == 'gbdt':
            return {
                "type": "Gradient Boosting Decision Tree + MLP refinement",
                "gbdt_params": {
                    "n_estimators": head_params.get('n_estimators', 100),
                    "max_depth": head_params.get('max_depth', 3),
                    "learning_rate": head_params.get('learning_rate', 0.1),
                    "min_samples_split": head_params.get('min_samples_split', 2),
                    "min_samples_leaf": head_params.get('min_samples_leaf', 1),
                    "subsample": head_params.get('subsample', 1.0)
                },
                "mlp_refinement": "1 -> 32 -> 16 -> 1" if head_params.get('use_mlp_refinement', True) else "disabled"
            }
        return "Unknown"

    # Save model parameters summary
    model_params = {
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "parameter_breakdown": {
            "input_projection_and_cls": other_params,
            "encoder_layers": encoder_params,
            "pooling_layer": pooling_params,
            "regression_head": head_params_count
        },
        "model_type": f"Encoder-MLP with {args.pooling.upper()} pooling and {args.head_type.upper()} head",
        "architecture_details": {
            "input_shape": list(dataset.X.shape),
            "d_model": args.d_model,
            "nhead": args.nhead,
            "num_encoder_layers": args.num_encoder_layers,
            "dim_feedforward": args.dim_feedforward,
            "pooling_method": args.pooling,
            "use_cls_token": args.pooling.upper() == 'CLS',
            "head_type": args.head_type,
            "head_structure": get_head_structure_description(args.head_type, head_params, args.d_model)
        }
    }

    with open(os.path.join(RESULTS_DIR, "model_parameters.json"), 'w') as f:
        json.dump(model_params, f, indent=4)
    print(f"Model parameters saved to {os.path.join(RESULTS_DIR, 'model_parameters.json')}")

    # Save final model
    torch.save(model.state_dict(), os.path.join(RESULTS_DIR, "final_model.pt"))
    print(f"Final model saved to {os.path.join(RESULTS_DIR, 'final_model.pt')}")

    # Generate regression head summary text
    def get_head_summary_text(head_type, head_params, d_model):
        if head_type == 'mlp':
            return f"  - MLP: {d_model} -> 128 -> 256 -> 128 -> 1"
        elif head_type == 'ensemble':
            text = f"  - Ensemble MLP Head with 3 pathways:\n"
            text += f"    - Deep pathway (5 layers): {d_model} -> 256 -> 128 -> 64 -> 32 -> 1\n"
            text += f"    - Shallow pathway (2 layers): {d_model} -> 64 -> 1\n"
            text += f"    - Medium pathway (3 layers): {d_model} -> 128 -> 32 -> 1\n"
            text += f"    - Ensemble weights (learnable): deep={head_params.get('deep_weight', 1.0)}, "
            text += f"shallow={head_params.get('shallow_weight', 1.0)}, medium={head_params.get('medium_weight', 1.0)}"
            return text
        elif head_type == 'lasso':
            text = f"  - Lasso MLP Head with L1 regularization:\n"
            text += f"    - Structure: {d_model} -> 128 -> 256 -> 128 -> 1\n"
            text += f"    - L1 penalty (lambda): {head_params.get('lasso_lambda', 0.01)}"
            return text
        elif head_type == 'gbdt':
            text = f"  - GBDT + MLP Hybrid Head:\n"
            text += f"    - GBDT: {head_params.get('n_estimators', 100)} estimators, "
            text += f"max_depth={head_params.get('max_depth', 3)}, "
            text += f"lr={head_params.get('learning_rate', 0.1)}\n"
            if head_params.get('use_mlp_refinement', True):
                text += f"    - MLP refinement: 1 -> 32 -> 16 -> 1"
            else:
                text += f"    - MLP refinement: disabled"
            return text
        return "  - Unknown head type"

    # Create summary report
    summary = f"""
Model 5: Encoder-MLP with {args.pooling.upper()} Pooling and {args.head_type.upper()} Head - Training Summary
{'=' * 80}
Timestamp: {timestamp}
Device: {device}

Data:
- Input shape: {dataset.X.shape}
- Total samples: {len(dataset)}
- Training samples: {train_size}
- Validation samples: {val_size}

Hyperparameters:
- Pooling method: {args.pooling.upper()}
- Regression head type: {args.head_type.upper()}
- Batch size: {args.batch_size}
- Learning rate: {args.learning_rate}
- Number of epochs: {args.num_epochs}
- Transformer d_model: {args.d_model}
- Number of attention heads: {args.nhead}
- Number of encoder layers: {args.num_encoder_layers}
- Feedforward dimension: {args.dim_feedforward}

Model Architecture:
- Type: Encoder-MLP with {args.pooling.upper()} pooling and {args.head_type.upper()} head
- CLS token: {'Yes' if args.pooling.upper() == 'CLS' else 'No'}
- Input: [B, 64] Pauli vectors -> [B, 64, 1] tokens
- Projection: 1 -> {args.d_model}
- Encoder: {args.num_encoder_layers} layers
- Pooling: {args.pooling.upper()} method
{get_head_summary_text(args.head_type, head_params, args.d_model)}

Parameter Counts:
- Input projection + CLS: {model_params['parameter_breakdown']['input_projection_and_cls']:,}
- Encoder layers: {model_params['parameter_breakdown']['encoder_layers']:,}
- Pooling layer: {model_params['parameter_breakdown']['pooling_layer']:,}
- Regression head: {model_params['parameter_breakdown']['regression_head']:,}
- Total parameters: {model_params['total_parameters']:,}

Results:
- Final training MSE: {train_losses[-1]:.6f}
- Final validation MSE: {val_metrics_history['mse'][-1]:.6f}
- Best validation MSE: {best_val_loss:.6f}

Evaluation Metrics (Final Epoch):
- R² Score: {val_metrics_history['r_squared'][-1]:.6f}
- Spearman Correlation: {val_metrics_history['spearman'][-1]:.6f}
- Pearson Correlation: {val_metrics_history['pearson'][-1]:.6f}

Files saved:
- best_model.pt: Best model checkpoint
- final_model.pt: Final model after all epochs
- training_curves.png: Comprehensive metrics visualization
- training_log.json: Detailed training log
- model_parameters.json: Model parameters summary
- epoch_log.txt: Per-epoch training log

Note: Edit regression head structures in encoder_mlp_model.py
"""

    with open(os.path.join(RESULTS_DIR, "summary.txt"), 'w') as f:
        f.write(summary)
    print(summary)
    print(f"Summary saved to {os.path.join(RESULTS_DIR, 'summary.txt')}")


if __name__ == "__main__":
    main()
