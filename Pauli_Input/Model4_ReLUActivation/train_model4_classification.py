"""
Model 4: MLP with ReLU + ReLU - Classification Version
Training script for binary classification: label==0 vs label!=0
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Paths
DATA_DIR = "/Users/guwenlan/Desktop/XAI/Full_Ori/2q"

# Dataset paths
X_PATH = os.path.join(DATA_DIR, "2q_100000samples_uniform_20251114_194912_pauli_vectors.npy")
Y_PATH = os.path.join(DATA_DIR, "2q_100000samples_uniform_20251114_194912_sn_labels.npy")


class QuantumDatasetClassification(Dataset):
    """Dataset for Pauli vectors and binary classification labels (0 vs non-0)"""
    def __init__(self, X_path, Y_path):
        self.X = torch.from_numpy(np.load(X_path)).float()

        # Load labels and convert to binary classification
        Y_continuous = np.load(Y_path)
        # Binary classification: 0 if label==0, 1 if label!=0
        Y_binary = (Y_continuous != 0).astype(np.int64)
        self.Y = torch.from_numpy(Y_binary).long()

        # Store class distribution info
        self.class_0_count = np.sum(Y_binary == 0)
        self.class_1_count = np.sum(Y_binary == 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


class MLPReLUActivationClassifier(nn.Module):
    """MLP with ReLU + ReLU for binary classification"""
    def __init__(self, input_dim, hidden_dim=64, dropout_rate=0.2, num_classes=2):
        super(MLPReLUActivationClassifier, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),      # MLP layer 1
            nn.ReLU(),                              # ReLU activation 1
            nn.Linear(hidden_dim, hidden_dim),      # MLP layer 2
            nn.ReLU(),                              # ReLU activation 2
            nn.Dropout(dropout_rate),               # Dropout
            nn.Linear(hidden_dim, num_classes)      # Output layer (2 classes)
        )

    def forward(self, x):
        return self.model(x)


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0
    correct = 0
    total = 0

    for X_batch, Y_batch in dataloader:
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)

        # Forward pass
        logits = model(X_batch)
        loss = criterion(logits, Y_batch)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        # Calculate accuracy
        _, predicted = torch.max(logits, 1)
        total += Y_batch.size(0)
        correct += (predicted == Y_batch).sum().item()

    avg_loss = total_loss / num_batches
    accuracy = correct / total
    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device):
    """Evaluate the model with comprehensive classification metrics"""
    model.eval()
    total_loss = 0
    num_batches = 0
    all_predictions = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for X_batch, Y_batch in dataloader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)

            logits = model(X_batch)
            loss = criterion(logits, Y_batch)

            total_loss += loss.item()
            num_batches += 1

            # Get predictions and probabilities
            probs = torch.softmax(logits, dim=1)
            _, predicted = torch.max(logits, 1)

            # Collect predictions, probabilities, and targets
            all_predictions.append(predicted.cpu().numpy())
            all_targets.append(Y_batch.cpu().numpy())
            all_probs.append(probs[:, 1].cpu().numpy())  # Probability of class 1

    # Concatenate all predictions and targets
    all_predictions = np.concatenate(all_predictions)
    all_targets = np.concatenate(all_targets)
    all_probs = np.concatenate(all_probs)

    # Compute metrics
    avg_loss = total_loss / num_batches
    accuracy = accuracy_score(all_targets, all_predictions)
    precision = precision_score(all_targets, all_predictions, zero_division=0)
    recall = recall_score(all_targets, all_predictions, zero_division=0)
    f1 = f1_score(all_targets, all_predictions, zero_division=0)

    # Confusion matrix
    cm = confusion_matrix(all_targets, all_predictions)

    # ROC-AUC
    try:
        roc_auc = roc_auc_score(all_targets, all_probs)
    except:
        roc_auc = 0.0

    metrics = {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm,
        'predictions': all_predictions,
        'targets': all_targets,
        'probs': all_probs
    }

    return metrics


def main():
    # Create timestamped results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    RESULTS_DIR = f"./results_classification_{timestamp}"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"Results will be saved to: {RESULTS_DIR}")

    # Hyperparameters
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 30
    TRAIN_SPLIT = 0.95
    HIDDEN_DIM = 64
    DROPOUT_RATE = 0.0

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
    dataset = QuantumDatasetClassification(X_PATH, Y_PATH)

    # Print class distribution
    print(f"\nClass Distribution:")
    print(f"  Class 0 (label==0): {dataset.class_0_count} samples ({100*dataset.class_0_count/len(dataset):.2f}%)")
    print(f"  Class 1 (label!=0): {dataset.class_1_count} samples ({100*dataset.class_1_count/len(dataset):.2f}%)")

    # Split into train and validation
    train_size = int(TRAIN_SPLIT * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Train samples: {train_size}, Validation samples: {val_size}")

    # Create model
    input_dim = dataset.X.shape[1]
    model = MLPReLUActivationClassifier(input_dim, hidden_dim=HIDDEN_DIM, dropout_rate=DROPOUT_RATE, num_classes=2).to(device)

    # Print model architecture
    print("\nModel Architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters())}")

    # Loss and optimizer - CrossEntropyLoss for classification
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    print("\nStarting training...")
    print("=" * 80)

    # Open epoch log file
    epoch_log_path = os.path.join(RESULTS_DIR, "epoch_log.txt")
    with open(epoch_log_path, 'w') as epoch_log:
        epoch_log.write("Epoch\tTrain_Loss\tTrain_Acc\tVal_Loss\tVal_Acc\tVal_Precision\tVal_Recall\tVal_F1\tVal_ROC_AUC\n")

        train_losses = []
        train_accuracies = []
        val_metrics_history = {
            'loss': [],
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'roc_auc': []
        }
        best_val_acc = 0.0
        best_val_f1 = 0.0

        for epoch in range(NUM_EPOCHS):
            train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            val_metrics = evaluate(model, val_loader, criterion, device)

            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            val_metrics_history['loss'].append(val_metrics['loss'])
            val_metrics_history['accuracy'].append(val_metrics['accuracy'])
            val_metrics_history['precision'].append(val_metrics['precision'])
            val_metrics_history['recall'].append(val_metrics['recall'])
            val_metrics_history['f1'].append(val_metrics['f1'])
            val_metrics_history['roc_auc'].append(val_metrics['roc_auc'])

            # Log each epoch to file
            epoch_log.write(f"{epoch+1}\t{train_loss:.6f}\t{train_acc:.6f}\t"
                          f"{val_metrics['loss']:.6f}\t{val_metrics['accuracy']:.6f}\t"
                          f"{val_metrics['precision']:.6f}\t{val_metrics['recall']:.6f}\t"
                          f"{val_metrics['f1']:.6f}\t{val_metrics['roc_auc']:.6f}\n")
            epoch_log.flush()

            # Print to console
            print(f"Epoch {epoch+1:3d}/{NUM_EPOCHS}: "
                  f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f} | "
                  f"Val Loss={val_metrics['loss']:.4f}, Val Acc={val_metrics['accuracy']:.4f}, "
                  f"F1={val_metrics['f1']:.4f}, ROC-AUC={val_metrics['roc_auc']:.4f}")

            # Save best model based on F1 score
            if val_metrics['f1'] > best_val_f1:
                best_val_f1 = val_metrics['f1']
                best_val_acc = val_metrics['accuracy']
                torch.save(model.state_dict(), os.path.join(RESULTS_DIR, "best_model.pt"))

    print("=" * 80)
    print(f"\nTraining completed! Best validation F1: {best_val_f1:.6f}, Best validation accuracy: {best_val_acc:.6f}")
    print(f"Epoch log saved to: {epoch_log_path}")

    # Get final evaluation on validation set
    final_val_metrics = evaluate(model, val_loader, criterion, device)

    # Plot training curves with multiple subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Plot 1: Loss
    axes[0, 0].plot(train_losses, label='Train Loss', color='blue')
    axes[0, 0].plot(val_metrics_history['loss'], label='Val Loss', color='red')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Cross Entropy Loss')
    axes[0, 0].set_title('Loss Curves')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Plot 2: Accuracy
    axes[0, 1].plot(train_accuracies, label='Train Accuracy', color='blue')
    axes[0, 1].plot(val_metrics_history['accuracy'], label='Val Accuracy', color='green')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Accuracy Over Training')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    axes[0, 1].set_ylim([0, 1])

    # Plot 3: F1 Score
    axes[0, 2].plot(val_metrics_history['f1'], label='Val F1', color='purple')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('F1 Score')
    axes[0, 2].set_title('F1 Score Over Training')
    axes[0, 2].legend()
    axes[0, 2].grid(True)
    axes[0, 2].set_ylim([0, 1])

    # Plot 4: Precision & Recall
    axes[1, 0].plot(val_metrics_history['precision'], label='Precision', color='orange')
    axes[1, 0].plot(val_metrics_history['recall'], label='Recall', color='cyan')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_title('Precision & Recall Over Training')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    axes[1, 0].set_ylim([0, 1])

    # Plot 5: ROC-AUC
    axes[1, 1].plot(val_metrics_history['roc_auc'], label='ROC-AUC', color='magenta')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('ROC-AUC Score')
    axes[1, 1].set_title('ROC-AUC Over Training')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    axes[1, 1].set_ylim([0, 1])

    # Plot 6: Confusion Matrix (final epoch)
    cm = final_val_metrics['confusion_matrix']
    im = axes[1, 2].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    axes[1, 2].set_title('Confusion Matrix (Final Epoch)')
    axes[1, 2].set_xlabel('Predicted Label')
    axes[1, 2].set_ylabel('True Label')

    # Add colorbar
    cbar = plt.colorbar(im, ax=axes[1, 2])

    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axes[1, 2].text(j, i, format(cm[i, j], 'd'),
                          ha="center", va="center",
                          color="white" if cm[i, j] > thresh else "black")

    axes[1, 2].set_xticks([0, 1])
    axes[1, 2].set_yticks([0, 1])
    axes[1, 2].set_xticklabels(['Class 0', 'Class 1'])
    axes[1, 2].set_yticklabels(['Class 0', 'Class 1'])

    plt.suptitle('Model 4: MLP with ReLU + ReLU - Classification Training Metrics', fontsize=16, y=0.995)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "training_curves.png"), dpi=150, bbox_inches='tight')
    print(f"Training curves saved to {os.path.join(RESULTS_DIR, 'training_curves.png')}")

    # Plot ROC Curve
    fpr, tpr, thresholds = roc_curve(final_val_metrics['targets'], final_val_metrics['probs'])
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {final_val_metrics["roc_auc"]:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Binary Classification')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(RESULTS_DIR, "roc_curve.png"), dpi=150, bbox_inches='tight')
    print(f"ROC curve saved to {os.path.join(RESULTS_DIR, 'roc_curve.png')}")

    # Save training log
    log = {
        "model": "MLP with ReLU + ReLU - Binary Classification",
        "task": "Binary classification: label==0 vs label!=0",
        "timestamp": timestamp,
        "device": str(device),
        "hyperparameters": {
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "num_epochs": NUM_EPOCHS,
            "train_split": TRAIN_SPLIT,
            "hidden_dim": HIDDEN_DIM,
            "dropout_rate": DROPOUT_RATE
        },
        "data": {
            "X_path": X_PATH,
            "Y_path": Y_PATH,
            "input_dim": input_dim,
            "train_samples": train_size,
            "val_samples": val_size,
            "class_0_count": int(dataset.class_0_count),
            "class_1_count": int(dataset.class_1_count)
        },
        "results": {
            "final_train_loss": float(train_losses[-1]),
            "final_train_accuracy": float(train_accuracies[-1]),
            "final_val_loss": float(val_metrics_history['loss'][-1]),
            "final_val_accuracy": float(val_metrics_history['accuracy'][-1]),
            "final_val_precision": float(val_metrics_history['precision'][-1]),
            "final_val_recall": float(val_metrics_history['recall'][-1]),
            "final_val_f1": float(val_metrics_history['f1'][-1]),
            "final_val_roc_auc": float(val_metrics_history['roc_auc'][-1]),
            "best_val_f1": float(best_val_f1),
            "best_val_accuracy": float(best_val_acc),
            "confusion_matrix": cm.tolist()
        }
    }

    with open(os.path.join(RESULTS_DIR, "training_log.json"), 'w') as f:
        json.dump(log, f, indent=4)
    print(f"Training log saved to {os.path.join(RESULTS_DIR, 'training_log.json')}")

    # Save model parameters summary
    model_params = {
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "model_architecture": str(model),
        "layer_details": {
            "input_dim": input_dim,
            "hidden_dim": HIDDEN_DIM,
            "output_dim": 2,
            "num_classes": 2,
            "activation_1": "ReLU",
            "activation_2": "ReLU",
            "dropout_rate": DROPOUT_RATE
        }
    }

    with open(os.path.join(RESULTS_DIR, "model_parameters.json"), 'w') as f:
        json.dump(model_params, f, indent=4)
    print(f"Model parameters saved to {os.path.join(RESULTS_DIR, 'model_parameters.json')}")

    # Save final model
    torch.save(model.state_dict(), os.path.join(RESULTS_DIR, "final_model.pt"))
    print(f"Final model saved to {os.path.join(RESULTS_DIR, 'final_model.pt')}")

    # Create summary report
    summary = f"""
Model 4: MLP with ReLU + ReLU - Binary Classification Training Summary
======================================================================
Task: Binary Classification (label==0 vs label!=0)
Timestamp: {timestamp}
Device: {device}

Data:
- Input dimension: {input_dim}
- Total samples: {len(dataset)}
- Training samples: {train_size}
- Validation samples: {val_size}
- Class 0 (label==0): {dataset.class_0_count} ({100*dataset.class_0_count/len(dataset):.2f}%)
- Class 1 (label!=0): {dataset.class_1_count} ({100*dataset.class_1_count/len(dataset):.2f}%)

Hyperparameters:
- Batch size: {BATCH_SIZE}
- Learning rate: {LEARNING_RATE}
- Number of epochs: {NUM_EPOCHS}
- Hidden dimension: {HIDDEN_DIM}
- Dropout rate: {DROPOUT_RATE}

Model Architecture:
- Type: MLP with Sequential layers
- Layer 1: Linear({input_dim} -> {HIDDEN_DIM})
- Activation 1: ReLU
- Layer 2: Linear({HIDDEN_DIM} -> {HIDDEN_DIM})
- Activation 2: ReLU
- Dropout: {DROPOUT_RATE}
- Output: Linear({HIDDEN_DIM} -> 2) [Binary Classification]
- Total parameters: {model_params['total_parameters']}
- Loss function: CrossEntropyLoss

Results:
- Final training loss: {train_losses[-1]:.6f}
- Final training accuracy: {train_accuracies[-1]:.6f}
- Final validation loss: {val_metrics_history['loss'][-1]:.6f}
- Best validation F1: {best_val_f1:.6f}
- Best validation accuracy: {best_val_acc:.6f}

Evaluation Metrics (Final Epoch):
- Accuracy: {val_metrics_history['accuracy'][-1]:.6f}
- Precision: {val_metrics_history['precision'][-1]:.6f}
- Recall: {val_metrics_history['recall'][-1]:.6f}
- F1 Score: {val_metrics_history['f1'][-1]:.6f}
- ROC-AUC: {val_metrics_history['roc_auc'][-1]:.6f}

Confusion Matrix (Final Epoch):
{cm}

Files saved:
- best_model.pt: Best model checkpoint (based on F1 score)
- final_model.pt: Final model after all epochs
- training_curves.png: Comprehensive metrics visualization
- roc_curve.png: ROC curve
- training_log.json: Detailed training log
- model_parameters.json: Model parameters summary
- epoch_log.txt: Per-epoch training log
"""

    with open(os.path.join(RESULTS_DIR, "summary.txt"), 'w') as f:
        f.write(summary)
    print(summary)
    print(f"Summary saved to {os.path.join(RESULTS_DIR, 'summary.txt')}")


if __name__ == "__main__":
    main()
