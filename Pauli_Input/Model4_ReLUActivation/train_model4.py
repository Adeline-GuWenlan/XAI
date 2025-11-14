"""
Model 4: MLP with ReLU + ReLU
Training script for MLP with ReLU activation functions
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

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Paths
DATA_DIR = "/Users/guwenlan/Desktop/XAI/INSPECT/Data"

# Dataset paths - Updated to 50000 samples
X_PATH = os.path.join(DATA_DIR, "2q_50000samples_poisson_lam1.1_20251111_180142_pauli_vectors.npy")
Y_PATH = os.path.join(DATA_DIR, "2q_50000samples_poisson_lam1.1_20251111_180142_sn_labels.npy")


class QuantumDataset(Dataset):
    """Dataset for Pauli vectors and SN labels"""
    def __init__(self, X_path, Y_path):
        self.X = torch.from_numpy(np.load(X_path)).float()
        self.Y = torch.from_numpy(np.load(Y_path)).float()

        # Reshape Y if necessary
        if len(self.Y.shape) == 1:
            self.Y = self.Y.unsqueeze(1)

        print(f"Loaded data: X shape={self.X.shape}, Y shape={self.Y.shape}")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


class MLPReLUActivation(nn.Module):
    """MLP with ReLU + ReLU"""
    def __init__(self, input_dim, hidden_dim=64, dropout_rate=0.2, output_dim=1):
        super(MLPReLUActivation, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),      # MLP layer 1
            nn.ReLU(),                              # ReLU activation 1
            nn.Linear(hidden_dim, hidden_dim),      # MLP layer 2
            nn.ReLU(),                              # ReLU activation 2
            nn.Dropout(dropout_rate),               # Dropout
            nn.Linear(hidden_dim, output_dim)       # Output layer
        )

    def forward(self, x):
        return self.model(x)


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0

    for X_batch, Y_batch in dataloader:
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)

        # Forward pass
        predictions = model(X_batch)
        loss = criterion(predictions, Y_batch)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def evaluate(model, dataloader, criterion, device):
    """Evaluate the model"""
    model.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for X_batch, Y_batch in dataloader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)

            predictions = model(X_batch)
            loss = criterion(predictions, Y_batch)

            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches


def main():
    # Create timestamped results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    RESULTS_DIR = f"./results_{timestamp}"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"Results will be saved to: {RESULTS_DIR}")

    # Hyperparameters
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 200
    TRAIN_SPLIT = 0.8
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
    dataset = QuantumDataset(X_PATH, Y_PATH)

    # Split into train and validation
    train_size = int(TRAIN_SPLIT * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Train samples: {train_size}, Validation samples: {val_size}")

    # Create model
    input_dim = dataset.X.shape[1]
    model = MLPReLUActivation(input_dim, hidden_dim=HIDDEN_DIM, dropout_rate=DROPOUT_RATE).to(device)

    # Print model architecture
    print("\nModel Architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters())}")

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    print("\nStarting training...")
    print("=" * 80)

    # Open epoch log file
    epoch_log_path = os.path.join(RESULTS_DIR, "epoch_log.txt")
    with open(epoch_log_path, 'w') as epoch_log:
        epoch_log.write("Epoch\tTrain_Loss\tVal_Loss\n")

        train_losses = []
        val_losses = []
        best_val_loss = float('inf')

        for epoch in range(NUM_EPOCHS):
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss = evaluate(model, val_loader, criterion, device)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            # Log each epoch to file
            epoch_log.write(f"{epoch+1}\t{train_loss:.6f}\t{val_loss:.6f}\n")
            epoch_log.flush()  # Ensure it's written immediately

            # Print to console
            print(f"Epoch {epoch+1:3d}/{NUM_EPOCHS}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), os.path.join(RESULTS_DIR, "best_model.pt"))

    print("=" * 80)
    print(f"\nTraining completed! Best validation loss: {best_val_loss:.6f}")
    print(f"Epoch log saved to: {epoch_log_path}")

    # Plot training curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Model 4: MLP with ReLU + ReLU - Training Curves')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(RESULTS_DIR, "training_curves.png"), dpi=150, bbox_inches='tight')
    print(f"Training curves saved to {os.path.join(RESULTS_DIR, 'training_curves.png')}")

    # Save training log
    log = {
        "model": "MLP with ReLU + ReLU",
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
            "val_samples": val_size
        },
        "results": {
            "final_train_loss": float(train_losses[-1]),
            "final_val_loss": float(val_losses[-1]),
            "best_val_loss": float(best_val_loss)
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
            "output_dim": 1,
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
Model 4: MLP with ReLU + ReLU - Training Summary
=================================================
Timestamp: {timestamp}
Device: {device}

Data:
- Input dimension: {input_dim}
- Total samples: {len(dataset)}
- Training samples: {train_size}
- Validation samples: {val_size}

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
- Output: Linear({HIDDEN_DIM} -> 1)
- Total parameters: {model_params['total_parameters']}

Results:
- Final training loss (MSE): {train_losses[-1]:.6f}
- Final validation loss (MSE): {val_losses[-1]:.6f}
- Best validation loss (MSE): {best_val_loss:.6f}

Files saved:
- best_model.pt: Best model checkpoint
- final_model.pt: Final model after all epochs
- training_curves.png: Loss curves visualization
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
