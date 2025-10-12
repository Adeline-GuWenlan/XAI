#!/usr/bin/env python3
"""
Single experiment training script - runs one experiment configuration
Updated with clean parameters and dummy test mode for local testing
"""

import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import csv
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr
from quantum_dataset import create_dataloaders
from updated_training_model import CompleteQuantumMagicPredictor
import argparse
import sys
import json
from datetime import datetime
import time
import fcntl
from pathlib import Path
# Updated import for PyTorch 2.5.1+
from torch.amp import GradScaler, autocast

torch.backends.cudnn.benchmark = True

class GPUManager:
    """Automatic GPU assignment to prevent conflicts"""
    
    @staticmethod
    def get_gpu_memory_usage():
        """Get memory usage for all available GPUs"""
        if not torch.cuda.is_available():
            return []
        
        gpu_memory = []
        for i in range(torch.cuda.device_count()):
            torch.cuda.set_device(i)
            memory_allocated = torch.cuda.memory_allocated(i)
            memory_reserved = torch.cuda.memory_reserved(i)
            memory_total = torch.cuda.get_device_properties(i).total_memory
            
            usage_percent = (memory_reserved / memory_total) * 100
            gpu_memory.append({
                'gpu_id': i,
                'allocated': memory_allocated,
                'reserved': memory_reserved,
                'total': memory_total,
                'usage_percent': usage_percent,
                'free_memory': memory_total - memory_reserved
            })
        
        return gpu_memory
    
    @staticmethod
    def select_best_gpu():
        """Select GPU with most available memory"""
        if not torch.cuda.is_available():
            print("🔧 No CUDA GPUs available, using CPU")
            return torch.device('cpu')
        
        gpu_info = GPUManager.get_gpu_memory_usage()
        if not gpu_info:
            return torch.device('cpu')
        
        # Sort by free memory (descending)
        best_gpu = max(gpu_info, key=lambda x: x['free_memory'])
        
        print(f"🔧 GPU Selection Summary:")
        for gpu in gpu_info:
            marker = "👉" if gpu['gpu_id'] == best_gpu['gpu_id'] else "  "
            print(f"   {marker} GPU {gpu['gpu_id']}: {gpu['usage_percent']:.1f}% used, "
                  f"{gpu['free_memory']/1e9:.1f}GB free")
        
        selected_device = torch.device(f"cuda:{best_gpu['gpu_id']}")
        print(f"✅ Auto-selected: {selected_device}")
        
        return selected_device
    
    @staticmethod
    def create_gpu_lock_file(gpu_id, exp_name):
        """Create a lock file to coordinate GPU usage across processes"""
        lock_dir = "/tmp/gpu_locks"
        os.makedirs(lock_dir, exist_ok=True)
        
        lock_file = f"{lock_dir}/gpu_{gpu_id}_{exp_name}.lock"
        
        try:
            with open(lock_file, 'w') as f:
                f.write(f"pid:{os.getpid()}\nexp:{exp_name}\ntime:{datetime.now().isoformat()}\n")
            print(f"🔒 Created GPU lock: {lock_file}")
            return lock_file
        except Exception as e:
            print(f"⚠️ Could not create lock file: {e}")
            return None
    
    @staticmethod
    def cleanup_gpu_lock(lock_file):
        """Clean up GPU lock file"""
        try:
            if lock_file and os.path.exists(lock_file):
                os.remove(lock_file)
                print(f"🔓 Cleaned up GPU lock: {lock_file}")
        except Exception as e:
            print(f"⚠️ Could not clean up lock file: {e}")


class ExperimentPathManager:
    """Manages unique paths for each experiment to prevent conflicts"""
    
    def __init__(self, exp_name, base_log_dir="experiments", base_model_dir="models"):
        self.exp_name = exp_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create unique experiment directory
        self.exp_dir = Path(base_log_dir) / f"{exp_name}_{self.timestamp}"
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Create unique model directory  
        self.model_dir = Path(base_model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Define all unique paths
        self.log_path = self.exp_dir / f"{exp_name}.csv"
        self.metrics_path = self.exp_dir / f"{exp_name}_final_metrics.json"
        self.model_path = self.model_dir / f"best_{exp_name}_{self.timestamp}.pth"
        self.config_path = self.exp_dir / f"{exp_name}_config.json"
        
        print(f"📁 Experiment Paths Created:")
        print(f"   📊 Log: {self.log_path}")
        print(f"   📈 Metrics: {self.metrics_path}")
        print(f"   🤖 Model: {self.model_path}")
        print(f"   ⚙️ Config: {self.config_path}")
    
    def save_experiment_config(self, config_dict):
        """Save complete experiment configuration"""
        config_dict['experiment_info'] = {
            'exp_name': self.exp_name,
            'timestamp': self.timestamp,
            'exp_dir': str(self.exp_dir),
            'model_path': str(self.model_path),
            'log_path': str(self.log_path)
        }
        
        with open(self.config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        print(f"✅ Experiment config saved: {self.config_path}")
    
    def get_visualization_command(self, data_folder, labels_path):
        """Generate visualization command for this specific experiment"""
        return f"""python Visualization.py \\
    --model_path '{self.model_path}' \\
    --data_folder '{data_folder}' \\
    --labels_path '{labels_path}' \\
    --config_from_json '{self.config_path}' \\
    --output_prefix '{self.exp_name}_{self.timestamp}'"""


def create_dummy_data(batch_size=8, matrix_dim=4):
    """Create dummy data for testing"""
    print(f"🧪 Creating dummy data: batch_size={batch_size}, matrix_dim={matrix_dim}")
    
    # Create dummy quantum matrices
    rho_real = torch.randn(batch_size, matrix_dim, matrix_dim)
    rho_imag = torch.randn(batch_size, matrix_dim, matrix_dim)
    
    # Create dummy magic labels (random values between 0 and 1)
    magic_labels = torch.rand(batch_size, 1)
    
    # Create simple dataset
    dummy_dataset = torch.utils.data.TensorDataset(rho_real, rho_imag, magic_labels)
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(dummy_dataset, batch_size=4, shuffle=True)
    val_loader = torch.utils.data.DataLoader(dummy_dataset, batch_size=4, shuffle=False)
    test_loader = torch.utils.data.DataLoader(dummy_dataset, batch_size=4, shuffle=False)
    
    print(f"✅ Dummy data created: {len(dummy_dataset)} samples")
    return train_loader, val_loader, test_loader


def setup_logging(log_path, model_params):
    """Setup file logging with model parameters in header"""
    class EpochLogger:
        def __init__(self, log_path, model_params):
            self.log_path = log_path
            # Match the tolerances used in compute_metrics function
            self.fieldnames = ["epoch","split","mse","mae","r2","spearman","pearson","lr","loss"] + [f"acc@{tol}" for tol in [0.01,0.05,0.1,0.2,0.3,0.4,0.5]]
            
            # Initialize CSV file with model parameters as comments and headers
            with open(log_path, 'w', newline='') as f:
                # Write model parameters as comments at the top
                f.write(f"# Model Parameters for Reconstruction:\n")
                for key, value in model_params.items():
                    f.write(f"# {key}: {value}\n")
                f.write(f"# training_started: {datetime.now().isoformat()}\n")
                f.write(f"#\n")
                
                # Write CSV headers
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()
        
        def log_epoch(self, metrics):
            """Log only epoch-level results to file"""
            with open(self.log_path, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writerow(metrics)
    
    return EpochLogger(log_path, model_params)

def train_model(model, train_loader, val_loader, test_loader, device,
                epochs=50, lr=1e-4, path_manager=None, exp_name="experiment",
                model_params=None, gpu_lock_file=None):
    """
    Training function with isolated paths, optimized for GPU with Automatic Mixed Precision.
    Updated for PyTorch 2.5.1+ torch.amp API
    ❌ REMOVED: mask_threshold parameter (not valid for this function)
    """
    # Setup logging with model parameters
    epoch_logger = setup_logging(path_manager.log_path, model_params)

    print(f"\n🚀 Starting training for {exp_name}")
    print(f"   Model: {model.__class__.__name__}")
    print(f"   Device: {device}")
    print(f"   Epochs: {epochs}, LR: {lr}")
    print(f"   PyTorch: {torch.__version__}")

    model.to(device)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.7, patience=10, min_lr=1e-7
    )
    loss_fn = nn.MSELoss()
    
    # Initialize the GradScaler for PyTorch 2.5.1+ (no device_type needed in constructor)
    device_type = "cuda" if device.type == "cuda" else "cpu"
    scaler = GradScaler()
    
    best_val_loss, patience_counter, patience = float('inf'), 0, 20

    # Debug shapes
    print("🐛 Debugging model shapes with a sample batch...")
    sample_batch = next(iter(train_loader))
    debug_model_shapes(model, sample_batch[:2], device)

    try:
        for epoch in range(1, epochs + 1):
            current_lr = optimizer.param_groups[0]['lr']

            for split, loader in [("train", train_loader), ("val", val_loader)]:
                if split == "train":
                    model.train()
                else:
                    model.eval()

                epoch_loss = 0.0
                y_preds, y_trues = [], []  # Reset for each split

                # Force TQDM to console
                try:
                    console_file = open('/dev/tty', 'w')
                except:
                    console_file = sys.stdout

                progress_bar = tqdm(
                    loader,
                    desc=f"🚀 [{exp_name}] {split.capitalize()} Epoch {epoch}",
                    leave=False,
                    file=console_file,
                    dynamic_ncols=True
                )

                for rho_real, rho_imag, y in progress_bar:
                    rho_real = rho_real.to(device, non_blocking=True)
                    rho_imag = rho_imag.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)

                    # Use autocast with device_type for PyTorch 2.5.1+ 
                    # Mixed-precision computation on Tensor Cores for both training and evaluation
                    with autocast(device_type=device_type, dtype=torch.float16):
                        y_pred = model(rho_real, rho_imag)
                        loss = loss_fn(y_pred, y)

                    # Separate logic for train vs. val/test
                    if split == "train":
                        # Backward pass and optimization ONLY in training mode
                        optimizer.zero_grad()
                        # Scale the loss and call backward to create scaled gradients
                        scaler.scale(loss).backward()
                        # Unscale gradients before clipping to prevent inf/NaN
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        # Step the optimizer and update the scaler
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        # For validation, just collect predictions. Gradients are not needed.
                        y_preds.append(y_pred.detach().cpu())
                        y_trues.append(y.detach().cpu())
                    
                    epoch_loss += loss.item()
                    progress_bar.set_postfix({
                        'loss': f'{loss.item():.6f}',
                        'lr': f'{current_lr:.2e}'
                    })

                progress_bar.close()
                if console_file != sys.stdout:
                    console_file.close()
                
                avg_loss = epoch_loss / len(loader)

                # Log metrics and check for improvement only on the validation set
                if split == "val":
                    y_pred_cat = torch.cat(y_preds).numpy()
                    y_true_cat = torch.cat(y_trues).numpy()
                    metrics = compute_metrics(y_true_cat, y_pred_cat)
                    metrics.update({"epoch": epoch, "split": split, "lr": current_lr, "loss": avg_loss})
                    
                    print(f"📊 [{exp_name}] Epoch {epoch:2d}|{split:5s} | Loss: {avg_loss:.6f} | MAE: {metrics['mae']:.6f} | R²: {metrics['r2']:.4f}")
                    epoch_logger.log_epoch(metrics)

                    scheduler.step(avg_loss)
                    if avg_loss < best_val_loss:
                        best_val_loss, patience_counter = avg_loss, 0
                        torch.save(model.state_dict(), path_manager.model_path)
                        print(f"✅ [{exp_name}] New best model saved! Val Loss: {avg_loss:.6f}")
                    else:
                        patience_counter += 1
                else: # For the training split
                    print(f"💪 [{exp_name}] Epoch {epoch:2d}|{split:5s} | Loss: {avg_loss:.6f}")

            if patience_counter >= patience:
                print(f"\n🛑 [{exp_name}] Early stopping at epoch {epoch}")
                break

    finally:
        # Always cleanup GPU lock
        if gpu_lock_file:
            GPUManager.cleanup_gpu_lock(gpu_lock_file)

    # Final test evaluation
    print(f"\n🧪 [{exp_name}] Final Test Evaluation...")
    model.load_state_dict(torch.load(path_manager.model_path)) # Load best model
    test_metrics = evaluate(model, test_loader, device)
    test_metrics.update({"epoch": "final", "split": "test", "lr": current_lr})
    epoch_logger.log_epoch(test_metrics)

    print(f"📋 [{exp_name}] Final Test Results:")
    print(f"   MSE: {test_metrics['mse']:.6f}")
    print(f"   MAE: {test_metrics['mae']:.6f}")
    print(f"   R²:  {test_metrics['r2']:.4f}")
    print(f"   Spearman: {test_metrics['spearman']:.4f}")

    return test_metrics

def compute_metrics(y_true, y_pred, tolerances=[0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]):
    """Compute evaluation metrics"""
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    try:
        spearman = spearmanr(y_true, y_pred).correlation or 0.0
    except:
        spearman = 0.0
    try:
        pearson = pearsonr(y_true, y_pred)[0] or 0.0
    except:
        pearson = 0.0
    
    accs = {f'acc@{tol}': float((abs(y_pred - y_true) < tol).mean()) for tol in tolerances}
    return {"mse": mse, "mae": mae, "r2": r2, "spearman": spearman, "pearson": pearson, **accs}

def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj
    

def evaluate(model, loader, device):
    """Evaluate model on a dataset with PyTorch 2.5.1+ autocast"""
    model.eval()
    y_preds, y_trues = [], []
    
    device_type = "cuda" if device.type == "cuda" else "cpu"
    
    try:
        console_file = open('/dev/tty', 'w')
    except:
        console_file = sys.stdout
    
    with torch.no_grad():
        for rho_real, rho_imag, y in tqdm(loader, desc="🔍 Evaluating", file=console_file):
            rho_real, rho_imag, y = rho_real.to(device), rho_imag.to(device), y.to(device)
            
            # Use autocast for evaluation too
            with autocast(device_type=device_type, dtype=torch.float16):
                y_pred = model(rho_real, rho_imag)
            
            y_preds.append(y_pred.cpu())
            y_trues.append(y.cpu())
    
    if console_file != sys.stdout:
        console_file.close()
    
    y_pred_cat = torch.cat(y_preds).numpy()
    y_true_cat = torch.cat(y_trues).numpy()
    return compute_metrics(y_true_cat, y_pred_cat)

def debug_model_shapes(model, sample_data, device):
    """Debug model shapes with autocast"""
    model.eval()
    device_type = "cuda" if device.type == "cuda" else "cpu"
    
    print(f"\n🔍 DEBUG: Checking model shapes...")
    rho_real, rho_imag = sample_data
    rho_real, rho_imag = rho_real.to(device), rho_imag.to(device)
    print(f"   Input shapes: rho_real={rho_real.shape}, rho_imag={rho_imag.shape}")
    
    with torch.no_grad():
        with autocast(device_type=device_type, dtype=torch.float16):
            tokens = model.embedding(rho_real, rho_imag)
            print(f"   After embedding: {tokens.shape}")
            
            # Add CLS token if needed (same as in model forward)
            if model.use_cls_token:
                batch_size = tokens.size(0)
                cls_tokens = model.cls_token.expand(batch_size, -1, -1)
                tokens = torch.cat([cls_tokens, tokens], dim=1)
                print(f"   After adding CLS: {tokens.shape}")
            
            for i, layer in enumerate(model.encoder_layers):
                tokens = layer(tokens)
                print(f"   After encoder layer {i+1}: {tokens.shape}")
            
            if model.pooling_type == "cls":
                global_features = tokens[:, 0, :]
                print(f"   After CLS extraction: {global_features.shape}")
            else:
                global_features = model.physics_pooling(tokens)
                print(f"   After {model.pooling_type} pooling: {global_features.shape}")
            
            output = model.mlp_head(global_features)
            print(f"   Final output: {output.shape}")
    print("   ✅ All shapes look correct!")

def main():
    parser = argparse.ArgumentParser(description="Train single quantum magic prediction experiment with isolated outputs (PyTorch 2.5.1+ compatible)")
    
    # Experiment identification
    parser.add_argument("--exp_name", required=True, type=str, help="Experiment name (must be unique)")
    
    # Model configuration
    parser.add_argument("--num_qubits", default=2, type=int, help="Number of qubits")
    parser.add_argument("--d_model", type=int, default=64, help="Model dimension")
    parser.add_argument("--pooling_type", type=str, default="cls", 
                        choices=["cls", "mean", "attention", "structured"], 
                        help="Pooling type")
    parser.add_argument("--mlp_type", type=str, default="standard", 
                        choices=["standard", "physics_aware", "attention_enhanced", 
                               "mixture_of_experts", "asymmetric_ensemble"], 
                        help="MLP type")
    parser.add_argument("--use_physics_mask", 
                        type=lambda x: x.lower() == 'true', 
                        default=False, 
                        help="Use physics mask in transformer encoder (True/False)")
    parser.add_argument("--mask_threshold", type=float, default=1, 
                        help="Threshold for physics mask (only used if use_physics_mask=True)")
    parser.add_argument("--nhead", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--use_cls_token", 
                        type=lambda x: x.lower() == 'true', 
                        default=True, 
                        help="Use CLS token in transformer encoder (True/False)")
    
    # Training configuration
    parser.add_argument("--epochs", type=int, default=40, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size")

    # Data configuration
    parser.add_argument("--data_folder", default="/home/gwl/MixMLP/Decoded_Tokens", type=str, help="Data folder path")
    parser.add_argument("--labels_path", default="/home/gwl/MixMLP/Label", type=str, help="Labels file path OR folder containing label files")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Training ratio")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation ratio")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    
    # Device configuration
    parser.add_argument("--device", type=str, default="auto", 
                        help="Device to use (auto, cpu, cuda, cuda:0, etc.)")
    parser.add_argument("--force_gpu", type=int, default=None, 
                        help="Force specific GPU ID (overrides auto-selection)")
    
    # Output isolation
    parser.add_argument("--base_log_dir", type=str, default="experiments", 
                        help="Base directory for experiment logs")
    parser.add_argument("--base_model_dir", type=str, default="models", 
                        help="Base directory for model saves")
    
    # 🧪 NEW: Dummy test mode for local testing
    parser.add_argument("--dummy_test", action="store_true", 
                        help="Use dummy data for local testing (ignores data_folder/labels_path)")
    parser.add_argument("--dummy_samples", type=int, default=32,
                        help="Number of dummy samples to generate (for dummy_test mode)")
    
    args = parser.parse_args()
    
    print(f"🔧 PyTorch Version: {torch.__version__}")
    print(f"🔧 CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"🔧 CUDA Version: {torch.version.cuda}")
        print(f"🔧 GPU Count: {torch.cuda.device_count()}")
    
    # CREATE ISOLATED PATHS
    path_manager = ExperimentPathManager(
        exp_name=args.exp_name,
        base_log_dir=args.base_log_dir,
        base_model_dir=args.base_model_dir
    )
    
    # Enhanced device setup
    gpu_lock_file = None
    
    if args.force_gpu is not None:
        if torch.cuda.is_available() and args.force_gpu < torch.cuda.device_count():
            device = torch.device(f"cuda:{args.force_gpu}")
            gpu_lock_file = GPUManager.create_gpu_lock_file(args.force_gpu, args.exp_name)
            print(f"🔧 Forced GPU: {device}")
        else:
            print(f"❌ Requested GPU {args.force_gpu} not available, falling back to auto")
            device = GPUManager.select_best_gpu()
    elif args.device == "auto":
        device = GPUManager.select_best_gpu()
        if device.type == 'cuda':
            gpu_id = device.index
            gpu_lock_file = GPUManager.create_gpu_lock_file(gpu_id, args.exp_name)
    elif args.device == "cpu":
        device = torch.device("cpu")
        print("🔧 Using CPU (forced)")
    else:
        device = torch.device(args.device)
        if device.type == 'cuda':
            gpu_lock_file = GPUManager.create_gpu_lock_file(device.index, args.exp_name)
        print(f"🔧 Using specified device: {device}")
    
    print(f"🔧 Experiment: {args.exp_name}")
    print(f"🔧 Configuration:")
    print(f"   Qubits: {args.num_qubits}")
    print(f"   d_model: {args.d_model}")
    print(f"   Physics mask: {args.use_physics_mask}")
    print(f"   Pooling: {args.pooling_type}")
    print(f"   MLP: {args.mlp_type}")
    print(f"   Device: {device}")
    print(f"   Dummy test mode: {args.dummy_test}")
    
    # Load data - either real or dummy
    if args.dummy_test:
        print("🧪 DUMMY TEST MODE - Using synthetic data for local testing")
        matrix_dim = 2**args.num_qubits
        train_loader, val_loader, test_loader = create_dummy_data(
            batch_size=args.dummy_samples, 
            matrix_dim=matrix_dim
        )
        # Override epochs for quick testing
        args.epochs = min(args.epochs, 3)
        print(f"🧪 Reduced epochs to {args.epochs} for dummy testing")
    else:
        print("📥 Loading real data...")
        train_loader, val_loader, test_loader = create_dataloaders(
            data_folder=args.data_folder,
            labels_path=args.labels_path,
            batch_size=args.batch_size,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            num_workers=args.num_workers
        )
    
    # Create model
    print("🏗️ Creating model...")
    matrix_dim = 2**args.num_qubits
    model = CompleteQuantumMagicPredictor(
        matrix_dim=matrix_dim,
        n_qubits=args.num_qubits,
        d_model=args.d_model,
        pooling_type=args.pooling_type,
        mlp_type=args.mlp_type,
        use_physics_mask=args.use_physics_mask,
        mask_threshold=args.mask_threshold,
        nhead=args.nhead,
        use_cls_token=args.use_cls_token
    )
    
    # Prepare model parameters for logging and config
    model_params = {
        'exp_name': args.exp_name,
        'num_qubits': args.num_qubits,
        'd_model': args.d_model,
        'use_physics_mask': args.use_physics_mask,
        'pooling_type': args.pooling_type,
        'mlp_type': args.mlp_type,
        'matrix_dim': matrix_dim,
        'device': str(device),
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'lr': args.lr,
        'train_ratio': args.train_ratio,
        'val_ratio': args.val_ratio,
        'pytorch_version': torch.__version__,
        'mask_threshold': args.mask_threshold,
        'dummy_test': args.dummy_test,
        'nhead': args.nhead,
        'use_cls_token': args.use_cls_token
    }
    
    # Add data paths only if not using dummy data
    if not args.dummy_test:
        model_params.update({
            'data_folder': args.data_folder,
            'labels_path': args.labels_path
        })
    
    # Save experiment configuration
    path_manager.save_experiment_config(model_params)
    
    # Train model - ❌ REMOVED invalid mask_threshold parameter
    try:
        final_metrics = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            path_manager=path_manager,
            exp_name=args.exp_name,
            model_params=model_params,
            gpu_lock_file=gpu_lock_file
            # ✅ REMOVED: mask_threshold=args.mask_threshold (invalid parameter)
        )
        
        # Save final metrics
        final_results = {
            'model_parameters': model_params,
            'final_metrics': final_metrics,
            'model_path': str(path_manager.model_path),
            'experiment_paths': {
                'log_path': str(path_manager.log_path),
                'config_path': str(path_manager.config_path),
                'exp_dir': str(path_manager.exp_dir)
            }
        }
        final_results = convert_numpy_types(final_results)
        with open(path_manager.metrics_path, 'w') as f:
            json.dump(final_results, f, indent=2)
        
        print(f"\n✅ Experiment {args.exp_name} completed successfully!")
        print(f"📁 All outputs saved to: {path_manager.exp_dir}")
        print(f"📄 Training log: {path_manager.log_path}")
        print(f"📄 Final metrics: {path_manager.metrics_path}")
        print(f"🤖 Model weights: {path_manager.model_path}")
        print(f"⚙️ Configuration: {path_manager.config_path}")
        
        if args.dummy_test:
            print(f"\n🧪 DUMMY TEST COMPLETED!")
            print(f"   This was a local test run with synthetic data")
            print(f"   For real training, remove --dummy_test flag")
        else:
            # Generate visualization command only for real data
            viz_cmd = path_manager.get_visualization_command(args.data_folder, args.labels_path)
            print(f"\n🔍 To visualize this model, run:")
            print(viz_cmd)
        
    except Exception as e:
        print(f"❌ Experiment {args.exp_name} failed with error: {e}")
        import traceback
        traceback.print_exc()
        
        # Cleanup GPU lock on failure
        if gpu_lock_file:
            GPUManager.cleanup_gpu_lock(gpu_lock_file)
        
        sys.exit(1)

if __name__ == "__main__":
    main()