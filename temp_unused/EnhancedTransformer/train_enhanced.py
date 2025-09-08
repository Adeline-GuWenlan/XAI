#!/usr/bin/env python3
"""
Enhanced Training Script - extends existing train_single.py with minimal changes

Usage:
python train_enhanced.py --exp_name test_enhanced --num_qubits 3 --d_model 512 \
    --num_layers 8 --loss_type smoothl1 --epochs 50 --batch_size 256
"""

import sys
import os
sys.path.append('..')

import argparse
import torch
from enhanced_model import EnhancedQuantumPredictor

# Import most functions from existing training script
from quantum_dataset import create_dataloaders
from train_single import (
    ExperimentPathManager, setup_device
)
import pandas as pd
import numpy as np
import json
import datetime
from tqdm import tqdm
from scipy.stats import spearmanr

def parse_enhanced_args():
    """Extend base arguments with enhanced model parameters"""
    # Get base parser
    base_parser = argparse.ArgumentParser(add_help=False)
    
    # Copy arguments from base parser (manually add key ones)
    base_parser.add_argument('--exp_name', required=True, help='Experiment name')
    base_parser.add_argument('--num_qubits', type=int, default=3, help='Number of qubits')
    base_parser.add_argument('--d_model', type=int, default=512, help='Model dimension')
    base_parser.add_argument('--nhead', type=int, default=8, help='Number of attention heads')
    base_parser.add_argument('--pooling_type', default='cls', help='Pooling type')
    base_parser.add_argument('--mlp_type', default='mixture_of_experts', help='MLP type')
    base_parser.add_argument('--use_cls_token', action='store_true', default=True, help='Use CLS token')
    
    # Training parameters
    base_parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    base_parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    base_parser.add_argument('--batch_size', type=int, default=512, help='Batch size')
    
    # Data parameters
    base_parser.add_argument('--data_folder', default='/home/gwl/3q/MixMLP/Decoded_Tokens', help='Data folder')
    base_parser.add_argument('--labels_path', 
                           default='/home/gwl/3q/MixMLP/Label/magic_labels_for_input_for_3_qubits_mixed_10000_datapoints.npy',
                           help='Labels path')
    base_parser.add_argument('--train_ratio', type=float, default=0.8, help='Train ratio')
    base_parser.add_argument('--val_ratio', type=float, default=0.1, help='Validation ratio')
    base_parser.add_argument('--num_workers', type=int, default=4, help='Data workers')
    
    # Output directories
    base_parser.add_argument('--base_log_dir', default='experiments', help='Log directory')
    base_parser.add_argument('--base_model_dir', default='models', help='Model directory')
    
    # Device
    base_parser.add_argument('--device', default='auto', help='Device')
    
    # Debug
    base_parser.add_argument('--dummy_test', action='store_true', help='Use dummy data')
    base_parser.add_argument('--dummy_samples', type=int, default=32, help='Dummy samples')
    
    # Enhanced model parameters
    base_parser.add_argument('--num_layers', type=int, default=6, 
                           help='Number of transformer encoder layers')
    base_parser.add_argument('--loss_type', choices=['mse', 'smoothl1'], default='mse',
                           help='Loss function type')
    
    return base_parser.parse_args()

def main():
    args = parse_enhanced_args()
    
    # Setup experiment
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"{args.exp_name}_{timestamp}"
    
    path_manager = ExperimentPathManager(
        base_log_dir=args.base_log_dir,
        base_model_dir=args.base_model_dir,
        experiment_name=experiment_name
    )
    path_manager.setup_directories()
    
    # Device setup
    device = setup_device(args.device, experiment_name)
    
    # Create enhanced model
    model = EnhancedQuantumPredictor(
        matrix_dim=None,  # Auto-calculated
        n_qubits=args.num_qubits,
        d_model=args.d_model,
        num_layers=args.num_layers,  # Enhanced parameter
        pooling_type=args.pooling_type,
        mlp_type=args.mlp_type,
        nhead=args.nhead,
        use_cls_token=args.use_cls_token,
        loss_type=args.loss_type,  # Enhanced parameter
        dropout=0.1
    )
    model.to(device)
    
    print(f"📊 Enhanced Model - Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Save model config
    config = {
        'model_config': model.get_config(),
        'training_args': vars(args)
    }
    
    config_path = path_manager.log_dir / f"{experiment_name}_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Create data loaders
    if args.dummy_test:
        print("🔧 Using dummy data for testing")
        from torch.utils.data import TensorDataset, DataLoader
        
        matrix_dim = 2 ** args.num_qubits
        dummy_real = torch.randn(args.dummy_samples, matrix_dim, matrix_dim)
        dummy_imag = torch.randn(args.dummy_samples, matrix_dim, matrix_dim)
        dummy_labels = torch.randn(args.dummy_samples)
        
        dataset = TensorDataset(dummy_real, dummy_imag, dummy_labels)
        train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        test_loader = val_loader
    else:
        train_loader, val_loader, test_loader = create_dataloaders(
            data_folder=args.data_folder,
            labels_path=args.labels_path,
            batch_size=args.batch_size,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            num_workers=args.num_workers
        )
    
    print(f"📊 Data: Train={len(train_loader.dataset)}, Val={len(val_loader.dataset)}, Test={len(test_loader.dataset)}")
    
    # Training setup
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-5,
        betas=(0.9, 0.95)
    )
    
    total_steps = args.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=total_steps,
        pct_start=0.1,
        anneal_strategy='cos'
    )
    
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    
    # Training tracking
    training_history = []
    best_val_loss = float('inf')
    patience_counter = 0
    early_stopping_patience = 15
    
    print(f"🚀 Training Enhanced Model: {args.num_layers} layers, {args.loss_type} loss")
    print(f"🔧 FFN dimension: {4 * args.d_model} (4x d_model)")
    
    # Training loop
    for epoch in range(1, args.epochs + 1):
        # Training phase
        model.train()
        train_loss = 0.0
        train_predictions = []
        train_targets = []
        
        progress_bar = tqdm(train_loader, desc=f"🚀 [{experiment_name}] Epoch {epoch:2d}", 
                           leave=False, dynamic_ncols=True)
        
        for rho_real, rho_imag, labels in progress_bar:
            rho_real, rho_imag, labels = rho_real.to(device), rho_imag.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            if scaler:
                with torch.cuda.amp.autocast():
                    predictions = model(rho_real, rho_imag).squeeze()
                    loss = model.compute_loss(predictions, labels)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                predictions = model(rho_real, rho_imag).squeeze()
                loss = model.compute_loss(predictions, labels)
                loss.backward()
                optimizer.step()
            
            scheduler.step()
            
            train_loss += loss.item()
            train_predictions.extend(predictions.detach().cpu().numpy())
            train_targets.extend(labels.detach().cpu().numpy())
            
            progress_bar.set_postfix({
                'loss': f'{loss.item():.6f}',
                'lr': f'{scheduler.get_last_lr()[0]:.2e}'
            })
        
        avg_train_loss = train_loss / len(train_loader)
        print(f"💪 [{experiment_name}] Epoch {epoch:2d}|train | Loss: {avg_train_loss:.6f}")
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_predictions = []
        val_targets = []
        
        with torch.no_grad():
            for rho_real, rho_imag, labels in val_loader:
                rho_real, rho_imag, labels = rho_real.to(device), rho_imag.to(device), labels.to(device)
                
                if scaler:
                    with torch.cuda.amp.autocast():
                        predictions = model(rho_real, rho_imag).squeeze()
                        loss = model.compute_loss(predictions, labels)
                else:
                    predictions = model(rho_real, rho_imag).squeeze()
                    loss = model.compute_loss(predictions, labels)
                
                val_loss += loss.item()
                val_predictions.extend(predictions.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader)
        
        # Calculate metrics
        val_predictions = np.array(val_predictions)
        val_targets = np.array(val_targets)
        val_mae = np.mean(np.abs(val_predictions - val_targets))
        
        ss_res = np.sum((val_targets - val_predictions) ** 2)
        ss_tot = np.sum((val_targets - np.mean(val_targets)) ** 2)
        val_r2 = 1 - (ss_res / (ss_tot + 1e-8))
        
        print(f"📊 [{experiment_name}] Epoch {epoch:2d}|val   | Loss: {avg_val_loss:.6f} | MAE: {val_mae:.6f} | R²: {val_r2:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), path_manager.model_path)
            print(f"✅ [{experiment_name}] New best model saved! Val Loss: {avg_val_loss:.6f}")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= early_stopping_patience:
            print(f"🛑 Early stopping after {epoch} epochs")
            break
        
        # Record training history
        epoch_data = {
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'val_mae': val_mae,
            'val_r2': val_r2,
            'lr': scheduler.get_last_lr()[0]
        }
        training_history.append(epoch_data)
        
        # Save training log
        history_df = pd.DataFrame(training_history)
        history_df.to_csv(path_manager.log_path, index=False)
    
    # Final test evaluation
    print(f"🧪 [{experiment_name}] Final Test Evaluation...")
    model.load_state_dict(torch.load(path_manager.model_path))
    model.eval()
    
    test_predictions = []
    test_targets = []
    
    with torch.no_grad():
        for rho_real, rho_imag, labels in tqdm(test_loader, desc="🔍 Evaluating"):
            rho_real, rho_imag, labels = rho_real.to(device), rho_imag.to(device), labels.to(device)
            
            if scaler:
                with torch.cuda.amp.autocast():
                    predictions = model(rho_real, rho_imag).squeeze()
            else:
                predictions = model(rho_real, rho_imag).squeeze()
            
            test_predictions.extend(predictions.cpu().numpy())
            test_targets.extend(labels.cpu().numpy())
    
    # Final metrics
    test_predictions = np.array(test_predictions)
    test_targets = np.array(test_targets)
    
    test_mse = np.mean((test_predictions - test_targets) ** 2)
    test_mae = np.mean(np.abs(test_predictions - test_targets))
    
    ss_res = np.sum((test_targets - test_predictions) ** 2)
    ss_tot = np.sum((test_targets - np.mean(test_targets)) ** 2)
    test_r2 = 1 - (ss_res / (ss_tot + 1e-8))
    
    test_spearman, _ = spearmanr(test_predictions, test_targets)
    
    print(f"📋 [{experiment_name}] Final Test Results:")
    print(f"   MSE: {test_mse:.6f}")
    print(f"   MAE: {test_mae:.6f}")
    print(f"   R²:  {test_r2:.4f}")
    print(f"   Spearman: {test_spearman:.4f}")
    
    # Save final metrics
    final_metrics = {
        'test_mse': test_mse,
        'test_mae': test_mae,
        'test_r2': test_r2,
        'test_spearman': test_spearman,
        'best_val_loss': best_val_loss,
        'model_config': model.get_config()
    }
    
    metrics_path = path_manager.log_dir / f"{experiment_name}_final_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(final_metrics, f, indent=2)
    
    print(f"✅ Experiment {experiment_name} completed successfully!")
    print(f"📁 All outputs saved to: {path_manager.log_dir}")
    print(f"📄 Training log: {path_manager.log_path}")
    print(f"📄 Final metrics: {metrics_path}")
    print(f"🤖 Model weights: {path_manager.model_path}")
    print(f"⚙️ Configuration: {config_path}")

if __name__ == '__main__':
    main()