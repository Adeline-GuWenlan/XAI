#!/usr/bin/env python3
"""
Simple training script for Enhanced Transformer - standalone implementation
"""

import sys
import os
sys.path.append('..')

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from scipy.stats import spearmanr

from enhanced_model import EnhancedQuantumPredictor
from quantum_dataset import create_dataloaders

def parse_args():
    parser = argparse.ArgumentParser(description='Enhanced Quantum Transformer Training')
    
    # Experiment
    parser.add_argument('--exp_name', required=True, help='Experiment name')
    
    # Model
    parser.add_argument('--num_qubits', type=int, default=3, help='Number of qubits')
    parser.add_argument('--d_model', type=int, default=512, help='Model dimension')
    parser.add_argument('--num_layers', type=int, default=6, help='Number of transformer layers')
    parser.add_argument('--nhead', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--pooling_type', default='cls', help='Pooling type')
    parser.add_argument('--mlp_type', default='mixture_of_experts', help='MLP type')
    parser.add_argument('--loss_type', choices=['mse', 'smoothl1'], default='mse', help='Loss function')
    
    # Training
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size')
    
    # Data
    parser.add_argument('--data_folder', default='../Decoded_Tokens', help='Data folder')
    parser.add_argument('--labels_path', 
                       default='../Label/magic_labels_for_input_for_3_qubits_mixed_10000_datapoints.npy',
                       help='Labels path')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Train ratio')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='Validation ratio')
    parser.add_argument('--num_workers', type=int, default=4, help='Data workers')
    
    # Output
    parser.add_argument('--output_dir', default='./experiments', help='Output directory')
    
    # Device
    parser.add_argument('--device', default='auto', help='Device')
    
    # Debug
    parser.add_argument('--dummy_test', action='store_true', help='Use dummy data')
    parser.add_argument('--dummy_samples', type=int, default=32, help='Dummy samples')
    
    return parser.parse_args()

def setup_device(device_arg):
    if device_arg == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_arg)
    
    print(f"🔧 Using device: {device}")
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name()}")
    
    return device

def create_experiment_dir(output_dir, exp_name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(output_dir) / f"{exp_name}_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir, f"{exp_name}_{timestamp}"

def compute_metrics(predictions, targets):
    pred = np.array(predictions)
    target = np.array(targets)
    
    mse = float(np.mean((pred - target) ** 2))
    mae = float(np.mean(np.abs(pred - target)))
    
    ss_res = np.sum((target - pred) ** 2)
    ss_tot = np.sum((target - np.mean(target)) ** 2)
    r2 = float(1 - (ss_res / (ss_tot + 1e-8)))
    
    spearman_corr, _ = spearmanr(pred.flatten(), target.flatten())
    
    return {
        'mse': mse,
        'mae': mae,
        'r2': r2,
        'spearman': float(spearman_corr)
    }

def main():
    args = parse_args()
    
    # Setup
    device = setup_device(args.device)
    exp_dir, experiment_name = create_experiment_dir(args.output_dir, args.exp_name)
    
    print(f"📁 Experiment: {experiment_name}")
    print(f"📁 Directory: {exp_dir}")
    
    # Create model
    model = EnhancedQuantumPredictor(
        n_qubits=args.num_qubits,
        d_model=args.d_model,
        num_layers=args.num_layers,
        pooling_type=args.pooling_type,
        mlp_type=args.mlp_type,
        nhead=args.nhead,
        use_cls_token=True,
        loss_type=args.loss_type
    )
    model.to(device)
    
    print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create data loaders
    if args.dummy_test:
        print("🔧 Using dummy data for testing")
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
    
    print(f"📊 Data - Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")
    
    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr,
        total_steps=args.epochs * len(train_loader),
        pct_start=0.1, anneal_strategy='cos'
    )
    
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    
    # Training tracking
    training_history = []
    best_val_loss = float('inf')
    patience_counter = 0
    early_stopping_patience = 15
    
    print(f"🚀 Training: {args.num_layers} layers, {args.loss_type} loss, FFN: {4 * args.d_model}")
    
    # Training loop
    for epoch in range(1, args.epochs + 1):
        # Training
        model.train()
        train_loss = 0.0
        train_preds, train_tgts = [], []
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch:2d}", leave=False)
        
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
            train_preds.extend(predictions.detach().cpu().numpy())
            train_tgts.extend(labels.detach().cpu().numpy())
            
            progress_bar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_preds, val_tgts = [], []
        
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
                val_preds.extend(predictions.cpu().numpy())
                val_tgts.extend(labels.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader)
        val_metrics = compute_metrics(val_preds, val_tgts)
        
        print(f"Epoch {epoch:3d} - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, R²: {val_metrics['r2']:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': model.get_config(),
                'epoch': epoch,
                'val_loss': avg_val_loss
            }, exp_dir / 'best_model.pth')
            print(f"✅ New best model saved! Val Loss: {avg_val_loss:.6f}")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= early_stopping_patience:
            print(f"🛑 Early stopping after {epoch} epochs")
            break
        
        # Record history
        training_history.append({
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'val_r2': val_metrics['r2'],
            'val_mae': val_metrics['mae'],
            'lr': scheduler.get_last_lr()[0]
        })
        
        # Save training history
        pd.DataFrame(training_history).to_csv(exp_dir / 'training_log.csv', index=False)
    
    # Final test evaluation
    print(f"🧪 Final Test Evaluation...")
    checkpoint = torch.load(exp_dir / 'best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    test_preds, test_tgts = [], []
    
    with torch.no_grad():
        for rho_real, rho_imag, labels in tqdm(test_loader, desc="Testing"):
            rho_real, rho_imag, labels = rho_real.to(device), rho_imag.to(device), labels.to(device)
            
            if scaler:
                with torch.cuda.amp.autocast():
                    predictions = model(rho_real, rho_imag).squeeze()
            else:
                predictions = model(rho_real, rho_imag).squeeze()
            
            test_preds.extend(predictions.cpu().numpy())
            test_tgts.extend(labels.cpu().numpy())
    
    test_metrics = compute_metrics(test_preds, test_tgts)
    
    print(f"📋 Final Test Results:")
    print(f"   MSE: {test_metrics['mse']:.6f}")
    print(f"   MAE: {test_metrics['mae']:.6f}")
    print(f"   R²:  {test_metrics['r2']:.4f}")
    print(f"   Spearman: {test_metrics['spearman']:.4f}")
    
    # Save final results
    final_results = {
        'test_metrics': test_metrics,
        'best_val_loss': best_val_loss,
        'model_config': model.get_config(),
        'training_args': vars(args)
    }
    
    with open(exp_dir / 'final_results.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"✅ Experiment completed successfully!")
    print(f"📁 Results saved to: {exp_dir}")

if __name__ == '__main__':
    main()