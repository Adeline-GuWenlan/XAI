#!/usr/bin/env python3
# finetune_robust.py - Robust fine-tuning with automatic batch size optimization
import torch
import torch.backends.cudnn as cudnn
import argparse
import sys
import os
from updated_training_model import CompleteQuantumMagicPredictor
from quantum_dataset import create_dataloaders
from train_single import train_model, ExperimentPathManager

def main():
    parser = argparse.ArgumentParser(description="Robust fine-tuning with automatic optimization")
    parser.add_argument("--pretrained_model", required=True, type=str, help="Path to pretrained model")
    parser.add_argument("--mlp_type", required=True)
    parser.add_argument("--exp_name", default="robust_finetune", type=str, help="Experiment name")
    parser.add_argument("--epochs", default=15, type=int, help="Number of epochs")
    parser.add_argument("--lr", default=1e-4, type=float, help="Learning rate")
    parser.add_argument("--use_physics_mask", 
                        type=lambda x: x.lower() == 'true', 
                        default=False, 
                        help="Use physics mask in transformer encoder (True/False)")
    parser.add_argument("--mask_threshold", type=float, default=1, 
                        help="Threshold for physics mask (only used if use_physics_mask=True)")
    

    parser.add_argument("--gpu_id", default=0, type=int, help="GPU ID to use")
    
    args = parser.parse_args()
    
    # 🚀 PERFORMANCE OPTIMIZATIONS
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True
    
    print(f"🔧 Robust Fine-tuning Setup")
    print(f"🔧 GPU: cuda:{args.gpu_id}")
    print(f"🔧 Pretrained: {args.pretrained_model}")
    print(f"🔧 Max epochs: {args.epochs}")
    print(f"🔧 Learning rate: {args.lr}")
    
    # Setup device
    device = torch.device(f"cuda:{args.gpu_id}")
    
    # Load model
    print("🏗️  Creating model...")
    model = CompleteQuantumMagicPredictor(
        matrix_dim=4, n_qubits=2, d_model=64,
        pooling_type="cls", mlp_type=args.mlp_type,
        use_physics_mask=args.use_physics_mask,
        mask_threshold=args.mask_threshold
    )
    
    print(f"🔄 Loading pretrained weights: {args.pretrained_model}")
    model.load_state_dict(torch.load(args.pretrained_model, map_location='cpu'))
    
    # Detect optimal batch size
    # Create optimized dataloaders
    print("📥 Creating optimized dataloaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_folder="/home/gwl/DURF/Transformer/dataset/Decoded_Tokens",
        labels_path="/home/gwl/DURF/Transformer/dataset/Label/magic_labels_for_input_for_2_qubits_mixed_1_200000_datapoints.npy",
        batch_size=512,
        num_workers=4,  # Increased workers
        validate=False   # Skip validation for speed
    )
    
    # Setup experiment tracking
    path_manager = ExperimentPathManager(exp_name=args.exp_name)
    
    # Model parameters for logging
    model_params = {
        'exp_name': args.exp_name,
        'num_qubits': 2, 'd_model': 64, 'use_physics_mask': args.use_physics_mask,
        'pooling_type': 'cls', 'mlp_type': args.mlp_type, 'matrix_dim': 4,
        'device': str(device), 'batch_size': 512, 
        'epochs': args.epochs, 'lr': args.lr,
        'optimizations': 'cuDNN_benchmark + TF32 + auto_batch_size + robust_tmux',
        'pretrained_from': args.pretrained_model,
        'pytorch_version': torch.__version__,
        'mask':args.mask_threshold
    }
    
    path_manager.save_experiment_config(model_params)
    
    print(f"🚀 Starting robust fine-tuning...")
    print(f"📊 Logs: {path_manager.log_path}")
    print(f"🤖 Model will be saved to: {path_manager.model_path}")
    
    try:
        final_metrics = train_model(
            model=model, train_loader=train_loader, val_loader=val_loader,
            test_loader=test_loader, device=device,
            epochs=args.epochs, lr=args.lr, path_manager=path_manager,
            exp_name=args.exp_name, model_params=model_params
        )
        
        print(f"✅ Robust fine-tuning completed successfully!")
        print(f"📈 Final test metrics: {final_metrics}")
        
    except Exception as e:
        print(f"❌ Fine-tuning failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
