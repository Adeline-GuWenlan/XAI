#!/usr/bin/env python3
"""
XAI Analysis Script for Quantum Magic Prediction Model
Supports both attention weight visualization and saliency analysis
"""

import argparse
import json
import os
import sys
import torch
import numpy as np
from pathlib import Path

# Import analysis modules
import sys
sys.path.append('..')
sys.path.append('../Saliency')
sys.path.append('../Atten')
sys.path.append('../Model_Archi')

from saliency_analyzer import SaliencyAnalyzer
from saliency_visualizer import SaliencyVisualizer
from attention_analyzer import AttentionAnalyzer
from attention_visualizer import AttentionVisualizer

def load_config(config_path):
    """Load model configuration from JSON"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def create_compatible_model(config, device):
    """
    Create model architecture that matches the saved weights exactly
    """
    print("Creating model architecture to match saved weights...")
    
    # Import with error handling
    try:
        sys.path.append('../Model_Archi')
        from updated_training_model import CompleteQuantumMagicPredictor
    except ImportError as e:
        print(f"Error importing model: {e}")
        print("Make sure all required modules are in the current directory")
        raise
    
    # Create model with exact config parameters
    model = CompleteQuantumMagicPredictor(
        matrix_dim=config['matrix_dim'],
        n_qubits=config['num_qubits'], 
        d_model=config['d_model'],
        pooling_type=config['pooling_type'],
        mlp_type=config['mlp_type'],
        use_physics_mask=config.get('use_physics_mask', False),  # Default to False if not specified
        mask_threshold=config.get('mask_threshold', 1),
        nhead=config['nhead'],
        use_cls_token=config.get('use_cls_token', True)
    )
    
    model.to(device)
    return model

def load_model_weights(model, model_path, device, strict=False):
    """
    Load model weights with compatibility checking
    """
    print(f"Loading model weights from: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load saved state
    saved_state = torch.load(model_path, map_location=device, weights_only=True)
    model_state = model.state_dict()
    
    print("Checking weight compatibility...")
    
    # Check for missing and unexpected keys
    saved_keys = set(saved_state.keys())
    model_keys = set(model_state.keys())
    
    missing_keys = model_keys - saved_keys
    unexpected_keys = saved_keys - model_keys
    
    if missing_keys:
        print(f"⚠️  Missing keys in saved model: {len(missing_keys)}")
        for key in sorted(missing_keys)[:5]:  # Show first 5
            print(f"    {key}")
        if len(missing_keys) > 5:
            print(f"    ... and {len(missing_keys) - 5} more")
    
    if unexpected_keys:
        print(f"⚠️  Unexpected keys in saved model: {len(unexpected_keys)}")
        for key in sorted(unexpected_keys)[:5]:  # Show first 5
            print(f"    {key}")
        if len(unexpected_keys) > 5:
            print(f"    ... and {len(unexpected_keys) - 5} more")
    
    if not missing_keys and not unexpected_keys:
        print("✓ Perfect weight compatibility!")
    
    # Load weights
    try:
        if strict and (missing_keys or unexpected_keys):
            raise RuntimeError("Strict mode enabled but weights don't match exactly")
        
        model.load_state_dict(saved_state, strict=strict)
        print("✓ Weights loaded successfully")
        
    except RuntimeError as e:
        if strict:
            raise e
        else:
            print(f"⚠️  Loading with strict=False due to: {e}")
            # Try loading only matching keys
            matching_state = {k: v for k, v in saved_state.items() if k in model_state}
            model.load_state_dict(matching_state, strict=False)
            print(f"✓ Loaded {len(matching_state)} matching parameters")
    
    model.eval()
    return model

def test_model(model, device, n_qubits):
    """Test model functionality"""
    print("Testing model functionality...")
    
    matrix_dim = 2 ** n_qubits
    batch_size = 2
    
    # Create test inputs
    rho_real = torch.randn(batch_size, matrix_dim, matrix_dim, device=device) * 0.1
    rho_imag = torch.randn(batch_size, matrix_dim, matrix_dim, device=device) * 0.1
    
    try:
        # Test normal forward pass
        with torch.no_grad():
            output = model(rho_real, rho_imag)
        print(f"✓ Normal forward pass: output shape {output.shape}")
        
        # Test attention extraction
        with torch.no_grad():
            output_attn, attention_weights = model(rho_real, rho_imag, return_attention=True)
        print(f"✓ Attention extraction: {len(attention_weights)} layers")
        print(f"  Attention shape per layer: {attention_weights[0].shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_dummy_dataloader(config, device, batch_size=32, num_batches=5):
    """Create dummy dataloader for testing when real data is not available"""
    print("Creating dummy dataloader for testing...")
    
    matrix_dim = config['matrix_dim']
    n_qubits = config['num_qubits']
    
    class DummyDataset:
        def __init__(self, num_samples):
            self.num_samples = num_samples
            
        def __len__(self):
            return self.num_samples
            
        def __getitem__(self, idx):
            # Create random density matrices
            rho_real = torch.randn(matrix_dim, matrix_dim) * 0.1
            rho_imag = torch.randn(matrix_dim, matrix_dim) * 0.1
            
            # Make them Hermitian
            rho_real = (rho_real + rho_real.T) / 2
            rho_imag = (rho_imag - rho_imag.T) / 2
            
            # Dummy magic value
            magic = torch.rand(1) * 0.8 + 0.1  # Between 0.1 and 0.9
            
            return rho_real, rho_imag, magic
    
    from torch.utils.data import DataLoader
    dataset = DummyDataset(batch_size * num_batches)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    return dataloader

def try_load_real_data(data_folder, labels_path, batch_size=64):
    """Try to load real data if available"""
    try:
        sys.path.append('../Model_Archi')
        from quantum_dataset import create_dataloaders
        
        if not data_folder or not labels_path:
            print("⚠️  Data paths not specified")
            return None
            
        if not os.path.exists(data_folder) or not os.path.exists(labels_path):
            print(f"⚠️  Data files not found:")
            print(f"    Data folder: {data_folder} (exists: {os.path.exists(data_folder)})")
            print(f"    Labels path: {labels_path} (exists: {os.path.exists(labels_path)})")
            return None
            
        print(f"Loading real data from:")
        print(f"  Data folder: {data_folder}")
        print(f"  Labels path: {labels_path}")
        
        # Import dataset class directly to avoid DataLoader issues  
        from quantum_dataset import QuantumDataset
        from torch.utils.data import DataLoader, random_split
        
        # Load dataset
        dataset = QuantumDataset(data_folder, labels_path)
        total_size = len(dataset)
        print(f"  Total dataset size: {total_size}")
        
        # Split dataset (use only test split for analysis)
        train_ratio, val_ratio = 0.8, 0.1
        test_ratio = 1.0 - train_ratio - val_ratio
        
        train_size = int(train_ratio * total_size)
        val_size = int(val_ratio * total_size)
        test_size = total_size - train_size - val_size
        
        _, _, test_set = random_split(dataset, [train_size, val_size, test_size])
        
        # Create DataLoader with safe parameters
        test_loader = DataLoader(
            test_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,  # Disable for CPU
            persistent_workers=False,
            prefetch_factor=None  # Must be None when num_workers=0
        )
        
        print(f"  Test set size: {len(test_set)} samples")
        
        print("✓ Real data loaded successfully")
        return test_loader
        
    except Exception as e:
        print(f"⚠️  Could not load real data: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    parser = argparse.ArgumentParser(description='XAI Analysis for Quantum Magic Prediction')
    
    # Model and data
    parser.add_argument('--model_path', type=str, default='../CONFIGs/best_128_CLS_attention_enhanced_20250810_214428.pth',
                       help='Path to model weights')
    parser.add_argument('--config_path', type=str, 
                       default='../CONFIGs/128_CLS_attention_enhanced_config.json',
                       help='Path to model config JSON')
    parser.add_argument('--data_folder', type=str, default='../Rawdata/DecodedTokens',
                       help='Path to data folder containing _real.npy and _imag.npy files')
    parser.add_argument('--labels_path', type=str, default='../Rawdata/label/magic_labels_for_input_for_2_qubits_mixed_2_200000_datapoints.npy',
                       help='Path to labels .npy file')
    parser.add_argument('--output_dir', type=str, default='xai_analysis_results',
                       help='Output directory for results')
    
    # Analysis options
    parser.add_argument('--analysis_types', nargs='+', 
                       choices=['attention', 'saliency', 'both'],
                       default=['both'],
                       help='Types of analysis to run')
    parser.add_argument('--max_samples', type=int, default=1024,
                       help='Maximum samples to analyze')
    parser.add_argument('--individual_samples', type=int, default=10,
                       help='Number of individual samples for detailed analysis')
    
    # Data options
    parser.add_argument('--use_dummy_data', action='store_true',
                       help='Force use of dummy data instead of real data')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for analysis')
    
    # Model loading options
    parser.add_argument('--strict_loading', action='store_true',
                       help='Use strict weight loading (fail if weights don\'t match exactly)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("XAI ANALYSIS FOR QUANTUM MAGIC PREDICTION")
    print("="*80)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load configuration
    print(f"\nLoading configuration from: {args.config_path}")
    config = load_config(args.config_path)
    print("Configuration:")
    for key, value in config.items():
        if key not in ['data_folder', 'labels_path']:  # Skip potentially long paths
            print(f"  {key}: {value}")
    
    # Create model
    print(f"\nCreating model architecture...")
    model = create_compatible_model(config, device)
    
    # Load weights
    print(f"\nLoading model weights...")
    model = load_model_weights(model, args.model_path, device, args.strict_loading)
    
    # Test model
    print(f"\nTesting model...")
    if not test_model(model, device, config['num_qubits']):
        print("Model test failed. Exiting.")
        return 1
    
    # Prepare data
    print(f"\nPreparing data...")
    if args.use_dummy_data:
        dataloader = create_dummy_dataloader(config, device, args.batch_size)
        print("Using dummy data for analysis")
    else:
        dataloader = try_load_real_data(args.data_folder, args.labels_path, args.batch_size)
        if dataloader is None:
            print("Falling back to dummy data")
            dataloader = create_dummy_dataloader(config, device, args.batch_size)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Generate quantum labels for visualization
    quantum_labels = []
    for i in range(2**config['num_qubits']):
        for j in range(2**config['num_qubits']):
            i_binary = format(i, f'0{config["num_qubits"]}b')
            j_binary = format(j, f'0{config["num_qubits"]}b')
            quantum_labels.append(f'⟨{i_binary}|ρ|{j_binary}⟩')
    
    results = {}
    
    # Run attention analysis
    if 'attention' in args.analysis_types or 'both' in args.analysis_types:
        print(f"\n{'='*50}")
        print("ATTENTION WEIGHT ANALYSIS")
        print(f"{'='*50}")
        
        attention_analyzer = AttentionAnalyzer(model, device)
        attention_visualizer = AttentionVisualizer(
            str(output_dir / "attention_analysis"),
            figsize_scale=1.0
        )
        
        print("Running attention analysis...")
        attention_results = attention_analyzer.analyze_dataset(
            dataloader, 
            max_samples=args.max_samples
        )
        
        print("Creating attention visualizations...")
        attention_visualizer.create_attention_dashboard(attention_results, quantum_labels)
        
        results['attention'] = attention_results
        print(f"✓ Attention analysis complete! Results in {output_dir}/attention_analysis/")
    
    # Run saliency analysis  
    if 'saliency' in args.analysis_types or 'both' in args.analysis_types:
        print(f"\n{'='*50}")
        print("SALIENCY ANALYSIS")
        print(f"{'='*50}")
        
        saliency_config = {'ig_steps': 50}
        saliency_analyzer = SaliencyAnalyzer(model, device, saliency_config)
        saliency_visualizer = SaliencyVisualizer(str(output_dir / "saliency_analysis"))
        
        methods = ['gradient', 'integrated_gradients']
        print(f"Running saliency analysis with methods: {methods}")
        
        saliency_results = saliency_analyzer.analyze_dataset(
            dataloader,
            methods=methods,
            max_samples=args.max_samples
        )
        
        print("Creating saliency visualizations...")
        saliency_visualizer.plot_aggregated_patterns(saliency_results)
        
        # Individual sample analysis
        n_individual = min(args.individual_samples, len(saliency_results[methods[0]]['sample_results']))
        for i in range(n_individual):
            saliency_visualizer.compare_methods(saliency_results, sample_idx=i, quantum_labels=quantum_labels)
        
        results['saliency'] = saliency_results
        print(f"✓ Saliency analysis complete! Results in {output_dir}/saliency_analysis/")
    
    # Final summary
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE!")
    print(f"{'='*80}")
    print(f"Results saved to: {output_dir}")
    print(f"Model: {config['exp_name']}")
    print(f"Analyzed {args.max_samples} samples")
    
    if 'attention' in results:
        attn_stats = results['attention']['summary_stats']
        print(f"\nAttention Analysis:")
        print(f"  - Total samples: {attn_stats['total_samples']}")
        print(f"  - Avg prediction error: {attn_stats['avg_prediction_error']:.4f}")
    
    if 'saliency' in results:
        print(f"\nSaliency Analysis:")
        for method in methods:
            sample_count = len(results['saliency'][method]['sample_results'])
            errors = [s['prediction_error'] for s in results['saliency'][method]['sample_results']]
            print(f"  - {method}: {sample_count} samples, avg error: {np.mean(errors):.4f}")
    
    print(f"\n🎯 Most valuable insights:")
    print(f"   1. Check attention_analysis/ for transformer behavior patterns")
    print(f"   2. Check saliency_analysis/ for input importance maps")
    print(f"   3. Focus on CLS attention evolution - shows learning progression")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())