#!/usr/bin/env python3
"""
Run comprehensive MoE analysis on the quantum magic prediction model
"""
import os
import sys
import json
import torch
import numpy as np
from pathlib import Path

# Add paths
sys.path.append('/Users/guwenlan/Desktop/XAI')
sys.path.append('/Users/guwenlan/Desktop/XAI/Model_Archi')
sys.path.append('/Users/guwenlan/Desktop/XAI/MOE_ANA')

from Model_Archi.updated_training_model import CompleteQuantumMagicPredictor
from Model_Archi.quantum_dataset import QuantumDataset
from moe_attention_analyzer import MoEAttentionAnalyzer
from moe_visualizer import MoEVisualizer

def load_model_and_config(model_path: str, config_path: str):
    """Load the trained MoE model and its configuration"""
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Initialize model
    model = CompleteQuantumMagicPredictor(
        matrix_dim=config['matrix_dim'],
        n_qubits=config['num_qubits'],
        d_model=config['d_model'],
        pooling_type=config['pooling_type'],
        mlp_type=config['mlp_type'],
        use_physics_mask=config['use_physics_mask'],
        mask_threshold=config['mask_threshold'],
        nhead=config['nhead'],
        use_cls_token=config.get('use_cls_token', True)
    )
    
    # Load trained weights
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    print(f"Model loaded successfully!")
    print(f"Model type: {config['mlp_type']}")
    print(f"Device: {device}")
    
    return model, config, device

def create_dataloader(config, batch_size=32, max_samples=500):
    """Create dataloader for analysis using correct 50K labels"""
    # Dataset parameters
    data_folder = Path("/Users/guwenlan/Desktop/XAI") / config['data_folder']
    # Use the correct matching 50K label file instead of the 200K one
    labels_path = Path("/Users/guwenlan/Desktop/XAI/Rawdata/label/magic_labels_for_input_for_2_qubits_mixed_2_50000_datapoints.npy")
    
    print(f"Loading data from: {data_folder}")
    print(f"Loading labels from: {labels_path}")
    
    try:
        # Create dataset with matching 50K labels
        dataset = QuantumDataset(
            data_folder=str(data_folder),
            labels_path=str(labels_path),
            validate=True  # Enable validation now that we have matching data/labels
        )
        
        # Get actual available data size
        actual_size = min(len(dataset), max_samples)
        print(f"Using {actual_size} samples from dataset")
        
        # Create random subset for analysis
        if len(dataset) > max_samples:
            indices = np.random.choice(len(dataset), actual_size, replace=False)
            dataset = torch.utils.data.Subset(dataset, indices)
        
    except Exception as e:
        print(f"Dataset loading failed: {e}")
        print("Creating synthetic test data for analysis...")
        
        # Create synthetic data for testing
        matrix_dim = config['matrix_dim']
        num_samples = min(max_samples, 100)
        
        # Generate random density matrices
        rho_real_data = np.random.randn(num_samples, matrix_dim, matrix_dim).astype(np.float32) * 0.5
        rho_imag_data = np.random.randn(num_samples, matrix_dim, matrix_dim).astype(np.float32) * 0.5
        
        # Make them Hermitian
        for i in range(num_samples):
            rho_real_data[i] = (rho_real_data[i] + rho_real_data[i].T) / 2
            rho_imag_data[i] = (rho_imag_data[i] - rho_imag_data[i].T) / 2
        
        # Generate random magic labels
        labels_data = np.random.uniform(0, 2, num_samples).astype(np.float32)
        
        # Create simple dataset
        class SyntheticDataset(torch.utils.data.Dataset):
            def __init__(self, rho_real, rho_imag, labels):
                self.rho_real = torch.from_numpy(rho_real)
                self.rho_imag = torch.from_numpy(rho_imag)
                self.labels = torch.from_numpy(labels)
                
            def __len__(self):
                return len(self.labels)
                
            def __getitem__(self, idx):
                return self.rho_real[idx], self.rho_imag[idx], self.labels[idx]
        
        dataset = SyntheticDataset(rho_real_data, rho_imag_data, labels_data)
        print(f"Created synthetic dataset with {len(dataset)} samples")
    
    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0  # Set to 0 to avoid multiprocessing issues
    )
    
    print(f"Final dataset size: {len(dataset)} samples")
    
    return dataloader

def main():
    # Configuration
    model_path = "/Users/guwenlan/Desktop/XAI/CONFIGs/2_128_cls_moe_20250817_202509/best_2_128_cls_moe_20250817_202509.pth"
    config_path = "/Users/guwenlan/Desktop/XAI/CONFIGs/2_128_cls_moe_20250817_202509/2_128_cls_moe_config.json"
    output_dir = "/Users/guwenlan/Desktop/XAI/MOE_ANA/results"
    
    print("=" * 80)
    print("MoE QUANTUM MAGIC PREDICTION MODEL ANALYSIS")
    print("=" * 80)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model and config
    print("\n1. Loading model and configuration...")
    model, config, device = load_model_and_config(model_path, config_path)
    
    # Verify this is a MoE model
    if config['mlp_type'] != 'mixture_of_experts':
        print(f"WARNING: Model type is '{config['mlp_type']}', not 'mixture_of_experts'")
        print("Analysis will proceed but MoE-specific features may not be available.")
    
    # Create dataloader
    print("\n2. Creating dataloader...")
    dataloader = create_dataloader(config, batch_size=16, max_samples=500)
    
    # Initialize analyzer
    print("\n3. Initializing MoE analyzer...")
    analyzer = MoEAttentionAnalyzer(model, device, config)
    
    # Initialize visualizer
    visualizer = MoEVisualizer(output_dir)
    
    # Run analysis
    print("\n4. Running comprehensive MoE analysis...")
    analysis_results = analyzer.analyze_dataset_with_moe(dataloader, max_samples=500)
    
    print(f"\nAnalysis completed for {len(analysis_results['sample_results'])} samples")
    
    # Create visualizations
    print("\n5. Creating visualizations...")
    visualizer.create_moe_analysis_dashboard(analysis_results)
    
    # Print summary
    print("\n6. Analysis Summary:")
    print("=" * 50)
    
    summary = analysis_results.get('summary', {})
    if 'prediction_stats' in summary:
        pred_stats = summary['prediction_stats']
        print(f"Mean Absolute Error: {pred_stats['mae']:.4f}")
        print(f"Prediction Range: [{pred_stats['prediction_range'][0]:.4f}, {pred_stats['prediction_range'][1]:.4f}]")
        print(f"True Range: [{pred_stats['true_range'][0]:.4f}, {pred_stats['true_range'][1]:.4f}]")
    
    if 'expert_stats' in summary:
        expert_stats = summary['expert_stats']
        print(f"Expert Agreement (mean): {expert_stats['mean_expert_agreement']:.4f}")
        print(f"Expert Balance: {expert_stats['expert_balance']:.3f}")
        print(f"Dominant Expert Distribution: {expert_stats['dominant_expert_distribution']}")
    
    expert_statistics = analysis_results.get('expert_statistics', {})
    if 'avg_expert_usage' in expert_statistics:
        avg_usage = expert_statistics['avg_expert_usage']
        print("Average Expert Usage:")
        for i, usage in enumerate(avg_usage):
            print(f"  Expert {i}: {usage:.1%}")
    
    if 'expert_diversity' in expert_statistics:
        diversity = expert_statistics['expert_diversity']
        print(f"Expert Diversity (normalized entropy): {diversity['normalized_entropy']:.3f}")
    
    print(f"\nAll results saved to: {output_dir}")
    print("Files generated:")
    print("- moe_analysis_report.txt: Comprehensive analysis report")
    print("- expert_usage_analysis.png: Expert usage distribution and diversity")
    print("- attention_moe_correlation.png: Attention-expert correlations")
    print("- moe_analysis_sample_*.png: Individual sample analyses")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)

if __name__ == "__main__":
    main()