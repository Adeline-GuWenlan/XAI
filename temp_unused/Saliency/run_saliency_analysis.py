import torch
import argparse
import numpy as np
from pathlib import Path

from saliency_analyzer import SaliencyAnalyzer
from saliency_visualizer import SaliencyVisualizer
from Atten.attention_analyzer import AttentionAnalyzer
from Atten.attention_visualizer import AttentionVisualizer

# Import your actual functions
from Model_Archi.quantum_dataset import create_dataloaders
from Model_Archi.updated_training_model import CompleteQuantumMagicPredictor

def generate_quantum_labels(n_qubits):
    """Generate |00⟩, |01⟩, etc. labels for visualization"""
    labels = []
    for i in range(2**n_qubits):
        binary = format(i, f'0{n_qubits}b')
        labels.append(f'|{binary}⟩')
    return labels

def load_model(model_path, model_config, device):
    """Load model with your specific parameters"""
    model = CompleteQuantumMagicPredictor(
        matrix_dim=model_config['matrix_dim'],
        n_qubits=model_config['n_qubits'],
        d_model=model_config['d_model'],
        pooling_type=model_config['pooling_type'],
        mlp_type=model_config['mlp_type'],
        use_physics_mask=model_config['use_physics_mask'],
        mask_threshold=model_config['mask_threshold'],
        nhead=model_config['nhead']
    )
    
    # Load weights
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    return model

def main():
    parser = argparse.ArgumentParser(description='Saliency Analysis for Quantum Magic Detection')
    
    # Model and data paths
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model .pth file')
    parser.add_argument('--data_folder', type=str, 
                       default='Rawdata/DecodedTokens', 
                       help='Path to data folder containing _real.npy and _imag.npy files')
    parser.add_argument('--labels_path', type=str,
                       default='Rawdata/label/magic_labels_for_input_for_2_qubits_mixed_2_200000_datapoints.npy',
                       help='Path to labels .npy file')
    
    # Model configuration parameters
    parser.add_argument('--n_qubits', type=int, default=2, help='Number of qubits')
    parser.add_argument('--d_model', type=int, default=64, help='Model dimension')
    parser.add_argument('--pooling_type', type=str, default='cls', 
                       choices=['cls', 'mean', 'attention', 'structured'], help='Pooling type')
    parser.add_argument('--mlp_type', type=str, default='standard',
                       choices=['standard', 'physics_aware', 'attention_enhanced', 'mixture_of_experts', 'asymmetric_ensemble'],
                       help='MLP type')
    parser.add_argument('--use_physics_mask', action='store_true', help='Use physics mask in transformer')
    parser.add_argument('--mask_threshold', type=int, default=1, help='Physics mask threshold')
    parser.add_argument('--nhead', type=int, default=8, help='Number of attention heads')    
    # Data loading parameters
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for data loading')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of data loading workers')
    
    # Analysis parameters
    parser.add_argument('--methods', nargs='+', default=['gradient', 'integrated_gradients'],
                       choices=['gradient', 'integrated_gradients'], help='Saliency methods to use')
    parser.add_argument('--max_samples', type=int, default=150000, help='Max samples to analyze')
    parser.add_argument('--output_dir', type=str, default='xai_analysis_results', help='Output directory')
    parser.add_argument('--ig_steps', type=int, default=50, help='Integration steps for IG')
    
    # Attention analysis parameters
    parser.add_argument('--include_attention', action='store_true', default=True, 
                       help='Include attention weight analysis')
    parser.add_argument('--individual_samples', type=int, default=20, 
                       help='Number of individual samples for detailed analysis')

    # Integrated gradients parameters    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Prepare model configuration
    matrix_dim = 2 ** args.n_qubits
    model_config = {
        'matrix_dim': matrix_dim,
        'n_qubits': args.n_qubits,
        'd_model': args.d_model,
        'pooling_type': args.pooling_type,
        'mlp_type': args.mlp_type,
        'use_physics_mask': args.use_physics_mask,
        'mask_threshold': args.mask_threshold,
        'nhead': args.nhead
    }
    
    print("Model configuration:")
    for key, value in model_config.items():
        print(f"  {key}: {value}")
    
    # Load model
    print(f"Loading model from: {args.model_path}")
    model = load_model(args.model_path, model_config, device)
    
    # Load data - only use test set for saliency analysis
    print(f"Loading data from: {args.data_folder}")
    print(f"Labels from: {args.labels_path}")
    
    _, _, test_loader = create_dataloaders(
        data_folder=args.data_folder,
        labels_path=args.labels_path,
        batch_size=args.batch_size,
        train_ratio=0.8,
        val_ratio=0.1,
        num_workers=args.num_workers,
        validate=False  # Skip validation for speed
    )
    
    # Initialize analyzer
    config = {
        'ig_steps': args.ig_steps,
    }
    analyzer = SaliencyAnalyzer(model, device, config)
    
    # Initialize visualizer
    visualizer = SaliencyVisualizer(args.output_dir)
    
    print(f"Running saliency analysis with methods: {args.methods}")
    print(f"Analyzing up to {args.max_samples} samples...")
    
    # Run saliency analysis
    print("Running saliency analysis...")
    saliency_results = analyzer.analyze_dataset(test_loader, methods=args.methods, max_samples=args.max_samples)
    
    # Generate saliency visualizations
    quantum_labels = generate_quantum_labels(args.n_qubits)
    
    print("Generating saliency visualizations...")
    
    # Plot aggregated patterns
    visualizer.plot_aggregated_patterns(saliency_results)
    
    # Compare methods for first few samples
    n_comparison_samples = min(args.individual_samples, len(saliency_results[args.methods[0]]['sample_results']))
    for i in range(n_comparison_samples):
        visualizer.compare_methods(saliency_results, sample_idx=i, quantum_labels=quantum_labels)
    
    # Run attention analysis if requested
    if args.include_attention:
        print("\n" + "="*50)
        print("ATTENTION WEIGHT ANALYSIS")
        print("="*50)
        
        # Initialize attention analyzer and visualizer
        attention_analyzer = AttentionAnalyzer(model, device)
        attention_visualizer = AttentionVisualizer(
            args.output_dir + "/attention_analysis", 
            figsize_scale=1.2
        )
        
        # Run comprehensive attention analysis
        attention_results = attention_analyzer.analyze_dataset(
            test_loader, 
            max_samples=args.max_samples
        )
        
        # Create attention visualization dashboard
        print("Creating attention visualization dashboard...")
        attention_visualizer.create_attention_dashboard(attention_results, quantum_labels)
        
        print(f"Attention analysis complete! Results saved to {args.output_dir}/attention_analysis/")
    
    print(f"\nComplete XAI analysis finished! Results saved to {args.output_dir}/")
    
    # Print summary statistics
    print("\n" + "="*50)
    print("SALIENCY ANALYSIS SUMMARY")
    print("="*50)
    for method in args.methods:
        sample_results = saliency_results[method]['sample_results']
        errors = [r['prediction_error'] for r in sample_results]
        true_magics = [r['true_magic'] for r in sample_results]
        pred_magics = [r['predicted_magic'] for r in sample_results]
        
        print(f"\n{method} Summary:")
        print(f"  Samples analyzed: {len(sample_results)}")
        print(f"  Mean prediction error: {np.mean(errors):.4f} ± {np.std(errors):.4f}")
        print(f"  True magic range: [{np.min(true_magics):.4f}, {np.max(true_magics):.4f}]")
        print(f"  Predicted magic range: [{np.min(pred_magics):.4f}, {np.max(pred_magics):.4f}]")
    
    if args.include_attention:
        print("\n" + "="*50)
        print("ATTENTION ANALYSIS SUMMARY")
        print("="*50)
        attention_summary = attention_results['summary_stats']
        print(f"  Total samples: {attention_summary['total_samples']}")
        print(f"  Avg prediction error: {attention_summary['avg_prediction_error']:.4f}")
        print(f"  True magic range: [{attention_summary['true_magic_range'][0]:.4f}, {attention_summary['true_magic_range'][1]:.4f}]")
        print(f"  Predicted range: [{attention_summary['predicted_magic_range'][0]:.4f}, {attention_summary['predicted_magic_range'][1]:.4f}]")

if __name__ == "__main__":
    main()