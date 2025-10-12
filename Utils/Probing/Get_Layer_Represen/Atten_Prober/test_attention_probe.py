#!/usr/bin/env python3
"""
Test script for AttentionPoolingProbe
Demonstrates how to extract attention-weighted pooling vectors and MLP representations
"""
import torch
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from Probing.Get_Layer_Represen.Atten_Prober.attention_pooling_probe import AttentionPoolingProbe

def test_attention_probe():
    """Test the attention pooling probe with a model"""
    
    # Example usage - you'll need to adapt this to your actual model loading
    print("Testing AttentionPoolingProbe...")
    
    # This is a template - replace with your actual model loading code
    """
    # Load your trained model
    from Model_Archi.updated_training_model import CompleteQuantumMagicPredictor
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Model configuration - adjust to match your trained model
    model = CompleteQuantumMagicPredictor(
        matrix_dim=4,
        n_qubits=2, 
        d_model=128,
        pooling_type="attention",  # Important: must be attention pooling
        mlp_type="attention_enhanced",
        use_physics_mask=False
    )
    
    # Load trained weights
    model_path = "path/to/your/trained/model.pth"
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # Create probe
    probe = AttentionPoolingProbe(model, device)
    
    # Load test data (replace with your data loading)
    test_data_path = "path/to/test_data.pth"
    test_data = torch.load(test_data_path, map_location=device)
    
    rho_real = test_data['rho_real'][:10]  # First 10 samples
    rho_imag = test_data['rho_imag'][:10]
    
    print(f"Input shapes: rho_real={rho_real.shape}, rho_imag={rho_imag.shape}")
    
    # Extract representations
    representations = probe.extract_representations(rho_real, rho_imag)
    
    print("Extracted representations:")
    for name, data in representations.items():
        print(f"  {name}: {data.shape}")
    
    # Save to CSV files
    output_dir = "./attention_pooling_representations"
    probe.save_representations_csv(representations, output_dir, prefix="test")
    
    print(f"Representations saved to {output_dir}")
    """
    
    print("Template created. Please:")
    print("1. Update the model loading code with your actual model path and configuration")
    print("2. Update the data loading code with your actual test data")
    print("3. Run the script to extract attention pooling representations")
    
    # Show available representations
    print("\nThe probe can capture these representations:")
    
    # Mock model for demonstration
    class MockModel:
        def __init__(self):
            self.pooling_type = "attention"
            self.physics_pooling = None
            
    mock_model = MockModel()
    device = torch.device('cpu')
    probe = AttentionPoolingProbe(mock_model, device)
    
    for rep in probe.get_available_representations():
        print(f"  - {rep}")

if __name__ == "__main__":
    test_attention_probe()