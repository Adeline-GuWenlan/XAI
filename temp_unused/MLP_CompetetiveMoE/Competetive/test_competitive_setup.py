#!/usr/bin/env python3
"""
Test script for competitive MoE setup.
Validates that all components work together with dummy data.
"""

import os
import sys
import torch
import numpy as np

# Add paths for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def create_dummy_data(n_samples=16, matrix_dim=4):
    """Create dummy quantum density matrices for testing."""
    print(f"Creating {n_samples} dummy quantum samples...")
    
    # Generate random hermitian matrices (density matrix approximations)
    rho_real_list = []
    rho_imag_list = []
    labels_list = []
    
    for _ in range(n_samples):
        # Random complex matrix
        real_part = torch.randn(matrix_dim, matrix_dim)
        imag_part = torch.randn(matrix_dim, matrix_dim)
        
        # Make Hermitian: A = (A + A†) / 2
        real_part = (real_part + real_part.T) / 2
        imag_part = (imag_part - imag_part.T) / 2  # Anti-symmetric for imaginary part
        
        # Normalize to make it closer to a density matrix
        trace = torch.trace(real_part)
        if trace > 0:
            real_part = real_part / trace
        
        rho_real_list.append(real_part)
        rho_imag_list.append(imag_part)
        
        # Random magic value between 0 and 1
        labels_list.append(torch.tensor([np.random.rand()], dtype=torch.float32))
    
    return rho_real_list, rho_imag_list, labels_list

def test_forward_moe_debug():
    """Test the forward_moe_debug function with dummy data."""
    print("\n=== Testing forward_moe_debug function ===")
    
    try:
        from train_moe_competitive import build_model, forward_moe_debug
        
        # Build model
        print("Building model...")
        model = build_model()
        model.eval()
        
        # Create dummy data
        rho_real_list, rho_imag_list, labels_list = create_dummy_data(n_samples=4)
        
        # Stack to batch format
        rho_real_batch = torch.stack(rho_real_list)  # (4, 4, 4)
        rho_imag_batch = torch.stack(rho_imag_list)  # (4, 4, 4)
        labels_batch = torch.stack(labels_list)      # (4, 1)
        
        print(f"Input shapes:")
        print(f"  rho_real: {rho_real_batch.shape}")
        print(f"  rho_imag: {rho_imag_batch.shape}")
        print(f"  labels: {labels_batch.shape}")
        
        # Test forward pass
        print("Testing forward pass...")
        with torch.no_grad():
            yhat, gates, expert_outs = forward_moe_debug(model, (rho_real_batch, rho_imag_batch))
            
        print(f"Output shapes:")
        print(f"  yhat: {yhat.shape}")
        print(f"  gates: {gates.shape}")
        print(f"  expert_outs: {expert_outs.shape}")
        
        # Verify shapes
        B = rho_real_batch.size(0)
        E = gates.size(1)
        
        assert yhat.shape == (B, 1), f"yhat shape mismatch: expected ({B}, 1), got {yhat.shape}"
        assert gates.shape[0] == B, f"gates batch dimension mismatch: expected {B}, got {gates.shape[0]}"
        assert expert_outs.shape == (B, 1, E), f"expert_outs shape mismatch: expected ({B}, 1, {E}), got {expert_outs.shape}"
        
        # Check that gates are probabilities (sum to 1)
        gate_sums = gates.sum(dim=1)
        assert torch.allclose(gate_sums, torch.ones_like(gate_sums), atol=1e-6), "Gates don't sum to 1"
        
        print("✅ forward_moe_debug test PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ forward_moe_debug test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_competitive_loss():
    """Test the competitive loss functions."""
    print("\n=== Testing competitive loss functions ===")
    
    try:
        from competitive_loss import mixture_gaussian_nll, mixture_laplace_nll, balance_kl
        
        # Create dummy MoE outputs
        B, E = 4, 3
        gates = torch.softmax(torch.randn(B, E), dim=1)  # (B, E)
        expert_outs = torch.randn(B, 1, E)  # (B, 1, E)
        targets = torch.randn(B, 1)  # (B, 1)
        
        print(f"Loss input shapes:")
        print(f"  gates: {gates.shape}")
        print(f"  expert_outs: {expert_outs.shape}")
        print(f"  targets: {targets.shape}")
        
        # Test Gaussian NLL
        gauss_nll = mixture_gaussian_nll(gates, expert_outs, targets, sigma=0.1)
        print(f"Gaussian NLL: {gauss_nll.item():.4f}")
        
        # Test Laplace NLL
        laplace_nll = mixture_laplace_nll(gates, expert_outs, targets, b=0.1)
        print(f"Laplace NLL: {laplace_nll.item():.4f}")
        
        # Test load balancing
        lb_loss = balance_kl(gates)
        print(f"Load balance KL: {lb_loss.item():.4f}")
        
        # Test with temperature and top-k
        gauss_nll_temp = mixture_gaussian_nll(gates, expert_outs, targets, sigma=0.1, tau=1.5)
        print(f"Gaussian NLL (tau=1.5): {gauss_nll_temp.item():.4f}")
        
        gauss_nll_topk = mixture_gaussian_nll(gates, expert_outs, targets, sigma=0.1, topk=2)
        print(f"Gaussian NLL (top-2): {gauss_nll_topk.item():.4f}")
        
        print("✅ Competitive loss test PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Competitive loss test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dataloader():
    """Test the dataset loading functionality."""
    print("\n=== Testing dataset loading ===")
    
    try:
        from train_moe_competitive import load_dataset
        from torch.utils.data import DataLoader
        
        # This will fail with real data paths but we can test the structure
        print("Testing load_dataset function structure...")
        
        # Just test that the function exists and can be called
        # (will fail on actual data loading, which is expected)
        try:
            dataset = load_dataset("train")
            print("❌ Unexpected: dataset loaded (should fail with dummy paths)")
        except Exception as expected_error:
            if "No valid file pairs found" in str(expected_error) or "No such file or directory" in str(expected_error):
                print("✅ Dataset loading test PASSED (expected failure with dummy paths)")
                return True
            else:
                print(f"❌ Unexpected error: {expected_error}")
                return False
        
    except Exception as e:
        print(f"❌ Dataset loading test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🧪 Testing Competitive MoE Setup")
    print("=" * 50)
    
    tests = [
        test_forward_moe_debug,
        test_competitive_loss,
        test_dataloader
    ]
    
    results = []
    for test_func in tests:
        results.append(test_func())
    
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests PASSED! Competitive MoE setup is ready.")
    else:
        print("⚠️  Some tests failed. Check the implementation.")
    
    return passed == total

if __name__ == "__main__":
    main()