#!/usr/bin/env python3
"""
Test script to verify mixture_of_experts hooks work correctly.
"""
import os
import sys
import torch

# GPU environment setup
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Add paths for imports
sys.path.append('../..')
sys.path.append('../../Model_Archi')

try:
    from hooks_regist_only import LayerwiseCLSProbeExact
    from Model_Archi.updated_training_model import CompleteQuantumMagicPredictor
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

def test_moe_architecture():
    """Test the mixture_of_experts architecture hooks"""
    print("🧪 Testing Mixture of Experts Architecture")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Device: {device}")
    
    # Create MoE model configuration
    model_cfg = {
        'matrix_dim': 4,
        'n_qubits': 2,
        'd_model': 128,
        'pooling_type': 'cls',
        'mlp_type': 'mixture_of_experts',  # Key change
        'use_physics_mask': False,
        'mask_threshold': 1,
        'nhead': 8,
        'use_cls_token': True
    }
    
    print(f"   Model config: {model_cfg['mlp_type']}")
    
    try:
        # Create model
        model = CompleteQuantumMagicPredictor(**model_cfg)
        model.to(device).eval()
        
        print(f"   ✅ Model created successfully")
        print(f"   Model MLP type: {getattr(model.mlp_head, 'mlp_type', 'unknown')}")
        
        # Create prober
        prober = LayerwiseCLSProbeExact(model, device)
        
        # Get expected keys
        expected_keys = prober.get_expected_keys()
        print(f"   Expected keys: {expected_keys}")
        
        # Create dummy data
        batch_size = 4
        rho_real = torch.randn(batch_size, 4, 4).to(device)
        rho_imag = torch.randn(batch_size, 4, 4).to(device)
        
        print(f"   ✅ Created dummy tensors: {rho_real.shape}")
        
        # Test hook registration and forward pass
        cache = prober.run_once(rho_real, rho_imag)
        
        print(f"   ✅ Forward pass successful")
        print(f"   Actual keys captured: {list(cache.keys())}")
        print(f"   Number of keys: {len(cache)}")
        
        # Check key shapes
        for k, v in cache.items():
            print(f"      {k}: {v.shape}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_attention_enhanced_compatibility():
    """Test that attention_enhanced still works"""
    print("\n🧪 Testing Attention Enhanced Compatibility")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create attention_enhanced model configuration
    model_cfg = {
        'matrix_dim': 4,
        'n_qubits': 2,
        'd_model': 128,
        'pooling_type': 'cls',
        'mlp_type': 'attention_enhanced',
        'use_physics_mask': False,
        'mask_threshold': 1,
        'nhead': 8,
        'use_cls_token': True
    }
    
    try:
        model = CompleteQuantumMagicPredictor(**model_cfg)
        model.to(device).eval()
        
        prober = LayerwiseCLSProbeExact(model, device)
        expected_keys = prober.get_expected_keys()
        
        batch_size = 4
        rho_real = torch.randn(batch_size, 4, 4).to(device)
        rho_imag = torch.randn(batch_size, 4, 4).to(device)
        
        cache = prober.run_once(rho_real, rho_imag)
        
        print(f"   ✅ Attention enhanced still works")
        print(f"   Keys captured: {len(cache)}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Attention enhanced test failed: {e}")
        return False

def main():
    print("🚀 Testing Architecture-Agnostic Hooks")
    print("=" * 50)
    
    tests = [
        test_moe_architecture,
        test_attention_enhanced_compatibility
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"   ❌ Test failed with exception: {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    print("📊 Test Summary")
    
    passed = sum(results)
    total = len(results)
    
    print(f"   Passed: {passed}/{total}")
    
    if passed == total:
        print("   🎉 All tests passed! Architecture-agnostic hooks working correctly.")
    else:
        print("   ⚠️ Some tests failed. Check the output above for details.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)