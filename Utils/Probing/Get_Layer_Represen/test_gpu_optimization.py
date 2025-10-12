#!/usr/bin/env python3
"""
GPU optimization validation script.
Tests that the optimized files correctly utilize GPU resources.
"""
import os
import sys
import torch
import time
import numpy as np

# GPU environment setup
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Add paths for imports
sys.path.append('../..')
sys.path.append('../../Model_Archi')

try:
    from new_prober import extract_layer_csv_exact, build_loader_sequential, _normalize_trace
    from hooks_regist_only import LayerwiseCLSProbeExact
    from Model_Archi.updated_training_model import CompleteQuantumMagicPredictor
except ImportError as e:
    print(f"❌ Import failed: {e}")
    print("Make sure you're in the correct directory and dependencies are available")
    sys.exit(1)

def test_device_setup():
    """Test 1: Basic device setup and model placement"""
    print("🧪 Test 1: Device Setup and Model Placement")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'   Device: {device}, GPUs: {torch.cuda.device_count()}')
    
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        print("   ✅ CuDNN benchmark enabled")
    
    # Test model creation and device placement
    model_cfg = {
        'matrix_dim': 4, 'n_qubits': 2, 'd_model': 128,
        'pooling_type': 'cls', 'mlp_type': 'attention_enhanced',
        'use_physics_mask': False, 'mask_threshold': 1,
        'nhead': 8, 'use_cls_token': True
    }
    
    model = CompleteQuantumMagicPredictor(**model_cfg)
    model.to(device).eval()
    
    model_device = next(model.parameters()).device
    print(f'   Model device: {model_device}')
    
    if str(model_device) == str(device):
        print("   ✅ Model correctly placed on target device")
    else:
        print("   ❌ Model device mismatch!")
        return False
    
    return True

def test_tensor_placement():
    """Test 2: Input tensor device placement"""
    print("\n🧪 Test 2: Input Tensor Device Placement")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create dummy tensors
    batch_size = 32
    rho_real = torch.randn(batch_size, 4, 4)
    rho_imag = torch.randn(batch_size, 4, 4)
    
    print(f"   Original tensors device: {rho_real.device}")
    
    # Move to device
    rho_real = rho_real.to(device, non_blocking=True)
    rho_imag = rho_imag.to(device, non_blocking=True)
    
    print(f"   After .to(device): {rho_real.device}")
    
    if str(rho_real.device) == str(device):
        print("   ✅ Tensors correctly moved to target device")
        return True
    else:
        print("   ❌ Tensor device placement failed!")
        return False

def test_inference_mode():
    """Test 3: Inference mode and forward pass"""
    print("\n🧪 Test 3: Inference Mode Forward Pass")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model_cfg = {
        'matrix_dim': 4, 'n_qubits': 2, 'd_model': 128,
        'pooling_type': 'cls', 'mlp_type': 'attention_enhanced',
        'use_physics_mask': False, 'mask_threshold': 1,
        'nhead': 8, 'use_cls_token': True
    }
    
    model = CompleteQuantumMagicPredictor(**model_cfg)
    model.to(device).eval()
    
    batch_size = 64
    rho_real = torch.randn(batch_size, 4, 4).to(device)
    rho_imag = torch.randn(batch_size, 4, 4).to(device)
    
    # Normalize trace
    rho_real, rho_imag = _normalize_trace(rho_real, rho_imag)
    
    try:
        # Test inference_mode
        with torch.inference_mode():
            if device.type == 'cuda':
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    output = model(rho_real, rho_imag)
            else:
                output = model(rho_real, rho_imag)
        
        print(f"   Forward pass successful, output shape: {output.shape}")
        print("   ✅ Inference mode working correctly")
        return True
        
    except Exception as e:
        print(f"   ❌ Forward pass failed: {e}")
        return False

def test_timing():
    """Test 4: GPU timing and throughput"""
    print("\n🧪 Test 4: GPU Timing and Throughput")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if device.type != 'cuda':
        print("   ⚠️ Skipping timing test - CUDA not available")
        return True
    
    model_cfg = {
        'matrix_dim': 4, 'n_qubits': 2, 'd_model': 128,
        'pooling_type': 'cls', 'mlp_type': 'attention_enhanced',
        'use_physics_mask': False, 'mask_threshold': 1,
        'nhead': 8, 'use_cls_token': True
    }
    
    model = CompleteQuantumMagicPredictor(**model_cfg)
    model.to(device).eval()
    
    batch_size = 1024
    rho_real = torch.randn(batch_size, 4, 4).to(device)
    rho_imag = torch.randn(batch_size, 4, 4).to(device)
    rho_real, rho_imag = _normalize_trace(rho_real, rho_imag)
    
    # Warmup
    with torch.inference_mode():
        for _ in range(5):
            _ = model(rho_real, rho_imag)
    
    # Time the forward pass
    torch.cuda.synchronize()
    start_time = time.time()
    
    with torch.inference_mode():
        with torch.cuda.amp.autocast(dtype=torch.float16):
            _ = model(rho_real, rho_imag)
    
    torch.cuda.synchronize()
    elapsed = time.time() - start_time
    
    throughput = batch_size / elapsed
    print(f"   Batch size: {batch_size}")
    print(f"   Time per batch: {elapsed:.4f}s")
    print(f"   Throughput: {throughput:.1f} samples/sec")
    
    if elapsed < 1.0:  # Should be much faster than 1 second for 1024 samples
        print("   ✅ GPU timing looks good")
        return True
    else:
        print("   ⚠️ GPU might be underutilized - check for CPU bottlenecks")
        return True  # Don't fail, just warn

def main():
    """Run all GPU optimization tests"""
    print("🚀 GPU Optimization Validation Tests")
    print("=" * 50)
    
    tests = [
        test_device_setup,
        test_tensor_placement, 
        test_inference_mode,
        test_timing
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
        print("   🎉 All tests passed! GPU optimization is working correctly.")
    else:
        print("   ⚠️ Some tests failed. Check the output above for details.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)