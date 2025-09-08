#!/usr/bin/env python3
"""
Test script to check if all dependencies are available and run a minimal analysis
"""

import sys
import os
import traceback

def check_imports():
    """Check if all required modules can be imported"""
    print("Checking imports...")
    
    try:
        import torch
        print("✓ PyTorch available")
    except ImportError:
        print("✗ PyTorch not available")
        return False
    
    try:
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        print("✓ Basic scientific libraries available")
    except ImportError:
        print("✗ Missing basic scientific libraries")
        return False
    
    # Check our custom modules
    modules_to_check = [
        'saliency_analyzer',
        'saliency_visualizer', 
        'attention_analyzer',
        'attention_visualizer',
        'quantum_dataset',
        'updated_training_model',
        'saliency_core'
    ]
    
    missing_modules = []
    for module in modules_to_check:
        try:
            __import__(module)
            print(f"✓ {module} available")
        except ImportError as e:
            print(f"✗ {module} missing: {e}")
            missing_modules.append(module)
    
    # Check model dependencies
    try:
        from updated_training_model import CompleteQuantumMagicPredictor
        print("✓ Main model class available")
    except ImportError as e:
        print(f"✗ Model import failed: {e}")
        missing_modules.append('model_dependencies')
        
    return len(missing_modules) == 0

def check_data_files():
    """Check if model and data files exist"""
    print("\nChecking data files...")
    
    model_path = "best_128_CLS_attention_enhanced_20250810_214428.pth"
    config_path = "128_CLS_attention_enhanced_20250810_214428/128_CLS_attention_enhanced_config.json"
    
    if os.path.exists(model_path):
        print(f"✓ Model file found: {model_path}")
    else:
        print(f"✗ Model file missing: {model_path}")
        return False
        
    if os.path.exists(config_path):
        print(f"✓ Config file found: {config_path}")
    else:
        print(f"✗ Config file missing: {config_path}")
        return False
        
    return True

def create_minimal_test():
    """Create a minimal test to verify the analysis pipeline"""
    print("\nRunning minimal test...")
    
    try:
        import torch
        import numpy as np
        from saliency_analyzer import SaliencyAnalyzer
        from attention_analyzer import AttentionAnalyzer
        
        # Create dummy model (just for testing imports)
        device = torch.device('cpu')
        
        print("✓ All analyzers imported successfully")
        print("✓ Ready to run full analysis!")
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("XAI Analysis Dependency Check")
    print("=" * 40)
    
    imports_ok = check_imports()
    files_ok = check_data_files() 
    test_ok = create_minimal_test() if imports_ok else False
    
    print("\n" + "=" * 40)
    print("SUMMARY:")
    print(f"Imports: {'✓ PASS' if imports_ok else '✗ FAIL'}")
    print(f"Files: {'✓ PASS' if files_ok else '✗ FAIL'}")
    print(f"Test: {'✓ PASS' if test_ok else '✗ FAIL'}")
    
    if imports_ok and files_ok and test_ok:
        print("\n🎉 All checks passed! Ready to run analysis.")
        print("\nRecommended analysis order:")
        print("1. Attention Weight Visualization (most interpretable)")
        print("2. Integrated Gradients Saliency (most reliable)")
        print("3. Layer-wise Probing (reveals information flow)")
        print("\nTo run: python run_simplified_analysis.py")
    else:
        print("\n⚠️  Some issues detected. Please resolve them first.")