#!/usr/bin/env python3
"""
Test script to verify that the QuantumDataset implementation works correctly
with the existing data structure in Rawdata/
"""
import torch
import numpy as np
from Model_Archi.quantum_dataset import create_dataloaders, test_dataloader

def test_dataset_structure():
    """Test that the dataset loads correctly with the existing data structure"""
    
    data_folder = "Rawdata/DecodedTokens"
    labels_path = "Rawdata/label/magic_labels_for_input_for_2_qubits_mixed_2_200000_datapoints.npy"
    
    print("Testing QuantumDataset with existing data structure...")
    print(f"Data folder: {data_folder}")
    print(f"Labels path: {labels_path}")
    
    try:
        # Test the simple test function first
        test_dataloader(data_folder, labels_path)
        
        print("\n" + "="*50)
        print("FULL DATASET TEST")
        print("="*50)
        
        # Create actual dataloaders
        train_loader, val_loader, test_loader = create_dataloaders(
            data_folder=data_folder,
            labels_path=labels_path,
            batch_size=16,
            train_ratio=0.8,
            val_ratio=0.1,
            num_workers=2,
            validate=True
        )
        
        print(f"✅ Created dataloaders successfully!")
        print(f"   Train batches: {len(train_loader)}")
        print(f"   Val batches: {len(val_loader)}")
        print(f"   Test batches: {len(test_loader)}")
        
        # Test iterating through test loader
        print("\nTesting data iteration...")
        for i, (rho_real, rho_imag, labels) in enumerate(test_loader):
            print(f"  Batch {i+1}:")
            print(f"    rho_real: {rho_real.shape}, dtype: {rho_real.dtype}")
            print(f"    rho_imag: {rho_imag.shape}, dtype: {rho_imag.dtype}")
            print(f"    labels: {labels.shape}, dtype: {labels.dtype}")
            print(f"    Label range: [{labels.min():.4f}, {labels.max():.4f}]")
            
            # Test a few batches
            if i >= 2:
                break
        
        print("✅ Dataset loading test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Dataset loading test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_dataset_structure()
    if success:
        print("\n🎉 All tests passed! The quantum dataset implementation is working correctly.")
    else:
        print("\n💥 Tests failed! Please check the data structure and implementation.")