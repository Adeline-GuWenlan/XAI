# quantum_dataset.py
# 7.31: 更新Dataloaders，增加了pin_memory和persistent_workers参数
# 7.31: 更新batch_size参数，默认值改为2048

import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from typing import Tuple

class QuantumDataset(Dataset):
    """
    Simple dataset for pre-processed quantum density matrices.
    Expects _real.npy and _imag.npy files containing [B, D, D] matrices.
    """
    def __init__(self, 
                 data_folder: str,
                 labels_path: str,
                 validate: bool = True):
        """
        Args:
            data_folder: Folder containing _real.npy and _imag.npy files
            labels_path: Path to labels .npy file OR folder containing multiple .npy label files
            validate: Whether to validate data consistency
        """
        self.data_folder = data_folder
        self.labels_path = labels_path
        
        # Find and pair data and label files
        self._find_file_pairs()
        
        # Validate if requested
        if validate:
            self._validate_data()
    
    def _find_file_pairs(self):
        """Find and pair matrix files with corresponding label files."""
        # Get matrix files
        data_files = os.listdir(self.data_folder)
        real_files = sorted([f for f in data_files if f.endswith('_real.npy')])
        imag_files = sorted([f for f in data_files if f.endswith('_imag.npy')])
        
        # Get label files
        if os.path.isdir(self.labels_path):
            label_files = sorted([f for f in os.listdir(self.labels_path) if f.endswith('.npy')])
        else:
            # Single label file
            label_files = [os.path.basename(self.labels_path)]
            self.labels_path = os.path.dirname(self.labels_path)
        
        print(f"📊 Found {len(real_files)} real files, {len(imag_files)} imag files, {len(label_files)} label files")
        
        # Match files by base name
        self.file_triplets = []
        for real_file in real_files:
            base_name = real_file[:-9]  # Remove '_real.npy'
            imag_file = base_name + '_imag.npy'
            
            # Find matching label file
            matching_label = None
            for label_file in label_files:
                label_base = label_file[:-4]  # Remove '.npy'
                if base_name in label_base or label_base in base_name:
                    matching_label = label_file
                    break
            
            if imag_file in imag_files and matching_label:
                self.file_triplets.append((real_file, imag_file, matching_label))
                print(f"   Matched: {base_name} → {matching_label}")
            else:
                missing = []
                if imag_file not in imag_files:
                    missing.append("imag")
                if not matching_label:
                    missing.append("label")
                print(f"⚠️  Missing {', '.join(missing)} for {real_file}")
        
        print(f"📊 Found {len(self.file_triplets)} complete file triplets")
        
        if len(self.file_triplets) == 0:
            raise ValueError(f"No valid file triplets found")
    
    def _validate_data(self):
        """Validate data consistency."""
        total_samples = 0
        
        for real_file, imag_file, label_file in self.file_triplets:
            real_path = os.path.join(self.data_folder, real_file)
            imag_path = os.path.join(self.data_folder, imag_file)
            label_path = os.path.join(self.labels_path, label_file)
            
            real_data = np.load(real_path)
            imag_data = np.load(imag_path)
            labels = np.load(label_path)
            
            # Check matrix shapes match
            if real_data.shape != imag_data.shape:
                raise ValueError(f"Shape mismatch: {real_file} vs {imag_file}")
            
            # Check 3D format [B, D, D]
            if real_data.ndim != 3:
                raise ValueError(f"Expected 3D array [B,D,D], got {real_data.shape}")
            
            # Check label count matches matrix count
            if len(labels) != real_data.shape[0]:
                raise ValueError(f"Label count {len(labels)} != matrix count {real_data.shape[0]} for {real_file}")
            
            total_samples += real_data.shape[0]
        
        print(f"✅ Validation passed: {total_samples} samples")
    
    def __len__(self):
        # Calculate total samples across all file triplets
        if not hasattr(self, '_total_samples'):
            self._total_samples = 0
            for real_file, _, _ in self.file_triplets:
                real_path = os.path.join(self.data_folder, real_file)
                real_data = np.load(real_path)
                self._total_samples += real_data.shape[0]
        return self._total_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            rho_real: [D, D] real part of density matrix
            rho_imag: [D, D] imaginary part  
            label: [1] magic monotone value
        """
        # Find which file triplet contains this index
        current_idx = 0
        for real_file, imag_file, label_file in self.file_triplets:
            real_path = os.path.join(self.data_folder, real_file)
            imag_path = os.path.join(self.data_folder, imag_file)
            label_path = os.path.join(self.labels_path, label_file)
            
            # Load data to get file size
            real_data = np.load(real_path)
            file_size = real_data.shape[0]
            
            if current_idx <= idx < current_idx + file_size:
                # Found the right file triplet
                local_idx = idx - current_idx
                
                # Load imaginary and label data
                imag_data = np.load(imag_path)
                labels = np.load(label_path)
                
                # Convert to tensors
                rho_real = torch.from_numpy(real_data[local_idx]).float()  # [D, D]
                rho_imag = torch.from_numpy(imag_data[local_idx]).float()  # [D, D]
                label = torch.tensor([labels[local_idx]], dtype=torch.float32)  # [1]
                
                return rho_real, rho_imag, label
            
            current_idx += file_size
        
        raise IndexError(f"Index {idx} out of range")


def create_dataloaders(data_folder: str, 
                      labels_path: str,
                      batch_size: int = 512,
                      train_ratio: float = 0.8,
                      val_ratio: float = 0.1,
                      num_workers: int = 2,
                      validate: bool = True):
    """
    Create train/val/test dataloaders.
    
    Returns:
        Each batch: (rho_real, rho_imag, labels) with shapes ([B,D,D], [B,D,D], [B,1])
    """
    # Create dataset
    dataset = QuantumDataset(
        data_folder=data_folder,
        labels_path=labels_path,
        validate=validate
    )
    
    # Split dataset
    n_total = len(dataset)
    n_train = int(train_ratio * n_total)
    n_val = int(val_ratio * n_total)
    n_test = n_total - n_train - n_val
    
    print(f"📊 Split: Train={n_train}, Val={n_val}, Test={n_test}")
    
    train_set, val_set, test_set = random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create dataloaders
    common_kwargs = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'pin_memory': torch.cuda.is_available(),
        'persistent_workers': True,
        'prefetch_factor': 2,
    }
    
    train_loader = DataLoader(train_set, shuffle=True, **common_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **common_kwargs)
    test_loader = DataLoader(test_set, shuffle=False, **common_kwargs)
    
    return train_loader, val_loader, test_loader


# -----------------------------------------------
# Simple test function
# -----------------------------------------------
def test_dataloader(data_folder: str, labels_path: str):
    """Quick test to verify dataloader works."""
    print("🧪 Testing dataloader...")
    
    train_loader, _, _ = create_dataloaders(
        data_folder=data_folder,
        labels_path=labels_path,
        batch_size=4
    )
    
    # Test one batch
    rho_real, rho_imag, labels = next(iter(train_loader))
    
    print(f"✅ Batch shapes:")
    print(f"   rho_real: {rho_real.shape}")
    print(f"   rho_imag: {rho_imag.shape}")
    print(f"   labels: {labels.shape}")
    print(f"   Label range: [{labels.min():.3f}, {labels.max():.3f}]")


if __name__ == "__main__":
    # Example test
    test_dataloader("path/to/data", "path/to/labels.npy")