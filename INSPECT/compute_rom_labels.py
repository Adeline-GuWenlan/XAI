#!/usr/bin/env python3
"""
Compute RoM labels for existing 2-qubit density matrices.
"""

import numpy as np
import sys
import os

# Add RoMHandbook to path to import the function
sys.path.insert(0, '/Users/guwenlan/Desktop/XAI/RoMHandbook/RoM_handbook')

from generate_balanced_magic_dataset import calculate_RoM_for_2qubits


def compute_rom_labels(input_file: str, output_file: str):
    """
    Compute RoM labels for all matrices in the input file.

    Args:
        input_file (str): Path to .npy file containing density matrices
        output_file (str): Path to save RoM labels as .npy file
    """
    print("=" * 60)
    print("Computing RoM labels for 2-qubit density matrices")
    print("=" * 60)
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print()

    # Load the density matrices
    print("Loading density matrices...")
    matrices = np.load(input_file)
    num_samples = len(matrices)
    print(f"Loaded {num_samples} matrices with shape {matrices.shape}")
    print()

    # Initialize array for RoM values
    rom_values = np.zeros(num_samples)

    # Compute RoM for each matrix
    print("Computing RoM values...")
    failed_count = 0

    for i in range(num_samples):
        try:
            rom = calculate_RoM_for_2qubits(matrices[i])
            rom_values[i] = rom

            # Progress update every 1000 samples
            if (i + 1) % 1000 == 0:
                print(f"Progress: {i + 1}/{num_samples} ({100*(i+1)/num_samples:.1f}%)")

        except Exception as e:
            print(f"Warning: Failed to calculate RoM for matrix {i}: {e}")
            rom_values[i] = np.nan
            failed_count += 1

    print()
    print("=" * 60)
    print("Computation completed!")
    print("=" * 60)
    print(f"Total samples: {num_samples}")
    print(f"Successfully computed: {num_samples - failed_count}")
    print(f"Failed: {failed_count}")
    print()
    print("RoM Statistics:")
    print(f"  Mean: {np.nanmean(rom_values):.6f}")
    print(f"  Std:  {np.nanstd(rom_values):.6f}")
    print(f"  Min:  {np.nanmin(rom_values):.6f}")
    print(f"  Max:  {np.nanmax(rom_values):.6f}")
    print()

    # Count magic vs non-magic states
    magic_count = np.sum(rom_values >= 1)
    non_magic_count = np.sum(rom_values < 1)
    print(f"Magic states (RoM >= 1): {magic_count} ({100*magic_count/num_samples:.1f}%)")
    print(f"Non-magic states (RoM < 1): {non_magic_count} ({100*non_magic_count/num_samples:.1f}%)")
    print("=" * 60)

    # Save RoM values
    np.save(output_file, rom_values)
    print(f"\nSaved RoM labels to: {output_file}")
    print()


if __name__ == "__main__":
    input_file = "/Users/guwenlan/Desktop/XAI/INSPECT/Data/2q_50000samples_poisson_lam1.1_20251111_180142_full_matrices.npy"
    output_file = "/Users/guwenlan/Desktop/XAI/INSPECT/Data/2q_50000samples_poisson_lam1.1_20251111_180142_RoM.npy"

    compute_rom_labels(input_file, output_file)