#!/usr/bin/env python3
#NOTE updated with the correct mixing method;
"""
Integrated pipeline for complete quantum state dataset generation.

This script generates random quantum states and computes:
1. Full density matrices (complex Hermitian matrices)
2. Upper triangular vector representation (real-valued flattened form)
3. Pauli expectation values (expectation vector)
4. Stabilizer norm (SN) labels

Usage:
    python generate_complete_dataset.py --n_qubits 2 --num_samples 1000 --output ./output

Author: Integrated from Matrix, Expectation, and SN_Label modules
"""

import argparse
import numpy as np
import os
from datetime import datetime

# Import from local modules
from random_mix import random_mixed_state, random_mixed_state_poisson, validate_rho
from upper_tri import rho_to_vec
from vector_to_hermitian import vec_to_rho
from pauli_vector_code import get_explist
from generate_stabilizer_dataset_notFull import get_sn
def random_mixed_state_uniform_radius(D):
    psi = np.random.uniform(-1, 1, D) + 1.j * np.random.uniform(-1, 1, D)
    psi = psi/np.linalg.norm(psi)
    rho = np.tensordot(psi, np.conjugate(psi), axes = 0)
    tensproduct = np.tensordot(psi, np.conjugate(psi), axes = 0)
    mu = np.random.uniform(0,1)
    return (1-mu) * np.identity(tensproduct[0].size) / D + mu * tensproduct


def generate_complete_dataset(n_qubits, num_samples, output_dir=None, seed=None,
                              method='uniform', verbose=True):
    """
    Generate complete quantum state dataset with all representations.

    Args:
        n_qubits (int): Number of qubits (determines matrix dimension as 2^n_qubits)
        num_samples (int): Number of samples to generate
        output_dir (str): Directory to save .npy files (optional)
        seed (int): Random seed for reproducibility (optional)
        method (str): Mixing method - 'convex' or 'poisson' (default: 'convex')
        lam (float): Lambda parameter for Poisson method (default: 1.0)
        verbose (bool): Print progress information

    Returns:
        dict: Dictionary containing all generated data arrays
    """
    if seed is not None:
        np.random.seed(seed)

    D = 2 ** n_qubits

    if verbose:
        print("=" * 60)
        print(f"Quantum State Dataset Generation Pipeline")
        print("=" * 60)
        print(f"Number of qubits: {n_qubits}")
        print(f"Matrix dimension: {D}x{D}")
        print(f"Number of samples: {num_samples}")
        if seed is not None:
            print(f"Random seed: {seed}")
        print("=" * 60)

    # Initialize storage arrays
    full_matrices = []
    upper_tri_vectors = []
    pauli_vectors = []
    sn_labels = []

    # Progress tracking
    checkpoint_interval = max(1, num_samples // 10)

    for i in range(num_samples):
        if verbose and (i + 1) % checkpoint_interval == 0:
            print(f"Progress: {i + 1}/{num_samples} ({(i+1)/num_samples*100:.1f}%)")

        # 1. Generate random mixed state density matrix
        """if method == 'poisson':
            rho = random_mixed_state_poisson(D=D, lam=lam)
        elif method == 'convex':
            rho = random_mixed_state(D=D)"""
        if method == 'uniform':
            rho = random_mixed_state_uniform_radius(D)

        # Validate the density matrix (optional but recommended)
        is_valid, msg = validate_rho(rho, base_eps=1e-10)
        if not is_valid:
            if verbose:
                print(f"Warning: Sample {i} validation failed: {msg}")
                print("Regenerating...")
            # Retry with a new matrix
                rho = random_mixed_state_uniform_radius(D)
            is_valid, msg = validate_rho(rho, base_eps=1e-10)
            if not is_valid:
                raise ValueError(f"Failed to generate valid matrix after retry: {msg}")

        # 2. Compute upper triangular vector representation
        upper_tri_vec = rho_to_vec(rho)

        # 3. Compute Pauli expectation values
        pauli_vec = get_explist(rho)

        # 4. Compute stabilizer norm label
        sn_label = get_sn(rho, qubits= n_qubits)

        # Store results
        full_matrices.append(rho)
        upper_tri_vectors.append(upper_tri_vec)
        pauli_vectors.append(pauli_vec)
        sn_labels.append(sn_label)

    # Convert to numpy arrays
    full_matrices = np.array(full_matrices)
    upper_tri_vectors = np.array(upper_tri_vectors)
    pauli_vectors = np.array(pauli_vectors)
    sn_labels = np.array(sn_labels)

    if verbose:
        print("\n" + "=" * 60)
        print("Dataset Generation Completed!")
        print("=" * 60)
        print(f"Full matrices shape: {full_matrices.shape}")
        print(f"Upper triangular vectors shape: {upper_tri_vectors.shape}")
        print(f"Pauli vectors shape: {pauli_vectors.shape}")
        print(f"SN labels shape: {sn_labels.shape}")
        print("\nSN Label Statistics:")
        print(f"  Range: [{np.min(sn_labels):.6f}, {np.max(sn_labels):.6f}]")
        print(f"  Mean: {np.mean(sn_labels):.6f}")
        print(f"  Std: {np.std(sn_labels):.6f}")
        print("=" * 60)

    # Save to files if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

        # Create timestamp for unique filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Include method and lambda in filename for uniform method
        if method == 'uniform':
            prefix = f"{n_qubits}q_{num_samples}samples_uniform_{timestamp}"

        files = {
            'full_matrices': os.path.join(output_dir, f"{prefix}_full_matrices.npy"),
            'upper_tri_vectors': os.path.join(output_dir, f"{prefix}_upper_tri_vectors.npy"),
            'pauli_vectors': os.path.join(output_dir, f"{prefix}_pauli_vectors.npy"),
            'sn_labels': os.path.join(output_dir, f"{prefix}_sn_labels.npy")
        }

        np.save(files['full_matrices'], full_matrices)
        np.save(files['upper_tri_vectors'], upper_tri_vectors)
        np.save(files['pauli_vectors'], pauli_vectors)
        np.save(files['sn_labels'], sn_labels)

        if verbose:
            print("\nSaved files:")
            for key, filepath in files.items():
                print(f"  {key}: {filepath}")
            print("=" * 60)

    return {
        'full_matrices': full_matrices,
        'upper_tri_vectors': upper_tri_vectors,
        'pauli_vectors': pauli_vectors,
        'sn_labels': sn_labels
    }


def main():
    parser = argparse.ArgumentParser(
        description='Generate complete quantum state dataset with all representations',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--n_qubits', '-n', type=int, required=True,
                        help='Number of qubits (determines matrix dimension as 2^n)')
    parser.add_argument('--num_samples', '-s', type=int, required=True,
                        help='Number of samples to generate')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output directory to save .npy files (optional)')
    parser.add_argument('--method', '-m', type=str, default='uniform',
                        choices=["uniform"],
                        help='Mixing method: convex (p*pure + (1-p)*I/D) or poisson (Poisson-distributed mixing)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility (optional)')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Suppress progress output')

    args = parser.parse_args()

    # Validate arguments
    if args.n_qubits < 1:
        parser.error("n_qubits must be at least 1")
    if args.num_samples < 1:
        parser.error("num_samples must be at least 1")

    # Generate dataset
    try:
        results = generate_complete_dataset(
            n_qubits=args.n_qubits,
            num_samples=args.num_samples,
            output_dir=args.output,
            seed=args.seed,
            method=args.method,
            verbose=not args.quiet
        )

        if not args.quiet:
            print("\n✓ Pipeline completed successfully!")

    except Exception as e:
        print(f"\n✗ Error during generation: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
