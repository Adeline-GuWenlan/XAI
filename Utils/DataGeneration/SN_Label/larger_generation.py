#!/usr/bin/env python3
"""
Generate dataset of random quantum states and their stabilizer norm labels for larger qubit systems (3-5 qubits).
Adapted from generate_stabilizer_dataset.py with improved stabilizer norm calculation.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import random

# Random state generation functions
def random_pure_state(D: int) -> np.ndarray:
    """Generate a random pure state density matrix."""
    psi = np.random.randn(D) + 1j * np.random.randn(D)
    psi /= np.linalg.norm(psi)
    return np.outer(psi, psi.conj())

def random_mixed_state_poisson(D, lam=1.0):
    """
    Generate random mixed state using Poisson distribution.
    K ~ Poisson(lam) + 1
    """
    K = np.random.poisson(lam) + 1
    rho = np.zeros((D, D), dtype=np.complex128)

    for _ in range(K):
        rho += random_pure_state(D)
    rho /= K
    rho /= np.trace(rho)
    return rho

# Pauli matrices
I = np.array([[1,0],[0,1]], dtype=np.complex128)
X = np.array([[0,1],[1,0]], dtype=np.complex128)
Y = np.array([[0,-1j],[1j,0]], dtype=np.complex128)
Z = np.array([[1,0],[0,-1]], dtype=np.complex128)
paulis = [I, X, Y, Z]

def number_to_base(n, b, length):
    """Convert integer n to length-digit base-b representation."""
    digits = []
    while n:
        digits.append(int(n % b))
        n //= b
    while len(digits) < length:
        digits.append(0)
    return digits[::-1]

def get_sn_general(rho, qubits):
    """
    General stabilizer norm calculation for arbitrary number of qubits.
    Returns stabilizer norm normalized so that threshold is always 1.

    For all qubits: Returns sum_{non-identity P} |Tr(rho P)| / (2^qubits - 1)
    This normalization ensures non-magic states have SN <= 1, magic states have SN > 1
    """
    # Ensure proper normalization
    rho = 0.5 * (rho + rho.conj().T)
    rho = rho / np.trace(rho)

    if qubits == 1:
        # Single qubit case - direct calculation
        rx = np.trace(rho @ X).real
        ry = np.trace(rho @ Y).real
        rz = np.trace(rho @ Z).real
        # Normalize by (2^1 - 1) = 1 to get threshold of 1
        return abs(rx) + abs(ry) + abs(rz)
    else:
        # Multi-qubit case - enumerate all Pauli combinations
        sn = 0.0
        dim_pauli = 4 ** qubits

        for idx in range(dim_pauli):
            # Convert idx to qubits-digit base-4 representation
            digits = number_to_base(idx, 4, qubits)

            # Skip identity operator (all zeros)
            if all(d == 0 for d in digits):
                continue

            # Construct tensor product Pauli matrix
            sigma = paulis[digits[0]]
            for d in digits[1:]:
                sigma = np.kron(sigma, paulis[d])

            # Add absolute value of trace
            sn += abs(np.trace(rho @ sigma))

        # Normalize by (2^qubits - 1) to get threshold of 1
        return sn / (2**qubits-1)

def generate_dataset(num_samples, qubits, lam, output_path=None, seed=None):
    """
    Generate dataset of quantum states and their stabilizer norm labels.

    Args:
        num_samples (int): Number of samples to generate
        qubits (int): Number of qubits (determines matrix dimension as 2^qubits)
        lam (float): Lambda parameter for Poisson distribution
        output_path (str): Path to save .npy files (optional)
        seed (int): Random seed for reproducibility (optional)

    Returns:
        tuple: (states, labels) - numpy arrays of quantum states and their labels
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    D = 2 ** qubits

    print(f"Generating {num_samples} samples for {qubits} qubits (dimension {D}x{D})")
    print(f"Poisson parameter lambda = {lam}")
    print(f"Total Pauli operators: {4**qubits}")
    print(f"Non-identity Pauli operators: {4**qubits - 1}")

    states = []
    labels = []

    for i in range(num_samples):
        if (i + 1) % 100 == 0:
            print(f"Progress: {i + 1}/{num_samples}")

        # Generate random mixed state
        rho = random_mixed_state_poisson(D, lam)

        # Normalize trace to exactly 1
        rho = rho / np.trace(rho)

        # Compute stabilizer norm label using general function
        sn_label = get_sn_general(rho, qubits)

        states.append(rho)
        labels.append(sn_label)

    states = np.array(states)
    labels = np.array(labels)

    print(f"Dataset generation completed!")
    print(f"States shape: {states.shape}")
    print(f"Labels shape: {labels.shape}")

    # Save to .npy files if output path provided
    if output_path:
        os.makedirs(output_path, exist_ok=True)
        states_file = os.path.join(output_path, f"{qubits}q_{num_samples}_states.npy")
        labels_file = os.path.join(output_path, f"{qubits}q_{num_samples}_labels.npy")

        np.save(states_file, states)
        np.save(labels_file, labels)

        print(f"Saved states to: {states_file}")
        print(f"Saved labels to: {labels_file}")

    return states, labels

def analyze_distribution(labels, qubits):
    """
    Analyze the distribution of stabilizer norm labels with universal threshold of 1.
    """
    print("\n=== Label Distribution Analysis ===")
    print(f"Total samples: {len(labels)}")
    print(f"Label range: [{np.min(labels):.6f}, {np.max(labels):.6f}]")
    print(f"Mean: {np.mean(labels):.6f}")
    print(f"Std: {np.std(labels):.6f}")

    # Universal threshold for non-magic states
    threshold = 1.0
    print(f"Non-magic threshold (universal): {threshold:.6f}")

    # Count above/below threshold
    below_threshold = np.sum(labels <= threshold)
    above_threshold = np.sum(labels > threshold)

    print(f"Labels <= {threshold:.3f} (non-magic): {below_threshold} ({below_threshold/len(labels)*100:.2f}%)")
    print(f"Labels > {threshold:.3f} (magic): {above_threshold} ({above_threshold/len(labels)*100:.2f}%)")

    # Additional statistics
    percentiles = [25, 50, 75, 90, 95, 99]
    print(f"\nPercentiles:")
    for p in percentiles:
        val = np.percentile(labels, p)
        print(f"  {p}th percentile: {val:.6f}")

def main():
    parser = argparse.ArgumentParser(description='Generate quantum state dataset with stabilizer norm labels for larger qubit systems')
    parser.add_argument('--num', type=int, required=True, help='Number of samples to generate')
    parser.add_argument('--qubits', type=int, required=True, help='Number of qubits (3-5 recommended)')
    parser.add_argument('--lam', type=float, default=1.0, help='Lambda parameter for Poisson distribution (default: 1.0)')
    parser.add_argument('--output', type=str, help='Output directory to save .npy files')
    parser.add_argument('--seed', type=int, help='Random seed for reproducibility')

    args = parser.parse_args()

    # Validate qubit number
    if args.qubits < 1:
        print("Error: Number of qubits must be at least 1")
        return

    if args.qubits > 5:
        print(f"Warning: {args.qubits} qubits will require computation of {4**args.qubits} Pauli operators.")
        print("This may be very slow. Consider using qubits <= 5.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return

    # Generate dataset
    print(f"Starting dataset generation for {args.qubits} qubits...")
    states, labels = generate_dataset(args.num, args.qubits, args.lam, args.output, args.seed)

    # Analyze distribution
    analyze_distribution(labels, args.qubits)

if __name__ == "__main__":
    main()