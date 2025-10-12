import os
import argparse
import numpy as np
import torch
from tqdm import tqdm
from typing import Tuple

def complex_matrix_to_upper_triangle_vector(rho: np.ndarray) -> np.ndarray:
    """
    Convert complex Hermitian matrix to upper triangular vector format.

    Args:
        rho: Complex matrix of shape (D, D)

    Returns:
        Vector with diagonal elements followed by upper triangular real/imag pairs
    """
    D = rho.shape[0]
    vec = []

    # Add diagonal elements (real-valued for Hermitian matrices)
    for i in range(D):
        vec.append(rho[i, i].real)

    # Add upper triangular elements (real then imaginary parts)
    for i in range(D):
        for j in range(i + 1, D):
            vec.append(rho[i, j].real)
            vec.append(rho[i, j].imag)

    return np.array(vec)

def decode_upper_triangle_vector(vec: torch.Tensor, D: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert upper triangle vectors back to real and imaginary matrix parts.

    Args:
        vec: Tensor of shape (B, D²) containing flattened upper triangular data
        D: Dimension of the square matrix

    Returns:
        Tuple of (real_matrices, imag_matrices) each of shape (B, D, D)
    """
    B = vec.shape[0]
    device = vec.device

    rho_real = torch.zeros((B, D, D), dtype=torch.float32, device=device)
    rho_imag = torch.zeros((B, D, D), dtype=torch.float32, device=device)

    k = 0
    # Fill diagonal with real values
    for i in range(D):
        rho_real[:, i, i] = vec[:, k]
        k += 1

    # Fill upper triangular part with real/imag pairs
    for i in range(D):
        for j in range(i + 1, D):
            real = vec[:, k]
            imag = vec[:, k + 1]
            rho_real[:, i, j] = real
            rho_real[:, j, i] = real  # Symmetric for real part
            rho_imag[:, i, j] = imag
            rho_imag[:, j, i] = -imag  # Anti-symmetric for imaginary part
            k += 2

    return rho_real, rho_imag

def process_folder(input_folder: str, output_folder: str, D: int, debug=False):
    """
    Process folder containing full complex matrices and convert to decoder format.

    Args:
        input_folder: Path to folder with .npy files containing full complex matrices
        output_folder: Path to save decoded real/imag matrices
        D: Expected matrix dimension
        debug: Enable debug output
    """
    os.makedirs(output_folder, exist_ok=True)
    files = [f for f in os.listdir(input_folder) if f.endswith(".npy")]

    for filename in tqdm(files, desc="Processing complex matrices"):
        path = os.path.join(input_folder, filename)
        try:
            data = np.load(path)
            print(f"Processing {filename}: shape {data.shape}, dtype {data.dtype}")

            # Handle different input formats
            if filename.endswith("_labels.npy"):
                # Labels file - just copy as is
                if data.ndim == 1:
                    print(f"✅ Copying labels file: {filename}")
                    np.save(os.path.join(output_folder, filename), data)
                    continue
                else:
                    print(f"❌ Skipping {filename}: labels should be 1D, got shape {data.shape}")
                    continue

            # Process matrix files
            if data.ndim == 3 and data.shape[1] == D and data.shape[2] == D:
                # Format: (N, D, D) - full complex matrices
                print(f"✅ Processing {filename}: {data.shape[0]} matrices of size {D}x{D}")

                # Convert each matrix to upper triangular vector
                vectors = []
                for i in range(data.shape[0]):
                    vec = complex_matrix_to_upper_triangle_vector(data[i])
                    vectors.append(vec)

                vectors = np.array(vectors)  # Shape: (N, D²)

                if debug and len(vectors) > 0:
                    print("=== Debug: First matrix conversion ===")
                    print("Original matrix shape:", data[0].shape)
                    print("Vector shape:", vectors[0].shape)
                    print("Original matrix[0,0]:", data[0][0,0])
                    print("Vector first element:", vectors[0][0])

                # Convert to tensor and decode back to real/imag matrices
                vec_tensor = torch.from_numpy(vectors).float()
                rho_real, rho_imag = decode_upper_triangle_vector(vec_tensor, D)

                # Save the decoded matrices
                base = os.path.splitext(filename)[0]
                real_file = os.path.join(output_folder, f"{base}_real.npy")
                imag_file = os.path.join(output_folder, f"{base}_imag.npy")

                np.save(real_file, rho_real.cpu().numpy())
                np.save(imag_file, rho_imag.cpu().numpy())

                print(f"✅ Saved {real_file} and {imag_file}")

            elif data.ndim == 2 and data.shape[1] == D * D:
                # Format: (N, D²) - already flattened vectors (original decoder format)
                print(f"✅ Processing flattened vectors: {filename}")
                vec_tensor = torch.from_numpy(data).float()
                rho_real, rho_imag = decode_upper_triangle_vector(vec_tensor, D)

                base = os.path.splitext(filename)[0]
                np.save(os.path.join(output_folder, f"{base}_real.npy"), rho_real.cpu().numpy())
                np.save(os.path.join(output_folder, f"{base}_imag.npy"), rho_imag.cpu().numpy())

            else:
                print(f"❌ Skipping {filename}: shape {data.shape} not compatible with D={D}")
                print(f"   Expected: (N, {D}, {D}) for matrices or (N, {D*D}) for vectors")

        except Exception as e:
            print(f"⚠️ Error processing {filename}: {e}")

def main():
    parser = argparse.ArgumentParser(description="New decoder that handles both full complex matrices and flattened vectors.")
    parser.add_argument("--input", required=True, help="Path to input folder containing .npy files.")
    parser.add_argument("--output", required=True, help="Path to output folder for decoded matrices.")
    parser.add_argument("--dim", type=int, required=True, help="Dimension D of the square density matrix.")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode to print intermediate results.")

    args = parser.parse_args()
    process_folder(args.input, args.output, args.dim, args.debug)

if __name__ == "__main__":
    main()