import os
import argparse
import numpy as np
import torch
from tqdm import tqdm
from typing import Tuple
## for debug and test case. 
def vec_to_rho(vec: np.ndarray) -> np.ndarray:
    """
    反展开：把向量还原成 Hermitian 矩阵。
    """
    # 通过方程 D^2 = len(vec) 求 D
    D = int(np.sqrt(len(vec)))
    if D * D != len(vec):
        raise ValueError("Length of vec must be a perfect square.")
    rho = np.zeros((D, D), dtype=np.complex128)
    k = 0
    for i in range(D):
        rho[i, i] = vec[k]
        k += 1
        for j in range(i + 1, D):
            rho[i, j] = vec[k] + 1j * vec[k + 1]
            rho[j, i] = rho[i, j].conj()
            k += 2
    return rho


def decode_upper_triangle_vector(vec: torch.Tensor, D: int) -> Tuple[torch.Tensor, torch.Tensor]:
    B = vec.shape[0]
    device = vec.device

    rho_real = torch.zeros((B, D, D), dtype=torch.float32, device=device)
    rho_imag = torch.zeros((B, D, D), dtype=torch.float32, device=device)

    k = 0
    for i in range(D):
        rho_real[:, i, i] = vec[:, k]
        k += 1
        for j in range(i + 1, D):
            real = vec[:, k]
            imag = vec[:, k + 1]
            rho_real[:, i, j] = real
            rho_real[:, j, i] = real
            rho_imag[:, i, j] = imag
            rho_imag[:, j, i] = -imag
            k += 2

    return rho_real, rho_imag

def process_folder(input_folder: str, output_folder: str, D: int, debug=False):
    os.makedirs(output_folder, exist_ok=True)
    files = [f for f in os.listdir(input_folder) if f.endswith(".npy")]

    for filename in tqdm(files, desc="Decoding vectors"):
        path = os.path.join(input_folder, filename)
        try:
            vec = np.load(path)
            print(vec.shape, vec.dtype)
            if vec.ndim != 2 or vec.shape[1] != D * D:
                print(f"❌ Skipping {filename}: shape {vec.shape} incompatible with D={D}")
                continue
            '''vec_tensor = torch.tensor(vec, dtype=torch.float32)'''
            vec_tensor = torch.from_numpy(vec).float()
            if debug:
                    print("=== Raw vector[0] ===")
                    print(vec_tensor[0])                     # first sample
                    rho_real, rho_imag = decode_upper_triangle_vector(vec_tensor, D)
                    ref = vec_to_rho(vec_tensor[0].numpy())
                    print("=== Reference Hermitian matrix ===")
                    print("ref (shape):", ref.shape)
                    print("ref (full):\n", ref)
                    print("ours (real):\n", rho_real[0])
                    print("ours (imag):\n", rho_imag[0])

            rho_real, rho_imag = decode_upper_triangle_vector(vec_tensor, D)
            base = os.path.splitext(filename)[0]

            np.save(os.path.join(output_folder, f"{base}_real.npy"), rho_real.cpu().numpy())
            np.save(os.path.join(output_folder, f"{base}_imag.npy"), rho_imag.cpu().numpy())
        except Exception as e:
            print(f"⚠️ Error processing {filename}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Decode upper triangle vectors into Hermitian matrices.")
    parser.add_argument("--input", required=True, help="Path to input folder containing .npy vector files.")
    parser.add_argument("--output", required=True, help="Path to output folder for decoded matrices.")
    parser.add_argument("--dim", type=int, required=True, help="Dimension D of the square density matrix.")
    parser.add_argument("--debug", type=bool, help="Enable debug mode to print intermediate results.")

    args = parser.parse_args()
    process_folder(args.input, args.output, args.dim, args.debug)

if __name__ == "__main__":
    main()



