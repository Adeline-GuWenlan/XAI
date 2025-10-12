import numpy as np
import os

# Load the numpy files
decoded_tokens_dir = "/Users/guwenlan/Desktop/XAI/MixMLP/Decoded_Tokens/"

# Load the data
real_data = np.load(os.path.join(decoded_tokens_dir, "2q_100000_states_real.npy"))
imag_data = np.load(os.path.join(decoded_tokens_dir, "2q_100000_states_imag.npy"))

# Take the first matrix as example
sample_real = real_data[0]
sample_imag = imag_data[0]
sample_complex = sample_real + 1j * sample_imag

print("Original 4x4 matrix (real part):")
print(sample_real)
print(f"Shape: {sample_real.shape}")

print("\nOriginal 4x4 matrix (imaginary part):")
print(sample_imag)

print("\nOriginal 4x4 complex matrix:")
print(sample_complex)

print("\nFlattened matrix (default numpy order - row-major/C-style):")
flattened = sample_complex.flatten()
print(flattened)
print(f"Length: {len(flattened)}")

print("\nElement mapping (row, col) -> flattened index:")
for i in range(4):
    for j in range(4):
        flat_index = i * 4 + j
        print(f"({i},{j}) -> index {flat_index}: {sample_complex[i,j]}")

print("\nSequence explanation:")
print("Row 0: indices 0-3   -> (0,0), (0,1), (0,2), (0,3)")
print("Row 1: indices 4-7   -> (1,0), (1,1), (1,2), (1,3)")
print("Row 2: indices 8-11  -> (2,0), (2,1), (2,2), (2,3)")
print("Row 3: indices 12-15 -> (3,0), (3,1), (3,2), (3,3)")