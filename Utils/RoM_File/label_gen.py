import numpy as np
import os
import argparse
from scipy.optimize import linprog
# ========================================================
# Import functions from rom_function.ipynb
# ========================================================
# helper functions
import numpy as np
from scipy.optimize import linprog
from random import uniform
si = np.array([[1,0], [0,1]])
sx = np.array([[0,1] ,[1,0]])
sy = np.array([[0,-1j], [1j,0]])
sz = np.array([[1,0], [0,-1]])

pauli = [si, sx, sy, sz]

def numberToBase(n, b):
    if n == 0:
        return [0]
    digits = []
    while n:
        digits.append(int(n % b))
        n //= b
    return digits[::-1]

f = open("Amat2.txt", "r")
amat2 = []
for line in f:
    amat2.append(line.replace('{', '').replace('}', '').split(", "))
amat2 = np.array(amat2).astype(int)
f.close()

def l1_minimization(A, b):
    m, n = A.shape

    # Variables are (u, v), total 2n
    c = np.ones(2*n)  # objective coefficients: sum(u+v)

    # Constraint: A_n u - A_n v = b
    A_eq = np.hstack([A, -A])
    b_eq = b

    # Bounds: u, v >= 0
    bounds = [(0, None)] * (2*n)

    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")

    if res.success:
        return res.fun
    else:
        raise RuntimeError("LP failed: " + res.message)

# robustness of magic (density matrix -> real number)
# 类似basis？
def get_rom(rho, amat=amat2):

    p = int(np.log2(np.size(rho[0])))
    dim = 2**p
    dim2 = 4**p
    sdim = amat[0].size
    
    explist = []
    for nu in range(dim**2):
        ntb = numberToBase(nu, 4)
        op_nu = np.pad(ntb, (p-len(ntb), 0), 'constant')
        op = [[1]]
        for i in range(p):
            op = np.kron(op,pauli[op_nu[i]])
        explist.append(np.trace(np.dot(rho,op)))
    bvec = explist
    
    return l1_minimization(amat, bvec)

def main():
    parser = argparse.ArgumentParser(description='Generate labels for quantum states')
    parser.add_argument('--file_path', type=str, default='Decoded_Tokens',
                       help='Path to directory containing real and imaginary state files')
    args = parser.parse_args()

    # Get all .npy files in the directory
    npy_files = []
    for filename in os.listdir(args.file_path):
        if filename.endswith(".npy"):
            print(filename)
            npy_files.append(filename)

    # Look for real/imag pairs
    real_files = [f for f in npy_files if 'real' in f.lower()]
    imag_files = [f for f in npy_files if 'imag' in f.lower()]

    if not real_files or not imag_files:
        print("No real/imag pairs found in directory")
        return

    # Process each real/imag pair
    for real_file in real_files:
        # Find corresponding imag file
        base_name = real_file.replace('_real.npy', '').replace('real.npy', '')
        imag_file = None

        for imag in imag_files:
            if base_name in imag and 'imag' in imag.lower():
                imag_file = imag
                break

        if imag_file is None:
            print(f"No corresponding imaginary file found for {real_file}")
            continue

        print(f"Processing pair: {real_file} and {imag_file}")

        # Load the data
        states_real = np.load(os.path.join(args.file_path, real_file), mmap_mode="r")
        states_imag = np.load(os.path.join(args.file_path, imag_file), mmap_mode="r")

        # Combine real and imaginary parts
        states_rhos = states_real + 1j * states_imag

        # Generate ROM labels
        roms = []
        N = len(states_rhos)
        c = 0
        checks = set(np.linspace(0, N - 1, 100, dtype=int))

        for state_rho in states_rhos:
            if c in checks:
                print(f'{c}/{N}')
            c += 1
            rom = get_rom(state_rho)
            roms.append(rom)

        # Save the labels
        output_filename = f"{base_name}_rom_labels.npy"
        output_path = os.path.join(args.file_path, output_filename)
        np.save(output_path, roms)
        print(f"Saved ROM labels to {output_path}")

if __name__ == "__main__":
    main()