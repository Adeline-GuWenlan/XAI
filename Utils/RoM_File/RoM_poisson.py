import numpy as np
import os
import argparse
from scipy.optimize import linprog


# ========================================================
# generate rho
# ========================================================
import random
import numpy as np
# Sep. 15 
# UNCHANGED

def random_pure_state(D: int) -> np.ndarray:
    psi = np.random.randn(D) + 1j * np.random.randn(D)
    psi /= np.linalg.norm(psi)
    return np.outer(psi, psi.conj())

# Sep.15 Update 2 qubits
# start with lam == 1
def random_mixed_state_poisson(D, lam=1.0):
    # lam == hyper para. args input
    """
    K ~ Poisson(lam) + 1 
    """
    K = np.random.poisson(lam) + 1
    # To make sure that we at least have one to mix with
    # K = selecting how much we want to mix
    rho = np.zeros((D, D), dtype=np.complex128)
    # Empty rho
    for _ in range(K):
        rho += random_pure_state(D)
    rho /= K
    # here we giving all pure states equal weights;
    # which can be modified later to give more weoghts to pure states on certain direction?               
    rho /= np.trace(rho)
    return rho


# ========================================================
# rom_function.ipynb
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


def generate_dataset(num_samples, qubits, lam, output_path=None, seed=None):
    """
    Generate dataset of quantum states and their RoM labels.
    
    Args:
        num_samples (int): Number of samples to generate
        qubits (int): Number of qubits (determines matrix dimension as 2^qubits)
        output_path (str): Path to save .npy files (optional)
        seed (int): Random seed for reproducibility (optional)
    
    Returns:
        tuple: (states, labels) - numpy arrays of quantum states and their labels
    """
    if seed is not None:
        np.random.seed(seed)
    
    D = 2 ** qubits
    
    print(f"Generating {num_samples} samples for {qubits} qubits (dimension {D}x{D})")
    
    states = []
    labels = []
    
    for i in range(num_samples):
        if (i + 1) % 1000 == 0:
            print(f"Progress: {i + 1}/{num_samples}")
        
        # Generate random mixed state
        rho = random_mixed_state_poisson(D,lam)
                
        # Normalize trace to exactly 1 (good practice)
        rho = rho / np.trace(rho)
        
        # Compute stabilizer norm label
        RoM_label = get_rom(rho)
        
        states.append(rho)
        labels.append(RoM_label)
    
    states = np.array(states)
    labels = np.array(labels)
    
    print(f"Dataset generation completed!")
    print(f"States shape: {states.shape}")
    print(f"Labels shape: {labels.shape}")
    
    # Save to .npy files if output path provided
    if output_path:
        os.makedirs(output_path, exist_ok=True)
        states_file = os.path.join(output_path, f"{qubits}q_{num_samples}_states.npy")
        labels_file = os.path.join(output_path, f"{qubits}q_{num_samples}_states_labels.npy")
        
        np.save(states_file, states)
        np.save(labels_file, labels)
        
        print(f"Saved states to: {states_file}")
        print(f"Saved labels to: {labels_file}")
    
    return states, labels


def analyze_distribution(labels):
    """
    Analyze the distribution of stabilizer norm labels.
    """
    print("\n=== Label Distribution Analysis ===")
    print(f"Total samples: {len(labels)}")
    print(f"Label range: [{np.min(labels):.6f}, {np.max(labels):.6f}]")
    print(f"Mean: {np.mean(labels):.6f}")
    print(f"Std: {np.std(labels):.6f}")
    # standard deviation
    # Count above/below 1
    below_1 = np.sum(labels < 1.0)
    above_1 = np.sum(labels >= 1.0)
    
    print(f"Labels < 1.0: {below_1} ({below_1/len(labels)*100:.2f}%)")
    print(f"Labels >= 1.0: {above_1} ({above_1/len(labels)*100:.2f}%)")

def main():
    parser = argparse.ArgumentParser(description='Generate quantum state dataset with stabilizer norm labels')
    parser.add_argument('--num', type=int, required=True, help='Number of samples to generate')
    parser.add_argument('--qubits', type=int, required=True, help='Number of qubits')
    parser.add_argument('--lam', type=float, help='lam in poisson')
    parser.add_argument('--output', type=str, help='Output directory to save .npy files')
    parser.add_argument('--seed', type=int, help='Random seed for reproducibility')
    
    
    args = parser.parse_args()
    
    # Generate dataset
    states, labels = generate_dataset(args.num, args.qubits, args.lam, args.output, args.seed)
    
    # Analyze distribution
    analyze_distribution(labels)
if __name__ == "__main__":
    main()

