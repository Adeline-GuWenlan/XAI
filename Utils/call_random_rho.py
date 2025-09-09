# call_random_rho.py - for usingt the random_mixed_state function
import argparse
from random_mix import random_mixed_state, validate_rho
import numpy as np

parser = argparse.ArgumentParser(
    description="Generate random mixed-state density matrices")
parser.add_argument("-n", "--num",      type=int, default=3,
                    help="how many matrices to generate")
parser.add_argument("-q", "--qubits",   type=int, default=2,
                    help="number of qubits (matrix dim = 2**q)")

args = parser.parse_args()

dim = 2 ** args.qubits

for idx in range(args.num):
    rho = random_mixed_state(D=dim)

    ok, msg = validate_rho(rho)
    print(f"\n=== #{idx+1} | {args.qubits}-qubit ({dim}×{dim}) | {msg} ===")
    with np.printoptions(precision=3, suppress=True):
        print(rho)
