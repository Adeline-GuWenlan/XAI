# peek_data.py
import numpy as np
file = np.load('/Users/guwenlan/Desktop/XAI/Utils/Data/input_for_4_qubits_mixed_1_50000_datapoints.npy')
print(file[0].shape)