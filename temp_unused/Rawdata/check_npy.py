import numpy as np

# Load your npy file
labels = np.load("Rawdata/label/magic_labels_for_input_for_2_qubits_mixed_2_200000_datapoints.npy", allow_pickle=True)

# Overall shape
print("Full array shape:", labels.shape)

# Data type
print("Dtype:", labels.dtype)

# Peek at the first few entries
print("First 5 entries:", labels[:5])

# If it’s a structured array or object array, drill down
if labels.dtype == object:
    for i, row in enumerate(labels[:5]):
        print(f"Row {i} type: {type(row)}, shape: {getattr(row, 'shape', None)}")
        print(row)

# If it’s a numeric ndarray, check each row shape explicitly
for i in range(min(5, labels.shape[0])):
    try:
        print(f"Row {i} shape: {labels[i].shape}")
    except AttributeError:
        print(f"Row {i} is scalar:", labels[i])
