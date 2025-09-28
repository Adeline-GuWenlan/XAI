(base) guwenlan@guwenlans-MacBook-Pro XAI % python /Users/guwenlan/Desktop/XAI/Utils/larger_generation.py --num 1000 --qubits 4 --lam 10 --output /Users/guwenlan/Desktop/XAI/Dataset --seed 114514
Starting dataset generation for 4 qubits...
Generating 1000 samples for 4 qubits (dimension 16x16)
Poisson parameter lambda = 10.0
Total Pauli operators: 256
Non-identity Pauli operators: 255
Progress: 100/1000
Progress: 200/1000
Progress: 300/1000
Progress: 400/1000
Progress: 500/1000
Progress: 600/1000
Progress: 700/1000
Progress: 800/1000
Progress: 900/1000
Progress: 1000/1000
Dataset generation completed!
States shape: (1000, 16, 16)
Labels shape: (1000,)
Saved states to: /Users/guwenlan/Desktop/XAI/Dataset/4q_1000_states.npy
Saved labels to: /Users/guwenlan/Desktop/XAI/Dataset/4q_1000_labels.npy

=== Label Distribution Analysis ===
Total samples: 1000
Label range: [0.626158, 1.714495]
Mean: 1.026380
Std: 0.167044
Non-magic threshold (universal): 1.000000
Labels <= 1.000 (non-magic): 508 (50.80%)
Labels > 1.000 (magic): 492 (49.20%)

Percentiles:
  25th percentile: 0.904471
  50th percentile: 0.998290
  75th percentile: 1.132549
  90th percentile: 1.258496
  95th percentile: 1.318116
  99th percentile: 1.494467


