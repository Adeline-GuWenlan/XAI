### dataset
1. load data from /Users/guwenlan/Desktop/XAI/INSPECT/Data - pick those with same timestamp
### Model archi
Remove all dropout layers. (for now )
2. for model 1: - Linear regression
   1. load /Users/guwenlan/Desktop/XAI/INSPECT/Data/2q_1000samples_poisson_lam1.1_20251111_163332_pauli_vectors.npy as X, /Users/guwenlan/Desktop/XAI/INSPECT/Data/2q_1000samples_poisson_lam1.1_20251111_163332_sn_labels.npy as Y
   2. build a linear regressor for it training
3. Model 2: - MLP with Tanh (symmetric) + ReLU
   1. load /Users/guwenlan/Desktop/XAI/INSPECT/Data/2q_1000samples_poisson_lam1.1_20251111_163332_pauli_vectors.npy as X, /Users/guwenlan/Desktop/XAI/INSPECT/Data/2q_1000samples_poisson_lam1.1_20251111_163332_sn_labels.npy as Y
   2. build nn.Sequential{mlp+activation+mlp+activation+drop+output} to fit. the first activation must be one symmetric acti vation func with exsiting pytorch api the second one Relu
4. Model 3:  - MLP with custom Abs activation + ReLU (overfit)
   1. load /Users/guwenlan/Desktop/XAI/INSPECT/Data/2q_1000samples_poisson_lam1.1_20251111_163332_pauli_vectors.npy as X, /Users/guwenlan/Desktop/XAI/INSPECT/Data/2q_1000samples_poisson_lam1.1_20251111_163332_sn_labels.npy as Y
   2. build nn.Sequential{mlp+activation+mlp+activation+drop+output} to fit. the first activation must be designed as takiing the absoluate value of vevery input. the secong one Relu
5. Model 4:  - MLP with ReLU + ReLU (best performing)
   1. load /Users/guwenlan/Desktop/XAI/INSPECT/Data/2q_1000samples_poisson_lam1.1_20251111_163332_pauli_vectors.npy as X, /Users/guwenlan/Desktop/XAI/INSPECT/Data/2q_1000samples_poisson_lam1.1_20251111_163332_sn_labels.npy as Y
   2. build nn.Sequential{mlp+activation+mlp+activation+drop+output} to fit. both activation func should be Relu
### Todo
for all model, create a subfolder in /Users/guwenlan/Desktop/XAI/AbsValue, build their dataset+model def+train loop with MSE eval all in one file. save the output, log, result summary, model para log all in a subfolder parallel to the model. 


1. log the training process through training, each epoch a line with error
2. update the data path to 
3. for each training, give a timestamp for the result folder



DONE: Nov. 11

Still 4 give the best performance w/ 
Epoch 132/200: Train Loss = 0.000012, Val Loss = 0.000000


Relu better then Abs activation + ReLU. Why? - not smooth enoough for gradient dscent


What if we put just the abs values? Linear model would be enough
- still with abs activation VS symmetric avativation vs Relu
And put the abs values + density
Density + abs value 
Setting dropput to 0 would help: which i should ve foreseen ealier. since it wont fit to noise
 - then how come still in some archi tranin loss << val loss and other in reverse???


Based on the given data in "/Users/guwenlan/Desktop/XAI/INSPECT/Data/2q_50000samples_poisson_lam1.1_20251111_180142_full_matrices.npy", use the "def calculate_RoM_for_2qubits(rho: np.ndarray) -> float:
" deifned in "RoMHandbook/RoM_handbook/generate_balanced_magic_dataset.py" to compute all the RoM label of the given matricx and store under /Users/guwenlan/Desktop/XAI/INSPECT/Data as a .npy, with naming style following "/Users/guwenlan/Desktop/XAI/INSPECT/Data/2q_50000samples_poisson_lam1.1_20251111_180142_RoM.npy"


/home/gwl/compute_rom_only/Bal_Rom
### OH HPC

python generate_balanced_magic_dataset.py --num 1000 --output /home/gwl/compute_rom_only/Bal_Rom --lam_magic 40 --seed 114
python generate_balanced_magic_dataset.py --num 100000 --output /home/gwl/compute_rom_only/Bal_Rom --lam_magic 40 --seed 114


Dataset to use
Saved files:
  States: /home/gwl/compute_rom_only/Bal_Rom/2q_100000_balanced_states.npy
  Labels: /home/gwl/compute_rom_only/Bal_Rom/2q_100000_balanced_labels.npy
  RoM values: /home/gwl/compute_rom_only/Bal_Rom/2q_100000_balanced_rom_values.npy
Next steps:
upload a Pauli exp computation, get Pauli vec based on States
comb out a pipeline for training BERT rn. 


