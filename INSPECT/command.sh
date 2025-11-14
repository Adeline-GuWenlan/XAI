python '/Users/guwenlan/Desktop/INSPECT/DataGeneration/RoM_label/RoM_poisson.py' --num 200 --qubits 2 --lam 1.5 --output '/Users/guwenlan/Desktop/INSPECT/Data' --seed 114514
python '/Users/guwenlan/Desktop/INSPECT/DataGeneration/RoM_label/RoM_poisson.py' --num 1000 --qubits 2 --lam 31 --output '/Users/guwenlan/Desktop/INSPECT/Data' --seed 114514
python '/Users/guwenlan/Desktop/XAI/INSPECT/DataGeneration/Expectation/generate_complete_dataset.py' -n 2 --num 50000 -o "/Users/guwenlan/Desktop/XAI/Full_Ori"

python '/Users/guwenlan/Desktop/XAI/INSPECT/DataGeneration/Expectation/generate_complete_dataset.py' -n 2 --num 100000 -o /Users/guwenlan/Desktop/XAI/Full_Ori/2q

python '/Users/guwenlan/Desktop/XAI/INSPECT/DataGeneration/Expectation/generate_complete_dataset.py' -n 3 --num 100000 -o /Users/guwenlan/Desktop/XAI/Full_Ori/3q
python '/Users/guwenlan/Desktop/XAI/INSPECT/DataGeneration/Expectation/generate_complete_dataset.py' -n 4 --num 150000 -o /Users/guwenlan/Desktop/XAI/Full_Ori/4q
python '/Users/guwenlan/Desktop/XAI/INSPECT/DataGeneration/Expectation/generate_complete_dataset.py' -n 5 --num 200000 -o /Users/guwenlan/Desktop/XAI/Full_Ori/5q