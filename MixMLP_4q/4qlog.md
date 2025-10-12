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


📊 [4q_d2048_h8_moe] Epoch  9|val   | Loss: 0.030793 | MAE: 0.134837 | R²: -0.0056                   
💪 [4q_d2048_h8_moe] Epoch 10|train | Loss: 0.030519                                                 
📊 [4q_d2048_h8_moe] Epoch 10|val   | Loss: 0.030863 | MAE: 0.131246 | R²: -0.0069                   
💪 [4q_d2048_h8_moe] Epoch 11|train | Loss: 0.030575                                                 
📊 [4q_d2048_h8_moe] Epoch 11|val   | Loss: 0.030660 | MAE: 0.133531 | R²: -0.0010                   
💪 [4q_d2048_h8_moe] Epoch 12|train | Loss: 0.030544                                                 
📊 [4q_d2048_h8_moe] Epoch 12|val   | Loss: 0.030735 | MAE: 0.131619 | R²: -0.0029                   
💪 [4q_d2048_h8_moe] Epoch 13|train | Loss: 0.030530                                                 
📊 [4q_d2048_h8_moe] Epoch 13|val   | Loss: 0.030929 | MAE: 0.131128 | R²: -0.0090                   
💪 [4q_d2048_h8_moe] Epoch 14|train | Loss: 0.030554                                                 
📊 [4q_d2048_h8_moe] Epoch 14|val   | Loss: 0.030638 | MAE: 0.132480 | R²: -0.0001                   
💪 [4q_d2048_h8_moe] Epoch 15|train | Loss: 0.030549                                                 
📊 [4q_d2048_h8_moe] Epoch 15|val   | Loss: 0.030899 | MAE: 0.135579 | R²: -0.0092                   
💪 [4q_d2048_h8_moe] Epoch 16|train | Loss: 0.030555                                                 
📊 [4q_d2048_h8_moe] Epoch 16|val   | Loss: 0.030638 | MAE: 0.132500 | R²: -0.0001                   
💪 [4q_d2048_h8_moe] Epoch 17|train | Loss: 0.030466                                                 
📊 [4q_d2048_h8_moe] Epoch 17|val   | Loss: 0.030977 | MAE: 0.131064 | R²: -0.0105                   
💪 [4q_d2048_h8_moe] Epoch 18|train | Loss: 0.030514                                                 
📊 [4q_d2048_h8_moe] Epoch 18|val   | Loss: 0.030634 | MAE: 0.132787 | R²: -0.0000                   
✅ [4q_d2048_h8_moe] New best model saved! Val Loss: 0.030634
💪 [4q_d2048_h8_moe] Epoch 19|train | Loss: 0.030469                                                 
/home/gwl/MixMLP_4q/train_single.py:372: ConstantInputWarning: An input array is constant; the correlation coefficient is not defined.
  spearman = spearmanr(y_true, y_pred).correlation or 0.0
/home/gwl/MixMLP_4q/train_single.py:376: ConstantInputWarning: An input array is constant; the correlation coefficient is not defined.
  pearson = pearsonr(y_true, y_pred)[0] or 0.0
📊 [4q_d2048_h8_moe] Epoch 19|val   | Loss: 0.030635 | MAE: 0.132610 | R²: -0.0000
💪 [4q_d2048_h8_moe] Epoch 20|train | Loss: 0.030495                                                 
/home/gwl/MixMLP_4q/train_single.py:372: ConstantInputWarning: An input array is constant; the correlation coefficient is not defined.
  spearman = spearmanr(y_true, y_pred).correlation or 0.0
/home/gwl/MixMLP_4q/train_single.py:376: ConstantInputWarning: An input array is constant; the correlation coefficient is not defined.
  pearson = pearsonr(y_true, y_pred)[0] or 0.0
📊 [4q_d2048_h8_moe] Epoch 20|val   | Loss: 0.030683 | MAE: 0.131898 | R²: -0.0014
💪 [4q_d2048_h8_moe] Epoch 21|train | Loss: 0.030511                                                 
📊 [4q_d2048_h8_moe] Epoch 21|val   | Loss: 0.030891 | MAE: 0.131190 | R²: -0.0078                   
💪 [4q_d2048_h8_moe] Epoch 22|train | Loss: 0.030475                                                 
📊 [4q_d2048_h8_moe] Epoch 22|val   | Loss: 0.030952 | MAE: 0.131096 | R²: -0.0097                   
💪 [4q_d2048_h8_moe] Epoch 23|train | Loss: 0.030477                                                 
📊 [4q_d2048_h8_moe] Epoch 23|val   | Loss: 0.030649 | MAE: 0.132252 | R²: -0.0003                   
💪 [4q_d2048_h8_moe] Epoch 24|train | Loss: 0.030417                                                 
/home/gwl/MixMLP_4q/train_single.py:376: NearConstantInputWarning: An input array is nearly constant; the computed correlation coefficient may be inaccurate.
  pearson = pearsonr(y_true, y_pred)[0] or 0.0
📊 [4q_d2048_h8_moe] Epoch 24|val   | Loss: 0.030634 | MAE: 0.132731 | R²: -0.0000
✅ [4q_d2048_h8_moe] New best model saved! Val Loss: 0.030634
💪 [4q_d2048_h8_moe] Epoch 25|train | Loss: 0.030471                                                 
📊 [4q_d2048_h8_moe] Epoch 25|val   | Loss: 0.030637 | MAE: 0.132526 | R²: -0.0000                   
💪 [4q_d2048_h8_moe] Epoch 26|train | Loss: 0.030481                                                 
📊 [4q_d2048_h8_moe] Epoch 26|val   | Loss: 0.030785 | MAE: 0.131442 | R²: -0.0045                   
💪 [4q_d2048_h8_moe] Epoch 27|train | Loss: 0.030446                                                 
📊 [4q_d2048_h8_moe] Epoch 27|val   | Loss: 0.030653 | MAE: 0.132197 | R²: -0.0005                   
💪 [4q_d2048_h8_moe] Epoch 28|train | Loss: 0.030442                                                 
📊 [4q_d2048_h8_moe] Epoch 28|val   | Loss: 0.030652 | MAE: 0.132201 | R²: -0.0004                   
💪 [4q_d2048_h8_moe] Epoch 29|train | Loss: 0.030472                                                 
📊 [4q_d2048_h8_moe] Epoch 29|val   | Loss: 0.030661 | MAE: 0.132093 | R²: -0.0007                   
💪 [4q_d2048_h8_moe] Epoch 30|train | Loss: 0.030395                                                 
📊 [4q_d2048_h8_moe] Epoch 30|val   | Loss: 0.030853 | MAE: 0.131267 | R²: -0.0066                   
💪 [4q_d2048_h8_moe] Epoch 31|train | Loss: 0.030437                                                 
📊 [4q_d2048_h8_moe] Epoch 31|val   | Loss: 0.030649 | MAE: 0.132252 | R²: -0.0003                   
💪 [4q_d2048_h8_moe] Epoch 32|train | Loss: 0.030432                                                 
📊 [4q_d2048_h8_moe] Epoch 32|val   | Loss: 0.030643 | MAE: 0.133203 | R²: -0.0004                   
💪 [4q_d2048_h8_moe] Epoch 33|train | Loss: 0.030438                                                 
/home/gwl/MixMLP_4q/train_single.py:376: NearConstantInputWarning: An input array is nearly constant; the computed correlation coefficient may be inaccurate.
  pearson = pearsonr(y_true, y_pred)[0] or 0.0
📊 [4q_d2048_h8_moe] Epoch 33|val   | Loss: 0.030650 | MAE: 0.132239 | R²: -0.0004
💪 [4q_d2048_h8_moe] Epoch 34|train | Loss: 0.030439                                                 
📊 [4q_d2048_h8_moe] Epoch 34|val   | Loss: 0.030684 | MAE: 0.133845 | R²: -0.0019                   
💪 [4q_d2048_h8_moe] Epoch 35|train | Loss: 0.030448                                                 
📊 [4q_d2048_h8_moe] Epoch 35|val   | Loss: 0.030641 | MAE: 0.132412 | R²: -0.0001                   
💪 [4q_d2048_h8_moe] Epoch 36|train | Loss: 0.030440                                                 
/home/gwl/MixMLP_4q/train_single.py:372: ConstantInputWarning: An input array is constant; the correlation coefficient is not defined.
  spearman = spearmanr(y_true, y_pred).correlation or 0.0
/home/gwl/MixMLP_4q/train_single.py:376: ConstantInputWarning: An input array is constant; the correlation coefficient is not defined.
  pearson = pearsonr(y_true, y_pred)[0] or 0.0
📊 [4q_d2048_h8_moe] Epoch 36|val   | Loss: 0.030654 | MAE: 0.133431 | R²: -0.0008
💪 [4q_d2048_h8_moe] Epoch 37|train | Loss: 0.030426                                                 
📊 [4q_d2048_h8_moe] Epoch 37|val   | Loss: 0.030642 | MAE: 0.133176 | R²: -0.0004                   
💪 [4q_d2048_h8_moe] Epoch 38|train | Loss: 0.049946                                                 
/home/gwl/MixMLP_4q/train_single.py:372: ConstantInputWarning: An input array is constant; the correlation coefficient is not defined.
  spearman = spearmanr(y_true, y_pred).correlation or 0.0
/home/gwl/MixMLP_4q/train_single.py:376: ConstantInputWarning: An input array is constant; the correlation coefficient is not defined.
  pearson = pearsonr(y_true, y_pred)[0] or 0.0
📊 [4q_d2048_h8_moe] Epoch 38|val   | Loss: 0.030872 | MAE: 0.131229 | R²: -0.0072
💪 [4q_d2048_h8_moe] Epoch 39|train | Loss: 0.031215                                                 
/home/gwl/MixMLP_4q/train_single.py:372: ConstantInputWarning: An input array is constant; the correlation coefficient is not defined.
  spearman = spearmanr(y_true, y_pred).correlation or 0.0
/home/gwl/MixMLP_4q/train_single.py:376: ConstantInputWarning: An input array is constant; the correlation coefficient is not defined.
  pearson = pearsonr(y_true, y_pred)[0] or 0.0
📊 [4q_d2048_h8_moe] Epoch 39|val   | Loss: 0.030656 | MAE: 0.132152 | R²: -0.0005
💪 [4q_d2048_h8_moe] Epoch 40|train | Loss: 0.031063                                                 
/home/gwl/MixMLP_4q/train_single.py:372: ConstantInputWarning: An input array is constant; the correlation coefficient is not defined.
  spearman = spearmanr(y_true, y_pred).correlation or 0.0
/home/gwl/MixMLP_4q/train_single.py:376: ConstantInputWarning: An input array is constant; the correlation coefficient is not defined.
  pearson = pearsonr(y_true, y_pred)[0] or 0.0
📊 [4q_d2048_h8_moe] Epoch 40|val   | Loss: 0.030636 | MAE: 0.132928 | R²: -0.0001
💪 [4q_d2048_h8_moe] Epoch 41|train | Loss: 0.031006                                                 
/home/gwl/MixMLP_4q/train_single.py:372: ConstantInputWarning: An input array is constant; the correlation coefficient is not defined.
  spearman = spearmanr(y_true, y_pred).correlation or 0.0
/home/gwl/MixMLP_4q/train_single.py:376: ConstantInputWarning: An input array is constant; the correlation coefficient is not defined.
  pearson = pearsonr(y_true, y_pred)[0] or 0.0
📊 [4q_d2048_h8_moe] Epoch 41|val   | Loss: 0.030699 | MAE: 0.134009 | R²: -0.0024
💪 [4q_d2048_h8_moe] Epoch 42|train | Loss: 0.030980                                                 
/home/gwl/MixMLP_4q/train_single.py:372: ConstantInputWarning: An input array is constant; the correlation coefficient is not defined.
  spearman = spearmanr(y_true, y_pred).correlation or 0.0
/home/gwl/MixMLP_4q/train_single.py:376: ConstantInputWarning: An input array is constant; the correlation coefficient is not defined.
  pearson = pearsonr(y_true, y_pred)[0] or 0.0
📊 [4q_d2048_h8_moe] Epoch 42|val   | Loss: 0.030725 | MAE: 0.131659 | R²: -0.0026
💪 [4q_d2048_h8_moe] Epoch 43|train | Loss: 0.030905                                                 
/home/gwl/MixMLP_4q/train_single.py:372: ConstantInputWarning: An input array is constant; the correlation coefficient is not defined.
  spearman = spearmanr(y_true, y_pred).correlation or 0.0
/home/gwl/MixMLP_4q/train_single.py:376: ConstantInputWarning: An input array is constant; the correlation coefficient is not defined.
  pearson = pearsonr(y_true, y_pred)[0] or 0.0
📊 [4q_d2048_h8_moe] Epoch 43|val   | Loss: 0.030708 | MAE: 0.131747 | R²: -0.0021
💪 [4q_d2048_h8_moe] Epoch 44|train | Loss: 0.030897                                                 
/home/gwl/MixMLP_4q/train_single.py:372: ConstantInputWarning: An input array is constant; the correlation coefficient is not defined.
  spearman = spearmanr(y_true, y_pred).correlation or 0.0
/home/gwl/MixMLP_4q/train_single.py:376: ConstantInputWarning: An input array is constant; the correlation coefficient is not defined.
  pearson = pearsonr(y_true, y_pred)[0] or 0.0
📊 [4q_d2048_h8_moe] Epoch 44|val   | Loss: 0.031129 | MAE: 0.130939 | R²: -0.0153

🛑 [4q_d2048_h8_moe] Early stopping at epoch 44
🔓 Cleaned up GPU lock: /tmp/gpu_locks/gpu_0_4q_d2048_h8_moe.lock

🧪 [4q_d2048_h8_moe] Final Test Evaluation...
/home/gwl/MixMLP_4q/train_single.py:350: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  model.load_state_dict(torch.load(path_manager.model_path)) # Load best model
🔍 Evaluating: 100%|██████████| 79/79 [01:04<00:00,  1.23it/s]
/home/gwl/MixMLP_4q/train_single.py:376: NearConstantInputWarning: An input array is nearly constant; the computed correlation coefficient may be inaccurate.
  pearson = pearsonr(y_true, y_pred)[0] or 0.0
📋 [4q_d2048_h8_moe] Final Test Results:
   MSE: 0.030736
   MAE: 0.131967
   R²:  -0.0002
   Spearman: 0.0196

✅ Experiment 4q_d2048_h8_moe completed successfully!
📁 All outputs saved to: experiments/4q_d2048_h8_moe_20250928_233035
📄 Training log: experiments/4q_d2048_h8_moe_20250928_233035/4q_d2048_h8_moe.csv
📄 Final metrics: experiments/4q_d2048_h8_moe_20250928_233035/4q_d2048_h8_moe_final_metrics.json
🤖 Model weights: models/best_4q_d2048_h8_moe_20250928_233035.pth
⚙️ Configuration: experiments/4q_d2048_h8_moe_20250928_233035/4q_d2048_h8_moe_config.json


(magic) [gwl@cdsw01 MixMLP_4q]$ python train_single.py \
>   --exp_name "4q_d2048_h8_moe" \
>   --num_qubits 4 \
>   --d_model 2048 \
>   --nhead 8 \
>   --pooling_type cls \
>   --mlp_type mixture_of_experts \
>   --use_cls_token True \
>   --epochs 100 \
>   --lr 1e-4 \
>   --batch_size 128 \
>   --data_folder Decoded_Tokens \
>   --labels_path Decoded_Tokens/magic_labels_for_input_for_4_qubits_mixed_1_50000_datapoints.npy \
>   --train_ratio 0.8 \
>   --val_ratio 0.1 \
>   --num_workers 8 \
>   --device auto \
>   --base_log_dir experiments \
>   --base_model_dir models
🔧 PyTorch Version: 2.5.1+cu121
🔧 CUDA Available: True
🔧 CUDA Version: 12.1
🔧 GPU Count: 2
📁 Experiment Paths Created:
   📊 Log: experiments/4q_d2048_h8_moe_20250928_211220/4q_d2048_h8_moe.csv
   📈 Metrics: experiments/4q_d2048_h8_moe_20250928_211220/4q_d2048_h8_moe_final_metrics.json
   🤖 Model: models/best_4q_d2048_h8_moe_20250928_211220.pth
   ⚙️ Config: experiments/4q_d2048_h8_moe_20250928_211220/4q_d2048_h8_moe_config.json
🔧 GPU Selection Summary:
   👉 GPU 0: 0.0% used, 47.6GB free
      GPU 1: 0.0% used, 47.6GB free
✅ Auto-selected: cuda:0
🔒 Created GPU lock: /tmp/gpu_locks/gpu_0_4q_d2048_h8_moe.lock
🔧 Experiment: 4q_d2048_h8_moe
🔧 Configuration:
   Qubits: 4
   d_model: 2048
   Pooling: cls
   MLP: mixture_of_experts
   Device: cuda:0
   Dummy test mode: False
📥 Loading real data...
📊 Found 1 real files, 1 imag files, 1 label files
   Matched: input_for_4_qubits_mixed_1_50000_datapoints → magic_labels_for_input_for_4_qubits_mixed_1_50000_datapoints.npy
📊 Found 1 complete file triplets
✅ Validation passed: 50000 samples
📊 Split: Train=40000, Val=5000, Test=5000
🏗️ Creating model...
🔧 Model Config:
   Pooling: cls, Use CLS: True
   MLP: mixture_of_experts
✅ Experiment config saved: experiments/4q_d2048_h8_moe_20250928_211220/4q_d2048_h8_moe_config.json

🚀 Starting training for 4q_d2048_h8_moe
   Model: CompleteQuantumMagicPredictor
   Device: cuda:0
   Epochs: 100, LR: 0.0001
   PyTorch: 2.5.1+cu121
🐛 Debugging model shapes with a sample batch...

