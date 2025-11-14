#HPC_command.sh
ssh gwl@172.16.51.235
git reflog
128_Attention_attention_enhanced_20250815_120257
##############################################################
# ---------------------- 上 传 ----------------------
##############################################################
      # ----------------------语法----------------------
      scp -r /path/to/local_dir username@remote.host:/path/to/remote_dir
      rsync -av --exclude='.git' Decoded_Tokens gwl@172.16.51.235:/home/gwl/3q
          #  ----------------------批量----------------------
            # 更新：cab20fd - cab20fd (HEAD -> 7.31whther_attention+Mask) HEAD@{2}: commit: Add experiment_manager.sh
            scp -r /Users/guwenlan/Desktop/MixMLP/Decoded_Tokens\
            gwl@172.16.51.235:/home/gwl/MixMLP
          
          # ---------------------- 单独文件 ----------------------
            scp /Users/guwenlan/Desktop/MixMLP/train_single.py gwl@172.16.51.235:/home/gwl/MixMLP
scp EnhancedTransformer/train_enhanced.py gwl@172.16.51.235:/home/gwl/EnhancedTransformer


##############################################################
# ---------------------- 下 载 ----------------------
##############################################################
scp -r gwl@172.16.51.235:/home/gwl/3q/MixMLP/experiments /Users/guwenlan/Desktop
scp gwl@172.16.51.235:/home/gwl/NewTrans_MLP/models/best_2_128_cls_moe_20250817_202509.pth /Users/guwenlan/Desktop
cat logs/stdout/Msk_CLS_PA.log
cat NewTrans_MLP/experiments/CLS_Stand_20250808_145448/CLS_Stand.csv
##############################################################
# ----------------------TMUX----------------------
##############################################################
scp -r gwl@172.16.51.235:/home/gwl/EnhancedTransformer/experiments /Users/guwenlan/Desktop/experiments/Deepened
tmux attach -t WhichMLP
/home/gwl/DURF
/home/gwl/DURF/NewTrans_Try_MLP/models

watch -n 0.1 nvidia-smi

##############################################################
# ——------------------ Python us!! ——------------------
##############################################################


##############################################################
# ——------------------ past run ——------------------
##############################################################

# batch run

# check error and log
cat logs/stdout/*.log
cat logs/stderr/*.log
cat /home/gwl/DURF/Visual/7.30/logs/Baseline_CLS_MixtureExperts.csv
# 可视化
# finetune

(base) [gwl@cdsw01 NewTrans_MLP]$ ls -lh models
total 8.8M
-rw-rw-r-- 1 gwl gwl 1.6M Aug  8 20:17 best_CLS_Stand_20250808_145448.pth
-rw-rw-r-- 1 gwl gwl 1.6M Aug  8 20:22 best_Formal_20250808_151844.pth
-rw-rw-r-- 1 gwl gwl 2.6M Aug  9 01:35 best_Msk_CLS_AE_20250809_012527.pth
-rw-rw-r-- 1 gwl gwl 1.7M Aug  9 08:43 best_Msk_cls_AE_20250809_013105.pth
python finetune_robust.py \
 --pretrained_model /home/gwl/NewTrans_MLP/models/best_Msk_CLS_moe_20250809_012529.pth \
 --epochs 200 \
 --gpu_id 0\
 --exp_name Msk_CLS_moe_20250809_012529_finetune\
 --mlp_type mixture_of_experts\
 --use_physics_mask True\
 --mask_threshold 2
# latest model
/home/gwl/DURF/NewTrans_Try_MLP/models/best_robust_finetune_20250805_203810.pth


python train_single.py --exp_name 2_128_cls_moe --d_model 128 --pooling_type cls --mlp_type mixture_of_experts --nhead 8 --epochs 200 --batch_size 512 --num_workers 8


python decoder.py --input /Users/guwenlan/Desktop/MixMLP/Decoded_Tokens --output Decoded_Tokens --dim 8






cd Decoded_Tokens && mv input_for_3_qubits_mixed_10000_datapoints_imag.npy InputFor3QubitsMixed10000DatapointsImag.npy 
mv input_for_3_qubits_mixed_10000_datapoints_real.npy InputFor3QubitsMixed10000DatapointsReal.npy



  mv "Label/MagicLabelsForInputFor3QubitsMixed10000Datapoints.npy" "Label/magic_labels_for_input_for_3_qubits_mixed_10000_datapoints.npy"
scp /Users/guwenlan/Desktop/MixMLP/batch_runner_tmux.py gwl@172.16.51.235:/home/gwl/3q/MixMLP
  
conda activate magic 

  python train_simple.py --exp_name 512_6lay_8head --num_qubits 3 --d_model 512 --num_layers 6\
  --data_folder /home/gwl/3q/MixMLP/Decoded_Tokens --labels_path /home/gwl/3q/MixMLP/Label/magic_labels_for_input_for_3_qubits_mixed_10000_datapoints.npy\
  --loss_type smoothl1

scp transformer_embedding.py gwl@172.16.51.235:/home/gwl/EnhancedTransformer


  python train_single.py \
    --exp_name "128_2q_SN_balanced" \
    --num_qubits 2 \
    --d_model 128 \
    --pooling_type $POOLING_TYPE \
    --mlp_type $MLP_TYPE \
    --use_cls_token $USE_CLS_TOKEN \
    --nhead $NHEAD \
    --epochs $EPOCHS \
    --lr $LEARNING_RATE \
    --batch_size $BATCH_SIZE \
    --data_folder "$DATA_FOLDER" \
    --labels_path "$LABELS_PATH" \
    --train_ratio 0.8 \
    --val_ratio 0.1 \
    --num_workers $NUM_WORKERS \
    --device $DEVICE \
    --base_log_dir "experiments" \
    --base_model_dir "models"


python train_single.py --exp_name "128_2q_SN_balanced" --num_qubits 2 --d_model 128 --pooling_type cls --mlp_type mixture_of_experts\
  --use_cls_token True --nhead 8 --epochs 100 --lr 1e-4 --batch_size 512 --data_folder "/home/gwl/quantum_archi/Decoded_Tokens"\
  --labels_path "/home/gwl/quantum_archi/Decoded_Tokens" --train_ratio 0.8 --val_ratio 0.1 --num_workers 8 --device auto --base_log_dir\
  "experiments" --base_model_dir "models"


  python -c \" from quantum_dataset import QuantumDataset \
  import torch\ 
  dataset = QuantumDataset(
      data_folder='Decoded_Tokens',
      labels_path='Decoded_Tokens',
      validate=False
  )\
  print('Dataset length:', len(dataset))
  rho_real, rho_imag, label = dataset[0]
  print('Sample 0 shapes:')
  print('  rho_real:', rho_real.shape)
  print('  rho_imag:', rho_imag.shape)
  print('  label:', label.shape, '← This should be (1,) not (1, 4, 4)')
  print('  label value:', label)
  \"



python 
python generate_complete_dataset.py -n 2 -s 50000 -o /Users/guwenlan/Desktop/XAI/INSPECT/Data --method poisson --lam 1.1 --seed 114514 


python /Users/guwenlan/Desktop/XAI/RoMHandbook/RoM_handbook/generate_balanced_magic_dataset.py --num 1000 --output "/Users/guwenlan/Desktop/XAI/INSPECT/Architec/TESTWITHRoM" --lam_magic 1.1 --seed 114514

 /Users/guwenlan/Desktop/XAI/RoMHandbook/RoM_handbook/generate_balanced_magic_dataset.py --num 1000 --output "/Users/guwenlan/Desktop/XAI/INSPECT/Architec/TESTWITHRoM" --lam_magic 40 --seed 114514
python /Users/guwenlan/Desktop/XAI/INSPECT/DataGeneration/Expectation/generate_complete_dataset.py --n_qubits 3 --num_samples 10000 --method poisson --lam 1 --seed 114