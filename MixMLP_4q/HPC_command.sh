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
            scp -r /Users/guwenlan/Desktop/XAI/MixMLP/Decoded_Tokens\
            gwl@172.16.51.235:/home/gwl/MixMLP
          
          # ---------------------- 单独文件 ----------------------
            scp /Users/guwenlan/Desktop/XAI/MixMLP_4q/train_parallel.py gwl@172.16.51.235:/home/gwl/MixMLP_4q
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



scp -r /Users/guwenlan/Desktop/XAI/MixMLP/Decoded_Tokens
##############################################################
# ——------------------ Python us!! ——------------------
##############################################################
tmux new -s "4q_baseline"
python train_single.py \
  --exp_name "4q_d2048_h8_moe" \
  --num_qubits 4 \
  --d_model 2048 \
  --nhead 8 \
  --pooling_type cls \
  --mlp_type mixture_of_experts \
  --use_cls_token True \
  --epochs 100 \
  --lr 1e-4 \
  --batch_size 32 \
  --data_folder Decoded_Tokens \
  --labels_path Decoded_Tokens/magic_labels_for_input_for_4_qubits_mixed_1_50000_datapoints.npy \
  --train_ratio 0.8 \
  --val_ratio 0.1 \
  --num_workers 8 \
  --device auto \
  --base_log_dir experiments \
  --base_model_dir models

##############################################################
# NEW: 5-Pathway Asymmetric Ensemble V2 (for d_model >= 1024)
##############################################################
tmux new -s "4q_asym_v2"
python train_single.py \
  --exp_name "4q_d2048_h8_asym_v2" \
  --num_qubits 4 \
  --d_model 2048 \
  --nhead 8 \
  --pooling_type cls \
  --mlp_type asymmetric_ensemble_v2 \
  --use_cls_token True \
  --epochs 100 \
  --lr 1e-4 \
  --batch_size 16 \
  --data_folder Decoded_Tokens \
  --labels_path Decoded_Tokens/4q_50000_states_labels.npy \
  --train_ratio 0.8 \
  --val_ratio 0.1 \
  --num_workers 8 \
  --device auto \
  --base_log_dir experiments \
  --base_model_dir models


rsync -avz --progress /Users/guwenlan/Desktop/XAI/MixMLP_4q/mlp.py /Users/guwenlan/Desktop/XAI/MixMLP_4q/train_single.py /Users/guwenlan/Desktop/XAI/MixMLP_4q/updated_training_model.py gwl@172.16.51.235:/home/gwl/MixMLP_4q/
