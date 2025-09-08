#!/bin/bash
#SBATCH --job-name=competitive_moe
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --output=competitive_moe_%j.log
#SBATCH --error=competitive_moe_%j.err

# Activate conda environment
source ~/.bashrc
conda activate competitive_moe

# Change to script directory
cd MLP_MoE/Competetive

# Run competitive training
python train_moe_competitive.py \
  --init_ckpt ../../CONFIGs/2_128_cls_moe_20250817_202509/best_2_128_cls_moe_20250817_202509.pth \
  --out_dir ./competitive_checkpoints \
  --epochs 50 \
  --lr 3e-4 \
  --bs 512 \
  --tau_start 1.8 \
  --tau_end 1.0 \
  --tau_warmup 15 \
  --topk_start 20 \
  --topk_target 1 \
  --lb_lambda 0.01 \
  --lik gauss \
  --sigma 0.1 \
  --clip_grad 1.0

echo "Competitive MoE training completed!"

# Run final evaluation
python eval_moe_competitive.py \
  --ckpt ./competitive_checkpoints/moe_competitive_$(printf "%08d" 50).pt \
  --bs 512

echo "Evaluation completed!"