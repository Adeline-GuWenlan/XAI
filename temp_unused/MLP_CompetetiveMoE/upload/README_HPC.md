# Competitive MoE Training - HPC Deployment

## Directory Structure
```
upload/
├── MLP_MoE/Competetive/          # Competitive training scripts
│   ├── competitive_loss.py       # Competitive loss functions
│   ├── train_moe_competitive.py  # Main training script
│   ├── eval_moe_competitive.py   # Evaluation script  
│   └── test_competitive_setup.py # Test script (optional)
├── Model_Archi/                  # Model architecture
│   ├── quantum_dataset.py        # Dataset handling
│   ├── updated_training_model.py # Complete model definition
│   ├── transformer_embedding.py  # Embedding layers
│   ├── transformer_encoder.py    # Encoder layers
│   └── mlp.py                    # MLP/MoE heads
├── CONFIGs/2_128_cls_moe_20250817_202509/  # Model config & weights
│   ├── 2_128_cls_moe_config.json          # Model configuration
│   └── best_2_128_cls_moe_20250817_202509.pth  # Pretrained weights
├── requirements.txt              # Python dependencies
└── README_HPC.md                # This file
```

## Setup Instructions

### 1. Upload to HPC
```bash
# Upload the entire upload/ directory to your HPC workspace
scp -r upload/ username@hpc-cluster:/path/to/your/workspace/
```

### 2. Environment Setup
```bash
# Create conda environment
conda create -n competitive_moe python=3.9
conda activate competitive_moe

# Install dependencies
pip install -r requirements.txt

# Verify PyTorch CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 3. Data Preparation
You need to upload your quantum dataset separately:
```bash
# Create data directories (adjust paths in train script if different)
mkdir -p Rawdata/DecodedTokens
mkdir -p Rawdata/label

# Upload your data files:
# - Rawdata/DecodedTokens/*_real.npy, *_imag.npy
# - Rawdata/label/magic_labels_for_input_for_2_qubits_mixed_2_200000_datapoints.npy
```

## Running Competitive Training

### Basic Usage
```bash
cd upload/MLP_MoE/Competetive

python train_moe_competitive.py \
  --init_ckpt ../../CONFIGs/2_128_cls_moe_20250817_202509/best_2_128_cls_moe_20250817_202509.pth \
  --out_dir ./competitive_checkpoints \
  --epochs 30 \
  --lr 3e-4 \
  --bs 256
```

### Recommended HPC Settings
```bash
# For SLURM submission
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
  --sigma 0.1
```

### Key Parameters
- `--init_ckpt`: Path to collaborative MoE checkpoint (inheritance)
- `--tau_start/end`: Temperature annealing (1.8→1.0 for soft→competitive)  
- `--topk_start/target`: Epoch to enable top-k, final k value
- `--lb_lambda`: Load balancing weight (prevent expert collapse)
- `--lik`: Loss type (gauss/laplace)

## Monitoring Training

### Real-time Logs
Training outputs JSON logs per epoch:
```json
{
  "epoch": 15,
  "wall_s": 1234.5,
  "tau": 1.2,
  "topk": null,
  "train_mse": 0.0123,
  "train_nll": 2.345,
  "val_mse": 0.0145,
  "val_nll": 2.456
}
```

### Evaluation
```bash
python eval_moe_competitive.py \
  --ckpt ./competitive_checkpoints/moe_competitive_00000025.pt \
  --bs 512
```

## Expected Behavior

### Training Phases
1. **Soft Competition** (epochs 1-15): τ=1.8→1.0, all experts active
2. **Hard Competition** (epochs 20+): τ=1.0, top-k=1, winner-takes-all

### Success Metrics  
- **Routing Entropy**: Should decrease as competition increases
- **Expert Usage**: Should become more specialized/imbalanced
- **NLL**: Should decrease (competitive loss optimization)
- **MSE**: Should remain stable or improve

## Troubleshooting

### Common Issues
1. **CUDA OOM**: Reduce `--bs` (try 256, 128)
2. **Expert Collapse**: Increase `--lb_lambda` (try 0.05)
3. **Path Errors**: Verify data/config paths relative to script location
4. **Import Errors**: Check sys.path additions in scripts

### Performance Tips
- Use mixed precision: `--no_amp` flag to disable if issues
- Gradient clipping: `--clip_grad 1.0` (default)
- Batch size scaling: Larger batches = more stable gradients

Good luck with your competitive MoE training! 🚀