# run_probe_exact.py
import os, json, sys

# GPU environment setup for optimal performance
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use GPU 0 as recommended in claude.md

# Add paths for imports
sys.path.append('../..')
sys.path.append('../../Model_Archi')

from new_prober import extract_layer_csv_exact

# paste your JSON here or load from file
CONFIG = {
  "exp_name": "Msk_CLS_moe",
  "num_qubits": 2,
  "d_model": 128,
  "use_physics_mask": False,
  "pooling_type": "cls",
  "mlp_type": "mixture_of_experts",
  "matrix_dim": 4,
  "data_folder": "Rawdata/DecodedTokens",
  "labels_path": "Rawdata/label/magic_labels_for_input_for_2_qubits_mixed_2_50000_datapoints.npy",
  "device": "cuda:1",
  "batch_size": 512,
  "epochs": 40,
  "lr": 0.0001,
  "train_ratio": 0.8,
  "val_ratio": 0.1,
  "nhead":8, 
  "mask_threshold": 1, 
  "use_cls_token": True,
  "pytorch_version": "2.5.1+cu121",
  "experiment_info": {
    "exp_name": "Msk_CLS_moe",
    "timestamp": "20250809_012529",
    "exp_dir": "experiments/Msk_CLS_moe_20250809_012529",
    "model_path": "CONFIGs/2_128_cls_moe_20250817_202509/best_2_128_cls_moe_20250817_202509.pth",
    "log_path": "experiments/Msk_CLS_moe_20250809_012529/Msk_CLS_moe.csv"
  }
}

def cfg_to_model_kwargs(cfg):
    return dict(
        matrix_dim       = cfg["matrix_dim"],
        n_qubits         = cfg["num_qubits"],
        d_model          = cfg["d_model"],
        pooling_type     = cfg["pooling_type"],
        mlp_type         = cfg["mlp_type"],
        use_physics_mask = cfg["use_physics_mask"],
        mask_threshold   = cfg["mask_threshold"],
        nhead            = cfg["nhead"],
        use_cls_token    = cfg["use_cls_token"],
    )

if __name__ == "__main__":
    model_cfg = cfg_to_model_kwargs(CONFIG)
    ckpt_path = CONFIG["experiment_info"]["model_path"]
    out_dir   = os.path.join(CONFIG["experiment_info"]["exp_dir"], "layer_features_csv_exact")

    extract_layer_csv_exact(
        model_cfg     = model_cfg,
        ckpt_path     = ckpt_path,
        data_folder   = CONFIG["data_folder"],
        labels_path   = CONFIG["labels_path"],
        out_dir       = out_dir,
        device        = CONFIG["device"],
        batch_size    = CONFIG["batch_size"],
        num_workers   = 4,  # Increased from 4 to min(8, cpu_count()) as per claude.md
        normalize_trace = True
    )



