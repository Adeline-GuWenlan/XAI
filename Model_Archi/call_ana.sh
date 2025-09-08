# ---- Basic ----
python  run_saliency_analysis.py \
    --model_path /home/gwl/DURF/NewTrans_Try_MLP/models/best_robust_finetune_20250806_112411.pth \
    --n_qubits 2 \
    --d_model 64 \
    --mask_threshold 2 \
    --use_physics_mask \
    --mlp_type physics_aware
# ---- physics ----
python run_saliency_analysis.py \
    --model_path /path/to/your/physics_model.pth \
    --n_qubits 2 \
    --d_model 64 \
    --pooling_type cls \
    --mlp_type physics_aware \
    --use_physics_mask \
    --mask_threshold 2 \
    --methods gradient integrated_gradients