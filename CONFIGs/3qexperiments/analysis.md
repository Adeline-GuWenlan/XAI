⏺ Write(experiment_performance_ranking_updated.csv)
  ⎿  Wrote 14 lines to experiment_performance_ranking_updated.csv                                                         
     experiment_name,r2_score,d_model,nhead,epochs,mlp_type,mse,mae,spearman,pearson,timestamp
     3q_d1024_h8_moe_20250902_211047,0.8884,1024,8,100,mixture_of_experts,0.001953,0.03374,0.9314,0.9427,20250902_211047
     3q_d1024_h4_moe_20250902_210914,0.8865,1024,4,100,mixture_of_experts,0.001987,0.03414,0.9324,0.9425,20250902_210914
     3q_d512_h8_moe_20250902_204025,0.8851,512,8,100,mixture_of_experts,0.002011,0.03458,0.9315,0.9417,20250902_204025
     3q_d512_h4_moe_20250902_203848,0.8821,512,4,100,mixture_of_experts,0.002063,0.03467,0.9316,0.9392,20250902_203848
     3q_d2048_h4_moe_20250902_211142,0.8815,2048,4,100,mixture_of_experts,0.002074,0.03493,0.9278,0.9393,20250902_211142
     3q_d2048_h8_asym_20250903_125041,0.8803,2048,8,100,asymmetric_ensemble,0.002095,0.03488,0.9273,0.9386,20250903_125041
     3q_512_cls_moe_4h_20250902_182736,0.8425,512,4,20,mixture_of_experts,0.002756,0.04026,0.9113,0.9197,20250902_182736
     3q_512_cls_moe_4h_20250902_183252,0.8291,512,64,20,mixture_of_experts,0.002991,0.04301,0.9058,0.9182,20250902_183252
     3q_256_cls_moe_64h_20250902_181813,0.5954,256,32,20,mixture_of_experts,0.007081,0.06565,0.7818,0.7793,20250902_181813
     … +4 lines (ctrl+r to expand)

⏺ Updated analysis with 13 experiments total. Here are the key findings:

  Performance Ranking (Best to Worst by R²):
  1. 3q_d1024_h8_moe (R² = 0.8884) - d_model=1024, nhead=8, 100 epochs
  2. 3q_d1024_h4_moe (R² = 0.8865) - d_model=1024, nhead=4, 100 epochs
  3. 3q_d512_h8_moe (R² = 0.8851) - d_model=512, nhead=8, 100 epochs
  4. 3q_d512_h4_moe (R² = 0.8821) - d_model=512, nhead=4, 100 epochs
  5. 3q_d2048_h4_moe (R² = 0.8815) - d_model=2048, nhead=4, 100 epochs
  6. 3q_d2048_h8_asym (R² = 0.8803) - d_model=2048, nhead=8, asymmetric_ensemble

  Key Parameters & Differences:
  - d_model: 128, 256, 512, 1024, 2048
  - nhead: 4, 8, 16, 32, 64, 128
  - epochs: 20 vs 100 (major difference)
  - mlp_type: mixture_of_experts vs asymmetric_ensemble

  Critical Observations:
  1. Training epochs matter most: 100-epoch models (R² ~0.88) vastly outperform 20-epoch models (R² 0.84 max)
  2. Optimal d_model is 1024: Better than both 512 and 2048, suggesting sweet spot for this task
  3. Head count has minimal impact: 4 vs 8 heads show similar performance within same d_model
  4. 2048-dim shows diminishing returns: Despite more parameters, doesn't outperform 1024-dim
  5. asymmetric_ensemble vs mixture_of_experts: Similar performance levels

  The updated CSV shows 1024-dimensional models with 100 epochs achieve the best R² scores around 0.888.