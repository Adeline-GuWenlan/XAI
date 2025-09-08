conmand.sh


python visualizations/Visualization.py\
 --model_path CONFIGs/2_128_cls_moe_20250817_202509/best_2_128_cls_moe_20250817_202509.pth\
 --config_from_json CONFIGs/2_128_cls_moe_20250817_202509/2_128_cls_moe_config.json

python visualizations/Visualization.py\
 --model_path CONFIGs/128_CLS_attention_enhanced/best_128_CLS_attention_enhanced_20250810_214428.pth\
 --config_from_json CONFIGs/128_CLS_attention_enhanced/128_CLS_attention_enhanced_config.json



python AttentionPoolingVisualization.py\
 --model_path CONFIGs/128_Attention_attention_enhanced_20250815_120257/best_128_Attention_attention_enhanced_20250815_120257.pth\
 --config_from_json CONFIGs/128_Attention_attention_enhanced_20250815_120257/128_Attention_attention_enhanced_config.json



# Attention
python Saliency/run_saliency_analysis.py\
 --model_path CONFIGs/2_128_cls_moe_20250817_202509/best_2_128_cls_moe_20250817_202509.pth\
 --d_model 128\
 --mlp_type mixture_of_experts\
 --methods integrated_gradients\
 --individual_samples 0



python Model_Archi/decoder.py\
 --input Rawdata/Input\
 --output Rawdata/DecodedTokens\
 --dim 4

 python -m Probing.Get_Layer_Represen.Atten_Prober.run_attention_probe --model_path CONFIGs/128_Attention_attention_enhanced_20250815_120257/best_128_Attention_attention_enhanced_20250815_120257.pth