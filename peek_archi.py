import torch
data = torch.load("/Users/guwenlan/Desktop/XAI/CONFIGs/2_128_cls_moe_20250817_202509/best_2_128_cls_moe_20250817_202509.pth", map_location=torch.device('cpu'))
print(type(data))
import torch

# 安全加载模型（使用 weights_only=True 避免潜在风险）
try:
    # 首先尝试安全模式加载
    state_dict = torch.load(
        "/Users/guwenlan/Desktop/XAI/CONFIGs/2_128_cls_moe_20250817_202509/best_2_128_cls_moe_20250817_202509.pth",
        map_location=torch.device('cpu'),
        weights_only=True
    )
except TypeError:
    # 如果安全模式失败（旧版本PyTorch不支持），回退到非安全模式
    print("except TypeError:如果安全模式失败（旧版本PyTorch不支持），回退到非安全模式")
    state_dict = torch.load(
        "/Users/guwenlan/Desktop/XAI/CONFIGs/2_128_cls_moe_20250817_202509/best_2_128_cls_moe_20250817_202509.pth",
        map_location=torch.device('cpu')
    )

# 分析模型结构
print(f"模型包含 {len(state_dict)} 个参数层")
print("\n各层参数形状:")

# 遍历并打印每一层的形状
for name, param in state_dict.items():
    print(f"{name:<40} | 形状: {tuple(param.shape)}")

# 提取关键结构信息
transformer_layers = set()
mlp_layers = set()

for name in state_dict.keys():
    if 'transformer' in name.lower():
        # 提取 transformer 层编号（如 transformer.encoder.layers.0...）
        parts = name.split('.')
        for part in parts:
            if part.isdigit():
                transformer_layers.add(f"层 {part}")
                break
    elif 'mlp' in name.lower() or 'feedforward' in name.lower():
        parts = name.split('.')
        for part in parts:
            if part.isdigit():
                mlp_layers.add(f"层 {part}")
                break

print("\n推断的模型结构:")
print(f"- Transformer 层数: {len(transformer_layers) or '未知'}")
print(f"- MLP 层数: {len(mlp_layers) or '未知'}")

# 检查是否有嵌入层
embedding_keys = [k for k in state_dict.keys() if 'embed' in k.lower()]
if embedding_keys:
    print(f"- 检测到嵌入层: {embedding_keys[0]} (维度: {state_dict[embedding_keys[0]].shape[-1]})")

# 检查输出层
output_keys = [k for k in state_dict.keys() if 'out' in k.lower() or 'head' in k.lower() or 'classifier' in k.lower()]
if output_keys:
    output_dim = state_dict[output_keys[0]].shape[0]
    print(f"- 输出层维度: {output_dim} (可能对应分类类别数)")