# 解读
# 修改
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
#| 模型架构   | `Linear(16→64→32→1)`, 中间带 `ReLU + Dropout(0.2)` |
class MagicMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.model(x)

class MagicDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
### 加载数据
def load_and_preprocess(input_path, label_path):
    X = np.load(input_path)
    y = np.load(label_path)
    #| 输入数据清洗 | `np.real_if_close` 移除虚部                         |
    X = np.real_if_close(X, tol=1e5)  ### 清理数据：去除微小虚部，会把形如 (1+1e-17j) 转为 real
    #| 特征缩放   | `StandardScaler` 标准化                            |
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, scaler


