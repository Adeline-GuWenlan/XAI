'''# train.py
import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
from MLP_2_baseline_Model import MagicMLP, MagicDataset, load_and_preprocess

# 1. 加载与预处理数据
X, y, scaler = load_and_preprocess("input_for_2_qubits_mixed_10000_datapoints.npy", "magic_labels_for_input_for_2_qubits_mixed_10000_datapoints.npy")
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. 构建数据集与模型
train_loader = torch.utils.data.DataLoader(MagicDataset(X_train, y_train), batch_size=64, shuffle=True)
val_loader = torch.utils.data.DataLoader(MagicDataset(X_val, y_val), batch_size=64)
model = MagicMLP()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.MSELoss()

# 3. 训练循环
for epoch in range(50):
    model.train()
    for xb, yb in train_loader:
        pred = model(xb)
        loss = loss_fn(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 验证评估
    model.eval()
    with torch.no_grad():
        val_loss = sum(loss_fn(model(xb), yb).item() for xb, yb in val_loader) / len(val_loader)
    print(f"Epoch {epoch+1}, Val Loss: {val_loss:.5f}")
'''
# train.py
import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
from MLP_2_baseline_Model import MagicMLP, MagicDataset, load_and_preprocess


# wandb
import wandb
wandb.init(
    project="magic cal",
    name="MLP_baseline_2",  # 每次实验的名字，便于追踪
    config={
        "lr": 1e-3,
        "epochs": 50,
        "batch_size": 64,
        "model": "MLP-16-64-32-1",
        "dropout": 0.2,
        "optimizer": "Adam",
        "loss_fn": "MSELoss"
    }
)

import matplotlib.pyplot as plt
#4. 可选：绘制 True vs. Predicted scatter plot
# 预测 vs. 真实值 scatter 图上传 W&B🖼️ 
def log_prediction_plot(model, val_loader):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            y_hat = model(xb)
            preds.extend(y_hat.squeeze().tolist())
            trues.extend(yb.squeeze().tolist())
    
    plt.figure()
    plt.scatter(trues, preds, alpha=0.4)
    plt.xlabel("True Magic")
    plt.ylabel("Predicted Magic")
    plt.title("Prediction Scatter")
    plt.grid(True)
    wandb.log({"scatter": wandb.Image(plt)})
    plt.close()
    from sklearn.metrics import mean_squared_error, r2_score

    # 预测值和真实值数组
    preds = np.array(preds)
    trues = np.array(trues)

    mse = mean_squared_error(trues, preds)
    rmse = mse ** 0.5
    r2 = r2_score(trues, preds)
    rel_err = np.mean(np.abs(preds - trues) / (trues + 1e-8))

    print(f"RMSE: {rmse:.6f}")
    print(f"R² Score: {r2:.6f}")
    print(f"Mean Relative Error: {rel_err:.6%}")



# 1. 加载与预处理数据
X, y, scaler = load_and_preprocess("input_for_2_qubits_mixed_10000_datapoints.npy", "magic_labels_for_input_for_2_qubits_mixed_10000_datapoints.npy")
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. 构建数据集与模型
train_loader = torch.utils.data.DataLoader(MagicDataset(X_train, y_train), batch_size=64, shuffle=True)
val_loader = torch.utils.data.DataLoader(MagicDataset(X_val, y_val), batch_size=64)
model = MagicMLP()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = torch.nn.MSELoss()

# 3. 训练循环
#✅ 1. 在 for epoch in range(...) 之前：初始化最优值记录变量：
best_val = float('inf')  # 👈 放在训练循环外部

for epoch in range(50):
    model.train()
    train_loss_total = 0
    for xb, yb in train_loader:
        pred = model(xb)
        loss = loss_fn(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss_total += loss.item()
    train_loss = train_loss_total / len(train_loader)
        # 验证评估
    model.eval()
    with torch.no_grad():
        val_loss = sum(loss_fn(model(xb), yb).item() for xb, yb in val_loader) / len(val_loader)
    print(f"Epoch {epoch+1}, Val Loss: {val_loss:.5f}")
    # val_loss 已经被你计算完（对整个 val_loader 求平均）
    wandb.log({
        "epoch": epoch + 1,
        "val_loss": val_loss,
        "train_loss": train_loss  # 如果你计算了
    })
    if val_loss < best_val:
        best_val = val_loss
        torch.save(model.state_dict(), "best_model.pt")
        wandb.run.summary["best_val_loss"] = best_val
        wandb.save("best_model.pt")

# === 所有 epoch 完成后 ===
log_prediction_plot(model, val_loader)
wandb.finish()