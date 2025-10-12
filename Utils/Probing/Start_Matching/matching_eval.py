# probe_eval_csv.py
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr, spearmanr

def _rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def _norm_rmse(y_true, y_pred):
    rng = float(np.max(y_true) - np.min(y_true) + 1e-12)
    return _rmse(y_true, y_pred) / rng

def evaluate_one_layer(layer_csv: str, targets_csv: str, target_cols, model_type="linear", alpha=1.0, n_splits=5):
    """
    读入一层的 CSV (idx, f0..fd-1) 与 目标 CSV (必须包含 idx 和 target_cols)
    返回: DataFrame, 每个目标一行，含 CV-R2, Pearson r, Spearman ρ, NRMSE
    """
    Xdf = pd.read_csv(layer_csv)
    Tdf = pd.read_csv(targets_csv)

    # 仅按 idx 合并
    df = Xdf.merge(Tdf, on="idx", how="inner")
    feat_cols = [c for c in df.columns if c.startswith("f")]
    X = df[feat_cols].values.astype(np.float64)

    results = []
    for tgt in target_cols:
        y = df[tgt].values.astype(np.float64)

        # 选择模型
        if model_type == "linear":
            reg = LinearRegression()
        elif model_type == "ridge":
            reg = Ridge(alpha=alpha)
        elif model_type == "lasso":
            reg = Lasso(alpha=alpha, max_iter=10000)
        else:
            raise ValueError("model_type must be one of {'linear','ridge','lasso'}")

        # K 折交叉验证（用折外预测评估）
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        y_pred_cv = np.zeros_like(y)
        for train_idx, test_idx in kf.split(X):
            reg.fit(X[train_idx], y[train_idx])
            y_pred_cv[test_idx] = reg.predict(X[test_idx])

        # 指标
        r2 = r2_score(y, y_pred_cv)
        pr, _ = pearsonr(y, y_pred_cv)
        sr, _ = spearmanr(y, y_pred_cv)
        nrmse = _norm_rmse(y, y_pred_cv)

        results.append({
            "layer_csv": os.path.basename(layer_csv),
            "target": tgt,
            "cv_r2": float(r2),
            "pearson_r": float(pr),
            "spearman_rho": float(sr),
            "nrmse": float(nrmse),
            "n_samples": int(len(y)),
            "n_features": int(X.shape[1]),
            "model_type": model_type,
            "alpha": float(alpha),
        })

    return pd.DataFrame(results)

def evaluate_many_layers(layers_manifest_csv: str, targets_csv: str, target_cols, out_csv: str,
                         model_type="linear", alpha=1.0, n_splits=5):
    """
    扫描 manifest_layers.csv 中的每一层 CSV，汇总成一个评分表。
    """
    mani = pd.read_csv(layers_manifest_csv)  # 列: layer, path, dim
    all_rows = []
    for _, row in mani.iterrows():
        layer_path = row["path"]
        df = evaluate_one_layer(layer_path, targets_csv, target_cols,
                                model_type=model_type, alpha=alpha, n_splits=n_splits)
        df.insert(0, "layer", row["layer"])
        all_rows.append(df)
        print(f"✓ evaluated {row['layer']}")

    out = pd.concat(all_rows, axis=0, ignore_index=True)
    out.to_csv(out_csv, index=False)
    print(f"✅ saved summary: {out_csv}")
    return out
