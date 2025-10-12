import os
import re
import json
import ast
import argparse
import warnings
import numpy as np
import pandas as pd

from typing import Dict, Tuple, List, Any
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr, spearmanr

warnings.filterwarnings("ignore", category=UserWarning)

# ---- 1) 目标字段形态配置（可根据你的表结构调整/增补） ----
# 形态: "scalar" | "vec3" | "mat3x3"
TARGET_SPECS = {
    "S_rho_bits": "scalar",
    "purity_scalar": "scalar",
    "Hinf_bits": "scalar",
    "CHSH_bound_max": "scalar",
    "T_fro_norm": "scalar",
    "C_rel_bits": "scalar",
    "C_conc": "scalar",
    "N_neg": "scalar",
    "r_bloch_A": "vec3",
    "s_bloch_B": "vec3",
    "T_corr": "mat3x3",
}

# 可选：如果你的标签文件把 vec/mat 拆成多个列，按需在这里声明映射
# 例如：
# VEC3_COLUMNS = {
#     "r_bloch_A": ["r_bloch_A_x", "r_bloch_A_y", "r_bloch_A_z"],
#     "s_bloch_B": ["s_bloch_B_x", "s_bloch_B_y", "s_bloch_B_z"],
# }
VEC3_COLUMNS = {}
MAT3x3_COLUMNS = {}  # e.g., {"T_corr": ["T11","T12",...,"T33"]}

# ---- 2) 工具函数 ----
import re
import numpy as np
import ast

def _from_numpy_style_string(s: str) -> np.ndarray:
    """
    解析类似 '[-0.1 0.2 0.3]' 或 '[[...][...][...]]' 的空格分隔、无逗号 numpy 风格字符串。
    """
    # 去掉多余空白和换行
    s_clean = re.sub(r'\s+', ' ', s.strip())
    # 去掉方括号方便提取数值
    s_flat = s_clean.replace('[', ' ').replace(']', ' ')
    arr = np.fromstring(s_flat, sep=' ')
    return arr

def parse_obj_cell(cell: object) -> np.ndarray:
    """
    把存成字符串/对象的向量/矩阵解析为 numpy 数组。
    首选 literal_eval（逗号分隔等标准 Python 表达式），失败后用 numpy 风格解析兜底。
    """
    if isinstance(cell, (list, tuple, np.ndarray)):
        return np.array(cell)

    if isinstance(cell, str):
        s = cell.strip()
        # 先试标准 Python 表达式（适合逗号分隔）
        try:
            obj = ast.literal_eval(s)
            return np.array(obj)
        except Exception:
            # 再试 numpy 风格（空格分隔、无逗号）
            try:
                return _from_numpy_style_string(s)
            except Exception as e:
                raise ValueError(f"Cannot parse array-like string: {cell}") from e

    raise ValueError(f"Cannot parse cell to array: {cell}")

def load_targets(df_labels, target: str) -> tuple[np.ndarray, str]:
    """
    根据 TARGET_SPECS/VEC3_COLUMNS/MAT3x3_COLUMNS，产出 y 和 target_type。
    支持：
      - 单列字符串（空格/换行、无逗号的 numpy 风格）
      - 多列拆分（若在映射里声明）
    """
    ttype = TARGET_SPECS[target]

    if ttype == "scalar":
        y = df_labels[target].to_numpy().reshape(-1, 1)
        return y, ttype

    if ttype == "vec3":
        if target in VEC3_COLUMNS and VEC3_COLUMNS[target]:
            cols = VEC3_COLUMNS[target]
            y = df_labels[cols].to_numpy()
        else:
            y = np.vstack(df_labels[target].apply(parse_obj_cell).to_numpy())
        if y.shape[1] != 3:
            raise ValueError(f"{target} expected 3-dim, got {y.shape}")
        return y, ttype

    if ttype == "mat3x3":
        if target in MAT3x3_COLUMNS and MAT3x3_COLUMNS[target]:
            cols = MAT3x3_COLUMNS[target]
            y = df_labels[cols].to_numpy().reshape(-1, 3, 3)
        else:
            mats = [parse_obj_cell(v) for v in df_labels[target].tolist()]
            mats = [m.reshape(3, 3) if m.size == 9 else m for m in mats]
            y = np.stack(mats, axis=0)  # [N, 3, 3]
        if y.shape[1:] != (3, 3):
            raise ValueError(f"{target} expected (3,3), got {y.shape}")
        # 扁平化成 9 维用于多输出回归
        y = y.reshape(len(y), 9)
        return y, ttype

    raise KeyError(f"Unknown target type for {target}")


def choose_estimator(output_dim: int, alpha: float = 1.0):
    """
    标量/多输出统一选择：Ridge + 标准化；多输出使用 MultiOutput 包裹。
    """
    base = Pipeline([("scaler", StandardScaler(with_mean=True, with_std=True)),
                     ("reg", Ridge(alpha=alpha, random_state=42))])
    if output_dim == 1:
        return base
    else:
        return MultiOutputRegressor(base)

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, target_type: str) -> Dict[str, float]:
    """
    汇总指标；多输出用 macro 平均；矩阵多给个 Frobenius RMSE（按扁平 9维近似）。
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Handle 1D arrays by reshaping to 2D
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)
    
    n_out = y_true.shape[1]

    # 每维
    r2_list, mae_list, rmse_list = [], [], []
    for j in range(n_out):
        r2_list.append(r2_score(y_true[:, j], y_pred[:, j]))
        mae_list.append(mean_absolute_error(y_true[:, j], y_pred[:, j]))
        rmse_list.append(np.sqrt(mean_squared_error(y_true[:, j], y_pred[:, j])))

    metrics = {
        "r2_macro": float(np.mean(r2_list)),
        "mae_macro": float(np.mean(mae_list)),
        "rmse_macro": float(np.mean(rmse_list)),
    }

    # 相关性只在标量时给更有意义（也可对多输出做平均）
    if n_out == 1:
        y0, yhat0 = y_true[:, 0], y_pred[:, 0]
        try:
            pr, _ = pearsonr(y0, yhat0)
        except Exception:
            pr = np.nan
        try:
            sr, _ = spearmanr(y0, yhat0)
        except Exception:
            sr = np.nan
        metrics.update({"pearson": float(pr), "spearman": float(sr)})

    # 矩阵（扁平 9 维）补充 Frobenius RMSE（按 9维等价）
    if target_type == "mat3x3":
        diff = y_true - y_pred
        # Fro RMSE per sample = sqrt(sum(diff^2)), 再对样本求均值
        fro_rmse = float(np.mean(np.sqrt(np.sum(diff**2, axis=1))))
        metrics.update({"fro_rmse": fro_rmse})

    return metrics

def read_layer_csv(path: str) -> pd.DataFrame:
    """
    读取层 CSV。要求列名包含 f0..f127；如有 sample_id 列则保留。
    """
    df = pd.read_csv(path)
    # 若无 sample_id，用行号生成
    if "sample_id" not in df.columns:
        df = df.reset_index().rename(columns={"index": "sample_id"})
    # 只保留特征列 + sample_id
    feat_cols = [c for c in df.columns if re.match(r"f\d+$", c)]
    cols = ["sample_id"] + feat_cols
    return df[cols]

# ---- 3) 主流程 ----
def run_batch(layers_dir: str,
              labels_path: str,
              out_dir: str,
              targets: List[str],
              max_layers: int = None,
              cv_folds: int = 5,
              alpha: float = 1.0,
              subsample: int = None):

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "details"), exist_ok=True)

    # 读取标签表
    labels_df = pd.read_csv(labels_path)
    if "sample_id" not in labels_df.columns:
        labels_df = labels_df.reset_index().rename(columns={"index": "sample_id"})

    # 遍历层文件
    layer_files = sorted([f for f in os.listdir(layers_dir) if f.endswith(".csv")])
    if max_layers is not None:
        layer_files = layer_files[:max_layers]

    summary_rows = []

    for lf in layer_files:
        layer_name = os.path.splitext(lf)[0]
        layer_path = os.path.join(layers_dir, lf)
        layer_df = read_layer_csv(layer_path)

        # join 对齐
        merged = layer_df.merge(labels_df, on="sample_id", how="inner")
        # X
        feat_cols = [c for c in merged.columns if re.match(r"f\d+$", c)]
        X = merged[feat_cols].to_numpy()

        # 可选下采样
        if subsample is not None and subsample < X.shape[0]:
            X = X[:subsample]
            sub_df = merged.iloc[:subsample].copy()
        else:
            sub_df = merged

        for tgt in targets:
            try:
                y, ttype = load_targets(sub_df, tgt)
            except Exception as e:
                print(f"[WARN] Skip {layer_name} × {tgt}: {e}")
                continue

            # 模型
            est = choose_estimator(output_dim=y.shape[1], alpha=alpha)

            # 交叉验证预测（防止泄漏）
            cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
            y_pred = cross_val_predict(est, X, y, cv=cv, n_jobs=-1, verbose=0)

            # 指标
            metrics = compute_metrics(y, y_pred, ttype)

            # 汇总
            row = {
                "layer": layer_name,
                "target": tgt,
                "target_type": ttype,
                "n_samples": int(X.shape[0]),
                **metrics
            }
            summary_rows.append(row)

            # 详细保存
            detail = {
                "layer": layer_name,
                "target": tgt,
                "target_type": ttype,
                "metrics": metrics,
                "cv_folds": cv_folds,
                "alpha": alpha,
                "n_samples": int(X.shape[0]),
            }
            with open(os.path.join(out_dir, "details", f"{layer_name}__{tgt}.json"), "w") as f:
                json.dump(detail, f, indent=2)

        print(f"[DONE] {layer_name}")

    # 写总表
    summ = pd.DataFrame(summary_rows)
    summ.sort_values(by=["target", "r2_macro"], ascending=[True, False], inplace=True)
    summ.to_csv(os.path.join(out_dir, "results_summary.csv"), index=False)

    # 每个 target 选最佳层
    best_rows = []
    for tgt, sub in summ.groupby("target"):
        best = sub.sort_values("r2_macro", ascending=False).iloc[0].to_dict()
        best_rows.append(best)
    pd.DataFrame(best_rows).to_csv(os.path.join(out_dir, "best_layers_by_target.csv"), index=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers_dir", required=True, help="目录：逐层 CLS CSV")
    parser.add_argument("--labels_path", required=True, help="物理标签 CSV")
    parser.add_argument("--out_dir", required=True, help="结果输出目录")
    parser.add_argument("--targets", nargs="+", default=list(TARGET_SPECS.keys()), help="要评估的目标列名")
    parser.add_argument("--max_layers", type=int, default=None)
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--subsample", type=int, default=None, help="可选：只用前 N 条做快速跑通")
    args = parser.parse_args()

    run_batch(layers_dir=args.layers_dir,
              labels_path=args.labels_path,
              out_dir=args.out_dir,
              targets=args.targets,
              max_layers=args.max_layers,
              cv_folds=args.cv,
              alpha=args.alpha,
              subsample=args.subsample)

if __name__ == "__main__":
    main()
