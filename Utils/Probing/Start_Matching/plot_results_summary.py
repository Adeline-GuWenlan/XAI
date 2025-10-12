
#!/usr/bin/env python3
"""
Plot XAI probing results by layer order.

Input:
  - results_summary.csv (required): columns must include
        layer, target, target_type, r2_macro, [optional: n_samples, mae_macro, rmse_macro, ...]
  - Optional layer order sources (checked in this order):
        1) --layer_order path/to/layer_order.txt  (one layer name per line)
        2) manifest_layers.csv in the same folder as results_summary.csv (uses its "layer" column order)
        3) Heuristic ordering by known prefixes + numbers, with lexicographic fallback

Output:
  - Three figures (PNG + displayed): scalar.png, vec3.png, mat3x3.png
    X-axis: layers in chronological/architectural order
    Y-axis: r2_macro (already R^2 — DO NOT square again)
    One line per target, legend on the right
"""

import argparse
import os
import re
from pathlib import Path
from typing import List, Dict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def read_layer_order_from_txt(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def read_layer_order_from_manifest(manifest_path: Path) -> List[str]:
    try:
        dfm = pd.read_csv(manifest_path)
        # Expect columns: layer, path, dim
        if "layer" in dfm.columns:
            return dfm["layer"].tolist()
    except Exception:
        pass
    return []


def heuristic_sort_layers(layer_names: List[str]) -> List[str]:
    """
    Sort layers in architectural order for MoE transformer:
    enc_00_post → enc_01_post → enc_02_post → enc_03_post 
    → mlp_in_pre → mlp_in_post 
    → mlp_gating_pre → mlp_gating_post 
    → mlp_expert_00_pre → mlp_expert_00_post 
    → mlp_expert_01_pre → mlp_expert_01_post 
    → mlp_expert_02_pre → mlp_expert_02_post
    """
    def enc_key(name: str):
        m = re.match(r"enc_(\d{2})_(pre|post)$", name)
        if m:
            block = int(m.group(1))
            sub = 0 if m.group(2) == "pre" else 1
            return (0, block, sub, 0, 0, 0, name)
        return None

    def mlp_in_key(name: str):
        m = re.match(r"mlp_in_(pre|post)$", name)
        if m:
            sub = 0 if m.group(1) == "pre" else 1
            return (1, 0, sub, 0, 0, 0, name)
        return None

    def mlp_gating_key(name: str):
        m = re.match(r"mlp_gating_(pre|post)$", name)
        if m:
            sub = 0 if m.group(1) == "pre" else 1
            return (2, 0, sub, 0, 0, 0, name)
        return None

    def mlp_expert_key(name: str):
        m = re.match(r"mlp_expert_(\d{2})_(pre|post)$", name)
        if m:
            expert = int(m.group(1))  # 00, 01, 02
            sub = 0 if m.group(2) == "pre" else 1
            return (3, expert, sub, 0, 0, 0, name)
        return None

    def fallback_key(name: str):
        # Unknowns go after knowns; keep alphabetical within this group.
        return (9, 0, 0, 0, 0, 0, name)

    keys = []
    for n in layer_names:
        for keyfun in (enc_key, mlp_in_key, mlp_gating_key, mlp_expert_key):
            k = keyfun(n)
            if k is not None:
                keys.append((k, n))
                break
        else:
            keys.append((fallback_key(n), n))

    keys.sort(key=lambda x: x[0])
    return [n for _, n in keys]


def build_layer_order(results_df: pd.DataFrame, csv_path: Path, layer_order_txt: Path = None) -> List[str]:
    # 1) explicit file if provided
    if layer_order_txt and layer_order_txt.exists():
        order = read_layer_order_from_txt(layer_order_txt)
        existing = set(results_df["layer"].unique())
        order = [x for x in order if x in existing]
        return order

    # 2) manifest in same folder
    src_dir = csv_path.parent
    manifest = src_dir / "manifest_layers.csv"
    if manifest.exists():
        order = read_layer_order_from_manifest(manifest)
        existing = set(results_df["layer"].unique())
        order = [x for x in order if x in existing]
        if order:
            return order

    # 3) heuristic
    unique_layers = results_df["layer"].unique().tolist()
    return heuristic_sort_layers(unique_layers)


def plot_group(df: pd.DataFrame, layer_order: List[str], target_type: str, out_dir: Path):
    sub = df[df["target_type"] == target_type].copy()
    if sub.empty:
        print(f"[WARN] No rows for target_type={target_type}")
        return

    # ensure all listed layers exist in the data and keep order
    layer_to_pos = {layer: i for i, layer in enumerate(layer_order)}
    sub = sub[sub["layer"].isin(layer_to_pos.keys())]
    sub["xpos"] = sub["layer"].map(layer_to_pos)

    # pivot to have one line per target
    # y-axis is r2_macro (already R^2)
    # We aggregate by mean in case there are duplicates for a (layer, target) pair.
    pivot = (sub.groupby(["target", "layer"], as_index=False)["r2_macro"]
                .mean()
                .assign(xpos=lambda d: d["layer"].map(layer_to_pos))
                .sort_values(["target", "xpos"]))

    # Prepare plotting
    fig, ax = plt.subplots(figsize=(max(10, len(layer_order) * 0.4), 6))
    for tgt, g in pivot.groupby("target"):
        g_sorted = g.sort_values("xpos")
        ax.plot(g_sorted["xpos"], g_sorted["r2_macro"], marker="o", label=str(tgt))

    ax.set_title(f"Layer-wise linear decodability: {target_type} (r2_macro)")
    ax.set_xlabel("Layer (architectural order)")
    ax.set_ylabel("r2_macro (R^2)")
    ax.set_xticks(range(len(layer_order)))
    ax.set_xticklabels(layer_order, rotation=60, ha="right")
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.)

    out_path = out_dir / f"{target_type}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"[Saved] {out_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", required=True, help="Path to results_summary.csv")
    parser.add_argument("--layer_order", default=None, help="Optional path to a text file with one layer name per line")
    args_local = parser.parse_args()

    csv_path = Path(args_local.csv_path)
    out_dir = csv_path.parent

    df = pd.read_csv(csv_path)

    required_cols = {"layer", "target", "target_type", "r2_macro"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required column(s) in CSV: {missing}")

    # Ensure target_type normalization: scalar / vec3 / mat3x3
    # Try to coerce common synonyms if they appear
    mapping = {
        "scalar": "scalar",
        "vec3": "vec3",
        "vector": "vec3",
        "mat3x3": "mat3x3",
        "matrix": "mat3x3",
        "3x3": "mat3x3"
    }
    df["target_type"] = df["target_type"].map(lambda x: mapping.get(str(x).strip().lower(), str(x).strip().lower()))

    # r2_macro should be numeric
    df["r2_macro"] = pd.to_numeric(df["r2_macro"], errors="coerce")

    # Build layer order
    order_txt = Path(args_local.layer_order) if args_local.layer_order else None
    layer_order = build_layer_order(df, csv_path, order_txt)

    # Plot per group
    for ttype in ["scalar", "mat3x3", "vec3"]:
        plot_group(df, layer_order, ttype, out_dir)


if __name__ == "__main__":
    main()
