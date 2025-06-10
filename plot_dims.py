#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import shutil
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

def load_mat(runs_dir, dim, mat):
    path = os.path.join(runs_dir, str(dim), f"{mat}.csv")
    df = pd.read_csv(path, index_col=0)
    df.columns = range(df.shape[1])
    return df.astype(float)

def main():
    parser = argparse.ArgumentParser(
        description="Search for optimal ICA dimensionality from a runs directory"
    )
    parser.add_argument(
        "--runs_dir", "-r",
        required=True,
        help="Path containing subdirs 50,51,... each with M.csv and A.csv"
    )
    args = parser.parse_args()
    runs_dir = args.runs_dir

    # find all numeric subdirs
    dims = sorted(
        int(d) for d in os.listdir(runs_dir)
        if os.path.isdir(os.path.join(runs_dir, d)) and d.isdigit()
    )

    # load all M and A
    M_data = [load_mat(runs_dir, d,"tmp\\robust_M_consensus") for d in dims]
    A_data = [load_mat(runs_dir, d, "tmp\\robust_A_consensus") for d in dims]

    # trim off any all‐zero final A
    while np.allclose(A_data[-1], 0, atol=0.01):
        M_data.pop(); A_data.pop(); dims.pop()

    final_m = M_data[-1]
    n_components = [m.shape[1] for m in M_data]

    # compute statistics at each dim
    thresh = 0.7
    n_final_mods = []
    n_single_genes = []
    for m in M_data:
        # cosine similarity to final_m
        l2_final = np.linalg.norm(final_m.values, axis=0)
        l2_m     = np.linalg.norm(m.values,      axis=0)
        dist = (np.abs(final_m.values.T @ m.values)
                / l2_final[:,None] / l2_m[None,:])
        n_final_mods.append(int((dist > thresh).sum()))

        # count single‐gene modules
        cnt = 0
        for c in m.columns:
            col = m[c].abs().sort_values(ascending=False)
            if col.iloc[0] > 2 * col.iloc[1]:
                cnt += 1
        n_single_genes.append(cnt)

    non_single = np.array(n_components) - np.array(n_single_genes)

    # build DataFrame
    DF = pd.DataFrame({
        "Robust Components":    n_components,
        "Final Components":     n_final_mods,
        "Multi-gene Components": non_single,
        "Single Gene Components": n_single_genes
    }, index=dims)

    # pick optimal: first dim with Final >= Multi-gene
    mask = DF["Final Components"] >= DF["Multi-gene Components"]
    dimensionality = int(DF[mask].index[0])

    print("Optimal Dimensionality:", dimensionality)

    import seaborn as sns
    sns.set_style("whitegrid")               # 灰白网格背景
    sns.set_context("talk")                  # 字体/粗细和示例接近

    colors = {
        "Final":        "#f5a623",  # 橙黄
        "Robust":       "#009e73",  # 草绿
        "Multi":        "#0082c8",  # 天蓝
        "Single":       "#cc79a7"   # 玫红
    }

    fig, ax = plt.subplots()

    ax.plot(dims, DF["Final Components"],
            color=colors["Final"],  linewidth=3,
            label="Final Components")

    ax.plot(dims, DF["Robust Components"],
            color=colors["Robust"], linewidth=3,
            label="Robust Components")

    ax.plot(dims, DF["Multi-gene Components"],
            color=colors["Multi"],  linewidth=3,
            label="Multi gene Components")

    ax.plot(dims, DF["Single Gene Components"],
            color=colors["Single"], linewidth=3,
            label="Single Gene Components")

    # 垂直虚线标记最佳维度
    ax.axvline(dimensionality, color="k",
            linestyle="--", linewidth=2)

    ax.set_xlabel("Dimensionality")
    ax.set_ylabel("Component Count")



    plt.tight_layout()
    pdf_out = os.path.join(runs_dir, "dimension_analysis.pdf")
    fig.savefig(pdf_out, transparent=True)

if __name__ == "__main__":
    main()
