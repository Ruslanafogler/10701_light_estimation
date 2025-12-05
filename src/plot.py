#!/usr/bin/env python3
"""
Plot hyperparameter experiments for LightNet.

Reads experiment_summary.csv and produces:
- Euclidean error vs λ_phys
- Angular error vs learning rate
- Heatmap of angular error (batch size vs λ_phys)
- 3D scatter of hyperparameters
- Saves figures to: experiments/<model>/plots/
"""

import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)


def plot_euclid_vs_lambda(df, outdir):
    plt.figure(figsize=(7,5))

    for q in sorted(df.q.unique()):
        sub = df[df.q == q]
        plt.plot(
            sub["lambda_phys"],
            sub["Mean_Euclid"],
            marker='o',
            label=f"q={q}"
        )

    plt.xlabel("λ_phys")
    plt.ylabel("Mean Euclidean Error")
    plt.title("Effect of λ_phys on Euclidean Error")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(outdir / "euclid_vs_lambda.png")
    plt.close()


def plot_angle_vs_lr(df, outdir):
    plt.figure(figsize=(7,5))

    for bs in sorted(df.batch.unique()):
        sub = df[df.batch == bs]
        plt.plot(
            sub["lr"],
            sub["Mean_Angle"],
            marker='o',
            label=f"batch={bs}"
        )

    plt.xscale("log")
    plt.xlabel("Learning Rate (log scale)")
    plt.ylabel("Mean Angular Error (deg)")
    plt.title("Angular Error vs Learning Rate")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(outdir / "angle_vs_lr.png")
    plt.close()


def plot_heatmap(df, outdir):
    pivot = df.pivot_table(index="batch", columns="lambda_phys", values="Mean_Angle")

    plt.figure(figsize=(6,5))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis")
    plt.title("Angular Error Heatmap (batch vs λ_phys)")
    plt.xlabel("λ_phys")
    plt.ylabel("batch")
    plt.tight_layout()

    plt.savefig(outdir / "heatmap_angle.png")
    plt.close()


def plot_3d_scatter(df, outdir):
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection="3d")

    p = ax.scatter(
        df["lr"], df["lambda_phys"], df["Mean_Euclid"],
        c=df["Mean_Euclid"], cmap="coolwarm", s=60
    )

    ax.set_xlabel("Learning Rate")
    ax.set_ylabel("λ_phys")
    ax.set_zlabel("Euclidean Error")
    ax.set_title("3D Hyperparameter Interaction")

    fig.colorbar(p, ax=ax, shrink=0.6)
    plt.tight_layout()

    plt.savefig(outdir / "3d_hyperparams.png")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="sphere",
                        help="Model name: sphere, bunny, etc.")
    args = parser.parse_args()

    # ----------------------------------------------------------------------
    # Load CSV
    # ----------------------------------------------------------------------
    base_dir = Path("experiments") / args.model
    csv_path = base_dir / "experiment_summary.csv"
    plot_dir = base_dir / "plots"
    ensure_dir(plot_dir)

    print(f"📄 Loading: {csv_path}")
    df = pd.read_csv(csv_path)

    # ----------------------------------------------------------------------
    # Generate plots
    # ----------------------------------------------------------------------
    print("📊 Generating plots...")

    plot_euclid_vs_lambda(df, plot_dir)
    plot_angle_vs_lr(df, plot_dir)
    # plot_heatmap(df, plot_dir)
    plot_3d_scatter(df, plot_dir)

    # ----------------------------------------------------------------------
    # Report best configuration
    # ----------------------------------------------------------------------
    best_idx = df["Mean_Euclid"].idxmin()
    best = df.loc[best_idx]

    print("\n✨ BEST MODEL (lowest Euclidean error):")
    print(best)

    # ----------------------------------------------------------------------
    # Optional: print LaTeX table
    # ----------------------------------------------------------------------
    latex_table = df.to_latex(index=False)
    with open(plot_dir / "results_table.tex", "w") as f:
        f.write(latex_table)

    print(f"\n📁 Saved LaTeX table to: {plot_dir / 'results_table.tex'}")
    print(f"📁 All plots saved to: {plot_dir}")


if __name__ == "__main__":
    main()
