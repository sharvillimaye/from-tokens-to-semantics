#!/usr/bin/env python3
"""
Analyze and visualize polytope density results across all models and datasets.

Reads direct_pair_density.csv files from results/<model>/<dataset>/ and produces
publication-quality figures showing:
  1. Density vs layer curves (per model, all datasets overlaid)
  2. Normalized Hamming vs layer curves
  3. Cross-model comparison (density drop from early to late layers)
  4. Cross-dataset comparison
  5. Euclidean distance vs layer (to disentangle numerator/denominator effects)
  6. Model family comparison (Pythia vs OLMo scaling)
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, List

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

# =============================================================================
# STYLE
# =============================================================================

COLORS_DATASET = {
    "emotion": "#E69F00",    # orange
    "medical": "#56B4E9",    # sky blue
    "legal": "#009E73",      # green
    "scientific": "#CC79A7", # pink
    "verb": "#0072B2",       # blue
}

COLORS_MODEL = {
    "pythia-70m": "#E69F00",
    "pythia-1b": "#56B4E9",
    "pythia-6.9b": "#0072B2",
    "olmo-1b": "#009E73",
    "olmo-7b": "#CC79A7",
}

MODEL_ORDER = ["pythia-70m", "pythia-1b", "pythia-6.9b", "olmo-1b", "olmo-7b"]
DATASET_ORDER = ["emotion", "medical", "legal", "scientific", "verb"]
MODEL_DISPLAY = {
    "pythia-70m": "Pythia-70M",
    "pythia-1b": "Pythia-1B",
    "pythia-6.9b": "Pythia-6.9B",
    "olmo-1b": "OLMo-1B",
    "olmo-7b": "OLMo-7B",
}
MODEL_PARAMS = {
    "pythia-70m": 70e6,
    "pythia-1b": 1e9,
    "pythia-6.9b": 6.9e9,
    "olmo-1b": 1e9,
    "olmo-7b": 7e9,
}


def set_paper_style():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times", "Times New Roman", "DejaVu Serif"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.5,
        "lines.markersize": 4,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.3,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def save(fig, path: Path, close=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".pdf"), format="pdf", bbox_inches="tight", pad_inches=0.05)
    fig.savefig(path.with_suffix(".png"), format="png", dpi=300, bbox_inches="tight")
    if close:
        plt.close(fig)
    print(f"  saved: {path.with_suffix('.pdf')}")


# =============================================================================
# DATA LOADING
# =============================================================================

def load_all_results(results_dir: Path) -> pd.DataFrame:
    """Load all direct_pair_density.csv files into a single DataFrame."""
    rows = []
    for csv_path in sorted(results_dir.rglob("direct_pair_density.csv")):
        parts = csv_path.relative_to(results_dir).parts
        model, dataset = parts[0], parts[1]
        df = pd.read_csv(csv_path)
        df["model"] = model
        df["dataset"] = dataset
        rows.append(df)

    combined = pd.concat(rows, ignore_index=True)
    # Normalize layer to fractional depth (0 to 1)
    for model in combined["model"].unique():
        mask = combined["model"] == model
        n_layers = combined.loc[mask, "layer"].max() + 1
        combined.loc[mask, "layer_frac"] = combined.loc[mask, "layer"] / (n_layers - 1)
        combined.loc[mask, "n_total_layers"] = n_layers

    return combined


# =============================================================================
# FIGURE 1: Per-model density profiles (all datasets overlaid)
# =============================================================================

def fig_density_per_model(df: pd.DataFrame, outdir: Path):
    """One subplot per model: density_mean vs layer, colored by dataset."""
    models = [m for m in MODEL_ORDER if m in df["model"].unique()]
    n = len(models)

    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 3.5), sharey=False)
    if n == 1:
        axes = [axes]

    for i, model in enumerate(models):
        ax = axes[i]
        sub = df[df["model"] == model]

        for dataset in DATASET_ORDER:
            ds_data = sub[sub["dataset"] == dataset].sort_values("layer")
            if ds_data.empty:
                continue
            ax.plot(ds_data["layer"], ds_data["density_mean"],
                    color=COLORS_DATASET[dataset], marker="o", markersize=2.5,
                    linewidth=1.3, label=dataset.capitalize())
            ax.fill_between(ds_data["layer"],
                            ds_data["density_mean"] - ds_data["density_std"],
                            ds_data["density_mean"] + ds_data["density_std"],
                            color=COLORS_DATASET[dataset], alpha=0.08)

        ax.set_xlabel("Layer")
        ax.set_title(MODEL_DISPLAY[model], fontsize=11)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.set_ylabel("Polytope Density (ρ)")

    # Shared legend
    handles = [Line2D([0], [0], color=COLORS_DATASET[d], marker="o", markersize=4,
                       label=d.capitalize()) for d in DATASET_ORDER]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.08),
               ncol=len(DATASET_ORDER), frameon=False, fontsize=9)
    fig.suptitle("Polytope Boundary Density Across Layers", y=1.12, fontsize=13)
    plt.tight_layout()
    save(fig, outdir / "fig1_density_per_model")


# =============================================================================
# FIGURE 2: Normalized Hamming per model
# =============================================================================

def fig_norm_hamming_per_model(df: pd.DataFrame, outdir: Path):
    """Normalized Hamming distance vs layer, one subplot per model."""
    models = [m for m in MODEL_ORDER if m in df["model"].unique()]
    n = len(models)

    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 3.5), sharey=False)
    if n == 1:
        axes = [axes]

    for i, model in enumerate(models):
        ax = axes[i]
        sub = df[df["model"] == model]

        for dataset in DATASET_ORDER:
            ds_data = sub[sub["dataset"] == dataset].sort_values("layer")
            if ds_data.empty:
                continue
            ax.plot(ds_data["layer"], ds_data["normalized_hamming_mean"],
                    color=COLORS_DATASET[dataset], marker="o", markersize=2.5,
                    linewidth=1.3, label=dataset.capitalize())

        ax.set_xlabel("Layer")
        ax.set_title(MODEL_DISPLAY[model], fontsize=11)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.set_ylabel("Normalized Hamming Distance")

    handles = [Line2D([0], [0], color=COLORS_DATASET[d], marker="o", markersize=4,
                       label=d.capitalize()) for d in DATASET_ORDER]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.08),
               ncol=len(DATASET_ORDER), frameon=False, fontsize=9)
    fig.suptitle("Normalized Hamming Distance (Spline Code Divergence)", y=1.12, fontsize=13)
    plt.tight_layout()
    save(fig, outdir / "fig2_norm_hamming_per_model")


# =============================================================================
# FIGURE 3: Cross-model comparison at fractional depth
# =============================================================================

def fig_cross_model_fractional(df: pd.DataFrame, outdir: Path):
    """All models on same axes, x = fractional layer depth, averaged across datasets."""
    models = [m for m in MODEL_ORDER if m in df["model"].unique()]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    metrics = [
        ("density_mean", "Polytope Density (ρ)", axes[0]),
        ("normalized_hamming_mean", "Normalized Hamming", axes[1]),
        ("euclidean_mean", "Euclidean Distance", axes[2]),
    ]

    for metric, ylabel, ax in metrics:
        for model in models:
            sub = df[df["model"] == model]
            # Average across datasets per layer
            agg = sub.groupby("layer_frac")[metric].agg(["mean", "std"]).reset_index()
            agg = agg.sort_values("layer_frac")
            ax.plot(agg["layer_frac"], agg["mean"],
                    color=COLORS_MODEL[model], marker="o", markersize=3,
                    linewidth=1.8, label=MODEL_DISPLAY[model])
            ax.fill_between(agg["layer_frac"], agg["mean"] - agg["std"],
                            agg["mean"] + agg["std"],
                            color=COLORS_MODEL[model], alpha=0.1)

        ax.set_xlabel("Fractional Depth (layer / total)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="best")

    fig.suptitle("Cross-Model Comparison at Matched Fractional Depth", fontsize=13, y=1.02)
    plt.tight_layout()
    save(fig, outdir / "fig3_cross_model_fractional")


# =============================================================================
# FIGURE 4: Density decomposition — Hamming & Euclidean components
# =============================================================================

def fig_density_decomposition(df: pd.DataFrame, outdir: Path):
    """Show Hamming distance and Euclidean distance separately per model."""
    models = [m for m in MODEL_ORDER if m in df["model"].unique()]
    n = len(models)

    fig, axes = plt.subplots(2, n, figsize=(3.5 * n, 6), sharey="row")
    if n == 1:
        axes = axes.reshape(2, 1)

    for col, model in enumerate(models):
        sub = df[df["model"] == model]

        for dataset in DATASET_ORDER:
            ds_data = sub[sub["dataset"] == dataset].sort_values("layer")
            if ds_data.empty:
                continue
            # Hamming
            axes[0, col].plot(ds_data["layer"], ds_data["hamming_mean"],
                              color=COLORS_DATASET[dataset], marker="o", markersize=2,
                              linewidth=1.2)
            # Euclidean
            axes[1, col].plot(ds_data["layer"], ds_data["euclidean_mean"],
                              color=COLORS_DATASET[dataset], marker="o", markersize=2,
                              linewidth=1.2)

        axes[0, col].set_title(MODEL_DISPLAY[model], fontsize=10)
        axes[1, col].set_xlabel("Layer")
        axes[0, col].grid(True, alpha=0.3)
        axes[1, col].grid(True, alpha=0.3)

    axes[0, 0].set_ylabel("Hamming Distance")
    axes[1, 0].set_ylabel("Euclidean Distance\n(MLP Input)")

    handles = [Line2D([0], [0], color=COLORS_DATASET[d], marker="o", markersize=4,
                       label=d.capitalize()) for d in DATASET_ORDER]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.05),
               ncol=len(DATASET_ORDER), frameon=False, fontsize=9)
    fig.suptitle("Density Decomposition: Hamming (numerator) vs Euclidean (denominator)",
                 y=1.09, fontsize=12)
    plt.tight_layout()
    save(fig, outdir / "fig4_density_decomposition")


# =============================================================================
# FIGURE 5: Scaling law — summary statistics vs model size
# =============================================================================

def fig_scaling(df: pd.DataFrame, outdir: Path):
    """Plot summary metrics vs model parameters, split by family."""
    models = [m for m in MODEL_ORDER if m in df["model"].unique()]

    # Compute summary stats per model (averaged across datasets)
    summary_rows = []
    for model in models:
        sub = df[df["model"] == model]
        n_layers = int(sub["n_total_layers"].iloc[0])
        first_layer = sub[sub["layer"] == 0]
        last_layer = sub[sub["layer"] == n_layers - 1]
        mid_layer = sub[sub["layer"] == n_layers // 2]

        summary_rows.append({
            "model": model,
            "params": MODEL_PARAMS[model],
            "family": "Pythia" if "pythia" in model else "OLMo",
            "n_layers": n_layers,
            "density_first": first_layer["density_mean"].mean(),
            "density_last": last_layer["density_mean"].mean(),
            "density_mid": mid_layer["density_mean"].mean(),
            "density_drop": first_layer["density_mean"].mean() - last_layer["density_mean"].mean(),
            "density_drop_pct": (first_layer["density_mean"].mean() - last_layer["density_mean"].mean()) / first_layer["density_mean"].mean() * 100,
            "hamming_first": first_layer["normalized_hamming_mean"].mean(),
            "hamming_last": last_layer["normalized_hamming_mean"].mean(),
            "hamming_drop": first_layer["normalized_hamming_mean"].mean() - last_layer["normalized_hamming_mean"].mean(),
            "euclid_first": first_layer["euclidean_mean"].mean(),
            "euclid_last": last_layer["euclidean_mean"].mean(),
        })

    summary = pd.DataFrame(summary_rows)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    family_colors = {"Pythia": "#0072B2", "OLMo": "#D55E00"}
    family_markers = {"Pythia": "o", "OLMo": "s"}

    plots = [
        (axes[0, 0], "density_first", "Density (First Layer)"),
        (axes[0, 1], "density_last", "Density (Last Layer)"),
        (axes[0, 2], "density_drop_pct", "Density Drop (%)"),
        (axes[1, 0], "hamming_first", "Norm. Hamming (First Layer)"),
        (axes[1, 1], "hamming_last", "Norm. Hamming (Last Layer)"),
        (axes[1, 2], "hamming_drop", "Hamming Drop (First - Last)"),
    ]

    for ax, col, title in plots:
        for family in ["Pythia", "OLMo"]:
            fsub = summary[summary["family"] == family].sort_values("params")
            ax.plot(fsub["params"], fsub[col],
                    color=family_colors[family], marker=family_markers[family],
                    markersize=8, linewidth=1.5, label=family)
            for _, row in fsub.iterrows():
                ax.annotate(MODEL_DISPLAY[row["model"]],
                           (row["params"], row[col]),
                           textcoords="offset points", xytext=(5, 5),
                           fontsize=6, color=family_colors[family])

        ax.set_xscale("log")
        ax.set_xlabel("Parameters")
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle("Polytope Density Scaling Across Model Families", fontsize=13, y=1.02)
    plt.tight_layout()
    save(fig, outdir / "fig5_scaling")

    return summary


# =============================================================================
# FIGURE 6: Dataset comparison heatmap
# =============================================================================

def fig_dataset_heatmap(df: pd.DataFrame, outdir: Path):
    """Heatmap: model x dataset showing final-layer density and hamming."""
    models = [m for m in MODEL_ORDER if m in df["model"].unique()]

    # Build matrices
    density_mat = np.full((len(models), len(DATASET_ORDER)), np.nan)
    hamming_mat = np.full((len(models), len(DATASET_ORDER)), np.nan)

    for i, model in enumerate(models):
        sub = df[df["model"] == model]
        n_layers = int(sub["n_total_layers"].iloc[0])
        last = sub[sub["layer"] == n_layers - 1]
        for j, dataset in enumerate(DATASET_ORDER):
            ds = last[last["dataset"] == dataset]
            if not ds.empty:
                density_mat[i, j] = ds["density_mean"].values[0]
                hamming_mat[i, j] = ds["normalized_hamming_mean"].values[0]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, mat, title, cmap in [
        (axes[0], density_mat, "Final Layer Density (ρ)", "YlOrRd"),
        (axes[1], hamming_mat, "Final Layer Norm. Hamming", "YlGnBu"),
    ]:
        im = ax.imshow(mat, aspect="auto", cmap=cmap)
        ax.set_xticks(range(len(DATASET_ORDER)))
        ax.set_xticklabels([d.capitalize() for d in DATASET_ORDER], fontsize=9)
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels([MODEL_DISPLAY[m] for m in models], fontsize=9)
        ax.set_title(title, fontsize=11)

        # Annotate cells
        for i in range(len(models)):
            for j in range(len(DATASET_ORDER)):
                if np.isfinite(mat[i, j]):
                    ax.text(j, i, f"{mat[i, j]:.1f}" if mat[i, j] > 1 else f"{mat[i, j]:.3f}",
                            ha="center", va="center", fontsize=8,
                            color="white" if mat[i, j] > np.nanmedian(mat) else "black")
        plt.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle("Dataset Comparison: Final Layer Metrics", fontsize=13, y=1.02)
    plt.tight_layout()
    save(fig, outdir / "fig6_dataset_heatmap")


# =============================================================================
# FIGURE 7: Combined "story" figure — the key result
# =============================================================================

def fig_key_result(df: pd.DataFrame, outdir: Path):
    """
    2-panel figure showing the main finding:
    (a) Density vs fractional depth for all models (dataset-averaged)
    (b) Normalized Hamming vs fractional depth
    With annotation highlighting the key pattern.
    """
    models = [m for m in MODEL_ORDER if m in df["model"].unique()]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    for metric, ylabel, ax in [
        ("density_mean", "Polytope Boundary Density (ρ)", axes[0]),
        ("normalized_hamming_mean", "Normalized Hamming Distance", axes[1]),
    ]:
        for model in models:
            sub = df[df["model"] == model]
            agg = sub.groupby("layer_frac")[metric].mean().reset_index()
            agg = agg.sort_values("layer_frac")
            ax.plot(agg["layer_frac"], agg["mean"] if "mean" in agg.columns else agg[metric],
                    color=COLORS_MODEL[model], marker="o", markersize=4,
                    linewidth=2, label=MODEL_DISPLAY[model])

        ax.set_xlabel("Fractional Depth")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    axes[0].set_title("(a) Polytope Density Decreases Through Layers", fontsize=11)
    axes[1].set_title("(b) Synonym Pairs Cross More Boundaries Early On", fontsize=11)

    fig.suptitle("Synonym Pairs: Polytope Structure Across Model Scale",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    save(fig, outdir / "fig7_key_result")


# =============================================================================
# FIGURE 8: Per-model 2D (Hamming vs Euclidean) at select layers
# =============================================================================

def fig_hamming_vs_euclidean(df: pd.DataFrame, outdir: Path):
    """Scatter-like: each model gets early/mid/late layer comparison."""
    models = [m for m in MODEL_ORDER if m in df["model"].unique()]
    n = len(models)

    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 3.5))
    if n == 1:
        axes = [axes]

    layer_fracs = [0.0, 0.5, 1.0]
    layer_labels = ["First", "Middle", "Last"]
    layer_markers = ["o", "s", "D"]
    layer_colors_cycle = ["#4477AA", "#EE6677", "#228833"]

    for i, model in enumerate(models):
        ax = axes[i]
        sub = df[df["model"] == model]
        n_layers = int(sub["n_total_layers"].iloc[0])

        for lf, label, marker, color in zip(layer_fracs, layer_labels, layer_markers, layer_colors_cycle):
            layer_idx = int(round(lf * (n_layers - 1)))
            layer_data = sub[sub["layer"] == layer_idx]
            ax.scatter(layer_data["euclidean_mean"], layer_data["hamming_mean"],
                       c=color, marker=marker, s=50, label=f"L{layer_idx} ({label})",
                       edgecolors="white", linewidths=0.5, zorder=5)

        ax.set_xlabel("Euclidean Distance (MLP Input)")
        ax.set_title(MODEL_DISPLAY[model], fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="best")
        if i == 0:
            ax.set_ylabel("Hamming Distance")

    fig.suptitle("Hamming vs Euclidean at Different Depths", fontsize=12, y=1.05)
    plt.tight_layout()
    save(fig, outdir / "fig8_hamming_vs_euclidean")


# =============================================================================
# PRINT SUMMARY TABLE
# =============================================================================

def print_summary(df: pd.DataFrame):
    """Print a concise text summary of key metrics."""
    models = [m for m in MODEL_ORDER if m in df["model"].unique()]

    print("\n" + "=" * 90)
    print("POLYTOPE DENSITY RESULTS SUMMARY")
    print("=" * 90)

    print(f"\n{'Model':<15} {'Layers':>6} {'ρ(L0)':>8} {'ρ(mid)':>8} {'ρ(last)':>8} "
          f"{'H(L0)':>8} {'H(last)':>8} {'E(L0)':>8} {'E(last)':>8}")
    print("-" * 90)

    for model in models:
        sub = df[df["model"] == model]
        n_layers = int(sub["n_total_layers"].iloc[0])
        mid = n_layers // 2
        last = n_layers - 1

        l0 = sub[sub["layer"] == 0]
        lm = sub[sub["layer"] == mid]
        ll = sub[sub["layer"] == last]

        print(f"{MODEL_DISPLAY[model]:<15} {n_layers:>6} "
              f"{l0['density_mean'].mean():>8.1f} {lm['density_mean'].mean():>8.1f} "
              f"{ll['density_mean'].mean():>8.1f} "
              f"{l0['normalized_hamming_mean'].mean():>8.3f} "
              f"{ll['normalized_hamming_mean'].mean():>8.3f} "
              f"{l0['euclidean_mean'].mean():>8.1f} "
              f"{ll['euclidean_mean'].mean():>8.1f}")

    # Dataset breakdown
    print(f"\n\nPer-Dataset Final Layer Density:")
    print(f"{'Model':<15}", end="")
    for d in DATASET_ORDER:
        print(f" {d:>10}", end="")
    print()
    print("-" * 70)

    for model in models:
        sub = df[df["model"] == model]
        n_layers = int(sub["n_total_layers"].iloc[0])
        last = sub[sub["layer"] == n_layers - 1]
        print(f"{MODEL_DISPLAY[model]:<15}", end="")
        for d in DATASET_ORDER:
            ds = last[last["dataset"] == d]
            if not ds.empty:
                print(f" {ds['density_mean'].values[0]:>10.1f}", end="")
            else:
                print(f" {'---':>10}", end="")
        print()

    print("\n" + "=" * 90)


# =============================================================================
# MAIN
# =============================================================================

def main():
    set_paper_style()

    results_dir = Path(__file__).parent / "results"
    outdir = Path(__file__).parent / "figures"
    outdir.mkdir(parents=True, exist_ok=True)

    print("Loading results...")
    df = load_all_results(results_dir)
    print(f"Loaded {len(df)} rows: {df['model'].nunique()} models x {df['dataset'].nunique()} datasets")

    print_summary(df)

    print("\nGenerating figures...")
    fig_density_per_model(df, outdir)
    fig_norm_hamming_per_model(df, outdir)
    fig_cross_model_fractional(df, outdir)
    fig_density_decomposition(df, outdir)
    summary = fig_scaling(df, outdir)
    fig_dataset_heatmap(df, outdir)
    fig_key_result(df, outdir)
    fig_hamming_vs_euclidean(df, outdir)

    # Save summary table as CSV
    summary.to_csv(outdir / "scaling_summary.csv", index=False)
    print(f"\n  saved: {outdir / 'scaling_summary.csv'}")

    print(f"\nAll figures saved to {outdir.resolve()}")


if __name__ == "__main__":
    main()
