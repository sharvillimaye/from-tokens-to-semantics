"""
Visualization for JSD Synonym Pairs Analysis

Generates plots to analyze the relationship between frequency ratios
and JSD divergence across model layers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

# =============================================================================
# PLOTTING SETUP
# =============================================================================


def setup_plotting():
    """Configure matplotlib for clean, publication-ready plots."""
    mpl.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


# =============================================================================
# DATA LOADING
# =============================================================================


def load_results(results_path: Path) -> pd.DataFrame:
    """Load JSD results from JSONL file into DataFrame."""
    records = []
    with results_path.open("r") as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                # Flatten jsd_by_layer
                row = {
                    "high_freq_ngram": data["high_freq_ngram"],
                    "low_freq_ngram": data["low_freq_ngram"],
                    "high_freq_count": data.get("high_freq_count", -1),
                    "low_freq_count": data.get("low_freq_count", -1),
                    "frequency_ratio": data.get("frequency_ratio", -1),
                    "log_ratio": data.get("log_ratio", -1),
                    "category": data.get("category", "unknown"),
                }
                for layer, jsd in data.get("jsd_by_layer", {}).items():
                    row[f"jsd_layer_{layer}"] = jsd
                records.append(row)

    return pd.DataFrame(records)


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================


def plot_jsd_vs_log_ratio(
    df: pd.DataFrame,
    output_dir: Path,
    title_prefix: str = "",
):
    """
    Create scatter plots of JSD vs log(frequency_ratio) for each layer.

    Includes correlation statistics (Spearman and Pearson).
    """
    # Find JSD columns
    jsd_cols = [c for c in df.columns if c.startswith("jsd_layer_")]
    layers = sorted([int(c.split("_")[-1]) for c in jsd_cols])

    n_layers = len(layers)
    n_cols = min(3, n_layers)
    n_rows = (n_layers + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 4 * n_rows))
    if n_layers == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)

    x = df["log_ratio"].values

    for idx, layer in enumerate(layers):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row, col]

        y = df[f"jsd_layer_{layer}"].values

        # Scatter plot
        ax.scatter(x, y, alpha=0.6, s=40, edgecolors="white", linewidth=0.5)

        # Fit line
        valid = np.isfinite(x) & np.isfinite(y)
        if valid.sum() > 2:
            z = np.polyfit(x[valid], y[valid], 1)
            p = np.poly1d(z)
            x_line = np.linspace(x[valid].min(), x[valid].max(), 100)
            ax.plot(x_line, p(x_line), "r--", alpha=0.7, linewidth=1.5)

            # Correlation stats
            r_spearman, p_spearman = spearmanr(x[valid], y[valid])
            r_pearson, p_pearson = pearsonr(x[valid], y[valid])

            stats_text = f"Spearman r={r_spearman:.3f} (p={p_spearman:.3f})\nPearson r={r_pearson:.3f} (p={p_pearson:.3f})"
            ax.text(
                0.05,
                0.95,
                stats_text,
                transform=ax.transAxes,
                fontsize=8,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )

        ax.set_xlabel("log₁₀(frequency ratio)")
        ax.set_ylabel("JSD")
        ax.set_title(f"Layer {layer}")

    # Hide unused subplots
    for idx in range(len(layers), n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row, col].set_visible(False)

    fig.suptitle(f"{title_prefix}JSD vs Frequency Ratio by Layer", fontsize=14, y=1.02)
    plt.tight_layout()

    out_path = output_dir / "jsd_vs_log_ratio.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_jsd_heatmap(
    df: pd.DataFrame,
    output_dir: Path,
    title_prefix: str = "",
):
    """
    Create heatmap of JSD values.

    Rows = pairs (sorted by frequency ratio), Columns = layers.
    """
    jsd_cols = [c for c in df.columns if c.startswith("jsd_layer_")]
    layers = sorted([int(c.split("_")[-1]) for c in jsd_cols])

    # Sort by frequency ratio
    df_sorted = df.sort_values("frequency_ratio", ascending=False).reset_index(drop=True)

    # Extract JSD matrix
    jsd_matrix = df_sorted[[f"jsd_layer_{L}" for L in layers]].values

    # Create labels
    pair_labels = [
        f"{r['high_freq_ngram']} / {r['low_freq_ngram']}" for _, r in df_sorted.iterrows()
    ]

    # Truncate labels if too long
    pair_labels = [l[:30] + "..." if len(l) > 30 else l for l in pair_labels]

    fig, ax = plt.subplots(figsize=(8, max(6, len(df) * 0.25)))

    im = ax.imshow(jsd_matrix, aspect="auto", cmap="viridis", interpolation="nearest")

    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels([f"L{L}" for L in layers])
    ax.set_xlabel("Layer")

    # Only show yticks if not too many
    if len(df) <= 30:
        ax.set_yticks(range(len(pair_labels)))
        ax.set_yticklabels(pair_labels, fontsize=8)
    else:
        ax.set_ylabel(f"Pairs (n={len(df)}, sorted by freq ratio)")

    ax.set_title(f"{title_prefix}JSD by Pair and Layer\n(sorted by frequency ratio, high to low)")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("JSD")

    plt.tight_layout()

    out_path = output_dir / "jsd_heatmap.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_jsd_by_layer(
    df: pd.DataFrame,
    output_dir: Path,
    title_prefix: str = "",
):
    """
    Box plot of JSD distribution by layer.
    """
    jsd_cols = [c for c in df.columns if c.startswith("jsd_layer_")]
    layers = sorted([int(c.split("_")[-1]) for c in jsd_cols])

    data = [df[f"jsd_layer_{L}"].dropna().values for L in layers]

    fig, ax = plt.subplots(figsize=(8, 5))

    bp = ax.boxplot(data, labels=[f"Layer {L}" for L in layers], patch_artist=True)

    # Color boxes
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(layers)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xlabel("Layer")
    ax.set_ylabel("JSD")
    ax.set_title(f"{title_prefix}JSD Distribution by Layer")

    # Add mean line
    means = [d.mean() for d in data]
    ax.plot(range(1, len(layers) + 1), means, "ro-", markersize=6, label="Mean", alpha=0.7)
    ax.legend()

    plt.tight_layout()

    out_path = output_dir / "jsd_by_layer_boxplot.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_jsd_by_category(
    df: pd.DataFrame,
    output_dir: Path,
    title_prefix: str = "",
):
    """
    Box plot of JSD by category (if multiple categories exist).
    """
    if "category" not in df.columns:
        return

    categories = df["category"].unique()
    if len(categories) <= 1:
        return

    # Use mean JSD across layers
    jsd_cols = [c for c in df.columns if c.startswith("jsd_layer_")]
    df["jsd_mean"] = df[jsd_cols].mean(axis=1)

    fig, ax = plt.subplots(figsize=(10, 5))

    data = [df[df["category"] == cat]["jsd_mean"].dropna().values for cat in categories]

    bp = ax.boxplot(data, labels=categories, patch_artist=True)

    colors = plt.cm.Set2(np.linspace(0, 1, len(categories)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xlabel("Category")
    ax.set_ylabel("Mean JSD (across layers)")
    ax.set_title(f"{title_prefix}JSD Distribution by Category")
    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()

    out_path = output_dir / "jsd_by_category.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_summary_stats(
    df: pd.DataFrame,
    output_dir: Path,
    title_prefix: str = "",
):
    """
    Create summary statistics table as image.
    """
    jsd_cols = [c for c in df.columns if c.startswith("jsd_layer_")]
    layers = sorted([int(c.split("_")[-1]) for c in jsd_cols])

    stats = []
    for L in layers:
        col = f"jsd_layer_{L}"
        vals = df[col].dropna()

        # Correlation with log_ratio
        valid = df["log_ratio"].notna() & df[col].notna()
        if valid.sum() > 2:
            r, p = spearmanr(df.loc[valid, "log_ratio"], df.loc[valid, col])
        else:
            r, p = np.nan, np.nan

        stats.append(
            {
                "Layer": L,
                "Mean JSD": f"{vals.mean():.6f}",
                "Std JSD": f"{vals.std():.6f}",
                "Min": f"{vals.min():.6f}",
                "Max": f"{vals.max():.6f}",
                "Spearman r": f"{r:.3f}",
                "p-value": f"{p:.4f}" if p < 0.05 else f"{p:.3f}",
            }
        )

    stats_df = pd.DataFrame(stats)

    # Save as CSV
    stats_path = output_dir / "summary_stats.csv"
    stats_df.to_csv(stats_path, index=False)
    print(f"Saved: {stats_path}")

    # Create table figure
    fig, ax = plt.subplots(figsize=(10, 2 + len(layers) * 0.4))
    ax.axis("off")

    table = ax.table(
        cellText=stats_df.values,
        colLabels=stats_df.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    ax.set_title(f"{title_prefix}Summary Statistics", fontsize=14, pad=20)

    plt.tight_layout()

    out_path = output_dir / "summary_stats.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# =============================================================================
# MAIN
# =============================================================================


def generate_all_plots(
    results_path: Path,
    output_dir: Path,
    title_prefix: str = "",
):
    """Generate all visualization plots."""
    setup_plotting()

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from {results_path}...")
    df = load_results(results_path)
    print(f"Loaded {len(df)} pairs")

    if len(df) == 0:
        print("No data to visualize!")
        return

    print("\nGenerating plots...")

    plot_jsd_vs_log_ratio(df, output_dir, title_prefix)
    plot_jsd_heatmap(df, output_dir, title_prefix)
    plot_jsd_by_layer(df, output_dir, title_prefix)
    plot_jsd_by_category(df, output_dir, title_prefix)
    plot_summary_stats(df, output_dir, title_prefix)

    print(f"\nAll plots saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Visualize JSD results")
    parser.add_argument(
        "--input", type=str, required=True, help="Path to results directory or JSONL file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for plots (default: {input}/plots)",
    )
    parser.add_argument("--title", type=str, default="", help="Title prefix for plots")
    args = parser.parse_args()

    input_path = Path(args.input)

    # Find results file
    if input_path.is_file():
        results_path = input_path
        output_dir = input_path.parent / "plots" if args.output is None else Path(args.output)
    else:
        # Look for JSONL file in directory
        jsonl_files = list(input_path.glob("*_jsd_results.jsonl"))
        if not jsonl_files:
            print(f"No *_jsd_results.jsonl files found in {input_path}")
            return
        results_path = jsonl_files[0]
        output_dir = input_path / "plots" if args.output is None else Path(args.output)

    # Extract dataset name for title
    title_prefix = args.title
    if not title_prefix:
        dataset_name = results_path.stem.replace("_jsd_results", "").replace("_", " ").title()
        title_prefix = f"{dataset_name}: "

    generate_all_plots(results_path, output_dir, title_prefix)


if __name__ == "__main__":
    main()
