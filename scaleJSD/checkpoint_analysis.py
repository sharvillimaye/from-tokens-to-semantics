"""
Checkpoint Analysis for JSD Synonym Pairs

Analyzes how JSD decay pattern emerges over training by comparing
results across multiple model checkpoints.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# =============================================================================
# CONFIG
# =============================================================================

RUNS_DIR = Path(__file__).parent / "runs"
LAYERS = list(range(6))

# Checkpoint steps to analyze
CHECKPOINTS = [3000, 23000, 83000, 143000]

# Color scheme for checkpoints (light to dark as training progresses)
CHECKPOINT_COLORS = {
    3000: "#a6cee3",  # light blue
    23000: "#1f78b4",  # medium blue
    83000: "#33a02c",  # green
    143000: "#e31a1c",  # red (final)
}


# =============================================================================
# DATA LOADING
# =============================================================================


def load_checkpoint_results(dataset: str, step: int) -> pd.DataFrame:
    """Load JSD results for a specific checkpoint."""
    if step == 143000:
        # Final checkpoint is in the _templates directory
        run_dir = RUNS_DIR / f"{dataset}_jsd_templates"
    else:
        run_dir = RUNS_DIR / f"{dataset}_step{step}"

    if not run_dir.exists():
        return pd.DataFrame()

    jsonl_files = list(run_dir.glob("*_jsd_results.jsonl"))
    if not jsonl_files:
        return pd.DataFrame()

    records = []
    with jsonl_files[0].open("r") as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                freq_ratio = data.get("frequency_ratio", -1)
                log_ratio = np.log10(freq_ratio) if freq_ratio > 0 else np.nan

                row = {
                    "high_freq_ngram": data["high_freq_ngram"],
                    "low_freq_ngram": data["low_freq_ngram"],
                    "frequency_ratio": freq_ratio,
                    "log_ratio": log_ratio,
                }
                for layer, jsd in data.get("jsd_by_layer", {}).items():
                    row[f"jsd_layer_{layer}"] = jsd
                records.append(row)

    df = pd.DataFrame(records)
    df["checkpoint"] = step
    return df


def load_all_checkpoints(dataset: str) -> dict[int, pd.DataFrame]:
    """Load results for all checkpoints."""
    results = {}
    for step in CHECKPOINTS:
        df = load_checkpoint_results(dataset, step)
        if len(df) > 0:
            results[step] = df
            print(f"Loaded step {step}: {len(df)} pairs")
    return results


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================


def compute_decay_stats(df: pd.DataFrame) -> dict:
    """Compute decay statistics for a single checkpoint."""
    l0_mean = df["jsd_layer_0"].mean()
    l5_mean = df["jsd_layer_5"].mean()
    decay_ratio = l5_mean / l0_mean if l0_mean > 0 else np.nan

    layer_means = [df[f"jsd_layer_{L}"].mean() for L in LAYERS]

    return {
        "l0_mean": l0_mean,
        "l5_mean": l5_mean,
        "decay_ratio": decay_ratio,
        "layer_means": layer_means,
    }


def analyze_checkpoints(checkpoint_data: dict[int, pd.DataFrame]) -> dict:
    """Analyze patterns across checkpoints."""
    stats = {}
    for step, df in checkpoint_data.items():
        stats[step] = compute_decay_stats(df)
    return stats


# =============================================================================
# VISUALIZATION
# =============================================================================


def plot_jsd_by_checkpoint(
    checkpoint_data: dict[int, pd.DataFrame],
    stats: dict,
    output_dir: Path,
    dataset: str,
):
    """Plot mean JSD by layer for each checkpoint."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for step in sorted(checkpoint_data.keys()):
        layer_means = stats[step]["layer_means"]
        color = CHECKPOINT_COLORS.get(step, "gray")
        label = f"Step {step:,}"
        if step == 143000:
            label += " (final)"
        ax.plot(
            LAYERS, layer_means, marker="o", linewidth=2, markersize=8, color=color, label=label
        )

    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Mean JSD", fontsize=12)
    ax.set_title(f"JSD Decay Across Training Checkpoints\n({dataset} dataset)", fontsize=14)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xticks(LAYERS)
    ax.set_xticklabels([f"L{L}" for L in LAYERS])

    plt.tight_layout()
    out_path = output_dir / "jsd_by_checkpoint.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_decay_over_training(
    stats: dict,
    output_dir: Path,
    dataset: str,
):
    """Plot how decay ratio changes over training."""
    steps = sorted(stats.keys())
    decay_ratios = [stats[s]["decay_ratio"] for s in steps]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Decay ratio over training
    ax1 = axes[0]
    colors = [CHECKPOINT_COLORS.get(s, "gray") for s in steps]
    ax1.bar(range(len(steps)), decay_ratios, color=colors)
    ax1.set_xticks(range(len(steps)))
    ax1.set_xticklabels([f"{s // 1000}k" for s in steps])
    ax1.set_xlabel("Training Step", fontsize=12)
    ax1.set_ylabel("Decay Ratio (L5/L0)", fontsize=12)
    ax1.set_title("JSD Decay Ratio Over Training", fontsize=14)
    ax1.set_ylim(0, 1)

    # Add value labels
    for i, (step, ratio) in enumerate(zip(steps, decay_ratios)):
        ax1.text(i, ratio + 0.02, f"{ratio:.2f}", ha="center", fontsize=10)

    # Right: L0 and L5 means over training
    ax2 = axes[1]
    l0_means = [stats[s]["l0_mean"] for s in steps]
    l5_means = [stats[s]["l5_mean"] for s in steps]

    x = range(len(steps))
    width = 0.35
    ax2.bar([i - width / 2 for i in x], l0_means, width, label="Layer 0", color="#1f78b4")
    ax2.bar([i + width / 2 for i in x], l5_means, width, label="Layer 5", color="#e31a1c")

    ax2.set_xticks(range(len(steps)))
    ax2.set_xticklabels([f"{s // 1000}k" for s in steps])
    ax2.set_xlabel("Training Step", fontsize=12)
    ax2.set_ylabel("Mean JSD", fontsize=12)
    ax2.set_title("Layer 0 vs Layer 5 JSD Over Training", fontsize=14)
    ax2.legend()

    plt.tight_layout()
    out_path = output_dir / "decay_over_training.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_correlation_evolution(
    checkpoint_data: dict[int, pd.DataFrame],
    output_dir: Path,
    dataset: str,
):
    """Plot how correlation with frequency ratio evolves over training."""
    steps = sorted(checkpoint_data.keys())

    # Compute correlations for each checkpoint and layer
    correlations = {step: [] for step in steps}
    p_values = {step: [] for step in steps}

    for step in steps:
        df = checkpoint_data[step]
        for L in LAYERS:
            valid = df["log_ratio"].notna() & df[f"jsd_layer_{L}"].notna()
            if valid.sum() > 2:
                r, p = spearmanr(df.loc[valid, "log_ratio"], df.loc[valid, f"jsd_layer_{L}"])
            else:
                r, p = np.nan, np.nan
            correlations[step].append(r)
            p_values[step].append(p)

    # Plot: Correlation by layer for each checkpoint
    fig, ax = plt.subplots(figsize=(10, 6))

    for step in steps:
        color = CHECKPOINT_COLORS.get(step, "gray")
        label = f"Step {step:,}"
        if step == 143000:
            label += " (final)"

        # Mark significant correlations
        rs = correlations[step]
        ps = p_values[step]

        ax.plot(LAYERS, rs, marker="o", linewidth=2, markersize=8, color=color, label=label)

        # Add stars for significant correlations
        for L, (r, p) in enumerate(zip(rs, ps)):
            if p < 0.05:
                ax.annotate(
                    "*",
                    (L, r),
                    textcoords="offset points",
                    xytext=(0, 5),
                    ha="center",
                    fontsize=14,
                    color=color,
                )

    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Spearman Correlation (JSD vs log freq ratio)", fontsize=12)
    ax.set_title(f"Correlation Evolution Over Training\n({dataset} dataset)", fontsize=14)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xticks(LAYERS)
    ax.set_xticklabels([f"L{L}" for L in LAYERS])
    ax.set_ylim(-0.6, 0.8)

    plt.tight_layout()
    out_path = output_dir / "correlation_evolution.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def generate_summary_table(stats: dict, output_dir: Path):
    """Generate summary table of checkpoint statistics."""
    rows = []
    for step in sorted(stats.keys()):
        s = stats[step]
        rows.append(
            {
                "Checkpoint": f"step{step}",
                "L0 Mean": f"{s['l0_mean']:.4f}",
                "L5 Mean": f"{s['l5_mean']:.4f}",
                "Decay (L5/L0)": f"{s['decay_ratio']:.3f}",
            }
        )

    df = pd.DataFrame(rows)

    # Save as CSV
    csv_path = output_dir / "checkpoint_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # Print table
    print("\n" + "=" * 60)
    print("CHECKPOINT SUMMARY")
    print("=" * 60)
    print(df.to_string(index=False))

    return df


# =============================================================================
# MAIN
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Analyze JSD across training checkpoints")
    parser.add_argument(
        "--dataset",
        type=str,
        default="verb",
        help="Dataset name (default: verb)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: runs/checkpoint_analysis/)",
    )
    args = parser.parse_args()

    # Output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = RUNS_DIR / "checkpoint_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.dataset} dataset across checkpoints...")
    checkpoint_data = load_all_checkpoints(args.dataset)

    if len(checkpoint_data) < 2:
        print(f"Need at least 2 checkpoints, found {len(checkpoint_data)}")
        return

    print(f"\nAnalyzing {len(checkpoint_data)} checkpoints...")
    stats = analyze_checkpoints(checkpoint_data)

    # Generate summary table
    generate_summary_table(stats, output_dir)

    # Generate plots
    print("\nGenerating plots...")
    plot_jsd_by_checkpoint(checkpoint_data, stats, output_dir, args.dataset)
    plot_decay_over_training(stats, output_dir, args.dataset)
    plot_correlation_evolution(checkpoint_data, output_dir, args.dataset)

    # Print key findings
    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)

    steps = sorted(stats.keys())
    early_decay = stats[steps[0]]["decay_ratio"]
    final_decay = stats[steps[-1]]["decay_ratio"]

    print(f"\n1. DECAY PATTERN EMERGENCE:")
    print(f"   Early (step {steps[0]}): decay ratio = {early_decay:.3f}")
    print(f"   Final (step {steps[-1]}): decay ratio = {final_decay:.3f}")
    print(
        f"   Change: {early_decay:.3f} → {final_decay:.3f} ({(1 - final_decay / early_decay) * 100:.1f}% reduction)"
    )

    print(f"\n2. LAYER 5 JSD REDUCTION:")
    early_l5 = stats[steps[0]]["l5_mean"]
    final_l5 = stats[steps[-1]]["l5_mean"]
    print(f"   Early: L5 mean = {early_l5:.4f}")
    print(f"   Final: L5 mean = {final_l5:.4f}")
    print(f"   Reduction: {(1 - final_l5 / early_l5) * 100:.1f}%")

    print(f"\n3. INTERPRETATION:")
    if final_decay < early_decay * 0.5:
        print(f"   Strong evidence that JSD decay emerges during training.")
        print(
            f"   The model learns to abstract synonyms to similar representations in deeper layers."
        )
    else:
        print(f"   Moderate decay pattern change during training.")

    print(f"\nAll outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
