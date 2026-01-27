"""
Generate JSD decay line plots across models.

The key visualization showing how JSD decays across layers for different models.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# =============================================================================
# CONFIG
# =============================================================================

RUNS_DIR = Path(__file__).parent / "runs"

# Model configurations with new directory structure
MODELS = {
    "pythia-70m": {
        "dir": "pythia-70m",
        "num_layers": 6,
        "color": "#1f78b4",
        "linestyle": "-",
        "marker": "o",
    },
    "olmo-1b": {
        "dir": "olmo-1b",
        "num_layers": 16,
        "color": "#33a02c",
        "linestyle": "-",
        "marker": "s",
    },
    "olmo-7b": {
        "dir": "olmo-7b",
        "num_layers": 32,
        "color": "#e31a1c",
        "linestyle": "-",
        "marker": "^",
    },
}

DATASETS = ["emotion", "medical", "legal", "scientific", "verb"]


# =============================================================================
# DATA LOADING
# =============================================================================


def load_results(model: str, dataset: str) -> pd.DataFrame:
    """Load JSD results for a model/dataset combination."""
    config = MODELS[model]
    run_dir = RUNS_DIR / config["dir"] / dataset

    if not run_dir.exists():
        return pd.DataFrame()

    # Find the results file
    jsonl_files = list(run_dir.glob("*_jsd_results.jsonl"))
    if not jsonl_files:
        return pd.DataFrame()

    records = []
    with jsonl_files[0].open("r") as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                row = {}
                for layer, jsd in data.get("jsd_by_layer", {}).items():
                    row[f"jsd_layer_{layer}"] = jsd
                records.append(row)

    return pd.DataFrame(records)


def get_layer_means(model: str, dataset: str) -> tuple[list, list]:
    """Get layer indices and mean JSD values for a model/dataset."""
    df = load_results(model, dataset)
    if len(df) == 0:
        return [], []

    num_layers = MODELS[model]["num_layers"]
    layers = list(range(num_layers))
    means = []

    for L in layers:
        col = f"jsd_layer_{L}"
        if col in df.columns:
            means.append(df[col].mean())
        else:
            means.append(np.nan)

    return layers, means


# =============================================================================
# PLOTTING
# =============================================================================


def plot_decay_all_models(output_dir: Path):
    """Generate the main decay plot showing all models on one figure."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create figure with subplots for each dataset
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, dataset in enumerate(DATASETS):
        ax = axes[idx]

        for model, config in MODELS.items():
            layers, means = get_layer_means(model, dataset)
            if not layers:
                continue

            # Normalize layer positions to [0, 1] for comparison
            norm_layers = np.array(layers) / (len(layers) - 1)

            ax.plot(
                norm_layers,
                means,
                marker=config["marker"],
                linestyle=config["linestyle"],
                color=config["color"],
                linewidth=2,
                markersize=6,
                label=model,
                alpha=0.8,
            )

        ax.set_xlabel("Normalized Layer Position (0=first, 1=last)")
        ax.set_ylabel("Mean JSD")
        ax.set_title(f"{dataset.capitalize()}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 0.55)

    # Hide unused subplot
    axes[-1].set_visible(False)

    fig.suptitle("JSD Decay Across Model Layers\n(Pythia-70M vs OLMo-1B vs OLMo-7B)", fontsize=14)
    plt.tight_layout()

    out_path = output_dir / "jsd_decay_all_models.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_decay_averaged(output_dir: Path):
    """Generate decay plot averaged across all datasets."""
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 7))

    for model, config in MODELS.items():
        all_means = []

        for dataset in DATASETS:
            layers, means = get_layer_means(model, dataset)
            if means:
                # Normalize to [0, 1] range
                norm_layers = np.array(layers) / (len(layers) - 1)
                all_means.append((norm_layers, means))

        if not all_means:
            continue

        # Interpolate all to same normalized positions and average
        norm_positions = np.linspace(0, 1, 50)
        interpolated = []

        for norm_layers, means in all_means:
            interp = np.interp(norm_positions, norm_layers, means)
            interpolated.append(interp)

        avg_means = np.mean(interpolated, axis=0)
        std_means = np.std(interpolated, axis=0)

        ax.plot(
            norm_positions,
            avg_means,
            linestyle=config["linestyle"],
            color=config["color"],
            linewidth=3,
            label=model,
            alpha=0.9,
        )

        ax.fill_between(
            norm_positions,
            avg_means - std_means,
            avg_means + std_means,
            color=config["color"],
            alpha=0.2,
        )

    ax.set_xlabel("Normalized Layer Position (0=first, 1=last)", fontsize=12)
    ax.set_ylabel("Mean JSD (averaged across datasets)", fontsize=12)
    ax.set_title(
        "JSD Decay Pattern: Model Comparison\n(shaded region = ±1 std across datasets)",
        fontsize=14,
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 0.5)

    plt.tight_layout()

    out_path = output_dir / "jsd_decay_averaged.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_decay_summary_table(output_dir: Path):
    """Generate summary statistics table."""
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for model, config in MODELS.items():
        for dataset in DATASETS:
            layers, means = get_layer_means(model, dataset)
            if not means:
                continue

            first_jsd = means[0]
            last_jsd = means[-1]
            decay_ratio = last_jsd / first_jsd if first_jsd > 0 else np.nan
            min_jsd = min(means)
            min_layer = means.index(min_jsd)

            rows.append(
                {
                    "Model": model,
                    "Dataset": dataset,
                    "Layers": config["num_layers"],
                    "First JSD": f"{first_jsd:.4f}",
                    "Last JSD": f"{last_jsd:.4f}",
                    "Min JSD": f"{min_jsd:.4f}",
                    "Min Layer": min_layer,
                    "Decay Ratio": f"{decay_ratio:.3f}",
                }
            )

    df = pd.DataFrame(rows)

    # Save as CSV
    csv_path = output_dir / "decay_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("DECAY SUMMARY")
    print("=" * 80)
    print(df.to_string(index=False))

    # Compute averages per model
    print("\n" + "-" * 80)
    print("AVERAGE DECAY RATIOS BY MODEL")
    print("-" * 80)

    for model in MODELS:
        model_rows = [r for r in rows if r["Model"] == model]
        if model_rows:
            avg_decay = np.mean([float(r["Decay Ratio"]) for r in model_rows])
            print(f"{model}: {avg_decay:.3f}")


# =============================================================================
# MAIN
# =============================================================================


def main():
    output_dir = RUNS_DIR / "analysis" / "decay_plots"

    print("Generating JSD decay plots...")

    plot_decay_all_models(output_dir)
    plot_decay_averaged(output_dir)
    plot_decay_summary_table(output_dir)

    print(f"\nAll plots saved to {output_dir}")


if __name__ == "__main__":
    main()
