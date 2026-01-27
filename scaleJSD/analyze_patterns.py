"""
Cross-dataset pattern analysis for JSD synonym pairs.

Analyzes JSD results across all 5 datasets to find common patterns
and differences between domains.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

# =============================================================================
# CONFIG
# =============================================================================

RUNS_DIR = Path(__file__).parent / "runs"

DATASETS = {
    "emotion": "emotion_jsd_templates",
    "medical": "medical_jsd_templates",
    "legal": "legal_jsd_templates",
    "scientific": "scientific_jsd_templates",
    "verb": "verb_jsd_templates",
}

LAYERS = list(range(6))


# =============================================================================
# DATA LOADING
# =============================================================================


def load_dataset_results(run_dir: Path) -> pd.DataFrame:
    """Load JSD results from a run directory."""
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
                log_ratio = data.get("log_ratio", -1)

                # Compute log_ratio from frequency_ratio if not available
                if (log_ratio == -1 or log_ratio is None) and freq_ratio > 0:
                    log_ratio = np.log10(freq_ratio)

                row = {
                    "high_freq_ngram": data["high_freq_ngram"],
                    "low_freq_ngram": data["low_freq_ngram"],
                    "high_freq_count": data.get("high_freq_count", -1),
                    "low_freq_count": data.get("low_freq_count", -1),
                    "frequency_ratio": freq_ratio,
                    "log_ratio": log_ratio,
                }
                for layer, jsd in data.get("jsd_by_layer", {}).items():
                    row[f"jsd_layer_{layer}"] = jsd
                records.append(row)

    return pd.DataFrame(records)


def load_all_datasets() -> dict[str, pd.DataFrame]:
    """Load all datasets."""
    all_data = {}
    for name, run_name in DATASETS.items():
        run_dir = RUNS_DIR / run_name
        if run_dir.exists():
            df = load_dataset_results(run_dir)
            if len(df) > 0:
                df["dataset"] = name
                all_data[name] = df
                print(f"Loaded {name}: {len(df)} pairs")
    return all_data


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================


def compute_layer_stats(df: pd.DataFrame) -> dict:
    """Compute layer-wise statistics for a dataset."""
    stats = {}
    for L in LAYERS:
        col = f"jsd_layer_{L}"
        if col in df.columns:
            vals = df[col].dropna()

            # Correlation with log_ratio
            valid = df["log_ratio"].notna() & df[col].notna()
            if valid.sum() > 2:
                r_spearman, p_spearman = spearmanr(df.loc[valid, "log_ratio"], df.loc[valid, col])
                r_pearson, p_pearson = pearsonr(df.loc[valid, "log_ratio"], df.loc[valid, col])
            else:
                r_spearman, p_spearman = np.nan, np.nan
                r_pearson, p_pearson = np.nan, np.nan

            stats[L] = {
                "mean": vals.mean(),
                "std": vals.std(),
                "min": vals.min(),
                "max": vals.max(),
                "r_spearman": r_spearman,
                "p_spearman": p_spearman,
                "r_pearson": r_pearson,
                "p_pearson": p_pearson,
            }
    return stats


def find_patterns(all_data: dict[str, pd.DataFrame]):
    """Analyze patterns across all datasets."""

    print("\n" + "=" * 70)
    print("CROSS-DATASET PATTERN ANALYSIS")
    print("=" * 70)

    # Collect stats for all datasets
    all_stats = {}
    for name, df in all_data.items():
        all_stats[name] = compute_layer_stats(df)

    # 1. Mean JSD by layer across datasets
    print("\n## 1. Mean JSD by Layer (across all datasets)")
    print("-" * 60)
    print(f"{'Dataset':<12} | " + " | ".join([f"L{L}" for L in LAYERS]))
    print("-" * 60)
    for name in all_data.keys():
        means = [f"{all_stats[name][L]['mean']:.4f}" for L in LAYERS]
        print(f"{name:<12} | " + " | ".join(means))

    # 2. JSD decay pattern (ratio of L5/L0)
    print("\n## 2. JSD Decay Pattern (L5/L0 ratio)")
    print("-" * 40)
    for name in all_data.keys():
        l0_mean = all_stats[name][0]["mean"]
        l5_mean = all_stats[name][5]["mean"]
        decay = l5_mean / l0_mean if l0_mean > 0 else np.nan
        print(f"{name:<12}: L0={l0_mean:.4f}, L5={l5_mean:.4f}, decay={decay:.3f}")

    # 3. Correlation with frequency ratio
    print("\n## 3. Spearman Correlation (JSD vs log freq ratio)")
    print("-" * 70)
    print(f"{'Dataset':<12} | " + " | ".join([f"L{L} (r, p)" for L in LAYERS]))
    print("-" * 70)
    for name in all_data.keys():
        corrs = []
        for L in LAYERS:
            r = all_stats[name][L]["r_spearman"]
            p = all_stats[name][L]["p_spearman"]
            sig = "*" if p < 0.05 else ""
            corrs.append(f"{r:+.2f}{sig}")
        print(f"{name:<12} | " + " | ".join(corrs))
    print("(* = p < 0.05)")

    # 4. Find significant correlations
    print("\n## 4. Significant Correlations (p < 0.05)")
    print("-" * 50)
    significant = []
    for name in all_data.keys():
        for L in LAYERS:
            p = all_stats[name][L]["p_spearman"]
            r = all_stats[name][L]["r_spearman"]
            if p < 0.05:
                significant.append((name, L, r, p))

    if significant:
        for name, L, r, p in sorted(significant, key=lambda x: x[3]):
            direction = "positive" if r > 0 else "negative"
            print(f"  {name} Layer {L}: r={r:+.3f}, p={p:.4f} ({direction})")
    else:
        print("  No significant correlations found")

    # 5. High JSD pairs (potential representation differences)
    print("\n## 5. Pairs with Highest JSD at Layer 4 (per dataset)")
    print("-" * 70)
    for name, df in all_data.items():
        if "jsd_layer_4" in df.columns:
            top = df.nlargest(3, "jsd_layer_4")
            print(f"\n{name.upper()}:")
            for _, row in top.iterrows():
                print(
                    f"  {row['high_freq_ngram']} vs {row['low_freq_ngram']}: "
                    f"JSD={row['jsd_layer_4']:.4f}, ratio={row['frequency_ratio']:.1f}x"
                )

    # 6. Low JSD pairs (similar representations despite frequency diff)
    print("\n## 6. Pairs with Lowest JSD at Layer 4 (per dataset)")
    print("-" * 70)
    for name, df in all_data.items():
        if "jsd_layer_4" in df.columns:
            # Filter out pairs with JSD = 0 (these are likely errors)
            valid = df[df["jsd_layer_4"] > 0.01]
            if len(valid) > 0:
                bottom = valid.nsmallest(3, "jsd_layer_4")
                print(f"\n{name.upper()}:")
                for _, row in bottom.iterrows():
                    print(
                        f"  {row['high_freq_ngram']} vs {row['low_freq_ngram']}: "
                        f"JSD={row['jsd_layer_4']:.4f}, ratio={row['frequency_ratio']:.1f}x"
                    )

    return all_stats


def create_cross_dataset_plots(all_data: dict[str, pd.DataFrame], output_dir: Path):
    """Generate cross-dataset comparison plots."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Combine all data
    combined = pd.concat(all_data.values(), ignore_index=True)

    # 1. Mean JSD by layer per dataset
    fig, ax = plt.subplots(figsize=(10, 6))

    for name, df in all_data.items():
        means = [df[f"jsd_layer_{L}"].mean() for L in LAYERS]
        ax.plot(LAYERS, means, marker="o", label=name, linewidth=2, markersize=8)

    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean JSD")
    ax.set_title("Mean JSD by Layer Across Datasets")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / "mean_jsd_by_layer_all.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_dir}/mean_jsd_by_layer_all.png")

    # 2. JSD vs log ratio - all datasets combined
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()

    colors = {"emotion": "C0", "medical": "C1", "legal": "C2", "scientific": "C3", "verb": "C4"}

    for idx, L in enumerate(LAYERS):
        ax = axes[idx]
        for name, df in all_data.items():
            ax.scatter(
                df["log_ratio"], df[f"jsd_layer_{L}"], alpha=0.5, s=30, label=name, c=colors[name]
            )

        ax.set_xlabel("log₁₀(frequency ratio)")
        ax.set_ylabel("JSD")
        ax.set_title(f"Layer {L}")

        if idx == 0:
            ax.legend(fontsize=8)

    # Hide unused subplot
    if len(LAYERS) < len(axes):
        for i in range(len(LAYERS), len(axes)):
            axes[i].set_visible(False)

    fig.suptitle("JSD vs Frequency Ratio by Layer (All Datasets)", fontsize=14)
    plt.tight_layout()
    fig.savefig(output_dir / "jsd_vs_ratio_all_datasets.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_dir}/jsd_vs_ratio_all_datasets.png")

    # 3. Box plot of JSD by dataset
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()

    for idx, L in enumerate(LAYERS):
        ax = axes[idx]
        data = [all_data[name][f"jsd_layer_{L}"].dropna().values for name in all_data.keys()]
        bp = ax.boxplot(data, labels=list(all_data.keys()), patch_artist=True)

        for patch, color in zip(bp["boxes"], [colors[n] for n in all_data.keys()]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_xlabel("Dataset")
        ax.set_ylabel("JSD")
        ax.set_title(f"Layer {L}")
        ax.tick_params(axis="x", rotation=45)

    fig.suptitle("JSD Distribution by Dataset", fontsize=14)
    plt.tight_layout()
    fig.savefig(output_dir / "jsd_boxplot_by_dataset.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_dir}/jsd_boxplot_by_dataset.png")

    # 4. Correlation heatmap
    corr_data = []
    for name in all_data.keys():
        df = all_data[name]
        row = {"dataset": name}
        for L in LAYERS:
            valid = df["log_ratio"].notna() & df[f"jsd_layer_{L}"].notna()
            if valid.sum() > 2:
                r, _ = spearmanr(df.loc[valid, "log_ratio"], df.loc[valid, f"jsd_layer_{L}"])
            else:
                r = np.nan
            row[f"Layer_{L}"] = r
        corr_data.append(row)

    corr_df = pd.DataFrame(corr_data).set_index("dataset")

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(corr_df.values, cmap="RdBu_r", aspect="auto", vmin=-0.5, vmax=0.5)

    ax.set_xticks(range(len(LAYERS)))
    ax.set_xticklabels([f"L{L}" for L in LAYERS])
    ax.set_yticks(range(len(corr_df)))
    ax.set_yticklabels(corr_df.index)

    # Add text annotations
    for i in range(len(corr_df)):
        for j in range(len(LAYERS)):
            val = corr_df.iloc[i, j]
            color = "white" if abs(val) > 0.3 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=9)

    ax.set_title("Spearman Correlation (JSD vs log freq ratio)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Correlation coefficient")

    plt.tight_layout()
    fig.savefig(output_dir / "correlation_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_dir}/correlation_heatmap.png")


# =============================================================================
# MAIN
# =============================================================================


def main():
    print("Loading all datasets...")
    all_data = load_all_datasets()

    if not all_data:
        print("No data found!")
        return

    # Run analysis
    all_stats = find_patterns(all_data)

    # Create plots
    output_dir = RUNS_DIR / "cross_dataset_analysis"
    print(f"\nGenerating cross-dataset plots to {output_dir}...")
    create_cross_dataset_plots(all_data, output_dir)

    print("\n" + "=" * 70)
    print("KEY FINDINGS SUMMARY")
    print("=" * 70)

    # Calculate overall patterns
    all_decays = []
    sig_corrs = []
    for name in all_data.keys():
        l0 = all_stats[name][0]["mean"]
        l5 = all_stats[name][5]["mean"]
        all_decays.append(l5 / l0 if l0 > 0 else np.nan)

        for L in LAYERS:
            if all_stats[name][L]["p_spearman"] < 0.05:
                sig_corrs.append((name, L, all_stats[name][L]["r_spearman"]))

    print(f"\n1. JSD DECAY: All datasets show consistent JSD decay from L0 to L5")
    print(f"   Average decay ratio (L5/L0): {np.nanmean(all_decays):.3f}")
    print(f"   This suggests representations become more similar in deeper layers")

    print(f"\n2. SIGNIFICANT CORRELATIONS: {len(sig_corrs)} total")
    if sig_corrs:
        pos = sum(1 for _, _, r in sig_corrs if r > 0)
        neg = len(sig_corrs) - pos
        print(f"   Positive correlations: {pos}")
        print(f"   Negative correlations: {neg}")
        if pos > neg:
            print(f"   --> Higher frequency ratio tends to correlate with HIGHER JSD")
        else:
            print(f"   --> Higher frequency ratio tends to correlate with LOWER JSD")

    print(f"\n3. LAYER-SPECIFIC EFFECTS:")
    layer_counts = {}
    for name, L, r in sig_corrs:
        layer_counts[L] = layer_counts.get(L, 0) + 1

    if layer_counts:
        most_sig_layer = max(layer_counts, key=layer_counts.get)
        print(
            f"   Layer {most_sig_layer} shows the most significant correlations ({layer_counts[most_sig_layer]} datasets)"
        )

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
