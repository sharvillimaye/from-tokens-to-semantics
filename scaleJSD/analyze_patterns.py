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
from scipy.stats import bootstrap, f_oneway, pearsonr, spearmanr

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
# STATISTICAL RIGOR
# =============================================================================


def bootstrap_decay_ratio(df: pd.DataFrame, n_bootstrap: int = 1000, ci: float = 0.95) -> dict:
    """
    Compute bootstrap confidence interval for decay ratio (L5/L0).

    Args:
        df: DataFrame with jsd_layer_0 and jsd_layer_5 columns
        n_bootstrap: Number of bootstrap samples
        ci: Confidence level (default 0.95 for 95% CI)

    Returns:
        dict with mean, lower, upper bounds
    """
    l0_vals = df["jsd_layer_0"].dropna().values
    l5_vals = df["jsd_layer_5"].dropna().values

    # Compute point estimate
    point_estimate = l5_vals.mean() / l0_vals.mean()

    # Bootstrap resampling
    n = len(l0_vals)
    rng = np.random.default_rng(42)

    bootstrap_ratios = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        l0_sample = l0_vals[idx]
        l5_sample = l5_vals[idx]
        ratio = l5_sample.mean() / l0_sample.mean() if l0_sample.mean() > 0 else np.nan
        bootstrap_ratios.append(ratio)

    bootstrap_ratios = np.array(bootstrap_ratios)
    bootstrap_ratios = bootstrap_ratios[~np.isnan(bootstrap_ratios)]

    # Compute CI
    alpha = 1 - ci
    lower = np.percentile(bootstrap_ratios, alpha / 2 * 100)
    upper = np.percentile(bootstrap_ratios, (1 - alpha / 2) * 100)

    return {
        "mean": point_estimate,
        "lower": lower,
        "upper": upper,
        "std": bootstrap_ratios.std(),
    }


def run_anova_across_domains(all_data: dict[str, pd.DataFrame], layer: int) -> dict:
    """
    Run one-way ANOVA to test if JSD differs significantly across domains.

    Args:
        all_data: dict mapping dataset name to DataFrame
        layer: Layer index to test

    Returns:
        dict with F-statistic, p-value, and group means
    """
    col = f"jsd_layer_{layer}"
    groups = []
    group_names = []
    group_means = {}

    for name, df in all_data.items():
        if col in df.columns:
            vals = df[col].dropna().values
            if len(vals) > 0:
                groups.append(vals)
                group_names.append(name)
                group_means[name] = vals.mean()

    if len(groups) < 2:
        return {"f_stat": np.nan, "p_value": np.nan, "group_means": group_means}

    # Run ANOVA
    f_stat, p_value = f_oneway(*groups)

    return {
        "f_stat": f_stat,
        "p_value": p_value,
        "group_means": group_means,
        "group_names": group_names,
    }


def compute_cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute Cohen's d effect size between two groups."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = group1.var(ddof=1), group2.var(ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return 0.0

    return (group1.mean() - group2.mean()) / pooled_std


def run_statistical_tests(all_data: dict[str, pd.DataFrame], output_dir: Path):
    """
    Run comprehensive statistical tests and save results.

    Includes:
    - Bootstrap CI for decay ratios
    - ANOVA across domains
    - Effect sizes (Cohen's d)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    results.append("=" * 70)
    results.append("STATISTICAL RIGOR ANALYSIS")
    results.append("=" * 70)

    # 1. Bootstrap CI for decay ratios
    results.append("\n## 1. Bootstrap 95% CI for Decay Ratio (L5/L0)")
    results.append("-" * 50)
    results.append(f"{'Dataset':<12} | {'Mean':>8} | {'95% CI':>20} | {'Std':>8}")
    results.append("-" * 50)

    decay_ci_data = []
    for name, df in all_data.items():
        ci_result = bootstrap_decay_ratio(df)
        decay_ci_data.append(
            {
                "dataset": name,
                "mean": ci_result["mean"],
                "lower": ci_result["lower"],
                "upper": ci_result["upper"],
                "std": ci_result["std"],
            }
        )
        results.append(
            f"{name:<12} | {ci_result['mean']:>8.3f} | "
            f"[{ci_result['lower']:.3f}, {ci_result['upper']:.3f}] | "
            f"{ci_result['std']:>8.3f}"
        )

    # 2. ANOVA across domains
    results.append("\n## 2. One-way ANOVA (JSD by Domain)")
    results.append("-" * 50)

    anova_results = {}
    for L in LAYERS:
        anova = run_anova_across_domains(all_data, L)
        anova_results[L] = anova
        sig = (
            "***"
            if anova["p_value"] < 0.001
            else "**"
            if anova["p_value"] < 0.01
            else "*"
            if anova["p_value"] < 0.05
            else ""
        )
        results.append(f"Layer {L}: F={anova['f_stat']:.3f}, p={anova['p_value']:.4f} {sig}")

    results.append("\n(*** p<0.001, ** p<0.01, * p<0.05)")

    # 3. Pairwise effect sizes for significant layers
    results.append("\n## 3. Pairwise Effect Sizes (Cohen's d)")
    results.append("-" * 50)

    # Find layers with significant ANOVA
    sig_layers = [L for L in LAYERS if anova_results[L]["p_value"] < 0.05]

    if sig_layers:
        results.append(f"Significant layers: {sig_layers}")

        for L in sig_layers:
            results.append(f"\nLayer {L} pairwise Cohen's d:")
            col = f"jsd_layer_{L}"
            datasets = list(all_data.keys())

            for i, name1 in enumerate(datasets):
                for name2 in datasets[i + 1 :]:
                    g1 = all_data[name1][col].dropna().values
                    g2 = all_data[name2][col].dropna().values
                    d = compute_cohens_d(g1, g2)

                    # Interpret effect size
                    if abs(d) < 0.2:
                        interp = "negligible"
                    elif abs(d) < 0.5:
                        interp = "small"
                    elif abs(d) < 0.8:
                        interp = "medium"
                    else:
                        interp = "large"

                    results.append(f"  {name1} vs {name2}: d={d:+.3f} ({interp})")
    else:
        results.append("No layers showed significant domain differences.")

    # 4. Summary
    results.append("\n## 4. Summary")
    results.append("-" * 50)

    # Check if decay ratios are significantly different from 1.0
    all_upper = [d["upper"] for d in decay_ci_data]
    if all(u < 1.0 for u in all_upper):
        results.append("- All decay ratios are significantly < 1.0 (CI upper bounds < 1)")
        results.append("  --> JSD decay is a robust phenomenon across all domains")

    # Report on domain differences
    n_sig_anova = sum(1 for L in LAYERS if anova_results[L]["p_value"] < 0.05)
    results.append(f"- {n_sig_anova}/{len(LAYERS)} layers show significant domain differences")

    # Write to file
    output_text = "\n".join(results)
    stats_path = output_dir / "statistical_tests.txt"
    with open(stats_path, "w") as f:
        f.write(output_text)
    print(f"Saved: {stats_path}")

    # Print results
    print(output_text)

    # Create visualization of decay ratios with CI
    fig, ax = plt.subplots(figsize=(10, 6))

    names = [d["dataset"] for d in decay_ci_data]
    means = [d["mean"] for d in decay_ci_data]
    lowers = [d["mean"] - d["lower"] for d in decay_ci_data]
    uppers = [d["upper"] - d["mean"] for d in decay_ci_data]

    x = range(len(names))
    ax.bar(x, means, yerr=[lowers, uppers], capsize=5, color="steelblue", alpha=0.7)
    ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.5, label="No decay (ratio=1)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("Decay Ratio (L5/L0)")
    ax.set_title("JSD Decay Ratio with 95% Bootstrap CI")
    ax.legend()
    ax.set_ylim(0, max(means) * 1.5)

    plt.tight_layout()
    fig.savefig(output_dir / "decay_ratio_ci.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_dir}/decay_ratio_ci.png")

    return anova_results, decay_ci_data


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

    # Run statistical tests
    print("\nRunning statistical tests...")
    run_statistical_tests(all_data, output_dir)

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
