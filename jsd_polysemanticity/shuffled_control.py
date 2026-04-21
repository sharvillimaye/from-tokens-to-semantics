# -*- coding: utf-8 -*-
"""
Shuffled Control Experiment for JSD Polysemanticity Analysis

This script runs the SAME JSD analysis as the main pipeline, but with
randomly shuffled phrase-to-bucket assignments. If the JSD effect is
driven by genuine frequency differences, the effect should disappear
(or drastically shrink) under random assignment.

Design:
  1. Pool all phrases from bucket 0 and bucket 7
  2. Randomly assign them to two equal-sized groups (N_SHUFFLES times)
  3. Run the same JSD sweep for each shuffle
  4. Compare shuffled JSD distribution against the real JSD values

Usage:
  1. First run the notebook cells that create df_candidates and cumulative.parquet
  2. Then: python shuffled_control.py
  OR run inside a notebook after df_candidates is defined.
"""

from __future__ import annotations
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import mannwhitneyu

# Import the real pipeline's functions
from pipeline_script import (
    MODELS,
    BUCKETS_TO_COMPARE,
    BATCH_SIZE,
    DTYPE,
    DEVICE_MAP,
    TEMPLATE_PREFIX,
    TEMPLATE_SUFFIX,
    COUNTS_PARQUET,
    EPS,
    setup_plotting,
    build_text,
    get_anchor_indices,
    capture_activations,
    relu_probs,
    jsd_per_dim,
    savefig_dual,
)
from nnsight import LanguageModel

# =============================================================================
# CONFIGURATION
# =============================================================================

N_SHUFFLES = 10          # Number of random reshuffles (10 is usually enough)
SEED = 42                # Base seed for reproducibility
OUTROOT = Path("./runs_jsd_poly_shuffled_control")
OUTROOT.mkdir(parents=True, exist_ok=True)


# =============================================================================
# SHUFFLED JSD COMPUTATION
# =============================================================================

def compute_jsd_for_assignment(
    lm: LanguageModel,
    group_a_phrases: List[str],
    group_b_phrases: List[str],
    layers: List[int],
    df_counts_all: pd.DataFrame,
    count_col: str,
    step: int,
    batch_size: int = 32,
) -> Dict[int, float]:
    """Compute JSD between two phrase groups for a single model checkpoint.

    Returns dict mapping layer -> JSD value.
    """
    results = {}

    for layer in layers:
        # Process group A
        avgP_a = _compute_avg_probs(lm, group_a_phrases, layer, batch_size)
        # Process group B
        avgP_b = _compute_avg_probs(lm, group_b_phrases, layer, batch_size)

        if avgP_a.size and avgP_b.size and avgP_a.shape == avgP_b.shape:
            contrib = jsd_per_dim(avgP_a, avgP_b, eps=EPS)
            results[layer] = float(contrib.sum())
        else:
            results[layer] = np.nan

    return results


def _compute_avg_probs(
    lm: LanguageModel,
    phrases: List[str],
    layer: int,
    batch_size: int,
) -> np.ndarray:
    """Compute average ReLU-probability distribution for a set of phrases at one layer."""
    texts = [build_text(p) for p in phrases]
    anchors = get_anchor_indices(lm, phrases)
    rows = []

    for i0 in range(0, len(phrases), batch_size):
        sl = slice(i0, i0 + batch_size)
        caps = capture_activations(lm, texts[sl], anchors[sl], [layer])
        A = caps[layer]
        P = relu_probs(A, eps=EPS)
        rows.append(P)

    if not rows:
        return np.array([])

    P_all = np.vstack(rows)
    meanP = P_all.mean(axis=0)
    meanP = meanP / (meanP.sum() + EPS)
    return meanP


# =============================================================================
# MAIN SHUFFLED CONTROL PIPELINE
# =============================================================================

def run_shuffled_control(
    bucket2phrases: Dict[int, List[str]],
    df_counts_all: pd.DataFrame,
    count_col: str,
):
    """Run the full shuffled control experiment."""
    setup_plotting()

    b0, b1 = BUCKETS_TO_COMPARE[:2]
    all_phrases = bucket2phrases[b0] + bucket2phrases[b1]
    n_total = len(all_phrases)
    n_a = len(bucket2phrases[b0])  # Keep group sizes equal to original

    print(f"[shuffled control] {n_total} phrases total, "
          f"group A size = {n_a}, group B size = {n_total - n_a}")
    print(f"[shuffled control] Running {N_SHUFFLES} random shuffles")

    rng = np.random.default_rng(SEED)

    for model_config in MODELS:
        tag = model_config["tag"]
        model_title = model_config["title"]
        model_id = model_config["model_id"]
        steps = model_config["steps"]
        layers = model_config["layers"]

        print(f"\n{'='*60}")
        print(f"Processing {model_title} (shuffled control)")
        print(f"{'='*60}")

        outdir = OUTROOT / tag
        outdir.mkdir(parents=True, exist_ok=True)

        # Collect: real JSD + shuffled JSD per (step, layer)
        real_jsd_rows = []
        shuffled_jsd_rows = []

        for step in steps:
            rev = f"step{step}"
            print(f"\n[info] loading {model_title} @ {rev}")

            lm = LanguageModel(
                model_id, revision=rev,
                device_map=DEVICE_MAP, torch_dtype=DTYPE, dispatch=True
            )

            # --- Real JSD (original bucket assignment) ---
            real_jsd = compute_jsd_for_assignment(
                lm, bucket2phrases[b0], bucket2phrases[b1],
                layers, df_counts_all, count_col, step, BATCH_SIZE,
            )
            for layer, jsd_val in real_jsd.items():
                real_jsd_rows.append({
                    "step": step, "layer": layer,
                    "JSD": jsd_val, "condition": "real",
                })

            # --- Shuffled JSD ---
            for shuffle_i in range(N_SHUFFLES):
                perm = rng.permutation(n_total)
                group_a = [all_phrases[i] for i in perm[:n_a]]
                group_b = [all_phrases[i] for i in perm[n_a:]]

                shuf_jsd = compute_jsd_for_assignment(
                    lm, group_a, group_b,
                    layers, df_counts_all, count_col, step, BATCH_SIZE,
                )
                for layer, jsd_val in shuf_jsd.items():
                    shuffled_jsd_rows.append({
                        "step": step, "layer": layer,
                        "JSD": jsd_val, "condition": "shuffled",
                        "shuffle_i": shuffle_i,
                    })

                print(f"  shuffle {shuffle_i+1}/{N_SHUFFLES} done")

            del lm
            torch.cuda.empty_cache()

        # --- Save results ---
        df_real = pd.DataFrame(real_jsd_rows)
        df_shuffled = pd.DataFrame(shuffled_jsd_rows)
        df_all = pd.concat([df_real, df_shuffled], ignore_index=True)
        df_all.to_csv(outdir / f"{tag}_shuffled_control_jsd.csv", index=False)

        # --- Generate comparison plots ---
        _plot_real_vs_shuffled(df_real, df_shuffled, layers, steps, outdir, model_title, tag)
        _compute_statistics(df_real, df_shuffled, layers, steps, outdir, tag)

        print(f"\nCompleted {model_title}. Results in {outdir}")


# =============================================================================
# VISUALIZATION
# =============================================================================

def _plot_real_vs_shuffled(
    df_real: pd.DataFrame,
    df_shuffled: pd.DataFrame,
    layers: List[int],
    steps: List[int],
    outdir: Path,
    model_title: str,
    tag: str,
):
    """Plot real JSD vs shuffled JSD distribution over training steps."""

    # --- Plot 1: JSD over steps, real vs shuffled mean+band per layer ---
    for layer in layers:
        fig, ax = plt.subplots(figsize=(10, 5.5))

        # Real
        dr = df_real[df_real["layer"] == layer].sort_values("step")
        ax.plot(dr["step"], dr["JSD"], "o-", color="C0", lw=2, label="Real (bucket 0 vs 7)")

        # Shuffled: mean + 95% CI band
        ds = df_shuffled[df_shuffled["layer"] == layer]
        shuf_agg = ds.groupby("step")["JSD"].agg(["mean", "std", "count"]).reindex(steps)
        shuf_mean = shuf_agg["mean"].values
        shuf_std = shuf_agg["std"].values

        ax.plot(steps, shuf_mean, "s--", color="C3", lw=1.5, label="Shuffled (mean)")
        ax.fill_between(
            steps,
            shuf_mean - 2 * shuf_std,
            shuf_mean + 2 * shuf_std,
            color="C3", alpha=0.15, label="Shuffled (±2σ)"
        )

        ax.set_xlabel("Training step")
        ax.set_ylabel("JSD")
        ax.set_title(f"Real vs Shuffled Control — {model_title}, Layer {layer}")
        ax.legend()

        savefig_dual(fig, outdir / f"{tag}_real_vs_shuffled_layer{layer}.png")
        plt.close(fig)

    # --- Plot 2: Summary across all layers at final step ---
    final_step = max(steps)
    fig, ax = plt.subplots(figsize=(10, 5.5))

    dr_final = df_real[df_real["step"] == final_step].sort_values("layer")
    ds_final = df_shuffled[df_shuffled["step"] == final_step]
    shuf_final = ds_final.groupby("layer")["JSD"].agg(["mean", "std"]).reindex(layers)

    ax.bar(
        np.array(layers) - 0.15, dr_final["JSD"].values, width=0.3,
        color="C0", alpha=0.8, label="Real"
    )
    ax.bar(
        np.array(layers) + 0.15, shuf_final["mean"].values, width=0.3,
        color="C3", alpha=0.8, label="Shuffled (mean)",
        yerr=2 * shuf_final["std"].values, capsize=3
    )

    ax.set_xlabel("Layer")
    ax.set_ylabel("JSD")
    ax.set_title(f"Real vs Shuffled at final step ({final_step}) — {model_title}")
    ax.legend()
    ax.set_xticks(layers)

    savefig_dual(fig, outdir / f"{tag}_real_vs_shuffled_final_step_bars.png")
    plt.close(fig)

    # --- Plot 3: Effect size (real / shuffled mean) over steps ---
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for layer in layers:
        dr = df_real[df_real["layer"] == layer].sort_values("step")
        ds = df_shuffled[df_shuffled["layer"] == layer]
        shuf_agg = ds.groupby("step")["JSD"].mean().reindex(steps)

        ratio = dr["JSD"].values / np.maximum(shuf_agg.values, 1e-12)
        ax.plot(steps, ratio, "o-", lw=1.5, label=f"Layer {layer}")

    ax.axhline(1.0, ls="--", color="gray", alpha=0.6, label="No effect (ratio=1)")
    ax.set_xlabel("Training step")
    ax.set_ylabel("JSD_real / JSD_shuffled")
    ax.set_title(f"Effect Size Ratio — {model_title}")
    ax.legend(ncols=min(4, len(layers)))

    savefig_dual(fig, outdir / f"{tag}_effect_size_ratio.png")
    plt.close(fig)


# =============================================================================
# STATISTICAL SUMMARY
# =============================================================================

def _compute_statistics(
    df_real: pd.DataFrame,
    df_shuffled: pd.DataFrame,
    layers: List[int],
    steps: List[int],
    outdir: Path,
    tag: str,
):
    """Compute and save statistical comparison between real and shuffled JSD."""
    stat_rows = []

    for layer in layers:
        for step in steps:
            real_val = df_real[
                (df_real["layer"] == layer) & (df_real["step"] == step)
            ]["JSD"].values

            shuf_vals = df_shuffled[
                (df_shuffled["layer"] == layer) & (df_shuffled["step"] == step)
            ]["JSD"].values

            if len(real_val) == 0 or len(shuf_vals) == 0:
                continue

            real_v = real_val[0]
            shuf_mean = np.mean(shuf_vals)
            shuf_std = np.std(shuf_vals)

            # How many SDs above shuffled mean is the real value?
            z_score = (real_v - shuf_mean) / max(shuf_std, 1e-12)

            # What fraction of shuffles exceeded the real value?
            p_empirical = np.mean(shuf_vals >= real_v)

            stat_rows.append({
                "layer": layer,
                "step": step,
                "real_jsd": real_v,
                "shuffled_mean": shuf_mean,
                "shuffled_std": shuf_std,
                "z_score": z_score,
                "p_empirical": p_empirical,
                "ratio": real_v / max(shuf_mean, 1e-12),
            })

    df_stats = pd.DataFrame(stat_rows)
    df_stats.to_csv(outdir / f"{tag}_shuffled_control_stats.csv", index=False)

    # Print summary
    print(f"\n{'='*60}")
    print(f"STATISTICAL SUMMARY: {tag}")
    print(f"{'='*60}")
    print(f"Mean z-score (real vs shuffled): {df_stats['z_score'].mean():.2f}")
    print(f"Mean ratio (real/shuffled):      {df_stats['ratio'].mean():.2f}")
    print(f"Fraction with p_emp < 0.05:      {(df_stats['p_empirical'] < 0.05).mean():.1%}")
    print(f"Fraction with p_emp == 0:        {(df_stats['p_empirical'] == 0).mean():.1%}")
    print()
    print(df_stats.groupby("layer")[["z_score", "ratio", "p_empirical"]].mean().to_string())


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # When run as a script, load data the same way the main pipeline does
    print("Loading data...")

    # You need df_candidates in scope — either:
    #   (a) pickle/parquet it from the notebook and load here, or
    #   (b) run this inside the notebook after the sampling cells

    # Try loading from a saved file first
    candidates_path = Path("./df_candidates.parquet")
    if candidates_path.exists():
        df_candidates = pd.read_parquet(candidates_path)
        print(f"Loaded {len(df_candidates)} candidates from {candidates_path}")
    else:
        raise RuntimeError(
            "df_candidates not found. Either:\n"
            "  1. Save it from the notebook: df_candidates.to_parquet('df_candidates.parquet')\n"
            "  2. Run this script inside the notebook after the sampling cells.\n"
        )

    # Build bucket mapping
    present_buckets = sorted(df_candidates["bucket"].dropna().unique().astype(int).tolist())
    use_buckets = [b for b in BUCKETS_TO_COMPARE if b in present_buckets]

    bucket2phrases = {
        b: sorted(df_candidates.loc[df_candidates["bucket"] == b, "phrase"].unique().tolist())
        for b in use_buckets
    }

    # Load frequency data
    df_counts_all = pd.read_parquet(COUNTS_PARQUET)
    count_col = ("per_million_cum" if "per_million_cum" in df_counts_all.columns
                 else "count_cum")

    all_phrases = [p for b in use_buckets for p in bucket2phrases[b]]
    df_counts_all = df_counts_all[df_counts_all["phrase"].isin(all_phrases)].copy()

    # Run the experiment
    run_shuffled_control(bucket2phrases, df_counts_all, count_col)
