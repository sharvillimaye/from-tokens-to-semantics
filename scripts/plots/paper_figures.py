#!/usr/bin/env python3
"""
Publication-Quality Figures for Polytope Superposition Paper

Generates a complete figure set from:
  1. Polytope analysis results (JSON) — density, participation ratio, pattern reuse
  2. JSD pipeline results (CSV) — JSD between frequency buckets
  3. Shuffled control results (CSV) — null distribution validation

Designed for NeurIPS/ICML submission. All figures are PDF-ready for Overleaf.

Usage:
    python -m scripts.plots.paper_figures \
        --polytope-dir ./polytope_results \
        --jsd-dir ./jsd_polysemanticity/runs_jsd_poly \
        --shuffled-dir ./jsd_polysemanticity/runs_jsd_poly_shuffled_control \
        --output-dir ./neurips-2026/figures/main
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

# =============================================================================
# PUBLICATION STYLE
# =============================================================================

# Color-blind safe palette (Okabe-Ito inspired)
COLORS = {
    "high_freq": "#0072B2",   # blue
    "low_freq": "#D55E00",    # vermilion
    "real": "#0072B2",
    "shuffled": "#999999",
    "effect": "#009E73",      # green
    "neutral": "#555555",
}

LAYER_CMAP = "viridis"
DIFF_CMAP = "coolwarm"

def set_paper_style():
    """Set matplotlib rcParams for publication figures."""
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
    """Save figure as both PDF and PNG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".pdf"), format="pdf", bbox_inches="tight", pad_inches=0.05)
    fig.savefig(path.with_suffix(".png"), format="png", dpi=300, bbox_inches="tight")
    if close:
        plt.close(fig)
    print(f"  saved: {path.with_suffix('.pdf')}")


# =============================================================================
# DATA LOADING
# =============================================================================

def load_polytope_results(json_path: Path) -> pd.DataFrame:
    """Load polytope analysis JSON into a tidy DataFrame."""
    with open(json_path) as f:
        results = json.load(f)

    rows = []
    for result in results:
        if "error" in result:
            continue
        checkpoint = result.get("checkpoint_step", "unknown")
        try:
            checkpoint = int(checkpoint)
        except (ValueError, TypeError):
            continue

        analysis_type = result.get("analysis_type", "legacy")

        if analysis_type == "layer_wise":
            for layer_idx, layer_data in result.get("layer_results", {}).items():
                if "error" in layer_data:
                    continue
                for freq_type in ["high_freq", "low_freq"]:
                    density = layer_data.get(f"{freq_type}_density", {})
                    rows.append({
                        "checkpoint": checkpoint,
                        "layer": int(layer_idx),
                        "group": freq_type,
                        "density_mean": density.get("density_mean", np.nan),
                        "density_std": density.get("density_std", np.nan),
                        "boundary_crossing_rate": density.get("boundary_crossing_rate", np.nan),
                        "participation_ratio": layer_data.get(f"{freq_type}_participation_ratio", np.nan),
                        "sparsity": layer_data.get(f"{freq_type}_sparsity", np.nan),
                        "activation_norm": layer_data.get(f"{freq_type}_activation_norm", np.nan),
                        "pattern_reuse_rate": layer_data.get(f"{freq_type}_pattern_reuse_rate", np.nan),
                    })
        else:
            for freq_type in ["high_freq", "low_freq"]:
                density = result.get(f"{freq_type}_density", {})
                rows.append({
                    "checkpoint": checkpoint,
                    "layer": -1,  # averaged
                    "group": freq_type,
                    "density_mean": density.get("density_mean", np.nan),
                    "density_std": density.get("density_std", np.nan),
                    "participation_ratio": result.get(f"{freq_type}_participation_ratio", np.nan),
                    "sparsity": result.get(f"{freq_type}_sparsity", np.nan),
                    "activation_norm": result.get(f"{freq_type}_activation_norm", np.nan),
                    "pattern_reuse_rate": 0,
                })

    return pd.DataFrame(rows)


def load_jsd_results(csv_path: Path) -> pd.DataFrame:
    """Load JSD between-bin CSV."""
    return pd.read_csv(csv_path)


def load_shuffled_results(csv_path: Path) -> pd.DataFrame:
    """Load shuffled control CSV."""
    return pd.read_csv(csv_path)


def find_data_files(polytope_dir: Path, jsd_dir: Path, shuffled_dir: Path) -> Dict:
    """Auto-discover result files organized by model tag."""
    models = {}

    # Scan JSD dir for model tags
    if jsd_dir.exists():
        for subdir in sorted(jsd_dir.iterdir()):
            if subdir.is_dir():
                tag = subdir.name
                jsd_csv = list(subdir.rglob("*between_bin_jsd.csv"))
                contrib_csv = list(subdir.rglob("*per_neuron_jsd_contrib.csv"))
                models.setdefault(tag, {})
                if jsd_csv:
                    models[tag]["jsd"] = jsd_csv[0]
                if contrib_csv:
                    models[tag]["contrib"] = contrib_csv[0]

    # Scan shuffled dir
    if shuffled_dir.exists():
        for subdir in sorted(shuffled_dir.iterdir()):
            if subdir.is_dir():
                tag = subdir.name
                shuf_csv = list(subdir.rglob("*shuffled_control_jsd.csv"))
                stat_csv = list(subdir.rglob("*shuffled_control_stats.csv"))
                models.setdefault(tag, {})
                if shuf_csv:
                    models[tag]["shuffled"] = shuf_csv[0]
                if stat_csv:
                    models[tag]["shuffled_stats"] = stat_csv[0]

    # Scan polytope dir for JSON results
    if polytope_dir.exists():
        for json_file in sorted(polytope_dir.rglob("polytope_analysis_results*.json")):
            # Try to match to a model tag from the path
            for tag in list(models.keys()):
                if tag in str(json_file):
                    models[tag]["polytope"] = json_file
                    break
            else:
                # If no match, use filename as tag
                models.setdefault("polytope", {})["polytope"] = json_file

    return models


# =============================================================================
# FIGURE 1: POLYTOPE TEMPORAL EVOLUTION (per Pythia scale)
# =============================================================================

def fig_polytope_temporal(
    df: pd.DataFrame,
    model_title: str,
    outdir: Path,
    tag: str,
):
    """
    Small-multiples: per-layer polytope density + participation ratio over checkpoints.
    Two rows of subplots — top row = density, bottom row = participation ratio.
    Each column is a layer.
    """
    layer_data = df[df["layer"] >= 0]
    if layer_data.empty:
        return

    layers = sorted(layer_data["layer"].unique())
    checkpoints = sorted(layer_data["checkpoint"].unique())

    # Subsample layers if too many
    max_cols = 8
    if len(layers) > max_cols:
        idxs = np.linspace(0, len(layers) - 1, max_cols).astype(int)
        layers = [layers[i] for i in idxs]

    n = len(layers)
    fig, axes = plt.subplots(2, n, figsize=(2.2 * n, 4.5), sharey="row")
    if n == 1:
        axes = axes.reshape(2, 1)

    metrics = [("density_mean", "Polytope Density"), ("participation_ratio", "Participation Ratio")]

    for row, (metric, ylabel) in enumerate(metrics):
        for col, layer in enumerate(layers):
            ax = axes[row, col]
            for group in ["high_freq", "low_freq"]:
                sub = layer_data[(layer_data["layer"] == layer) & (layer_data["group"] == group)]
                sub = sub.sort_values("checkpoint")
                if sub.empty:
                    continue
                ax.plot(sub["checkpoint"], sub[metric],
                        color=COLORS[group], marker="o" if group == "high_freq" else "s",
                        markersize=2.5, linewidth=1.2)

            ax.grid(True, alpha=0.3)
            if row == 0:
                ax.set_title(f"L{layer}", fontsize=9)
            if col == 0:
                ax.set_ylabel(ylabel, fontsize=9)
            if row == 1:
                # Only show a few x-ticks
                ax.set_xlabel("Step", fontsize=8)
                if len(checkpoints) > 5:
                    ticks = checkpoints[::max(1, len(checkpoints) // 3)]
                    ax.set_xticks(ticks)
                    ax.set_xticklabels([f"{t // 1000}k" for t in ticks], fontsize=7, rotation=45)
            else:
                ax.set_xticklabels([])

    # Legend
    handles = [
        Line2D([0], [0], color=COLORS["high_freq"], marker="o", markersize=4, label="High Freq"),
        Line2D([0], [0], color=COLORS["low_freq"], marker="s", markersize=4, label="Low Freq"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.02),
               ncol=2, frameon=False, fontsize=9)
    fig.suptitle(f"{model_title} — Polytope Metrics Over Training", y=1.06, fontsize=11)
    plt.tight_layout()
    save(fig, outdir / f"{tag}_polytope_temporal")


# =============================================================================
# FIGURE 2: LAYER × CHECKPOINT HEATMAPS (density + PR + diff)
# =============================================================================

def fig_heatmaps(
    df: pd.DataFrame,
    model_title: str,
    outdir: Path,
    tag: str,
):
    """
    2×3 grid of heatmaps:
      Row 1: density (high, low, diff)
      Row 2: participation_ratio (high, low, diff)
    """
    layer_data = df[df["layer"] >= 0]
    if layer_data.empty:
        return

    layers = sorted(layer_data["layer"].unique())
    checkpoints = sorted(layer_data["checkpoint"].unique())

    def make_grid(metric, group):
        sub = layer_data[layer_data["group"] == group]
        pivot = sub.pivot_table(index="layer", columns="checkpoint", values=metric, aggfunc="mean")
        pivot = pivot.reindex(index=layers, columns=checkpoints)
        return pivot.values

    metrics = [("density_mean", "Polytope Density"), ("participation_ratio", "Participation Ratio")]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))

    for row, (metric, title) in enumerate(metrics):
        gh = make_grid(metric, "high_freq")
        gl = make_grid(metric, "low_freq")
        diff = gh - gl

        # High freq
        im0 = axes[row, 0].imshow(gh, aspect="auto", origin="lower", cmap=LAYER_CMAP)
        axes[row, 0].set_title(f"{title} — High Freq", fontsize=9)
        plt.colorbar(im0, ax=axes[row, 0], fraction=0.046)

        # Low freq
        im1 = axes[row, 1].imshow(gl, aspect="auto", origin="lower", cmap=LAYER_CMAP)
        axes[row, 1].set_title(f"{title} — Low Freq", fontsize=9)
        plt.colorbar(im1, ax=axes[row, 1], fraction=0.046)

        # Difference
        vmax = np.nanmax(np.abs(diff)) if np.isfinite(diff).any() else 1.0
        vmax = max(vmax, 1e-6)
        im2 = axes[row, 2].imshow(diff, aspect="auto", origin="lower",
                                    cmap=DIFF_CMAP, vmin=-vmax, vmax=vmax)
        axes[row, 2].set_title(f"Δ (High − Low)", fontsize=9)
        plt.colorbar(im2, ax=axes[row, 2], fraction=0.046)

        # Axis labels
        for col in range(3):
            ax = axes[row, col]
            ax.set_ylabel("Layer")
            ax.set_xlabel("Checkpoint")

            # Subsample ticks
            if len(checkpoints) > 8:
                step = max(1, len(checkpoints) // 6)
                xt = list(range(0, len(checkpoints), step))
            else:
                xt = list(range(len(checkpoints)))
            ax.set_xticks(xt)
            ax.set_xticklabels([f"{checkpoints[i] // 1000}k" for i in xt], rotation=45, fontsize=7)

            if len(layers) > 15:
                step = max(1, len(layers) // 10)
                yt = list(range(0, len(layers), step))
            else:
                yt = list(range(len(layers)))
            ax.set_yticks(yt)
            ax.set_yticklabels([str(layers[i]) for i in yt], fontsize=7)

    fig.suptitle(f"{model_title} — Layer × Checkpoint Heatmaps", fontsize=12, y=1.01)
    plt.tight_layout()
    save(fig, outdir / f"{tag}_heatmaps")


# =============================================================================
# FIGURE 3: JSD OVER TRAINING (per layer)
# =============================================================================

def fig_jsd_evolution(
    df_jsd: pd.DataFrame,
    model_title: str,
    outdir: Path,
    tag: str,
):
    """JSD(bucket 0 vs 7) over training steps, one line per layer."""
    if df_jsd.empty:
        return

    layers = sorted(df_jsd["layer"].dropna().unique().astype(int))
    cmap = plt.cm.get_cmap(LAYER_CMAP, len(layers))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, layer in enumerate(layers):
        sub = df_jsd[df_jsd["layer"] == layer].sort_values("step")
        ax.plot(sub["step"], sub["JSD"], color=cmap(i), marker="o",
                markersize=3, linewidth=1.3, label=f"L{layer}")

    ax.set_xlabel("Training Step")
    ax.set_ylabel("JSD (Low vs High Freq)")
    ax.set_title(f"{model_title} — JSD Evolution Over Training", fontsize=11)
    ax.legend(fontsize=7, ncol=min(4, len(layers)), loc="best")
    ax.grid(True, alpha=0.3)

    # Format x-axis
    ticks = ax.get_xticks()
    ax.set_xticklabels([f"{int(t) // 1000}k" if t >= 1000 else str(int(t)) for t in ticks])

    plt.tight_layout()
    save(fig, outdir / f"{tag}_jsd_evolution")


# =============================================================================
# FIGURE 4: JSD + POLYTOPE JOINT TEMPORAL (the key story figure)
# =============================================================================

def fig_joint_temporal(
    df_poly: Optional[pd.DataFrame],
    df_jsd: Optional[pd.DataFrame],
    model_title: str,
    outdir: Path,
    tag: str,
):
    """
    Two-panel figure showing polytope density divergence and JSD growing together.
    Left: Density gap (high - low) over training, averaged across layers with band.
    Right: JSD over training, averaged across layers with band.
    This is the "intuitive story" figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # --- Left: Polytope density gap ---
    if df_poly is not None and not df_poly.empty:
        layer_data = df_poly[df_poly["layer"] >= 0]
        if not layer_data.empty:
            # Compute per-checkpoint, per-layer density gap
            high = layer_data[layer_data["group"] == "high_freq"].set_index(["checkpoint", "layer"])["density_mean"]
            low = layer_data[layer_data["group"] == "low_freq"].set_index(["checkpoint", "layer"])["density_mean"]
            gap = (high - low).reset_index()
            gap.columns = ["checkpoint", "layer", "density_gap"]

            agg = gap.groupby("checkpoint")["density_gap"].agg(["mean", "std"])
            agg = agg.sort_index()

            ax = axes[0]
            ax.plot(agg.index, agg["mean"], color=COLORS["effect"], linewidth=2, marker="o", markersize=3)
            ax.fill_between(agg.index, agg["mean"] - agg["std"], agg["mean"] + agg["std"],
                            color=COLORS["effect"], alpha=0.15)
            ax.axhline(0, ls="--", color=COLORS["neutral"], alpha=0.5, linewidth=0.8)
            ax.set_xlabel("Training Step")
            ax.set_ylabel("Density Gap (High − Low)")
            ax.set_title("Polytope Density Divergence", fontsize=10)
            ax.grid(True, alpha=0.3)

            # Format ticks
            ticks = agg.index.values
            if len(ticks) > 8:
                ticks = ticks[::max(1, len(ticks) // 6)]
            ax.set_xticks(ticks)
            ax.set_xticklabels([f"{t // 1000}k" for t in ticks], fontsize=7, rotation=45)

    # --- Right: JSD ---
    if df_jsd is not None and not df_jsd.empty:
        agg_jsd = df_jsd.groupby("step")["JSD"].agg(["mean", "std"]).sort_index()

        ax = axes[1]
        ax.plot(agg_jsd.index, agg_jsd["mean"], color=COLORS["real"], linewidth=2, marker="o", markersize=3)
        ax.fill_between(agg_jsd.index, agg_jsd["mean"] - agg_jsd["std"],
                        agg_jsd["mean"] + agg_jsd["std"], color=COLORS["real"], alpha=0.15)
        ax.set_xlabel("Training Step")
        ax.set_ylabel("JSD (Low vs High Freq)")
        ax.set_title("Frequency Bucket JSD", fontsize=10)
        ax.grid(True, alpha=0.3)

        ticks = agg_jsd.index.values
        if len(ticks) > 8:
            ticks = ticks[::max(1, len(ticks) // 6)]
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t // 1000}k" for t in ticks], fontsize=7, rotation=45)

    fig.suptitle(f"{model_title} — Polytope Structure and Frequency Separation Co-evolve",
                 fontsize=11, y=1.02)
    plt.tight_layout()
    save(fig, outdir / f"{tag}_joint_temporal")


# =============================================================================
# FIGURE 5: SHUFFLED CONTROL VALIDATION
# =============================================================================

def fig_shuffled_control(
    df_shuffled: pd.DataFrame,
    model_title: str,
    outdir: Path,
    tag: str,
):
    """
    Three-panel shuffled control figure:
    (a) Real vs shuffled JSD over training (layer-averaged)
    (b) Histogram of shuffled JSD at final step vs real value
    (c) Effect size ratio over training
    """
    if df_shuffled.empty:
        return

    df_real = df_shuffled[df_shuffled["condition"] == "real"]
    df_shuf = df_shuffled[df_shuffled["condition"] == "shuffled"]

    if df_real.empty or df_shuf.empty:
        return

    steps = sorted(df_real["step"].unique())
    layers = sorted(df_real["layer"].unique())

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    # --- (a) Real vs Shuffled over training (layer-averaged) ---
    ax = axes[0]
    real_agg = df_real.groupby("step")["JSD"].mean().sort_index()
    shuf_agg = df_shuf.groupby("step")["JSD"].agg(["mean", "std"]).sort_index()

    ax.plot(real_agg.index, real_agg.values, color=COLORS["real"],
            linewidth=2, marker="o", markersize=3, label="Real", zorder=5)
    ax.plot(shuf_agg.index, shuf_agg["mean"].values, color=COLORS["shuffled"],
            linewidth=1.5, marker="s", markersize=2.5, ls="--", label="Shuffled (mean)")
    ax.fill_between(shuf_agg.index,
                    shuf_agg["mean"].values - 2 * shuf_agg["std"].values,
                    shuf_agg["mean"].values + 2 * shuf_agg["std"].values,
                    color=COLORS["shuffled"], alpha=0.2, label="Shuffled ±2σ")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("JSD (layer-avg)")
    ax.set_title("(a) Real vs Shuffled", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    if len(steps) > 6:
        ticks = steps[::max(1, len(steps) // 5)]
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t // 1000}k" for t in ticks], fontsize=7, rotation=45)

    # --- (b) Histogram at final checkpoint ---
    ax = axes[1]
    final_step = max(steps)
    real_final = df_real[df_real["step"] == final_step]["JSD"].values
    shuf_final = df_shuf[df_shuf["step"] == final_step]["JSD"].values

    if len(shuf_final) > 0:
        ax.hist(shuf_final, bins=max(10, len(shuf_final) // 3), color=COLORS["shuffled"],
                alpha=0.6, edgecolor="white", label="Shuffled")
        for rv in real_final:
            ax.axvline(rv, color=COLORS["real"], linewidth=2, ls="-", label="Real")
        ax.set_xlabel("JSD")
        ax.set_ylabel("Count")
        ax.set_title(f"(b) Distribution at Step {final_step // 1000}k", fontsize=10)
        # Deduplicate legend
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), fontsize=7)

    # --- (c) Effect size ratio ---
    ax = axes[2]
    for i, layer in enumerate(layers):
        dr = df_real[df_real["layer"] == layer].set_index("step")["JSD"]
        ds = df_shuf[df_shuf["layer"] == layer].groupby("step")["JSD"].mean()

        common = sorted(set(dr.index) & set(ds.index))
        if not common:
            continue
        ratio = dr.reindex(common).values / np.maximum(ds.reindex(common).values, 1e-12)

        cmap = plt.cm.get_cmap(LAYER_CMAP, len(layers))
        ax.plot(common, ratio, color=cmap(i), linewidth=1.2, marker="o",
                markersize=2, label=f"L{layer}")

    ax.axhline(1.0, ls="--", color=COLORS["neutral"], alpha=0.6, linewidth=0.8)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("JSD_real / JSD_shuffled")
    ax.set_title("(c) Effect Size Ratio", fontsize=10)
    ax.legend(fontsize=6, ncol=min(3, len(layers)), loc="best")
    ax.grid(True, alpha=0.3)

    if len(steps) > 6:
        ticks = steps[::max(1, len(steps) // 5)]
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t // 1000}k" for t in ticks], fontsize=7, rotation=45)

    fig.suptitle(f"{model_title} — Shuffled Control Validates Frequency Effect",
                 fontsize=11, y=1.02)
    plt.tight_layout()
    save(fig, outdir / f"{tag}_shuffled_control")


# =============================================================================
# FIGURE 6: CROSS-SCALE COMPARISON (Pythia 70M → 6.9B)
# =============================================================================

def fig_cross_scale(
    model_data: Dict[str, Dict],
    outdir: Path,
):
    """
    Compare final-checkpoint metrics across model scales.
    Bar chart: density gap and JSD at final step for each model.
    """
    scale_labels = []
    density_gaps = []
    jsd_vals = []

    # Sort models by rough parameter count
    def sort_key(tag):
        for suffix in ["70m", "160m", "410m", "1b", "1.4b", "2.8b", "6.9b", "7b", "12b"]:
            if suffix in tag.lower():
                return float(suffix.replace("m", "e6").replace("b", "e9"))
        return 0

    sorted_tags = sorted(model_data.keys(), key=sort_key)

    for tag in sorted_tags:
        data = model_data[tag]

        # Density gap at final checkpoint
        if "polytope_df" in data:
            df = data["polytope_df"]
            layer_data = df[df["layer"] >= 0]
            if not layer_data.empty:
                final_ckpt = layer_data["checkpoint"].max()
                final = layer_data[layer_data["checkpoint"] == final_ckpt]
                high_d = final[final["group"] == "high_freq"]["density_mean"].mean()
                low_d = final[final["group"] == "low_freq"]["density_mean"].mean()
                density_gaps.append(high_d - low_d)
            else:
                density_gaps.append(np.nan)
        else:
            density_gaps.append(np.nan)

        # JSD at final step
        if "jsd_df" in data:
            df_jsd = data["jsd_df"]
            final_step = df_jsd["step"].max()
            jsd_vals.append(df_jsd[df_jsd["step"] == final_step]["JSD"].mean())
        else:
            jsd_vals.append(np.nan)

        scale_labels.append(tag.replace("pythia-", "").replace("olmo-", "OLMo-").upper())

    if not scale_labels:
        return

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    x = np.arange(len(scale_labels))
    w = 0.5

    # Density gap bars
    ax = axes[0]
    valid = ~np.isnan(density_gaps)
    if valid.any():
        ax.bar(x[valid], np.array(density_gaps)[valid], width=w, color=COLORS["effect"], alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(scale_labels, fontsize=8)
    ax.set_ylabel("Density Gap (High − Low)")
    ax.set_title("Polytope Density Gap", fontsize=10)
    ax.axhline(0, ls="--", color=COLORS["neutral"], alpha=0.4)
    ax.grid(True, axis="y", alpha=0.3)

    # JSD bars
    ax = axes[1]
    valid = ~np.isnan(jsd_vals)
    if valid.any():
        ax.bar(x[valid], np.array(jsd_vals)[valid], width=w, color=COLORS["real"], alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(scale_labels, fontsize=8)
    ax.set_ylabel("JSD (Low vs High Freq)")
    ax.set_title("Frequency Separation (JSD)", fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Cross-Scale Comparison at Final Checkpoint", fontsize=11, y=1.02)
    plt.tight_layout()
    save(fig, outdir / "cross_scale_comparison")


# =============================================================================
# FIGURE 7: PER-LAYER TEMPORAL DEEP DIVE (Pythia detailed)
# =============================================================================

def fig_layer_deep_dive(
    df_poly: Optional[pd.DataFrame],
    df_jsd: Optional[pd.DataFrame],
    model_title: str,
    outdir: Path,
    tag: str,
):
    """
    Detailed per-layer view: each layer gets a row with 4 metrics side by side.
    Columns: density, participation ratio, pattern reuse, JSD (if available).
    Only show a representative subset of layers (early, mid, late).
    """
    if df_poly is None or df_poly.empty:
        return

    layer_data = df_poly[df_poly["layer"] >= 0]
    if layer_data.empty:
        return

    all_layers = sorted(layer_data["layer"].unique())

    # Pick representative layers: first, ~25%, ~50%, ~75%, last
    if len(all_layers) > 5:
        idxs = np.linspace(0, len(all_layers) - 1, 5).astype(int)
        layers = [all_layers[i] for i in idxs]
    else:
        layers = all_layers

    has_jsd = df_jsd is not None and not df_jsd.empty
    n_cols = 4 if has_jsd else 3
    n_rows = len(layers)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 2.5 * n_rows),
                              sharex=True)
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    poly_metrics = [
        ("density_mean", "Polytope Density"),
        ("participation_ratio", "Participation Ratio"),
        ("pattern_reuse_rate", "Pattern Reuse Rate"),
    ]

    for row, layer in enumerate(layers):
        # Polytope metrics
        for col, (metric, ylabel) in enumerate(poly_metrics):
            ax = axes[row, col]
            for group in ["high_freq", "low_freq"]:
                sub = layer_data[(layer_data["layer"] == layer) & (layer_data["group"] == group)]
                sub = sub.sort_values("checkpoint")
                if sub.empty or metric not in sub.columns:
                    continue
                ax.plot(sub["checkpoint"], sub[metric],
                        color=COLORS[group], marker="o" if group == "high_freq" else "s",
                        markersize=2, linewidth=1.2)
            ax.grid(True, alpha=0.3)
            if row == 0:
                ax.set_title(ylabel, fontsize=9)
            if col == 0:
                ax.set_ylabel(f"Layer {layer}", fontsize=9)
            if row == n_rows - 1:
                checkpoints = sorted(layer_data["checkpoint"].unique())
                if len(checkpoints) > 5:
                    ticks = checkpoints[::max(1, len(checkpoints) // 4)]
                    ax.set_xticks(ticks)
                    ax.set_xticklabels([f"{t // 1000}k" for t in ticks], fontsize=7, rotation=45)

        # JSD column
        if has_jsd:
            ax = axes[row, n_cols - 1]
            jsd_layer = df_jsd[df_jsd["layer"] == layer].sort_values("step")
            if not jsd_layer.empty:
                ax.plot(jsd_layer["step"], jsd_layer["JSD"],
                        color=COLORS["real"], marker="o", markersize=2, linewidth=1.2)
            ax.grid(True, alpha=0.3)
            if row == 0:
                ax.set_title("JSD", fontsize=9)
            if row == n_rows - 1:
                steps = sorted(df_jsd["step"].unique())
                if len(steps) > 5:
                    ticks = steps[::max(1, len(steps) // 4)]
                    ax.set_xticks(ticks)
                    ax.set_xticklabels([f"{t // 1000}k" for t in ticks], fontsize=7, rotation=45)

    # Legend
    handles = [
        Line2D([0], [0], color=COLORS["high_freq"], marker="o", markersize=4, label="High Freq"),
        Line2D([0], [0], color=COLORS["low_freq"], marker="s", markersize=4, label="Low Freq"),
    ]
    if has_jsd:
        handles.append(Line2D([0], [0], color=COLORS["real"], marker="o", markersize=4, label="JSD"))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.02),
               ncol=len(handles), frameon=False, fontsize=8)
    fig.suptitle(f"{model_title} — Per-Layer Deep Dive", y=1.05, fontsize=11)
    plt.tight_layout()
    save(fig, outdir / f"{tag}_layer_deep_dive")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate paper figures")
    parser.add_argument("--polytope-dir", type=Path, default=Path("./polytope_results"),
                        help="Directory containing polytope_analysis_results*.json")
    parser.add_argument("--jsd-dir", type=Path, default=Path("./jsd_polysemanticity/runs_jsd_poly"),
                        help="Directory containing JSD run outputs")
    parser.add_argument("--shuffled-dir", type=Path,
                        default=Path("./jsd_polysemanticity/runs_jsd_poly_shuffled_control"),
                        help="Directory containing shuffled control outputs")
    parser.add_argument("--output-dir", type=Path, default=Path("./neurips-2026/figures/main"),
                        help="Output directory for figures")
    args = parser.parse_args()

    set_paper_style()
    outdir = args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)

    print("Discovering data files...")
    models = find_data_files(args.polytope_dir, args.jsd_dir, args.shuffled_dir)

    if not models:
        print("No data files found. Check --polytope-dir, --jsd-dir, --shuffled-dir paths.")
        print(f"  Looked in: {args.polytope_dir}, {args.jsd_dir}, {args.shuffled_dir}")
        return

    print(f"Found models: {list(models.keys())}")

    # Load all data
    all_model_data = {}
    for tag, paths in models.items():
        print(f"\nLoading {tag}...")
        data = {}

        if "polytope" in paths:
            data["polytope_df"] = load_polytope_results(paths["polytope"])
            print(f"  polytope: {len(data['polytope_df'])} rows")

        if "jsd" in paths:
            data["jsd_df"] = load_jsd_results(paths["jsd"])
            print(f"  jsd: {len(data['jsd_df'])} rows")

        if "shuffled" in paths:
            data["shuffled_df"] = load_shuffled_results(paths["shuffled"])
            print(f"  shuffled: {len(data['shuffled_df'])} rows")

        if "shuffled_stats" in paths:
            data["shuffled_stats_df"] = pd.read_csv(paths["shuffled_stats"])

        all_model_data[tag] = data

    # Generate figures per model
    for tag, data in all_model_data.items():
        title = tag.replace("-", " ").title().replace("Pythia", "Pythia").replace("Olmo", "OLMo")
        model_outdir = outdir / tag
        print(f"\nGenerating figures for {tag}...")

        if "polytope_df" in data:
            print("  Fig: polytope temporal evolution")
            fig_polytope_temporal(data["polytope_df"], title, model_outdir, tag)

            print("  Fig: layer × checkpoint heatmaps")
            fig_heatmaps(data["polytope_df"], title, model_outdir, tag)

        if "jsd_df" in data:
            print("  Fig: JSD evolution")
            fig_jsd_evolution(data["jsd_df"], title, model_outdir, tag)

        if "polytope_df" in data or "jsd_df" in data:
            print("  Fig: joint temporal (polytope + JSD)")
            fig_joint_temporal(
                data.get("polytope_df"), data.get("jsd_df"),
                title, model_outdir, tag,
            )

        if "polytope_df" in data:
            print("  Fig: per-layer deep dive")
            fig_layer_deep_dive(
                data.get("polytope_df"), data.get("jsd_df"),
                title, model_outdir, tag,
            )

        if "shuffled_df" in data:
            print("  Fig: shuffled control validation")
            fig_shuffled_control(data["shuffled_df"], title, model_outdir, tag)

    # Cross-scale comparison
    if len(all_model_data) > 1:
        print("\nFig: cross-scale comparison")
        fig_cross_scale(all_model_data, outdir)

    print(f"\nAll figures saved to {outdir.resolve()}")
    print("PDF files are Overleaf-ready (Type 42 fonts, tight bbox).")


if __name__ == "__main__":
    main()
