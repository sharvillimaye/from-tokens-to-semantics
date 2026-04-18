#!/usr/bin/env python3
"""
Compare write-site and reader-site patterns between the frequency direction
and the sentiment direction for the SAME model.

Inputs (two attribution CSVs, same model):
  --freq-dir      dir containing frequency pipeline outputs
                    (attribution_scores.csv, top_readers.csv, summary.json, probe.json)
  --sent-dir      dir containing sentiment pipeline outputs
                    (sentiment_attribution_scores.csv, sentiment_top_readers.csv,
                     sentiment_qkv_summary.json, sentiment_probe.json)
  --output-dir    where to write the comparison

Outputs:
  comparison_freq_vs_sentiment.json
  comparison_plot.png

Shows:
  - Top-10 write-site split (MLP vs attn_head) per direction
  - Ratio Σ|attr| over MLPs vs Σ|attr| over attn heads, per direction
  - Overlap of top-K attention heads across directions
  - QKV reader significant-head counts per direction
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _load_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        print(f"  WARN: missing {path}")
        return None
    return pd.read_csv(path)


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def top_k_split(df: pd.DataFrame, k: int = 10) -> Dict[str, int]:
    top = df.sort_values("abs_score", ascending=False).head(k)
    counts = top["component_type"].value_counts().to_dict()
    return {"mlp": int(counts.get("mlp", 0)),
            "attn_head": int(counts.get("attn_head", 0))}


def mlp_attn_mass_ratio(df: pd.DataFrame) -> Dict[str, float]:
    attn_mass = float(df.loc[df.component_type == "attn_head", "abs_score"].sum())
    mlp_mass = float(df.loc[df.component_type == "mlp", "abs_score"].sum())
    ratio = mlp_mass / attn_mass if attn_mass > 0 else float("inf")
    return {"mlp_total_abs_score": mlp_mass,
            "attn_total_abs_score": attn_mass,
            "mlp_over_attn_ratio": ratio}


def top_components(df: pd.DataFrame, k: int = 10) -> List[Dict[str, Any]]:
    top = df.sort_values("abs_score", ascending=False).head(k)
    return [
        {
            "layer": int(r["layer"]),
            "component_type": str(r["component_type"]),
            "component_index": int(r["component_index"]),
            "attribution_score": float(r["attribution_score"]),
            "abs_score": float(r["abs_score"]),
        }
        for _, r in top.iterrows()
    ]


def head_overlap(freq_df: pd.DataFrame, sent_df: pd.DataFrame, k: int = 20
                 ) -> Dict[str, Any]:
    def top_heads(df):
        heads = df[df.component_type == "attn_head"].sort_values(
            "abs_score", ascending=False).head(k)
        return {(int(r["layer"]), int(r["component_index"])) for _, r in heads.iterrows()}
    f = top_heads(freq_df)
    s = top_heads(sent_df)
    inter = f & s
    return {
        "k": k,
        "freq_top_heads": sorted(list(f)),
        "sentiment_top_heads": sorted(list(s)),
        "intersection": sorted(list(inter)),
        "jaccard": float(len(inter) / max(len(f | s), 1)),
    }


def make_plot(freq_split: Dict[str, int], sent_split: Dict[str, int],
              freq_mass: Dict[str, float], sent_mass: Dict[str, float],
              out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # LEFT: top-10 write-site split
    ax = axes[0]
    labels = ["frequency", "sentiment"]
    mlp_vals = [freq_split["mlp"], sent_split["mlp"]]
    attn_vals = [freq_split["attn_head"], sent_split["attn_head"]]
    x = np.arange(len(labels))
    w = 0.35
    ax.bar(x - w/2, mlp_vals, w, label="MLP", color="#4b7bec")
    ax.bar(x + w/2, attn_vals, w, label="Attn head", color="#eb3b5a")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Count in top-10 |attribution|")
    ax.set_title("Top-10 writer split: MLP vs Attention")
    ax.legend()
    for xi, (m, a) in enumerate(zip(mlp_vals, attn_vals)):
        ax.text(xi - w/2, m + 0.1, str(m), ha="center")
        ax.text(xi + w/2, a + 0.1, str(a), ha="center")

    # RIGHT: total |attribution| mass ratio
    ax = axes[1]
    fr = freq_mass["mlp_over_attn_ratio"]
    sr = sent_mass["mlp_over_attn_ratio"]
    ratios = [fr, sr]
    colors = ["#4b7bec" if r > 1 else "#eb3b5a" for r in ratios]
    bars = ax.bar(labels, ratios, color=colors)
    ax.axhline(1.0, color="grey", linestyle="--", linewidth=1)
    ax.set_ylabel("Σ|attr|(MLP) / Σ|attr|(Attn)")
    ax.set_title("Writer mass ratio (>1 = MLP-dominant)")
    for b, r in zip(bars, ratios):
        label = f"{r:.2f}" if np.isfinite(r) else "inf"
        ax.text(b.get_x() + b.get_width()/2, b.get_height(), label,
                ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--freq-dir", required=True, help="Frequency pipeline results dir")
    p.add_argument("--sent-dir", required=True, help="Sentiment pipeline results dir")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--reader-top-k", type=int, default=20)
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    freq_dir = Path(args.freq_dir)
    sent_dir = Path(args.sent_dir)

    # Frequency artifacts: attribution_scores.csv (from frequency_circuit_eap.py)
    freq_attr = _load_csv(freq_dir / "attribution_scores.csv")
    freq_probe = _load_json(freq_dir / "probe.json")
    freq_qkv_summary = _load_json(freq_dir / "summary.json")

    # Sentiment artifacts
    sent_attr = _load_csv(sent_dir / "sentiment_attribution_scores.csv")
    sent_probe = _load_json(sent_dir / "sentiment_probe.json")
    sent_qkv_summary = _load_json(sent_dir / "sentiment_qkv_summary.json")

    comparison: Dict[str, Any] = {
        "freq_probe": freq_probe,
        "sent_probe": sent_probe,
    }

    if freq_attr is not None and sent_attr is not None:
        comparison["top_k_split"] = {
            "k": args.top_k,
            "frequency": top_k_split(freq_attr, args.top_k),
            "sentiment": top_k_split(sent_attr, args.top_k),
        }
        comparison["writer_mass"] = {
            "frequency": mlp_attn_mass_ratio(freq_attr),
            "sentiment": mlp_attn_mass_ratio(sent_attr),
        }
        comparison["top_components"] = {
            "frequency": top_components(freq_attr, args.top_k),
            "sentiment": top_components(sent_attr, args.top_k),
        }
        comparison["attn_head_overlap"] = head_overlap(
            freq_attr, sent_attr, args.reader_top_k)

        make_plot(
            comparison["top_k_split"]["frequency"],
            comparison["top_k_split"]["sentiment"],
            comparison["writer_mass"]["frequency"],
            comparison["writer_mass"]["sentiment"],
            out / "comparison_plot.png",
        )
    else:
        print("  Missing attribution CSV; skipping plot.")

    if freq_qkv_summary is not None and sent_qkv_summary is not None:
        def _reader_row(s):
            return {
                k: s.get(k) for k in (
                    "n_heads_significant_Q", "n_heads_significant_K", "n_heads_significant_V",
                    "n_total_Q", "n_total_K", "n_total_V",
                    "max_max_cos_Q", "max_max_cos_K", "max_max_cos_V",
                    "best_layer_L_star",
                )
            }
        comparison["qkv_readers"] = {
            "frequency": _reader_row(freq_qkv_summary),
            "sentiment": _reader_row(sent_qkv_summary),
        }

    with open(out / "comparison_freq_vs_sentiment.json", "w") as f:
        json.dump(comparison, f, indent=2, default=str)

    print(f"Wrote comparison to {out / 'comparison_freq_vs_sentiment.json'}")
    if (out / "comparison_plot.png").exists():
        print(f"Wrote plot to {out / 'comparison_plot.png'}")


if __name__ == "__main__":
    main()
