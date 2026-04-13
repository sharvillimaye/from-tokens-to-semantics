#!/usr/bin/env python3
"""
Analyze the per-neuron frequency projection distributions from circuit tracing.

Key questions:
  1. Is frequency writing concentrated in a small fraction of neurons,
     or distributed across many?
  2. What fraction of total |diff_proj| is accounted for by top 1%, 5%, 10%?
  3. How does this vary by layer?
  4. Do the top-projection neurons across layers form a coherent circuit
     (i.e., consistent across datasets)?

Usage:
    python -m scripts.plots.analyze_circuit_concentration \
        --circuit-dir /data/ani/mechinterp/runs/circuit_v2/pythia-6.9b \
        --model-name pythia-6.9b \
        --output-dir /data/ani/mechinterp/runs/circuit_analysis/pythia-6.9b
"""

from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


DATASETS = ["emotion", "medical", "legal", "scientific", "verb"]


def load_all_projections(circuit_dir: str) -> pd.DataFrame:
    dfs = []
    for ds in DATASETS:
        path = Path(circuit_dir) / f"{ds}_neuron_freq_projections.csv"
        if path.exists():
            df = pd.read_csv(path)
            dfs.append(df)
    if not dfs:
        raise FileNotFoundError(f"No CSVs in {circuit_dir}")
    return pd.concat(dfs, ignore_index=True)


def concentration_by_layer(df: pd.DataFrame) -> pd.DataFrame:
    """For each layer, compute what % of total |diff_proj| is in top-k% neurons."""
    rows = []
    # Average across datasets first for stable ranking
    per_neuron = df.groupby(["layer", "neuron_idx"])["abs_diff_proj"].mean().reset_index()

    for L, group in per_neuron.groupby("layer"):
        vals = group["abs_diff_proj"].values
        sorted_desc = np.sort(vals)[::-1]
        total = sorted_desc.sum()
        if total == 0:
            continue

        n = len(sorted_desc)
        row = {"layer": int(L), "n_neurons": n, "total_abs_diff_proj": float(total)}
        for pct in [1, 5, 10, 25, 50]:
            k = max(1, int(n * pct / 100))
            row[f"top_{pct}pct_mass"] = float(sorted_desc[:k].sum() / total)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("layer")


def consistency_across_datasets(df: pd.DataFrame, pct: float = 5.0) -> pd.DataFrame:
    """For each layer, compute the Jaccard overlap between top-pct% neuron sets
    identified from each dataset. High overlap = dataset-invariant circuit."""
    rows = []
    for L in sorted(df["layer"].unique()):
        layer_df = df[df["layer"] == L]
        ds_sets: Dict[str, set] = {}
        for ds in DATASETS:
            ds_rows = layer_df[layer_df["dataset"] == ds]
            if len(ds_rows) == 0:
                continue
            n_top = max(1, int(len(ds_rows) * pct / 100))
            top_ids = ds_rows.nlargest(n_top, "abs_diff_proj")["neuron_idx"].values
            ds_sets[ds] = set(int(x) for x in top_ids)

        if len(ds_sets) < 2:
            continue

        ds_names = list(ds_sets.keys())
        overlaps = []
        for i in range(len(ds_names)):
            for j in range(i + 1, len(ds_names)):
                a, b = ds_sets[ds_names[i]], ds_sets[ds_names[j]]
                if not a and not b:
                    jac = 1.0
                else:
                    jac = len(a & b) / len(a | b)
                overlaps.append(jac)

        rows.append({
            "layer": int(L),
            "pct": pct,
            "mean_jaccard": float(np.mean(overlaps)),
            "std_jaccard": float(np.std(overlaps)),
            "n_pairs": len(overlaps),
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--circuit-dir", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading circuit data from {args.circuit_dir}")
    df = load_all_projections(args.circuit_dir)
    print(f"  Loaded {len(df):,} rows ({df['layer'].nunique()} layers, "
          f"{df['neuron_idx'].nunique():,} unique neurons, "
          f"{df['dataset'].nunique()} datasets)")

    # 1. Concentration
    print("\n=== Frequency projection concentration by layer ===")
    conc = concentration_by_layer(df)
    conc.to_csv(Path(args.output_dir) / "concentration.csv", index=False)
    print(conc.to_string(index=False))

    # 2. Cross-dataset consistency
    print("\n=== Cross-dataset consistency of top-5% neurons (Jaccard) ===")
    cons = consistency_across_datasets(df, pct=5.0)
    cons.to_csv(Path(args.output_dir) / "consistency_pct5.csv", index=False)
    print(cons.to_string(index=False))

    # 3. Also check top-1% consistency (tighter circuit)
    cons1 = consistency_across_datasets(df, pct=1.0)
    cons1.to_csv(Path(args.output_dir) / "consistency_pct1.csv", index=False)

    # 4. Summary stats
    summary = {
        "model": args.model_name,
        "n_neurons_total": int(df["neuron_idx"].nunique()),
        "n_layers": int(df["layer"].nunique()),
        "n_datasets": int(df["dataset"].nunique()),
        "mean_top1pct_mass": float(conc["top_1pct_mass"].mean()),
        "mean_top5pct_mass": float(conc["top_5pct_mass"].mean()),
        "mean_top10pct_mass": float(conc["top_10pct_mass"].mean()),
        "mean_jaccard_pct5": float(cons["mean_jaccard"].mean()) if len(cons) else 0.0,
        "mean_jaccard_pct1": float(cons1["mean_jaccard"].mean()) if len(cons1) else 0.0,
    }
    print("\n=== Summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    with open(Path(args.output_dir) / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
