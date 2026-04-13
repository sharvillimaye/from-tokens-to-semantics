#!/usr/bin/env python3
"""
Circuit-based ablation: ablate neurons selected by their circuit-level
frequency projection (from frequency_circuit_tracing.py), not abstract
descriptive axes.

This is the proper causal test of the frequency routing circuit:
  - Circuit tracing identified neurons with large |diff_proj| as
    "frequency writers" — their output aligns with the local frequency
    direction
  - If the circuit hypothesis is correct, ablating these specific
    neurons should drop frequency probe AUROC more than ablating
    random neurons

Conditions:
  - clean (no ablation)
  - circuit_<pct>: top-k neurons by |diff_proj| per layer
  - circuit_pos_<pct>: top-k by diff_proj (high-freq writers)
  - circuit_neg_<pct>: bottom-k by diff_proj (low-freq writers)
  - random_<pct>: random ablation (averaged over seeds)

Ablation percentages: 5, 10, 25, 50

Usage:
    python -m scripts.interventions.circuit_ablation \
        --model EleutherAI/pythia-6.9b-deduped --revision step143000 \
        --dataset-dir ~/scaleJSD/dataset/legacy/filtered \
        --circuit-dir /data/ani/mechinterp/runs/circuit_v2/pythia-6.9b \
        --output-dir /data/ani/mechinterp/runs/circuit_ablation/pythia-6.9b \
        --ablate-layers 22,23,24,25,26,27,28,29,30,31 \
        --ablate-pct 5,10,25,50
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch

try:
    from scripts.metrics.coverage_affinity_experiment import (
        _infer_layers,
        load_synonym_pairs,
        pairs_to_samples,
    )
    from scripts.interventions.probe_under_ablation import (
        extract_residual_stream,
        train_frequency_probe,
    )
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from scripts.metrics.coverage_affinity_experiment import (
        _infer_layers,
        load_synonym_pairs,
        pairs_to_samples,
    )
    from scripts.interventions.probe_under_ablation import (
        extract_residual_stream,
        train_frequency_probe,
    )


DATASET_FILES = {
    "emotion": "emotion_ngrams_dedup_filtered.jsonl",
    "medical": "medical_ngrams_dedup_filtered.jsonl",
    "legal": "legal_ngrams_dedup_filtered.jsonl",
    "scientific": "scientific_ngrams_dedup_filtered.jsonl",
    "verb": "verb_ngrams_dedup_filtered.jsonl",
}


def load_circuit_neuron_projections(circuit_dir: str) -> pd.DataFrame:
    """Load per-neuron frequency projections, averaged across datasets."""
    dfs = []
    for ds in DATASET_FILES.keys():
        path = Path(circuit_dir) / f"{ds}_neuron_freq_projections.csv"
        if path.exists():
            df = pd.read_csv(path)
            dfs.append(df)
        else:
            print(f"  WARNING: {path} not found")

    if not dfs:
        raise FileNotFoundError(f"No neuron projection files in {circuit_dir}")

    full = pd.concat(dfs, ignore_index=True)
    # Average across datasets per (layer, neuron_idx)
    agg = full.groupby(["layer", "neuron_idx"])[
        ["diff_proj", "abs_diff_proj", "mean_proj_high", "mean_proj_low"]
    ].mean().reset_index()
    return agg


def select_circuit_neurons(
    circuit_df: pd.DataFrame,
    layer: int,
    pct: float,
    strategy: str,
    seed: int = 42,
) -> np.ndarray:
    """Select neurons by circuit-level metric."""
    layer_df = circuit_df[circuit_df["layer"] == layer]
    if len(layer_df) == 0:
        return np.array([], dtype=np.int64)
    n_total = len(layer_df)
    n_select = int(n_total * pct / 100.0)

    if strategy == "circuit":
        order = layer_df["abs_diff_proj"].values.argsort()[::-1]
    elif strategy == "circuit_pos":
        order = layer_df["diff_proj"].values.argsort()[::-1]
    elif strategy == "circuit_neg":
        order = layer_df["diff_proj"].values.argsort()
    elif strategy == "random":
        rng = np.random.RandomState(seed)
        order = rng.permutation(n_total)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    neuron_idx_vals = layer_df["neuron_idx"].values
    return neuron_idx_vals[order[:n_select]].astype(np.int64)


def run_condition(
    model,
    samples: List[Dict[str, Any]],
    all_layers: List[int],
    freq_labels: np.ndarray,
    pair_ids: np.ndarray,
    ablation: Optional[Dict[int, np.ndarray]],
    device: str,
    condition: str,
    pct: float,
    seed: int,
    dataset: str,
    n_per_layer: int,
) -> List[Dict[str, Any]]:
    """Run one ablation condition: extract residuals, train probes per layer."""
    residual = extract_residual_stream(
        model, samples, all_layers, device, neurons_to_ablate=ablation)

    results = []
    for L in all_layers:
        X = residual[L]
        probe = train_frequency_probe(X, freq_labels, pair_ids)
        results.append({
            "condition": condition,
            "pct": pct,
            "seed": seed,
            "dataset": dataset,
            "layer": L,
            "n_ablated_per_layer": n_per_layer,
            "freq_auroc": probe["frequency_auroc"],
            "freq_auroc_std": probe["frequency_auroc_std"],
            "freq_accuracy": probe["frequency_accuracy"],
        })
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--dataset-dir", type=str, required=True)
    parser.add_argument("--circuit-dir", type=str, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--ablate-layers", type=str, required=True)
    parser.add_argument("--ablate-pct", type=str, default="5,10,25,50")
    parser.add_argument("--n-random-seeds", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.makedirs(args.output_dir, exist_ok=True)
    ablate_layers = [int(x) for x in args.ablate_layers.split(",")]
    pcts = [float(x) for x in args.ablate_pct.split(",")]

    print(f"Loading circuit projections from {args.circuit_dir}")
    circuit_df = load_circuit_neuron_projections(args.circuit_dir)
    print(f"  Loaded {len(circuit_df)} (layer, neuron) rows")

    print(f"\nLoading model {args.model} (revision={args.revision})")
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision,
        torch_dtype=torch.float32,
    ).to(args.device)
    model.eval()

    all_layers = _infer_layers(args.model)
    all_results: List[Dict[str, Any]] = []

    for ds_name, fname in DATASET_FILES.items():
        ds_path = Path(args.dataset_dir) / fname
        if not ds_path.exists():
            print(f"  Skipping {ds_name}: not found")
            continue

        print(f"\n{'='*60}\n  Dataset: {ds_name}\n{'='*60}")
        pairs = load_synonym_pairs(str(ds_path))
        samples = pairs_to_samples(pairs, tokenizer)
        if not samples:
            continue

        freq_labels = np.array(
            [1 if s["frequency_category"] == "high_freq" else 0 for s in samples])
        pair_ids = np.array([s["pair_id"] for s in samples])

        # Clean baseline
        print("  → clean")
        all_results.extend(run_condition(
            model, samples, all_layers, freq_labels, pair_ids,
            None, args.device, "clean", 0, 0, ds_name, 0))

        # Circuit-based ablations
        for strategy in ["circuit", "circuit_pos", "circuit_neg"]:
            for pct in pcts:
                print(f"  → {strategy}_{pct:.0f}pct")
                ablation = {
                    L: select_circuit_neurons(circuit_df, L, pct, strategy)
                    for L in ablate_layers
                }
                n_per_layer = int(list(ablation.values())[0].shape[0]) if ablate_layers else 0
                all_results.extend(run_condition(
                    model, samples, all_layers, freq_labels, pair_ids,
                    ablation, args.device, strategy, pct, 0, ds_name, n_per_layer))

        # Random baselines
        for pct in pcts:
            for seed in range(args.n_random_seeds):
                print(f"  → random_{pct:.0f}pct_seed{seed}")
                ablation = {
                    L: select_circuit_neurons(circuit_df, L, pct, "random", seed=seed)
                    for L in ablate_layers
                }
                n_per_layer = int(list(ablation.values())[0].shape[0]) if ablate_layers else 0
                all_results.extend(run_condition(
                    model, samples, all_layers, freq_labels, pair_ids,
                    ablation, args.device, "random", pct, seed, ds_name, n_per_layer))

        # Incremental save
        pd.DataFrame(all_results).to_csv(
            Path(args.output_dir) / "circuit_ablation.csv", index=False)

    df = pd.DataFrame(all_results)
    df.to_csv(Path(args.output_dir) / "circuit_ablation.csv", index=False)

    print(f"\n{'='*60}")
    print("  SUMMARY — Freq AUROC at ablated layers, avg across datasets")
    print(f"{'='*60}")
    ablated_layer_set = set(ablate_layers)
    mask = df["layer"].apply(lambda L: L in ablated_layer_set)
    summary = df[mask].groupby(["condition", "pct"])["freq_auroc"].agg(["mean", "std", "count"])
    print(summary.to_string())


if __name__ == "__main__":
    main()
