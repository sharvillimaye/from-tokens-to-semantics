#!/usr/bin/env python3
"""
Targeted ablation of frequency-specialized neurons.

Tests whether high-affinity neurons in late layers are causally responsible for
the frequency signal remaining in the residual stream. If ablating top-K%
highest-affinity neurons collapses the JSD between synonym pairs more than
ablating random neurons or top-coverage neurons, then affinity identifies
the causal mechanism for residual frequency encoding.

Ablation method: zero the post-activation output of selected neurons during
the forward pass (equivalent to removing their contribution to the residual
stream via the down-projection).

Three conditions:
  1. affinity: ablate neurons with highest |affinity - 0.5| (most frequency-selective)
  2. coverage: ablate neurons with highest binary coverage (broadest-firing)
  3. random: ablate random neurons (control)

Measurements:
  - Group JSD between high-freq and low-freq synonym activations (per layer)
  - Pairwise JSD between matched synonym pairs
  - Output KL divergence between clean and ablated model

Usage:
    python targeted_ablation.py \
        --model EleutherAI/pythia-70m-deduped \
        --revision step143000 \
        --neuron-metrics results/coverage_affinity/pythia-70m/emotion/neuron_metrics.csv \
        --dataset ~/dev/personal/mechinterp/scaleJSD/dataset/legacy/filtered/emotion_ngrams_dedup_filtered.jsonl \
        --output-dir results/ablation/pythia-70m/emotion/ \
        --ablate-layers 3,4,5 \
        --ablate-pct 5

    # All datasets
    python targeted_ablation.py \
        --model EleutherAI/pythia-70m-deduped \
        --revision step143000 \
        --all-datasets \
        --dataset-dir ~/dev/personal/mechinterp/scaleJSD/dataset/legacy/filtered \
        --metrics-dir results/coverage_affinity/pythia-70m \
        --output-dir results/ablation/pythia-70m/ \
        --ablate-pct 1,2,5,10
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


EPS = 1e-12

# ─────────────────────────────────────────────────────────────────────────────
# Reuse ActivationCapture + data loading from coverage_affinity_experiment
# ─────────────────────────────────────────────────────────────────────────────

from coverage_affinity_experiment import (
    ActivationCapture,
    load_synonym_pairs,
    pairs_to_samples,
    compute_group_jsd,
    compute_pairwise_jsd,
    _infer_layers,
)


# ─────────────────────────────────────────────────────────────────────────────
# Neuron selection
# ─────────────────────────────────────────────────────────────────────────────

def select_neurons(
    neuron_df: pd.DataFrame,
    layer: int,
    pct: float,
    strategy: str,
    seed: int = 42,
) -> np.ndarray:
    """Select top-pct% neurons from a layer by strategy.

    Returns array of neuron indices to ablate.
    """
    layer_df = neuron_df[neuron_df["layer"] == layer].copy()
    n_total = len(layer_df)
    k = max(1, int(n_total * pct / 100))

    if strategy == "affinity":
        # Most frequency-selective: furthest from 0.5
        layer_df["aff_dev"] = (layer_df["frequency_affinity"] - 0.5).abs()
        selected = layer_df.nlargest(k, "aff_dev")["neuron"].values
    elif strategy == "coverage":
        selected = layer_df.nlargest(k, "coverage_total")["neuron"].values
    elif strategy == "jsd":
        selected = layer_df.nlargest(k, "jsd_contrib")["neuron"].values
    elif strategy == "random":
        rng = np.random.RandomState(seed)
        selected = rng.choice(layer_df["neuron"].values, size=k, replace=False)
    elif strategy == "mass":
        selected = layer_df.nlargest(k, "raw_mass_total")["neuron"].values
    elif strategy == "low_coverage":
        # Narrowest-firing neurons (bottom pct% by coverage)
        # Filter out dead neurons (coverage == 0) to avoid trivial null
        active = layer_df[layer_df["coverage_total"] > 0]
        if len(active) < k:
            active = layer_df
        selected = active.nsmallest(k, "coverage_total")["neuron"].values
    elif strategy == "affinity_high_freq":
        # Neurons that prefer high-frequency tokens (affinity >> 0.5)
        selected = layer_df.nlargest(k, "frequency_affinity")["neuron"].values
    elif strategy == "affinity_low_freq":
        # Neurons that prefer low-frequency tokens (affinity << 0.5)
        selected = layer_df.nsmallest(k, "frequency_affinity")["neuron"].values
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return selected.astype(int)


# ─────────────────────────────────────────────────────────────────────────────
# Ablation hooks
# ─────────────────────────────────────────────────────────────────────────────

class AblationHook:
    """Zero out specific neuron dimensions in the post-activation MLP intermediate.

    Hooks into the down-projection input and zeros selected dimensions,
    effectively removing those neurons' contribution to the residual stream.
    """

    def __init__(self, neurons_to_ablate: Dict[int, np.ndarray]):
        """neurons_to_ablate: dict mapping layer_idx -> array of neuron indices."""
        self.neurons_to_ablate = neurons_to_ablate
        self.hooks: list = []

    def _find_model_layers(self, model):
        for attr in [
            "gpt_neox.layers", "model.layers", "transformer.h",
            "transformer.layers", "model.decoder.layers",
        ]:
            obj = model
            try:
                for part in attr.split("."):
                    obj = getattr(obj, part)
                return obj
            except AttributeError:
                continue
        raise RuntimeError("Cannot find transformer layers in model")

    @staticmethod
    def _find_mlp_and_down(layer_mod):
        mlp_names = ["mlp", "feed_forward", "ff"]
        down_names = ["dense_4h_to_h", "down_proj", "c_proj", "fc2", "w2"]
        for mn in mlp_names:
            mlp = getattr(layer_mod, mn, None)
            if mlp is None:
                continue
            for dn in down_names:
                down = getattr(mlp, dn, None)
                if down is not None:
                    return mlp, down
        return None, None

    def register(self, model):
        model_layers = self._find_model_layers(model)
        for L, neurons in self.neurons_to_ablate.items():
            layer_mod = model_layers[L]
            _, down_mod = self._find_mlp_and_down(layer_mod)
            if down_mod is None:
                raise RuntimeError(f"Cannot find down-projection in layer {L}")

            neuron_mask = torch.tensor(neurons, dtype=torch.long)

            def make_hook(mask):
                def hook_fn(_mod, inputs):
                    x = inputs[0] if isinstance(inputs, tuple) else inputs
                    x[:, :, mask] = 0.0
                    return (x,) if isinstance(inputs, tuple) else x
                return hook_fn

            # Use forward_pre_hook on down_proj to zero neurons before down-projection
            self.hooks.append(
                down_mod.register_forward_pre_hook(make_hook(neuron_mask))
            )

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


# ─────────────────────────────────────────────────────────────────────────────
# Measurement: extract activations + logits with optional ablation
# ─────────────────────────────────────────────────────────────────────────────

def run_forward(
    model,
    samples: List[Dict],
    layers: List[int],
    device: str,
    ablation_hook: Optional[AblationHook] = None,
) -> Tuple[Dict[int, np.ndarray], List[np.ndarray]]:
    """Run forward pass, return (layer_activations, per_sample_logits).

    layer_activations: dict[layer] -> np.ndarray [N_samples, 4H]
    logits: list of np.ndarray [vocab_size] per sample (at anchor position)
    """
    capture = ActivationCapture(model, layers)
    capture.register()

    if ablation_hook is not None:
        ablation_hook.register(model)

    activations: Dict[int, list] = {L: [] for L in layers}
    all_logits: List[np.ndarray] = []

    for sample in samples:
        capture.clear()
        with torch.no_grad():
            inputs = torch.tensor([sample["token_ids"]], device=device)
            outputs = model(input_ids=inputs)

        pos = sample["anchor"]
        for L in layers:
            act = capture.post_act.get(L)
            if act is None:
                raise RuntimeError(f"No activation captured for layer {L}")
            pos_clamped = min(pos, act.shape[1] - 1)
            activations[L].append(act[0, pos_clamped, :].float().numpy())

        # Get logits at anchor position
        logits = outputs.logits[0, min(pos, outputs.logits.shape[1] - 1), :]
        all_logits.append(logits.float().cpu().numpy())

    capture.remove()
    if ablation_hook is not None:
        ablation_hook.remove()

    return (
        {L: np.stack(vecs) for L, vecs in activations.items()},
        all_logits,
    )


# ─────────────────────────────────────────────────────────────────────────────
# KL divergence between logit distributions
# ─────────────────────────────────────────────────────────────────────────────

def compute_output_kl(
    clean_logits: List[np.ndarray],
    ablated_logits: List[np.ndarray],
) -> float:
    """Mean KL(clean || ablated) across samples using softmax probabilities."""
    kls = []
    for clean, ablated in zip(clean_logits, ablated_logits):
        p = torch.softmax(torch.tensor(clean), dim=-1)
        q = torch.softmax(torch.tensor(ablated), dim=-1)
        kl = F.kl_div(q.log(), p, reduction="sum").item()
        kls.append(kl)
    return float(np.mean(kls))


# ─────────────────────────────────────────────────────────────────────────────
# Main experiment
# ─────────────────────────────────────────────────────────────────────────────

def run_ablation_experiment(
    model,
    samples: List[Dict],
    neuron_df: pd.DataFrame,
    layers: List[int],
    ablate_layers: List[int],
    ablate_pcts: List[float],
    device: str,
    n_random_seeds: int = 5,
) -> List[Dict[str, Any]]:
    """Run ablation experiment across strategies and percentages."""

    # 1. Clean baseline
    print("  Running clean baseline...")
    clean_acts, clean_logits = run_forward(model, samples, layers, device)

    clean_jsds = {}
    for L in layers:
        total_jsd, _ = compute_group_jsd(clean_acts[L], samples)
        pair_jsd = compute_pairwise_jsd(clean_acts[L], samples)
        clean_jsds[L] = {
            "group_jsd": total_jsd,
            "mean_pair_jsd": float(pair_jsd["jsd"].mean()),
        }

    results = []

    strategies = ["affinity", "affinity_high_freq", "affinity_low_freq", "coverage", "low_coverage", "jsd", "mass", "random"]

    for pct in ablate_pcts:
        for strategy in strategies:
            n_seeds = n_random_seeds if strategy == "random" else 1

            for seed in range(n_seeds):
                # Select neurons to ablate in specified layers
                neurons_to_ablate = {}
                total_ablated = 0
                for L in ablate_layers:
                    selected = select_neurons(neuron_df, L, pct, strategy, seed=seed)
                    neurons_to_ablate[L] = selected
                    total_ablated += len(selected)

                label = f"{strategy}_pct{pct}"
                if strategy == "random":
                    label += f"_seed{seed}"
                print(f"  Ablating {label}: {total_ablated} neurons across {len(ablate_layers)} layers")

                # Run with ablation
                hook = AblationHook(neurons_to_ablate)
                abl_acts, abl_logits = run_forward(
                    model, samples, layers, device, ablation_hook=hook,
                )

                # Measure impact
                output_kl = compute_output_kl(clean_logits, abl_logits)

                for L in layers:
                    total_jsd, _ = compute_group_jsd(abl_acts[L], samples)
                    pair_jsd = compute_pairwise_jsd(abl_acts[L], samples)

                    clean_gj = clean_jsds[L]["group_jsd"]
                    clean_pj = clean_jsds[L]["mean_pair_jsd"]
                    abl_gj = total_jsd
                    abl_pj = float(pair_jsd["jsd"].mean())

                    results.append({
                        "strategy": strategy,
                        "pct": pct,
                        "seed": seed if strategy == "random" else -1,
                        "layer": L,
                        "ablate_layers": ",".join(map(str, ablate_layers)),
                        "n_ablated": total_ablated,
                        "clean_group_jsd": clean_gj,
                        "ablated_group_jsd": abl_gj,
                        "jsd_change": abl_gj - clean_gj,
                        "jsd_change_pct": (abl_gj - clean_gj) / (clean_gj + EPS) * 100,
                        "clean_mean_pair_jsd": clean_pj,
                        "ablated_mean_pair_jsd": abl_pj,
                        "pair_jsd_change_pct": (abl_pj - clean_pj) / (clean_pj + EPS) * 100,
                        "output_kl": output_kl,
                    })

    return results


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

DATASET_NAMES = ["emotion", "medical", "legal", "scientific", "verb"]


def _find_dataset(dataset_dir: Path, name: str) -> Optional[Path]:
    for pat in [
        "{name}_ngrams_dedup_filtered.jsonl",
        "{name}_ngrams_filtered_pairs.jsonl",
    ]:
        p = dataset_dir / pat.format(name=name)
        if p.exists():
            return p
    return None


def main():
    p = argparse.ArgumentParser(description="Targeted ablation of frequency-specialized neurons")

    p.add_argument("--model", default="EleutherAI/pythia-70m-deduped")
    p.add_argument("--revision", default="step143000")
    p.add_argument("--device", default=None)

    # Single dataset mode
    p.add_argument("--dataset", default=None)
    p.add_argument("--neuron-metrics", default=None, help="Path to neuron_metrics.csv")

    # All datasets mode
    p.add_argument("--all-datasets", action="store_true")
    p.add_argument("--dataset-dir", default=None)
    p.add_argument("--metrics-dir", default=None, help="Dir with {dataset}/neuron_metrics.csv")

    p.add_argument("--output-dir", required=True)
    p.add_argument("--ablate-layers", default=None,
                   help="Comma-separated layers to ablate in (default: last third of model)")
    p.add_argument("--ablate-pct", default="1,2,5,10",
                   help="Comma-separated percentages of neurons to ablate")
    p.add_argument("--n-random-seeds", type=int, default=5)

    args = p.parse_args()

    device = args.device or (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    ablate_pcts = [float(x) for x in args.ablate_pct.split(",")]

    # Load model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    load_kw: Dict[str, Any] = {"torch_dtype": torch.float16}
    if args.revision != "main":
        load_kw["revision"] = args.revision

    print(f"Loading {args.model} @ {args.revision} on {device}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, **{k: v for k, v in load_kw.items() if k == "revision"})
    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kw).to(device)
    model.eval()

    layers = _infer_layers(args.model)
    n_layers = len(layers)

    if args.ablate_layers:
        ablate_layers = [int(x) for x in args.ablate_layers.split(",")]
    else:
        # Default: last third of model
        start = n_layers * 2 // 3
        ablate_layers = list(range(start, n_layers))

    print(f"Layers: {n_layers}, ablating in: {ablate_layers}")

    # Resolve datasets
    if args.all_datasets:
        dataset_dir = Path(args.dataset_dir)
        metrics_dir = Path(args.metrics_dir)
        datasets = []
        for name in DATASET_NAMES:
            ds_path = _find_dataset(dataset_dir, name)
            met_path = metrics_dir / name / "neuron_metrics.csv"
            if ds_path and met_path.exists():
                datasets.append((name, str(ds_path), str(met_path)))
            else:
                print(f"SKIP: {name} (dataset={'found' if ds_path else 'missing'}, metrics={'found' if met_path.exists() else 'missing'})")
    elif args.dataset and args.neuron_metrics:
        name = Path(args.dataset).stem.split("_")[0]
        datasets = [(name, args.dataset, args.neuron_metrics)]
    else:
        p.error("Provide --dataset + --neuron-metrics, or --all-datasets + --dataset-dir + --metrics-dir")

    # Run per dataset
    all_results = []

    for name, dataset_path, metrics_path in datasets:
        print(f"\n{'='*60}")
        print(f"  Dataset: {name}")
        print(f"{'='*60}")

        pairs = load_synonym_pairs(dataset_path)
        samples = pairs_to_samples(pairs, tokenizer)

        # Load neuron metrics (filter to the revision we're using)
        neuron_df = pd.read_csv(metrics_path)
        if "checkpoint" in neuron_df.columns and args.revision in neuron_df["checkpoint"].values:
            neuron_df = neuron_df[neuron_df["checkpoint"] == args.revision]
        print(f"  Neuron metrics: {len(neuron_df)} rows, layers {sorted(neuron_df['layer'].unique())}")

        results = run_ablation_experiment(
            model=model,
            samples=samples,
            neuron_df=neuron_df,
            layers=layers,
            ablate_layers=ablate_layers,
            ablate_pcts=ablate_pcts,
            device=device,
            n_random_seeds=args.n_random_seeds,
        )

        for r in results:
            r["dataset"] = name

        all_results.extend(results)

        # Print summary for this dataset
        df = pd.DataFrame(results)
        summary = df.groupby(["strategy", "pct"]).agg({
            "jsd_change_pct": "mean",
            "output_kl": "mean",
        }).reset_index()
        print(f"\n  Summary (mean across layers):")
        print(f"  {'Strategy':<12} {'%abl':>5} {'JSD change%':>12} {'Output KL':>10}")
        print(f"  {'-'*42}")
        for _, row in summary.iterrows():
            print(f"  {row['strategy']:<12} {row['pct']:>5.0f} {row['jsd_change_pct']:>+12.2f} {row['output_kl']:>10.4f}")

    # Save
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df_all = pd.DataFrame(all_results)
    df_all.to_csv(out / "ablation_results.csv", index=False)
    print(f"\nSaved {len(df_all)} rows to {out / 'ablation_results.csv'}")

    # Global summary
    if len(df_all) > 0:
        print(f"\n{'='*60}")
        print(f"  GLOBAL SUMMARY (all datasets)")
        print(f"{'='*60}")
        summary = df_all.groupby(["strategy", "pct"]).agg({
            "jsd_change_pct": ["mean", "std"],
            "output_kl": ["mean", "std"],
        }).reset_index()
        summary.columns = ["strategy", "pct", "jsd_mean", "jsd_std", "kl_mean", "kl_std"]
        print(f"  {'Strategy':<12} {'%abl':>5} {'JSD chg%':>10} {'(std)':>8} {'KL':>8} {'(std)':>8}")
        print(f"  {'-'*56}")
        for _, row in summary.iterrows():
            print(f"  {row['strategy']:<12} {row['pct']:>5.0f} {row['jsd_mean']:>+10.2f} {row['jsd_std']:>8.2f} {row['kl_mean']:>8.4f} {row['kl_std']:>8.4f}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
