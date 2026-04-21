#!/usr/bin/env python3
"""
Frequency circuit tracing: decompose how each transformer component
(attention, MLP, per-neuron) reads and writes frequency information
in the residual stream.


For each layer, we capture:
  1. Attention output contribution to residual stream (Δ_attn)
  2. MLP output contribution to residual stream (Δ_mlp)
  3. Per-neuron MLP contribution: activation_h * W_down[h, :]

We project each onto the local frequency direction and compute:
  - Signed projection (does this component write/erase frequency?)
  - Differential projection (high-freq vs low-freq inputs separately)
  - Cross-reference with neuron axes (coverage, affinity, mass)

Outputs:
  component_freq_flow.csv   — per (layer, component): freq projection for high/low/diff
  neuron_freq_projections.csv — per (layer, neuron): freq projection + axes
  freq_directions.csv       — per layer: probe weight vector stats, probe AUROC
  summary.json              — aggregate statistics

Usage:
    python -m scripts.interventions.frequency_circuit_tracing \
        --model EleutherAI/pythia-70m-deduped \
        --revision step143000 \
        --all-datasets \
        --dataset-dir ~/scaleJSD/dataset/legacy/filtered \
        --metrics-dir results/coverage_affinity/pythia-70m \
        --output-dir results/circuit_v2/pythia-70m \
        --device cuda
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
import torch

try:
    from scripts.metrics.coverage_affinity_experiment import (
        _infer_layers,
        load_synonym_pairs,
        pairs_to_samples,
    )
except ImportError:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from scripts.metrics.coverage_affinity_experiment import (
        _infer_layers,
        load_synonym_pairs,
        pairs_to_samples,
    )


class ComponentCapture:
    """Capture attention and MLP contributions to the residual stream separately.

    For each transformer layer, captures:
      - attn_out: output of the attention sublayer (before adding to residual)
      - mlp_out: output of the MLP sublayer (before adding to residual)
      - residual_pre: residual stream BEFORE this layer
      - residual_post: residual stream AFTER this layer
    """

    def __init__(self, model, layers: List[int], token_indices: List[int]):
        self.layers = layers
        self.token_indices = token_indices  # which token positions to capture
        self.hooks = []

        # Storage
        self.attn_out: Dict[int, List[np.ndarray]] = {l: [] for l in layers}
        self.mlp_out: Dict[int, List[np.ndarray]] = {l: [] for l in layers}
        self.residual_post: Dict[int, List[np.ndarray]] = {l: [] for l in layers}

        # Also capture per-neuron MLP activations (post-activation, pre-down_proj)
        self.mlp_post_act: Dict[int, List[np.ndarray]] = {l: [] for l in layers}

        self._register_hooks(model)

    def _register_hooks(self, model):
        """Register hooks on attention output, MLP output, and MLP intermediate."""
        model_name = model.config.model_type if hasattr(model.config, "model_type") else ""

        # Find transformer layers
        if hasattr(model, "gpt_neox"):
            tf_layers = model.gpt_neox.layers
        elif hasattr(model, "model") and hasattr(model.model, "layers"):
            tf_layers = model.model.layers
        elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            tf_layers = model.transformer.h
        elif (
            hasattr(model, "model")
            and hasattr(model.model, "transformer")
            and hasattr(model.model.transformer, "blocks")
        ):
            tf_layers = model.model.transformer.blocks
        else:
            raise ValueError(f"Cannot find transformer layers for {type(model)}")

        for layer_idx in self.layers:
            layer = tf_layers[layer_idx]

            # Hook on attention output
            attn_module = None
            for name in ["attention", "self_attn", "attn"]:
                if hasattr(layer, name):
                    attn_module = getattr(layer, name)
                    break
            if attn_module is not None:
                h = attn_module.register_forward_hook(
                    self._make_capture_hook(self.attn_out, layer_idx)
                )
                self.hooks.append(h)

            # Hook on MLP output
            mlp_module = None
            for name in ["mlp", "feed_forward", "ff"]:
                if hasattr(layer, name):
                    mlp_module = getattr(layer, name)
                    break
            if mlp_module is not None:
                h = mlp_module.register_forward_hook(
                    self._make_capture_hook(self.mlp_out, layer_idx)
                )
                self.hooks.append(h)

            # Hook on full layer output (residual stream after this layer)
            h = layer.register_forward_hook(self._make_capture_hook(self.residual_post, layer_idx))
            self.hooks.append(h)

            # Hook on MLP intermediate (post-activation, pre-down_proj)
            down_proj = None
            if mlp_module is not None:
                for name in ["dense_4h_to_h", "down_proj", "c_proj", "fc2", "w2"]:
                    if hasattr(mlp_module, name):
                        down_proj = getattr(mlp_module, name)
                        break
            if down_proj is not None:
                h = down_proj.register_forward_pre_hook(
                    self._make_pre_hook(self.mlp_post_act, layer_idx)
                )
                self.hooks.append(h)

    def _make_capture_hook(self, storage, layer_idx):
        def hook(module, input, output):
            # Handle tuple outputs (common for attention)
            if isinstance(output, tuple):
                out = output[0]
            else:
                out = output
            # Extract at target token positions
            for ti in self.token_indices:
                vec = out[:, ti, :].detach().cpu().float().numpy()
                storage[layer_idx].append(vec)

        return hook

    def _make_pre_hook(self, storage, layer_idx):
        def hook(module, input):
            inp = input[0] if isinstance(input, tuple) else input
            for ti in self.token_indices:
                vec = inp[:, ti, :].detach().cpu().float().numpy()
                storage[layer_idx].append(vec)

        return hook

    def clear(self):
        for l in self.layers:
            self.attn_out[l].clear()
            self.mlp_out[l].clear()
            self.residual_post[l].clear()
            self.mlp_post_act[l].clear()

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


# ── Frequency direction computation ─────────────────────────────────


def train_frequency_direction(
    X: np.ndarray,
    freq_labels: np.ndarray,
    pair_ids: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """Train a logistic regression probe and return its weight vector as
    the frequency direction, plus the probe's AUROC.

    The weight vector of a logistic regression trained to classify high vs low
    frequency IS the direction in activation space that maximally separates them
    (in the linear sense). This is the per-layer frequency direction.
    """
    # Group-aware split
    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    train_idx, test_idx = next(gss.split(X, freq_labels, groups=pair_ids))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = freq_labels[train_idx], freq_labels[test_idx]

    clf = LogisticRegression(max_iter=1000, solver="lbfgs", C=1.0)
    clf.fit(X_train, y_train)

    # AUROC
    if len(np.unique(y_test)) > 1:
        probs = clf.predict_proba(X_test)[:, 1]
        auroc = roc_auc_score(y_test, probs)
    else:
        auroc = 0.5

    # The weight vector IS the frequency direction
    # Shape: (hidden_dim,)
    freq_direction = clf.coef_[0]
    freq_direction = freq_direction / (np.linalg.norm(freq_direction) + 1e-12)

    return freq_direction, auroc


# ── Per-neuron MLP decomposition ────────────────────────────────────


def decompose_mlp_by_neuron(
    post_act: np.ndarray,  # [N_samples, 4H] — post-activation values
    down_proj_weight: np.ndarray,  # [H, 4H] — down_proj weight matrix
    freq_direction: np.ndarray,  # [H] — local frequency direction
    freq_labels: np.ndarray,  # [N_samples] — 0/1 high/low freq
) -> pd.DataFrame:
    """Decompose MLP output into per-neuron contributions projected onto
    the frequency direction.

    Each neuron h contributes: activation_h * W_down[:, h] to the residual stream.
    We project this onto freq_direction to get a signed frequency contribution.
    """
    N, four_H = post_act.shape
    H = down_proj_weight.shape[0]

    # Each neuron h's contribution to residual = activation_h * W_down[:, h]
    # Projection onto freq_dir = activation_h * <W_down[:, h], freq_dir>
    # = activation_h * (W_down.T @ freq_dir)[h]
    # down_proj is [H, 4H], so .T is [4H, H], and .T @ freq_dir is [4H]
    neuron_freq_alignment = down_proj_weight.T @ freq_direction  # [4H]

    # Per-neuron, per-sample frequency projection
    # proj[n, h] = post_act[n, h] * neuron_freq_alignment[h]
    per_sample_proj = post_act * neuron_freq_alignment[None, :]  # [N, 4H]

    # Aggregate by frequency group
    high_mask = freq_labels == 1
    low_mask = freq_labels == 0

    mean_proj_high = (
        per_sample_proj[high_mask].mean(axis=0) if high_mask.sum() > 0 else np.zeros(four_H)
    )
    mean_proj_low = (
        per_sample_proj[low_mask].mean(axis=0) if low_mask.sum() > 0 else np.zeros(four_H)
    )
    mean_proj_all = per_sample_proj.mean(axis=0)

    # The differential projection is what matters for frequency ROUTING:
    # positive = this neuron writes more frequency info for high-freq tokens
    # negative = this neuron writes more frequency info for low-freq tokens
    diff_proj = mean_proj_high - mean_proj_low

    # Also compute mean activation per group (for cross-referencing with axes)
    mean_act_high = post_act[high_mask].mean(axis=0) if high_mask.sum() > 0 else np.zeros(four_H)
    mean_act_low = post_act[low_mask].mean(axis=0) if low_mask.sum() > 0 else np.zeros(four_H)

    records = []
    for h in range(four_H):
        records.append(
            {
                "neuron_idx": h,
                "freq_alignment": float(neuron_freq_alignment[h]),
                "mean_proj_high": float(mean_proj_high[h]),
                "mean_proj_low": float(mean_proj_low[h]),
                "mean_proj_all": float(mean_proj_all[h]),
                "diff_proj": float(diff_proj[h]),
                "abs_diff_proj": float(abs(diff_proj[h])),
                "mean_act_high": float(mean_act_high[h]),
                "mean_act_low": float(mean_act_low[h]),
            }
        )

    return pd.DataFrame(records)


# ── Component-level frequency flow ──────────────────────────────────


def compute_component_freq_flow(
    attn_out: np.ndarray,  # [N, H]
    mlp_out: np.ndarray,  # [N, H]
    freq_direction: np.ndarray,  # [H]
    freq_labels: np.ndarray,  # [N]
) -> Dict[str, float]:
    """Project attention and MLP outputs onto frequency direction,
    separately for high-freq and low-freq inputs."""

    high_mask = freq_labels == 1
    low_mask = freq_labels == 0

    results = {}
    for name, component in [("attn", attn_out), ("mlp", mlp_out)]:
        proj = component @ freq_direction  # [N]

        results[f"{name}_proj_mean"] = float(proj.mean())
        results[f"{name}_proj_high"] = (
            float(proj[high_mask].mean()) if high_mask.sum() > 0 else 0.0
        )
        results[f"{name}_proj_low"] = float(proj[low_mask].mean()) if low_mask.sum() > 0 else 0.0
        results[f"{name}_proj_diff"] = results[f"{name}_proj_high"] - results[f"{name}_proj_low"]
        results[f"{name}_proj_abs_diff"] = abs(results[f"{name}_proj_diff"])
        results[f"{name}_proj_std"] = float(proj.std())

    # Combined: total contribution
    total = attn_out + mlp_out
    proj_total = total @ freq_direction
    results["total_proj_diff"] = (
        float(proj_total[high_mask].mean() - proj_total[low_mask].mean())
        if (high_mask.sum() > 0 and low_mask.sum() > 0)
        else 0.0
    )

    # Fraction of frequency info from MLP vs attention
    attn_diff = abs(results["attn_proj_diff"])
    mlp_diff = abs(results["mlp_proj_diff"])
    total_diff = attn_diff + mlp_diff + 1e-12
    results["mlp_freq_fraction"] = float(mlp_diff / total_diff)
    results["attn_freq_fraction"] = float(attn_diff / total_diff)

    return results


# ── Main experiment ──────────────────────────────────────────────────


def run_circuit_tracing(
    model_id: str,
    revision: str,
    dataset_path: str,
    metrics_dir: Optional[str],
    output_dir: str,
    device: str = "cuda",
):
    """Run frequency circuit tracing for one model + one dataset."""

    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.makedirs(output_dir, exist_ok=True)
    dataset_name = Path(dataset_path).stem.replace("_ngrams_dedup_filtered", "")

    print(f"  Loading model {model_id} (revision={revision})...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        torch_dtype=torch.float32,
        device_map=device,
    )
    model.eval()

    all_layers = _infer_layers(model_id)
    n_layers = len(all_layers)

    # Load data
    print(f"  Loading dataset {dataset_path}...")
    pairs = load_synonym_pairs(dataset_path)
    samples = pairs_to_samples(pairs, tokenizer)

    if not samples:
        print(f"  WARNING: No valid samples for {dataset_path}, skipping")
        return

    # Prepare inputs
    token_ids_list = [s["token_ids"] for s in samples]
    anchor_positions = [s["anchor"] for s in samples]
    freq_labels = np.array([1 if s["frequency_category"] == "high_freq" else 0 for s in samples])
    pair_ids = np.array([s["pair_id"] for s in samples])

    # Set up hooks
    capture = ComponentCapture(model, all_layers, token_indices=[])
    # We'll manually handle token positions since they vary per sample
    capture.remove_hooks()  # Remove — we'll do manual forward passes

    print(f"  Running forward passes ({len(samples)} samples)...")

    # Storage for per-layer activations
    attn_outs = {l: [] for l in all_layers}
    mlp_outs = {l: [] for l in all_layers}
    residuals = {l: [] for l in all_layers}
    mlp_post_acts = {l: [] for l in all_layers}

    # We need to capture per-sample because anchor positions differ
    for i, sample in enumerate(samples):
        input_ids = torch.tensor([sample["token_ids"]], device=device)
        anchor_pos = sample["anchor"]

        # Register hooks for this forward pass
        cap = ComponentCapture(model, all_layers, token_indices=[anchor_pos])

        with torch.no_grad():
            _ = model(input_ids)

        for l in all_layers:
            if cap.attn_out[l]:
                attn_outs[l].append(cap.attn_out[l][0][0])  # [H]
            if cap.mlp_out[l]:
                mlp_outs[l].append(cap.mlp_out[l][0][0])  # [H]
            if cap.residual_post[l]:
                residuals[l].append(cap.residual_post[l][0][0])  # [H]
            if cap.mlp_post_act[l]:
                mlp_post_acts[l].append(cap.mlp_post_act[l][0][0])  # [4H]

        cap.remove_hooks()

        if (i + 1) % 50 == 0:
            print(f"    {i + 1}/{len(samples)} samples processed")

    # Stack into arrays
    for l in all_layers:
        attn_outs[l] = np.stack(attn_outs[l]) if attn_outs[l] else None
        mlp_outs[l] = np.stack(mlp_outs[l]) if mlp_outs[l] else None
        residuals[l] = np.stack(residuals[l]) if residuals[l] else None
        mlp_post_acts[l] = np.stack(mlp_post_acts[l]) if mlp_post_acts[l] else None

    # Get down_proj weights for per-neuron decomposition
    print("  Extracting down_proj weights...")
    down_proj_weights = {}
    if hasattr(model, "gpt_neox"):
        tf_layers = model.gpt_neox.layers
    elif hasattr(model, "model") and hasattr(model.model, "layers"):
        tf_layers = model.model.layers
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        tf_layers = model.transformer.h
    elif hasattr(model, "model") and hasattr(model.model, "transformer"):
        tf_layers = model.model.transformer.blocks
    else:
        tf_layers = None

    if tf_layers is not None:
        for l in all_layers:
            layer = tf_layers[l]
            mlp_module = None
            for name in ["mlp", "feed_forward", "ff"]:
                if hasattr(layer, name):
                    mlp_module = getattr(layer, name)
                    break
            if mlp_module is not None:
                for name in ["dense_4h_to_h", "down_proj", "c_proj", "fc2", "w2"]:
                    if hasattr(mlp_module, name):
                        W = getattr(mlp_module, name).weight.detach().cpu().float().numpy()
                        down_proj_weights[l] = W  # [H, 4H]
                        break

    # Load neuron metrics for cross-referencing (if available)
    neuron_metrics = None
    if metrics_dir:
        metrics_path = Path(metrics_dir) / dataset_name / "neuron_metrics.csv"
        if metrics_path.exists():
            print(f"  Loading neuron metrics from {metrics_path}...")
            neuron_metrics = pd.read_csv(metrics_path)

    # ── Per-layer analysis ──
    print("  Computing per-layer frequency directions and projections...")

    component_rows = []
    freq_dir_rows = []
    all_neuron_rows = []

    for l in all_layers:
        if residuals[l] is None:
            continue

        X = residuals[l]  # [N, H]

        # Train frequency probe → get local frequency direction
        freq_dir, probe_auroc = train_frequency_direction(X, freq_labels, pair_ids)

        freq_dir_rows.append(
            {
                "layer": l,
                "probe_auroc": probe_auroc,
                "freq_dir_norm": float(np.linalg.norm(freq_dir)),
                "freq_dir_mean": float(freq_dir.mean()),
                "freq_dir_std": float(freq_dir.std()),
            }
        )

        # Component-level frequency flow
        if attn_outs[l] is not None and mlp_outs[l] is not None:
            flow = compute_component_freq_flow(attn_outs[l], mlp_outs[l], freq_dir, freq_labels)
            flow["layer"] = l
            flow["probe_auroc"] = probe_auroc
            component_rows.append(flow)

        # Per-neuron MLP decomposition
        if mlp_post_acts[l] is not None and l in down_proj_weights:
            neuron_df = decompose_mlp_by_neuron(
                mlp_post_acts[l], down_proj_weights[l], freq_dir, freq_labels
            )
            neuron_df["layer"] = l

            # Cross-reference with axes if available
            if neuron_metrics is not None:
                layer_metrics = neuron_metrics[neuron_metrics["layer"] == l].copy()
                if len(layer_metrics) > 0 and "neuron_idx" in layer_metrics.columns:
                    neuron_df = neuron_df.merge(
                        layer_metrics[
                            [
                                "neuron_idx",
                                "binary_coverage",
                                "raw_mass_total",
                                "frequency_affinity",
                                "jsd_contrib",
                            ]
                        ].rename(
                            columns={
                                "binary_coverage": "coverage",
                                "raw_mass_total": "mass",
                                "frequency_affinity": "affinity",
                            }
                        ),
                        on="neuron_idx",
                        how="left",
                    )

            all_neuron_rows.append(neuron_df)

        print(
            f"    Layer {l}: probe AUROC={probe_auroc:.3f}, "
            f"MLP freq fraction={flow.get('mlp_freq_fraction', 0):.3f}"
            if flow
            else f"    Layer {l}: probe AUROC={probe_auroc:.3f}"
        )

    # ── Save results ──
    print(f"  Saving results to {output_dir}...")

    # Component flow
    if component_rows:
        comp_df = pd.DataFrame(component_rows)
        comp_df["dataset"] = dataset_name
        comp_path = os.path.join(output_dir, f"{dataset_name}_component_freq_flow.csv")
        comp_df.to_csv(comp_path, index=False)
        print(f"    Saved {comp_path} ({len(comp_df)} rows)")

    # Frequency directions
    if freq_dir_rows:
        fd_df = pd.DataFrame(freq_dir_rows)
        fd_df["dataset"] = dataset_name
        fd_path = os.path.join(output_dir, f"{dataset_name}_freq_directions.csv")
        fd_df.to_csv(fd_path, index=False)

    # Per-neuron projections
    if all_neuron_rows:
        neuron_full = pd.concat(all_neuron_rows, ignore_index=True)
        neuron_full["dataset"] = dataset_name
        neuron_path = os.path.join(output_dir, f"{dataset_name}_neuron_freq_projections.csv")
        neuron_full.to_csv(neuron_path, index=False)
        print(f"    Saved {neuron_path} ({len(neuron_full)} rows)")

    # Summary
    summary = {
        "model": model_id,
        "revision": revision,
        "dataset": dataset_name,
        "n_samples": len(samples),
        "n_layers": n_layers,
        "component_flow": component_rows,
        "freq_directions": freq_dir_rows,
    }
    summary_path = os.path.join(output_dir, f"{dataset_name}_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Cleanup
    del model
    torch.cuda.empty_cache()

    return summary


# ── CLI ──────────────────────────────────────────────────────────────

DATASET_FILES = {
    "emotion": "emotion_ngrams_dedup_filtered.jsonl",
    "medical": "medical_ngrams_dedup_filtered.jsonl",
    "legal": "legal_ngrams_dedup_filtered.jsonl",
    "scientific": "scientific_ngrams_dedup_filtered.jsonl",
    "verb": "verb_ngrams_dedup_filtered.jsonl",
}


def main():
    parser = argparse.ArgumentParser(description="Frequency circuit tracing")
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--dataset", type=str, default=None, help="Path to specific dataset JSONL")
    parser.add_argument(
        "--all-datasets", action="store_true", help="Run on all 5 ScaleJSD datasets"
    )
    parser.add_argument(
        "--dataset-dir", type=str, default=None, help="Directory containing dataset JSONLs"
    )
    parser.add_argument(
        "--metrics-dir",
        type=str,
        default=None,
        help="Directory containing neuron_metrics.csv files",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.all_datasets:
        assert args.dataset_dir, "--dataset-dir required with --all-datasets"
        datasets = []
        for name, fname in DATASET_FILES.items():
            path = os.path.join(args.dataset_dir, fname)
            if os.path.exists(path):
                datasets.append(path)
            else:
                print(f"  WARNING: {path} not found, skipping {name}")
    elif args.dataset:
        datasets = [args.dataset]
    else:
        parser.error("Provide --dataset or --all-datasets")

    print(f"Frequency circuit tracing: {args.model}")
    print(f"  Datasets: {len(datasets)}")
    print(f"  Device: {args.device}")
    print()

    for ds_path in datasets:
        ds_name = Path(ds_path).stem.replace("_ngrams_dedup_filtered", "")
        print(f"{'=' * 60}")
        print(f"  Dataset: {ds_name}")
        print(f"{'=' * 60}")

        metrics_subdir = None
        if args.metrics_dir:
            candidate = os.path.join(args.metrics_dir, ds_name)
            if os.path.isdir(candidate):
                metrics_subdir = args.metrics_dir

        run_circuit_tracing(
            model_id=args.model,
            revision=args.revision,
            dataset_path=ds_path,
            metrics_dir=metrics_subdir,
            output_dir=args.output_dir,
            device=args.device,
        )
        print()

    print("Done!")


if __name__ == "__main__":
    main()
