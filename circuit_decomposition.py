#!/usr/bin/env python3
"""
Circuit decomposition: trace frequency information flow through the transformer.

For each layer, decompose the residual stream update into attention and MLP
contributions, then project each onto the "frequency direction" — the direction
in residual stream space that separates high-freq from low-freq synonym
representations.

This answers: which components WRITE frequency info (positive projection) and
which ERASE it (negative projection)?

Within each MLP, further decompose by neuron: each neuron contributes
activation_h * W_down[h, :] to the residual stream. Project each neuron's
contribution onto the frequency direction.

Outputs:
  component_projections.csv — per (layer, component): signed projection onto freq direction
  neuron_freq_projections.csv — per (layer, neuron): signed projection, plus existing axes
  summary.json — per-layer summary statistics

Usage:
    python circuit_decomposition.py \
        --model EleutherAI/pythia-70m-deduped \
        --revision step143000 \
        --dataset ~/dev/personal/mechinterp/scaleJSD/dataset/legacy/filtered/emotion_ngrams_dedup_filtered.jsonl \
        --output-dir results/circuit/pythia-70m/emotion/

    python circuit_decomposition.py \
        --model EleutherAI/pythia-70m-deduped \
        --revision step143000 \
        --all-datasets --dataset-dir ~/dev/personal/mechinterp/scaleJSD/dataset/legacy/filtered \
        --output-dir results/circuit/pythia-70m/
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

import sys, importlib.util

# Dynamic import: handle both flat and reorganized repo layouts
_cae_candidates = [
    "coverage_affinity_experiment",
    "scripts.metrics.coverage_affinity_experiment",
]
_cae = None
for _name in _cae_candidates:
    try:
        _cae = importlib.import_module(_name)
        break
    except ModuleNotFoundError:
        continue

# If module import fails, try direct file path
if _cae is None:
    for _path in [
        Path(__file__).parent / "scripts" / "metrics" / "coverage_affinity_experiment.py",
        Path(__file__).parent / "coverage_affinity_experiment.py",
    ]:
        if _path.exists():
            spec = importlib.util.spec_from_file_location("coverage_affinity_experiment", _path)
            _cae = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(_cae)
            break

if _cae is None:
    raise ImportError("Cannot find coverage_affinity_experiment.py in any expected location")

load_synonym_pairs = _cae.load_synonym_pairs
pairs_to_samples = _cae.pairs_to_samples
_infer_layers = _cae._infer_layers


# ─────────────────────────────────────────────────────────────────────────────
# Architecture detection (reuse patterns from existing scripts)
# ─────────────────────────────────────────────────────────────────────────────

LAYER_PATHS = ["gpt_neox.layers", "model.layers", "transformer.h",
               "transformer.layers", "model.decoder.layers"]
MLP_NAMES = ["mlp", "feed_forward", "ff"]
DOWN_NAMES = ["dense_4h_to_h", "down_proj", "c_proj", "fc2", "w2"]


def _resolve_attr(obj, dotted):
    for part in dotted.split("."):
        obj = getattr(obj, part)
    return obj


def find_model_layers(model):
    for path in LAYER_PATHS:
        try:
            return _resolve_attr(model, path)
        except AttributeError:
            continue
    raise RuntimeError("Cannot find transformer layers")


def find_mlp_and_down(layer_mod):
    for mn in MLP_NAMES:
        mlp = getattr(layer_mod, mn, None)
        if mlp is None:
            continue
        for dn in DOWN_NAMES:
            down = getattr(mlp, dn, None)
            if down is not None:
                return mlp, down
    return None, None


# ─────────────────────────────────────────────────────────────────────────────
# Hook-based capture of attention output, MLP output, and layer output
# ─────────────────────────────────────────────────────────────────────────────

class CircuitCapture:
    """Capture attention output, MLP input, MLP post-activation, and layer output.

    For the residual stream decomposition:
      h_{L+1} = h_L + attn_L(h_L) + mlp_L(h_L)

    We need:
      - layer_input[L]: h_L (residual stream entering the layer)
      - attn_output[L]: the attention block's contribution
      - mlp_output[L]: the MLP block's contribution
      - post_act[L]: the 4H-dim post-activation (for per-neuron decomposition)
      - layer_output[L]: h_{L+1} (residual stream leaving the layer)

    Architecture note: In GPT-NeoX (Pythia), attention and MLP run in PARALLEL
    on the same input. In LLaMA/OLMo, they run SEQUENTIALLY (attn first, then
    MLP on attn output + residual). The decomposition h = h_prev + attn + mlp
    holds in both cases, but the "input" to MLP differs.
    """

    def __init__(self, model, layers):
        self.model = model
        self.layers = layers
        self.hooks = []

        # Captured tensors per layer
        self.layer_input: Dict[int, torch.Tensor] = {}
        self.attn_output: Dict[int, torch.Tensor] = {}
        self.mlp_output: Dict[int, torch.Tensor] = {}
        self.post_act: Dict[int, torch.Tensor] = {}  # 4H-dim for per-neuron decomp
        self.layer_output: Dict[int, torch.Tensor] = {}

    def register(self):
        model_layers = find_model_layers(self.model)

        for L in self.layers:
            layer_mod = model_layers[L]

            # Hook the full layer to get input and output
            def make_layer_hook(layer_idx):
                def hook_fn(_mod, inp, out):
                    x_in = inp[0] if isinstance(inp, tuple) else inp
                    x_out = out[0] if isinstance(out, tuple) else out
                    self.layer_input[layer_idx] = x_in.detach().cpu()
                    self.layer_output[layer_idx] = x_out.detach().cpu()
                return hook_fn
            self.hooks.append(layer_mod.register_forward_hook(make_layer_hook(L)))

            # Hook attention block to get its output
            attn_mod = None
            for name in ["attention", "self_attn", "attn", "self_attention"]:
                attn_mod = getattr(layer_mod, name, None)
                if attn_mod is not None:
                    break

            if attn_mod is not None:
                def make_attn_hook(layer_idx):
                    def hook_fn(_mod, _inp, out):
                        x = out[0] if isinstance(out, tuple) else out
                        self.attn_output[layer_idx] = x.detach().cpu()
                    return hook_fn
                self.hooks.append(attn_mod.register_forward_hook(make_attn_hook(L)))

            # Hook MLP block to get its output
            mlp_mod, down_mod = find_mlp_and_down(layer_mod)
            if mlp_mod is not None:
                def make_mlp_hook(layer_idx):
                    def hook_fn(_mod, _inp, out):
                        x = out[0] if isinstance(out, tuple) else out
                        self.mlp_output[layer_idx] = x.detach().cpu()
                    return hook_fn
                self.hooks.append(mlp_mod.register_forward_hook(make_mlp_hook(L)))

            # Hook down_proj to get 4H-dim post-activation (input to down_proj)
            if down_mod is not None:
                def make_postact_hook(layer_idx):
                    def hook_fn(_mod, inp, _out):
                        x = inp[0] if isinstance(inp, tuple) else inp
                        self.post_act[layer_idx] = x.detach().cpu()
                    return hook_fn
                self.hooks.append(down_mod.register_forward_hook(make_postact_hook(L)))

    def clear(self):
        self.layer_input.clear()
        self.attn_output.clear()
        self.mlp_output.clear()
        self.post_act.clear()
        self.layer_output.clear()

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


# ─────────────────────────────────────────────────────────────────────────────
# Frequency direction computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_frequency_direction(
    high_freq_vecs: np.ndarray,
    low_freq_vecs: np.ndarray,
) -> np.ndarray:
    """Compute unit frequency direction: normalize(mean(high) - mean(low))."""
    diff = high_freq_vecs.mean(axis=0) - low_freq_vecs.mean(axis=0)
    norm = np.linalg.norm(diff)
    if norm < 1e-10:
        return diff  # degenerate case
    return diff / norm


# ─────────────────────────────────────────────────────────────────────────────
# Main experiment
# ─────────────────────────────────────────────────────────────────────────────

def run_circuit_decomposition(
    model,
    samples: List[Dict],
    layers: List[int],
    device: str,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Run the circuit decomposition.

    Returns:
      component_rows: per (layer, sample_group) component projections
      neuron_rows: per (layer, neuron) frequency direction projections
      summary_rows: per-layer summary
    """
    capture = CircuitCapture(model, layers)
    capture.register()

    # Separate high and low freq sample indices
    high_indices = [i for i, s in enumerate(samples) if s["frequency_category"] == "high_freq"]
    low_indices = [i for i, s in enumerate(samples) if s["frequency_category"] == "low_freq"]

    # Storage: per-layer, per-sample residual stream vectors and component outputs
    layer_inputs = {L: [] for L in layers}     # [N, H]
    attn_outputs = {L: [] for L in layers}     # [N, H]
    mlp_outputs = {L: [] for L in layers}      # [N, H]
    post_acts = {L: [] for L in layers}        # [N, 4H]
    layer_outputs_all = {L: [] for L in layers}  # [N, H]

    # Run all samples
    for idx, sample in enumerate(samples):
        if idx % 50 == 0:
            print(f"  [{idx+1}/{len(samples)}] {sample['phrase']}")

        capture.clear()
        with torch.no_grad():
            inputs = torch.tensor([sample["token_ids"]], device=device)
            model(input_ids=inputs)

        pos = sample["anchor"]

        for L in layers:
            def get_vec(tensor_dict, layer, position):
                t = tensor_dict.get(layer)
                if t is None:
                    return None
                p = min(position, t.shape[1] - 1)
                return t[0, p, :].float().numpy()

            h_in = get_vec(capture.layer_input, L, pos)
            a_out = get_vec(capture.attn_output, L, pos)
            m_out = get_vec(capture.mlp_output, L, pos)
            h_out = get_vec(capture.layer_output, L, pos)

            if h_in is not None:
                layer_inputs[L].append(h_in)
            if a_out is not None:
                attn_outputs[L].append(a_out)
            if m_out is not None:
                mlp_outputs[L].append(m_out)
            if h_out is not None:
                layer_outputs_all[L].append(h_out)

            # 4H post-activation for neuron decomposition
            pa = capture.post_act.get(L)
            if pa is not None:
                p_clamped = min(pos, pa.shape[1] - 1)
                post_acts[L].append(pa[0, p_clamped, :].float().numpy())

    capture.remove()

    # Stack into arrays
    for L in layers:
        layer_inputs[L] = np.stack(layer_inputs[L]) if layer_inputs[L] else None
        attn_outputs[L] = np.stack(attn_outputs[L]) if attn_outputs[L] else None
        mlp_outputs[L] = np.stack(mlp_outputs[L]) if mlp_outputs[L] else None
        post_acts[L] = np.stack(post_acts[L]) if post_acts[L] else None
        layer_outputs_all[L] = np.stack(layer_outputs_all[L]) if layer_outputs_all[L] else None

    # ── Compute frequency direction (from residual stream at each layer) ──

    # Use a FIXED frequency direction from the embedding layer (layer 0 input)
    # AND per-layer directions for comparison
    h0 = layer_inputs[layers[0]]
    if h0 is not None:
        fixed_freq_dir = compute_frequency_direction(
            h0[high_indices], h0[low_indices]
        )
    else:
        fixed_freq_dir = None

    component_rows = []
    neuron_rows = []
    summary_rows = []

    for L in layers:
        h_in = layer_inputs[L]
        a_out = attn_outputs[L]
        m_out = mlp_outputs[L]
        h_out = layer_outputs_all[L]
        pa = post_acts[L]

        if h_in is None:
            continue

        # Per-layer frequency direction
        per_layer_freq_dir = compute_frequency_direction(
            h_in[high_indices], h_in[low_indices]
        )

        # Frequency content of residual stream (norm of projection)
        freq_content_in = np.linalg.norm(
            h_in[high_indices].mean(0) - h_in[low_indices].mean(0)
        )
        freq_content_out = np.linalg.norm(
            h_out[high_indices].mean(0) - h_out[low_indices].mean(0)
        ) if h_out is not None else float('nan')

        # ── Component projections (mean across samples) ──
        # For each component, compute mean projection onto fixed freq direction

        for freq_dir, dir_name in [(fixed_freq_dir, "fixed"), (per_layer_freq_dir, "per_layer")]:
            if freq_dir is None:
                continue

            # Mean component output across ALL samples, then project
            # But more informative: mean of (high) minus mean of (low), projected
            # This tells us: does this component push high and low APART (positive)
            # or TOGETHER (negative)?

            if a_out is not None:
                attn_diff = a_out[high_indices].mean(0) - a_out[low_indices].mean(0)
                attn_proj = float(np.dot(attn_diff, freq_dir))
            else:
                attn_proj = float('nan')

            if m_out is not None:
                mlp_diff = m_out[high_indices].mean(0) - m_out[low_indices].mean(0)
                mlp_proj = float(np.dot(mlp_diff, freq_dir))
            else:
                mlp_proj = float('nan')

            # Residual (input contribution = just carrying forward)
            residual_diff = h_in[high_indices].mean(0) - h_in[low_indices].mean(0)
            residual_proj = float(np.dot(residual_diff, freq_dir))

            # Total output
            if h_out is not None:
                output_diff = h_out[high_indices].mean(0) - h_out[low_indices].mean(0)
                output_proj = float(np.dot(output_diff, freq_dir))
            else:
                output_proj = float('nan')

            component_rows.append({
                "layer": L,
                "freq_dir_type": dir_name,
                "residual_proj": residual_proj,
                "attn_proj": attn_proj,
                "mlp_proj": mlp_proj,
                "output_proj": output_proj,
                "freq_content_in": freq_content_in,
                "freq_content_out": freq_content_out,
            })

        # ── Per-neuron decomposition within MLP ──
        # Each neuron h contributes: a_h * W_down[h, :] to residual stream
        # Project onto frequency direction: a_h * dot(W_down[h, :], freq_dir)

        if pa is not None and m_out is not None:
            model_layers = find_model_layers(model)
            _, down_mod = find_mlp_and_down(model_layers[L])
            if down_mod is not None:
                W_down = down_mod.weight.detach().float().cpu().numpy()  # [H, 4H]
                # W_down[h_dim, neuron] — each column is a neuron's contribution direction
                # dot(W_down[:, neuron], freq_dir) = how aligned this neuron is with freq
                neuron_freq_alignment = W_down.T @ fixed_freq_dir  # [4H]

                # Mean activation per neuron for high vs low
                pa_high = pa[high_indices]  # [N_high, 4H]
                pa_low = pa[low_indices]    # [N_low, 4H]
                mean_act_high = pa_high.mean(axis=0)
                mean_act_low = pa_low.mean(axis=0)
                mean_act_all = pa.mean(axis=0)

                # Per-neuron frequency contribution:
                # (mean_act_high - mean_act_low) * alignment
                # This is how much this neuron pushes high and low freq APART
                # along the frequency direction
                act_diff = mean_act_high - mean_act_low
                neuron_freq_contrib = act_diff * neuron_freq_alignment

                # Also compute affinity for cross-referencing
                mass_high = np.maximum(mean_act_high, 0)
                mass_low = np.maximum(mean_act_low, 0)
                affinity = mass_high / (mass_high + mass_low + 1e-12)
                coverage = (pa > 0).mean(axis=0)

                n_neurons = len(neuron_freq_alignment)
                for h in range(n_neurons):
                    neuron_rows.append({
                        "layer": L,
                        "neuron": h,
                        "freq_alignment": float(neuron_freq_alignment[h]),
                        "freq_contrib": float(neuron_freq_contrib[h]),
                        "mean_act_diff": float(act_diff[h]),
                        "mean_act_all": float(mean_act_all[h]),
                        "affinity": float(affinity[h]),
                        "coverage": float(coverage[h]),
                    })

        # ── Summary ──
        summary_rows.append({
            "layer": L,
            "freq_content_in": freq_content_in,
            "freq_content_out": freq_content_out,
            "freq_change": freq_content_out - freq_content_in,
            "freq_change_pct": (freq_content_out - freq_content_in) / (freq_content_in + 1e-12) * 100,
            "attn_proj_fixed": next((r["attn_proj"] for r in component_rows if r["layer"] == L and r["freq_dir_type"] == "fixed"), float('nan')),
            "mlp_proj_fixed": next((r["mlp_proj"] for r in component_rows if r["layer"] == L and r["freq_dir_type"] == "fixed"), float('nan')),
        })

    return component_rows, neuron_rows, summary_rows


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

DATASET_NAMES = ["emotion", "medical", "legal", "scientific", "verb"]


def _find_dataset(dataset_dir, name):
    for pat in ["{name}_ngrams_dedup_filtered.jsonl", "{name}_ngrams_filtered_pairs.jsonl"]:
        p = Path(dataset_dir) / pat.format(name=name)
        if p.exists():
            return p
    return None


def main():
    p = argparse.ArgumentParser(description="Circuit decomposition: frequency direction projections")
    p.add_argument("--model", default="EleutherAI/pythia-70m-deduped")
    p.add_argument("--revision", default="step143000")
    p.add_argument("--device", default=None)

    p.add_argument("--dataset", default=None)
    p.add_argument("--all-datasets", action="store_true")
    p.add_argument("--dataset-dir", default=None)

    p.add_argument("--output-dir", required=True)
    args = p.parse_args()

    device = args.device or (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    from transformers import AutoModelForCausalLM, AutoTokenizer

    load_kw: Dict[str, Any] = {"torch_dtype": torch.float16}
    if args.revision != "main":
        load_kw["revision"] = args.revision

    print(f"Loading {args.model} @ {args.revision} on {device}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, **{k: v for k, v in load_kw.items() if k == "revision"}
    )
    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kw).to(device)
    model.eval()

    layers = _infer_layers(args.model)

    # Resolve datasets
    if args.all_datasets:
        dataset_dir = Path(args.dataset_dir)
        datasets = []
        for name in DATASET_NAMES:
            ds = _find_dataset(dataset_dir, name)
            if ds:
                datasets.append((name, str(ds)))
        if not datasets:
            raise RuntimeError(f"No datasets found in {dataset_dir}")
    elif args.dataset:
        name = Path(args.dataset).stem.split("_")[0]
        datasets = [(name, args.dataset)]
    else:
        p.error("Provide --dataset or --all-datasets + --dataset-dir")

    all_component = []
    all_neuron = []
    all_summary = []

    for name, dataset_path in datasets:
        print(f"\n{'='*60}")
        print(f"  Dataset: {name}")
        print(f"{'='*60}")

        pairs = load_synonym_pairs(dataset_path)
        samples = pairs_to_samples(pairs, tokenizer)

        comp, neur, summ = run_circuit_decomposition(model, samples, layers, device)

        for r in comp:
            r["dataset"] = name
        for r in neur:
            r["dataset"] = name
        for r in summ:
            r["dataset"] = name

        all_component.extend(comp)
        all_neuron.extend(neur)
        all_summary.extend(summ)

        # Print summary for this dataset
        print(f"\n  {'Layer':>5} {'Freq In':>10} {'Freq Out':>10} {'Change%':>10} {'Attn Proj':>10} {'MLP Proj':>10}")
        print(f"  {'-'*57}")
        for r in summ:
            print(f"  {r['layer']:>5} {r['freq_content_in']:>10.4f} {r['freq_content_out']:>10.4f} "
                  f"{r['freq_change_pct']:>+10.2f} {r['attn_proj_fixed']:>10.4f} {r['mlp_proj_fixed']:>+10.4f}")

    # Save
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Component projections
    if all_component:
        with open(out / "component_projections.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=all_component[0].keys())
            w.writeheader()
            w.writerows(all_component)

    # Neuron projections
    if all_neuron:
        with open(out / "neuron_freq_projections.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=all_neuron[0].keys())
            w.writeheader()
            w.writerows(all_neuron)

    # Summary
    if all_summary:
        with open(out / "summary.json", "w") as f:
            json.dump(all_summary, f, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else str(x))

    print(f"\nSaved to {out}:")
    print(f"  component_projections.csv: {len(all_component)} rows")
    print(f"  neuron_freq_projections.csv: {len(all_neuron)} rows")
    print(f"  summary.json: {len(all_summary)} entries")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
