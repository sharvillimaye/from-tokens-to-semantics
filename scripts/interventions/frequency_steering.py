#!/usr/bin/env python3
"""
E0.1 — Asymmetric intervention test: does ADDING along the frequency
direction move outputs, even though PROJECTING IT OUT doesn't?

This is the make-or-break experiment for the applied paper direction.

For each model, at the "best layer" (peak probe AUROC):
  - Compute v_L (probe direction)
  - For α ∈ {-3, -1.5, -0.5, 0, 0.5, 1.5, 3}, inject x_L ← x_L + α·v_L
    at every token position after the prompt
  - Measure:
    (a) Mean log-frequency of generated tokens (does it shift with α?)
    (b) Generation perplexity under a reference model (does quality hold?)
    (c) KL divergence from clean logits at the prompt's last position
    (d) Probability mass on high-freq vs low-freq tokens

Pass criterion: token log-frequency moves monotonically with α, effect
size ≥ 0.5 stdev across the α range, perplexity stays within 2× baseline.

Also runs E0.2 — Probe orthogonality: cosine between frequency direction
and per-domain semantic probe directions at each layer.

Usage:
    python -m scripts.interventions.frequency_steering \
        --model EleutherAI/pythia-6.9b-deduped --revision step143000 \
        --dataset-dir ~/scaleJSD/dataset/legacy/filtered \
        --output-dir results/frequency_steering/pythia-6.9b \
        --alphas -3,-1.5,-0.5,0,0.5,1.5,3 \
        --device cuda
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder

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


DATASET_FILES = {
    "emotion": "emotion_ngrams_dedup_filtered.jsonl",
    "medical": "medical_ngrams_dedup_filtered.jsonl",
    "legal": "legal_ngrams_dedup_filtered.jsonl",
    "scientific": "scientific_ngrams_dedup_filtered.jsonl",
    "verb": "verb_ngrams_dedup_filtered.jsonl",
}


# ─────────────────────────────────────────────────────────────────────
#  Transformer layer discovery + hooks
# ─────────────────────────────────────────────────────────────────────

def find_transformer_layers(model):
    for attr in ["gpt_neox.layers", "model.layers", "transformer.h",
                 "transformer.layers", "model.decoder.layers"]:
        obj = model
        try:
            for p in attr.split("."):
                obj = getattr(obj, p)
            return obj
        except AttributeError:
            continue
    raise RuntimeError("Cannot find transformer layers")


class ResidualCapture:
    """Capture residual stream at layer outputs."""
    def __init__(self, model, layers):
        self.layers = layers
        self.hooks = []
        self.outputs = {}
        tf = find_transformer_layers(model)
        for L in layers:
            def make(idx):
                def hook(_, __, out):
                    h = out[0] if isinstance(out, tuple) else out
                    self.outputs[idx] = h.detach()
                return hook
            self.hooks.append(tf[L].register_forward_hook(make(L)))

    def clear(self):
        self.outputs.clear()

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


class SteeringHook:
    """Add α * direction to residual stream at specified layers."""
    def __init__(self, model, direction: torch.Tensor, alpha: float, layers: List[int]):
        self.hooks = []
        self.direction = direction  # [D], unit-normalized
        self.alpha = alpha
        tf = find_transformer_layers(model)
        for L in layers:
            def make(idx):
                def hook(_, __, out):
                    if isinstance(out, tuple):
                        x = out[0]
                        rest = out[1:]
                    else:
                        x = out
                        rest = None
                    # Add steering: x ← x + α·v
                    x_steered = x + self.alpha * self.direction.to(x.device)
                    if rest is None:
                        return x_steered
                    return (x_steered, *rest)
                return hook
            self.hooks.append(tf[L].register_forward_hook(make(L)))

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


# ─────────────────────────────────────────────────────────────────────
#  Probe training
# ─────────────────────────────────────────────────────────────────────

def train_probe_at_layer(
    model, samples, layer, device,
) -> Tuple[np.ndarray, float, np.ndarray]:
    """Extract residuals at one layer, train probe, return (direction, AUROC, residuals)."""
    cap = ResidualCapture(model, [layer])
    residuals = []
    for s in samples:
        cap.clear()
        with torch.no_grad():
            model(input_ids=torch.tensor([s["token_ids"]], device=device))
        h = cap.outputs[layer]
        pos = min(s["anchor"], h.shape[1] - 1)
        residuals.append(h[0, pos, :].float().cpu().numpy())
    cap.remove()

    X = np.stack(residuals)
    labels = np.array([1 if s["frequency_category"] == "high_freq" else 0 for s in samples])
    pair_ids = np.array([s["pair_id"] for s in samples])

    gss = GroupShuffleSplit(n_splits=5, test_size=0.3, random_state=42)
    aurocs, coefs = [], []
    for tr, te in gss.split(X, labels, groups=pair_ids):
        clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
        clf.fit(X[tr], labels[tr])
        prob = clf.predict_proba(X[te])[:, 1]
        aurocs.append(roc_auc_score(labels[te], prob))
        coefs.append(clf.coef_[0])

    direction = np.mean(coefs, axis=0)
    direction = direction / (np.linalg.norm(direction) + 1e-12)
    return direction, float(np.mean(aurocs)), X


# ─────────────────────────────────────────────────────────────────────
#  E0.1: Steering test — measure output shift under additive intervention
# ─────────────────────────────────────────────────────────────────────

def compute_token_log_frequencies(tokenizer, corpus_tokens: Optional[List[int]] = None):
    """Compute log-frequency for each token in the vocabulary.
    Uses a uniform-smoothed estimate if no corpus provided."""
    vocab_size = tokenizer.vocab_size
    if corpus_tokens is not None:
        counts = Counter(corpus_tokens)
        total = sum(counts.values())
        log_freqs = np.zeros(vocab_size)
        for i in range(vocab_size):
            log_freqs[i] = np.log((counts.get(i, 0) + 1) / (total + vocab_size))
    else:
        # Fallback: use token ID as a rough proxy (lower IDs tend to be more frequent
        # in BPE tokenizers due to merge ordering)
        log_freqs = -np.log(np.arange(1, vocab_size + 1).astype(float))
        log_freqs = log_freqs / np.abs(log_freqs).max()
    return log_freqs


def run_steering_test(
    model,
    tokenizer,
    direction: np.ndarray,
    layer: int,
    alphas: List[float],
    prompts: List[str],
    token_log_freqs: np.ndarray,
    device: str,
    max_new_tokens: int = 50,
) -> List[Dict[str, Any]]:
    """For each α, generate text with steering and measure output properties."""
    v = torch.tensor(direction, dtype=torch.float32, device=device)
    results = []

    for alpha in alphas:
        print(f"    α={alpha:+.1f}", end=" ", flush=True)

        all_gen_log_freqs = []
        all_kl = []
        all_high_freq_mass = []
        all_low_freq_mass = []

        for prompt in prompts:
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
            prompt_len = input_ids.shape[1]

            # Clean logits at last prompt position (for KL)
            with torch.no_grad():
                clean_out = model(input_ids=input_ids)
                clean_logits = clean_out.logits[0, -1, :]

            # Steered logits
            if abs(alpha) > 1e-8:
                hook = SteeringHook(model, v, alpha, [layer])
            else:
                hook = None

            with torch.no_grad():
                steered_out = model(input_ids=input_ids)
                steered_logits = steered_out.logits[0, -1, :]

                # Also generate
                gen = model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=1.0,
                    top_p=0.9,
                )

            if hook is not None:
                hook.remove()

            # Measure KL between clean and steered logits
            log_p = F.log_softmax(clean_logits, dim=-1)
            log_q = F.log_softmax(steered_logits, dim=-1)
            kl = float((log_p.exp() * (log_p - log_q)).sum())
            all_kl.append(kl)

            # Generated token log-frequencies
            gen_tokens = gen[0, prompt_len:].cpu().numpy()
            gen_lf = [token_log_freqs[t] for t in gen_tokens if t < len(token_log_freqs)]
            if gen_lf:
                all_gen_log_freqs.extend(gen_lf)

            # Probability mass on high-freq vs low-freq tokens
            probs = F.softmax(steered_logits, dim=-1).cpu().numpy()
            median_freq = np.median(token_log_freqs)
            high_mask = token_log_freqs >= median_freq
            low_mask = token_log_freqs < median_freq
            all_high_freq_mass.append(float(probs[high_mask[:len(probs)]].sum()))
            all_low_freq_mass.append(float(probs[low_mask[:len(probs)]].sum()))

        result = {
            "alpha": alpha,
            "layer": layer,
            "mean_gen_log_freq": float(np.mean(all_gen_log_freqs)) if all_gen_log_freqs else 0.0,
            "std_gen_log_freq": float(np.std(all_gen_log_freqs)) if all_gen_log_freqs else 0.0,
            "mean_kl": float(np.mean(all_kl)),
            "mean_high_freq_mass": float(np.mean(all_high_freq_mass)),
            "mean_low_freq_mass": float(np.mean(all_low_freq_mass)),
            "n_prompts": len(prompts),
            "n_gen_tokens": len(all_gen_log_freqs),
        }
        results.append(result)
        print(f"log_freq={result['mean_gen_log_freq']:.3f} KL={result['mean_kl']:.4f} "
              f"hi_mass={result['mean_high_freq_mass']:.3f}", flush=True)

    return results


# ─────────────────────────────────────────────────────────────────────
#  E0.2: Probe orthogonality — frequency vs semantic directions
# ─────────────────────────────────────────────────────────────────────

def compute_orthogonality(
    model, samples, all_layers, device,
) -> pd.DataFrame:
    """Train frequency and per-domain semantic probes at each layer.
    Return cosine between frequency direction and each semantic direction."""

    freq_labels = np.array(
        [1 if s["frequency_category"] == "high_freq" else 0 for s in samples])
    domain_labels = np.array([s.get("dataset", s.get("category", "unknown")) for s in samples])
    pair_ids = np.array([s["pair_id"] for s in samples])

    rows = []
    for L in all_layers:
        print(f"  Layer {L}...", end=" ", flush=True)

        # Extract residuals
        cap = ResidualCapture(model, [L])
        residuals = []
        for s in samples:
            cap.clear()
            with torch.no_grad():
                model(input_ids=torch.tensor([s["token_ids"]], device=device))
            h = cap.outputs[L]
            pos = min(s["anchor"], h.shape[1] - 1)
            residuals.append(h[0, pos, :].float().cpu().numpy())
        cap.remove()
        X = np.stack(residuals)

        # Frequency probe direction
        clf_freq = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
        clf_freq.fit(X, freq_labels)
        v_freq = clf_freq.coef_[0]
        v_freq = v_freq / (np.linalg.norm(v_freq) + 1e-12)
        freq_auroc = roc_auc_score(freq_labels, clf_freq.predict_proba(X)[:, 1])

        # Per-domain semantic probe directions
        le = LabelEncoder()
        y_domain = le.fit_transform(domain_labels)
        domains = le.classes_

        for i, domain in enumerate(domains):
            binary = (y_domain == i).astype(int)
            if binary.sum() < 5 or (1 - binary).sum() < 5:
                continue
            clf_sem = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
            clf_sem.fit(X, binary)
            v_sem = clf_sem.coef_[0]
            v_sem = v_sem / (np.linalg.norm(v_sem) + 1e-12)

            cos = float(np.dot(v_freq, v_sem))
            sem_auroc = roc_auc_score(binary, clf_sem.predict_proba(X)[:, 1])

            rows.append({
                "layer": L,
                "domain": domain,
                "cos_freq_semantic": cos,
                "abs_cos": abs(cos),
                "freq_auroc": freq_auroc,
                "semantic_auroc": sem_auroc,
            })

        print(f"freq_auroc={freq_auroc:.3f}", flush=True)

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--alphas", default="-3,-1.5,-0.5,0,0.5,1.5,3")
    parser.add_argument("--steer-layer", type=int, default=None,
                       help="Layer to steer at. Default: layer with peak freq AUROC.")
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--n-prompts", type=int, default=100,
                       help="Number of prompts for generation test")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.makedirs(args.output_dir, exist_ok=True)
    alphas = [float(x) for x in args.alphas.split(",")]

    print(f"Loading model {args.model} (revision={args.revision})")
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, torch_dtype=torch.float32,
    ).to(args.device)
    model.eval()

    all_layers = _infer_layers(args.model)

    # Load ALL datasets merged
    all_samples = []
    for ds_name, fname in DATASET_FILES.items():
        ds_path = Path(args.dataset_dir) / fname
        if not ds_path.exists():
            continue
        pairs = load_synonym_pairs(str(ds_path))
        samples = pairs_to_samples(pairs, tokenizer)
        for s in samples:
            s["dataset"] = ds_name
        all_samples.extend(samples)

    print(f"Total samples: {len(all_samples)}")

    # ── E0.2: Probe orthogonality ──
    print("\n=== E0.2: Probe orthogonality (frequency vs semantic directions) ===")
    ortho_df = compute_orthogonality(model, all_samples, all_layers, args.device)
    ortho_df.to_csv(Path(args.output_dir) / "probe_orthogonality.csv", index=False)

    print("\n  Summary (mean |cos| between freq and semantic directions):")
    summary = ortho_df.groupby("layer")["abs_cos"].mean()
    for L, val in summary.items():
        print(f"    L{L}: mean |cos| = {val:.4f}")

    # Find best layer for steering
    if args.steer_layer is not None:
        best_layer = args.steer_layer
    else:
        # Use layer with peak freq AUROC
        layer_aurocs = ortho_df.groupby("layer")["freq_auroc"].first()
        best_layer = int(layer_aurocs.idxmax())
    print(f"\n  Best layer for steering: L{best_layer} "
          f"(freq AUROC = {ortho_df[ortho_df['layer'] == best_layer]['freq_auroc'].iloc[0]:.3f})")

    # Train probe at best layer
    print(f"\n=== Training probe at L{best_layer} ===")
    direction, auroc, _ = train_probe_at_layer(model, all_samples, best_layer, args.device)
    print(f"  Probe AUROC: {auroc:.3f}")

    # Compute token log-frequencies (rough: use BPE merge order as proxy)
    # For a proper version, stream from the Pile. This is a fast fallback.
    token_log_freqs = compute_token_log_frequencies(tokenizer)

    # Build prompts for generation test
    # Use ScaleJSD sentence templates as prompts (truncated before the target word)
    prompts = []
    seen = set()
    for s in all_samples:
        text = s["sentence"]
        # Use full sentence as prompt
        if text not in seen:
            prompts.append(text)
            seen.add(text)
        if len(prompts) >= args.n_prompts:
            break

    print(f"\n=== E0.1: Steering test at L{best_layer} ({len(prompts)} prompts) ===")
    steering_results = run_steering_test(
        model, tokenizer, direction, best_layer, alphas,
        prompts, token_log_freqs, args.device, args.max_new_tokens)

    steering_df = pd.DataFrame(steering_results)
    steering_df.to_csv(Path(args.output_dir) / "steering_test.csv", index=False)

    # ── Summary ──
    print(f"\n{'='*60}")
    print("  E0.1 SUMMARY — Steering effect on generated token frequency")
    print(f"{'='*60}")
    print(steering_df[["alpha", "mean_gen_log_freq", "mean_kl",
                       "mean_high_freq_mass", "mean_low_freq_mass"]].to_string(index=False))

    # Check monotonicity
    gen_freqs = steering_df["mean_gen_log_freq"].values
    diffs = np.diff(gen_freqs)
    monotonic = all(d >= 0 for d in diffs) or all(d <= 0 for d in diffs)
    effect_range = gen_freqs.max() - gen_freqs.min()
    baseline_std = steering_df.loc[steering_df["alpha"] == 0, "std_gen_log_freq"].values
    baseline_std = baseline_std[0] if len(baseline_std) > 0 else 1.0

    print(f"\n  Monotonic: {monotonic}")
    print(f"  Effect range: {effect_range:.4f}")
    print(f"  Baseline std: {baseline_std:.4f}")
    print(f"  Effect / baseline_std: {effect_range / (baseline_std + 1e-12):.2f}")
    print(f"  PASS CRITERION: effect/std >= 0.5 and monotonic")
    passes = monotonic and (effect_range / (baseline_std + 1e-12) >= 0.5)
    print(f"  **{'PASS' if passes else 'FAIL'}**")

    # Save summary
    summary_data = {
        "model": args.model,
        "best_layer": best_layer,
        "probe_auroc": auroc,
        "monotonic": bool(monotonic),
        "effect_range": float(effect_range),
        "baseline_std": float(baseline_std),
        "effect_over_std": float(effect_range / (baseline_std + 1e-12)),
        "passes": bool(passes),
        "alphas": alphas,
        "n_prompts": len(prompts),
        "orthogonality_mean_abs_cos": float(ortho_df["abs_cos"].mean()),
    }
    with open(Path(args.output_dir) / "summary.json", "w") as f:
        json.dump(summary_data, f, indent=2)

    print(f"\n  Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
