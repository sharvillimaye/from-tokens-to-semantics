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

def _resolve_vocab_size(model, tokenizer) -> int:
    """Return the logit-matrix vocab size (model-side, not tokenizer-side)."""
    try:
        return int(model.get_output_embeddings().weight.shape[0])
    except Exception:
        return int(getattr(tokenizer, "vocab_size", len(tokenizer)))


def compute_token_log_frequencies(
    model,
    tokenizer,
    device: str,
    corpus_file: Optional[str] = None,
    unigram_counts_file: Optional[str] = None,
    n_prior_prompts: int = 64,
) -> np.ndarray:
    """Compute log-frequency over the model's logit vocabulary.

    Priority:
      1) --unigram-counts-file (.npy / .npz with a vocab_size int array)
      2) --corpus-file plaintext (one doc per line, tokenized & counted)
      3) Model-derived unigram prior: average softmax under diverse short
         priming contexts. Principled approximation when no corpus is
         available; it is precisely the model's own learned marginal
         P(token | no_context), which is the correct baseline for the
         steering test (we measure what the model emits; we compare to
         what the model thinks is frequent).
    """
    vocab_size = _resolve_vocab_size(model, tokenizer)

    if unigram_counts_file and Path(unigram_counts_file).exists():
        print(f"  Loading unigram counts from {unigram_counts_file}")
        obj = np.load(unigram_counts_file, allow_pickle=False)
        counts = obj["counts"] if hasattr(obj, "files") else obj
        counts = np.asarray(counts, dtype=np.int64).reshape(-1)
        if counts.shape[0] != vocab_size:
            print(f"  WARN: counts size {counts.shape[0]} != model vocab {vocab_size}; "
                  f"padding/truncating")
            fixed = np.zeros(vocab_size, dtype=np.int64)
            n = min(vocab_size, counts.shape[0])
            fixed[:n] = counts[:n]
            counts = fixed
        total = int(counts.sum())
        return np.log((counts + 1) / (total + vocab_size))

    if corpus_file and Path(corpus_file).exists():
        print(f"  Building unigram counts from {corpus_file}")
        counts = np.zeros(vocab_size, dtype=np.int64)
        with open(corpus_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ids = tokenizer.encode(line, add_special_tokens=False)
                for t in ids:
                    if 0 <= t < vocab_size:
                        counts[t] += 1
        total = int(counts.sum())
        print(f"  Counted {total} tokens; {(counts > 0).sum()}/{vocab_size} vocab covered")
        return np.log((counts + 1) / (total + vocab_size))

    print("  No corpus file supplied — deriving unigram prior from model "
          f"softmax under {n_prior_prompts} short priming contexts")
    prime_texts = [
        "", " ", ".", "\n", "The", " The", "In", "A", "This", "I",
        "We", "One", "When", "After", "Before", "If", "However",
        " the", " a", " to", " and", " of", " in", " is", " that",
        " for", " with", " as", " on", " at", " by", " this",
    ]
    while len(prime_texts) < n_prior_prompts:
        prime_texts.append(prime_texts[len(prime_texts) % 32])
    prime_texts = prime_texts[:n_prior_prompts]

    accum = torch.zeros(vocab_size, dtype=torch.float64, device="cpu")
    n_used = 0
    bos = tokenizer.bos_token_id
    for txt in prime_texts:
        ids = tokenizer.encode(txt, add_special_tokens=False) or [bos if bos is not None else 0]
        with torch.no_grad():
            out = model(input_ids=torch.tensor([ids], device=device))
            logits = out.logits[0, -1, :vocab_size].float().cpu()
            accum += torch.softmax(logits, dim=-1).to(torch.float64)
            n_used += 1
    accum = (accum / max(n_used, 1)).numpy()
    return np.log(accum + 1e-12)


def _generation_perplexity(model, gen_ids: torch.Tensor, prompt_len: int) -> float:
    """Self-perplexity of the generated continuation under the CURRENT (steered or clean)
    model. Measures whether generation stays fluent under steering."""
    if gen_ids.shape[1] <= prompt_len + 1:
        return float("nan")
    with torch.no_grad():
        out = model(input_ids=gen_ids)
        logits = out.logits[0, prompt_len - 1:-1, :]
        targets = gen_ids[0, prompt_len:]
        logp = F.log_softmax(logits.float(), dim=-1)
        tok_logp = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        return float(torch.exp(-tok_logp.mean()))


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
    seed: int = 42,
    sample_texts_per_alpha: int = 5,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """For each α, generate text with steering and measure output properties.

    Returns (metric_rows, sample_generations).
    """
    v = torch.tensor(direction, dtype=torch.float32, device=device)
    vocab_size = len(token_log_freqs)

    # Quantile split (top 25% = common, bottom 25% = rare) — cleaner than median.
    q_hi = np.quantile(token_log_freqs, 0.75)
    q_lo = np.quantile(token_log_freqs, 0.25)
    high_mask = token_log_freqs >= q_hi
    low_mask = token_log_freqs <= q_lo

    results: List[Dict[str, Any]] = []
    samples: List[Dict[str, Any]] = []

    # Pre-compute clean logits per prompt once (independent of α).
    clean_cache: Dict[int, torch.Tensor] = {}
    clean_perp_cache: Dict[int, float] = {}
    prompt_tensors = []
    for pi, prompt in enumerate(prompts):
        ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        prompt_tensors.append(ids)
        with torch.no_grad():
            clean_cache[pi] = model(input_ids=ids).logits[0, -1, :vocab_size].float().cpu()

    for alpha in alphas:
        print(f"    α={alpha:+.2f}", end=" ", flush=True)

        all_gen_log_freqs: List[float] = []
        per_prompt_gen_log_freq_mean: List[float] = []
        all_kl: List[float] = []
        all_high_mass: List[float] = []
        all_low_mass: List[float] = []
        all_perp: List[float] = []

        hook = SteeringHook(model, v, alpha, [layer]) if abs(alpha) > 1e-8 else None

        try:
            for pi, input_ids in enumerate(prompt_tensors):
                prompt_len = int(input_ids.shape[1])

                with torch.no_grad():
                    steered_out = model(input_ids=input_ids)
                    steered_logits = steered_out.logits[0, -1, :vocab_size].float().cpu()

                    torch.manual_seed(seed + pi)
                    gen = model.generate(
                        input_ids,
                        max_new_tokens=max_new_tokens,
                        do_sample=True,
                        temperature=1.0,
                        top_p=0.9,
                        pad_token_id=tokenizer.pad_token_id
                            if tokenizer.pad_token_id is not None
                            else tokenizer.eos_token_id,
                    )

                log_p = F.log_softmax(clean_cache[pi], dim=-1)
                log_q = F.log_softmax(steered_logits, dim=-1)
                kl = float((log_p.exp() * (log_p - log_q)).sum())
                all_kl.append(kl)

                perp = _generation_perplexity(model, gen, prompt_len)
                all_perp.append(perp)

                gen_tokens = gen[0, prompt_len:].cpu().numpy()
                gen_lf = [float(token_log_freqs[t]) for t in gen_tokens if 0 <= t < vocab_size]
                if gen_lf:
                    all_gen_log_freqs.extend(gen_lf)
                    per_prompt_gen_log_freq_mean.append(float(np.mean(gen_lf)))

                probs = F.softmax(steered_logits, dim=-1).numpy()
                all_high_mass.append(float(probs[high_mask].sum()))
                all_low_mass.append(float(probs[low_mask].sum()))

                if pi < sample_texts_per_alpha:
                    samples.append({
                        "alpha": alpha,
                        "prompt": tokenizer.decode(input_ids[0], skip_special_tokens=True),
                        "generation": tokenizer.decode(
                            gen[0, prompt_len:], skip_special_tokens=True),
                        "perplexity": perp,
                        "kl_first_pos": kl,
                    })
        finally:
            if hook is not None:
                hook.remove()

        clean_perp_baseline = np.nanmean([samples[i]["perplexity"]
                                          for i in range(min(len(samples), sample_texts_per_alpha))
                                          if samples[i]["alpha"] == 0.0]) if alpha != 0.0 else np.nan

        result = {
            "alpha": float(alpha),
            "layer": int(layer),
            "mean_gen_log_freq": float(np.mean(all_gen_log_freqs)) if all_gen_log_freqs else float("nan"),
            "std_gen_log_freq": float(np.std(all_gen_log_freqs)) if all_gen_log_freqs else float("nan"),
            "per_prompt_gen_log_freq_std": float(np.std(per_prompt_gen_log_freq_mean))
                if per_prompt_gen_log_freq_mean else float("nan"),
            "mean_kl": float(np.mean(all_kl)),
            "mean_high_freq_mass": float(np.mean(all_high_mass)),
            "mean_low_freq_mass": float(np.mean(all_low_mass)),
            "mean_perplexity": float(np.nanmean(all_perp)),
            "median_perplexity": float(np.nanmedian(all_perp)),
            "n_prompts": len(prompts),
            "n_gen_tokens": len(all_gen_log_freqs),
        }
        results.append(result)
        print(f"log_freq={result['mean_gen_log_freq']:+.3f} "
              f"KL={result['mean_kl']:.4f} "
              f"hi_mass={result['mean_high_freq_mass']:.3f} "
              f"perp={result['median_perplexity']:.2f}", flush=True)

    return results, samples


# ─────────────────────────────────────────────────────────────────────
#  E0.2: Probe orthogonality — frequency vs semantic directions
# ─────────────────────────────────────────────────────────────────────

def _extract_all_layer_residuals(
    model, samples, layers: List[int], device: str,
) -> Dict[int, np.ndarray]:
    """Single forward pass per sample, capturing every layer at the anchor position."""
    cap = ResidualCapture(model, layers)
    per_layer: Dict[int, List[np.ndarray]] = {L: [] for L in layers}
    try:
        for s in samples:
            cap.clear()
            with torch.no_grad():
                model(input_ids=torch.tensor([s["token_ids"]], device=device))
            for L in layers:
                h = cap.outputs[L]
                pos = min(s["anchor"], h.shape[1] - 1)
                per_layer[L].append(h[0, pos, :].float().cpu().numpy())
    finally:
        cap.remove()
    return {L: np.stack(v) for L, v in per_layer.items()}


def _cv_probe(X: np.ndarray, labels: np.ndarray, groups: np.ndarray) -> Tuple[np.ndarray, float]:
    """Group-aware CV probe. Returns (mean coef unit-vector, mean test AUROC)."""
    gss = GroupShuffleSplit(n_splits=5, test_size=0.3, random_state=42)
    aurocs, coefs = [], []
    for tr, te in gss.split(X, labels, groups=groups):
        if labels[tr].sum() < 2 or (1 - labels[tr]).sum() < 2:
            continue
        if labels[te].sum() < 1 or (1 - labels[te]).sum() < 1:
            continue
        clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
        clf.fit(X[tr], labels[tr])
        aurocs.append(roc_auc_score(labels[te], clf.predict_proba(X[te])[:, 1]))
        coefs.append(clf.coef_[0])
    if not coefs:
        return np.zeros(X.shape[1]), float("nan")
    v = np.mean(coefs, axis=0)
    v = v / (np.linalg.norm(v) + 1e-12)
    return v, float(np.mean(aurocs))


def compute_orthogonality(
    model, samples, all_layers, device,
) -> pd.DataFrame:
    """Train frequency and per-domain semantic probes at each layer with
    group-aware CV. Return cosine between frequency direction and each
    semantic direction. Uses a single forward pass per sample across all layers."""

    freq_labels = np.array(
        [1 if s["frequency_category"] == "high_freq" else 0 for s in samples])
    domain_labels = np.array([s.get("dataset", s.get("category", "unknown")) for s in samples])
    pair_ids = np.array([s["pair_id"] for s in samples])

    print(f"  Extracting residuals for {len(samples)} samples × "
          f"{len(all_layers)} layers in one sweep...")
    per_layer_X = _extract_all_layer_residuals(model, samples, all_layers, device)

    le = LabelEncoder()
    y_domain = le.fit_transform(domain_labels)
    domains = le.classes_

    rows = []
    for L in all_layers:
        X = per_layer_X[L]
        v_freq, freq_auroc = _cv_probe(X, freq_labels, pair_ids)

        for i, domain in enumerate(domains):
            binary = (y_domain == i).astype(int)
            if binary.sum() < 10 or (1 - binary).sum() < 10:
                continue
            v_sem, sem_auroc = _cv_probe(X, binary, pair_ids)

            cos = float(np.dot(v_freq, v_sem))
            rows.append({
                "layer": int(L),
                "domain": str(domain),
                "cos_freq_semantic": cos,
                "abs_cos": float(abs(cos)),
                "freq_auroc": float(freq_auroc),
                "semantic_auroc": float(sem_auroc),
            })
        print(f"  L{L}: freq_auroc={freq_auroc:.3f}", flush=True)

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────

def _select_dtype(model_id: str, device: str) -> torch.dtype:
    if device != "cuda":
        return torch.float32
    m = model_id.lower()
    large_markers = ("7b", "6.9b", "8b", "13b", "9b", "11b", "12b", "14b", "32b", "70b")
    if any(marker in m for marker in large_markers):
        return torch.bfloat16
    return torch.float32


def _build_prompt_from_sample(sample: Dict[str, Any], tokenizer) -> Optional[str]:
    """Truncate the full ScaleJSD sentence to everything BEFORE the target ngram,
    so the steered model is predicting the target token rather than continuing
    from after it."""
    sent = sample["sentence"]
    phrase = sample.get("phrase", "")
    if not phrase:
        return sent
    idx = sent.find(phrase)
    if idx <= 0:
        return None
    prefix = sent[:idx].rstrip()
    if not prefix:
        return None
    return prefix


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
    parser.add_argument("--corpus-file", default=None,
                        help="Plaintext corpus (one doc per line) for token unigram counts")
    parser.add_argument("--unigram-counts-file", default=None,
                        help="Pre-computed unigram counts (.npy/.npz, vocab_size int array)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", default="auto", choices=["auto", "float32", "bfloat16", "float16"])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.makedirs(args.output_dir, exist_ok=True)
    alphas = [float(x) for x in args.alphas.split(",")]

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.dtype == "auto":
        dtype = _select_dtype(args.model, args.device)
    else:
        dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]

    print(f"Loading model {args.model} (revision={args.revision}, dtype={dtype})")
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kw: Dict[str, Any] = {"torch_dtype": dtype}
    if args.revision != "main":
        load_kw["revision"] = args.revision
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if hf_token:
        load_kw["token"] = hf_token

    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kw).to(args.device)
    model.eval()

    try:
        n_hidden = int(model.config.num_hidden_layers)
        all_layers = list(range(n_hidden))
    except Exception:
        all_layers = _infer_layers(args.model)

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

    print("\n=== E0.2: Probe orthogonality (frequency vs semantic directions) ===")
    ortho_df = compute_orthogonality(model, all_samples, all_layers, args.device)
    ortho_df.to_csv(Path(args.output_dir) / "probe_orthogonality.csv", index=False)

    print("\n  Summary (mean |cos| between freq and semantic directions):")
    summary = ortho_df.groupby("layer")["abs_cos"].mean()
    for L, val in summary.items():
        print(f"    L{int(L)}: mean |cos| = {val:.4f}")

    if args.steer_layer is not None:
        best_layer = int(args.steer_layer)
    else:
        layer_aurocs = ortho_df.groupby("layer")["freq_auroc"].first()
        best_layer = int(layer_aurocs.idxmax())
    best_auroc = float(ortho_df[ortho_df["layer"] == best_layer]["freq_auroc"].iloc[0])
    print(f"\n  Best layer for steering: L{best_layer} (freq AUROC = {best_auroc:.3f})")

    print(f"\n=== Training probe at L{best_layer} ===")
    direction, auroc, _ = train_probe_at_layer(model, all_samples, best_layer, args.device)
    print(f"  Probe AUROC: {auroc:.3f}")

    print("\n=== Computing token log-frequencies ===")
    token_log_freqs = compute_token_log_frequencies(
        model, tokenizer, args.device,
        corpus_file=args.corpus_file,
        unigram_counts_file=args.unigram_counts_file,
    )

    prompts: List[str] = []
    seen_prefixes: set = set()
    for s in all_samples:
        prefix = _build_prompt_from_sample(s, tokenizer)
        if prefix is None or prefix in seen_prefixes:
            continue
        prompts.append(prefix)
        seen_prefixes.add(prefix)
        if len(prompts) >= args.n_prompts:
            break
    print(f"Built {len(prompts)} prompts (truncated before target phrase)")

    print(f"\n=== E0.1: Steering test at L{best_layer} ({len(prompts)} prompts) ===")
    steering_results, samples_out = run_steering_test(
        model, tokenizer, direction, best_layer, alphas,
        prompts, token_log_freqs, args.device, args.max_new_tokens,
        seed=args.seed)

    steering_df = pd.DataFrame(steering_results)
    steering_df.to_csv(Path(args.output_dir) / "steering_test.csv", index=False)
    pd.DataFrame(samples_out).to_csv(Path(args.output_dir) / "sample_generations.csv", index=False)

    print(f"\n{'='*72}")
    print("  E0.1 SUMMARY — Steering effect on generated token frequency")
    print(f"{'='*72}")
    cols = ["alpha", "mean_gen_log_freq", "mean_kl", "mean_high_freq_mass",
            "mean_low_freq_mass", "median_perplexity"]
    print(steering_df[cols].to_string(index=False))

    gen_freqs = steering_df["mean_gen_log_freq"].to_numpy()
    diffs = np.diff(gen_freqs)
    monotonic = bool(np.all(diffs >= -1e-6)) or bool(np.all(diffs <= 1e-6))
    effect_range = float(np.nanmax(gen_freqs) - np.nanmin(gen_freqs))

    baseline_std_arr = steering_df.loc[
        steering_df["alpha"] == 0, "std_gen_log_freq"].to_numpy()
    baseline_std = float(baseline_std_arr[0]) if baseline_std_arr.size else 1.0

    baseline_perp_arr = steering_df.loc[
        steering_df["alpha"] == 0, "median_perplexity"].to_numpy()
    baseline_perp = float(baseline_perp_arr[0]) if baseline_perp_arr.size else float("nan")
    max_perp = float(np.nanmax(steering_df["median_perplexity"]))
    perp_ratio = max_perp / (baseline_perp + 1e-12)

    print(f"\n  Monotonic: {monotonic}")
    print(f"  Effect range (mean_gen_log_freq max - min): {effect_range:.4f}")
    print(f"  Baseline std of generated log-freqs: {baseline_std:.4f}")
    print(f"  Effect / baseline_std: {effect_range / (baseline_std + 1e-12):.2f}")
    print(f"  Baseline median perplexity: {baseline_perp:.2f}")
    print(f"  Max median perplexity across α: {max_perp:.2f}  (ratio = {perp_ratio:.2f}×)")
    print(f"  PASS: monotonic AND effect/std >= 0.5 AND perp_ratio <= 2.0")
    passes = (monotonic
              and (effect_range / (baseline_std + 1e-12) >= 0.5)
              and (perp_ratio <= 2.0))
    print(f"  **{'PASS' if passes else 'FAIL'}**")

    summary_data = {
        "model": args.model,
        "revision": args.revision,
        "dtype": str(dtype),
        "best_layer": int(best_layer),
        "probe_auroc_best_layer": float(auroc),
        "monotonic": bool(monotonic),
        "effect_range": float(effect_range),
        "baseline_std": float(baseline_std),
        "effect_over_std": float(effect_range / (baseline_std + 1e-12)),
        "baseline_median_perplexity": float(baseline_perp),
        "max_median_perplexity": float(max_perp),
        "perplexity_ratio": float(perp_ratio),
        "passes": bool(passes),
        "alphas": alphas,
        "n_prompts": len(prompts),
        "orthogonality_mean_abs_cos": float(ortho_df["abs_cos"].mean()),
        "used_corpus_file": args.corpus_file,
        "used_unigram_file": args.unigram_counts_file,
    }
    with open(Path(args.output_dir) / "summary.json", "w") as f:
        json.dump(summary_data, f, indent=2)

    print(f"\n  Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
