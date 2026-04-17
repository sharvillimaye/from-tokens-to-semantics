#!/usr/bin/env python3
"""
Idea 3 — Frequency direction as learned code-length axis.

Tests whether the scalar projection `<residual[L*], v>` (residual at the
anchor-token position, projected onto the frequency probe direction at the
probe's best layer) correlates with an external corpus-derived unigram
`log P(token)`. If the probe direction IS a learned code-length axis,
cross-entropy training implicitly implements arithmetic coding — common
symbols get short codes, rare symbols long codes — and the linear fit
`proj = a + b * log P(token) + ε` should hold with high R².

This is complementary to Idea C (b_LN alignment), which tested vocabulary-
space alignment between W_U @ v and the final LayerNorm bias and found them
orthogonal. THIS test is in REPRESENTATION space and uses an EXTERNAL corpus
unigram (wordfreq's Zipf frequencies), not the model's learned output bias.

Evaluation sets:
  1. ScaleJSD anchor tokens — the probe's native task: ~582 per model.
  2. Wikitext sample or fallback: random tokens from ScaleJSD sentences —
     out-of-distribution generalization of the code-length hypothesis.

Interpretation:
  - Pearson r > 0.7 on wikitext set → the direction IS a code-length axis.
  - Pearson r < 0.3 → the direction captures something beyond unigram mass.

CPU-only, fp32, designed for the Oumi Lambda cluster.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr

try:
    from scripts.metrics.coverage_affinity_experiment import (
        load_synonym_pairs,
        pairs_to_samples,
    )
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from scripts.metrics.coverage_affinity_experiment import (
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
#  Transformer layer discovery + residual capture
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
    """Capture residual stream at the output of a specific layer."""

    def __init__(self, model, layers: List[int]):
        self.layers = layers
        self.hooks = []
        self.outputs: Dict[int, torch.Tensor] = {}
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


# ─────────────────────────────────────────────────────────────────────
#  Tokenizer prefix handling
# ─────────────────────────────────────────────────────────────────────

# Common BPE / SentencePiece leading-space markers.
_BPE_PREFIX_CHARS = ("\u2581", "\u0120", "Ġ", "▁")


def detokenize_for_wordfreq(token_text: str) -> str:
    """Strip BPE / SentencePiece prefix markers so wordfreq sees raw text."""
    if not token_text:
        return token_text
    s = token_text
    # Replace all known leading-space markers with a literal space, then strip.
    for ch in _BPE_PREFIX_CHARS:
        s = s.replace(ch, " ")
    return s.strip()


_WORD_RE = re.compile(r"^[A-Za-z]+$")
_NUMERIC_RE = re.compile(r"^[0-9]+(\.[0-9]+)?$")


def token_flags(text: str) -> Tuple[bool, bool]:
    """Return (is_word, is_numeric) given the cleaned token text."""
    if not text:
        return False, False
    return bool(_WORD_RE.match(text)), bool(_NUMERIC_RE.match(text))


# ─────────────────────────────────────────────────────────────────────
#  Evaluation set 1: ScaleJSD anchor tokens
# ─────────────────────────────────────────────────────────────────────

def build_scalejsd_records(
    model,
    tokenizer,
    samples: List[Dict[str, Any]],
    layer: int,
    v: np.ndarray,
    device: str,
    zipf_frequency,
) -> List[Dict[str, Any]]:
    """For each ScaleJSD sample, extract residual[layer] at the anchor token
    position and compute the scalar projection onto v."""
    cap = ResidualCapture(model, [layer])
    records: List[Dict[str, Any]] = []

    try:
        for i, s in enumerate(samples):
            if i % 50 == 0:
                print(f"    [{i + 1}/{len(samples)}] {s['phrase'][:32]!r}", flush=True)
            cap.clear()
            with torch.no_grad():
                model(input_ids=torch.tensor([s["token_ids"]], device=device))
            h = cap.outputs[layer]
            pos = min(s["anchor"], h.shape[1] - 1)
            residual = h[0, pos, :].float().cpu().numpy()

            proj = float(np.dot(residual, v))
            tok_id = int(s["token_ids"][pos])
            raw = tokenizer.decode([tok_id])
            clean = detokenize_for_wordfreq(raw)
            zipf = float(zipf_frequency(clean, "en", minimum=0.0)) if clean else 0.0
            is_word, is_numeric = token_flags(clean)

            records.append({
                "source": "scalejsd",
                "domain": s.get("dataset", s.get("category", "unknown")),
                "token_id": tok_id,
                "token_text": clean,
                "raw_token_text": raw,
                "proj": proj,
                "zipf_wordfreq": zipf,
                # ScaleJSD's own corpus-derived log-frequency (natural log of raw count).
                "log_freq_scalejsd": float(s["log_frequency"]),
                "token_length": len(clean),
                "is_word": is_word,
                "is_numeric": is_numeric,
                "phrase": s["phrase"],
                "frequency_category": s["frequency_category"],
            })
    finally:
        cap.remove()
    return records


# ─────────────────────────────────────────────────────────────────────
#  Evaluation set 2: Wikitext (or fallback: ScaleJSD sentences)
# ─────────────────────────────────────────────────────────────────────

WIKITEXT_CANDIDATES = [
    "/data/ani/mechinterp/corpora/wikitext_train.txt",
    "/data/ani/mechinterp/corpora/wikitext.txt",
    "/data/ani/mechinterp/data/wikitext_train.txt",
]


def _load_text_lines(path: str, max_bytes: int = 20 * 1024 * 1024) -> List[str]:
    """Read a plaintext corpus as a list of lines, stopping after `max_bytes`."""
    lines: List[str] = []
    total = 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            s = raw.strip()
            if not s:
                continue
            lines.append(s)
            total += len(s)
            if total >= max_bytes:
                break
    return lines


def sample_wikitext_records(
    model,
    tokenizer,
    layer: int,
    v: np.ndarray,
    device: str,
    zipf_frequency,
    fallback_sentences: List[str],
    n_tokens: int,
    seed: int,
    max_window_tokens: int = 64,
) -> List[Dict[str, Any]]:
    """Assemble ~n_tokens (text_window, target_token_position) samples from
    either a wikitext file or fallback ScaleJSD sentences, run forward passes,
    and record (proj, zipf) per target token."""
    rng = random.Random(seed)

    source_name = "wikitext"
    lines: List[str] = []
    for p in WIKITEXT_CANDIDATES:
        if Path(p).exists():
            print(f"  Wikitext source: {p}")
            lines = _load_text_lines(p)
            break

    if not lines:
        print("  No wikitext corpus found — falling back to ScaleJSD sentences")
        source_name = "wikitext_fallback_scalejsd"
        lines = list(fallback_sentences)

    rng.shuffle(lines)

    cap = ResidualCapture(model, [layer])
    records: List[Dict[str, Any]] = []
    n_passes = 0

    try:
        for line in lines:
            if len(records) >= n_tokens:
                break
            ids = tokenizer.encode(line, add_special_tokens=False)
            if len(ids) < 2:
                continue
            if len(ids) > max_window_tokens:
                start = rng.randint(0, len(ids) - max_window_tokens)
                ids = ids[start:start + max_window_tokens]

            # Run a single forward pass over the window and harvest multiple
            # (proj, zipf) pairs — one per position. Amortizes forward-pass cost.
            cap.clear()
            with torch.no_grad():
                model(input_ids=torch.tensor([ids], device=device))
            h = cap.outputs[layer]  # [1, T, D]
            n_passes += 1

            # Pick every few positions to get spread across the window.
            positions = list(range(1, len(ids)))  # skip pos 0 (no context)
            if len(positions) > 8:
                positions = rng.sample(positions, 8)

            for pos in positions:
                if len(records) >= n_tokens:
                    break
                residual = h[0, pos, :].float().cpu().numpy()
                proj = float(np.dot(residual, v))
                tok_id = int(ids[pos])
                raw = tokenizer.decode([tok_id])
                clean = detokenize_for_wordfreq(raw)
                zipf = float(zipf_frequency(clean, "en", minimum=0.0)) if clean else 0.0
                is_word, is_numeric = token_flags(clean)
                records.append({
                    "source": source_name,
                    "domain": None,
                    "token_id": tok_id,
                    "token_text": clean,
                    "raw_token_text": raw,
                    "proj": proj,
                    "zipf_wordfreq": zipf,
                    "log_freq_scalejsd": None,
                    "token_length": len(clean),
                    "is_word": is_word,
                    "is_numeric": is_numeric,
                    "phrase": None,
                    "frequency_category": None,
                })

            if n_passes % 25 == 0:
                print(f"    wikitext: {n_passes} passes, {len(records)}/{n_tokens} tokens",
                      flush=True)
    finally:
        cap.remove()

    print(f"  Collected {len(records)} wikitext tokens in {n_passes} forward passes")
    return records


# ─────────────────────────────────────────────────────────────────────
#  Correlation analysis
# ─────────────────────────────────────────────────────────────────────

def _safe_corr(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """Compute Pearson r, Spearman ρ, and a linear fit R² / slope / intercept."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return {"n": int(len(x)), "pearson_r": None, "spearman_rho": None,
                "R2": None, "slope": None, "intercept": None}
    pr, p_p = pearsonr(x, y)
    sr, p_s = spearmanr(x, y)
    slope, intercept = np.polyfit(x, y, 1)
    y_pred = slope * x + intercept
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else None
    return {
        "n": int(len(x)),
        "pearson_r": float(pr),
        "pearson_p": float(p_p),
        "spearman_rho": float(sr),
        "spearman_p": float(p_s),
        "R2": float(r2) if r2 is not None else None,
        "slope": float(slope),
        "intercept": float(intercept),
    }


def analyze(df: pd.DataFrame) -> Dict[str, Any]:
    """Main correlation breakdown: overall + per-source + stratified."""
    out: Dict[str, Any] = {}

    # Overall + per-source correlations (proj vs zipf_wordfreq).
    out["overall"] = _safe_corr(df["proj"].values, df["zipf_wordfreq"].values)
    for src, sub in df.groupby("source"):
        out[f"source:{src}"] = _safe_corr(sub["proj"].values,
                                          sub["zipf_wordfreq"].values)

    # ScaleJSD: also correlate proj vs its own log_freq_scalejsd (the probe's
    # native training signal).
    sj = df[df["source"] == "scalejsd"]
    if len(sj) >= 3:
        out["scalejsd_vs_log_freq_scalejsd"] = _safe_corr(
            sj["proj"].values, sj["log_freq_scalejsd"].values
        )

    # Stratification.
    stratified: Dict[str, Any] = {}

    # (a) Word vs subword — by source, so we see the generalization breakdown.
    by_word: Dict[str, Any] = {}
    for src, sub in df.groupby("source"):
        words = sub[sub["is_word"]]
        subwords = sub[~sub["is_word"]]
        by_word[src] = {
            "word": _safe_corr(words["proj"].values, words["zipf_wordfreq"].values),
            "subword": _safe_corr(subwords["proj"].values,
                                  subwords["zipf_wordfreq"].values),
        }
    stratified["word_vs_subword"] = by_word

    # (b) Token-length quartiles.
    by_len: Dict[str, Any] = {}
    if len(df) >= 8:
        try:
            qbins = pd.qcut(df["token_length"], q=4, duplicates="drop",
                            labels=[f"q{i}" for i in range(1, 5)])
            for lab, sub in df.groupby(qbins):
                by_len[str(lab)] = {
                    "token_length_range": [int(sub["token_length"].min()),
                                           int(sub["token_length"].max())],
                    "corr": _safe_corr(sub["proj"].values,
                                       sub["zipf_wordfreq"].values),
                }
        except ValueError:
            by_len["error"] = "could not compute length quartiles"
    stratified["token_length_quartiles"] = by_len

    # (c) ScaleJSD domain breakdown.
    by_domain: Dict[str, Any] = {}
    for dom, sub in sj.groupby("domain"):
        if dom is None:
            continue
        by_domain[str(dom)] = _safe_corr(sub["proj"].values,
                                         sub["zipf_wordfreq"].values)
    stratified["scalejsd_domain"] = by_domain

    out["stratified"] = stratified
    return out


# ─────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────

def _maybe_load_steer_layer(probe_dir: Path) -> Optional[int]:
    """Try to read the probe's best layer from probe_meta.json in the same dir
    as probe_direction.npy."""
    meta_path = probe_dir / "probe_meta.json"
    if not meta_path.exists():
        return None
    try:
        with open(meta_path) as f:
            meta = json.load(f)
    except Exception:
        return None
    for key in ("best_layer", "steer_layer", "probe_layer", "layer"):
        if key in meta:
            try:
                return int(meta[key])
            except Exception:
                pass
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--probe-direction", required=True,
                        help="Path to probe_direction.npy (shape [d_model])")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steer-layer", type=int, default=None,
                        help="Layer L*. Default: read from probe_meta.json "
                             "next to --probe-direction.")
    parser.add_argument("--dataset-dir",
                        default="/data/ani/mechinterp/code/scaleJSD/dataset/legacy/filtered")
    parser.add_argument("--n-wikitext-tokens", type=int, default=2000)
    parser.add_argument("--max-scalejsd-samples", type=int, default=None,
                        help="Optional cap on ScaleJSD samples for smoke tests.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu",
                        choices=["cpu", "cuda", "mps"])
    args = parser.parse_args()

    # wordfreq import is done here so `--help` works without the dep.
    try:
        from wordfreq import zipf_frequency
    except ImportError as e:
        raise SystemExit(
            "wordfreq is required — `pip install wordfreq`"
        ) from e

    from transformers import AutoModelForCausalLM, AutoTokenizer

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load probe direction ─────────────────────────────────────────
    probe_path = Path(args.probe_direction)
    print(f"Loading probe direction from {probe_path}")
    v_np = np.load(probe_path).astype(np.float64).reshape(-1)
    print(f"  probe direction shape: {v_np.shape}, norm: {np.linalg.norm(v_np):.6f}")

    # Use unit direction (scalar projection).
    v_unit = v_np / (np.linalg.norm(v_np) + 1e-12)

    # ── Determine steer layer ────────────────────────────────────────
    steer_layer = args.steer_layer
    if steer_layer is None:
        steer_layer = _maybe_load_steer_layer(probe_path.parent)
    if steer_layer is None:
        raise SystemExit(
            "Must provide --steer-layer (no probe_meta.json found with "
            "best_layer/steer_layer/probe_layer/layer)."
        )
    print(f"  steer layer L* = {steer_layer}")

    # ── Load model ───────────────────────────────────────────────────
    print(f"\nLoading model {args.model} @ {args.revision} on {args.device} (fp32)")
    load_kw: Dict[str, Any] = {"torch_dtype": torch.float32}
    if args.revision != "main":
        load_kw["revision"] = args.revision
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if hf_token:
        load_kw["token"] = hf_token

    tok_kw: Dict[str, Any] = {}
    if args.revision != "main":
        tok_kw["revision"] = args.revision
    if hf_token:
        tok_kw["token"] = hf_token

    tokenizer = AutoTokenizer.from_pretrained(args.model, **tok_kw)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kw).to(args.device)
    model.eval()

    d_model = int(model.config.hidden_size) if hasattr(model.config, "hidden_size") \
        else int(getattr(model.config, "n_embd", v_unit.shape[0]))
    if v_unit.shape[0] != d_model:
        raise SystemExit(
            f"Probe direction dim {v_unit.shape[0]} != model d_model {d_model}"
        )
    print(f"  d_model = {d_model}")

    # ── Load ScaleJSD samples (all 5 domains) ────────────────────────
    print(f"\nLoading ScaleJSD from {args.dataset_dir}")
    all_samples: List[Dict[str, Any]] = []
    for ds_name, fname in DATASET_FILES.items():
        ds_path = Path(args.dataset_dir) / fname
        if not ds_path.exists():
            print(f"  skipping {ds_name}: {ds_path} not found")
            continue
        pairs = load_synonym_pairs(str(ds_path))
        samples = pairs_to_samples(pairs, tokenizer)
        for s in samples:
            s["dataset"] = ds_name
        all_samples.extend(samples)
    print(f"Total ScaleJSD samples: {len(all_samples)}")

    if args.max_scalejsd_samples is not None and len(all_samples) > args.max_scalejsd_samples:
        rng = random.Random(args.seed)
        all_samples = rng.sample(all_samples, args.max_scalejsd_samples)
        print(f"  capped ScaleJSD samples to {len(all_samples)} (smoke test)")

    # ── Eval set 1: ScaleJSD ─────────────────────────────────────────
    print(f"\n=== Eval set 1: ScaleJSD anchor tokens at L{steer_layer} ===")
    sj_records = build_scalejsd_records(
        model, tokenizer, all_samples, steer_layer, v_unit,
        args.device, zipf_frequency,
    )

    # ── Eval set 2: Wikitext ─────────────────────────────────────────
    print(f"\n=== Eval set 2: Wikitext sample at L{steer_layer} ===")
    fallback_sentences = [s["sentence"] for s in all_samples]
    wt_records = sample_wikitext_records(
        model, tokenizer, steer_layer, v_unit, args.device, zipf_frequency,
        fallback_sentences=fallback_sentences,
        n_tokens=args.n_wikitext_tokens,
        seed=args.seed,
    )

    # ── Assemble DataFrame + write per_token.csv ─────────────────────
    all_records = sj_records + wt_records
    df = pd.DataFrame(all_records)
    # Add model column for provenance.
    df["model"] = args.model
    df["revision"] = args.revision
    df["layer"] = steer_layer

    per_token_path = output_dir / "per_token.csv"
    df.to_csv(per_token_path, index=False)
    print(f"\nWrote {per_token_path} ({len(df)} rows)")

    # ── Correlation analysis ─────────────────────────────────────────
    print("\n=== Correlation analysis ===")
    corr = analyze(df)
    overall = corr["overall"]
    print(f"  OVERALL: n={overall['n']}, pearson r={overall['pearson_r']}, "
          f"spearman ρ={overall['spearman_rho']}, R²={overall['R2']}")
    for key in [k for k in corr if k.startswith("source:")]:
        c = corr[key]
        print(f"  {key}: n={c['n']}, pearson r={c['pearson_r']}, R²={c['R2']}")

    summary = {
        "model": args.model,
        "revision": args.revision,
        "layer": steer_layer,
        "probe_direction_path": str(probe_path),
        "probe_direction_norm": float(np.linalg.norm(v_np)),
        "n_scalejsd": int((df["source"] == "scalejsd").sum()),
        "n_wikitext": int(df["source"].str.startswith("wikitext").sum()),
        "correlations": corr,
    }
    summary_path = output_dir / "correlation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {summary_path}")

    # ── Scatter data for paper figure ────────────────────────────────
    # Keep this compact: proj, zipf_wordfreq, source, is_word for each record.
    scatter = {
        "model": args.model,
        "layer": steer_layer,
        "points": df[["proj", "zipf_wordfreq", "source", "is_word",
                      "log_freq_scalejsd", "token_length"]]
        .to_dict(orient="records"),
    }
    scatter_path = output_dir / "scatter_data.json"
    with open(scatter_path, "w") as f:
        json.dump(scatter, f)
    print(f"Wrote {scatter_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
