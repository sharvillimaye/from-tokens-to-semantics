#!/usr/bin/env python3
"""
Idea G — Reorganization probes.

As frequency AUROC declines in late layers, WHAT grows in its place?
Is frequency erasure a zero-sum rotation (semantic gain = freq loss) or
complex reorganization?

For each model we train linear probes at EVERY layer on SEVEN concepts
and record per-layer CV AUROC (or R² for the position-regression probe):

1.  Frequency           — ScaleJSD high vs low (binary)
2.  Semantic domain     — ScaleJSD 5 domains (one-vs-rest ensemble, mean AUROC)
3.  Sentiment           — IMDB positive vs negative (binary)
4.  Token morphology    — ScaleJSD target-token length (<=4 vs >4 chars)
5.  Token casing        — ScaleJSD target-token begins-with-uppercase (binary)
6.  Sentence position   — within-sentence token index (linear regression → R²)
7.  Word class          — proper noun vs common noun vs function word (multi-class)

All probes use 5-fold CV; multi-class probes use one-vs-rest LR and report
mean AUROC across classes.

Usage:
    python -m scripts.interventions.reorganization_probes \
        --model allenai/OLMo-7B-hf --revision main \
        --dataset-dir datasets/filtered \
        --output-dir runs/reorg_probes/olmo-7b \
        --n-imdb-per-class 500 --device cuda --dtype auto --seed 42
"""

from __future__ import annotations

import argparse
import json
import os
import random
import string
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold, KFold

try:
    from scripts.interventions.frequency_steering import (
        ResidualCapture,
        _extract_all_layer_residuals,
        _select_dtype,
        find_transformer_layers,
    )
    from scripts.metrics.coverage_affinity_experiment import (
        _infer_layers,
        load_synonym_pairs,
        pairs_to_samples,
    )
except ImportError:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from scripts.interventions.frequency_steering import (
        ResidualCapture,
        _extract_all_layer_residuals,
        _select_dtype,
        find_transformer_layers,
    )
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

# Minimal function-word list (closed-class English words) — used for a
# lightweight syntactic categorisation when no POS tagger is available.
FUNCTION_WORDS = {
    # Articles / determiners
    "a", "an", "the", "this", "that", "these", "those",
    "some", "any", "no", "every", "each", "all", "both", "either", "neither",
    "my", "your", "his", "her", "its", "our", "their",
    # Pronouns
    "i", "me", "mine", "you", "yours", "he", "him", "she", "hers",
    "it", "we", "us", "ours", "they", "them", "theirs",
    "who", "whom", "whose", "which", "what",
    # Prepositions
    "of", "in", "on", "at", "by", "for", "with", "about", "against",
    "between", "into", "through", "during", "before", "after", "above",
    "below", "to", "from", "up", "down", "over", "under", "again",
    "further", "then", "once", "here", "there", "as", "until", "while",
    "per", "via",
    # Conjunctions
    "and", "but", "or", "nor", "so", "yet", "because", "if", "though",
    "although", "since", "unless", "while", "whereas",
    # Auxiliary / modal verbs
    "is", "am", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having",
    "do", "does", "did", "doing", "done",
    "can", "could", "shall", "should", "will", "would", "may", "might",
    "must", "ought", "'s", "'re", "'ve", "'d", "'ll", "'m",
    # Negation / particles
    "not", "n't", "no",
}


# ─────────────────────────────────────────────────────────────────────
#  Probe helpers
# ─────────────────────────────────────────────────────────────────────

def _kfold_binary_auroc(
    X: np.ndarray,
    y: np.ndarray,
    groups: Optional[np.ndarray],
    n_splits: int = 5,
    seed: int = 42,
) -> float:
    """5-fold CV (group-aware if groups is given) AUROC for binary labels."""
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return float("nan")
    if groups is not None:
        n_groups = len(np.unique(groups))
        k = min(n_splits, n_groups)
        splitter = GroupKFold(n_splits=k)
        iterator = splitter.split(X, y, groups=groups)
    else:
        k = min(n_splits, len(X))
        splitter = KFold(n_splits=k, shuffle=True, random_state=seed)
        iterator = splitter.split(X, y)

    aurocs: List[float] = []
    for tr, te in iterator:
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        clf = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")
        clf.fit(X[tr], y[tr])
        prob = clf.predict_proba(X[te])[:, 1]
        aurocs.append(roc_auc_score(y[te], prob))
    return float(np.mean(aurocs)) if aurocs else float("nan")


def _kfold_multiclass_mean_auroc(
    X: np.ndarray,
    y: np.ndarray,
    groups: Optional[np.ndarray],
    n_splits: int = 5,
    seed: int = 42,
) -> float:
    """One-vs-rest AUROC per class, averaged. Supports group-aware CV."""
    classes = np.unique(y)
    if len(classes) < 2:
        return float("nan")
    per_class: List[float] = []
    for c in classes:
        binary = (np.asarray(y) == c).astype(int)
        if binary.sum() < 5 or (1 - binary).sum() < 5:
            continue
        per_class.append(_kfold_binary_auroc(X, binary, groups, n_splits, seed))
    if not per_class:
        return float("nan")
    return float(np.nanmean(per_class))


def _kfold_regression_r2(
    X: np.ndarray,
    y: np.ndarray,
    groups: Optional[np.ndarray],
    n_splits: int = 5,
    seed: int = 42,
) -> float:
    """5-fold CV linear regression; report mean R² across folds."""
    if groups is not None:
        k = min(n_splits, len(np.unique(groups)))
        splitter = GroupKFold(n_splits=k)
        iterator = splitter.split(X, y, groups=groups)
    else:
        k = min(n_splits, len(X))
        splitter = KFold(n_splits=k, shuffle=True, random_state=seed)
        iterator = splitter.split(X, y)

    r2s: List[float] = []
    for tr, te in iterator:
        if len(tr) < 2 or len(te) < 2:
            continue
        reg = LinearRegression()
        reg.fit(X[tr], y[tr])
        r2s.append(r2_score(y[te], reg.predict(X[te])))
    return float(np.mean(r2s)) if r2s else float("nan")


# ─────────────────────────────────────────────────────────────────────
#  Feature extraction for ScaleJSD-based probes
# ─────────────────────────────────────────────────────────────────────

def _target_token_str(sample: Dict[str, Any], tokenizer) -> str:
    """Decode the anchor token only — used for morphology/casing probes."""
    try:
        tok_id = sample["token_ids"][sample["anchor"]]
        tok = tokenizer.decode([tok_id])
    except Exception:
        tok = sample.get("phrase", "")
    return tok


def _token_is_uppercase(token_str: str) -> int:
    """1 if the decoded token (after leading whitespace) starts uppercase."""
    stripped = token_str.lstrip()
    if not stripped:
        return 0
    first = stripped[0]
    return int(first.isalpha() and first.isupper())


def _token_is_long(token_str: str, threshold: int = 4) -> int:
    """1 if the decoded token (stripped of leading ws) is >threshold chars."""
    stripped = token_str.lstrip()
    # Ignore purely non-alpha tokens for a cleaner signal
    alpha = "".join(ch for ch in stripped if ch.isalpha())
    return int(len(alpha) > threshold)


def _word_class(phrase: str) -> str:
    """
    Lightweight heuristic word-class label for ScaleJSD n-grams:
      - 'function' if the n-gram (first token or whole phrase) is a closed-class word
      - 'proper'   if begins with an uppercase letter
      - 'common'   otherwise
    Used purely for an orthogonal syntactic probe; not meant to be linguistically perfect.
    """
    if not phrase:
        return "common"
    head = phrase.strip().split()[0] if phrase.strip() else ""
    head_lc = head.lower().strip(string.punctuation)
    if head_lc in FUNCTION_WORDS:
        return "function"
    # Proper-noun heuristic: starts with capital and isn't a sentence-initial common word
    if head and head[0].isalpha() and head[0].isupper():
        return "proper"
    return "common"


def _position_in_sentence(sample: Dict[str, Any]) -> float:
    """Fraction of the sentence's tokens preceding the anchor (in [0, 1])."""
    n = max(len(sample["token_ids"]), 1)
    return float(sample["anchor"]) / float(max(n - 1, 1))


# ─────────────────────────────────────────────────────────────────────
#  IMDB sentiment data (last-token residual extraction)
# ─────────────────────────────────────────────────────────────────────

def _build_imdb_samples(
    tokenizer,
    n_per_class: int,
    max_length: int = 256,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Load IMDB, balanced pos/neg sample, truncate to max_length tokens.

    Returns list of dicts with token_ids, anchor (= last token index), label.
    """
    from datasets import load_dataset

    ds = load_dataset("stanfordnlp/imdb", split="train")
    pos_indices: List[int] = []
    neg_indices: List[int] = []
    for i, lbl in enumerate(ds["label"]):
        if lbl == 1 and len(pos_indices) < n_per_class * 3:
            pos_indices.append(i)
        elif lbl == 0 and len(neg_indices) < n_per_class * 3:
            neg_indices.append(i)
        if len(pos_indices) >= n_per_class * 3 and len(neg_indices) >= n_per_class * 3:
            break

    rng = random.Random(seed)
    rng.shuffle(pos_indices)
    rng.shuffle(neg_indices)
    chosen = pos_indices[:n_per_class] + neg_indices[:n_per_class]

    samples: List[Dict[str, Any]] = []
    for idx in chosen:
        text = ds[int(idx)]["text"]
        label = int(ds[int(idx)]["label"])
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) == 0:
            continue
        ids = ids[:max_length]
        samples.append({
            "token_ids": ids,
            "anchor": len(ids) - 1,
            "label": label,
        })
    print(f"  IMDB: built {len(samples)} samples "
          f"(pos={sum(s['label'] for s in samples)}, "
          f"neg={sum(1 - s['label'] for s in samples)})")
    return samples


def _extract_last_token_residuals(
    model,
    samples: List[Dict[str, Any]],
    layers: List[int],
    device: str,
) -> Dict[int, np.ndarray]:
    """Single forward pass per sample; capture every layer at the final token."""
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


# ─────────────────────────────────────────────────────────────────────
#  Per-concept scoring
# ─────────────────────────────────────────────────────────────────────

def _score_concept(
    concept: str,
    X: np.ndarray,
    y: np.ndarray,
    groups: Optional[np.ndarray],
    is_regression: bool,
    is_multiclass: bool,
    seed: int,
) -> float:
    if is_regression:
        return _kfold_regression_r2(X, y, groups, seed=seed)
    if is_multiclass:
        return _kfold_multiclass_mean_auroc(X, y, groups, seed=seed)
    return _kfold_binary_auroc(X, y, groups, seed=seed)


# ─────────────────────────────────────────────────────────────────────
#  Summary + plotting
# ─────────────────────────────────────────────────────────────────────

def _summarise(df: pd.DataFrame) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    for concept in df["concept"].unique():
        sub = df[df["concept"] == concept].sort_values("layer")
        vals = sub["auroc"].to_numpy()
        layers = sub["layer"].to_numpy()
        if len(vals) == 0 or np.all(np.isnan(vals)):
            continue
        peak_idx = int(np.nanargmax(vals))
        valid = vals[~np.isnan(vals)]
        diffs = np.diff(valid)
        monotonic_up = bool(np.all(diffs >= -1e-3))
        monotonic_down = bool(np.all(diffs <= 1e-3))
        summary[concept] = {
            "peak_layer": int(layers[peak_idx]),
            "peak_value": float(vals[peak_idx]),
            "first_layer_value": float(vals[0]) if not np.isnan(vals[0]) else None,
            "last_layer_value": float(vals[-1]) if not np.isnan(vals[-1]) else None,
            "monotonic_increasing": monotonic_up,
            "monotonic_decreasing": monotonic_down,
            "n_layers": int(len(vals)),
        }
    return summary


def _plot_trajectories(df: pd.DataFrame, out_path: Path, model_name: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for concept in sorted(df["concept"].unique()):
        sub = df[df["concept"] == concept].sort_values("layer")
        ax.plot(sub["layer"], sub["auroc"], marker="o", markersize=3, label=concept, linewidth=1.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Probe AUROC  (R² for sentence_position)")
    ax.set_title(f"Reorganization probes — {model_name}")
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.7)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--dataset-dir", required=True,
                        help="Path to ScaleJSD filtered/ directory")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-imdb-per-class", type=int, default=500)
    parser.add_argument("--imdb-max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="auto",
                        choices=["auto", "float32", "bfloat16", "float16"])
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    dtype = (
        _select_dtype(args.model, args.device)
        if args.dtype == "auto"
        else {"float32": torch.float32, "bfloat16": torch.bfloat16,
              "float16": torch.float16}[args.dtype]
    )

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
    print(f"Model has {len(all_layers)} layers")

    # ── Load ScaleJSD samples ───────────────────────────────────────
    print("\n=== Loading ScaleJSD ===")
    jsd_samples: List[Dict[str, Any]] = []
    for ds_name, fname in DATASET_FILES.items():
        ds_path = Path(args.dataset_dir) / fname
        if not ds_path.exists():
            print(f"  WARN: missing {ds_path}")
            continue
        pairs = load_synonym_pairs(str(ds_path))
        per = pairs_to_samples(pairs, tokenizer)
        for s in per:
            s["dataset"] = ds_name
        jsd_samples.extend(per)
    print(f"Total ScaleJSD samples: {len(jsd_samples)}")

    # ── Extract ScaleJSD residuals at anchor, every layer, single pass ──
    print("\n=== Extracting ScaleJSD residuals (every layer, anchor pos) ===")
    jsd_X_per_layer = _extract_all_layer_residuals(
        model, jsd_samples, all_layers, args.device)

    # ── Build per-concept labels for ScaleJSD ───────────────────────
    freq_y = np.array(
        [1 if s["frequency_category"] == "high_freq" else 0 for s in jsd_samples])
    domain_y = np.array([s["dataset"] for s in jsd_samples])
    pair_ids = np.array([s["pair_id"] for s in jsd_samples])

    target_tokens = [_target_token_str(s, tokenizer) for s in jsd_samples]
    morphology_y = np.array([_token_is_long(t) for t in target_tokens])
    casing_y = np.array([_token_is_uppercase(t) for t in target_tokens])
    position_y = np.array([_position_in_sentence(s) for s in jsd_samples], dtype=float)
    wordclass_y = np.array([_word_class(s.get("phrase", "")) for s in jsd_samples])

    print(f"  freq balance     : high={int(freq_y.sum())}, low={int((1-freq_y).sum())}")
    print(f"  domain balance   : {dict(zip(*np.unique(domain_y, return_counts=True)))}")
    print(f"  morphology balance: long={int(morphology_y.sum())}, short={int((1-morphology_y).sum())}")
    print(f"  casing balance   : upper={int(casing_y.sum())}, lower={int((1-casing_y).sum())}")
    print(f"  wordclass balance: {dict(zip(*np.unique(wordclass_y, return_counts=True)))}")

    # ── Load IMDB and extract residuals ─────────────────────────────
    print("\n=== Loading IMDB ===")
    imdb_samples = _build_imdb_samples(
        tokenizer, args.n_imdb_per_class,
        max_length=args.imdb_max_length, seed=args.seed)
    imdb_y = np.array([s["label"] for s in imdb_samples])
    print("=== Extracting IMDB residuals (every layer, last token) ===")
    imdb_X_per_layer = _extract_last_token_residuals(
        model, imdb_samples, all_layers, args.device)

    # ── Train per-layer probes for each concept ─────────────────────
    print("\n=== Training per-layer probes ===")
    concept_specs: List[Tuple[str, bool, bool, str]] = [
        # (concept, is_regression, is_multiclass, source_dataset)
        ("frequency", False, False, "scaleJSD"),
        ("semantic_domain", False, True, "scaleJSD"),
        ("sentiment", False, False, "imdb"),
        ("token_morphology", False, False, "scaleJSD"),
        ("token_casing", False, False, "scaleJSD"),
        ("sentence_position", True, False, "scaleJSD"),
        ("word_class", False, True, "scaleJSD"),
    ]

    rows: List[Dict[str, Any]] = []
    for L in all_layers:
        X_jsd = jsd_X_per_layer[L]
        X_imdb = imdb_X_per_layer[L]

        # frequency (binary, group by pair)
        auroc_freq = _score_concept(
            "frequency", X_jsd, freq_y, pair_ids, False, False, args.seed)
        # semantic_domain (multi-class, group by pair)
        auroc_dom = _score_concept(
            "semantic_domain", X_jsd, domain_y, pair_ids, False, True, args.seed)
        # sentiment (binary, no groups)
        auroc_sent = _score_concept(
            "sentiment", X_imdb, imdb_y, None, False, False, args.seed)
        # token_morphology (binary, group by pair)
        auroc_morph = _score_concept(
            "token_morphology", X_jsd, morphology_y, pair_ids, False, False, args.seed)
        # token_casing (binary, group by pair)
        auroc_case = _score_concept(
            "token_casing", X_jsd, casing_y, pair_ids, False, False, args.seed)
        # sentence_position (regression → R², group by pair)
        r2_pos = _score_concept(
            "sentence_position", X_jsd, position_y, pair_ids, True, False, args.seed)
        # word_class (multi-class, group by pair)
        auroc_wc = _score_concept(
            "word_class", X_jsd, wordclass_y, pair_ids, False, True, args.seed)

        rows.extend([
            {"model": args.model, "layer": int(L), "concept": "frequency", "auroc": auroc_freq},
            {"model": args.model, "layer": int(L), "concept": "semantic_domain", "auroc": auroc_dom},
            {"model": args.model, "layer": int(L), "concept": "sentiment", "auroc": auroc_sent},
            {"model": args.model, "layer": int(L), "concept": "token_morphology", "auroc": auroc_morph},
            {"model": args.model, "layer": int(L), "concept": "token_casing", "auroc": auroc_case},
            {"model": args.model, "layer": int(L), "concept": "sentence_position", "auroc": r2_pos},
            {"model": args.model, "layer": int(L), "concept": "word_class", "auroc": auroc_wc},
        ])
        print(
            f"  L{L:02d}  freq={auroc_freq:.3f}  dom={auroc_dom:.3f}  "
            f"sent={auroc_sent:.3f}  morph={auroc_morph:.3f}  "
            f"case={auroc_case:.3f}  pos_r2={r2_pos:+.3f}  wc={auroc_wc:.3f}",
            flush=True,
        )

    df = pd.DataFrame(rows)
    df.to_csv(Path(args.output_dir) / "per_layer_auroc.csv", index=False)

    # ── Summary + plot ─────────────────────────────────────────────
    summary = _summarise(df)
    summary_out = {
        "model": args.model,
        "revision": args.revision,
        "dtype": str(dtype),
        "n_layers": len(all_layers),
        "n_jsd_samples": len(jsd_samples),
        "n_imdb_samples": len(imdb_samples),
        "concepts": summary,
    }
    with open(Path(args.output_dir) / "summary.json", "w") as f:
        json.dump(summary_out, f, indent=2)

    model_tag = args.model.split("/")[-1]
    _plot_trajectories(df, Path(args.output_dir) / "trajectory.png", model_tag)

    print(f"\nWrote {Path(args.output_dir) / 'per_layer_auroc.csv'}")
    print(f"Wrote {Path(args.output_dir) / 'summary.json'}")
    print(f"Wrote {Path(args.output_dir) / 'trajectory.png'}")

    # Brief text summary to stdout
    print("\n=== Peak layers per concept ===")
    for concept, info in summary.items():
        print(f"  {concept:20s}  peak L{info['peak_layer']:2d} "
              f"= {info['peak_value']:.3f}   final={info['last_layer_value']}")


if __name__ == "__main__":
    main()
