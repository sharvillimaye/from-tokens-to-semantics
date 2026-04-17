#!/usr/bin/env python3
"""
Track A — Shadow vs. functional direction comparison.

Extracts the "truth" direction (Marks & Tegmark 2023, "The Geometry of Truth")
using a binary true/false probe on factual statements, then runs the SAME
steering-test pipeline used for the frequency direction. At matched probe
AUROCs, quantify how much bigger the causal effect (KL, behavioral shift) of
the truth direction is compared to the frequency direction.

Why truth: Marks & Tegmark showed the truth direction is causally potent —
adding/subtracting flips veracity outputs. Canonical published example of a
functional direction. Works on any base LM, uses public factual-statement
datasets, cleaner than refusal.

Pipeline:
  1. Load 500-1000 balanced true/false factual statements (Azaria-Mitchell /
     Marks GitHub).
  2. Tokenize each statement; extract the last-token residual at every layer
     in a SINGLE forward pass per sample.
  3. Train a binary true-vs-false logistic-regression probe per layer with
     5-fold CV.  Save truth_probe_per_layer.csv.
  4. Pick peak-AUROC layer L*; extract unit-normalized direction v_L* from the
     mean-of-CV-coefs (same method as train_probe_at_layer).
  5. Run `run_steering_test` on ScaleJSD prompts with the SAME alpha grid used
     for the frequency steering pipeline; write truth_steering_test.csv and
     sample_generations.csv.
  6. If a frequency-steering summary.json exists on PVC, merge truth-vs-
     frequency comparison fields into comparison_summary.json.

Usage:
    python -m scripts.interventions.truth_steering_comparison \
        --model allenai/OLMo-7B-hf --revision main \
        --dataset-dir /data/.../scaleJSD/dataset/legacy/filtered \
        --output-dir /data/.../runs/truth_comparison/olmo-7b \
        --alphas=-2,-1,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1,2 \
        --deterministic \
        --device cuda
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

try:
    from scripts.interventions.frequency_steering import (
        ResidualCapture,
        SteeringHook,
        _build_prompt_from_sample,
        _extract_all_layer_residuals,
        _select_dtype,
        _resolve_vocab_size,
        compute_token_log_frequencies,
        run_steering_test,
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
        SteeringHook,
        _build_prompt_from_sample,
        _extract_all_layer_residuals,
        _select_dtype,
        _resolve_vocab_size,
        compute_token_log_frequencies,
        run_steering_test,
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


# ─────────────────────────────────────────────────────────────────────
# Truth dataset loading (cascade of fallbacks)
# ─────────────────────────────────────────────────────────────────────

# Marks GitHub raw CSVs (https://github.com/saprmarks/geometry-of-truth/tree/main/datasets).
# Each file is: {statement,label} with label ∈ {0,1}.
MARKS_RAW_URLS = [
    "https://raw.githubusercontent.com/saprmarks/geometry-of-truth/main/datasets/cities.csv",
    "https://raw.githubusercontent.com/saprmarks/geometry-of-truth/main/datasets/neg_cities.csv",
    "https://raw.githubusercontent.com/saprmarks/geometry-of-truth/main/datasets/sp_en_trans.csv",
    "https://raw.githubusercontent.com/saprmarks/geometry-of-truth/main/datasets/neg_sp_en_trans.csv",
    "https://raw.githubusercontent.com/saprmarks/geometry-of-truth/main/datasets/smaller_than.csv",
    "https://raw.githubusercontent.com/saprmarks/geometry-of-truth/main/datasets/larger_than.csv",
    "https://raw.githubusercontent.com/saprmarks/geometry-of-truth/main/datasets/common_claim_true_false.csv",
    "https://raw.githubusercontent.com/saprmarks/geometry-of-truth/main/datasets/companies_true_false.csv",
]


def _load_truth_hf(dataset_name: str) -> Optional[List[Dict[str, Any]]]:
    """Load a true/false dataset from HF Hub.  Returns [{statement,label}, ...]
    or None on failure."""
    try:
        from datasets import load_dataset
    except Exception as e:
        print(f"  datasets library unavailable: {e}")
        return None
    try:
        print(f"  Trying HF dataset: {dataset_name}")
        ds = load_dataset(dataset_name)
    except Exception as e:
        print(f"  HF load failed ({dataset_name}): {e}")
        return None
    rows: List[Dict[str, Any]] = []
    for split_name in ds.keys():
        split = ds[split_name]
        cols = split.column_names
        # Find statement / label columns heuristically
        stmt_col = next(
            (c for c in ["statement", "sentence", "claim", "text", "question"] if c in cols),
            None,
        )
        label_col = next(
            (c for c in ["label", "labels", "is_true", "truth", "y"] if c in cols),
            None,
        )
        if stmt_col is None or label_col is None:
            print(f"    skip split {split_name}: cols={cols}")
            continue
        for ex in split:
            lbl = ex[label_col]
            if isinstance(lbl, bool):
                lbl_int = int(lbl)
            elif isinstance(lbl, str):
                lbl_int = 1 if lbl.strip().lower() in ("true", "1", "yes") else 0
            else:
                try:
                    lbl_int = int(lbl)
                except Exception:
                    continue
            rows.append({"statement": str(ex[stmt_col]).strip(), "label": int(lbl_int)})
    print(f"  Loaded {len(rows)} rows from {dataset_name}")
    return rows if rows else None


def _load_truth_marks_github() -> Optional[List[Dict[str, Any]]]:
    """Last-resort fallback: download CSVs directly from Marks GitHub.
    Works without HF Hub access."""
    import csv
    import io
    try:
        import urllib.request
    except Exception:
        return None

    rows: List[Dict[str, Any]] = []
    for url in MARKS_RAW_URLS:
        try:
            print(f"  Fetching {url}")
            with urllib.request.urlopen(url, timeout=30) as resp:
                text = resp.read().decode("utf-8", errors="replace")
        except Exception as e:
            print(f"    skip: {e}")
            continue
        reader = csv.DictReader(io.StringIO(text))
        n0 = len(rows)
        for r in reader:
            stmt = r.get("statement") or r.get("sentence") or r.get("claim") or ""
            lbl = r.get("label") or r.get("truth") or r.get("is_true")
            if not stmt or lbl is None:
                continue
            try:
                lbl_int = int(str(lbl).strip())
            except ValueError:
                lbl_s = str(lbl).strip().lower()
                if lbl_s in ("true", "1", "yes"):
                    lbl_int = 1
                elif lbl_s in ("false", "0", "no"):
                    lbl_int = 0
                else:
                    continue
            rows.append({"statement": stmt.strip(), "label": int(lbl_int)})
        print(f"    +{len(rows) - n0} rows")
    return rows if rows else None


def load_truth_statements(choice: str = "auto",
                          max_samples: int = 1000,
                          seed: int = 42) -> Tuple[List[Dict[str, Any]], str]:
    """Load balanced true/false statements.  Returns (statements, source_tag).

    Each statement: {statement: str, label: int (0/1), stmt_id: int}
    """
    attempts: List[Tuple[str, Any]] = []
    if choice in ("auto", "azaria-mitchell"):
        attempts.append(("azaria-mitchell/true_false",
                         lambda: _load_truth_hf("azaria-mitchell/true_false")))
    if choice in ("auto", "azaria-mitchell-diverse"):
        attempts.append(("notrichardren/azaria-mitchell-diverse",
                         lambda: _load_truth_hf("notrichardren/azaria-mitchell-diverse")))
    if choice in ("auto", "cf-saplma"):
        attempts.append(("OamPatel/cf_saplma",
                         lambda: _load_truth_hf("OamPatel/cf_saplma")))
    if choice in ("auto", "marks-github"):
        attempts.append(("marks-github-raw", _load_truth_marks_github))

    rows: List[Dict[str, Any]] = []
    source_tag = "none"
    for tag, fn in attempts:
        try:
            result = fn()
        except Exception as e:
            print(f"  {tag} attempt errored: {e}")
            result = None
        if result:
            rows = result
            source_tag = tag
            break

    if not rows:
        raise RuntimeError(
            "Could not load any truth dataset — tried HF Azaria-Mitchell, "
            "diverse fork, cf_saplma, and Marks GitHub raw. "
            "Check HF_HUB_OFFLINE / network."
        )

    # Balance
    rng = np.random.default_rng(seed)
    pos = [r for r in rows if r["label"] == 1]
    neg = [r for r in rows if r["label"] == 0]
    n_each = min(len(pos), len(neg), max_samples // 2)
    print(f"  Source {source_tag}: {len(pos)} true, {len(neg)} false. "
          f"Sampling {n_each} each (total {2*n_each}).")
    rng.shuffle(pos)
    rng.shuffle(neg)
    balanced = pos[:n_each] + neg[:n_each]
    rng.shuffle(balanced)
    for i, r in enumerate(balanced):
        r["stmt_id"] = i
    return balanced, source_tag


# ─────────────────────────────────────────────────────────────────────
# Tokenize + last-token residual capture across all layers
# ─────────────────────────────────────────────────────────────────────

def _statements_to_samples(statements: List[Dict[str, Any]],
                           tokenizer) -> List[Dict[str, Any]]:
    """Convert truth statements to the dict-shape our extraction helper expects.
    anchor = last non-pad token index."""
    out: List[Dict[str, Any]] = []
    for r in statements:
        text = r["statement"].strip()
        if not text:
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) < 1:
            continue
        out.append({
            "pair_id": r["stmt_id"],
            "sentence": text,
            "phrase": "",
            "token_ids": ids,
            "anchor": len(ids) - 1,
            "label": int(r["label"]),
        })
    return out


def _cv_probe_no_groups(
    X: np.ndarray, labels: np.ndarray, n_splits: int = 5, seed: int = 42,
) -> Tuple[np.ndarray, float]:
    """Stratified k-fold CV LR probe — each statement is its own 'group', so
    plain stratified folds are the right design.  Returns (unit coef mean,
    mean AUROC across folds)."""
    if labels.sum() < 2 or (1 - labels).sum() < 2:
        return np.zeros(X.shape[1]), float("nan")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    aurocs, coefs = [], []
    for tr, te in skf.split(X, labels):
        if labels[tr].sum() < 2 or (1 - labels[tr]).sum() < 2:
            continue
        clf = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")
        clf.fit(X[tr], labels[tr])
        prob = clf.predict_proba(X[te])[:, 1]
        aurocs.append(roc_auc_score(labels[te], prob))
        coefs.append(clf.coef_[0])
    if not coefs:
        return np.zeros(X.shape[1]), float("nan")
    v = np.mean(coefs, axis=0)
    v = v / (np.linalg.norm(v) + 1e-12)
    return v, float(np.mean(aurocs))


def train_truth_probes_all_layers(
    model, samples: List[Dict[str, Any]], layers: List[int], device: str,
) -> Tuple[Dict[int, np.ndarray], pd.DataFrame]:
    """Single forward pass per sample; train per-layer binary truth probe with
    stratified CV.  Returns (directions_by_layer, per-layer dataframe)."""
    print(f"  Extracting last-token residuals for {len(samples)} statements × "
          f"{len(layers)} layers...")
    per_layer_X = _extract_all_layer_residuals(model, samples, layers, device)
    labels = np.array([s["label"] for s in samples], dtype=np.int64)

    directions: Dict[int, np.ndarray] = {}
    rows: List[Dict[str, Any]] = []
    for L in layers:
        v, auroc = _cv_probe_no_groups(per_layer_X[L], labels)
        directions[L] = v
        rows.append({
            "layer": int(L),
            "truth_probe_auroc": float(auroc),
            "n_samples": int(len(samples)),
        })
        print(f"    L{L}: truth AUROC = {auroc:.3f}", flush=True)
    return directions, pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def _existing_freq_summary(model_dir_name: str) -> Optional[Dict[str, Any]]:
    """Search known freq-steering output roots for a matching summary.json."""
    candidates = [
        Path(f"/data/ani/mechinterp/runs/frequency_steering/{model_dir_name}/summary.json"),
        Path(f"/data/ani/mechinterp/runs/steering_metrics/{model_dir_name}/summary.json"),
    ]
    for p in candidates:
        if p.exists():
            try:
                with open(p) as f:
                    return {"path": str(p), **json.load(f)}
            except Exception as e:
                print(f"  Failed to read {p}: {e}")
    return None


def _model_dir_name_from_id(model_id: str) -> str:
    m = model_id.lower()
    if "olmo-7b" in m:
        return "olmo-7b"
    if "llama-3.1-8b" in m or "llama3.1-8b" in m or "meta-llama/llama-3.1-8b" in m:
        return "llama-3.1-8b"
    if "pythia-6.9b" in m:
        return "pythia-6.9b"
    return m.split("/")[-1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--dataset-dir", required=True,
                        help="ScaleJSD filtered JSONL dir — prompts for the "
                             "steering test (matched to freq-steering).")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--truth-dataset", default="auto",
                        choices=["auto", "azaria-mitchell",
                                 "azaria-mitchell-diverse",
                                 "cf-saplma", "marks-github"])
    parser.add_argument("--max-truth-samples", type=int, default=1000)
    parser.add_argument("--alphas",
                        default="-2,-1,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1,2")
    parser.add_argument("--steer-layer", type=int, default=None,
                        help="Override peak-AUROC layer selection.")
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--n-prompts", type=int, default=200)
    parser.add_argument("--corpus-file", default=None)
    parser.add_argument("--unigram-counts-file", default=None)
    parser.add_argument("--deterministic", action="store_true",
                        help="Use greedy decoding in steering test.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", default="auto",
                        choices=["auto", "float32", "bfloat16", "float16"])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.makedirs(args.output_dir, exist_ok=True)
    out_dir = Path(args.output_dir)
    alphas = [float(x) for x in args.alphas.split(",")]

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ── Model + tokenizer ─────────────────────────────────────────────
    if args.dtype == "auto":
        dtype = _select_dtype(args.model, args.device)
    else:
        dtype = {"float32": torch.float32,
                 "bfloat16": torch.bfloat16,
                 "float16": torch.float16}[args.dtype]
    print(f"Loading model {args.model} (revision={args.revision}, dtype={dtype})")

    tok_kw: Dict[str, Any] = {}
    if args.revision != "main":
        tok_kw["revision"] = args.revision
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if hf_token:
        tok_kw["token"] = hf_token
    tokenizer = AutoTokenizer.from_pretrained(args.model, **tok_kw)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kw: Dict[str, Any] = {"torch_dtype": dtype}
    if args.revision != "main":
        load_kw["revision"] = args.revision
    if hf_token:
        load_kw["token"] = hf_token
    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kw).to(args.device)
    model.eval()

    try:
        n_hidden = int(model.config.num_hidden_layers)
        all_layers = list(range(n_hidden))
    except Exception:
        all_layers = _infer_layers(args.model)

    # ── Load truth statements ─────────────────────────────────────────
    print("\n=== Loading truth statements ===")
    statements, truth_source = load_truth_statements(
        choice=args.truth_dataset,
        max_samples=args.max_truth_samples,
        seed=args.seed,
    )
    samples = _statements_to_samples(statements, tokenizer)
    print(f"  Tokenized {len(samples)} statements; source={truth_source}")

    # ── Train truth probe per layer ───────────────────────────────────
    print("\n=== Training truth probes per layer ===")
    directions, probe_df = train_truth_probes_all_layers(
        model, samples, all_layers, args.device,
    )
    probe_df.to_csv(out_dir / "truth_probe_per_layer.csv", index=False)

    # ── Select peak layer ─────────────────────────────────────────────
    if args.steer_layer is not None:
        best_layer = int(args.steer_layer)
    else:
        valid = probe_df.dropna(subset=["truth_probe_auroc"])
        if len(valid) == 0:
            raise RuntimeError("No valid probe AUROCs; cannot choose steer layer.")
        best_layer = int(valid.loc[valid["truth_probe_auroc"].idxmax(), "layer"])
    best_auroc = float(probe_df.loc[probe_df["layer"] == best_layer,
                                    "truth_probe_auroc"].iloc[0])
    direction = directions[best_layer].astype(np.float32)
    print(f"\n  Peak truth layer: L{best_layer} (AUROC={best_auroc:.3f})")

    # ── Token log-frequencies (for steering-test metric columns) ──────
    print("\n=== Computing token log-frequencies (for matched freq-mass columns) ===")
    token_log_freqs = compute_token_log_frequencies(
        model, tokenizer, args.device,
        corpus_file=args.corpus_file,
        unigram_counts_file=args.unigram_counts_file,
    )

    # ── Build ScaleJSD prompts (match frequency steering distribution) ─
    print("\n=== Building prompts from ScaleJSD (match freq pipeline) ===")
    all_sj_samples: List[Dict[str, Any]] = []
    for ds_name, fname in DATASET_FILES.items():
        ds_path = Path(args.dataset_dir) / fname
        if not ds_path.exists():
            continue
        pairs = load_synonym_pairs(str(ds_path))
        sj_samples = pairs_to_samples(pairs, tokenizer)
        for s in sj_samples:
            s["dataset"] = ds_name
        all_sj_samples.extend(sj_samples)
    print(f"  Total ScaleJSD samples available: {len(all_sj_samples)}")

    prompts: List[str] = []
    seen: set = set()
    for s in all_sj_samples:
        prefix = _build_prompt_from_sample(s, tokenizer)
        if prefix is None or prefix in seen:
            continue
        prompts.append(prefix)
        seen.add(prefix)
        if len(prompts) >= args.n_prompts:
            break
    print(f"  Built {len(prompts)} prompts (truncated before target ngram)")

    # ── Steering test ─────────────────────────────────────────────────
    print(f"\n=== Truth-direction steering test at L{best_layer} "
          f"({len(prompts)} prompts) ===")
    rst_kwargs: Dict[str, Any] = dict(
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
    )
    if _run_steering_test_supports_deterministic():
        rst_kwargs["deterministic"] = args.deterministic
    steering_results, gen_samples = run_steering_test(
        model, tokenizer, direction, best_layer, alphas,
        prompts, token_log_freqs, args.device,
        **rst_kwargs,
    )

    steering_df = pd.DataFrame(steering_results)
    steering_df.to_csv(out_dir / "truth_steering_test.csv", index=False)
    pd.DataFrame(gen_samples).to_csv(out_dir / "sample_generations.csv", index=False)

    # ── Comparison summary ────────────────────────────────────────────
    def _kl_at(alpha_target: float) -> Optional[float]:
        row = steering_df[np.isclose(steering_df["alpha"], alpha_target)]
        if len(row):
            return float(row["mean_kl"].iloc[0])
        return None

    truth_kl_a1 = _kl_at(1.0)
    truth_kl_a2 = _kl_at(2.0)

    comparison: Dict[str, Any] = {
        "model": args.model,
        "revision": args.revision,
        "dtype": str(dtype),
        "truth_source": truth_source,
        "truth_n_statements": len(samples),
        "truth_best_layer": int(best_layer),
        "truth_probe_auroc": float(best_auroc),
        "truth_kl_at_alpha_1": truth_kl_a1,
        "truth_kl_at_alpha_2": truth_kl_a2,
        "alphas": alphas,
        "n_prompts": len(prompts),
        "deterministic": bool(args.deterministic),
    }

    # Try to read frequency-steering summary and compute ratio.
    model_dir_name = _model_dir_name_from_id(args.model)
    freq_summary = _existing_freq_summary(model_dir_name)
    if freq_summary is not None:
        comparison["frequency_summary_path"] = freq_summary.get("path")
        comparison["frequency_probe_auroc"] = freq_summary.get("probe_auroc_best_layer")
        comparison["frequency_best_layer"] = freq_summary.get("best_layer")
        # Frequency summary includes only PASS metrics — best effort to pull KL@α=1 from
        # the adjacent steering_test.csv if we can infer its location.
        freq_steering_csv = (
            Path(freq_summary["path"]).parent / "steering_test.csv"
        )
        if freq_steering_csv.exists():
            try:
                fdf = pd.read_csv(freq_steering_csv)
                row1 = fdf[np.isclose(fdf["alpha"], 1.0)]
                row2 = fdf[np.isclose(fdf["alpha"], 2.0)]
                freq_kl_a1 = float(row1["mean_kl"].iloc[0]) if len(row1) else None
                freq_kl_a2 = float(row2["mean_kl"].iloc[0]) if len(row2) else None
                comparison["frequency_kl_at_alpha_1"] = freq_kl_a1
                comparison["frequency_kl_at_alpha_2"] = freq_kl_a2
                if freq_kl_a1 and truth_kl_a1 is not None and freq_kl_a1 > 1e-12:
                    comparison["causal_effect_ratio_alpha_1"] = (
                        float(truth_kl_a1) / float(freq_kl_a1)
                    )
                if freq_kl_a2 and truth_kl_a2 is not None and freq_kl_a2 > 1e-12:
                    comparison["causal_effect_ratio_alpha_2"] = (
                        float(truth_kl_a2) / float(freq_kl_a2)
                    )
            except Exception as e:
                print(f"  Could not parse freq steering csv: {e}")
    else:
        comparison["frequency_summary_path"] = None
        comparison["note"] = ("No frequency-steering summary.json found on PVC — "
                              "causal_effect_ratio not computed.")

    with open(out_dir / "comparison_summary.json", "w") as f:
        json.dump(comparison, f, indent=2)

    # ── Console summary ───────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("  TRUTH STEERING COMPARISON — Summary")
    print(f"{'='*72}")
    cols = ["alpha", "layer", "mean_kl", "mean_high_freq_mass",
            "mean_low_freq_mass", "median_perplexity"]
    cols_present = [c for c in cols if c in steering_df.columns]
    print(steering_df[cols_present].to_string(index=False))

    print(f"\n  Truth probe AUROC @ L{best_layer} = {best_auroc:.3f}")
    if truth_kl_a1 is not None:
        print(f"  Truth KL at α=1: {truth_kl_a1:.4f}")
    if truth_kl_a2 is not None:
        print(f"  Truth KL at α=2: {truth_kl_a2:.4f}")
    if "causal_effect_ratio_alpha_1" in comparison:
        print(f"  Causal effect ratio (truth/freq) α=1: "
              f"{comparison['causal_effect_ratio_alpha_1']:.2f}×")
    if "causal_effect_ratio_alpha_2" in comparison:
        print(f"  Causal effect ratio (truth/freq) α=2: "
              f"{comparison['causal_effect_ratio_alpha_2']:.2f}×")
    print(f"\n  Results saved to {out_dir}")


def _run_steering_test_supports_deterministic() -> bool:
    """Detect whether run_steering_test accepts a 'deterministic' kwarg.
    Back-compat shim — the current frequency_steering.py does NOT take that
    arg, so if we pass it we get a TypeError.  We instead monkey-patch the
    generate call path below."""
    import inspect
    sig = inspect.signature(run_steering_test)
    return "deterministic" in sig.parameters


# ── Deterministic decoding monkey-patch ──────────────────────────────
# run_steering_test currently uses do_sample=True, top_p=0.9 for generation,
# which is the fair match to the frequency baseline.  We want deterministic
# generations for the truth comparison so α=0 vs α>0 differences aren't noise.
# We override at runtime via a torch.Tensor.generate wrapper when --deterministic
# is passed.  This keeps the script portable and reuses the helper unchanged.

if __name__ == "__main__":
    # If --deterministic is passed and run_steering_test doesn't natively
    # support it, wrap AutoModelForCausalLM.generate to force greedy.
    _saw_det = any(a == "--deterministic" for a in os.sys.argv)
    if _saw_det and not _run_steering_test_supports_deterministic():
        import transformers
        _orig_generate = transformers.PreTrainedModel.generate

        def _greedy_generate(self, *g_args, **g_kwargs):
            g_kwargs["do_sample"] = False
            g_kwargs.pop("temperature", None)
            g_kwargs.pop("top_p", None)
            return _orig_generate(self, *g_args, **g_kwargs)

        transformers.PreTrainedModel.generate = _greedy_generate  # type: ignore[assignment]
        print("[deterministic mode] patched model.generate → greedy")

    main()
