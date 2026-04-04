#!/usr/bin/env python3
"""
Behavioral Convergence Experiment

Measures how similarly the model treats synonym pairs at the OUTPUT level
across training checkpoints. If internal metrics (JSD, polytope density)
converge before output distributions do, they are leading indicators of
capability emergence.

For each synonym pair at each checkpoint:
  1. Feed high-freq and low-freq sentences through the model
  2. Extract logit distributions at the anchor token position
  3. Compute divergence between the two output distributions

Metrics:
  - Output JSD: Jensen-Shannon divergence between softmax distributions
  - Output KL (H→L, L→H): asymmetric KL divergences
  - Top-k overlap: fraction of top-k predicted tokens shared
  - Rank correlation: Spearman r between full logit vectors

Usage:
    # Single model, all checkpoints
    python -m scripts.metrics.behavioral_convergence \
        --all-datasets --dataset-dir ~/dev/personal/mechinterp/scaleJSD/dataset/legacy/filtered \
        --model EleutherAI/pythia-70m-deduped \
        --revisions step3000,step13000,...,step143000 \
        --output-dir results/behavioral_convergence/pythia-70m/

    # Quick validation
    python -m scripts.metrics.behavioral_convergence \
        --dataset emotion_ngrams_dedup_filtered.jsonl \
        --model EleutherAI/pythia-70m-deduped \
        --revisions step143000 \
        --output-dir /tmp/test --validate

Outputs:
    pair_divergence.csv  — per (checkpoint, pair_id): JSD, KL, top-k overlap, rank corr
    checkpoint_summary.csv — per checkpoint: mean/std of all metrics
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

EPS = 1e-12


def _model_torch_dtype(device: str) -> torch.dtype:
    """Avoid half precision on CPU, where many HF models will fail to run."""
    return torch.float16 if device in {"cuda", "mps"} else torch.float32


# ─────────────────────────────────────────────────────────────────────────────
# Data loading (shared with coverage_affinity_experiment.py)
# ─────────────────────────────────────────────────────────────────────────────

def load_synonym_pairs(path: str) -> List[Dict[str, Any]]:
    """Load synonym pairs from ScaleJSD JSONL file."""
    pairs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    print(f"Loaded {len(pairs)} synonym pairs from {Path(path).name}")
    return pairs


def pairs_to_samples(
    pairs: List[Dict[str, Any]],
    tokenizer,
) -> List[Dict[str, Any]]:
    """Convert synonym pairs into (high, low) sample dicts with token positions."""
    DEFAULT_TEMPLATE = "The word {ngram} means"
    samples = []

    for i, pair in enumerate(pairs):
        pair_id = pair.get("synonym_pair_seed", pair.get("id", f"pair_{i}"))
        template = pair.get("sentence_template")

        for ngram, freq_cat, count_key in [
            (pair["high_freq_ngram"], "high_freq", "high_freq_count"),
            (pair["low_freq_ngram"], "low_freq", "low_freq_count"),
        ]:
            if template and "[TERM]" in template:
                text = template.replace("[TERM]", ngram)
            else:
                text = DEFAULT_TEMPLATE.format(ngram=ngram)

            token_ids = tokenizer.encode(text, add_special_tokens=False)

            # Find anchor: last token of the ngram
            if template and "[TERM]" in template:
                pos = template.find("[TERM]")
                suffix = template[pos + 6:]
                if suffix:
                    suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
                    anchor = len(token_ids) - len(suffix_ids) - 1
                else:
                    anchor = len(token_ids) - 1
            else:
                suffix_ids = tokenizer.encode(" means", add_special_tokens=False)
                anchor = len(token_ids) - len(suffix_ids) - 1

            anchor = max(0, min(anchor, len(token_ids) - 1))

            raw_count = pair.get(count_key, 1)

            samples.append({
                "pair_id": pair_id,
                "phrase": ngram,
                "sentence": text,
                "frequency_category": freq_cat,
                "token_ids": token_ids,
                "anchor": anchor,
                "raw_frequency": raw_count,
                "category": pair.get("category", "unknown"),
            })

    n_high = sum(1 for s in samples if s["frequency_category"] == "high_freq")
    n_low = len(samples) - n_high
    print(f"Created {len(samples)} samples ({n_high} high-freq, {n_low} low-freq)")
    return samples


# ─────────────────────────────────────────────────────────────────────────────
# Output distribution extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_output_logits(
    model: torch.nn.Module,
    samples: List[Dict[str, Any]],
    device: str,
) -> np.ndarray:
    """Extract logit vectors at anchor positions for all samples.

    Returns np.ndarray [N_samples, vocab_size] of logits (float32).
    """
    all_logits = []

    for idx, sample in enumerate(samples):
        if idx % 50 == 0:
            print(f"  [{idx + 1}/{len(samples)}] {sample['phrase']}")

        with torch.no_grad():
            inputs = torch.tensor([sample["token_ids"]], device=device)
            outputs = model(input_ids=inputs)
            logits = outputs.logits  # [1, seq_len, vocab_size]

        pos = min(sample["anchor"], logits.shape[1] - 1)
        logit_vec = logits[0, pos, :].float().cpu().numpy()
        all_logits.append(logit_vec)

    return np.stack(all_logits)  # [N, V]


# ─────────────────────────────────────────────────────────────────────────────
# Divergence metrics
# ─────────────────────────────────────────────────────────────────────────────

def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    x = logits - logits.max()
    e = np.exp(x)
    return e / (e.sum() + EPS)


def _kl_div(p: np.ndarray, q: np.ndarray) -> float:
    """KL(P || Q) with epsilon smoothing."""
    p = np.maximum(p, EPS)
    q = np.maximum(q, EPS)
    return float(np.sum(p * (np.log(p) - np.log(q))))


def _jsd(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence."""
    m = 0.5 * (p + q)
    return 0.5 * _kl_div(p, m) + 0.5 * _kl_div(q, m)


def _top_k_overlap(logits_a: np.ndarray, logits_b: np.ndarray, k: int = 50) -> float:
    """Fraction of top-k tokens shared between two logit vectors."""
    top_a = set(np.argsort(logits_a)[-k:])
    top_b = set(np.argsort(logits_b)[-k:])
    return len(top_a & top_b) / k


def compute_pair_divergences(
    logits: np.ndarray,
    samples: List[Dict[str, Any]],
    top_k_values: List[int] = [10, 50, 100],
) -> pd.DataFrame:
    """Compute output distribution divergences for matched synonym pairs.

    Returns DataFrame with one row per pair.
    """
    # Group samples by pair_id
    by_pair: Dict[str, Dict[str, int]] = {}
    for i, s in enumerate(samples):
        by_pair.setdefault(s["pair_id"], {})[s["frequency_category"]] = i

    rows = []
    for pid, indices in sorted(by_pair.items()):
        if "high_freq" not in indices or "low_freq" not in indices:
            continue
        i_h, i_l = indices["high_freq"], indices["low_freq"]

        logits_h = logits[i_h]
        logits_l = logits[i_l]

        p_h = _softmax(logits_h)
        p_l = _softmax(logits_l)

        row = {
            "pair_id": pid,
            "phrase_high": samples[i_h]["phrase"],
            "phrase_low": samples[i_l]["phrase"],
            "category": samples[i_h]["category"],
            "output_jsd": _jsd(p_h, p_l),
            "kl_high_to_low": _kl_div(p_h, p_l),
            "kl_low_to_high": _kl_div(p_l, p_h),
        }

        # Top-k overlap at different k values
        for k in top_k_values:
            row[f"top{k}_overlap"] = _top_k_overlap(logits_h, logits_l, k=k)

        # Rank correlation of full logit vectors
        r, p = spearmanr(logits_h, logits_l)
        row["rank_correlation"] = float(r)
        row["rank_corr_pvalue"] = float(p)

        rows.append(row)

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_single_checkpoint(
    *,
    samples: List[Dict[str, Any]],
    model_id: str,
    revision: str,
    device: str,
) -> pd.DataFrame:
    """Run behavioral convergence for a single checkpoint.

    Returns pair_divergence DataFrame with checkpoint column.
    """
    from transformers import AutoModelForCausalLM

    print(f"\nLoading {model_id} @ {revision} on {device}")
    load_kw: Dict[str, Any] = {"torch_dtype": _model_torch_dtype(device)}
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if hf_token:
        load_kw["token"] = hf_token
    if revision != "main":
        load_kw["revision"] = revision

    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kw).to(device)
    model.eval()

    print("Extracting output logits ...")
    logits = extract_output_logits(model, samples, device)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"  Logits shape: {logits.shape}")

    pair_div = compute_pair_divergences(logits, samples)
    pair_div["checkpoint"] = revision

    # Print summary
    print(f"  Mean output JSD:     {pair_div['output_jsd'].mean():.6f}")
    print(f"  Mean top50 overlap:  {pair_div['top50_overlap'].mean():.3f}")
    print(f"  Mean rank corr:      {pair_div['rank_correlation'].mean():.3f}")

    return pair_div


def run_experiment(
    dataset_path: str,
    model_id: str,
    revisions: List[str],
    output_dir: str,
    device: Optional[str] = None,
    validate_only: bool = False,
):
    """Run behavioral convergence across checkpoints."""
    from transformers import AutoTokenizer

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )

    # Load data
    pairs = load_synonym_pairs(dataset_path)

    tok_kw: Dict[str, Any] = {}
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if hf_token:
        tok_kw["token"] = hf_token
    rev0 = revisions[0]
    if rev0 != "main":
        tok_kw["revision"] = rev0
    tokenizer = AutoTokenizer.from_pretrained(model_id, **tok_kw)

    samples = pairs_to_samples(pairs, tokenizer)

    # Validate mode
    if validate_only:
        print("\n=== VALIDATION MODE ===")
        test_samples = samples[:min(10, len(samples))]
        from transformers import AutoModelForCausalLM
        load_kw: Dict[str, Any] = {"torch_dtype": _model_torch_dtype(device)}
        if hf_token:
            load_kw["token"] = hf_token
        if rev0 != "main":
            load_kw["revision"] = rev0
        model = AutoModelForCausalLM.from_pretrained(model_id, **load_kw).to(device)
        model.eval()

        logits = extract_output_logits(model, test_samples, device)
        del model

        print(f"  Logits shape: {logits.shape}")
        pair_div = compute_pair_divergences(logits, test_samples)
        print(f"  Pairs computed: {len(pair_div)}")
        if len(pair_div) > 0:
            print(f"  Output JSD range: [{pair_div['output_jsd'].min():.6f}, {pair_div['output_jsd'].max():.6f}]")
            print(f"  Top50 overlap range: [{pair_div['top50_overlap'].min():.3f}, {pair_div['top50_overlap'].max():.3f}]")
        print("\n  VALIDATION PASSED")
        return

    # Run per checkpoint
    all_pairs = []
    for revision in revisions:
        print(f"\n{'=' * 60}")
        print(f"  Checkpoint: {revision}")
        print(f"{'=' * 60}")

        pair_div = run_single_checkpoint(
            samples=samples,
            model_id=model_id,
            revision=revision,
            device=device,
        )
        all_pairs.append(pair_div)

    # Save
    df_pairs = pd.concat(all_pairs, ignore_index=True)
    df_pairs.to_csv(out / "pair_divergence.csv", index=False)

    # Checkpoint summary
    summary_rows = []
    for ckpt, g in df_pairs.groupby("checkpoint"):
        row = {"checkpoint": ckpt, "n_pairs": len(g)}
        for col in ["output_jsd", "kl_high_to_low", "kl_low_to_high",
                     "top10_overlap", "top50_overlap", "top100_overlap",
                     "rank_correlation"]:
            if col in g.columns:
                row[f"{col}_mean"] = float(g[col].mean())
                row[f"{col}_std"] = float(g[col].std())
        # Per-category breakdown
        for cat, cg in g.groupby("category"):
            row[f"output_jsd_mean_{cat}"] = float(cg["output_jsd"].mean())
        summary_rows.append(row)

    df_summary = pd.DataFrame(summary_rows)
    # Sort by checkpoint step number
    df_summary["_step"] = df_summary["checkpoint"].str.extract(r"step(\d+)").astype(float)
    df_summary = df_summary.sort_values("_step").drop(columns=["_step"])
    df_summary.to_csv(out / "checkpoint_summary.csv", index=False)

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"  RESULTS SAVED TO {out}")
    print(f"{'=' * 60}")
    print(f"  pair_divergence.csv    : {len(df_pairs):,} rows")
    print(f"  checkpoint_summary.csv : {len(df_summary):,} rows")

    print(f"\n  Checkpoint progression (output JSD):")
    for _, row in df_summary.iterrows():
        print(f"    {row['checkpoint']:>12s}  "
              f"JSD={row['output_jsd_mean']:.6f}  "
              f"top50={row['top50_overlap_mean']:.3f}  "
              f"rank_r={row['rank_correlation_mean']:.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

DATASET_NAMES = ["emotion", "medical", "legal", "scientific", "verb"]
DATASET_PATTERNS = [
    "{name}_ngrams_dedup_filtered.jsonl",
    "{name}_ngrams_filtered_pairs.jsonl",
    "{name}_filtered_pairs.jsonl",
    "{name}_filtered.jsonl",
]


def _find_dataset(dataset_dir: Path, name: str) -> Optional[Path]:
    for pat in DATASET_PATTERNS:
        p = dataset_dir / pat.format(name=name)
        if p.exists():
            return p
    return None


def main():
    p = argparse.ArgumentParser(
        description="Behavioral Convergence Experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    p.add_argument("--dataset", type=str, default=None)
    p.add_argument("--all-datasets", action="store_true")
    p.add_argument("--dataset-dir", type=str, default=None)
    p.add_argument("--model", default="EleutherAI/pythia-70m-deduped")
    p.add_argument("--revisions", default="step143000",
                    help="Comma-separated checkpoint revisions")
    p.add_argument("--device", default=None)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--validate", action="store_true")

    args = p.parse_args()

    revisions = [r.strip() for r in args.revisions.split(",")]

    # Resolve datasets
    if args.all_datasets:
        if args.dataset_dir is None:
            p.error("--dataset-dir is required with --all-datasets")
        dataset_dir = Path(args.dataset_dir)
        datasets = []
        for name in DATASET_NAMES:
            path = _find_dataset(dataset_dir, name)
            if path:
                datasets.append((name, str(path)))
            else:
                print(f"SKIP: {name} not found in {dataset_dir}")
        if not datasets:
            p.error(f"No datasets found in {dataset_dir}")
    elif args.dataset:
        name = Path(args.dataset).stem.split("_")[0]
        datasets = [(name, args.dataset)]
    else:
        p.error("Provide --dataset or --all-datasets")

    for name, dataset_path in datasets:
        if args.all_datasets:
            out_dir = str(Path(args.output_dir) / name)
        else:
            out_dir = args.output_dir

        print(f"\n{'#' * 60}")
        print(f"  Dataset: {name} ({Path(dataset_path).name})")
        print(f"  Output:  {out_dir}")
        print(f"{'#' * 60}")

        run_experiment(
            dataset_path=dataset_path,
            model_id=args.model,
            revisions=revisions,
            output_dir=out_dir,
            device=args.device,
            validate_only=args.validate,
        )


if __name__ == "__main__":
    main()
