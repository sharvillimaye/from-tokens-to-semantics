#!/usr/bin/env python3
"""
Idea 2 — Frequency-sensitive task KL.

The cumulative-ablation experiment showed "small KL" on ScaleJSD synonym-pair
prompts, which are frequency-NEUTRAL by design (the only difference between
high/low-freq counterparts is the frequency direction itself, and synonyms
mostly have similar next-token continuations). That weakness of the KL signal
is actually a FEATURE: if the frequency direction is a specific, non-task
feature, we expect KL to stay small on frequency-neutral prompts and jump
dramatically on prompts where frequency MATTERS (rare-entity QA, rare-word
contexts, etc.).

For a fixed model and fixed probe-derived frequency direction v at layer L*,
we run CUMULATIVE rank-1 ablation (project out v at every layer) and measure
KL(clean || ablated) at the last position on four prompt distributions:

  (a) rare-entity QA (PopQA tail, popularity < 5) — frequency MATTERS
  (b) rare-word contexts (ScaleJSD low-freq anchor sentences, truncated)
  (c) common-word contexts (ScaleJSD high-freq anchor sentences, truncated)
  (d) ScaleJSD baseline (frequency-neutral, reference)

Prediction: median KL on (a), (b) is 10x–100x larger than on (d).

Saves:
  kl_per_distribution.csv — per-distribution aggregate stats
  per_prompt.csv         — per-prompt KL + top-5 token probs (clean vs ablated)
  summary.json           — overall stats + effect ratios vs ScaleJSD

Usage:
    python -m scripts.interventions.freq_sensitive_kl \\
        --model allenai/OLMo-7B-hf --revision main \\
        --dataset-dir /data/.../scaleJSD/dataset/legacy/filtered \\
        --output-dir /data/ani/mechinterp/runs/freq_sensitive_kl/olmo-7b \\
        --device cuda --dtype auto
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

try:
    from scripts.interventions.frequency_steering import (
        ResidualCapture,
        _build_prompt_from_sample,
        _select_dtype,
        find_transformer_layers,
        train_probe_at_layer,
    )
    from scripts.metrics.coverage_affinity_experiment import (
        _infer_layers,
        load_synonym_pairs,
        pairs_to_samples,
    )
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from scripts.interventions.frequency_steering import (  # noqa: E402
        ResidualCapture,
        _build_prompt_from_sample,
        _select_dtype,
        find_transformer_layers,
        train_probe_at_layer,
    )
    from scripts.metrics.coverage_affinity_experiment import (  # noqa: E402
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
#  Cumulative rank-1 ablation hook
# ─────────────────────────────────────────────────────────────────────

class CumulativeAblationHook:
    """Project residual stream onto complement of direction v at every
    layer in `layers`. Rank-1 projection:
        x_ablated = x - (x·v) v
    where v is unit-normalized.
    """

    def __init__(self, model, direction: torch.Tensor, layers: List[int]):
        self.hooks: list = []
        tf = find_transformer_layers(model)
        for L in layers:
            def make(idx):
                def hook(_mod, _inp, output):
                    v = direction.to(device=output[0].device, dtype=output[0].dtype) \
                        if isinstance(output, tuple) \
                        else direction.to(device=output.device, dtype=output.dtype)
                    if isinstance(output, tuple):
                        x = output[0]
                        rest = output[1:]
                    else:
                        x = output
                        rest = None
                    coeff = (x * v).sum(dim=-1, keepdim=True)  # [B, P, 1]
                    x_ablated = x - coeff * v
                    if rest is None:
                        return x_ablated
                    return (x_ablated, *rest)
                return hook
            self.hooks.append(tf[L].register_forward_hook(make(L)))

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


# ─────────────────────────────────────────────────────────────────────
#  Prompt distributions
# ─────────────────────────────────────────────────────────────────────

def _load_popqa_tail(n_prompts: int, max_popularity: int = 5) -> List[str]:
    """Load rare-entity QA prompts from PopQA tail.

    PopQA columns: id, subj, prop, obj, subj_id, prop_id, obj_id, s_aliases,
    o_aliases, s_uri, o_uri, s_wiki_title, o_wiki_title, s_pop, o_pop,
    question, possible_answers.

    "Tail" = subject popularity < max_popularity.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("  datasets library not available — skipping PopQA")
        return []

    try:
        ds = load_dataset("akariasai/PopQA", split="test")
    except Exception as e:
        print(f"  PopQA load failed: {e}")
        return []

    prompts: List[str] = []
    for row in ds:
        pop = row.get("s_pop")
        if pop is None:
            continue
        try:
            pop_f = float(pop)
        except (TypeError, ValueError):
            continue
        if pop_f >= max_popularity:
            continue
        question = row.get("question")
        if not question:
            continue
        prompts.append(f"Q: {question.strip()} A:")
        if len(prompts) >= n_prompts:
            break
    return prompts


def _scalejsd_anchor_prompts(
    samples: List[Dict[str, Any]],
    tokenizer,
    freq_category: str,
    n_prompts: int,
    rng: random.Random,
) -> List[str]:
    """Build prompts from ScaleJSD samples of a given frequency category,
    truncating before the target phrase (so the model is predicting
    a {rare,common}-word position)."""
    pool = [s for s in samples if s["frequency_category"] == freq_category]
    rng.shuffle(pool)
    prompts: List[str] = []
    seen = set()
    for s in pool:
        prefix = _build_prompt_from_sample(s, tokenizer)
        if prefix is None or prefix in seen:
            continue
        prompts.append(prefix)
        seen.add(prefix)
        if len(prompts) >= n_prompts:
            break
    return prompts


def _scalejsd_full_prompts(
    samples: List[Dict[str, Any]],
    tokenizer,
    n_prompts: int,
    rng: random.Random,
) -> List[str]:
    """ScaleJSD baseline: full-template prompts truncated before the target
    ngram, mixing high and low freq categories (frequency-neutral by
    virtue of the synonym-pair design)."""
    pool = list(samples)
    rng.shuffle(pool)
    prompts: List[str] = []
    seen = set()
    for s in pool:
        prefix = _build_prompt_from_sample(s, tokenizer)
        if prefix is None or prefix in seen:
            continue
        prompts.append(prefix)
        seen.add(prefix)
        if len(prompts) >= n_prompts:
            break
    return prompts


# ─────────────────────────────────────────────────────────────────────
#  KL measurement
# ─────────────────────────────────────────────────────────────────────

def _encode_prompt(tokenizer, text: str, device: str, max_len: int = 256) -> torch.Tensor:
    ids = tokenizer.encode(text, add_special_tokens=True, return_tensors="pt")
    if ids.shape[1] > max_len:
        ids = ids[:, -max_len:]
    return ids.to(device)


def _top_k_json(probs: np.ndarray, tokenizer, k: int = 5) -> str:
    idx = np.argsort(-probs)[:k]
    items = []
    for i in idx:
        try:
            tok = tokenizer.decode([int(i)])
        except Exception:
            tok = f"<tok:{int(i)}>"
        items.append([tok, float(probs[int(i)])])
    return json.dumps(items, ensure_ascii=False)


def measure_kl_on_distribution(
    model,
    tokenizer,
    prompts: List[str],
    direction: torch.Tensor,
    steer_layers: List[int],
    device: str,
    max_len: int,
    prompt_source: str,
) -> List[Dict[str, Any]]:
    """For each prompt, compute clean and ablated logits at last position,
    measure KL(clean || ablated)."""
    vocab_size = int(model.get_output_embeddings().weight.shape[0])
    rows: List[Dict[str, Any]] = []

    for pi, prompt in enumerate(prompts):
        try:
            input_ids = _encode_prompt(tokenizer, prompt, device, max_len=max_len)
        except Exception as e:
            print(f"    [{prompt_source} #{pi}] encode failed: {e}", flush=True)
            continue
        if input_ids.shape[1] < 2:
            continue

        with torch.no_grad():
            clean_logits = model(input_ids=input_ids).logits[0, -1, :vocab_size].float().cpu()

        hook = CumulativeAblationHook(model, direction, steer_layers)
        try:
            with torch.no_grad():
                abl_logits = model(input_ids=input_ids).logits[0, -1, :vocab_size].float().cpu()
        finally:
            hook.remove()

        log_p = F.log_softmax(clean_logits, dim=-1)
        log_q = F.log_softmax(abl_logits, dim=-1)
        kl = float((log_p.exp() * (log_p - log_q)).sum())

        p_probs = log_p.exp().numpy()
        q_probs = log_q.exp().numpy()

        rows.append({
            "prompt_source": prompt_source,
            "prompt": prompt,
            "n_tokens": int(input_ids.shape[1]),
            "kl": kl,
            "top5_clean_json": _top_k_json(p_probs, tokenizer, k=5),
            "top5_ablated_json": _top_k_json(q_probs, tokenizer, k=5),
        })

        if (pi + 1) % 25 == 0:
            med_so_far = float(np.median([r["kl"] for r in rows]))
            print(f"    [{prompt_source}] {pi+1}/{len(prompts)} median_KL={med_so_far:.4f}",
                  flush=True)

    return rows


def _aggregate(rows: List[Dict[str, Any]], model_id: str) -> Dict[str, Any]:
    if not rows:
        return {
            "model": model_id,
            "prompt_source": None,
            "n_prompts": 0,
            "median_kl": float("nan"),
            "mean_kl": float("nan"),
            "p90_kl": float("nan"),
            "p99_kl": float("nan"),
        }
    kls = np.array([r["kl"] for r in rows], dtype=np.float64)
    return {
        "model": model_id,
        "prompt_source": rows[0]["prompt_source"],
        "n_prompts": int(len(rows)),
        "median_kl": float(np.median(kls)),
        "mean_kl": float(np.mean(kls)),
        "p90_kl": float(np.percentile(kls, 90)),
        "p99_kl": float(np.percentile(kls, 99)),
    }


# ─────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--dataset-dir", required=True,
                        help="ScaleJSD dataset dir (for probe training + baseline prompts)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--probe-direction", default=None,
                        help="Optional path to saved probe_direction.npy. "
                             "If set, skips probe training.")
    parser.add_argument("--steer-layer", type=int, default=None,
                        help="Layer to extract v at. Default: mid-stack.")
    parser.add_argument("--n-prompts-per-dist", type=int, default=200)
    parser.add_argument("--popqa-max-pop", type=int, default=5)
    parser.add_argument("--max-prompt-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="auto",
                        choices=["auto", "float32", "bfloat16", "float16"])
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.makedirs(args.output_dir, exist_ok=True)
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.dtype == "auto":
        dtype = _select_dtype(args.model, args.device)
    else:
        dtype = {"float32": torch.float32,
                 "bfloat16": torch.bfloat16,
                 "float16": torch.float16}[args.dtype]

    # ── Load model ─────────────────────────────────────────────────────
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
    mid_layer = all_layers[len(all_layers) // 2]

    # ── Load ScaleJSD samples ──────────────────────────────────────────
    all_samples: List[Dict[str, Any]] = []
    for ds_name, fname in DATASET_FILES.items():
        ds_path = Path(args.dataset_dir) / fname
        if not ds_path.exists():
            continue
        pairs = load_synonym_pairs(str(ds_path))
        samples = pairs_to_samples(pairs, tokenizer)
        for s in samples:
            s["dataset"] = ds_name
        all_samples.extend(samples)
    print(f"Total ScaleJSD samples: {len(all_samples)}")
    if not all_samples:
        raise RuntimeError(f"No ScaleJSD samples found under {args.dataset_dir}")

    # ── Determine steer layer ─────────────────────────────────────────
    steer_layer = args.steer_layer if args.steer_layer is not None else mid_layer
    print(f"Steer layer (v extraction): L{steer_layer}")

    # ── Extract frequency direction v ─────────────────────────────────
    if args.probe_direction and Path(args.probe_direction).exists():
        print(f"Loading saved probe direction from {args.probe_direction}")
        direction_np = np.load(args.probe_direction)
        direction_np = direction_np / (np.linalg.norm(direction_np) + 1e-12)
        probe_auroc = float("nan")
    else:
        print(f"Training frequency probe at L{steer_layer} on ScaleJSD...")
        direction_np, probe_auroc, _ = train_probe_at_layer(
            model, all_samples, steer_layer, args.device)
        np.save(Path(args.output_dir) / "probe_direction.npy", direction_np)
        print(f"  probe AUROC = {probe_auroc:.3f}")

    direction = torch.tensor(direction_np, dtype=torch.float32, device=args.device)

    # ── Build prompt distributions ────────────────────────────────────
    print("\n=== Building prompt distributions ===")
    prompt_sets: Dict[str, List[str]] = {}

    # (a) rare-entity QA (PopQA tail)
    print(f"  (a) PopQA tail (pop<{args.popqa_max_pop}, n={args.n_prompts_per_dist})...")
    popqa_prompts = _load_popqa_tail(args.n_prompts_per_dist, args.popqa_max_pop)
    prompt_sets["popqa_tail"] = popqa_prompts
    print(f"      got {len(popqa_prompts)} prompts")

    # (b) rare-word contexts
    print(f"  (b) ScaleJSD rare-word (low_freq) contexts (n={args.n_prompts_per_dist})...")
    rare_prompts = _scalejsd_anchor_prompts(
        all_samples, tokenizer, "low_freq",
        args.n_prompts_per_dist, rng)
    prompt_sets["scalejsd_rare_word"] = rare_prompts
    print(f"      got {len(rare_prompts)} prompts")

    # (c) common-word contexts
    print(f"  (c) ScaleJSD common-word (high_freq) contexts (n={args.n_prompts_per_dist})...")
    common_prompts = _scalejsd_anchor_prompts(
        all_samples, tokenizer, "high_freq",
        args.n_prompts_per_dist, rng)
    prompt_sets["scalejsd_common_word"] = common_prompts
    print(f"      got {len(common_prompts)} prompts")

    # (d) ScaleJSD baseline (frequency-neutral)
    print(f"  (d) ScaleJSD baseline (all) (n={args.n_prompts_per_dist})...")
    baseline_prompts = _scalejsd_full_prompts(
        all_samples, tokenizer, args.n_prompts_per_dist, rng)
    prompt_sets["scalejsd_baseline"] = baseline_prompts
    print(f"      got {len(baseline_prompts)} prompts")

    # ── Run KL measurement under CUMULATIVE ablation ──────────────────
    print(f"\n=== Cumulative rank-1 ablation across {len(all_layers)} layers ===")
    all_rows: List[Dict[str, Any]] = []
    per_dist_agg: List[Dict[str, Any]] = []
    for src, prompts in prompt_sets.items():
        if not prompts:
            print(f"  [{src}] skipping (0 prompts)")
            continue
        print(f"\n  Measuring KL on {src} ({len(prompts)} prompts)...")
        rows = measure_kl_on_distribution(
            model, tokenizer, prompts, direction, all_layers,
            args.device, args.max_prompt_tokens, src)
        for r in rows:
            r["model"] = args.model
        all_rows.extend(rows)
        agg = _aggregate(rows, args.model)
        per_dist_agg.append(agg)
        print(f"  [{src}] median_KL={agg['median_kl']:.4f} "
              f"mean={agg['mean_kl']:.4f} "
              f"p90={agg['p90_kl']:.4f} p99={agg['p99_kl']:.4f}")

    # ── Save outputs ───────────────────────────────────────────────────
    out_dir = Path(args.output_dir)
    per_prompt_df = pd.DataFrame(all_rows,
        columns=["model", "prompt_source", "prompt", "n_tokens", "kl",
                 "top5_clean_json", "top5_ablated_json"])
    per_prompt_df.to_csv(out_dir / "per_prompt.csv", index=False)

    per_dist_df = pd.DataFrame(per_dist_agg,
        columns=["model", "prompt_source", "n_prompts",
                 "median_kl", "mean_kl", "p90_kl", "p99_kl"])
    per_dist_df.to_csv(out_dir / "kl_per_distribution.csv", index=False)

    baseline_med = None
    for agg in per_dist_agg:
        if agg["prompt_source"] == "scalejsd_baseline":
            baseline_med = agg["median_kl"]
            break

    effect_ratios: Dict[str, float] = {}
    if baseline_med is not None and baseline_med > 0:
        for agg in per_dist_agg:
            src = agg["prompt_source"]
            if src == "scalejsd_baseline":
                continue
            if np.isnan(agg["median_kl"]):
                continue
            effect_ratios[f"{src}_over_baseline"] = float(
                agg["median_kl"] / baseline_med)

    summary = {
        "model": args.model,
        "revision": args.revision,
        "dtype": str(dtype),
        "steer_layer": int(steer_layer),
        "probe_auroc": float(probe_auroc),
        "n_layers_ablated": len(all_layers),
        "n_prompts_per_dist": args.n_prompts_per_dist,
        "popqa_max_pop": args.popqa_max_pop,
        "per_distribution": per_dist_agg,
        "effect_ratios_vs_scalejsd_baseline": effect_ratios,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"\nWrote: {out_dir}")


if __name__ == "__main__":
    main()
