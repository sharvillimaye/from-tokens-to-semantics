#!/usr/bin/env python3
"""
Frequency steering v2 — stylistic re-characterization.

The v1 script (`frequency_steering.py`) reported `mean_gen_log_freq` as the
headline metric, but this turned out to be V-shaped at extreme α because the
model's fluency collapses and emits degenerate repetitive tokens whose
unigram prior is unusual. The QUALITATIVE signal — register, word choice,
brevity — is real and visible in the generations. This v2 script re-runs
the same additive intervention but adds PROPER stylistic metrics so we can
quantify the shift without relying on unigram-prior averages over degraded
text.

Metrics (per prompt, per α, averaged over N prompts):
  - Flesch-Kincaid grade level (textstat)
  - Mean word length (chars)
  - Type-token ratio over generated tokens
  - Mean Zipf frequency (wordfreq, en)
  - First-sentence length in characters
  - First-token high_freq_mass / low_freq_mass  (retained from v1)
  - Median self-perplexity of the generation (retained)

Modes:
  - Sampled (default): do_sample=True, temperature=1.0, top_p=0.9
  - Deterministic: --deterministic forces greedy decoding (do_sample=False)

Output:
  - steering_metrics.csv
  - sample_generations.csv (5 prompts × each α)
  - probe_orthogonality.csv (reused from v1 pipeline)
  - summary.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder

# Re-use hook / probe / orthogonality infra from v1.
try:
    from scripts.interventions.frequency_steering import (
        DATASET_FILES,
        ResidualCapture,
        SteeringHook,
        _build_prompt_from_sample,
        _select_dtype,
        compute_orthogonality,
        compute_token_log_frequencies,
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
    from scripts.interventions.frequency_steering import (  # type: ignore
        DATASET_FILES,
        ResidualCapture,
        SteeringHook,
        _build_prompt_from_sample,
        _select_dtype,
        compute_orthogonality,
        compute_token_log_frequencies,
        find_transformer_layers,
        train_probe_at_layer,
    )
    from scripts.metrics.coverage_affinity_experiment import (  # type: ignore
        _infer_layers,
        load_synonym_pairs,
        pairs_to_samples,
    )


# Lazy imports for optional deps so `python -m py_compile` works on dev laptop.
def _load_textstat():
    import textstat  # type: ignore
    return textstat


def _load_wordfreq():
    from wordfreq import zipf_frequency  # type: ignore
    return zipf_frequency


# ─────────────────────────────────────────────────────────────────────
#  Stylistic metrics
# ─────────────────────────────────────────────────────────────────────

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*")


def _tokenize_words(text: str) -> List[str]:
    return _WORD_RE.findall(text)


def _first_sentence_len(text: str) -> int:
    """Characters before the first period (or end of string if none)."""
    if not text:
        return 0
    idx = text.find(".")
    if idx < 0:
        return len(text)
    return idx


def compute_stylistic_metrics(text: str, zipf_fn) -> Dict[str, float]:
    """Return per-generation stylistic metrics.

    `text` is the decoded continuation (prompt stripped).
    """
    textstat = _load_textstat()
    text_stripped = (text or "").strip()
    words = _tokenize_words(text_stripped)
    n_words = len(words)

    if n_words == 0:
        return {
            "flesch_kincaid": float("nan"),
            "mean_word_len": float("nan"),
            "type_token_ratio": float("nan"),
            "mean_zipf": float("nan"),
            "first_sentence_len": float(_first_sentence_len(text_stripped)),
            "n_words": 0,
        }

    try:
        fk = float(textstat.flesch_kincaid_grade(text_stripped))
    except Exception:
        fk = float("nan")

    mean_word_len = float(np.mean([len(w) for w in words]))
    ttr = float(len(set(w.lower() for w in words)) / n_words)

    zipfs: List[float] = []
    for w in words:
        try:
            z = float(zipf_fn(w.lower(), "en"))
        except Exception:
            z = 0.0
        zipfs.append(z)
    mean_zipf = float(np.mean(zipfs)) if zipfs else float("nan")

    first_len = float(_first_sentence_len(text_stripped))

    return {
        "flesch_kincaid": fk,
        "mean_word_len": mean_word_len,
        "type_token_ratio": ttr,
        "mean_zipf": mean_zipf,
        "first_sentence_len": first_len,
        "n_words": int(n_words),
    }


# ─────────────────────────────────────────────────────────────────────
#  Generation + aggregation
# ─────────────────────────────────────────────────────────────────────

def _generation_perplexity(model, gen_ids: torch.Tensor, prompt_len: int) -> float:
    if gen_ids.shape[1] <= prompt_len + 1:
        return float("nan")
    with torch.no_grad():
        out = model(input_ids=gen_ids)
        logits = out.logits[0, prompt_len - 1:-1, :]
        targets = gen_ids[0, prompt_len:]
        logp = F.log_softmax(logits.float(), dim=-1)
        tok_logp = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        return float(torch.exp(-tok_logp.mean()))


def run_steering_stylistic(
    model,
    tokenizer,
    direction: np.ndarray,
    layer: int,
    alphas: List[float],
    prompts: List[str],
    token_log_freqs: np.ndarray,
    device: str,
    *,
    max_new_tokens: int = 50,
    seed: int = 42,
    deterministic: bool = False,
    sample_texts_per_alpha: int = 5,
    model_tag: str = "",
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Run the steering sweep and compute stylistic metrics.

    Returns (metric_rows, sample_generations).
    """
    zipf_fn = _load_wordfreq()

    v = torch.tensor(direction, dtype=torch.float32, device=device)
    vocab_size = len(token_log_freqs)

    q_hi = np.quantile(token_log_freqs, 0.75)
    q_lo = np.quantile(token_log_freqs, 0.25)
    high_mask = token_log_freqs >= q_hi
    low_mask = token_log_freqs <= q_lo

    prompt_tensors = []
    for prompt in prompts:
        ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        prompt_tensors.append(ids)

    results: List[Dict[str, Any]] = []
    samples_out: List[Dict[str, Any]] = []

    for alpha in alphas:
        print(f"    α={alpha:+.3f}", end=" ", flush=True)

        per_prompt_metrics: List[Dict[str, float]] = []
        per_prompt_high_mass: List[float] = []
        per_prompt_low_mass: List[float] = []
        per_prompt_perp: List[float] = []

        hook = SteeringHook(model, v, alpha, [layer]) if abs(alpha) > 1e-8 else None

        try:
            for pi, input_ids in enumerate(prompt_tensors):
                prompt_len = int(input_ids.shape[1])

                with torch.no_grad():
                    steered_out = model(input_ids=input_ids)
                    steered_logits = steered_out.logits[0, -1, :vocab_size].float().cpu()

                    gen_kwargs: Dict[str, Any] = dict(
                        max_new_tokens=max_new_tokens,
                        pad_token_id=tokenizer.pad_token_id
                            if tokenizer.pad_token_id is not None
                            else tokenizer.eos_token_id,
                    )
                    if deterministic:
                        gen_kwargs["do_sample"] = False
                    else:
                        torch.manual_seed(seed + pi)
                        gen_kwargs.update(
                            do_sample=True,
                            temperature=1.0,
                            top_p=0.9,
                        )

                    gen = model.generate(input_ids, **gen_kwargs)

                perp = _generation_perplexity(model, gen, prompt_len)
                per_prompt_perp.append(perp)

                gen_text = tokenizer.decode(
                    gen[0, prompt_len:], skip_special_tokens=True)
                m = compute_stylistic_metrics(gen_text, zipf_fn)
                per_prompt_metrics.append(m)

                probs = F.softmax(steered_logits, dim=-1).numpy()
                per_prompt_high_mass.append(float(probs[high_mask].sum()))
                per_prompt_low_mass.append(float(probs[low_mask].sum()))

                if pi < sample_texts_per_alpha:
                    samples_out.append({
                        "model": model_tag,
                        "alpha": float(alpha),
                        "prompt": tokenizer.decode(input_ids[0], skip_special_tokens=True),
                        "generation": gen_text,
                        "perplexity": perp,
                        "flesch_kincaid": m["flesch_kincaid"],
                        "mean_word_len": m["mean_word_len"],
                        "mean_zipf": m["mean_zipf"],
                        "first_sentence_len": m["first_sentence_len"],
                    })
        finally:
            if hook is not None:
                hook.remove()

        def _nanmean(vals: List[float]) -> float:
            arr = np.array(vals, dtype=float)
            if arr.size == 0 or np.all(np.isnan(arr)):
                return float("nan")
            return float(np.nanmean(arr))

        row = {
            "model": model_tag,
            "alpha": float(alpha),
            "layer": int(layer),
            "flesch_kincaid": _nanmean([m["flesch_kincaid"] for m in per_prompt_metrics]),
            "mean_word_len": _nanmean([m["mean_word_len"] for m in per_prompt_metrics]),
            "type_token_ratio": _nanmean([m["type_token_ratio"] for m in per_prompt_metrics]),
            "mean_zipf": _nanmean([m["mean_zipf"] for m in per_prompt_metrics]),
            "first_sentence_len": _nanmean([m["first_sentence_len"] for m in per_prompt_metrics]),
            "low_freq_mass": _nanmean(per_prompt_low_mass),
            "high_freq_mass": _nanmean(per_prompt_high_mass),
            "median_perplexity": float(np.nanmedian(per_prompt_perp))
                if per_prompt_perp else float("nan"),
            "n_prompts": len(prompts),
        }
        results.append(row)

        print(
            f"FK={row['flesch_kincaid']:.2f} "
            f"zipf={row['mean_zipf']:.2f} "
            f"wlen={row['mean_word_len']:.2f} "
            f"ttr={row['type_token_ratio']:.2f} "
            f"s1={row['first_sentence_len']:.0f} "
            f"hi={row['high_freq_mass']:.3f} lo={row['low_freq_mass']:.3f} "
            f"perp={row['median_perplexity']:.2f}",
            flush=True,
        )

    return results, samples_out


# ─────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--alphas",
                        default="-2,-1,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1,2")
    parser.add_argument("--steer-layer", type=int, default=None,
                        help="Layer to steer at. Default: layer with peak freq AUROC.")
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--n-prompts", type=int, default=200,
                        help="Number of prompts for generation test")
    parser.add_argument("--corpus-file", default=None)
    parser.add_argument("--unigram-counts-file", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true",
                        help="Use greedy decoding (do_sample=False).")
    parser.add_argument("--skip-orthogonality", action="store_true",
                        help="Skip the per-layer orthogonality sweep. "
                             "Implies --steer-layer must be set.")
    parser.add_argument("--dtype", default="auto",
                        choices=["auto", "float32", "bfloat16", "float16"])
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
        dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16,
                 "float16": torch.float16}[args.dtype]

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

    if not args.skip_orthogonality:
        print("\n=== Probe orthogonality (frequency vs semantic directions) ===")
        ortho_df = compute_orthogonality(model, all_samples, all_layers, args.device)
        ortho_df.to_csv(Path(args.output_dir) / "probe_orthogonality.csv", index=False)

        if args.steer_layer is not None:
            best_layer = int(args.steer_layer)
        else:
            layer_aurocs = ortho_df.groupby("layer")["freq_auroc"].first()
            best_layer = int(layer_aurocs.idxmax())
        best_auroc = float(
            ortho_df[ortho_df["layer"] == best_layer]["freq_auroc"].iloc[0])
        print(f"\n  Best layer for steering: L{best_layer} "
              f"(freq AUROC = {best_auroc:.3f})")
    else:
        if args.steer_layer is None:
            raise SystemExit("--skip-orthogonality requires --steer-layer to be set")
        best_layer = int(args.steer_layer)
        ortho_df = None

    print(f"\n=== Training probe at L{best_layer} ===")
    direction, auroc, _ = train_probe_at_layer(
        model, all_samples, best_layer, args.device)
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

    mode_tag = "greedy" if args.deterministic else "sampled"
    print(f"\n=== Steering sweep at L{best_layer}  "
          f"[{mode_tag}] ({len(prompts)} prompts × {len(alphas)} α) ===")
    rows, sample_rows = run_steering_stylistic(
        model, tokenizer, direction, best_layer, alphas, prompts,
        token_log_freqs, args.device,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
        deterministic=args.deterministic,
        model_tag=args.model,
    )

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(Path(args.output_dir) / "steering_metrics.csv", index=False)
    pd.DataFrame(sample_rows).to_csv(
        Path(args.output_dir) / "sample_generations.csv", index=False)

    print(f"\n{'='*72}")
    print(f"  STYLISTIC METRICS SUMMARY  [{mode_tag}]  model={args.model}")
    print(f"{'='*72}")
    cols = ["alpha", "flesch_kincaid", "mean_zipf", "mean_word_len",
            "type_token_ratio", "first_sentence_len",
            "high_freq_mass", "low_freq_mass", "median_perplexity"]
    print(metrics_df[cols].to_string(index=False))

    summary = {
        "model": args.model,
        "revision": args.revision,
        "dtype": str(dtype),
        "steer_layer": int(best_layer),
        "probe_auroc_best_layer": float(auroc),
        "deterministic": bool(args.deterministic),
        "alphas": alphas,
        "n_prompts": len(prompts),
        "max_new_tokens": int(args.max_new_tokens),
        "seed": int(args.seed),
    }
    with open(Path(args.output_dir) / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
