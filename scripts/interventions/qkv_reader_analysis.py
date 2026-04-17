#!/usr/bin/env python3
"""
Q/K/V reader test: does each attention head READ the frequency direction v
via its query/key/value input projections?

Attribution patching (already completed) measures whether a component's
OUTPUT projects onto v — that's a WRITER test.  It does not tell us whether
a component RECEIVES information about v.  Each attention head's per-head
Q, K, V projection matrices W_Q^h, W_K^h, W_V^h ∈ R^{d_head × d_model} map
the residual stream into head-space.  If a head's rows (= its linear
read-out directions) align with v, that head is a reader of the frequency
subspace — without any forward pass needed.

For every head h at every layer L > L*, we compute:

  max_cos_W[L, h]  = max_r |cos(v, W[r, :])|      (strongest row reader)
  matrix_proj_W[L, h] = ||W v||_2 / ||W||_F       (fraction of matrix norm
                                                   aligned with v)

for W ∈ {Q, K, V}.  A random-direction null baseline (100 Gaussian unit
vectors matched to d_model) gives per-model mean and 99th-percentile
thresholds; a head is "significant" if its max_cos exceeds p99.

Outputs:
    qkv_projections.csv  per (model, layer, head, w_type)
    top_readers.csv      top-20 heads by max_cos across Q/K/V
    summary.json         scalar summaries + null stats + best_layer
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

# Reuse scaleJSD sample building + probe training
try:
    from scripts.metrics.coverage_affinity_experiment import (
        load_synonym_pairs,
        pairs_to_samples,
    )
    from scripts.interventions.frequency_steering import (
        find_transformer_layers,
        train_probe_at_layer,
    )
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from scripts.metrics.coverage_affinity_experiment import (
        load_synonym_pairs,
        pairs_to_samples,
    )
    from scripts.interventions.frequency_steering import (
        find_transformer_layers,
        train_probe_at_layer,
    )


DATASET_FILES = {
    "emotion": "emotion_ngrams_dedup_filtered.jsonl",
    "medical": "medical_ngrams_dedup_filtered.jsonl",
    "legal": "legal_ngrams_dedup_filtered.jsonl",
    "scientific": "scientific_ngrams_dedup_filtered.jsonl",
    "verb": "verb_ngrams_dedup_filtered.jsonl",
}


# ─────────────────────────────────────────────────────────────────────
#  Attention-module introspection
# ─────────────────────────────────────────────────────────────────────

def _find_attention_module(layer):
    for name in ("self_attn", "attention", "attn"):
        if hasattr(layer, name):
            return getattr(layer, name), name
    raise RuntimeError(f"No attention module in layer {layer}")


def _get_num_heads(model) -> int:
    cfg = model.config
    for k in ("num_attention_heads", "n_head", "num_heads"):
        if hasattr(cfg, k):
            return int(getattr(cfg, k))
    raise RuntimeError("Cannot determine num_attention_heads")


def _get_num_kv_heads(model) -> int:
    """GQA/MQA — K,V may have fewer heads than Q."""
    cfg = model.config
    for k in ("num_key_value_heads", "num_kv_heads"):
        if hasattr(cfg, k):
            val = getattr(cfg, k)
            if val is not None:
                return int(val)
    return _get_num_heads(model)


def _get_hidden_size(model) -> int:
    cfg = model.config
    for k in ("hidden_size", "n_embd", "d_model"):
        if hasattr(cfg, k):
            return int(getattr(cfg, k))
    raise RuntimeError("Cannot determine hidden_size")


def _get_head_dim(model) -> int:
    cfg = model.config
    if hasattr(cfg, "head_dim") and getattr(cfg, "head_dim") is not None:
        return int(cfg.head_dim)
    return _get_hidden_size(model) // _get_num_heads(model)


def extract_qkv_weights(layer, model) -> Dict[str, torch.Tensor]:
    """Return {'Q': W_Q, 'K': W_K, 'V': W_V} as CPU fp32 tensors of shape
    (num_heads_for_this_stream, d_head, d_model).  Handles:
      - separate q_proj/k_proj/v_proj (Llama, OLMo-hf, Mistral)
      - fused query_key_value (Pythia / GPT-NeoX)
      - fused att_proj (some OLMo variants)
      - GQA: K and V use num_kv_heads, Q uses num_heads
    """
    attn, _name = _find_attention_module(layer)
    d_model = _get_hidden_size(model)
    n_heads = _get_num_heads(model)
    n_kv_heads = _get_num_kv_heads(model)
    d_head = _get_head_dim(model)

    def _to_fp32(w):
        return w.detach().to(dtype=torch.float32, device="cpu")

    # Case 1: separate q_proj / k_proj / v_proj (Llama, OLMo-7B-hf, Mistral, Qwen)
    if all(hasattr(attn, n) for n in ("q_proj", "k_proj", "v_proj")):
        Wq = _to_fp32(attn.q_proj.weight)  # (n_heads*d_head, d_model)
        Wk = _to_fp32(attn.k_proj.weight)  # (n_kv_heads*d_head, d_model)
        Wv = _to_fp32(attn.v_proj.weight)  # (n_kv_heads*d_head, d_model)
        # Reshape to per-head
        Q = Wq.view(n_heads, d_head, d_model)
        K = Wk.view(n_kv_heads, d_head, d_model)
        V = Wv.view(n_kv_heads, d_head, d_model)
        return {"Q": Q, "K": K, "V": V}

    # Case 2: fused query_key_value (Pythia / GPT-NeoX)
    #   HF stores weight as (3*n_heads*d_head, d_model) interleaved per head:
    #   [head0_q, head0_k, head0_v, head1_q, head1_k, head1_v, ...]
    if hasattr(attn, "query_key_value"):
        W = _to_fp32(attn.query_key_value.weight)  # (3*d_model, d_model)
        # GPT-NeoX layout: reshape -> (n_heads, 3, d_head, d_model)
        W_reshaped = W.view(n_heads, 3, d_head, d_model)
        Q = W_reshaped[:, 0, :, :].contiguous()
        K = W_reshaped[:, 1, :, :].contiguous()
        V = W_reshaped[:, 2, :, :].contiguous()
        return {"Q": Q, "K": K, "V": V}

    # Case 3: fused att_proj (OLMo original — unlikely on -hf variant but safe)
    if hasattr(attn, "att_proj"):
        W = _to_fp32(attn.att_proj.weight)  # (3*d_model, d_model) for MHA
        # OLMo layout: concatenated [Q; K; V] along output dim
        d_q = n_heads * d_head
        d_kv = n_kv_heads * d_head
        Wq = W[:d_q]
        Wk = W[d_q:d_q + d_kv]
        Wv = W[d_q + d_kv:d_q + 2 * d_kv]
        Q = Wq.view(n_heads, d_head, d_model)
        K = Wk.view(n_kv_heads, d_head, d_model)
        V = Wv.view(n_kv_heads, d_head, d_model)
        return {"Q": Q, "K": K, "V": V}

    raise RuntimeError(
        f"Unknown attention layout; attributes: {[a for a in dir(attn) if 'proj' in a.lower() or 'qkv' in a.lower()]}"
    )


# ─────────────────────────────────────────────────────────────────────
#  Alignment math
# ─────────────────────────────────────────────────────────────────────

def row_max_abs_cos(W: torch.Tensor, v: torch.Tensor) -> float:
    """W: (d_head, d_model), v: (d_model,) unit. Returns max_r |cos(v, W[r])|."""
    row_norms = W.norm(dim=1).clamp_min(1e-12)  # (d_head,)
    dots = (W @ v).abs()  # (d_head,)
    cos = dots / row_norms  # v already unit-normalized
    return float(cos.max().item())


def matrix_projection(W: torch.Tensor, v: torch.Tensor) -> float:
    """||W v||_2 / ||W||_F — proportion of W's Frobenius norm aligned with v."""
    numer = (W @ v).norm().item()
    denom = W.norm().item()  # Frobenius
    return float(numer / max(denom, 1e-12))


def compute_null_baseline(
    d_model: int,
    d_head: int,
    n_random: int,
    seed: int,
) -> Dict[str, Dict[str, float]]:
    """Generate n_random Gaussian unit vectors, compute max_abs_cos and
    matrix_proj stats against a fresh random matrix (d_head × d_model).
    Since max_abs_cos depends only on d_model and d_head (not the matrix
    magnitudes), and similarly for the normalized matrix projection, we
    can derive a single baseline distribution per (d_model, d_head) pair."""
    rng = np.random.default_rng(seed)
    # Sample n_random unit vectors in d_model.
    V = rng.standard_normal((n_random, d_model)).astype(np.float32)
    V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-12

    # Sample ONE random matrix and test alignment of each v with it; by
    # symmetry (v random vs W random == W random vs v random) this gives
    # the null distribution of max_abs_cos for a random direction vs a
    # random d_head × d_model matrix.
    W_null = rng.standard_normal((d_head, d_model)).astype(np.float32)
    W_null_t = torch.from_numpy(W_null)

    max_cos_samples = []
    matrix_proj_samples = []
    for vi in V:
        v_t = torch.from_numpy(vi)
        max_cos_samples.append(row_max_abs_cos(W_null_t, v_t))
        matrix_proj_samples.append(matrix_projection(W_null_t, v_t))
    mc = np.asarray(max_cos_samples)
    mp = np.asarray(matrix_proj_samples)
    return {
        "max_cos": {
            "mean": float(mc.mean()),
            "p50": float(np.percentile(mc, 50)),
            "p95": float(np.percentile(mc, 95)),
            "p99": float(np.percentile(mc, 99)),
        },
        "matrix_proj": {
            "mean": float(mp.mean()),
            "p50": float(np.percentile(mp, 50)),
            "p95": float(np.percentile(mp, 95)),
            "p99": float(np.percentile(mp, 99)),
        },
    }


# ─────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────

def _load_best_layer(
    probe_json_path: Optional[str],
    override: Optional[int],
) -> Optional[int]:
    if override is not None:
        return int(override)
    if probe_json_path and Path(probe_json_path).exists():
        with open(probe_json_path) as f:
            obj = json.load(f)
        for k in ("best_layer", "peak_layer"):
            if k in obj:
                return int(obj[k])
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--revision", default="main")
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--steer-layer", type=int, default=None,
                   help="L* override. If None, read from --probe-json.")
    p.add_argument("--probe-json", default=None,
                   help="Path to probe.json with best_layer")
    p.add_argument("--n-null", type=int, default=100,
                   help="Number of random unit vectors for null baseline")
    p.add_argument("--top-k", type=int, default=20,
                   help="Number of top heads to save per w_type")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.makedirs(args.output_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    print(f"Loading tokenizer {args.model} (revision={args.revision})")
    tok_kw: Dict[str, Any] = {}
    if hf_token:
        tok_kw["token"] = hf_token
    if args.revision != "main":
        tok_kw["revision"] = args.revision
    tokenizer = AutoTokenizer.from_pretrained(args.model, **tok_kw)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model {args.model} in fp32 on {args.device}")
    load_kw: Dict[str, Any] = {"torch_dtype": torch.float32}
    if args.revision != "main":
        load_kw["revision"] = args.revision
    if hf_token:
        load_kw["token"] = hf_token
    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kw)
    model = model.to(args.device)
    model.eval()

    d_model = _get_hidden_size(model)
    n_heads = _get_num_heads(model)
    n_kv_heads = _get_num_kv_heads(model)
    d_head = _get_head_dim(model)
    n_layers = int(model.config.num_hidden_layers)
    print(f"  d_model={d_model}  n_heads={n_heads}  n_kv_heads={n_kv_heads}  "
          f"d_head={d_head}  n_layers={n_layers}")

    # ── Resolve L* ──────────────────────────────────────────────────
    best_layer = _load_best_layer(args.probe_json, args.steer_layer)

    # ── Load ScaleJSD samples (used to train probe if we need to) ──
    print("Loading ScaleJSD samples")
    all_samples = []
    for ds_name, fname in DATASET_FILES.items():
        ds_path = Path(args.dataset_dir) / fname
        if not ds_path.exists():
            print(f"  skipping missing: {ds_path}")
            continue
        pairs = load_synonym_pairs(str(ds_path))
        samples = pairs_to_samples(pairs, tokenizer)
        for s in samples:
            s["dataset"] = ds_name
        all_samples.extend(samples)
    print(f"  total samples: {len(all_samples)}")
    if not all_samples:
        raise RuntimeError(f"No samples loaded from {args.dataset_dir}")

    # If no best_layer, we would need to sweep — refuse and require explicit.
    if best_layer is None:
        raise RuntimeError(
            "Must provide --steer-layer or --probe-json with best_layer field"
        )
    best_layer = int(best_layer)
    if best_layer >= n_layers:
        raise RuntimeError(f"best_layer={best_layer} >= n_layers={n_layers}")
    print(f"  best_layer L* = {best_layer}")

    # ── Train probe at L* to get v ──────────────────────────────────
    print(f"Training frequency probe at L*={best_layer}")
    v_np, auroc, _ = train_probe_at_layer(
        model, all_samples, best_layer, args.device)
    v = torch.from_numpy(v_np.astype(np.float32))  # (d_model,), unit norm
    v_norm = float(v.norm())
    print(f"  probe AUROC = {auroc:.3f}   ||v|| = {v_norm:.4f}")

    # ── Null baseline ───────────────────────────────────────────────
    print(f"Computing random-direction null baseline (n={args.n_null})")
    null_stats = compute_null_baseline(
        d_model=d_model, d_head=d_head, n_random=args.n_null, seed=args.seed)
    print(f"  null max_cos mean={null_stats['max_cos']['mean']:.4f}  "
          f"p99={null_stats['max_cos']['p99']:.4f}")
    print(f"  null matrix_proj mean={null_stats['matrix_proj']['mean']:.4f}  "
          f"p99={null_stats['matrix_proj']['p99']:.4f}")

    # ── Sweep layers > L* ───────────────────────────────────────────
    transformer_layers = find_transformer_layers(model)
    model_short = args.model.split("/")[-1]

    rows: List[Dict[str, Any]] = []
    for L in range(best_layer + 1, n_layers):
        qkv = extract_qkv_weights(transformer_layers[L], model)
        for w_type, W_all in qkv.items():
            # W_all: (n_heads_for_stream, d_head, d_model)
            n_h = W_all.shape[0]
            for h in range(n_h):
                W_h = W_all[h]  # (d_head, d_model)
                mc = row_max_abs_cos(W_h, v)
                mp = matrix_projection(W_h, v)
                p99_mc = null_stats["max_cos"]["p99"]
                significant = bool(mc > p99_mc)
                rows.append({
                    "model": model_short,
                    "revision": args.revision,
                    "layer": int(L),
                    "head": int(h),
                    "w_type": w_type,
                    "max_cos": mc,
                    "matrix_proj": mp,
                    "null_mean_max_cos": null_stats["max_cos"]["mean"],
                    "null_p99_max_cos": null_stats["max_cos"]["p99"],
                    "null_mean_matrix_proj": null_stats["matrix_proj"]["mean"],
                    "null_p99_matrix_proj": null_stats["matrix_proj"]["p99"],
                    "significant": significant,
                })
        if (L - best_layer) % 5 == 0 or L == n_layers - 1:
            print(f"  processed layer {L}/{n_layers-1}", flush=True)

    df = pd.DataFrame(rows)
    out_csv = Path(args.output_dir) / "qkv_projections.csv"
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}  ({len(df)} rows)")

    # ── Top-K readers per w_type ─────────────────────────────────────
    top_frames = []
    for w_type in ("Q", "K", "V"):
        sub = df[df["w_type"] == w_type].nlargest(args.top_k, "max_cos").copy()
        sub["rank"] = range(1, len(sub) + 1)
        top_frames.append(sub)
    top_df = pd.concat(top_frames, ignore_index=True)
    top_csv = Path(args.output_dir) / "top_readers.csv"
    top_df.to_csv(top_csv, index=False)
    print(f"Wrote {top_csv}  ({len(top_df)} rows)")

    # ── Summary ──────────────────────────────────────────────────────
    summary = {
        "model": args.model,
        "revision": args.revision,
        "model_short": model_short,
        "d_model": d_model,
        "n_heads": n_heads,
        "n_kv_heads": n_kv_heads,
        "d_head": d_head,
        "n_layers": n_layers,
        "best_layer_L_star": best_layer,
        "probe_auroc_at_L_star": float(auroc),
        "probe_direction_norm": v_norm,
        "null_stats": null_stats,
        "n_heads_significant_Q": int(df[(df.w_type == "Q") & df.significant].shape[0]),
        "n_heads_significant_K": int(df[(df.w_type == "K") & df.significant].shape[0]),
        "n_heads_significant_V": int(df[(df.w_type == "V") & df.significant].shape[0]),
        "n_total_Q": int((df.w_type == "Q").sum()),
        "n_total_K": int((df.w_type == "K").sum()),
        "n_total_V": int((df.w_type == "V").sum()),
        "max_max_cos_Q": float(df[df.w_type == "Q"]["max_cos"].max()),
        "max_max_cos_K": float(df[df.w_type == "K"]["max_cos"].max()),
        "max_max_cos_V": float(df[df.w_type == "V"]["max_cos"].max()),
    }
    with open(Path(args.output_dir) / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {Path(args.output_dir) / 'summary.json'}")

    print("\n=== Reader-head summary ===")
    for w_type in ("Q", "K", "V"):
        n_sig = summary[f"n_heads_significant_{w_type}"]
        n_tot = summary[f"n_total_{w_type}"]
        max_mc = summary[f"max_max_cos_{w_type}"]
        pct = 100.0 * n_sig / max(n_tot, 1)
        print(f"  {w_type}: {n_sig}/{n_tot} significant ({pct:.1f}%)  "
              f"max max_cos = {max_mc:.4f}  "
              f"(null p99 = {null_stats['max_cos']['p99']:.4f})")

    # ── Path-patch stub ──────────────────────────────────────────────
    # TODO: for each top-K reader head, patch v-component of residual at
    # L* on a single high-freq input and measure z change at that head.
    # Requires careful hook + residual writer/reader orchestration; the
    # weight-based analysis above is the primary deliverable. Stub left
    # here so downstream reviewers know we considered it.
    print("\n[INFO] Path-patch activation-space verification skipped (TODO).")


if __name__ == "__main__":
    main()
