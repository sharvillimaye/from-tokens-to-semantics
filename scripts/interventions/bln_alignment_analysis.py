#!/usr/bin/env python3
"""
Idea C — b_LN alignment test.

Tests whether the residual-stream frequency direction v, when un-embedded to
vocabulary space via u = W_U @ v, aligns with:

  1. The model's learned unigram, estimated from the model's own logits on a
     neutral context (BOS-only input).  This is the "b_LN-analog" for any
     architecture: the marginal P_model(token | no context).
  2. The actual final-LayerNorm bias b_LN (only present in Pythia / GPT-NeoX;
     Llama and OLMo-7B use RMSNorm which has no bias — we report None there).

Kobayashi et al. 2023 (arXiv:2305.18294) showed that in GPT-2 b_LN correlates
with corpus word frequency (Spearman rho ≈ 0.78), while hidden states are
nearly orthogonal to it.  Our 1D frequency direction v is an upstream,
residual-stream signal; this experiment asks: is it the SAME direction
accessed earlier in the pipeline, or an independent second frequency
pathway?

Three outcomes, all informative:
  - Aligned (cos(u, log P_model) high): upstream direction IS the b_LN
    direction, unifying the two findings.
  - Orthogonal: two independent frequency pathways (consistent with our
    cumulative-ablation KL-small finding).
  - Partially aligned: richer picture with overlap but not identity.

Outputs:
  bln_alignment.json  — cosines, spearman, u_norm_ratio, top/bottom tokens
  top_tokens.csv      — top 50 and bottom 50 tokens by u value, with
                        log_P_model and b_LN values if present
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
import torch.nn.functional as F
from scipy.stats import spearmanr

# Reuse infrastructure
try:
    from scripts.interventions.frequency_steering import (
        train_probe_at_layer,
    )
    from scripts.metrics.coverage_affinity_experiment import (
        load_synonym_pairs,
        pairs_to_samples,
    )
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from scripts.interventions.frequency_steering import (  # noqa: E402
        train_probe_at_layer,
    )
    from scripts.metrics.coverage_affinity_experiment import (  # noqa: E402
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
#  Model-side accessors
# ─────────────────────────────────────────────────────────────────────

def _get_unembed_matrix(model) -> torch.Tensor:
    """Return W_U as a CPU fp32 tensor of shape (vocab_size, d_model).

    Handles tied embeddings (use lm_head if present; otherwise fall back to
    model.get_output_embeddings()).  For Pythia/GPT-NeoX the output
    embedding is exposed as `embed_out`; get_output_embeddings covers it.
    """
    out_emb = None
    try:
        out_emb = model.get_output_embeddings()
    except Exception:
        out_emb = None

    if out_emb is not None and hasattr(out_emb, "weight"):
        W = out_emb.weight.detach().to(dtype=torch.float32, device="cpu")
        return W

    # Fallback: look directly at known attribute names.
    for attr_path in ("lm_head.weight", "embed_out.weight",
                      "model.lm_head.weight", "transformer.wte.weight"):
        obj = model
        try:
            for p in attr_path.split("."):
                obj = getattr(obj, p)
            return obj.detach().to(dtype=torch.float32, device="cpu")
        except AttributeError:
            continue
    raise RuntimeError("Could not locate unembedding matrix (W_U).")


def _get_final_layernorm_bias(model) -> Optional[torch.Tensor]:
    """Return the final-LayerNorm bias b_LN if present; else None.

    Pythia / GPT-NeoX: gpt_neox.final_layer_norm.bias
    GPT-2: transformer.ln_f.bias
    Llama / OLMo-7B (RMSNorm): no bias; return None.
    """
    candidate_paths = [
        "gpt_neox.final_layer_norm",
        "transformer.ln_f",
        "model.norm",
        "transformer.norm_f",
        "model.final_layer_norm",
    ]
    for path in candidate_paths:
        obj = model
        ok = True
        for p in path.split("."):
            if not hasattr(obj, p):
                ok = False
                break
            obj = getattr(obj, p)
        if not ok:
            continue
        # Check for a bias parameter with non-None value.
        bias = getattr(obj, "bias", None)
        if bias is not None and isinstance(bias, torch.Tensor):
            return bias.detach().to(dtype=torch.float32, device="cpu")
    return None


# ─────────────────────────────────────────────────────────────────────
#  Correlation helpers
# ─────────────────────────────────────────────────────────────────────

def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def _safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    try:
        r, _ = spearmanr(a, b)
        return float(r)
    except Exception:
        return float("nan")


# ─────────────────────────────────────────────────────────────────────
#  Neutral-context unigram
# ─────────────────────────────────────────────────────────────────────

def _neutral_input_ids(tokenizer, device: str) -> torch.Tensor:
    """Return input_ids for a minimal neutral context: [BOS] if defined, else
    a single space token.  Shape (1, T)."""
    if tokenizer.bos_token_id is not None:
        return torch.tensor([[int(tokenizer.bos_token_id)]], device=device)
    # Fallback: encode a single space (or the tokenizer's best neutral token)
    txt = tokenizer.eos_token if tokenizer.eos_token_id is not None else " "
    ids = tokenizer(txt, add_special_tokens=False, return_tensors="pt").input_ids
    if ids.numel() == 0:
        ids = tokenizer(" ", add_special_tokens=False, return_tensors="pt").input_ids
    return ids.to(device)


def compute_neutral_unigram(
    model, tokenizer, device: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run a forward pass on neutral input; return (log_P_model, P_model)
    over the vocabulary at the last position, as float32 numpy arrays."""
    input_ids = _neutral_input_ids(tokenizer, device)
    with torch.no_grad():
        out = model(input_ids=input_ids)
    logits = out.logits if hasattr(out, "logits") else out[0]
    # last position
    last = logits[0, -1, :].to(dtype=torch.float32, device="cpu")
    log_p = F.log_softmax(last, dim=-1).numpy()
    p = F.softmax(last, dim=-1).numpy()
    return log_p, p


# ─────────────────────────────────────────────────────────────────────
#  Probe direction resolution
# ─────────────────────────────────────────────────────────────────────

def _load_or_train_probe(
    probe_direction_path: Optional[str],
    model, tokenizer, device: str,
    steer_layer: Optional[int],
    dataset_dir: str,
) -> Tuple[np.ndarray, int, Optional[float]]:
    """Return (v_unit, best_layer, auroc_or_None).

    If --probe-direction is a valid .npy file, load it.  Otherwise train
    from ScaleJSD at `steer_layer` (required in that branch)."""
    # Try loading a precomputed direction.
    if probe_direction_path and Path(probe_direction_path).exists():
        v = np.load(probe_direction_path).astype(np.float32)
        v = v / (np.linalg.norm(v) + 1e-12)
        # Try loading a sibling probe_meta.json to recover best_layer.
        meta_path = Path(probe_direction_path).parent / "probe_meta.json"
        best_layer: Optional[int] = None
        auroc: Optional[float] = None
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            # Several possible keys used across scripts.
            for k in ("probe_layer", "best_layer", "peak_layer"):
                if k in meta:
                    best_layer = int(meta[k])
                    break
            if "probe_auroc" in meta:
                auroc = float(meta["probe_auroc"])
        if steer_layer is not None:
            best_layer = int(steer_layer)
        if best_layer is None:
            raise RuntimeError(
                f"Loaded probe from {probe_direction_path} but could not "
                "determine best_layer: pass --steer-layer, or place a "
                "probe_meta.json alongside the .npy with probe_layer."
            )
        return v, best_layer, auroc

    # Fallback: train from ScaleJSD at steer_layer.
    if steer_layer is None:
        raise RuntimeError(
            "No --probe-direction file and no --steer-layer; cannot train "
            "probe without a target layer."
        )
    print(f"  probe_direction file missing; training from ScaleJSD at L={steer_layer}")
    all_samples = []
    for ds_name, fname in DATASET_FILES.items():
        ds_path = Path(dataset_dir) / fname
        if not ds_path.exists():
            continue
        pairs = load_synonym_pairs(str(ds_path))
        samples = pairs_to_samples(pairs, tokenizer)
        for s in samples:
            s["dataset"] = ds_name
        all_samples.extend(samples)
    if not all_samples:
        raise RuntimeError(f"No ScaleJSD samples found in {dataset_dir}")
    v_np, auroc, _ = train_probe_at_layer(
        model, all_samples, int(steer_layer), device)
    v = v_np.astype(np.float32)
    v = v / (np.linalg.norm(v) + 1e-12)
    return v, int(steer_layer), float(auroc)


# ─────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--revision", default="main")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--probe-direction", default=None,
                   help="Path to probe_direction.npy (from Track C / halluc). "
                        "If missing, will train from ScaleJSD at --steer-layer.")
    p.add_argument("--steer-layer", type=int, default=None,
                   help="Layer L* override. If probe_meta.json lives next to "
                        "the .npy, we can recover L* from there; otherwise "
                        "this is required.")
    p.add_argument("--dataset-dir", default=None,
                   help="ScaleJSD dir (only needed if probe_direction.npy "
                        "is missing and we must re-train).")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--top-k", type=int, default=50,
                   help="How many top/bottom tokens by u to save in CSV.")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    # ── Load tokenizer + model ──────────────────────────────────────
    print(f"Loading tokenizer {args.model} (revision={args.revision})")
    tok_kw: Dict[str, Any] = {}
    if hf_token:
        tok_kw["token"] = hf_token
    if args.revision != "main":
        tok_kw["revision"] = args.revision
    tokenizer = AutoTokenizer.from_pretrained(args.model, **tok_kw)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model {args.model} (fp32, device={args.device})")
    load_kw: Dict[str, Any] = {"torch_dtype": torch.float32}
    if args.revision != "main":
        load_kw["revision"] = args.revision
    if hf_token:
        load_kw["token"] = hf_token
    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kw)
    model = model.to(args.device)
    model.eval()

    d_model = int(model.config.hidden_size) if hasattr(model.config, "hidden_size") \
        else int(getattr(model.config, "n_embd", 0))
    n_layers = int(model.config.num_hidden_layers) if hasattr(model.config, "num_hidden_layers") \
        else int(getattr(model.config, "n_layer", 0))

    # ── Resolve probe direction v + best_layer ──────────────────────
    print("Resolving probe direction v")
    v_np, best_layer, auroc = _load_or_train_probe(
        probe_direction_path=args.probe_direction,
        model=model, tokenizer=tokenizer, device=args.device,
        steer_layer=args.steer_layer,
        dataset_dir=args.dataset_dir or "",
    )
    print(f"  best_layer L* = {best_layer}   ||v|| (post-renorm) = {np.linalg.norm(v_np):.6f}"
          f"   probe_auroc = {auroc if auroc is not None else 'n/a'}")

    # ── Unembed matrix W_U ──────────────────────────────────────────
    print("Extracting W_U")
    W_U = _get_unembed_matrix(model)  # (vocab, d_model) fp32 CPU
    vocab_size = int(W_U.shape[0])
    d_model_from_W = int(W_U.shape[1])
    if d_model and d_model != d_model_from_W:
        print(f"  WARNING: config d_model={d_model} ≠ W_U d_model={d_model_from_W}; "
              f"using W_U.")
    d_model_eff = d_model_from_W

    # u = W_U @ v  — a vocabulary-space vector.
    v_t = torch.from_numpy(v_np.astype(np.float32))
    if v_t.shape[0] != d_model_eff:
        raise RuntimeError(
            f"Probe direction dim {v_t.shape[0]} ≠ W_U hidden dim {d_model_eff}; "
            "wrong model or wrong probe file."
        )
    u_t = (W_U @ v_t)  # (vocab,)
    u = u_t.numpy().astype(np.float32)
    u_norm = float(np.linalg.norm(u))
    W_U_frob = float(W_U.norm().item())
    u_norm_ratio = float(u_norm / max(W_U_frob, 1e-12))

    print(f"  W_U shape: {tuple(W_U.shape)}  ||u||={u_norm:.4f}  "
          f"||W_U||_F={W_U_frob:.4f}  ratio={u_norm_ratio:.6f}")

    # ── Model-derived unigram from neutral context ───────────────────
    print("Computing model-derived unigram from neutral context")
    log_P_model, P_model = compute_neutral_unigram(model, tokenizer, args.device)
    if log_P_model.shape[0] != vocab_size:
        # Logits can be narrower/wider than W_U (rare).  Truncate/pad by
        # intersection of indices; here we align by min.
        m = min(log_P_model.shape[0], vocab_size)
        log_P_model = log_P_model[:m]
        P_model = P_model[:m]
        u_for_corr = u[:m]
    else:
        u_for_corr = u

    cos_u_vs_unigram = _cos(u_for_corr, log_P_model)
    spearman_u_vs_unigram = _safe_spearman(u_for_corr, log_P_model)
    print(f"  cos(u, log P_model) = {cos_u_vs_unigram:.4f}")
    print(f"  spearman(u, log P_model) = {spearman_u_vs_unigram:.4f}")

    # ── b_LN if available ────────────────────────────────────────────
    b_LN = _get_final_layernorm_bias(model)
    has_b_LN = b_LN is not None
    cos_u_vs_b_LN: Optional[float] = None
    cos_unigram_vs_b_LN: Optional[float] = None
    if has_b_LN:
        b_LN_np = b_LN.numpy().astype(np.float32)
        if b_LN_np.shape[0] != d_model_eff:
            print(f"  WARNING: b_LN dim {b_LN_np.shape[0]} ≠ d_model {d_model_eff}; "
                  f"cannot compare.")
            has_b_LN = False
        else:
            # Compare u to W_U @ b_LN (b_LN is in residual space; projecting
            # it through W_U gives the vocabulary-space contribution).
            u_bLN = (W_U @ torch.from_numpy(b_LN_np)).numpy().astype(np.float32)
            # cos in residual space directly:
            cos_u_vs_b_LN = _cos(v_np.astype(np.float32), b_LN_np)
            # cos in vocabulary space between u and W_U @ b_LN:
            cos_unigram_vs_b_LN = _cos(log_P_model[:u_bLN.shape[0]],
                                       u_bLN[:log_P_model.shape[0]])
            print(f"  has b_LN (shape {b_LN_np.shape}) — "
                  f"cos(v, b_LN) = {cos_u_vs_b_LN:.4f}  "
                  f"cos(log P_model, W_U @ b_LN) = {cos_unigram_vs_b_LN:.4f}")
    else:
        print("  no b_LN parameter (model likely uses RMSNorm)")

    # ── Top-k / bottom-k tokens by u ─────────────────────────────────
    k = int(args.top_k)
    order = np.argsort(u)  # ascending
    bottom_idx = order[:k]
    top_idx = order[-k:][::-1]  # descending

    def _tok_str(tid: int) -> str:
        try:
            return tokenizer.decode([int(tid)], skip_special_tokens=False)
        except Exception:
            return f"<id={tid}>"

    csv_rows: List[Dict[str, Any]] = []
    for rank, tid in enumerate(top_idx, start=1):
        row: Dict[str, Any] = {
            "section": "top",
            "rank": int(rank),
            "token_id": int(tid),
            "token_str": _tok_str(int(tid)),
            "u_value": float(u[tid]),
            "log_P_model": float(log_P_model[tid]) if tid < log_P_model.shape[0] else None,
        }
        if has_b_LN:
            b_LN_val_np = b_LN.numpy() if b_LN is not None else None
            # Can't index into residual-space b_LN by token_id; report vocab-projected value.
            u_bLN_full = (W_U @ torch.from_numpy(b_LN_val_np.astype(np.float32))).numpy()
            row["W_U_dot_b_LN"] = float(u_bLN_full[tid]) if tid < u_bLN_full.shape[0] else None
        csv_rows.append(row)
    for rank, tid in enumerate(bottom_idx, start=1):
        row = {
            "section": "bottom",
            "rank": int(rank),
            "token_id": int(tid),
            "token_str": _tok_str(int(tid)),
            "u_value": float(u[tid]),
            "log_P_model": float(log_P_model[tid]) if tid < log_P_model.shape[0] else None,
        }
        if has_b_LN:
            b_LN_val_np = b_LN.numpy() if b_LN is not None else None
            u_bLN_full = (W_U @ torch.from_numpy(b_LN_val_np.astype(np.float32))).numpy()
            row["W_U_dot_b_LN"] = float(u_bLN_full[tid]) if tid < u_bLN_full.shape[0] else None
        csv_rows.append(row)
    top_df = pd.DataFrame(csv_rows)
    top_csv = Path(args.output_dir) / "top_tokens.csv"
    top_df.to_csv(top_csv, index=False)
    print(f"Wrote {top_csv} ({len(top_df)} rows)")

    # Compact json-friendly top-20 / bottom-20
    top20 = [
        {"token_id": int(tid),
         "token_str": _tok_str(int(tid)),
         "u_value": float(u[tid])}
        for tid in top_idx[:20]
    ]
    bottom20 = [
        {"token_id": int(tid),
         "token_str": _tok_str(int(tid)),
         "u_value": float(u[tid])}
        for tid in bottom_idx[:20]
    ]

    # ── Summary JSON ─────────────────────────────────────────────────
    summary = {
        "model": args.model,
        "revision": args.revision,
        "best_layer": int(best_layer),
        "d_model": int(d_model_eff),
        "vocab_size": int(vocab_size),
        "probe_auroc": (float(auroc) if auroc is not None else None),
        "cos_u_vs_model_unigram": float(cos_u_vs_unigram),
        "spearman_u_vs_model_unigram": float(spearman_u_vs_unigram),
        "has_b_LN": bool(has_b_LN),
        "cos_v_vs_b_LN": (float(cos_u_vs_b_LN) if cos_u_vs_b_LN is not None else None),
        "cos_model_unigram_vs_W_U_bLN": (float(cos_unigram_vs_b_LN)
                                         if cos_unigram_vs_b_LN is not None else None),
        "u_norm": float(u_norm),
        "W_U_frobenius": float(W_U_frob),
        "u_norm_ratio": float(u_norm_ratio),
        "top_20_tokens_by_u": top20,
        "bottom_20_tokens_by_u": bottom20,
    }
    out_json = Path(args.output_dir) / "bln_alignment.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {out_json}")

    print("\n=== b_LN alignment summary ===")
    print(f"  cos(u, log P_model)        = {cos_u_vs_unigram:+.4f}")
    print(f"  spearman(u, log P_model)   = {spearman_u_vs_unigram:+.4f}")
    if has_b_LN:
        print(f"  cos(v_residual, b_LN)      = {cos_u_vs_b_LN:+.4f}")
        print(f"  cos(log P_model, W_U·b_LN) = {cos_unigram_vs_b_LN:+.4f}")
    else:
        print("  (no b_LN — RMSNorm architecture)")
    print(f"  ||u|| / ||W_U||_F           = {u_norm_ratio:.6f}")

    print("\nTop-10 tokens by u (most upweighted by frequency direction):")
    for r in top20[:10]:
        print(f"  {r['u_value']:+.4f}  id={r['token_id']:<6}  {r['token_str']!r}")
    print("Bottom-10 tokens by u (most downweighted):")
    for r in bottom20[:10]:
        print(f"  {r['u_value']:+.4f}  id={r['token_id']:<6}  {r['token_str']!r}")


if __name__ == "__main__":
    main()
