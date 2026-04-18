#!/usr/bin/env python3
"""
Q/K-vs-OV dissociation: for each attention head, measure input-side
(Q/K reading) vs output-side (OV writing) alignment with a concept
direction v, and test whether ROUTING concepts (frequency) show
Q/K >> OV while AGGREGATION concepts (sentiment) show OV >> Q/K.

Following Elhage et al. 2021's QK/OV circuit decomposition:

  -- Q/K-read score per head (input side):
        max_r |cos(v, W_Q^h[r, :])|       (likewise W_K)
     Measures the strongest row of the query/key projection aligned
     with v — the degree to which this head *reads* v from its input.

  -- OV-write score per head (output side):
        ||P_v (W_O^h @ W_V^h)||_F / ||W_O^h @ W_V^h||_F
     where P_v = v v^T is the rank-1 projector onto v.  Equivalently
     ||v^T (W_O^h W_V^h)||_2 / ||W_O^h W_V^h||_F.  This is the fraction
     of the OV circuit's Frobenius norm whose *output column space*
     lives along v — the degree to which this head *writes* v into
     the residual stream.

GQA note: Llama-3.1/Qwen share K,V across several Q heads.  The OV
circuit for a q-head h is W_O^h @ W_V^{kv(h)} where kv(h) = h // (n_q/n_kv).

Null baseline: 100 random Gaussian unit vectors give per-model p99
thresholds for (Q-read, K-read, OV-write).  We report per-head scores
in units of null-p99 (the "excess over null").

Claim under test:
  Frequency  (routing):   mean_qk_excess / mean_ov_excess >> 1
  Sentiment  (aggregation): mean_qk_excess / mean_ov_excess << 1

Outputs:
  qk_vs_ov_scores.csv     per-row: (model, layer, head, direction,
                           score_type, score, null_p99, excess)
  dissociation_summary.json
  qk_vs_ov_scatter.png    2x3 panel grid (concept x model)
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

# ----------------------------------------------------------------------
# Reuse module helpers already written for the QKV-reader pipeline
# ----------------------------------------------------------------------
try:
    from scripts.interventions.frequency_steering import (
        find_transformer_layers,
        train_probe_at_layer,
    )
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parents[2]))
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


# ──────────────────────────────────────────────────────────────────────
#  Attention-module introspection
# ──────────────────────────────────────────────────────────────────────

def _find_attention_module(layer):
    for name in ("self_attn", "attention", "attn"):
        if hasattr(layer, name):
            return getattr(layer, name), name
    raise RuntimeError(f"No attention module in layer {layer}")


def _find_o_proj(attn_module):
    for name in ("o_proj", "out_proj", "dense", "c_proj", "wo"):
        if hasattr(attn_module, name):
            mod = getattr(attn_module, name)
            if isinstance(mod, torch.nn.Linear):
                return mod, name
    return None, None


def _get_num_heads(model) -> int:
    cfg = model.config
    for k in ("num_attention_heads", "n_head", "num_heads"):
        if hasattr(cfg, k):
            return int(getattr(cfg, k))
    raise RuntimeError("Cannot determine num_attention_heads")


def _get_num_kv_heads(model) -> int:
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


def _to_fp32(w: torch.Tensor) -> torch.Tensor:
    return w.detach().to(dtype=torch.float32, device="cpu")


def extract_layer_weights(layer, model) -> Dict[str, torch.Tensor]:
    """Return Q, K, V, O per-head tensors for a single transformer layer.

    Shapes:
      Q: (n_q_heads,  d_head, d_model)
      K: (n_kv_heads, d_head, d_model)
      V: (n_kv_heads, d_head, d_model)
      O: (n_q_heads,  d_model, d_head)   -- sliced from W_O (d_model, n_q_heads*d_head)

    Conventions (HuggingFace): nn.Linear.weight is (out, in), so
      q_proj: (n_q_heads * d_head, d_model)  → reshape (n_q, d_head, d_model)
      o_proj: (d_model, n_q_heads * d_head)  → reshape (d_model, n_q, d_head)
                                               → permute (n_q, d_model, d_head)

    Fused-QKV cases (Pythia/GPT-NeoX, some OLMo): handled below.
    """
    attn, _ = _find_attention_module(layer)
    d_model = _get_hidden_size(model)
    n_heads = _get_num_heads(model)
    n_kv_heads = _get_num_kv_heads(model)
    d_head = _get_head_dim(model)

    out: Dict[str, torch.Tensor] = {}

    # -- Q/K/V extraction --------------------------------------------------
    if all(hasattr(attn, n) for n in ("q_proj", "k_proj", "v_proj")):
        Wq = _to_fp32(attn.q_proj.weight)  # (n_heads*d_head, d_model)
        Wk = _to_fp32(attn.k_proj.weight)  # (n_kv_heads*d_head, d_model)
        Wv = _to_fp32(attn.v_proj.weight)  # (n_kv_heads*d_head, d_model)
        out["Q"] = Wq.view(n_heads, d_head, d_model)
        out["K"] = Wk.view(n_kv_heads, d_head, d_model)
        out["V"] = Wv.view(n_kv_heads, d_head, d_model)
    elif hasattr(attn, "query_key_value"):
        # GPT-NeoX / Pythia: interleaved per head.
        W = _to_fp32(attn.query_key_value.weight)  # (3*n_heads*d_head, d_model)
        W_r = W.view(n_heads, 3, d_head, d_model)
        out["Q"] = W_r[:, 0, :, :].contiguous()
        out["K"] = W_r[:, 1, :, :].contiguous()
        out["V"] = W_r[:, 2, :, :].contiguous()
    elif hasattr(attn, "att_proj"):
        # Original OLMo: concatenated [Q; K; V] along output dim.
        W = _to_fp32(attn.att_proj.weight)
        d_q = n_heads * d_head
        d_kv = n_kv_heads * d_head
        out["Q"] = W[:d_q].view(n_heads, d_head, d_model)
        out["K"] = W[d_q:d_q + d_kv].view(n_kv_heads, d_head, d_model)
        out["V"] = W[d_q + d_kv:d_q + 2 * d_kv].view(n_kv_heads, d_head, d_model)
    else:
        raise RuntimeError(
            f"Unknown attention QKV layout; attrs: "
            f"{[a for a in dir(attn) if 'proj' in a.lower() or 'qkv' in a.lower()]}"
        )

    # -- O extraction ------------------------------------------------------
    o_proj, o_name = _find_o_proj(attn)
    if o_proj is not None:
        # nn.Linear.weight is (out=d_model, in=n_heads*d_head)
        Wo = _to_fp32(o_proj.weight)  # (d_model, n_heads*d_head)
        # Reshape → (d_model, n_heads, d_head) → permute (n_heads, d_model, d_head)
        Wo_per = Wo.view(d_model, n_heads, d_head).permute(1, 0, 2).contiguous()
        out["O"] = Wo_per
    elif hasattr(attn, "attn_out"):
        # Original OLMo naming
        Wo = _to_fp32(attn.attn_out.weight)
        Wo_per = Wo.view(d_model, n_heads, d_head).permute(1, 0, 2).contiguous()
        out["O"] = Wo_per
    else:
        raise RuntimeError(
            f"Unknown attention O layout; attrs: "
            f"{[a for a in dir(attn) if 'proj' in a.lower() or 'out' in a.lower()]}"
        )

    return out


# ──────────────────────────────────────────────────────────────────────
#  Alignment math
# ──────────────────────────────────────────────────────────────────────

def row_max_abs_cos(W: torch.Tensor, v: torch.Tensor) -> float:
    """W: (d_head, d_model), v: (d_model,) unit. Return max_r |cos(v, W[r])|."""
    row_norms = W.norm(dim=1).clamp_min(1e-12)
    dots = (W @ v).abs()
    return float((dots / row_norms).max().item())


def ov_write_score(
    W_O_h: torch.Tensor,  # (d_model, d_head)
    W_V_kv: torch.Tensor,  # (d_head, d_model)
    v: torch.Tensor,       # (d_model,) unit
) -> float:
    """Fraction of the OV-circuit Frobenius norm projected onto v.

    OV = W_O_h @ W_V_kv ∈ R^{d_model x d_model}.
    Projection numerator: ||P_v OV||_F = ||v^T OV||_2 since P_v = v v^T and v unit.
    Denominator: ||OV||_F, computed without forming OV explicitly via
      ||OV||_F^2 = tr(W_V_kv^T W_O_h^T W_O_h W_V_kv)
                = tr((W_O_h^T W_O_h) (W_V_kv W_V_kv^T))
    which uses two d_head × d_head Gram matrices.
    """
    # Numerator: (v^T W_O_h) @ W_V_kv  -> (d_model,)
    left = v @ W_O_h            # (d_head,)  since W_O_h is (d_model, d_head)
    proj_vec = left @ W_V_kv    # (d_model,)
    numer = float(proj_vec.norm().item())

    # Denominator via Grams.
    G_O = W_O_h.T @ W_O_h           # (d_head, d_head)
    G_V = W_V_kv @ W_V_kv.T         # (d_head, d_head)
    fro_sq = float((G_O * G_V).sum().item())  # tr(G_O G_V) = sum of elementwise
    fro = float(np.sqrt(max(fro_sq, 0.0)))
    return numer / max(fro, 1e-12)


def compute_null_baseline(
    d_model: int,
    d_head: int,
    n_random: int,
    seed: int,
) -> Dict[str, Dict[str, float]]:
    """Random-direction baseline for (row_max_abs_cos, ov_write_score).

    Uses one fixed random (d_head, d_model) matrix for QKV-row baseline
    (by symmetry the null distribution depends only on d_model, d_head).
    For OV-write we use a fresh random (d_model, d_head) and (d_head, d_model)
    pair per null sample.
    """
    rng = np.random.default_rng(seed)

    # n_random random unit vectors in d_model
    V = rng.standard_normal((n_random, d_model)).astype(np.float32)
    V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-12

    # Fixed random (d_head, d_model) — QKV-row null
    W_null = torch.from_numpy(
        rng.standard_normal((d_head, d_model)).astype(np.float32)
    )

    # Fixed random OV null matrices
    W_O_null = torch.from_numpy(
        rng.standard_normal((d_model, d_head)).astype(np.float32)
    )
    W_V_null = torch.from_numpy(
        rng.standard_normal((d_head, d_model)).astype(np.float32)
    )

    max_cos_samples: List[float] = []
    ov_samples: List[float] = []
    for vi in V:
        v_t = torch.from_numpy(vi)
        max_cos_samples.append(row_max_abs_cos(W_null, v_t))
        ov_samples.append(ov_write_score(W_O_null, W_V_null, v_t))

    mc = np.asarray(max_cos_samples)
    ov = np.asarray(ov_samples)
    return {
        "qk_read": {
            "mean": float(mc.mean()),
            "p50": float(np.percentile(mc, 50)),
            "p95": float(np.percentile(mc, 95)),
            "p99": float(np.percentile(mc, 99)),
        },
        "ov_write": {
            "mean": float(ov.mean()),
            "p50": float(np.percentile(ov, 50)),
            "p95": float(np.percentile(ov, 95)),
            "p99": float(np.percentile(ov, 99)),
        },
    }


# ──────────────────────────────────────────────────────────────────────
#  Probe loading / training
# ──────────────────────────────────────────────────────────────────────

def _load_probe_npy(path: Path, d_model: int) -> np.ndarray:
    arr = np.load(path).astype(np.float32).flatten()
    if arr.shape[0] != d_model:
        raise RuntimeError(
            f"Probe direction {path} has shape {arr.shape}, expected ({d_model},)"
        )
    n = float(np.linalg.norm(arr))
    if n < 1e-9:
        raise RuntimeError(f"Probe direction {path} has zero norm")
    return (arr / n).astype(np.float32)


def _train_freq_probe_from_scalejsd(
    model,
    tokenizer,
    dataset_dir: str,
    probe_layer: int,
    device: str,
) -> np.ndarray:
    """Fallback: train frequency probe inline from ScaleJSD at a given layer."""
    # Imported lazily to avoid importing coverage_affinity_experiment in the
    # hot path when probe files already exist.
    try:
        from scripts.metrics.coverage_affinity_experiment import (
            load_synonym_pairs,
            pairs_to_samples,
        )
    except ImportError:
        sys.path.append(str(Path(__file__).resolve().parents[2]))
        from scripts.metrics.coverage_affinity_experiment import (
            load_synonym_pairs,
            pairs_to_samples,
        )

    all_samples: List[Dict[str, Any]] = []
    for _ds_name, fname in DATASET_FILES.items():
        p = Path(dataset_dir) / fname
        if not p.exists():
            continue
        pairs = load_synonym_pairs(str(p))
        samples = pairs_to_samples(pairs, tokenizer)
        all_samples.extend(samples)
    if not all_samples:
        raise RuntimeError(f"No ScaleJSD samples under {dataset_dir}")

    v_np, auroc, _ = train_probe_at_layer(model, all_samples, probe_layer, device)
    print(f"  [freq-fallback] trained probe at L={probe_layer}  auroc={auroc:.3f}")
    n = float(np.linalg.norm(v_np))
    return (v_np / max(n, 1e-12)).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────
#  Plotting
# ──────────────────────────────────────────────────────────────────────

def make_scatter(df: pd.DataFrame, out_png: Path) -> None:
    """2x3 panel grid: rows = {frequency, sentiment}, cols = models.
    Each panel: scatter of (qk_score / null_p99_qk) vs (ov_score / null_p99_ov).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directions = ["frequency", "sentiment"]
    models = sorted(df["model"].unique().tolist())

    fig, axes = plt.subplots(
        len(directions), len(models),
        figsize=(4.5 * len(models), 4.0 * len(directions)),
        squeeze=False,
    )
    for i, dir_name in enumerate(directions):
        for j, mdl in enumerate(models):
            ax = axes[i][j]
            sub = df[(df["model"] == mdl) & (df["direction"] == dir_name)]
            # Collapse to per-head max over Q and K for the QK axis.
            qk_rows = sub[sub["score_type"].isin(["Q_read", "K_read"])]
            ov_rows = sub[sub["score_type"] == "OV_write"]
            if qk_rows.empty or ov_rows.empty:
                ax.set_title(f"{dir_name} · {mdl}  (no data)")
                continue
            qk_agg = qk_rows.groupby(["layer", "head"])["excess"].max().reset_index()
            ov_agg = ov_rows.groupby(["layer", "head"])["excess"].max().reset_index()
            m = qk_agg.merge(ov_agg, on=["layer", "head"], suffixes=("_qk", "_ov"))
            ax.scatter(m["excess_qk"], m["excess_ov"], s=6, alpha=0.45)
            ax.axhline(1.0, color="grey", lw=0.5, ls="--")
            ax.axvline(1.0, color="grey", lw=0.5, ls="--")
            ax.set_xlabel("Q/K-read excess (× null p99)")
            ax.set_ylabel("OV-write excess (× null p99)")
            ax.set_title(f"{dir_name} · {mdl}")
            # Log scale shows the tails; fall back to linear if too few points.
            try:
                ax.set_xscale("log")
                ax.set_yscale("log")
            except Exception:
                pass
    fig.suptitle("Q/K-vs-OV dissociation (head-level)", y=1.02, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True,
                   help="HF model id (e.g. allenai/OLMo-7B-hf)")
    p.add_argument("--revision", default="main")
    p.add_argument("--freq-probe-path", default=None,
                   help="Path to probe_direction.npy for frequency direction")
    p.add_argument("--sent-probe-path", default=None,
                   help="Path to sentiment_probe_direction.npy")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--n-null", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    # Fallback: if freq probe is missing, train from ScaleJSD at this layer.
    p.add_argument("--freq-fallback-dataset-dir", default=None,
                   help="ScaleJSD filtered dataset dir (fallback if --freq-probe-path missing)")
    p.add_argument("--freq-fallback-layer", type=int, default=None,
                   help="Layer L* at which to train fallback frequency probe")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    tok_kw: Dict[str, Any] = {}
    load_kw: Dict[str, Any] = {"torch_dtype": torch.float32}
    if args.revision and args.revision != "main":
        tok_kw["revision"] = args.revision
        load_kw["revision"] = args.revision
    if hf_token:
        tok_kw["token"] = hf_token
        load_kw["token"] = hf_token

    print(f"[load] tokenizer {args.model} (rev={args.revision})", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, **tok_kw)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[load] model {args.model} fp32 on {args.device}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kw)
    model = model.to(args.device)
    model.eval()

    d_model = _get_hidden_size(model)
    n_heads = _get_num_heads(model)
    n_kv_heads = _get_num_kv_heads(model)
    d_head = _get_head_dim(model)
    n_layers = int(model.config.num_hidden_layers)
    print(
        f"[arch] d_model={d_model}  n_heads={n_heads}  n_kv={n_kv_heads}  "
        f"d_head={d_head}  n_layers={n_layers}",
        flush=True,
    )

    # ── Resolve direction vectors ────────────────────────────────────
    directions: Dict[str, np.ndarray] = {}

    if args.freq_probe_path and Path(args.freq_probe_path).exists():
        directions["frequency"] = _load_probe_npy(
            Path(args.freq_probe_path), d_model)
        print(f"[freq] loaded probe from {args.freq_probe_path}", flush=True)
    elif args.freq_fallback_dataset_dir and args.freq_fallback_layer is not None:
        print(
            f"[freq] probe missing → fallback train at layer "
            f"{args.freq_fallback_layer} from {args.freq_fallback_dataset_dir}",
            flush=True,
        )
        directions["frequency"] = _train_freq_probe_from_scalejsd(
            model, tokenizer,
            args.freq_fallback_dataset_dir,
            int(args.freq_fallback_layer),
            args.device,
        )
    else:
        print("[freq] no probe path + no fallback configured → SKIPPING frequency",
              flush=True)

    if args.sent_probe_path and Path(args.sent_probe_path).exists():
        directions["sentiment"] = _load_probe_npy(
            Path(args.sent_probe_path), d_model)
        print(f"[sent] loaded probe from {args.sent_probe_path}", flush=True)
    else:
        print("[sent] no probe path → SKIPPING sentiment", flush=True)

    if not directions:
        raise RuntimeError(
            "No probe directions resolved. Provide --freq-probe-path and/or "
            "--sent-probe-path, or --freq-fallback-* for inline training."
        )

    # Save the directions we used, for provenance.
    for name, v in directions.items():
        np.save(Path(args.output_dir) / f"direction_{name}.npy", v)

    # ── Null baseline ───────────────────────────────────────────────
    print(f"[null] computing baseline (n={args.n_null})", flush=True)
    null_stats = compute_null_baseline(
        d_model=d_model, d_head=d_head, n_random=args.n_null, seed=args.seed,
    )
    print(
        f"  qk_read   mean={null_stats['qk_read']['mean']:.4f}  "
        f"p99={null_stats['qk_read']['p99']:.4f}",
        flush=True,
    )
    print(
        f"  ov_write  mean={null_stats['ov_write']['mean']:.4f}  "
        f"p99={null_stats['ov_write']['p99']:.4f}",
        flush=True,
    )

    # Pre-convert direction to torch for matmuls.
    v_torch = {name: torch.from_numpy(v) for name, v in directions.items()}

    # ── Sweep all layers ────────────────────────────────────────────
    transformer_layers = find_transformer_layers(model)
    model_short = args.model.split("/")[-1]

    # Mapping from Q-head index → KV-head index for GQA.
    if n_heads % n_kv_heads != 0:
        raise RuntimeError(
            f"n_heads={n_heads} not divisible by n_kv_heads={n_kv_heads}")
    group_size = n_heads // n_kv_heads

    rows: List[Dict[str, Any]] = []
    for L in range(n_layers):
        try:
            W = extract_layer_weights(transformer_layers[L], model)
        except Exception as e:
            print(f"[warn] layer {L} extract failed: {e}", flush=True)
            continue
        Q = W["Q"]  # (n_heads, d_head, d_model)
        K = W["K"]  # (n_kv_heads, d_head, d_model)
        V = W["V"]  # (n_kv_heads, d_head, d_model)
        O = W["O"]  # (n_heads,  d_model, d_head)

        for dir_name, v_t in v_torch.items():
            p99_qk = null_stats["qk_read"]["p99"]
            p99_ov = null_stats["ov_write"]["p99"]

            for h in range(n_heads):
                kv_h = h // group_size
                # Q-read: row-max |cos| over W_Q^h rows vs v
                q_score = row_max_abs_cos(Q[h], v_t)
                # K-read: row-max |cos| over W_K^{kv(h)} rows vs v
                k_score = row_max_abs_cos(K[kv_h], v_t)
                # OV-write: ||P_v (W_O^h W_V^{kv(h)})||_F / ||.||_F
                ov_score = ov_write_score(O[h], V[kv_h], v_t)

                for stype, sval, p99 in (
                    ("Q_read",  q_score,  p99_qk),
                    ("K_read",  k_score,  p99_qk),
                    ("OV_write", ov_score, p99_ov),
                ):
                    rows.append({
                        "model": model_short,
                        "revision": args.revision,
                        "layer": int(L),
                        "head": int(h),
                        "kv_head": int(kv_h),
                        "direction": dir_name,
                        "score_type": stype,
                        "score": float(sval),
                        "null_mean": null_stats[
                            "qk_read" if stype != "OV_write" else "ov_write"
                        ]["mean"],
                        "null_p99": float(p99),
                        "excess": float(sval / max(p99, 1e-12)),
                        "significant": bool(sval > p99),
                    })

        if L % 4 == 0 or L == n_layers - 1:
            print(f"  [sweep] layer {L}/{n_layers - 1}", flush=True)

    df = pd.DataFrame(rows)
    out_csv = Path(args.output_dir) / "qk_vs_ov_scores.csv"
    df.to_csv(out_csv, index=False)
    print(f"[write] {out_csv}  ({len(df)} rows)", flush=True)

    # ── Dissociation summary ────────────────────────────────────────
    summary: Dict[str, Any] = {
        "model": args.model,
        "revision": args.revision,
        "model_short": model_short,
        "d_model": d_model, "n_heads": n_heads, "n_kv_heads": n_kv_heads,
        "d_head": d_head, "n_layers": n_layers,
        "null_stats": null_stats,
        "directions_analyzed": sorted(directions.keys()),
        "verdicts": {},
    }

    for dir_name in directions.keys():
        sub = df[df["direction"] == dir_name]
        # For each head, take max_qk = max(Q, K) excess and ov_excess.
        qk_max = (
            sub[sub["score_type"].isin(["Q_read", "K_read"])]
            .groupby(["layer", "head"])["excess"].max()
        )
        ov = (
            sub[sub["score_type"] == "OV_write"]
            .groupby(["layer", "head"])["excess"].first()
        )
        mean_qk = float(qk_max.mean())
        mean_ov = float(ov.mean())
        ratio = mean_qk / max(mean_ov, 1e-12)
        n_heads_significant_qk = int(
            sub[(sub["score_type"].isin(["Q_read", "K_read"]))
                & sub["significant"]]
            .groupby(["layer", "head"]).size().shape[0]
        )
        n_heads_significant_ov = int(
            sub[(sub["score_type"] == "OV_write") & sub["significant"]]
            .shape[0]
        )
        # Predicted regime
        if dir_name == "frequency":
            prediction = "Q/K >> OV (routing)"
            holds = ratio > 1.0
        elif dir_name == "sentiment":
            prediction = "OV >> Q/K (aggregation)"
            holds = ratio < 1.0
        else:
            prediction = "unclassified"
            holds = None
        summary["verdicts"][dir_name] = {
            "mean_qk_excess": mean_qk,
            "mean_ov_excess": mean_ov,
            "qk_over_ov_ratio": ratio,
            "n_heads_significant_qk": n_heads_significant_qk,
            "n_heads_significant_ov": n_heads_significant_ov,
            "prediction": prediction,
            "dissociation_holds": bool(holds) if holds is not None else None,
        }

    # Headline: does BOTH concepts dissociate in the predicted direction?
    freq_v = summary["verdicts"].get("frequency", {})
    sent_v = summary["verdicts"].get("sentiment", {})
    summary["headline"] = {
        "freq_qk_over_ov": freq_v.get("qk_over_ov_ratio"),
        "sent_qk_over_ov": sent_v.get("qk_over_ov_ratio"),
        "dissociation_ratio": (
            (freq_v.get("qk_over_ov_ratio", 0) or 0)
            / max(sent_v.get("qk_over_ov_ratio", 1) or 1, 1e-12)
        ) if freq_v and sent_v else None,
        "both_predictions_hold": (
            bool(freq_v.get("dissociation_holds"))
            and bool(sent_v.get("dissociation_holds"))
        ) if freq_v and sent_v else None,
    }

    with open(Path(args.output_dir) / "dissociation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[write] {Path(args.output_dir) / 'dissociation_summary.json'}",
          flush=True)

    print("\n=== dissociation verdicts ===")
    for dname, v in summary["verdicts"].items():
        print(
            f"  {dname}: mean_qk={v['mean_qk_excess']:.3f}  "
            f"mean_ov={v['mean_ov_excess']:.3f}  "
            f"qk/ov={v['qk_over_ov_ratio']:.3f}  "
            f"predicts={v['prediction']}  holds={v['dissociation_holds']}"
        )
    print(f"  headline: {summary['headline']}")

    # ── Scatter plot ────────────────────────────────────────────────
    try:
        out_png = Path(args.output_dir) / "qk_vs_ov_scatter.png"
        make_scatter(df, out_png)
        print(f"[write] {out_png}", flush=True)
    except Exception as e:
        print(f"[warn] scatter failed: {e}", flush=True)


if __name__ == "__main__":
    main()
