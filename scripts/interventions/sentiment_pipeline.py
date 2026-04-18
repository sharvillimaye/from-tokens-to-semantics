#!/usr/bin/env python3
"""
Sentiment-control pipeline for Idea 6 (universal-feature taxonomy).

Runs the EXACT SAME three-stage analysis pipeline we have for the frequency
direction, but targets the SENTIMENT direction (Tigges et al. 2023 IMDB
protocol). Comparing the frequency vs sentiment write-site / reader-site
patterns directly tests the hypothesis:

    "Write-site architecture (MLP-dominant vs attention-dominant) is
     concept-dependent, not model-dependent."

Stages (per model, single run):

  (A) Sentiment probe extraction  — Tigges 2023 protocol on IMDB:
        - Sample N_PER_CLASS positive / N_PER_CLASS negative reviews
        - Truncate to MAX_TOKENS tokens
        - Single forward per sample; capture residual at EVERY layer at last token
        - Train binary probe per layer (5-fold GroupShuffleSplit-safe CV)
        - Select L* (peak CV AUROC). v_sent = unit-normalized mean-of-CV-coefs.
        Outputs: sentiment_probe_per_layer.csv, sentiment_probe_direction.npy,
                 sentiment_probe.json

  (B) Signed DFA attribution (mirrors frequency_circuit_eap.py):
        Metric M(x) = <resid[L*, last_tok], v_sent>.
        For each (L, head) and (L, MLP), attribution
          a_c = <E_x[grad_c(M)], mean_pos(c) - mean_neg(c)>
        Positive ⇒ component writes toward POSITIVE sentiment.
        Also runs single-layer rank-1 ablation (project out v at L*) and
        measures downstream M trajectory → recovery_attribution.
        Outputs: sentiment_attribution_scores.csv, sentiment_recovery_attribution.csv

  (C) QKV reader analysis (mirrors qkv_reader_analysis.py):
        For every attention head at layers L > L*:
          max_cos_W = max_r |cos(v_sent, W[r,:])|,  matrix_proj = ||Wv||/||W||_F
        Null baseline: 100 random Gaussian unit vectors. p99 threshold.
        Outputs: sentiment_qkv_projections.csv, sentiment_top_readers.csv,
                 sentiment_qkv_summary.json

Usage:
  python -m scripts.interventions.sentiment_pipeline \
      --model allenai/OLMo-7B-hf --revision main \
      --output-dir /workspace/runs/sentiment_olmo-7b \
      --n-per-class 1000 --max-tokens 256 --device cuda
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

# Reuse hooks/utilities from the frequency pipeline.
try:
    from scripts.interventions.frequency_steering import (
        ResidualCapture,
        find_transformer_layers,
        _select_dtype,
    )
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from scripts.interventions.frequency_steering import (
        ResidualCapture,
        find_transformer_layers,
        _select_dtype,
    )


# ─────────────────────────────────────────────────────────────────────
#  IMDB loading + tokenization
# ─────────────────────────────────────────────────────────────────────

def load_imdb_samples(
    tokenizer,
    n_per_class: int = 1000,
    max_tokens: int = 256,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Load balanced IMDB train split via datasets; tokenize + truncate.

    Returns a list of sample dicts:
      {"token_ids": List[int], "label": int (0=neg, 1=pos), "idx": int}
    """
    from datasets import load_dataset

    ds = load_dataset("stanfordnlp/imdb", split="train")
    rng = np.random.default_rng(seed)

    pos_indices = [i for i, y in enumerate(ds["label"]) if y == 1]
    neg_indices = [i for i, y in enumerate(ds["label"]) if y == 0]
    rng.shuffle(pos_indices)
    rng.shuffle(neg_indices)
    pos_indices = pos_indices[:n_per_class]
    neg_indices = neg_indices[:n_per_class]

    samples: List[Dict[str, Any]] = []
    for idx in pos_indices + neg_indices:
        text = ds[int(idx)]["text"]
        ids = tokenizer.encode(
            text, add_special_tokens=True, truncation=True, max_length=max_tokens
        )
        if len(ids) < 4:
            continue
        samples.append({
            "token_ids": ids,
            "label": int(ds[int(idx)]["label"]),
            "idx": int(idx),
        })

    # Shuffle once so hi/lo aren't contiguous (helps gradient-pass noise).
    rng.shuffle(samples)
    return samples


# ─────────────────────────────────────────────────────────────────────
#  Residual extraction (single pass per sample, all layers, LAST TOKEN)
# ─────────────────────────────────────────────────────────────────────

def extract_all_layer_residuals_last_tok(
    model, samples: List[Dict[str, Any]], layers: List[int], device: str,
) -> Dict[int, np.ndarray]:
    """One forward per sample; extract last-token residual at every layer.

    Returns {layer: np.ndarray of shape [N, hidden]} in fp32 (CPU).
    """
    cap = ResidualCapture(model, layers)
    per_layer: Dict[int, List[np.ndarray]] = {L: [] for L in layers}
    try:
        for si, s in enumerate(samples):
            cap.clear()
            ids = torch.tensor([s["token_ids"]], device=device)
            with torch.no_grad():
                model(input_ids=ids)
            for L in layers:
                h = cap.outputs[L]
                per_layer[L].append(h[0, -1, :].float().cpu().numpy())
            if (si + 1) % 100 == 0:
                print(f"    residual extract: {si + 1}/{len(samples)}", flush=True)
    finally:
        cap.remove()
    return {L: np.stack(v) for L, v in per_layer.items()}


# ─────────────────────────────────────────────────────────────────────
#  5-fold stratified probe (no groups for IMDB — each review is its own unit)
# ─────────────────────────────────────────────────────────────────────

def cv_probe_stratified(
    X: np.ndarray, y: np.ndarray, n_splits: int = 5, seed: int = 42,
) -> Tuple[np.ndarray, float]:
    """5-fold stratified CV probe. Returns (unit-normalized mean coef, mean test AUROC)."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    aurocs, coefs = [], []
    for tr, te in skf.split(X, y):
        if y[tr].sum() < 2 or (1 - y[tr]).sum() < 2:
            continue
        if y[te].sum() < 1 or (1 - y[te]).sum() < 1:
            continue
        clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
        clf.fit(X[tr], y[tr])
        aurocs.append(roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1]))
        coefs.append(clf.coef_[0])
    if not coefs:
        return np.zeros(X.shape[1], dtype=np.float32), float("nan")
    v = np.mean(coefs, axis=0)
    v = v / (np.linalg.norm(v) + 1e-12)
    return v.astype(np.float32), float(np.mean(aurocs))


def find_peak_sentiment_layer(
    model, samples: List[Dict[str, Any]], layers: List[int], device: str, seed: int = 42,
) -> Tuple[int, float, Dict[int, float], np.ndarray, Dict[int, np.ndarray]]:
    """Sweep all layers; return (L*, AUROC*, all layer AUROCs, v_L*, per-layer X)."""
    per_layer_X = extract_all_layer_residuals_last_tok(model, samples, layers, device)
    y = np.array([s["label"] for s in samples])

    layer_aurocs: Dict[int, float] = {}
    best_L, best_auroc, best_v = -1, -1.0, None
    for L in layers:
        v, auroc = cv_probe_stratified(per_layer_X[L], y, seed=seed)
        layer_aurocs[L] = auroc
        if not np.isnan(auroc) and auroc > best_auroc:
            best_L, best_auroc, best_v = L, auroc, v
        print(f"    L{L}: sentiment_auroc={auroc:.3f}", flush=True)
    if best_v is None:
        raise RuntimeError("No layer produced a valid probe")
    return best_L, best_auroc, layer_aurocs, best_v, per_layer_X


# ─────────────────────────────────────────────────────────────────────
#  Stage B: Signed DFA attribution
#  (Direct port of frequency_circuit_eap.py logic, last-token metric)
# ─────────────────────────────────────────────────────────────────────

def _find_attention_module(layer):
    for name in ("self_attn", "attention", "attn"):
        if hasattr(layer, name):
            return getattr(layer, name), name
    raise RuntimeError(f"No attention module in {layer}")


def _find_mlp_module(layer):
    for name in ("mlp", "feed_forward", "ff"):
        if hasattr(layer, name):
            return getattr(layer, name), name
    raise RuntimeError(f"No mlp module in {layer}")


def _find_o_proj(attn_module):
    for name in ("o_proj", "out_proj", "dense", "c_proj", "wo"):
        if hasattr(attn_module, name):
            mod = getattr(attn_module, name)
            if isinstance(mod, torch.nn.Linear):
                return mod, name
    raise RuntimeError(f"No attention output projection in {type(attn_module)}")


def _get_num_heads(model) -> int:
    cfg = model.config
    for k in ("num_attention_heads", "n_head", "num_heads"):
        if hasattr(cfg, k):
            return int(getattr(cfg, k))
    raise RuntimeError("Cannot determine num_attention_heads")


def _get_hidden_size(model) -> int:
    cfg = model.config
    for k in ("hidden_size", "n_embd", "d_model"):
        if hasattr(cfg, k):
            return int(getattr(cfg, k))
    raise RuntimeError("Cannot determine hidden size")


class AttributionHooks:
    """Capture per-layer attn-pre-O input and MLP output with retained grad."""

    def __init__(self, model, layers: List[int]):
        self.layers = layers
        self.hooks = []
        self.attn_z_in: Dict[int, torch.Tensor] = {}
        self.mlp_z: Dict[int, torch.Tensor] = {}
        tf = find_transformer_layers(model)
        for L in layers:
            layer = tf[L]
            attn_module, _ = _find_attention_module(layer)
            o_proj, _ = _find_o_proj(attn_module)
            mlp_module, _ = _find_mlp_module(layer)

            def make_attn_hook(idx):
                def hook(mod, inp, out):
                    x = inp[0] if isinstance(inp, tuple) else inp
                    if x.requires_grad:
                        x.retain_grad()
                    self.attn_z_in[idx] = x
                return hook
            self.hooks.append(o_proj.register_forward_hook(make_attn_hook(L)))

            def make_mlp_hook(idx):
                def hook(mod, inp, out):
                    y = out[0] if isinstance(out, tuple) else out
                    if y.requires_grad:
                        y.retain_grad()
                    self.mlp_z[idx] = y
                return hook
            self.hooks.append(mlp_module.register_forward_hook(make_mlp_hook(L)))

    def clear(self):
        self.attn_z_in.clear()
        self.mlp_z.clear()

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


def compute_sentiment_attribution(
    model,
    samples: List[Dict[str, Any]],
    best_L: int,
    v_best: np.ndarray,
    layers: List[int],
    device: str,
    max_samples_grad: int = 200,
) -> pd.DataFrame:
    """Two-pass attribution patching; metric is <resid[L*, last_tok], v_sent>.

    Pass 1 (no grad): collect last-token attn-pre-O and MLP outputs per layer
                      per sample; compute class-mean deltas Δ = mean_pos - mean_neg.
    Pass 2 (grad): for each of up to max_samples_grad samples, compute
                   M = <resid[L*, last_tok], v>, backprop, accumulate
                   attribution = grad · Δ for each (L, head) and (L, MLP).
    """
    n_heads = _get_num_heads(model)
    hidden = _get_hidden_size(model)
    d_head = hidden // n_heads
    v_t = torch.tensor(v_best, device=device, dtype=torch.float32)
    labels = np.array([s["label"] for s in samples])

    print(f"  Pass 1: clean last-token activations ({len(samples)} samples)", flush=True)
    hooks = AttributionHooks(model, layers)
    attn_z_last: Dict[int, List[torch.Tensor]] = {L: [] for L in layers}
    mlp_z_last: Dict[int, List[torch.Tensor]] = {L: [] for L in layers}

    model.eval()
    for si, s in enumerate(samples):
        hooks.clear()
        ids = torch.tensor([s["token_ids"]], device=device)
        with torch.no_grad():
            model(input_ids=ids)
        for L in layers:
            attn_z_last[L].append(hooks.attn_z_in[L][0, -1].detach().float().cpu())
            mlp_z_last[L].append(hooks.mlp_z[L][0, -1].detach().float().cpu())
        if (si + 1) % 100 == 0:
            print(f"    {si + 1}/{len(samples)}", flush=True)
    hooks.remove()

    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]
    attn_delta: Dict[int, torch.Tensor] = {}
    mlp_delta: Dict[int, torch.Tensor] = {}
    for L in layers:
        A = torch.stack(attn_z_last[L])
        Mz = torch.stack(mlp_z_last[L])
        attn_delta[L] = A[pos_idx].mean(0) - A[neg_idx].mean(0)
        mlp_delta[L] = Mz[pos_idx].mean(0) - Mz[neg_idx].mean(0)

    # Pass 2 — gradient. Cap samples for budget; sample half from each class.
    if len(samples) > max_samples_grad:
        rng = np.random.default_rng(42)
        per_class = max_samples_grad // 2
        gi = rng.choice(pos_idx, size=min(per_class, len(pos_idx)), replace=False).tolist()
        gi += rng.choice(neg_idx, size=min(per_class, len(neg_idx)), replace=False).tolist()
        rng.shuffle(gi)
        grad_samples = [samples[i] for i in gi]
    else:
        grad_samples = samples
    print(f"  Pass 2: gradient pass ({len(grad_samples)} samples)", flush=True)

    tf = find_transformer_layers(model)
    attn_head_attr = {L: torch.zeros(n_heads, dtype=torch.float32) for L in layers}
    mlp_attr = {L: torch.tensor(0.0, dtype=torch.float32) for L in layers}
    n_used = 0

    for si, s in enumerate(grad_samples):
        hooks = AttributionHooks(model, layers)
        pass_cap: Dict[str, torch.Tensor] = {}

        def make_res_hook():
            def hook(_, __, out):
                h = out[0] if isinstance(out, tuple) else out
                h.retain_grad()
                pass_cap["h"] = h
            return hook

        res_hook = tf[best_L].register_forward_hook(make_res_hook())
        try:
            model.zero_grad(set_to_none=True)
            ids = torch.tensor([s["token_ids"]], device=device)
            _out = model(input_ids=ids)
            h_res = pass_cap["h"]
            resid_last = h_res[0, -1].float()
            M_scalar = torch.dot(resid_last, v_t.float())
            M_scalar.backward()

            for L in layers:
                gA = hooks.attn_z_in[L].grad if L in hooks.attn_z_in else None
                if gA is not None:
                    g_last = gA[0, -1].detach().float().cpu()
                    g_h = g_last.view(n_heads, d_head)
                    d_h = attn_delta[L].view(n_heads, d_head)
                    attn_head_attr[L] += (g_h * d_h).sum(dim=-1)

                gM = hooks.mlp_z[L].grad if L in hooks.mlp_z else None
                if gM is not None:
                    g_m = gM[0, -1].detach().float().cpu()
                    mlp_attr[L] += float((g_m * mlp_delta[L]).sum())
            n_used += 1
            del _out, h_res, resid_last, M_scalar
        finally:
            res_hook.remove()
            hooks.remove()

        if (si + 1) % 50 == 0:
            print(f"    {si + 1}/{len(grad_samples)}", flush=True)

    rows = []
    n = max(n_used, 1)
    for L in layers:
        per_head = attn_head_attr[L] / n
        for h_idx in range(n_heads):
            score = float(per_head[h_idx])
            rows.append({
                "layer": L,
                "component_type": "attn_head",
                "component_index": h_idx,
                "attribution_score": score,
                "abs_score": abs(score),
            })
        mscore = float(mlp_attr[L] / n)
        rows.append({
            "layer": L,
            "component_type": "mlp",
            "component_index": 0,
            "attribution_score": mscore,
            "abs_score": abs(mscore),
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────
#  Rank-1 ablation → recovery attribution
# ─────────────────────────────────────────────────────────────────────

class ProjectOutHook:
    """Project out unit direction v from residual at a given layer's output."""

    def __init__(self, model, layer: int, v: torch.Tensor):
        self.v = v
        tf = find_transformer_layers(model)
        self._h = tf[layer].register_forward_hook(self._hook)

    def _hook(self, _, __, out):
        if isinstance(out, tuple):
            x = out[0]
            rest = out[1:]
        else:
            x = out
            rest = None
        v = self.v.to(device=x.device, dtype=x.dtype)
        coef = (x * v).sum(dim=-1, keepdim=True)
        x_new = x - coef * v
        if rest is None:
            return x_new
        return (x_new, *rest)

    def remove(self):
        self._h.remove()


def compute_recovery_attribution(
    model,
    samples: List[Dict[str, Any]],
    best_L: int,
    v_best: np.ndarray,
    layers: List[int],
    device: str,
    subset_n: int = 200,
) -> pd.DataFrame:
    """Rank-1 project-out at L*; measure <resid[l, last_tok], v_sent> clean vs ablated.

    delta_M positive at downstream l ⇒ the direction gets REBUILT after ablation.
    """
    v_t = torch.tensor(v_best, device=device, dtype=torch.float32)

    if len(samples) > subset_n:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(samples), size=subset_n, replace=False).tolist()
        sub = [samples[i] for i in idx]
    else:
        sub = samples
    labels = np.array([s["label"] for s in sub])

    print(f"  Recovery: clean pass (n={len(sub)})", flush=True)
    cap_clean = ResidualCapture(model, layers)
    clean_M: Dict[int, List[float]] = {L: [] for L in layers}
    try:
        for s in sub:
            cap_clean.clear()
            with torch.no_grad():
                model(input_ids=torch.tensor([s["token_ids"]], device=device))
            for L in layers:
                h = cap_clean.outputs[L]
                x = h[0, -1].float()
                clean_M[L].append(float(torch.dot(x, v_t.float().to(x.device))))
    finally:
        cap_clean.remove()

    print(f"  Recovery: ablated pass (project out v at L{best_L})", flush=True)
    cap_abl = ResidualCapture(model, layers)
    abl_hook = ProjectOutHook(model, best_L, v_t)
    abl_M: Dict[int, List[float]] = {L: [] for L in layers}
    try:
        for s in sub:
            cap_abl.clear()
            with torch.no_grad():
                model(input_ids=torch.tensor([s["token_ids"]], device=device))
            for L in layers:
                h = cap_abl.outputs[L]
                x = h[0, -1].float()
                abl_M[L].append(float(torch.dot(x, v_t.float().to(x.device))))
    finally:
        cap_abl.remove()
        abl_hook.remove()

    rows = []
    for L in layers:
        c = np.array(clean_M[L])
        a = np.array(abl_M[L])
        rows.append({
            "layer": int(L),
            "clean_M_mean": float(c.mean()),
            "ablated_M_mean": float(a.mean()),
            "delta_M_mean": float((a - c).mean()),
            "clean_M_diff_pos_neg": float(c[labels == 1].mean() - c[labels == 0].mean()),
            "ablated_M_diff_pos_neg": float(a[labels == 1].mean() - a[labels == 0].mean()),
            "delta_M_diff_pos_neg": float(
                (a[labels == 1] - c[labels == 1]).mean()
                - (a[labels == 0] - c[labels == 0]).mean()
            ),
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────
#  Stage C: QKV reader analysis
# ─────────────────────────────────────────────────────────────────────

def _get_num_kv_heads(model) -> int:
    cfg = model.config
    for k in ("num_key_value_heads", "num_kv_heads"):
        if hasattr(cfg, k):
            val = getattr(cfg, k)
            if val is not None:
                return int(val)
    return _get_num_heads(model)


def _get_head_dim(model) -> int:
    cfg = model.config
    if hasattr(cfg, "head_dim") and getattr(cfg, "head_dim") is not None:
        return int(cfg.head_dim)
    return _get_hidden_size(model) // _get_num_heads(model)


def extract_qkv_weights(layer, model) -> Dict[str, torch.Tensor]:
    """Per-head Q/K/V weights: dict of (n_heads_for_stream, d_head, d_model) fp32 CPU."""
    attn, _ = _find_attention_module(layer)
    d_model = _get_hidden_size(model)
    n_heads = _get_num_heads(model)
    n_kv_heads = _get_num_kv_heads(model)
    d_head = _get_head_dim(model)

    def _to_fp32(w):
        return w.detach().to(dtype=torch.float32, device="cpu")

    if all(hasattr(attn, n) for n in ("q_proj", "k_proj", "v_proj")):
        Wq = _to_fp32(attn.q_proj.weight)
        Wk = _to_fp32(attn.k_proj.weight)
        Wv = _to_fp32(attn.v_proj.weight)
        Q = Wq.view(n_heads, d_head, d_model)
        K = Wk.view(n_kv_heads, d_head, d_model)
        V = Wv.view(n_kv_heads, d_head, d_model)
        return {"Q": Q, "K": K, "V": V}

    if hasattr(attn, "query_key_value"):
        W = _to_fp32(attn.query_key_value.weight)
        W_r = W.view(n_heads, 3, d_head, d_model)
        Q = W_r[:, 0, :, :].contiguous()
        K = W_r[:, 1, :, :].contiguous()
        V = W_r[:, 2, :, :].contiguous()
        return {"Q": Q, "K": K, "V": V}

    if hasattr(attn, "att_proj"):
        W = _to_fp32(attn.att_proj.weight)
        d_q = n_heads * d_head
        d_kv = n_kv_heads * d_head
        Wq = W[:d_q]
        Wk = W[d_q:d_q + d_kv]
        Wv = W[d_q + d_kv:d_q + 2 * d_kv]
        Q = Wq.view(n_heads, d_head, d_model)
        K = Wk.view(n_kv_heads, d_head, d_model)
        V = Wv.view(n_kv_heads, d_head, d_model)
        return {"Q": Q, "K": K, "V": V}

    raise RuntimeError(f"Unknown attention layout: {type(attn)}")


def row_max_abs_cos(W: torch.Tensor, v: torch.Tensor) -> float:
    row_norms = W.norm(dim=1).clamp_min(1e-12)
    dots = (W @ v).abs()
    cos = dots / row_norms
    return float(cos.max().item())


def matrix_projection(W: torch.Tensor, v: torch.Tensor) -> float:
    numer = (W @ v).norm().item()
    denom = W.norm().item()
    return float(numer / max(denom, 1e-12))


def compute_null_baseline(d_model: int, d_head: int, n_random: int, seed: int
                          ) -> Dict[str, Dict[str, float]]:
    rng = np.random.default_rng(seed)
    V = rng.standard_normal((n_random, d_model)).astype(np.float32)
    V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-12
    W_null = torch.from_numpy(rng.standard_normal((d_head, d_model)).astype(np.float32))

    mc_s, mp_s = [], []
    for vi in V:
        v_t = torch.from_numpy(vi)
        mc_s.append(row_max_abs_cos(W_null, v_t))
        mp_s.append(matrix_projection(W_null, v_t))
    mc = np.asarray(mc_s)
    mp = np.asarray(mp_s)
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


def run_qkv_reader(
    model,
    v_best: np.ndarray,
    best_L: int,
    n_layers: int,
    output_dir: Path,
    model_short: str,
    revision: str,
    n_null: int = 100,
    top_k: int = 20,
    seed: int = 42,
) -> Dict[str, Any]:
    d_model = _get_hidden_size(model)
    n_heads = _get_num_heads(model)
    n_kv_heads = _get_num_kv_heads(model)
    d_head = _get_head_dim(model)

    print(f"  QKV: d_model={d_model} n_heads={n_heads} n_kv={n_kv_heads} d_head={d_head}")
    null_stats = compute_null_baseline(d_model, d_head, n_null, seed)
    print(f"  QKV: null max_cos p99={null_stats['max_cos']['p99']:.4f}")

    v_t = torch.from_numpy(v_best.astype(np.float32))
    transformer_layers = find_transformer_layers(model)

    rows: List[Dict[str, Any]] = []
    for L in range(best_L + 1, n_layers):
        qkv = extract_qkv_weights(transformer_layers[L], model)
        for w_type, W_all in qkv.items():
            n_h = W_all.shape[0]
            for h in range(n_h):
                W_h = W_all[h]
                mc = row_max_abs_cos(W_h, v_t)
                mp = matrix_projection(W_h, v_t)
                rows.append({
                    "model": model_short,
                    "revision": revision,
                    "layer": int(L),
                    "head": int(h),
                    "w_type": w_type,
                    "max_cos": mc,
                    "matrix_proj": mp,
                    "null_mean_max_cos": null_stats["max_cos"]["mean"],
                    "null_p99_max_cos": null_stats["max_cos"]["p99"],
                    "null_mean_matrix_proj": null_stats["matrix_proj"]["mean"],
                    "null_p99_matrix_proj": null_stats["matrix_proj"]["p99"],
                    "significant": bool(mc > null_stats["max_cos"]["p99"]),
                })
        if (L - best_L) % 5 == 0 or L == n_layers - 1:
            print(f"    qkv layer {L}/{n_layers - 1}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "sentiment_qkv_projections.csv", index=False)

    top_frames = []
    for w_type in ("Q", "K", "V"):
        sub = df[df["w_type"] == w_type].nlargest(top_k, "max_cos").copy()
        sub["rank"] = range(1, len(sub) + 1)
        top_frames.append(sub)
    top_df = pd.concat(top_frames, ignore_index=True)
    top_df.to_csv(output_dir / "sentiment_top_readers.csv", index=False)

    summary = {
        "model_short": model_short,
        "revision": revision,
        "d_model": d_model,
        "n_heads": n_heads,
        "n_kv_heads": n_kv_heads,
        "d_head": d_head,
        "n_layers": n_layers,
        "best_layer_L_star": int(best_L),
        "null_stats": null_stats,
        "n_heads_significant_Q": int(df[(df.w_type == "Q") & df.significant].shape[0]),
        "n_heads_significant_K": int(df[(df.w_type == "K") & df.significant].shape[0]),
        "n_heads_significant_V": int(df[(df.w_type == "V") & df.significant].shape[0]),
        "n_total_Q": int((df.w_type == "Q").sum()),
        "n_total_K": int((df.w_type == "K").sum()),
        "n_total_V": int((df.w_type == "V").sum()),
        "max_max_cos_Q": float(df[df.w_type == "Q"]["max_cos"].max()) if (df.w_type == "Q").any() else float("nan"),
        "max_max_cos_K": float(df[df.w_type == "K"]["max_cos"].max()) if (df.w_type == "K").any() else float("nan"),
        "max_max_cos_V": float(df[df.w_type == "V"]["max_cos"].max()) if (df.w_type == "V").any() else float("nan"),
    }
    with open(output_dir / "sentiment_qkv_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


# ─────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-per-class", type=int, default=1000,
                        help="IMDB samples per class (pos and neg)")
    parser.add_argument("--max-tokens", type=int, default=256,
                        help="Truncate reviews to this many tokens")
    parser.add_argument("--max-samples-grad", type=int, default=200,
                        help="Cap for grad-pass in attribution (budget control)")
    parser.add_argument("--recovery-subset", type=int, default=200)
    parser.add_argument("--n-null", type=int, default=100,
                        help="Null baseline vectors for QKV test")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="auto",
                        choices=["auto", "float32", "bfloat16", "float16"])
    parser.add_argument("--skip-attribution", action="store_true")
    parser.add_argument("--skip-recovery", action="store_true")
    parser.add_argument("--skip-qkv", action="store_true")
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.dtype == "auto":
        dtype = _select_dtype(args.model, args.device)
    else:
        dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16,
                 "float16": torch.float16}[args.dtype]
    print(f"Loading {args.model} (revision={args.revision}, dtype={dtype})", flush=True)

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    tok_kw: Dict[str, Any] = {}
    if args.revision != "main":
        tok_kw["revision"] = args.revision
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

    # ── IMDB samples ───────────────────────────────────────────────
    print(f"\n=== Loading IMDB ({args.n_per_class} per class, max_tokens={args.max_tokens}) ===")
    samples = load_imdb_samples(
        tokenizer, n_per_class=args.n_per_class, max_tokens=args.max_tokens, seed=args.seed)
    n_pos = sum(1 for s in samples if s["label"] == 1)
    n_neg = sum(1 for s in samples if s["label"] == 0)
    print(f"  Loaded {len(samples)} samples ({n_pos} pos / {n_neg} neg)", flush=True)

    n_hidden = int(model.config.num_hidden_layers)
    all_layers = list(range(n_hidden))
    model_short = args.model.split("/")[-1].replace("/", "-")

    # ── Stage A: peak sentiment layer + probe direction ────────────
    print(f"\n=== Stage A: sentiment probe sweep across {len(all_layers)} layers ===",
          flush=True)
    best_L, best_auroc, layer_aurocs, v_best, _ = find_peak_sentiment_layer(
        model, samples, all_layers, args.device, seed=args.seed)
    print(f"\n  Peak: L{best_L} (AUROC={best_auroc:.3f})", flush=True)

    pd.DataFrame([
        {"layer": int(L), "sentiment_auroc": float(a)} for L, a in sorted(layer_aurocs.items())
    ]).to_csv(out / "sentiment_probe_per_layer.csv", index=False)
    np.save(out / "sentiment_probe_direction.npy", v_best.astype(np.float32))

    probe_info = {
        "model": args.model,
        "revision": args.revision,
        "best_layer": int(best_L),
        "best_probe_auroc": float(best_auroc),
        "layer_aurocs": {str(k): float(v) for k, v in layer_aurocs.items()},
        "n_samples": int(len(samples)),
        "n_pos": int(n_pos),
        "n_neg": int(n_neg),
        "max_tokens": int(args.max_tokens),
        "hidden_size": int(_get_hidden_size(model)),
        "n_heads": int(_get_num_heads(model)),
        "n_layers": int(n_hidden),
    }
    with open(out / "sentiment_probe.json", "w") as f:
        json.dump(probe_info, f, indent=2)

    # ── Stage B: attribution ───────────────────────────────────────
    if not args.skip_attribution:
        print(f"\n=== Stage B: signed attribution onto v_sent at L{best_L} ===", flush=True)
        attr_df = compute_sentiment_attribution(
            model, samples, best_L, v_best, all_layers, args.device,
            max_samples_grad=args.max_samples_grad)
        attr_df["model"] = model_short
        attr_df = attr_df[["model", "layer", "component_type",
                           "component_index", "attribution_score", "abs_score"]]
        attr_df.to_csv(out / "sentiment_attribution_scores.csv", index=False)
        top = attr_df.sort_values("abs_score", ascending=False).head(10)
        print("\n  Top 10 components by |attribution|:")
        print(top.to_string(index=False))

    # ── Recovery ────────────────────────────────────────────────────
    if not args.skip_recovery:
        print(f"\n=== Stage B': recovery attribution (project out v_sent at L{best_L}) ===",
              flush=True)
        rec_df = compute_recovery_attribution(
            model, samples, best_L, v_best, all_layers, args.device,
            subset_n=args.recovery_subset)
        rec_df["model"] = model_short
        rec_df.to_csv(out / "sentiment_recovery_attribution.csv", index=False)
        print("\n  Recovery per-layer ΔM_diff_pos_neg (ablated - clean):")
        for _, r in rec_df.iterrows():
            print(f"    L{int(r['layer']):3d}: clean={r['clean_M_diff_pos_neg']:+.4f}  "
                  f"ablated={r['ablated_M_diff_pos_neg']:+.4f}  "
                  f"Δ={r['delta_M_diff_pos_neg']:+.4f}")

    # ── Stage C: QKV readers ───────────────────────────────────────
    if not args.skip_qkv:
        print(f"\n=== Stage C: QKV reader analysis (layers > L{best_L}) ===", flush=True)
        run_qkv_reader(
            model, v_best, best_L, n_hidden, out, model_short, args.revision,
            n_null=args.n_null, top_k=args.top_k, seed=args.seed)

    print(f"\nDone. Results in {out}", flush=True)


if __name__ == "__main__":
    main()
