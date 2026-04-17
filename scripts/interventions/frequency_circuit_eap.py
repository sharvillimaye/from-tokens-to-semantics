#!/usr/bin/env python3
"""
Frequency circuit tracing via attribution patching.

GOAL
----
Identify which transformer components (per-layer MLPs and per-head attention
outputs) write, maintain, or erase the *frequency direction* in the residual
stream at the layer of peak probe AUROC. Works on OLMo-7B, Llama-3.1-8B,
Pythia-6.9B.

LIBRARY CHOICE
--------------
We implement attribution patching directly with PyTorch forward/backward hooks
on HuggingFace models, rather than routing through TransformerLens or the EAP-IG
package. Reasons:

  1. TransformerLens' HookedTransformer requires per-architecture weight
     conversions that are brittle for OLMo-7B and bleeding-edge for Llama-3.1
     (requires ≥ TL 2.x and still occasionally breaks on rotary embed details).
     Getting all three target models to load identically eats budget.
  2. EAP-IG (Hanna 2024) depends on TransformerLens.
  3. Anthropic's `circuit-tracer` is logit-focused and requires per-model
     pretrained cross-layer transcoders, which don't exist for our models.
  4. Attribution patching is *just* gradient × activation-diff for a scalar
     metric (Syed et al. 2023, "Attribution Patching: Activation Patching at
     Industrial Scale"). A few dozen lines of hooks give us the same result as
     a specialized library, but model-agnostic.

METHOD
------
Let M(x) = <residual[L*, anchor], v_L*>  (signed projection onto the unit probe
direction at the peak-AUROC layer).

For each component c (per-head attn_z output, per-layer MLP output) at every
layer ℓ, attribution patching computes

    a_c ≈ (∂M/∂z_c)|_clean · (z_c^patch - z_c^clean)

where z_c^clean is the component's output on the low-freq input and z_c^patch
is its output on the matched high-freq input (or the class-mean thereof).
Pairs are class-matched: ScaleJSD synonym pairs naturally give (high, low)
twins with identical sentence templates. We use mean-patching for stability:
z_c^patch = mean over high-freq class samples.

This yields a SIGNED score per component:
  - positive  ⇒ component writes frequency INTO the direction (promotes
    high-freq representation at anchor)
  - negative  ⇒ component erases / writes against frequency
  - |score|   ⇒ magnitude of causal effect

OUTPUTS
-------
  attribution_scores.csv
    columns: model, layer, component_type, component_index, attribution_score,
             abs_score
    component_type ∈ {attn_head, mlp}
    component_index = head index or 0 (for mlp)

  recovery_attribution.csv
    Post-ablation recovery: after projecting out v_L* at L* on the clean forward,
    measure ΔM at every downstream layer. This tells us which downstream layers
    REBUILD the frequency direction after it's been surgically removed.
    columns: model, layer, delta_M, clean_M, ablated_M

  probe.json
    best_layer (L*), probe_auroc, layer_aurocs

Usage
-----
  python -m scripts.interventions.frequency_circuit_eap \
      --model allenai/OLMo-7B-hf --revision main \
      --dataset-dir /data/ani/mechinterp/code/scaleJSD/dataset/legacy/filtered \
      --output-dir /data/ani/mechinterp/runs/circuits/olmo-7b \
      --device cuda
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
from sklearn.model_selection import GroupShuffleSplit

# Reuse scaleJSD sample building + layer discovery
try:
    from scripts.metrics.coverage_affinity_experiment import (
        _infer_layers,
        load_synonym_pairs,
        pairs_to_samples,
    )
except ImportError:
    sys.path.append(str(Path(__file__).resolve().parents[2]))
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
#  Model layer discovery
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


def find_attention_module(layer):
    for name in ["self_attn", "attention", "attn"]:
        if hasattr(layer, name):
            return getattr(layer, name), name
    raise RuntimeError(f"Cannot find attention module in layer {layer}")


def find_mlp_module(layer):
    for name in ["mlp", "feed_forward", "ff"]:
        if hasattr(layer, name):
            return getattr(layer, name), name
    raise RuntimeError(f"Cannot find mlp module in layer {layer}")


def get_num_heads(model) -> int:
    cfg = model.config
    for k in ("num_attention_heads", "n_head", "num_heads"):
        if hasattr(cfg, k):
            return int(getattr(cfg, k))
    raise RuntimeError("Cannot determine number of attention heads")


def get_hidden_size(model) -> int:
    cfg = model.config
    for k in ("hidden_size", "n_embd", "d_model"):
        if hasattr(cfg, k):
            return int(getattr(cfg, k))
    raise RuntimeError("Cannot determine hidden size")


# ─────────────────────────────────────────────────────────────────────
#  Residual capture (for probe training + recovery analysis)
# ─────────────────────────────────────────────────────────────────────

class ResidualCapture:
    """Capture residual stream at layer outputs (post-layer)."""

    def __init__(self, model, layers: List[int]):
        self.layers = layers
        self.hooks: List = []
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
#  Probe training (group-aware CV, reused pattern)
# ─────────────────────────────────────────────────────────────────────

def _cv_probe(X: np.ndarray, labels: np.ndarray, groups: np.ndarray
              ) -> Tuple[np.ndarray, float]:
    gss = GroupShuffleSplit(n_splits=5, test_size=0.3, random_state=42)
    aurocs, coefs = [], []
    for tr, te in gss.split(X, labels, groups=groups):
        if labels[tr].sum() < 2 or (1 - labels[tr]).sum() < 2:
            continue
        if labels[te].sum() < 1 or (1 - labels[te]).sum() < 1:
            continue
        clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
        clf.fit(X[tr], labels[tr])
        aurocs.append(roc_auc_score(labels[te], clf.predict_proba(X[te])[:, 1]))
        coefs.append(clf.coef_[0])
    if not coefs:
        return np.zeros(X.shape[1]), float("nan")
    v = np.mean(coefs, axis=0)
    v = v / (np.linalg.norm(v) + 1e-12)
    return v, float(np.mean(aurocs))


def extract_all_layer_residuals(
    model, samples, layers: List[int], device: str,
) -> Dict[int, np.ndarray]:
    """One forward pass per sample; capture residual at every layer, anchor position."""
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


def find_peak_probe_layer(
    model, samples, layers: List[int], device: str,
) -> Tuple[int, float, Dict[int, float], np.ndarray]:
    """Sweep all layers, return (L*, probe_auroc, per_layer_aurocs, v_L*)."""
    per_layer_X = extract_all_layer_residuals(model, samples, layers, device)
    freq_labels = np.array(
        [1 if s["frequency_category"] == "high_freq" else 0 for s in samples])
    pair_ids = np.array([s["pair_id"] for s in samples])

    layer_aurocs: Dict[int, float] = {}
    best_L, best_auroc, best_v = -1, -1.0, None
    for L in layers:
        v, auroc = _cv_probe(per_layer_X[L], freq_labels, pair_ids)
        layer_aurocs[L] = auroc
        if not np.isnan(auroc) and auroc > best_auroc:
            best_auroc = auroc
            best_L = L
            best_v = v
        print(f"    L{L}: freq_auroc={auroc:.3f}", flush=True)
    if best_v is None:
        raise RuntimeError(
            f"No layer yielded a valid probe (all NaN). "
            f"Check class balance in samples. "
            f"n_hi={int(freq_labels.sum())}, n_lo={int((1-freq_labels).sum())}")
    return best_L, best_auroc, layer_aurocs, best_v


# ─────────────────────────────────────────────────────────────────────
#  Attribution patching core
# ─────────────────────────────────────────────────────────────────────
#
# We capture two per-layer intermediates on every sample:
#
#   - attn_out_preO: input to attention output projection (o_proj / dense / c_proj)
#     shape [B, T, H]. Reshaped to [B, T, n_heads, d_head]. Sum over heads equals
#     attention contribution to residual (pre-bias). Per-head slice = per-head
#     contribution. We patch this intermediate because it's the clean per-head
#     residual write.
#
#   - mlp_out: output of the full MLP block, shape [B, T, H].
#
# We also hook the residual at L* to compute M(x) and its gradient.
#
# Attribution score per component c:
#     a_c = <grad_c, z_c^ref - z_c^clean>
# where z_c^clean is this-sample's value and z_c^ref is the per-position MEAN
# over the OPPOSITE class (mean-patching). We take sign convention:
#     sample low-freq → patch with high-freq mean → positive score means
#     component writes toward high-freq (i.e., toward +v_L*).

def _find_o_proj(attn_module):
    """Find the output projection of attention (W_O)."""
    # HF Llama: o_proj, HF OLMo: o_proj, GPT-NeoX (Pythia): dense
    for name in ["o_proj", "out_proj", "dense", "c_proj", "wo"]:
        if hasattr(attn_module, name):
            mod = getattr(attn_module, name)
            if isinstance(mod, torch.nn.Linear):
                return mod, name
    raise RuntimeError(f"Cannot find attention output projection in {type(attn_module)}")


class AttributionHooks:
    """Register forward hooks to capture per-head attn pre-O and mlp outputs.

    Stores activations with requires_grad so we can compute gradients via autograd.
    """

    def __init__(self, model, layers: List[int], n_heads: int, hidden: int):
        self.layers = layers
        self.n_heads = n_heads
        self.hidden = hidden
        self.d_head = hidden // n_heads
        self.hooks = []

        # Captured on each forward:
        # attn_z_in[L]: [B, T, H] — input to o_proj (pre-projection concatenated heads)
        # mlp_z[L]:     [B, T, H] — output of mlp block
        self.attn_z_in: Dict[int, torch.Tensor] = {}
        self.mlp_z: Dict[int, torch.Tensor] = {}

        tf = find_transformer_layers(model)
        for L in layers:
            layer = tf[L]
            attn_module, _ = find_attention_module(layer)
            o_proj, _ = _find_o_proj(attn_module)
            mlp_module, _ = find_mlp_module(layer)

            # Capture o_proj INPUT (this is the per-head concat). Only retain_grad
            # if gradients are being tracked (not under no_grad).
            def make_attn_hook(idx):
                def hook(mod, inp, out):
                    x = inp[0] if isinstance(inp, tuple) else inp
                    if x.requires_grad:
                        x.retain_grad()
                    self.attn_z_in[idx] = x
                return hook
            self.hooks.append(o_proj.register_forward_hook(make_attn_hook(L)))

            # Capture mlp output, retain gradient when tracking grads.
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


def compute_component_attributions(
    model,
    samples: List[Dict[str, Any]],
    best_L: int,
    v_best: np.ndarray,
    layers: List[int],
    device: str,
) -> pd.DataFrame:
    """Attribution patching for every (layer, head) and every (layer, MLP).

    Two-pass procedure:
      Pass 1 (no-grad): forward each sample, collect per-sample
        attn_z_in[L][anchor], mlp_z[L][anchor]. Compute per-class MEANS z_ref.
      Pass 2 (with grad): for each sample, forward with hooks+retain_grad,
        compute M = <resid[L*, anchor], v_best>, backprop.
        For each (L, component), attribution += <grad, z_ref - z_clean>.
    """
    n_heads = get_num_heads(model)
    hidden = get_hidden_size(model)
    d_head = hidden // n_heads
    v_t = torch.tensor(v_best, device=device)

    # ---- Pass 1: collect anchor-position clean activations per sample ----
    print(f"  Pass 1: collecting clean activations at anchor ({len(samples)} samples)", flush=True)
    hooks = AttributionHooks(model, layers, n_heads, hidden)

    # Per-sample storage at anchor position
    attn_z_anchor: Dict[int, List[torch.Tensor]] = {L: [] for L in layers}
    mlp_z_anchor: Dict[int, List[torch.Tensor]] = {L: [] for L in layers}
    labels = np.array(
        [1 if s["frequency_category"] == "high_freq" else 0 for s in samples])

    model.eval()
    for si, s in enumerate(samples):
        hooks.clear()
        ids = torch.tensor([s["token_ids"]], device=device)
        with torch.no_grad():
            model(input_ids=ids)
        pos = min(s["anchor"], ids.shape[1] - 1)
        for L in layers:
            attn_z_anchor[L].append(hooks.attn_z_in[L][0, pos].detach().float().cpu())
            mlp_z_anchor[L].append(hooks.mlp_z[L][0, pos].detach().float().cpu())
        if (si + 1) % 50 == 0:
            print(f"    {si + 1}/{len(samples)}", flush=True)
    hooks.remove()

    # Fixed class-delta: Δ_c = mean(component | high) - mean(component | low)
    # Attribution(c) = E_sample[grad_c(M) · Δ_c]
    # Positive ⇒ pushing c toward high-freq representation raises M (component
    # writes frequency); negative ⇒ component opposes.
    hi_idx = np.where(labels == 1)[0]
    lo_idx = np.where(labels == 0)[0]
    attn_delta: Dict[int, torch.Tensor] = {}
    mlp_delta: Dict[int, torch.Tensor] = {}
    for L in layers:
        A = torch.stack(attn_z_anchor[L])  # [N, H]
        M_ = torch.stack(mlp_z_anchor[L])  # [N, H]
        attn_delta[L] = A[hi_idx].mean(dim=0) - A[lo_idx].mean(dim=0)
        mlp_delta[L] = M_[hi_idx].mean(dim=0) - M_[lo_idx].mean(dim=0)

    # ---- Pass 2: gradient pass ----
    # For each sample compute grad of M(x) = <resid[L*, anchor], v> w.r.t. each
    # component activation, then accumulate attribution = grad · Δ.
    print(f"  Pass 2: gradient pass ({len(samples)} samples)", flush=True)

    tf = find_transformer_layers(model)
    attn_head_attr = {L: torch.zeros(n_heads, dtype=torch.float32) for L in layers}
    mlp_attr = {L: torch.tensor(0.0, dtype=torch.float32) for L in layers}
    n_samples_used = 0

    for si, s in enumerate(samples):
        # Register a grad-preserving residual hook at best_L and the component
        # hooks fresh each sample to ensure retained-grad tensors are local.
        hooks = AttributionHooks(model, layers, n_heads, hidden)
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
            pos = min(s["anchor"], ids.shape[1] - 1)

            out = model(input_ids=ids)
            h_res = pass_cap["h"]  # [1, T, H]
            resid_anchor = h_res[0, pos].float()
            M_scalar = torch.dot(resid_anchor, v_t.float())
            M_scalar.backward()

            for L in layers:
                gA = hooks.attn_z_in[L].grad if L in hooks.attn_z_in else None
                if gA is not None:
                    g_anchor = gA[0, pos].detach().float().cpu()
                    g_h = g_anchor.view(n_heads, d_head)
                    d_h = attn_delta[L].view(n_heads, d_head)
                    attn_head_attr[L] += (g_h * d_h).sum(dim=-1)

                gM = hooks.mlp_z[L].grad if L in hooks.mlp_z else None
                if gM is not None:
                    g_m_anchor = gM[0, pos].detach().float().cpu()
                    mlp_attr[L] += float((g_m_anchor * mlp_delta[L]).sum())
            n_samples_used += 1
            del out, h_res, resid_anchor, M_scalar
        finally:
            res_hook.remove()
            hooks.remove()

        if (si + 1) % 50 == 0:
            print(f"    {si + 1}/{len(samples)}", flush=True)

    # Normalize by sample count → per-sample mean attribution
    rows = []
    n = max(n_samples_used, 1)
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
#  Recovery attribution: after ablating v_L* at L*, how does the freq
#  component recover downstream?
# ─────────────────────────────────────────────────────────────────────

class ProjectOutHook:
    """Project out unit direction v from residual at given layer's output."""

    def __init__(self, model, layer: int, v: torch.Tensor, anchor_positions: Optional[List[int]] = None):
        self.v = v  # [H]
        self.anchor_positions = anchor_positions
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
        # Project out: x' = x - <x, v> v   (applied at every token position)
        coef = (x * v).sum(dim=-1, keepdim=True)  # [B, T, 1]
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
) -> pd.DataFrame:
    """Measure M_l = <resid[l, anchor], v_best> for clean vs single-layer-ablated
    forward. delta_M = M_l(ablated) - M_l(clean). Positive ⇒ direction was
    rebuilt downstream even though it was erased at L*.
    """
    v_t = torch.tensor(v_best, device=device, dtype=torch.float32)

    # Clean pass
    print(f"  Recovery: clean forward pass", flush=True)
    cap_clean = ResidualCapture(model, layers)
    clean_M: Dict[int, List[float]] = {L: [] for L in layers}
    try:
        for s in samples:
            cap_clean.clear()
            with torch.no_grad():
                model(input_ids=torch.tensor([s["token_ids"]], device=device))
            pos = min(s["anchor"], 10_000)
            for L in layers:
                h = cap_clean.outputs[L]
                p = min(pos, h.shape[1] - 1)
                x = h[0, p].float()
                clean_M[L].append(float(torch.dot(x, v_t.float().to(x.device))))
    finally:
        cap_clean.remove()

    # Ablated pass (project out v at L*)
    print(f"  Recovery: ablated forward pass (project out at L{best_L})", flush=True)
    cap_abl = ResidualCapture(model, layers)
    abl_hook = ProjectOutHook(model, best_L, v_t)
    abl_M: Dict[int, List[float]] = {L: [] for L in layers}
    try:
        for s in samples:
            cap_abl.clear()
            with torch.no_grad():
                model(input_ids=torch.tensor([s["token_ids"]], device=device))
            pos = s["anchor"]
            for L in layers:
                h = cap_abl.outputs[L]
                p = min(pos, h.shape[1] - 1)
                x = h[0, p].float()
                abl_M[L].append(float(torch.dot(x, v_t.float().to(x.device))))
    finally:
        cap_abl.remove()
        abl_hook.remove()

    rows = []
    labels = np.array(
        [1 if s["frequency_category"] == "high_freq" else 0 for s in samples])
    for L in layers:
        c = np.array(clean_M[L])
        a = np.array(abl_M[L])
        # Split by class so we can see whether downstream rewrites preserve sign
        d = a - c
        rows.append({
            "layer": int(L),
            "clean_M_mean": float(c.mean()),
            "ablated_M_mean": float(a.mean()),
            "delta_M_mean": float(d.mean()),
            "clean_M_diff_hi_lo": float(c[labels == 1].mean() - c[labels == 0].mean()),
            "ablated_M_diff_hi_lo": float(a[labels == 1].mean() - a[labels == 0].mean()),
            "delta_M_diff_hi_lo": float(
                (a[labels == 1] - c[labels == 1]).mean()
                - (a[labels == 0] - c[labels == 0]).mean()
            ),
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────

def _select_dtype(model_id: str, device: str) -> torch.dtype:
    if device != "cuda":
        return torch.float32
    m = model_id.lower()
    if any(k in m for k in ("7b", "6.9b", "8b", "13b", "9b", "11b", "12b", "14b", "32b", "70b")):
        return torch.bfloat16
    return torch.float32


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="auto",
                        choices=["auto", "float32", "bfloat16", "float16"])
    parser.add_argument("--max-samples", type=int, default=200,
                        help="Cap samples for budget; 200 gives stable attributions.")
    parser.add_argument("--skip-attribution", action="store_true",
                        help="Skip the gradient pass (recovery only)")
    parser.add_argument("--skip-recovery", action="store_true",
                        help="Skip recovery pass (attribution only)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.makedirs(args.output_dir, exist_ok=True)
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
    print(f"Loading {args.model} (revision={args.revision}, dtype={dtype})", flush=True)

    load_kw: Dict[str, Any] = {"torch_dtype": dtype}
    if args.revision != "main":
        load_kw["revision"] = args.revision
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if hf_token:
        load_kw["token"] = hf_token

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision,
                                              **({"token": hf_token} if hf_token else {}))
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kw).to(args.device)
    model.eval()

    # Load samples
    all_samples: List[Dict[str, Any]] = []
    for ds_name, fname in DATASET_FILES.items():
        p = Path(args.dataset_dir) / fname
        if not p.exists():
            continue
        pairs = load_synonym_pairs(str(p))
        s = pairs_to_samples(pairs, tokenizer)
        for x in s:
            x["dataset"] = ds_name
        all_samples.extend(s)
    print(f"Total samples: {len(all_samples)}", flush=True)

    # Cap for budget while preserving class balance (high/low pairs).
    # Samples come from pairs_to_samples as [hi0,lo0,hi1,lo1,...]; keep whole pairs.
    if len(all_samples) > args.max_samples:
        n_pairs_keep = max(1, args.max_samples // 2)
        n_total_pairs = len(all_samples) // 2
        step = max(1, n_total_pairs // n_pairs_keep)
        kept: List[Dict[str, Any]] = []
        for pi in range(0, n_total_pairs, step):
            kept.append(all_samples[2 * pi])      # hi
            kept.append(all_samples[2 * pi + 1])  # lo
            if len(kept) >= args.max_samples:
                break
        all_samples = kept
        n_hi = sum(1 for s in all_samples if s["frequency_category"] == "high_freq")
        print(f"Subsampled to {len(all_samples)} ({n_hi} hi / "
              f"{len(all_samples) - n_hi} lo) for budget", flush=True)

    try:
        n_hidden = int(model.config.num_hidden_layers)
        all_layers = list(range(n_hidden))
    except Exception:
        all_layers = _infer_layers(args.model)

    # ─── Find peak probe layer L* ───
    print(f"\n=== Finding peak probe layer across {len(all_layers)} layers ===",
          flush=True)
    best_L, best_auroc, layer_aurocs, v_best = find_peak_probe_layer(
        model, all_samples, all_layers, args.device)
    print(f"\n  Peak: L{best_L} (AUROC={best_auroc:.3f})", flush=True)

    probe_info = {
        "model": args.model,
        "revision": args.revision,
        "best_layer": int(best_L),
        "best_probe_auroc": float(best_auroc),
        "layer_aurocs": {str(k): float(v) for k, v in layer_aurocs.items()},
        "n_samples": int(len(all_samples)),
        "hidden_size": int(get_hidden_size(model)),
        "n_heads": int(get_num_heads(model)),
        "n_layers": int(len(all_layers)),
    }
    with open(Path(args.output_dir) / "probe.json", "w") as f:
        json.dump(probe_info, f, indent=2)

    model_short = args.model.split("/")[-1]

    # ─── Attribution patching ───
    if not args.skip_attribution:
        print(f"\n=== Attribution patching onto v_L{best_L} ===", flush=True)
        attr_df = compute_component_attributions(
            model, all_samples, best_L, v_best, all_layers, args.device)
        attr_df["model"] = model_short
        attr_df = attr_df[["model", "layer", "component_type",
                           "component_index", "attribution_score", "abs_score"]]
        attr_path = Path(args.output_dir) / "attribution_scores.csv"
        attr_df.to_csv(attr_path, index=False)
        print(f"  Wrote {attr_path} ({len(attr_df)} rows)", flush=True)

        # Top-10 components
        top = attr_df.sort_values("abs_score", ascending=False).head(10)
        print("\n  Top 10 components by |attribution|:")
        print(top.to_string(index=False))

    # ─── Recovery attribution ───
    if not args.skip_recovery:
        print(f"\n=== Recovery attribution (project out v at L{best_L}) ===",
              flush=True)
        rec_df = compute_recovery_attribution(
            model, all_samples, best_L, v_best, all_layers, args.device)
        rec_df["model"] = model_short
        rec_path = Path(args.output_dir) / "recovery_attribution.csv"
        rec_df.to_csv(rec_path, index=False)
        print(f"  Wrote {rec_path} ({len(rec_df)} rows)", flush=True)
        print("\n  Recovery per-layer ΔM_diff_hi_lo (ablated - clean):")
        for _, r in rec_df.iterrows():
            print(f"    L{int(r['layer']):3d}: clean={r['clean_M_diff_hi_lo']:+.4f}  "
                  f"ablated={r['ablated_M_diff_hi_lo']:+.4f}  "
                  f"Δ={r['delta_M_diff_hi_lo']:+.4f}")

    print(f"\nDone. Output: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
