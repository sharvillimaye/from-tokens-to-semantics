#!/usr/bin/env python3
"""
Subspace ablation: project the residual stream onto the subspace orthogonal
to the frequency direction at layer L, then measure downstream consequences.

This is the correct causal test for "frequency lives in a low-dimensional
subspace of the residual stream" — the superposition-consistent story.

Variants:
  A. Single-layer ablation: ablate at layer L, probe at all layers.
     Tells us whether frequency re-emerges downstream (via writers) or
     stays gone (no active re-writing).
  B. Cumulative ablation: ablate at every layer from L onward.
     Forces every subsequent component to re-introduce frequency or give up.
  C. Rank-k ablation: project out top-k SVD directions of residual
     difference between high/low freq means (generalizes beyond 1D).

Measurements per layer post-ablation:
  - Frequency probe AUROC (normalization-free, compared to clean baseline)
  - Cosine/Euclidean distance between mean high-freq and low-freq vectors
  - Output KL divergence between clean and ablated logits

Usage:
    python -m scripts.interventions.subspace_ablation \
        --model EleutherAI/pythia-6.9b-deduped --revision step143000 \
        --dataset-dir ~/scaleJSD/dataset/legacy/filtered \
        --output-dir /data/ani/mechinterp/runs/subspace_ablation/pythia-6.9b \
        --ablate-layers 5,10,15,20,25,30 \
        --ranks 1,2,4,8 \
        --device cuda
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

try:
    from scripts.metrics.coverage_affinity_experiment import (
        _infer_layers,
        load_synonym_pairs,
        pairs_to_samples,
    )
except ImportError:
    import sys
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
#  Transformer-layer discovery
# ─────────────────────────────────────────────────────────────────────

def find_transformer_layers(model):
    """Locate the list of transformer layer modules for hooking."""
    for attr in ["gpt_neox.layers", "model.layers", "transformer.h",
                 "transformer.layers", "model.decoder.layers"]:
        obj = model
        try:
            for p in attr.split("."):
                obj = getattr(obj, p)
            return obj
        except AttributeError:
            continue
    raise RuntimeError("Cannot find transformer layers in model")


# ─────────────────────────────────────────────────────────────────────
#  Step 1: clean forward pass — extract residual streams + logits
# ─────────────────────────────────────────────────────────────────────

class ResidualCapture:
    """Capture residual stream at each layer's output."""

    def __init__(self, model, layers: List[int]):
        self.model = model
        self.layers = layers
        self.hooks: list = []
        self.outputs: Dict[int, torch.Tensor] = {}

    def register(self):
        tf_layers = find_transformer_layers(self.model)
        for L in self.layers:
            def make_hook(layer_idx):
                def hook(_mod, _inp, output):
                    h = output[0] if isinstance(output, tuple) else output
                    self.outputs[layer_idx] = h.detach()
                return hook
            self.hooks.append(tf_layers[L].register_forward_hook(make_hook(L)))

    def clear(self):
        self.outputs.clear()

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


def clean_forward(
    model,
    samples: List[Dict],
    all_layers: List[int],
    device: str,
) -> Dict[str, Any]:
    """Run clean forward pass, extract residual streams and logits.

    Returns:
        residuals: {layer: np.ndarray [N, H]}
        logits: torch.Tensor [N, V] (at anchor positions)
    """
    cap = ResidualCapture(model, all_layers)
    cap.register()

    residuals: Dict[int, List[np.ndarray]] = {L: [] for L in all_layers}
    logits_list: List[torch.Tensor] = []

    for idx, sample in enumerate(samples):
        if idx % 100 == 0:
            print(f"    clean [{idx+1}/{len(samples)}]", flush=True)
        cap.clear()
        with torch.no_grad():
            inp = torch.tensor([sample["token_ids"]], device=device)
            out = model(input_ids=inp)
            # Extract anchor-position logits for KL
            anchor = min(sample["anchor"], out.logits.shape[1] - 1)
            logits_list.append(out.logits[0, anchor, :].detach().cpu())

        pos = sample["anchor"]
        for L in all_layers:
            h = cap.outputs[L]
            p = min(pos, h.shape[1] - 1)
            residuals[L].append(h[0, p, :].float().cpu().numpy())

    cap.remove()
    return {
        "residuals": {L: np.stack(vs) for L, vs in residuals.items()},
        "logits": torch.stack(logits_list),  # [N, V]
    }


# ─────────────────────────────────────────────────────────────────────
#  Step 2: compute frequency subspace at each layer
# ─────────────────────────────────────────────────────────────────────

def compute_frequency_directions(
    residuals: Dict[int, np.ndarray],
    freq_labels: np.ndarray,
    pair_ids: np.ndarray,
    max_rank: int = 8,
) -> Dict[int, Dict[str, Any]]:
    """At each layer compute two frequency-subspace representations:
      - probe_direction: logistic regression weight vector (rank 1)
      - svd_basis: top-k singular directions of (X_high - X_low) after
        mean-centering within group (for rank-k ablation)

    Also returns the clean probe AUROC for comparison.
    """
    freq_info = {}
    for L, X in residuals.items():
        # Probe direction (rank 1)
        gss = GroupShuffleSplit(n_splits=5, test_size=0.3, random_state=42)
        aurocs = []
        coefs = []
        for tr, te in gss.split(X, freq_labels, groups=pair_ids):
            clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
            clf.fit(X[tr], freq_labels[tr])
            prob = clf.predict_proba(X[te])[:, 1]
            aurocs.append(roc_auc_score(freq_labels[te], prob))
            coefs.append(clf.coef_[0])
        probe_dir = np.mean(coefs, axis=0)
        probe_dir = probe_dir / (np.linalg.norm(probe_dir) + 1e-12)

        # SVD basis for rank-k ablation
        # Subtract group mean, then stack; first singular vectors explain
        # between-group variance
        high_mean = X[freq_labels == 1].mean(axis=0)
        low_mean = X[freq_labels == 0].mean(axis=0)
        diff_matrix = np.stack([
            X[freq_labels == 1] - low_mean,   # residual of high given low
            X[freq_labels == 0] - high_mean,  # residual of low given high
        ]).reshape(-1, X.shape[1])
        # Fall back: use mean-diff direction as first axis
        U, S, Vt = np.linalg.svd(diff_matrix, full_matrices=False)
        svd_basis = Vt[:max_rank]  # [k, H]

        freq_info[L] = {
            "probe_direction": probe_dir,       # [H]
            "svd_basis": svd_basis,              # [max_rank, H]
            "clean_probe_auroc": float(np.mean(aurocs)),
            "clean_probe_auroc_std": float(np.std(aurocs)),
        }
    return freq_info


# ─────────────────────────────────────────────────────────────────────
#  Step 3: subspace ablation hook
# ─────────────────────────────────────────────────────────────────────

class SubspaceAblationHook:
    """Project residual stream onto subspace orthogonal to given directions
    at specified layers.

    For rank-k ablation: given basis V in [k, H] (rows are unit vectors),
    the projection is x - V.T @ (V @ x) so the k-dim subspace is removed.
    If V is not orthonormal, we orthonormalize first.
    """

    def __init__(self, directions: Dict[int, torch.Tensor], device: str):
        """directions[L] should be a [k, H] torch tensor on device."""
        # Orthonormalize each layer's basis
        self.proj_mats: Dict[int, torch.Tensor] = {}
        for L, V in directions.items():
            V = V.to(device)
            if V.dim() == 1:
                V = V.unsqueeze(0)
            # QR to orthonormalize: V = QR, keep first k columns of Q
            Q, _ = torch.linalg.qr(V.T, mode="reduced")  # [H, k]
            # Projection matrix onto complement: I - QQ^T, applied via x - Q @ (Q^T @ x)
            self.proj_mats[L] = Q  # keep orthonormal basis, apply on fly
        self.hooks: list = []

    def _make_hook(self, L):
        Q = self.proj_mats[L]  # [H, k]
        def hook(_mod, _inp, output):
            if isinstance(output, tuple):
                x = output[0]
                rest = output[1:]
            else:
                x = output
                rest = None
            # x: [batch, pos, H]
            # Project x onto complement of span(Q)
            # coeffs = x @ Q  [B, P, k]
            # x_ablated = x - coeffs @ Q.T
            coeffs = x @ Q
            x_ablated = x - coeffs @ Q.T
            if rest is None:
                return x_ablated
            return (x_ablated, *rest)
        return hook

    def register(self, model):
        tf_layers = find_transformer_layers(model)
        for L in self.proj_mats.keys():
            self.hooks.append(tf_layers[L].register_forward_hook(self._make_hook(L)))

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


# ─────────────────────────────────────────────────────────────────────
#  Step 4: probe trained on clean data, evaluated on ablated residuals
# ─────────────────────────────────────────────────────────────────────

def train_probes_on_clean(
    clean_residuals: Dict[int, np.ndarray],
    freq_labels: np.ndarray,
    pair_ids: np.ndarray,
) -> Dict[int, Any]:
    """Train a single frequency probe per layer on clean residuals.
    Returns the trained classifier + test split indices for evaluation."""
    probes = {}
    for L, X in clean_residuals.items():
        gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
        train_idx, test_idx = next(gss.split(X, freq_labels, groups=pair_ids))
        clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
        clf.fit(X[train_idx], freq_labels[train_idx])
        prob = clf.predict_proba(X[test_idx])[:, 1]
        auroc = roc_auc_score(freq_labels[test_idx], prob)
        probes[L] = {
            "clf": clf,
            "test_idx": test_idx,
            "clean_auroc": auroc,
        }
    return probes


def evaluate_probes(
    probes: Dict[int, Any],
    ablated_residuals: Dict[int, np.ndarray],
    freq_labels: np.ndarray,
) -> Dict[int, Dict[str, float]]:
    """Evaluate pre-trained probes on ablated residuals (same test split)."""
    results = {}
    for L, probe_info in probes.items():
        X = ablated_residuals[L]
        test_idx = probe_info["test_idx"]
        clf = probe_info["clf"]
        X_test = X[test_idx]
        y_test = freq_labels[test_idx]
        prob = clf.predict_proba(X_test)[:, 1]
        try:
            auroc = roc_auc_score(y_test, prob)
        except Exception:
            auroc = 0.5
        results[L] = {
            "ablated_auroc": float(auroc),
            "clean_auroc": float(probe_info["clean_auroc"]),
            "auroc_drop": float(probe_info["clean_auroc"] - auroc),
        }
    return results


# ─────────────────────────────────────────────────────────────────────
#  Step 5: forward pass with subspace ablation active
# ─────────────────────────────────────────────────────────────────────

def ablated_forward(
    model,
    samples: List[Dict],
    all_layers: List[int],
    ablation_directions: Dict[int, torch.Tensor],
    device: str,
) -> Dict[str, Any]:
    """Run forward pass with subspace ablation hooks active.

    Returns residual streams at all layers + anchor-position logits.
    """
    cap = ResidualCapture(model, all_layers)
    cap.register()

    ablator = SubspaceAblationHook(ablation_directions, device)
    ablator.register(model)

    residuals: Dict[int, List[np.ndarray]] = {L: [] for L in all_layers}
    logits_list: List[torch.Tensor] = []

    for idx, sample in enumerate(samples):
        cap.clear()
        with torch.no_grad():
            inp = torch.tensor([sample["token_ids"]], device=device)
            out = model(input_ids=inp)
            anchor = min(sample["anchor"], out.logits.shape[1] - 1)
            logits_list.append(out.logits[0, anchor, :].detach().cpu())

        pos = sample["anchor"]
        for L in all_layers:
            h = cap.outputs[L]
            p = min(pos, h.shape[1] - 1)
            residuals[L].append(h[0, p, :].float().cpu().numpy())

    cap.remove()
    ablator.remove()
    return {
        "residuals": {L: np.stack(vs) for L, vs in residuals.items()},
        "logits": torch.stack(logits_list),
    }


# ─────────────────────────────────────────────────────────────────────
#  Step 6: behavioral readout — KL and cos/eucl between groups
# ─────────────────────────────────────────────────────────────────────

def compute_output_kl(
    clean_logits: torch.Tensor,
    ablated_logits: torch.Tensor,
) -> float:
    """Mean KL(clean || ablated) across samples."""
    log_p = F.log_softmax(clean_logits, dim=-1)
    log_q = F.log_softmax(ablated_logits, dim=-1)
    p = log_p.exp()
    kl = (p * (log_p - log_q)).sum(dim=-1)
    return float(kl.mean())


def compute_distance_metrics(
    residuals: Dict[int, np.ndarray],
    freq_labels: np.ndarray,
) -> Dict[int, Dict[str, float]]:
    """Cosine and Euclidean distance between mean high-freq and low-freq
    residual vectors at each layer."""
    results = {}
    for L, X in residuals.items():
        mh = X[freq_labels == 1].mean(axis=0)
        ml = X[freq_labels == 0].mean(axis=0)
        diff = mh - ml
        norm_mh = np.linalg.norm(mh) + 1e-12
        norm_ml = np.linalg.norm(ml) + 1e-12
        cos = 1.0 - float(np.dot(mh, ml) / (norm_mh * norm_ml))
        euc = float(np.linalg.norm(diff))
        results[L] = {"cosine_dist": cos, "euclidean_dist": euc}
    return results


# ─────────────────────────────────────────────────────────────────────
#  Main experiment loop
# ─────────────────────────────────────────────────────────────────────

def run_one_ablation_condition(
    model,
    samples: List[Dict],
    all_layers: List[int],
    ablate_layers: List[int],
    directions: Dict[int, torch.Tensor],
    clean_probes: Dict[int, Any],
    clean_logits: torch.Tensor,
    freq_labels: np.ndarray,
    device: str,
) -> List[Dict[str, Any]]:
    """Run one ablation, return per-layer results."""
    out = ablated_forward(model, samples, all_layers, directions, device)
    probe_results = evaluate_probes(clean_probes, out["residuals"], freq_labels)
    distance_metrics = compute_distance_metrics(out["residuals"], freq_labels)
    output_kl = compute_output_kl(clean_logits, out["logits"])

    rows = []
    for L in all_layers:
        row = {
            "layer": L,
            "ablated_auroc": probe_results[L]["ablated_auroc"],
            "clean_auroc": probe_results[L]["clean_auroc"],
            "auroc_drop": probe_results[L]["auroc_drop"],
            "cosine_dist": distance_metrics[L]["cosine_dist"],
            "euclidean_dist": distance_metrics[L]["euclidean_dist"],
            "output_kl": output_kl,
        }
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--dataset-dir", type=str, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--ablate-layers", type=str, required=True,
                       help="Comma-separated layer indices to test single-layer ablation at")
    parser.add_argument("--ranks", type=str, default="1,2,4,8",
                       help="Comma-separated ranks for subspace ablation")
    parser.add_argument("--run-cumulative", action="store_true",
                       help="Also run cumulative ablation (ablate layer L and all after)")
    parser.add_argument("--run-random-baseline", action="store_true",
                       help="Run random-direction baseline (sanity check)")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    os.makedirs(args.output_dir, exist_ok=True)
    ablate_layers = [int(x) for x in args.ablate_layers.split(",")]
    ranks = [int(x) for x in args.ranks.split(",")]

    print(f"Loading model {args.model} (revision={args.revision})")
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, torch_dtype=torch.float32,
    ).to(args.device)
    model.eval()

    all_layers = _infer_layers(args.model)

    # Merge all datasets for robust probe training and category readout
    all_samples: List[Dict] = []
    category_labels: List[str] = []
    for ds_name, fname in DATASET_FILES.items():
        ds_path = Path(args.dataset_dir) / fname
        if not ds_path.exists():
            continue
        pairs = load_synonym_pairs(str(ds_path))
        samples = pairs_to_samples(pairs, tokenizer)
        for s in samples:
            s["dataset"] = ds_name
            all_samples.append(s)
            category_labels.append(ds_name)

    print(f"Total samples across all datasets: {len(all_samples)}")
    freq_labels = np.array(
        [1 if s["frequency_category"] == "high_freq" else 0 for s in all_samples])
    pair_ids = np.array([s["pair_id"] for s in all_samples])
    cat_array = np.array(category_labels)

    # === Clean forward pass ===
    print("\n=== Clean forward pass ===")
    clean = clean_forward(model, all_samples, all_layers, args.device)
    clean_residuals = clean["residuals"]
    clean_logits = clean["logits"]

    # === Compute frequency subspaces at each layer ===
    print("\n=== Computing frequency directions (probe + SVD) ===")
    max_rank = max(ranks)
    freq_info = compute_frequency_directions(
        clean_residuals, freq_labels, pair_ids, max_rank=max_rank)
    for L in ablate_layers:
        if L in freq_info:
            print(f"  L{L}: clean probe AUROC = {freq_info[L]['clean_probe_auroc']:.3f}")

    # === Train probes on clean data (to evaluate consistently across conditions) ===
    print("\n=== Training per-layer clean probes ===")
    clean_probes = train_probes_on_clean(clean_residuals, freq_labels, pair_ids)

    # === Run conditions ===
    all_results: List[Dict[str, Any]] = []

    # Clean baseline as a condition (no ablation)
    print("\n=== Clean baseline (no ablation) ===")
    # Just record clean AUROCs at all layers
    for L in all_layers:
        all_results.append({
            "condition": "clean",
            "ablate_layer": -1,
            "ablate_mode": "none",
            "rank": 0,
            "direction_type": "none",
            "layer": L,
            "ablated_auroc": clean_probes[L]["clean_auroc"],
            "clean_auroc": clean_probes[L]["clean_auroc"],
            "auroc_drop": 0.0,
            "cosine_dist": 0.0,
            "euclidean_dist": 0.0,
            "output_kl": 0.0,
        })

    # --- Variant A: single-layer ablation ---
    # For each layer L and each rank k, ablate only at L using probe or SVD direction
    for L in ablate_layers:
        if L not in freq_info:
            continue
        print(f"\n=== Single-layer ablation at L{L} ===")

        # Probe direction (rank 1)
        probe_dir = torch.tensor(
            freq_info[L]["probe_direction"][None, :], dtype=torch.float32)
        print(f"  → probe rank=1 at L{L}")
        rows = run_one_ablation_condition(
            model, all_samples, all_layers, [L],
            {L: probe_dir}, clean_probes, clean_logits, freq_labels, args.device)
        for r in rows:
            r.update({"condition": "single_probe",
                     "ablate_layer": L, "ablate_mode": "single",
                     "rank": 1, "direction_type": "probe"})
            all_results.append(r)

        # SVD directions (rank k)
        for k in ranks:
            svd_dir = torch.tensor(
                freq_info[L]["svd_basis"][:k], dtype=torch.float32)
            print(f"  → svd rank={k} at L{L}")
            rows = run_one_ablation_condition(
                model, all_samples, all_layers, [L],
                {L: svd_dir}, clean_probes, clean_logits, freq_labels, args.device)
            for r in rows:
                r.update({"condition": f"single_svd_k{k}",
                         "ablate_layer": L, "ablate_mode": "single",
                         "rank": k, "direction_type": "svd"})
                all_results.append(r)

        # Random-direction baseline (rank 1, same num directions as probe)
        if args.run_random_baseline:
            rng = np.random.RandomState(42)
            rand_dir = rng.randn(1, probe_dir.shape[1]).astype(np.float32)
            rand_dir = rand_dir / np.linalg.norm(rand_dir, axis=1, keepdims=True)
            rand_t = torch.tensor(rand_dir, dtype=torch.float32)
            print(f"  → random rank=1 at L{L}")
            rows = run_one_ablation_condition(
                model, all_samples, all_layers, [L],
                {L: rand_t}, clean_probes, clean_logits, freq_labels, args.device)
            for r in rows:
                r.update({"condition": "single_random",
                         "ablate_layer": L, "ablate_mode": "single",
                         "rank": 1, "direction_type": "random"})
                all_results.append(r)

        # Save incrementally
        pd.DataFrame(all_results).to_csv(
            Path(args.output_dir) / "subspace_ablation.csv", index=False)

    # --- Variant B: cumulative ablation ---
    if args.run_cumulative:
        for start in ablate_layers:
            layers_to_ablate = [L for L in all_layers if L >= start and L in freq_info]
            directions = {
                L: torch.tensor(
                    freq_info[L]["probe_direction"][None, :], dtype=torch.float32)
                for L in layers_to_ablate
            }
            print(f"\n=== Cumulative probe ablation from L{start} ({len(layers_to_ablate)} layers) ===")
            rows = run_one_ablation_condition(
                model, all_samples, all_layers, layers_to_ablate,
                directions, clean_probes, clean_logits, freq_labels, args.device)
            for r in rows:
                r.update({"condition": "cumulative_probe",
                         "ablate_layer": start, "ablate_mode": "cumulative",
                         "rank": 1, "direction_type": "probe"})
                all_results.append(r)
            pd.DataFrame(all_results).to_csv(
                Path(args.output_dir) / "subspace_ablation.csv", index=False)

    # Final save
    df = pd.DataFrame(all_results)
    df.to_csv(Path(args.output_dir) / "subspace_ablation.csv", index=False)

    # === Summary ===
    print(f"\n{'='*60}")
    print("  SUMMARY — AUROC drop at ablated layer (by condition)")
    print(f"{'='*60}")
    at_ablated = df[df.apply(lambda r: r["layer"] == r["ablate_layer"], axis=1)]
    if len(at_ablated) > 0:
        summary = at_ablated.groupby(["condition", "rank"])[
            ["clean_auroc", "ablated_auroc", "auroc_drop", "output_kl"]
        ].mean()
        print(summary.to_string())


if __name__ == "__main__":
    main()
