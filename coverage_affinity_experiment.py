#!/usr/bin/env python3
"""
Coverage & Frequency Affinity Experiment

Measures per-MLP-neuron coverage and frequency affinity using ScaleJSD synonym
pairs, then tests whether affinity independently predicts JSD contribution (and
optionally polysemanticity) after controlling for coverage.

Fixes over the original JSD polysemanticity pipeline (pipeline_notebook.ipynb):
  1. Binary coverage via activation > 0 threshold (not L1-normalized firing mass)
  2. Unnormalized activation mass (no inter-neuron competition from L1 norm)
  3. Group-size-balanced affinity using per-group mean (not raw sum)
  4. Actual log-frequency from dataset (not cumulative.parquet lookup)
  5. Pairwise JSD between matched synonym pairs (not bucket-averaged)
  6. Generic model support via PyTorch hooks (not Pythia-specific nnsight)

Usage:
    # Single checkpoint
    python coverage_affinity_experiment.py \\
        --dataset ~/dev/personal/mechinterp/scaleJSD/dataset/legacy/filtered/emotion_ngrams_dedup_filtered.jsonl \\
        --model EleutherAI/pythia-70m-deduped \\
        --revisions step143000 \\
        --output-dir results/coverage_affinity/pythia-70m/emotion/

    # Multiple checkpoints for training dynamics
    python coverage_affinity_experiment.py \\
        --dataset ... \\
        --revisions step1000,step43000,step143000 \\
        --output-dir results/coverage_affinity/pythia-70m/emotion/

    # All datasets at once
    python coverage_affinity_experiment.py \\
        --all-datasets --dataset-dir ~/dev/personal/mechinterp/scaleJSD/dataset/legacy/filtered \\
        --model EleutherAI/pythia-70m-deduped \\
        --revisions step143000 \\
        --output-dir results/coverage_affinity/pythia-70m/

    # Validate setup without full run (first 5 pairs, 1 layer)
    python coverage_affinity_experiment.py --dataset ... --output-dir /tmp/test --validate

Outputs:
    neuron_metrics.csv   — per (checkpoint, layer, neuron): coverage, mass, affinity, jsd_contrib
    pair_jsd.csv         — per (checkpoint, layer, pair_id): pairwise JSD
    group_jsd.csv        — per (checkpoint, layer): aggregate JSD
    analysis.json        — statistical tests per (checkpoint, layer)
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr, mannwhitneyu, norm

EPS = 1e-12


# ─────────────────────────────────────────────────────────────────────────────
# Activation capture
# ─────────────────────────────────────────────────────────────────────────────

class ActivationCapture:
    """Captures post-activation MLP intermediate (4H) and MLP input (H).

    Hooks the down-projection's forward call to grab its input, which is the
    post-activation-function intermediate — the canonical "neuron activation"
    vector for the MLP layer.

    Supports: GPT-NeoX (Pythia), LLaMA/Mistral, OLMo, GPT-2.
    """

    def __init__(self, model: torch.nn.Module, layers: List[int]):
        self.model = model
        self.layers = layers
        self.hooks: list = []
        self.post_act: Dict[int, torch.Tensor] = {}   # 4H-dim post-GELU/SiLU
        self.mlp_input: Dict[int, torch.Tensor] = {}   # H-dim residual stream

    # ── Architecture detection ──────────────────────────────────────────────

    def _find_model_layers(self):
        for attr in [
            "gpt_neox.layers", "model.layers", "transformer.h",
            "transformer.layers", "model.decoder.layers",
        ]:
            obj = self.model
            try:
                for part in attr.split("."):
                    obj = getattr(obj, part)
                return obj
            except AttributeError:
                continue
        raise RuntimeError("Cannot find transformer layers in model")

    @staticmethod
    def _find_mlp_and_down(layer_mod):
        """Return (mlp_module, down_projection_module) or (None, None)."""
        mlp_names = ["mlp", "feed_forward", "ff"]
        down_names = [
            "dense_4h_to_h", "down_proj", "c_proj", "fc2", "w2",
        ]
        for mn in mlp_names:
            mlp = getattr(layer_mod, mn, None)
            if mlp is None:
                continue
            for dn in down_names:
                down = getattr(mlp, dn, None)
                if down is not None:
                    return mlp, down
        return None, None

    # ── Hook registration ───────────────────────────────────────────────────

    def register(self):
        model_layers = self._find_model_layers()
        for L in self.layers:
            layer_mod = model_layers[L]
            mlp_mod, down_mod = self._find_mlp_and_down(layer_mod)
            if mlp_mod is None or down_mod is None:
                raise RuntimeError(
                    f"Cannot find MLP / down-projection in layer {L}. "
                    f"Submodules: {[n for n, _ in layer_mod.named_children()]}"
                )

            # MLP input = residual stream (H-dim)
            self.hooks.append(mlp_mod.register_forward_hook(
                lambda _mod, inp, _out, _L=L: self.mlp_input.__setitem__(
                    _L, (inp[0] if isinstance(inp, tuple) else inp).detach().cpu()
                )
            ))

            # Post-activation intermediate = down-projection input (4H-dim)
            self.hooks.append(down_mod.register_forward_hook(
                lambda _mod, inp, _out, _L=L: self.post_act.__setitem__(
                    _L, (inp[0] if isinstance(inp, tuple) else inp).detach().cpu()
                )
            ))

    def clear(self):
        self.post_act.clear()
        self.mlp_input.clear()

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
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
    """Convert synonym pairs into (high, low) sample dicts with token positions.

    Each pair produces TWO samples sharing the same pair_id.
    Uses actual frequency counts from the dataset for log-frequency.
    """
    DEFAULT_TEMPLATE = "The word {ngram} means"
    samples = []

    for i, pair in enumerate(pairs):
        pair_id = pair.get("synonym_pair_seed", pair.get("id", f"pair_{i}"))
        template = pair.get("sentence_template")

        for ngram, freq_cat, count_key in [
            (pair["high_freq_ngram"], "high_freq", "high_freq_count"),
            (pair["low_freq_ngram"], "low_freq", "low_freq_count"),
        ]:
            # Build text from template
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

            # Actual frequency from dataset (not looked up from cumulative.parquet)
            raw_count = pair.get(count_key, 1)
            log_freq = float(np.log(max(raw_count, 1)))

            samples.append({
                "pair_id": pair_id,
                "phrase": ngram,
                "sentence": text,
                "frequency_category": freq_cat,
                "token_ids": token_ids,
                "anchor": anchor,
                "log_frequency": log_freq,
                "raw_frequency": raw_count,
                "category": pair.get("category", "unknown"),
            })

    n_high = sum(1 for s in samples if s["frequency_category"] == "high_freq")
    n_low = len(samples) - n_high
    print(f"Created {len(samples)} samples ({n_high} high-freq, {n_low} low-freq)")
    return samples


# ─────────────────────────────────────────────────────────────────────────────
# Activation extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_activations(
    model: torch.nn.Module,
    samples: List[Dict[str, Any]],
    layers: List[int],
    device: str,
) -> Dict[int, np.ndarray]:
    """Extract post-activation MLP intermediate (4H) for all samples.

    Returns dict mapping layer -> np.ndarray [N_samples, 4H].
    """
    capture = ActivationCapture(model, layers)
    capture.register()

    results: Dict[int, list] = {L: [] for L in layers}

    for idx, sample in enumerate(samples):
        if idx % 50 == 0:
            print(f"  [{idx + 1}/{len(samples)}] {sample['phrase']}")

        capture.clear()
        with torch.no_grad():
            inputs = torch.tensor([sample["token_ids"]], device=device)
            model(input_ids=inputs)

        pos = sample["anchor"]
        for L in layers:
            act = capture.post_act.get(L)
            if act is None:
                raise RuntimeError(f"No activation captured for layer {L}")
            pos_clamped = min(pos, act.shape[1] - 1)
            results[L].append(act[0, pos_clamped, :].float().numpy())

    capture.remove()
    return {L: np.stack(vecs) for L, vecs in results.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Per-neuron metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_neuron_metrics(
    activations: np.ndarray,
    samples: List[Dict[str, Any]],
) -> pd.DataFrame:
    """Compute per-MLP-neuron coverage, raw mass, and frequency affinity.

    Operates on the post-activation intermediate (4H dim).

    Coverage (binary):
        Fraction of phrases where the neuron fires (activation > 0).
        No L1 normalization, no inter-neuron competition.

    Raw mass (unnormalized):
        Mean activation magnitude per group, normalized by group size.

    Frequency affinity (balanced ratio):
        raw_mass_high / (raw_mass_high + raw_mass_low).
        In [0, 1]; 0.5 = no preference; > 0.5 = prefers high-freq words.
        Group-size balanced because each group is independently averaged.

    Log-frequency affinity (continuous):
        Activation-weighted mean of log f(p), using unnormalized activations.
        Uses actual corpus frequencies from dataset, not L1-normalized probs.
    """
    N, D = activations.shape

    high_mask = np.array([s["frequency_category"] == "high_freq" for s in samples])
    low_mask = ~high_mask
    log_freqs = np.array([s["log_frequency"] for s in samples], dtype=np.float64)

    A_high = activations[high_mask]
    A_low = activations[low_mask]

    # ── Binary coverage: fraction of phrases activating each neuron ──
    # For post-GELU/SiLU activations, >0 is the natural firing threshold
    coverage_high = (A_high > 0).mean(axis=0).astype(np.float64)
    coverage_low = (A_low > 0).mean(axis=0).astype(np.float64)
    coverage_total = (activations > 0).mean(axis=0).astype(np.float64)

    # ── Raw mass: mean unnormalized ReLU activation per group ──
    A_high_relu = np.maximum(A_high, 0.0)
    A_low_relu = np.maximum(A_low, 0.0)
    raw_mass_high = A_high_relu.mean(axis=0).astype(np.float64)
    raw_mass_low = A_low_relu.mean(axis=0).astype(np.float64)
    raw_mass_total = np.maximum(activations, 0.0).mean(axis=0).astype(np.float64)

    # ── Frequency affinity (binary): balanced ratio of high vs low mass ──
    affinity = raw_mass_high / (raw_mass_high + raw_mass_low + EPS)

    # ── Log-frequency affinity (continuous): activation-weighted mean log f ──
    # For each neuron h: Σ_p ReLU(a_h(p)) * log f(p)  /  Σ_p ReLU(a_h(p))
    # Uses unnormalized activations — no inter-neuron competition
    A_relu = np.maximum(activations, 0.0)  # [N, D]
    weighted_logf = A_relu.T @ log_freqs                  # [D]
    mass_all = A_relu.sum(axis=0)                          # [D]
    logfreq_affinity = weighted_logf / np.maximum(mass_all, EPS)  # [D]

    return pd.DataFrame({
        "neuron": np.arange(D),
        "coverage_high": coverage_high,
        "coverage_low": coverage_low,
        "coverage_total": coverage_total,
        "raw_mass_high": raw_mass_high,
        "raw_mass_low": raw_mass_low,
        "raw_mass_total": raw_mass_total,
        "frequency_affinity": affinity,
        "logfreq_affinity": logfreq_affinity,
    })


# ─────────────────────────────────────────────────────────────────────────────
# JSD computation
# ─────────────────────────────────────────────────────────────────────────────

def _to_distribution(vec: np.ndarray) -> np.ndarray:
    """ReLU + L1 normalize to a probability distribution."""
    v = np.maximum(vec, 0.0).astype(np.float64)
    s = v.sum()
    return v / s if s > EPS else np.full_like(v, 1.0 / max(len(v), 1))


def _jsd_per_dim(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Per-dimension JSD between two probability distributions."""
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    P = P / (P.sum() + EPS)
    Q = Q / (Q.sum() + EPS)
    M = 0.5 * (P + Q)
    return 0.5 * (
        P * (np.log(P + EPS) - np.log(M + EPS))
        + Q * (np.log(Q + EPS) - np.log(M + EPS))
    )


def compute_group_jsd(
    activations: np.ndarray,
    samples: List[Dict[str, Any]],
) -> Tuple[float, np.ndarray]:
    """JSD between group-averaged high-freq and low-freq distributions.

    Returns (total_jsd, per_neuron_jsd_contributions).
    """
    high_mask = np.array([s["frequency_category"] == "high_freq" for s in samples])
    mean_high = np.maximum(activations[high_mask], 0.0).mean(axis=0)
    mean_low = np.maximum(activations[~high_mask], 0.0).mean(axis=0)

    P_high = _to_distribution(mean_high)
    P_low = _to_distribution(mean_low)
    contrib = _jsd_per_dim(P_high, P_low)
    return float(contrib.sum()), contrib


def compute_pairwise_jsd(
    activations: np.ndarray,
    samples: List[Dict[str, Any]],
) -> pd.DataFrame:
    """Pairwise JSD between matched synonym pairs."""
    by_pair: Dict[str, Dict[str, int]] = {}
    for i, s in enumerate(samples):
        by_pair.setdefault(s["pair_id"], {})[s["frequency_category"]] = i

    rows = []
    for pid, indices in sorted(by_pair.items()):
        if "high_freq" not in indices or "low_freq" not in indices:
            continue
        i_h, i_l = indices["high_freq"], indices["low_freq"]
        P_h = _to_distribution(activations[i_h])
        P_l = _to_distribution(activations[i_l])
        contrib = _jsd_per_dim(P_h, P_l)
        rows.append({
            "pair_id": pid,
            "phrase_high": samples[i_h]["phrase"],
            "phrase_low": samples[i_l]["phrase"],
            "jsd": float(contrib.sum()),
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Statistical analysis
# ─────────────────────────────────────────────────────────────────────────────

def _partial_spearman(x, y, z):
    """Partial Spearman correlation of x and y, controlling for z."""
    x, y, z = (np.asarray(v, float) for v in (x, y, z))
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[mask], y[mask], z[mask]
    if len(x) < 10:
        return np.nan, np.nan

    r_xy, _ = spearmanr(x, y)
    r_xz, _ = spearmanr(x, z)
    r_yz, _ = spearmanr(y, z)

    denom = np.sqrt(max((1 - r_xz ** 2) * (1 - r_yz ** 2), EPS))
    partial_r = (r_xy - r_xz * r_yz) / denom

    # Fisher z approximation for p-value
    n_eff = len(x) - 3
    if n_eff < 3 or abs(partial_r) >= 1.0:
        return float(partial_r), np.nan
    z_val = 0.5 * np.log((1 + partial_r) / (1 - partial_r + EPS))
    p_val = float(2 * norm.sf(abs(z_val) * np.sqrt(max(n_eff, 1))))
    return float(partial_r), p_val


def _cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Cliff's delta effect size (vectorized)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) == 0 or len(b) == 0:
        return np.nan
    diffs = a[:, None] - b[None, :]
    return float((np.sum(diffs > 0) - np.sum(diffs < 0)) / diffs.size)


def _coverage_matched_comparison(
    df: pd.DataFrame,
    outcome_col: str,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Within coverage deciles, compare outcome between high/low affinity neurons."""
    df = df.dropna(subset=["frequency_affinity", "coverage_total", outcome_col])
    if len(df) < 100:
        return pd.DataFrame()

    df = df.copy()
    df["cov_decile"] = pd.qcut(
        df["coverage_total"], q=n_bins, labels=False, duplicates="drop",
    )

    rows = []
    for decile, g in df.groupby("cov_decile"):
        if len(g) < 20:
            continue
        median_aff = g["frequency_affinity"].median()
        hi = g.loc[g["frequency_affinity"] >= median_aff, outcome_col].values
        lo = g.loc[g["frequency_affinity"] < median_aff, outcome_col].values
        if len(hi) < 5 or len(lo) < 5:
            continue

        _, p = mannwhitneyu(hi, lo, alternative="two-sided")
        rows.append({
            "coverage_decile": int(decile),
            "n_high_aff": len(hi),
            "n_low_aff": len(lo),
            "mean_outcome_high_aff": float(hi.mean()),
            "mean_outcome_low_aff": float(lo.mean()),
            "mann_whitney_p": float(p),
            "cliffs_delta": _cliffs_delta(hi, lo),
        })
    return pd.DataFrame(rows)


def run_analysis(
    neuron_df: pd.DataFrame,
    clusters_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Full statistical analysis: correlations, partial correlations, matched comparison."""
    results: Dict[str, Any] = {}

    n = len(neuron_df)
    results["n_neurons"] = n
    results["mean_coverage"] = float(neuron_df["coverage_total"].mean())
    results["median_coverage"] = float(neuron_df["coverage_total"].median())
    results["mean_affinity"] = float(neuron_df["frequency_affinity"].mean())

    # ── Affinity and coverage vs JSD contribution ────────────────────────
    if "jsd_contrib" not in neuron_df.columns:
        return results

    jsd = neuron_df["jsd_contrib"]
    cov = neuron_df["coverage_total"]
    aff = neuron_df["frequency_affinity"]

    # Raw Spearman
    for name, x in [("coverage", cov), ("affinity", aff), ("logfreq_affinity", neuron_df.get("logfreq_affinity"))]:
        if x is None:
            continue
        r, p = spearmanr(x, jsd, nan_policy="omit")
        results[f"spearman_{name}_jsd"] = {"r": float(r), "p": float(p)}

    # Partial Spearman: affinity → JSD | coverage
    pr, pp = _partial_spearman(aff, jsd, cov)
    results["partial_affinity_jsd_given_coverage"] = {"r": pr, "p": pp}

    # Partial Spearman: coverage → JSD | affinity
    pr, pp = _partial_spearman(cov, jsd, aff)
    results["partial_coverage_jsd_given_affinity"] = {"r": pr, "p": pp}

    # Coverage-matched comparison
    matched = _coverage_matched_comparison(neuron_df, "jsd_contrib")
    if len(matched) > 0:
        results["coverage_matched_jsd"] = matched.to_dict("records")
        results["coverage_matched_mean_delta"] = float(matched["cliffs_delta"].mean())
        sig = (matched["mann_whitney_p"] < 0.05).sum()
        results["coverage_matched_sig_deciles"] = f"{sig}/{len(matched)}"

    # ── Optional: merge with n_clusters ──────────────────────────────────
    if clusters_df is not None and "n_clusters" in clusters_df.columns:
        merged = neuron_df.merge(
            clusters_df[["neuron", "n_clusters"]], on="neuron", how="left",
        ).dropna(subset=["n_clusters"])

        if len(merged) > 20:
            nc = merged["n_clusters"]
            mc = merged["coverage_total"]
            ma = merged["frequency_affinity"]

            for name, x in [("coverage", mc), ("affinity", ma)]:
                r, p = spearmanr(x, nc, nan_policy="omit")
                results[f"spearman_{name}_nclusters"] = {"r": float(r), "p": float(p)}

            pr, pp = _partial_spearman(ma, nc, mc)
            results["partial_affinity_nclusters_given_coverage"] = {"r": pr, "p": pp}

            matched_nc = _coverage_matched_comparison(
                merged.rename(columns={"n_clusters": "jsd_contrib"}), "jsd_contrib",
            )
            if len(matched_nc) > 0:
                results["coverage_matched_nclusters"] = matched_nc.to_dict("records")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────

MODEL_LAYERS = {
    "pythia-70m": 6, "pythia-160m": 12, "pythia-410m": 24,
    "pythia-1b": 16, "pythia-1.4b": 24, "pythia-2.8b": 32,
    "pythia-6.9b": 32, "pythia-12b": 36,
    "olmo-1b": 16, "olmo-7b": 32,
}


def _infer_layers(model_id: str) -> List[int]:
    for key, n in MODEL_LAYERS.items():
        if key in model_id.lower():
            return list(range(n))
    return list(range(12))


def run_single_checkpoint(
    *,
    samples: List[Dict[str, Any]],
    model_id: str,
    revision: str,
    layers: List[int],
    device: str,
    output_dir: Path,
    clusters_df: Optional[pd.DataFrame],
    save_activations: bool,
    load_activations: Optional[Path],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[Dict]]:
    """Run the experiment for a single model checkpoint."""
    from transformers import AutoModelForCausalLM

    act_cache_path = output_dir / f"activations_{revision}.pkl"

    # ── Load or extract activations ──────────────────────────────────────
    if load_activations and load_activations.exists():
        print(f"Loading cached activations from {load_activations}")
        with open(load_activations, "rb") as f:
            layer_activations = pickle.load(f)
    elif act_cache_path.exists():
        print(f"Loading cached activations from {act_cache_path}")
        with open(act_cache_path, "rb") as f:
            layer_activations = pickle.load(f)
    else:
        print(f"\nLoading {model_id} @ {revision} on {device}")
        load_kw: Dict[str, Any] = {"torch_dtype": torch.float16}
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if hf_token:
            load_kw["token"] = hf_token
        if revision != "main":
            load_kw["revision"] = revision

        model = AutoModelForCausalLM.from_pretrained(model_id, **load_kw).to(device)
        model.eval()

        print("Extracting activations ...")
        layer_activations = extract_activations(model, samples, layers, device)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if save_activations and not act_cache_path.exists():
        with open(act_cache_path, "wb") as f:
            pickle.dump(layer_activations, f)
        print(f"Saved activations to {act_cache_path}")

    # ── Per-layer analysis ───────────────────────────────────────────────
    all_neuron_dfs = []
    all_pair_jsds = []
    group_jsd_rows = []
    all_analyses = []

    for L in layers:
        acts = layer_activations[L]  # [N, 4H]
        print(f"\n  Layer {L}  (4H = {acts.shape[1]})")

        # Per-neuron metrics
        neuron_df = compute_neuron_metrics(acts, samples)
        neuron_df["layer"] = L
        neuron_df["checkpoint"] = revision

        # Group JSD
        total_jsd, jsd_contrib = compute_group_jsd(acts, samples)
        neuron_df["jsd_contrib"] = jsd_contrib
        print(f"    Group JSD: {total_jsd:.6f}")

        # Pairwise JSD
        pair_jsd = compute_pairwise_jsd(acts, samples)
        pair_jsd["layer"] = L
        pair_jsd["checkpoint"] = revision
        print(f"    Mean pair JSD: {pair_jsd['jsd'].mean():.6f}  (n={len(pair_jsd)})")

        group_jsd_rows.append({
            "checkpoint": revision, "layer": L,
            "group_jsd": total_jsd,
            "mean_pair_jsd": float(pair_jsd["jsd"].mean()),
            "std_pair_jsd": float(pair_jsd["jsd"].std()),
            "n_pairs": len(pair_jsd),
        })

        # Statistics
        layer_clusters = None
        if clusters_df is not None and "layer" in clusters_df.columns:
            layer_clusters = clusters_df[clusters_df["layer"] == L]

        analysis = run_analysis(neuron_df, layer_clusters)
        analysis["layer"] = L
        analysis["checkpoint"] = revision
        analysis["group_jsd"] = total_jsd

        # Print key results
        if "spearman_coverage_jsd" in analysis:
            rc = analysis["spearman_coverage_jsd"]["r"]
            ra = analysis["spearman_affinity_jsd"]["r"]
            print(f"    Spearman(coverage, jsd):  r={rc:+.3f}")
            print(f"    Spearman(affinity, jsd):  r={ra:+.3f}")
        if "partial_affinity_jsd_given_coverage" in analysis:
            pr = analysis["partial_affinity_jsd_given_coverage"]["r"]
            pp = analysis["partial_affinity_jsd_given_coverage"]["p"]
            sig = "*" if pp is not None and pp < 0.05 else ""
            print(f"    Partial(aff, jsd | cov):  r={pr:+.3f} {sig}")
        if "coverage_matched_mean_delta" in analysis:
            md = analysis["coverage_matched_mean_delta"]
            sd = analysis["coverage_matched_sig_deciles"]
            print(f"    Coverage-matched delta:   δ={md:+.3f}  sig={sd}")

        all_analyses.append(analysis)
        all_neuron_dfs.append(neuron_df)
        all_pair_jsds.append(pair_jsd)

    df_neurons = pd.concat(all_neuron_dfs, ignore_index=True)
    df_pairs = pd.concat(all_pair_jsds, ignore_index=True)
    df_group = pd.DataFrame(group_jsd_rows)

    return df_neurons, df_pairs, df_group, all_analyses


def run_experiment(
    dataset_path: str,
    model_id: str,
    revisions: List[str],
    output_dir: str,
    layers: Optional[List[int]] = None,
    device: Optional[str] = None,
    clusters_path: Optional[str] = None,
    save_activations: bool = False,
    load_activations: Optional[str] = None,
    validate_only: bool = False,
):
    """Run the full experiment across one or more checkpoints."""
    from transformers import AutoTokenizer

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )

    # ── Load data ────────────────────────────────────────────────────────
    pairs = load_synonym_pairs(dataset_path)

    # Load tokenizer (use first revision for tokenizer — they share vocab)
    tok_kw: Dict[str, Any] = {}
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if hf_token:
        tok_kw["token"] = hf_token
    rev0 = revisions[0]
    if rev0 != "main":
        tok_kw["revision"] = rev0
    tokenizer = AutoTokenizer.from_pretrained(model_id, **tok_kw)

    samples = pairs_to_samples(pairs, tokenizer)

    if layers is None:
        layers = _infer_layers(model_id)

    # ── Validate mode: quick smoke test ──────────────────────────────────
    if validate_only:
        print("\n=== VALIDATION MODE ===")
        test_samples = samples[:min(10, len(samples))]
        test_layers = layers[:1]
        print(f"  Testing with {len(test_samples)} samples, layer {test_layers}")

        from transformers import AutoModelForCausalLM
        load_kw: Dict[str, Any] = {"torch_dtype": torch.float16}
        if hf_token:
            load_kw["token"] = hf_token
        if rev0 != "main":
            load_kw["revision"] = rev0
        model = AutoModelForCausalLM.from_pretrained(model_id, **load_kw).to(device)
        model.eval()

        acts = extract_activations(model, test_samples, test_layers, device)
        del model

        L = test_layers[0]
        print(f"  Activation shape: {acts[L].shape}")
        print(f"  Non-zero fraction: {(acts[L] > 0).mean():.3f}")
        print(f"  Mean activation: {acts[L][acts[L] > 0].mean():.4f}")

        ndf = compute_neuron_metrics(acts[L], test_samples)
        print(f"  Coverage range: [{ndf['coverage_total'].min():.3f}, {ndf['coverage_total'].max():.3f}]")
        print(f"  Affinity range: [{ndf['frequency_affinity'].min():.3f}, {ndf['frequency_affinity'].max():.3f}]")

        total_jsd, _ = compute_group_jsd(acts[L], test_samples)
        print(f"  Group JSD: {total_jsd:.6f}")

        pair_jsd = compute_pairwise_jsd(acts[L], test_samples)
        print(f"  Pair JSD: mean={pair_jsd['jsd'].mean():.6f} (n={len(pair_jsd)})")

        print("\n  VALIDATION PASSED")
        return

    # ── Load clusters ────────────────────────────────────────────────────
    clusters_df = None
    if clusters_path:
        clusters_df = pd.read_csv(clusters_path)
        print(f"Loaded clusters: {len(clusters_df)} rows")

    # ── Run per checkpoint ───────────────────────────────────────────────
    all_neurons, all_pairs, all_groups, all_analyses = [], [], [], []
    load_act_path = Path(load_activations) if load_activations else None

    for revision in revisions:
        print(f"\n{'=' * 60}")
        print(f"  Checkpoint: {revision}")
        print(f"{'=' * 60}")

        dn, dp, dg, analyses = run_single_checkpoint(
            samples=samples,
            model_id=model_id,
            revision=revision,
            layers=layers,
            device=device,
            output_dir=out,
            clusters_df=clusters_df,
            save_activations=save_activations,
            load_activations=load_act_path,
        )
        all_neurons.append(dn)
        all_pairs.append(dp)
        all_groups.append(dg)
        all_analyses.extend(analyses)

    # ── Save ─────────────────────────────────────────────────────────────
    df_neurons = pd.concat(all_neurons, ignore_index=True)
    df_pairs = pd.concat(all_pairs, ignore_index=True)
    df_groups = pd.concat(all_groups, ignore_index=True)

    df_neurons.to_csv(out / "neuron_metrics.csv", index=False)
    df_pairs.to_csv(out / "pair_jsd.csv", index=False)
    df_groups.to_csv(out / "group_jsd.csv", index=False)

    with open(out / "analysis.json", "w") as f:
        json.dump(all_analyses, f, indent=2, default=str)

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  RESULTS SAVED TO {out}")
    print(f"{'=' * 60}")
    print(f"  neuron_metrics.csv : {len(df_neurons):,} rows")
    print(f"  pair_jsd.csv       : {len(df_pairs):,} rows")
    print(f"  group_jsd.csv      : {len(df_groups):,} rows")
    print(f"  analysis.json      : {len(all_analyses)} layer-analyses")

    # Print summary table
    if len(df_groups) > 0:
        print(f"\n  Group JSD summary:")
        for _, row in df_groups.iterrows():
            print(f"    {row['checkpoint']} L{int(row['layer']):2d}  "
                  f"JSD={row['group_jsd']:.6f}  "
                  f"pair_mean={row['mean_pair_jsd']:.6f}")


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
        description="Coverage & Frequency Affinity Experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Input
    p.add_argument("--dataset", type=str, default=None,
                    help="Path to synonym pairs JSONL file")
    p.add_argument("--all-datasets", action="store_true",
                    help="Run all datasets (emotion, medical, legal, scientific, verb)")
    p.add_argument("--dataset-dir", type=str, default=None,
                    help="Directory containing dataset JSONL files (used with --all-datasets)")

    # Model
    p.add_argument("--model", default="EleutherAI/pythia-70m-deduped")
    p.add_argument("--revisions", default="step143000",
                    help="Comma-separated checkpoint revisions")
    p.add_argument("--layers", type=str, default=None,
                    help="Comma-separated layer indices (default: all)")
    p.add_argument("--device", default=None)

    # Output
    p.add_argument("--output-dir", required=True)

    # Optional
    p.add_argument("--clusters", default=None,
                    help="Pre-computed n_clusters CSV (from neuron embeddings pipeline)")
    p.add_argument("--save-activations", action="store_true",
                    help="Cache extracted activations to disk")
    p.add_argument("--load-activations", default=None,
                    help="Load activations from this path instead of extracting")
    p.add_argument("--validate", action="store_true",
                    help="Quick validation run (few samples, 1 layer)")

    args = p.parse_args()

    revisions = [r.strip() for r in args.revisions.split(",")]
    layers = [int(x) for x in args.layers.split(",")] if args.layers else None

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

    # Run each dataset
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
            layers=layers,
            device=args.device,
            clusters_path=args.clusters,
            save_activations=args.save_activations,
            load_activations=args.load_activations,
            validate_only=args.validate,
        )


if __name__ == "__main__":
    main()
