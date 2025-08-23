# -*- coding: utf-8 -*-
"""
JSD Polysemanticity Analysis Pipeline

This pipeline analyzes the relationship between Jensen-Shannon Divergence (JSD) 
and polysemanticity in language models, specifically examining how frequency 
affinity relates to neuron clustering behavior across training steps.

The pipeline:
1. Loads model checkpoints and computes activations for n-gram phrases
2. Calculates JSD between frequency buckets and per-neuron contributions
3. Analyzes correlations between frequency affinity and polysemanticity
4. Generates visualizations and statistical analyses
5. Performs layerwise analyses to understand spatial patterns

Requirements:
- df_candidates: DataFrame with columns [phrase, n_tokens, bucket]
- cumulative.parquet: Frequency data with columns [phrase, step, per_million_cum/count_cum]
- Model embedding parquet files for polysemanticity analysis
"""

from __future__ import annotations
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path
from nnsight import LanguageModel
from scipy.stats import spearmanr, mannwhitneyu, kruskal, zscore

# =============================================================================
# CONFIGURATION
# =============================================================================

# Model configurations
MODELS = [
    {
        "tag": "pythia-70m",
        "title": "Pythia-70M",
        "model_id": "EleutherAI/pythia-70m-deduped",
        "steps": [1000, 13000, 23000, 33000, 43000, 53000,
                  63000, 73000, 83000, 93000, 103000, 113000,
                  123000, 133000, 143000],
        "layers": list(range(0, 6)),
        "clusters_parquet": "./pythia70m_embeddings.parquet",
    },
    {
        "tag": "pythia-160m",
        "title": "Pythia-160M",
        "model_id": "EleutherAI/pythia-160m-deduped",
        "steps": [1000, 13000, 23000, 33000, 43000, 53000,
                  63000, 73000, 83000, 93000, 103000, 113000,
                  123000, 133000, 143000],
        "layers": list(range(0, 12)),
        "clusters_parquet": "./pythia160m_embeddings.parquet",
    },
]

# Analysis parameters
BUCKETS_TO_COMPARE = [0, 7]  # Compare lowest vs highest frequency buckets
BATCH_SIZE = 32
DTYPE = torch.float16
DEVICE_MAP = "auto"

# Text template for prompting
TEMPLATE_PREFIX = "This is a place in the world:"
TEMPLATE_SUFFIX = "."

# File paths
COUNTS_PARQUET = "./cumulative.parquet"  # Frequency data
OUTROOT = Path("./runs_jsd_poly")        # Output directory
OUTROOT.mkdir(parents=True, exist_ok=True)

# Numerical stability
EPS = 1e-12

# =============================================================================
# PLOTTING SETUP
# =============================================================================

def setup_plotting():
    """Configure matplotlib for consistent, publication-ready plots."""
    mpl.rcParams.update({
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    from cycler import cycler
    mpl.rcParams["axes.prop_cycle"] = cycler(color=mpl.cm.get_cmap("tab10").colors)

# =============================================================================
# DATA LOADING AND VALIDATION
# =============================================================================

def validate_inputs():
    """Validate that required data is available and properly formatted."""
    try:
        df_candidates  # noqa: F821
    except NameError:
        raise RuntimeError("df_candidates not found. Run your sampling cell to build it first.")
    
    present_buckets = sorted(df_candidates["bucket"].dropna().unique().astype(int).tolist())
    use_buckets = [b for b in BUCKETS_TO_COMPARE if b in present_buckets]
    
    if len(use_buckets) < 2:
        print(f"[warn] Not all requested buckets present. Found={present_buckets}, requested={BUCKETS_TO_COMPARE}")
    
    return use_buckets

def load_frequency_data(use_buckets: List[int]) -> Tuple[pd.DataFrame, str, Dict[int, List[str]]]:
    """Load and validate frequency data, create bucket-to-phrases mapping."""
    # Load frequency data
    df_counts_all = pd.read_parquet(COUNTS_PARQUET)
    
    # Validate columns
    if not {"phrase", "step"}.issubset(df_counts_all.columns):
        raise ValueError(f"{COUNTS_PARQUET} must have at least ['phrase','step', <count column>].")
    
    # Determine count column
    count_col = ("per_million_cum" if "per_million_cum" in df_counts_all.columns 
                 else "count_cum" if "count_cum" in df_counts_all.columns else None)
    if count_col is None:
        raise ValueError(f"{COUNTS_PARQUET} needs 'per_million_cum' or 'count_cum' column.")
    
    # Create bucket-to-phrases mapping
    bucket2phrases = {
        b: sorted(df_candidates.loc[df_candidates["bucket"] == b, "phrase"].unique().tolist())
        for b in use_buckets
    }
    
    # Filter to only phrases we'll probe
    df_counts_all = df_counts_all[df_counts_all["phrase"].isin(
        np.unique([p for b in use_buckets for p in bucket2phrases[b]])
    )].copy()
    
    return df_counts_all, count_col, bucket2phrases

# =============================================================================
# CORE COMPUTATION FUNCTIONS
# =============================================================================

def build_text(phrase: str) -> str:
    """Build the full text prompt for a given phrase."""
    return f"{TEMPLATE_PREFIX}{phrase}{TEMPLATE_SUFFIX}"

def get_anchor_indices(lm: LanguageModel, phrases: List[str]) -> np.ndarray:
    """Get token indices for the target phrases in the context."""
    tok = lm.tokenizer
    suffix_ids = tok.encode(TEMPLATE_SUFFIX, add_special_tokens=False)
    suffix_len = len(suffix_ids)
    
    anchors = []
    for phrase in phrases:
        full_text = build_text(phrase)
        full_ids = tok.encode(full_text, add_special_tokens=False)
        
        if suffix_len >= len(full_ids):
            raise ValueError("Suffix longer than full text; template issue.")
        
        # Anchor is the position of the last token of the phrase
        anchors.append(len(full_ids) - suffix_len - 1)
    
    return np.asarray(anchors, dtype=int)

def capture_activations(
    lm: LanguageModel,
    texts: List[str],
    anchors: np.ndarray,
    layers: List[int],
) -> Dict[int, np.ndarray]:
    """Capture post-MLP activations at specific token positions."""
    with lm.trace(texts) as tr:
        # Save activations from specified layers
        activations = {
            L: lm.gpt_neox.layers[L].mlp.dense_4h_to_h.input.save() 
            for L in layers
        }
        _ = lm.output.save()
    
    # Extract activations at anchor positions
    out = {}
    for L in layers:
        v = activations[L].value  # [B, S, H]
        v = v.to("cpu").float().detach().numpy()
        out[L] = v[np.arange(v.shape[0]), anchors, :]  # [B, H]
    
    return out

def relu_probs(activations: np.ndarray, eps: float = EPS) -> np.ndarray:
    """Convert activations to probability distribution using ReLU and normalization."""
    # Apply ReLU
    X = np.maximum(activations, 0.0)
    
    # Normalize to probabilities
    denom = X.sum(axis=1, keepdims=True)
    P = np.divide(X, np.maximum(denom, eps), out=np.zeros_like(X), where=(denom > 0))
    
    return P

def jsd_per_dim(P: np.ndarray, Q: np.ndarray, eps: float = EPS) -> np.ndarray:
    """Compute Jensen-Shannon divergence per dimension between distributions P and Q."""
    # Ensure inputs are arrays and normalize
    P = np.asarray(P, dtype=np.float64)
    P = P / (P.sum() + eps)
    
    Q = np.asarray(Q, dtype=np.float64)
    Q = Q / (Q.sum() + eps)
    
    # Compute midpoint distribution
    M = 0.5 * (P + Q)
    
    # Compute JSD: 0.5 * (KL(P||M) + KL(Q||M))
    jsd = 0.5 * (
        P * (np.log(P + eps) - np.log(M + eps)) +
        Q * (np.log(Q + eps) - np.log(M + eps))
    )
    
    return jsd

# =============================================================================
# STATISTICAL UTILITIES
# =============================================================================

_rng = np.random.default_rng(0)

def bootstrap_spearman_ci(x: np.ndarray, y: np.ndarray, n_boot: int = 1000, alpha: float = 0.05):
    """Bootstrap confidence interval for Spearman correlation."""
    n = x.shape[0]
    if n < 3:
        return np.nan, (np.nan, np.nan)
    
    boots = []
    for _ in range(n_boot):
        idx = _rng.integers(0, n, size=n)
        r, _ = spearmanr(x[idx], y[idx])
        boots.append(r if np.isfinite(r) else np.nan)
    
    boots = np.asarray(boots, dtype=float)
    lo = np.nanpercentile(boots, 100 * alpha / 2)
    hi = np.nanpercentile(boots, 100 * (1 - alpha / 2))
    
    return np.nanmean(boots), (float(lo), float(hi))

def fdr_bh(pvals: np.ndarray, alpha: float = 0.05):
    """Benjamini-Hochberg false discovery rate correction."""
    p = np.asarray(pvals, dtype=float)
    n = max(1, p.size)
    
    order = np.argsort(p)
    ranked = p[order]
    thresh = alpha * (np.arange(1, n + 1) / n)
    passed = ranked <= np.maximum.accumulate(thresh)
    
    mask = np.zeros_like(passed, dtype=bool)
    mask[order] = passed
    
    # Adjusted p-values
    adj = np.empty_like(p)
    adj[order] = np.minimum.accumulate((n / np.arange(n, 0, -1)) * ranked[::-1])[::-1]
    
    return mask, np.clip(adj, 0, 1)

def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Cliff's delta effect size measure."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    na, nb = len(a), len(b)
    
    if na == 0 or nb == 0:
        return np.nan
    
    A = np.sort(a)
    B = np.sort(b)
    i = j = more = less = 0
    
    while i < na and j < nb:
        if A[i] > B[j]:
            more += (na - i)
            j += 1
        elif A[i] < B[j]:
            less += (nb - j)
            i += 1
        else:
            i += 1
            j += 1
    
    return float((more - less) / (na * nb))

def cohens_d(a: np.ndarray, b: np.ndarray, hedges_correction: bool = True) -> tuple:
    """Cohen's d effect size with optional Hedges' correction."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    
    if a.size < 2 or b.size < 2:
        return np.nan, np.nan
    
    ma, mb = a.mean(), b.mean()
    sa2, sb2 = a.var(ddof=1), b.var(ddof=1)
    n1, n2 = a.size, b.size
    
    # Pooled standard deviation
    sp = np.sqrt(((n1 - 1) * sa2 + (n2 - 1) * sb2) / max(1, (n1 + n2 - 2)))
    
    d = (ma - mb) / sp if np.isfinite(sp) and sp > 0 else np.nan
    
    if hedges_correction and np.isfinite(d):
        J = 1.0 - 3.0 / (4.0 * (n1 + n2) - 9.0)
        return d, J * d
    
    return d, d

def bootstrap_ci(func, a: np.ndarray, b: np.ndarray, n_boot: int = 1000):
    """Bootstrap confidence interval for a function of two arrays."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    
    if a.size == 0 or b.size == 0:
        return np.nan, (np.nan, np.nan)
    
    na, nb = a.size, b.size
    vals = []
    
    for _ in range(n_boot):
        ai = _rng.integers(0, na, size=na)
        bi = _rng.integers(0, nb, size=nb)
        vals.append(func(a[ai], b[bi]))
    
    vals = np.asarray(vals, float)
    lo = np.nanpercentile(vals, 2.5)
    hi = np.nanpercentile(vals, 97.5)
    
    return float(np.nanmean(vals)), (float(lo), float(hi))

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def savefig_dual(fig, path_png: Path, path_pdf: Optional[Path] = None):
    """Save figure in both PNG and PDF formats."""
    fig.savefig(path_png, dpi=300, bbox_inches="tight")
    if path_pdf is None:
        path_pdf = path_png.with_suffix(".pdf")
    fig.savefig(path_pdf, dpi=300, bbox_inches="tight")
    print(f"[saved] {path_png}\n[saved] {path_pdf}")

def load_clusters_from_export_parquet(path: str) -> pd.DataFrame:
    """Load polysemanticity cluster data from exported parquet file."""
    dfw = pd.read_parquet(path)
    
    # Validate required columns
    if not {"layer", "neuron"}.issubset(dfw.columns):
        raise ValueError("Clusters parquet must contain 'layer' and 'neuron'.")
    
    # Find checkpoint columns
    idxs = []
    for i in range(1000):
        if (f"checkpoint_{i}_step" in dfw.columns and 
            f"checkpoint_{i}_num_clusters" in dfw.columns):
            idxs.append(i)
    
    if not idxs:
        raise ValueError("No checkpoint_{i}_step / checkpoint_{i}_num_clusters columns found.")
    
    # Build long-format DataFrame
    recs = []
    base = dfw[["layer", "neuron"]].astype(int)
    
    for i in idxs:
        steps = pd.to_numeric(dfw[f"checkpoint_{i}_step"], errors="coerce")
        ncl = pd.to_numeric(dfw[f"checkpoint_{i}_num_clusters"], errors="coerce")
        
        part = pd.DataFrame({
            "layer": base["layer"],
            "neuron": base["neuron"],
            "step": steps.astype("Int64"),
            "n_clusters": ncl
        }).dropna(subset=["step"])
        part["step"] = part["step"].astype(int)
        recs.append(part)
    
    # Combine and sort
    out = (pd.concat(recs, ignore_index=True)
             .sort_values(["layer", "neuron", "step"])
             .reset_index(drop=True))
    
    return out

# =============================================================================
# MAIN ANALYSIS PIPELINE
# =============================================================================

def sweep_bins_jsd_with_contrib_and_affinity(
    model_id: str,
    model_title: str,
    steps: List[int],
    layers: List[int],
    bucket_phrases: Dict[int, List[str]],
    df_counts_all: pd.DataFrame,
    count_col: str,
    batch_size: int = 32,
) -> Tuple[
    pd.DataFrame, pd.DataFrame,
    Dict[Tuple[int, int, int], np.ndarray],
    Dict[Tuple[int, int], Dict[str, np.ndarray]]
]:
    """
    Main analysis function that computes JSD, contributions, and affinity statistics.
    
    Returns:
        df_jsd: DataFrame with JSD between buckets per step/layer
        df_contrib: DataFrame with per-neuron JSD contributions
        avgP: Average probability distributions per bucket
        aff_stats: Affinity statistics (sum, mass, mean) per step/layer
    """
    jsd_rows, contrib_rows = [], []
    avgP: Dict[Tuple[int, int, int], np.ndarray] = {}
    aff_stats: Dict[Tuple[int, int], Dict[str, np.ndarray]] = {}
    
    for step in steps:
        rev = f"step{step}"
        print(f"[info] loading {model_title} @ {rev}")
        
        # Load model
        lm = LanguageModel(model_id, revision=rev, device_map=DEVICE_MAP, 
                          torch_dtype=DTYPE, dispatch=True)
        
        # Prepare bucket data for this step
        bucket_texts, bucket_anchors, bucket_logf = {}, {}, {}
        for bucket, phrases in bucket_phrases.items():
            texts = [build_text(p) for p in phrases]
            bucket_texts[bucket] = texts
            bucket_anchors[bucket] = get_anchor_indices(lm, phrases)
            
            # Get log frequencies for this step
            dfc = (df_counts_all[df_counts_all["step"] == step]
                   .set_index("phrase")[count_col]
                   .reindex(phrases).fillna(0.0))
            bucket_logf[bucket] = np.log(np.clip(dfc.to_numpy(dtype=float), 1e-12, None))
        
        # Process each layer
        for layer in layers:
            H_aff_sum = None    # Σ_p P(h|p) * log f(p)
            H_mass_sum = None   # Σ_p P(h|p)
            
            # Process each bucket
            for bucket, phrases in bucket_phrases.items():
                if not phrases:
                    continue
                
                texts = bucket_texts[bucket]
                anchors = bucket_anchors[bucket]
                rows = []
                
                # Process in batches
                for i0 in range(0, len(phrases), batch_size):
                    sl = slice(i0, i0 + batch_size)
                    caps = capture_activations(lm, texts[sl], anchors[sl], [layer])
                    A = caps[layer]                    # [B, H]
                    P = relu_probs(A, eps=EPS)         # [B, H]
                    rows.append(P)
                
                P_b = np.vstack(rows) if rows else np.zeros((0, 0), dtype=np.float64)
                
                if P_b.size:
                    # Average probability distribution
                    meanP = P_b.mean(axis=0)
                    meanP = meanP / (meanP.sum() + EPS)
                    avgP[(step, layer, bucket)] = meanP
                    
                    # Accumulate sums across buckets
                    # Sum affinity: Σ_p P(h|p) * log f(p)
                    contrib_aff = P_b.T @ bucket_logf[bucket]              # [H]
                    H_aff_sum = contrib_aff if H_aff_sum is None else (H_aff_sum + contrib_aff)
                    
                    # Activation mass: Σ_p P(h|p)
                    mass = P_b.T @ np.ones(P_b.shape[0], dtype=float)     # [H]
                    H_mass_sum = mass if H_mass_sum is None else (H_mass_sum + mass)
                else:
                    avgP[(step, layer, bucket)] = np.array([])
            
            # Store affinity stats per (step, layer)
            if H_aff_sum is None or H_mass_sum is None:
                aff_stats[(step, layer)] = {"sum": np.array([]), "mass": np.array([]), "mean": np.array([])}
            else:
                aff_mean = H_aff_sum / np.maximum(H_mass_sum, EPS)
                aff_stats[(step, layer)] = {"sum": H_aff_sum, "mass": H_mass_sum, "mean": aff_mean}
            
            # Compute JSD between first two buckets
            if len(bucket_phrases) >= 2:
                b0, b1 = list(bucket_phrases.keys())[:2]
                P0 = avgP.get((step, layer, b0), np.array([]))
                P1 = avgP.get((step, layer, b1), np.array([]))
                
                if P0.size and P1.size and P0.shape == P1.shape:
                    contrib = jsd_per_dim(P0, P1, eps=EPS)   # [H]
                    jsd_rows.append({
                        "step": step, "layer": layer, 
                        "bucket_a": b0, "bucket_b": b1, 
                        "JSD": float(contrib.sum())
                    })
                    
                    for i, c in enumerate(contrib):
                        contrib_rows.append({
                            "step": step, "layer": layer, "neuron": i, 
                            "jsd_contrib": float(c)
                        })
                else:
                    jsd_rows.append({
                        "step": step, "layer": layer, 
                        "bucket_a": b0, "bucket_b": b1, 
                        "JSD": np.nan
                    })
        
        # Clean up
        del lm
        torch.cuda.empty_cache()
    
    # Create DataFrames
    df_jsd = pd.DataFrame(jsd_rows).sort_values(["layer", "step"]).reset_index(drop=True)
    df_contrib = pd.DataFrame(contrib_rows).sort_values(["layer", "step", "neuron"]).reset_index(drop=True)
    
    return df_jsd, df_contrib, avgP, aff_stats

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_jsd_over_steps(df_jsd: pd.DataFrame, outdir: Path, model_title: str, tag: str):
    """Plot JSD evolution over training steps for each layer."""
    layers = sorted(df_jsd["layer"].dropna().unique().astype(int).tolist())
    
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    for layer in layers:
        d = df_jsd[df_jsd["layer"] == layer]
        ax.plot(d["step"], d["JSD"], marker="o", lw=1.8, label=f"Layer {layer}")
    
    ax.set_xlabel("Training step")
    ax.set_ylabel("JSD(bin0, bin7)")
    ax.set_title(f"JSD between bin 0 and bin 7 across steps ({model_title})")
    ax.legend(ncols=min(4, len(layers)))
    
    savefig_dual(fig, outdir / f"{tag}_jsd_over_steps.png")
    plt.close(fig)

def plot_avgprob_heatmaps(
    avgP: Dict[Tuple[int, int, int], np.ndarray],
    steps: List[int],
    layers: List[int],
    buckets: List[int],
    outdir: Path,
    model_title: str,
    tag: str,
):
    """Plot heatmaps of average probability distributions."""
    if len(buckets) < 2:
        return
    
    b0, b1 = buckets[:2]
    
    for step in steps:
        for layer in layers:
            P0 = avgP.get((step, layer, b0), np.array([]))
            P1 = avgP.get((step, layer, b1), np.array([]))
            
            if P0.size == 0 or P1.size == 0 or P0.shape[0] != P1.shape[0]:
                continue
            
            M = np.vstack([P0, P1])  # [2, H]
            
            fig, ax = plt.subplots(figsize=(11.0, 3.0), constrained_layout=True)
            im = ax.imshow(M, aspect="auto", interpolation="nearest", origin="upper", cmap="viridis")
            
            ax.set_xlabel("Neuron index")
            ax.set_yticks([0, 1])
            ax.set_yticklabels([f"bucket={b0}", f"bucket={b1}"])
            ax.set_title(f"Avg prob by neuron | step={step} | layer={layer} ({model_title})")
            
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label("Probability")
            
            savefig_dual(fig, outdir / f"{tag}_avgprob_step{step}_layer{layer}.png")
            plt.close(fig)

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function."""
    print("Starting JSD Polysemanticity Analysis Pipeline")
    
    # Setup
    setup_plotting()
    use_buckets = validate_inputs()
    df_counts_all, count_col, bucket2phrases = load_frequency_data(use_buckets)
    
    print(f"Analyzing {len(use_buckets)} buckets: {use_buckets}")
    print(f"Count column: {count_col}")
    
    # Process each model
    for model_config in MODELS:
        tag = model_config["tag"]
        model_title = model_config["title"]
        model_id = model_config["model_id"]
        steps = model_config["steps"]
        layers = model_config["layers"]
        clusters_path = model_config["clusters_parquet"]
        
        print(f"\nProcessing {model_title}...")
        
        # Create output directories
        outdir = OUTROOT / tag
        outdir.mkdir(parents=True, exist_ok=True)
        outdir_jsd = outdir / "jsd"
        outdir_jsd.mkdir(parents=True, exist_ok=True)
        
        # Run main analysis
        df_jsd, df_contrib, avgP, aff_stats = sweep_bins_jsd_with_contrib_and_affinity(
            model_id=model_id,
            model_title=model_title,
            steps=steps,
            layers=layers,
            bucket_phrases=bucket2phrases,
            df_counts_all=df_counts_all,
            count_col=count_col,
            batch_size=BATCH_SIZE
        )
        
        # Save results
        df_jsd.to_csv(outdir_jsd / f"{tag}_between_bin_jsd.csv", index=False)
        df_contrib.to_csv(outdir_jsd / f"{tag}_per_neuron_jsd_contrib.csv", index=False)
        
        # Generate plots
        plot_jsd_over_steps(df_jsd, outdir_jsd, model_title, tag)
        plot_avgprob_heatmaps(avgP, steps, layers, use_buckets, outdir_jsd, model_title, tag)
        
        print(f"Completed {model_title}")
    
    print(f"\nPipeline completed. Outputs saved to: {OUTROOT.resolve()}")

if __name__ == "__main__":
    main()
