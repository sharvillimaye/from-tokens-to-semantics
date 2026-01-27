"""
JSD Computation for Synonym Pairs

Computes Jensen-Shannon Divergence between high-frequency and low-frequency
synonym n-grams using Pythia-70M model activations.

Uses PyTorch hooks for activation capture (more reliable than nnsight).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_ID = "EleutherAI/pythia-70m-deduped"
MODEL_STEP = 143000  # Final checkpoint
REVISION = f"step{MODEL_STEP}"
LAYERS = list(range(6))  # Pythia-70M has 6 layers

DTYPE = torch.float16
DEVICE = (
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)

# Default context template for probing (used if no sentence_template in data)
DEFAULT_TEMPLATE = "The word {ngram} means"

# Numerical stability
EPS = 1e-12

# =============================================================================
# DATA LOADING
# =============================================================================


def load_filtered_pairs(path: Path) -> List[dict]:
    """Load filtered synonym pairs from JSONL file."""
    pairs = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    return pairs


# =============================================================================
# ACTIVATION CAPTURE WITH HOOKS
# =============================================================================


class ActivationCapture:
    """Captures activations from model layers using PyTorch hooks."""

    def __init__(self, model, layers: List[int]):
        self.model = model
        self.layers = layers
        self.activations: Dict[int, torch.Tensor] = {}
        self.hooks = []

    def _make_hook(self, layer_idx: int) -> Callable:
        def hook(module, input, output):
            # For Pythia MLP, input[0] is the input to dense_4h_to_h
            # We want the input to this layer (pre-projection activations)
            self.activations[layer_idx] = input[0].detach().cpu()

        return hook

    def register_hooks(self):
        """Register forward hooks on MLP layers."""
        for L in self.layers:
            # Hook on dense_4h_to_h (the down projection in MLP)
            layer = self.model.gpt_neox.layers[L].mlp.dense_4h_to_h
            hook = layer.register_forward_hook(self._make_hook(L))
            self.hooks.append(hook)

    def remove_hooks(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def clear(self):
        """Clear captured activations."""
        self.activations = {}


# =============================================================================
# CORE COMPUTATION FUNCTIONS
# =============================================================================


def build_text(ngram: str, template: str = None) -> str:
    """Build the full text prompt for a given n-gram.

    Args:
        ngram: The n-gram to insert
        template: Optional template with [TERM] placeholder. If None, uses DEFAULT_TEMPLATE.
    """
    if template and "[TERM]" in template:
        return template.replace("[TERM]", ngram)
    else:
        return DEFAULT_TEMPLATE.format(ngram=ngram)


def get_anchor_index(tokenizer, ngram: str, template: str = None) -> int:
    """
    Get token index for the last token of the n-gram in the context.

    For "The word tired means", we want the position of "tired" (or its last token).
    For templates with [TERM], we find the position of the n-gram's last token.
    """
    full_text = build_text(ngram, template)
    full_ids = tokenizer.encode(full_text, add_special_tokens=False)

    # Find where the ngram ends in the text
    if template and "[TERM]" in template:
        # Find position after the ngram by looking at what comes after [TERM]
        term_pos = template.find("[TERM]")
        suffix = template[term_pos + 6 :]  # Text after [TERM]
        if suffix:
            suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
            suffix_len = len(suffix_ids)
            anchor = len(full_ids) - suffix_len - 1
        else:
            # [TERM] is at the end
            anchor = len(full_ids) - 1
    else:
        # Default template: "The word {ngram} means"
        suffix = " means"
        suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
        suffix_len = len(suffix_ids)
        anchor = len(full_ids) - suffix_len - 1

    return anchor


def capture_activations(
    model,
    tokenizer,
    text: str,
    anchor: int,
    layers: List[int],
    device: str,
) -> Dict[int, np.ndarray]:
    """
    Capture post-MLP activations at specific token position.

    Returns dict mapping layer -> activation vector [H].
    """
    # Setup activation capture
    capture = ActivationCapture(model, layers)
    capture.register_hooks()
    capture.clear()

    try:
        # Tokenize and run forward pass
        inputs = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            _ = model(**inputs)

        # Extract activations at anchor position
        out = {}
        for L in layers:
            v = capture.activations[L]  # [1, S, H]
            out[L] = v[0, anchor, :].float().numpy()  # [H]

        return out
    finally:
        capture.remove_hooks()


def relu_probs(activations: np.ndarray, eps: float = EPS) -> np.ndarray:
    """Convert activations to probability distribution using ReLU and normalization."""
    X = np.maximum(activations, 0.0)
    denom = X.sum()
    if denom > eps:
        return X / denom
    return np.zeros_like(X)


def jsd_per_dim(P: np.ndarray, Q: np.ndarray, eps: float = EPS) -> np.ndarray:
    """
    Compute Jensen-Shannon divergence per dimension between distributions P and Q.

    Returns per-dimension contributions (not summed).
    """
    P = np.asarray(P, dtype=np.float64)
    P = P / (P.sum() + eps)

    Q = np.asarray(Q, dtype=np.float64)
    Q = Q / (Q.sum() + eps)

    # Midpoint distribution
    M = 0.5 * (P + Q)

    # JSD = 0.5 * (KL(P||M) + KL(Q||M))
    jsd = 0.5 * (P * (np.log(P + eps) - np.log(M + eps)) + Q * (np.log(Q + eps) - np.log(M + eps)))

    return jsd


def compute_pair_jsd(
    model,
    tokenizer,
    high_ngram: str,
    low_ngram: str,
    layers: List[int],
    device: str,
    template: str = None,
) -> Dict[str, any]:
    """
    Compute JSD between high and low frequency n-grams for all layers.

    Returns dict with per-layer JSD values.
    """
    # Build texts using the same template for both
    high_text = build_text(high_ngram, template)
    low_text = build_text(low_ngram, template)

    # Get anchor positions
    high_anchor = get_anchor_index(tokenizer, high_ngram, template)
    low_anchor = get_anchor_index(tokenizer, low_ngram, template)

    # Capture activations
    high_acts = capture_activations(model, tokenizer, high_text, high_anchor, layers, device)
    low_acts = capture_activations(model, tokenizer, low_text, low_anchor, layers, device)

    # Compute JSD per layer
    jsd_by_layer = {}
    for L in layers:
        P_high = relu_probs(high_acts[L])
        P_low = relu_probs(low_acts[L])

        jsd_contrib = jsd_per_dim(P_high, P_low)
        jsd_by_layer[L] = float(jsd_contrib.sum())

    return jsd_by_layer


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================


def run_experiment(
    dataset_path: Path,
    output_dir: Path,
    model_id: str = MODEL_ID,
    revision: str = REVISION,
    layers: List[int] = LAYERS,
    device: str = DEVICE,
):
    """
    Run JSD experiment on a filtered pairs dataset.

    Args:
        dataset_path: Path to filtered pairs JSONL file
        output_dir: Directory to save results
        model_id: HuggingFace model ID
        revision: Model checkpoint revision
        layers: List of layer indices to analyze
        device: Device to run on (cuda, mps, cpu)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"Loading pairs from {dataset_path}...")
    pairs = load_filtered_pairs(dataset_path)
    print(f"Loaded {len(pairs)} pairs")

    # Load model and tokenizer
    print(f"Loading model {model_id} @ {revision}...")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        torch_dtype=DTYPE,
    ).to(device)
    model.eval()
    print("Model loaded")

    # Process each pair
    results = []
    for i, pair in enumerate(pairs):
        high_ngram = pair["high_freq_ngram"]
        low_ngram = pair["low_freq_ngram"]
        template = pair.get("sentence_template")  # May be None

        template_info = " (template)" if template else " (default)"
        print(f"[{i + 1}/{len(pairs)}] {high_ngram} vs {low_ngram}{template_info}")

        try:
            jsd_by_layer = compute_pair_jsd(
                model, tokenizer, high_ngram, low_ngram, layers, device, template
            )

            result = {
                "high_freq_ngram": high_ngram,
                "low_freq_ngram": low_ngram,
                "high_freq_count": pair.get("high_freq_count", -1),
                "low_freq_count": pair.get("low_freq_count", -1),
                "frequency_ratio": pair.get("frequency_ratio", -1),
                "log_ratio": pair.get("log_ratio", -1),
                "category": pair.get("category", "unknown"),
                "sentence_template": template,
                "jsd_by_layer": jsd_by_layer,
            }
            results.append(result)

        except Exception as e:
            print(f"  Error: {e}")
            import traceback

            traceback.print_exc()
            continue

    # Save results
    dataset_name = dataset_path.stem.replace("_filtered_pairs", "").replace("_ngrams", "")
    results_path = output_dir / f"{dataset_name}_jsd_results.jsonl"

    with results_path.open("w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"\nSaved {len(results)} results to {results_path}")

    # Create summary DataFrame
    rows = []
    for r in results:
        row = {
            "high_freq_ngram": r["high_freq_ngram"],
            "low_freq_ngram": r["low_freq_ngram"],
            "frequency_ratio": r["frequency_ratio"],
            "log_ratio": r["log_ratio"],
            "category": r["category"],
        }
        for L, jsd in r["jsd_by_layer"].items():
            row[f"jsd_layer_{L}"] = jsd
        rows.append(row)

    df = pd.DataFrame(rows)
    summary_path = output_dir / f"{dataset_name}_jsd_summary.csv"
    df.to_csv(summary_path, index=False)
    print(f"Saved summary to {summary_path}")

    # Print basic stats
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    for L in layers:
        col = f"jsd_layer_{L}"
        if col in df.columns:
            print(
                f"Layer {L}: mean={df[col].mean():.6f}, std={df[col].std():.6f}, "
                f"min={df[col].min():.6f}, max={df[col].max():.6f}"
            )

    # Cleanup
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return df


# =============================================================================
# CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Compute JSD for synonym pairs")
    parser.add_argument(
        "--dataset",
        type=str,
        default="emotion",
        help="Dataset name (emotion, medical, legal, scientific, verb)",
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Output directory (default: runs/{dataset}_jsd/)"
    )
    parser.add_argument(
        "--model-step",
        type=int,
        default=MODEL_STEP,
        help=f"Model checkpoint step (default: {MODEL_STEP})",
    )
    args = parser.parse_args()

    # Resolve paths
    script_dir = Path(__file__).parent
    dataset_dir = script_dir / "dataset" / "filtered"

    # Find dataset file - prefer dedup_filtered (has sentence templates)
    dataset_file = dataset_dir / f"{args.dataset}_ngrams_dedup_filtered.jsonl"
    if not dataset_file.exists():
        # Try without dedup
        dataset_file = dataset_dir / f"{args.dataset}_ngrams_filtered_pairs.jsonl"
    if not dataset_file.exists():
        # Try without _ngrams
        dataset_file = dataset_dir / f"{args.dataset}_filtered_pairs.jsonl"
    if not dataset_file.exists():
        print(f"Dataset not found: {dataset_file}")
        print(f"Available datasets:")
        for f in sorted(dataset_dir.glob("*.jsonl")):
            print(f"  - {f.name}")
        return

    # Output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = script_dir / "runs" / f"{args.dataset}_jsd"

    # Update revision if custom step
    revision = f"step{args.model_step}"

    # Run experiment
    run_experiment(
        dataset_path=dataset_file,
        output_dir=output_dir,
        revision=revision,
    )


if __name__ == "__main__":
    main()
