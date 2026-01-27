"""
JSD Computation for Synonym Pairs using OLMo models.

Computes Jensen-Shannon Divergence between high-frequency and low-frequency
synonym n-grams using OLMo 1B model activations.

Uses PyTorch hooks for activation capture.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# =============================================================================
# CONFIGURATION
# =============================================================================

# OLMo model configurations
OLMO_MODELS = {
    "1b": {
        "model_id": "allenai/OLMo-1B-hf",
        "num_layers": 16,
    },
    "7b": {
        "model_id": "allenai/OLMo-7B-hf",
        "num_layers": 32,
    },
}

# Default model
DEFAULT_MODEL = "1b"
MODEL_ID = OLMO_MODELS[DEFAULT_MODEL]["model_id"]
NUM_LAYERS = OLMO_MODELS[DEFAULT_MODEL]["num_layers"]

# Use all layers for analysis
LAYERS = list(range(NUM_LAYERS))

DTYPE = torch.float16
DEVICE = (
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)

# Default context template for probing
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
# ACTIVATION CAPTURE WITH HOOKS (OLMo-specific)
# =============================================================================


class OLMoActivationCapture:
    """Captures activations from OLMo model layers using PyTorch hooks."""

    def __init__(self, model, layers: List[int]):
        self.model = model
        self.layers = layers
        self.activations: Dict[int, torch.Tensor] = {}
        self.hooks = []

    def _make_hook(self, layer_idx: int) -> Callable:
        def hook(module, input, output):
            # For OLMo MLP with SwiGLU, input[0] is the input to down_proj
            # This captures the post-activation (after SiLU gate) hidden states
            self.activations[layer_idx] = input[0].detach().cpu()

        return hook

    def register_hooks(self):
        """Register forward hooks on MLP down_proj layers."""
        for L in self.layers:
            # OLMo architecture: model.model.layers[L].mlp.down_proj
            layer = self.model.model.layers[L].mlp.down_proj
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
    """Build the full text prompt for a given n-gram."""
    if template and "[TERM]" in template:
        return template.replace("[TERM]", ngram)
    else:
        return DEFAULT_TEMPLATE.format(ngram=ngram)


def get_anchor_index(tokenizer, ngram: str, template: str = None) -> int:
    """Get token index for the last token of the n-gram in the context."""
    full_text = build_text(ngram, template)
    full_ids = tokenizer.encode(full_text, add_special_tokens=False)

    if template and "[TERM]" in template:
        term_pos = template.find("[TERM]")
        suffix = template[term_pos + 6 :]
        if suffix:
            suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
            suffix_len = len(suffix_ids)
            anchor = len(full_ids) - suffix_len - 1
        else:
            anchor = len(full_ids) - 1
    else:
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
    """Capture post-MLP activations at specific token position."""
    capture = OLMoActivationCapture(model, layers)
    capture.register_hooks()
    capture.clear()

    try:
        inputs = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            _ = model(**inputs)

        out = {}
        for L in layers:
            v = capture.activations[L]  # [1, S, H]
            out[L] = v[0, anchor, :].float().numpy()

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
    """Compute Jensen-Shannon divergence per dimension."""
    P = np.asarray(P, dtype=np.float64)
    P = P / (P.sum() + eps)

    Q = np.asarray(Q, dtype=np.float64)
    Q = Q / (Q.sum() + eps)

    M = 0.5 * (P + Q)
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
    """Compute JSD between high and low frequency n-grams for all layers."""
    high_text = build_text(high_ngram, template)
    low_text = build_text(low_ngram, template)

    high_anchor = get_anchor_index(tokenizer, high_ngram, template)
    low_anchor = get_anchor_index(tokenizer, low_ngram, template)

    high_acts = capture_activations(model, tokenizer, high_text, high_anchor, layers, device)
    low_acts = capture_activations(model, tokenizer, low_text, low_anchor, layers, device)

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
    layers: List[int] = LAYERS,
    device: str = DEVICE,
    num_layers: int = NUM_LAYERS,
    model_name: str = "olmo",
):
    """Run JSD experiment on a filtered pairs dataset using OLMo."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading pairs from {dataset_path}...")
    pairs = load_filtered_pairs(dataset_path)
    print(f"Loaded {len(pairs)} pairs")

    print(f"Loading model {model_id}...")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=DTYPE,
        trust_remote_code=True,
    ).to(device)
    model.eval()
    print(f"Model loaded ({num_layers} layers, analyzing all layers)")

    results = []
    for i, pair in enumerate(pairs):
        high_ngram = pair["high_freq_ngram"]
        low_ngram = pair["low_freq_ngram"]
        template = pair.get("sentence_template")

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
    dataset_name = (
        dataset_path.stem.replace("_filtered_pairs", "")
        .replace("_ngrams", "")
        .replace("_dedup_filtered", "")
    )
    results_path = output_dir / f"{dataset_name}_{model_name}_jsd_results.jsonl"

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
    summary_path = output_dir / f"{dataset_name}_{model_name}_jsd_summary.csv"
    df.to_csv(summary_path, index=False)
    print(f"Saved summary to {summary_path}")

    # Print basic stats
    print("\n" + "=" * 60)
    print(f"SUMMARY STATISTICS ({model_name.upper()})")
    print("=" * 60)
    for L in layers:
        col = f"jsd_layer_{L}"
        if col in df.columns:
            print(
                f"Layer {L:>2}: mean={df[col].mean():.6f}, std={df[col].std():.6f}, "
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
    parser = argparse.ArgumentParser(description="Compute JSD for synonym pairs using OLMo models")
    parser.add_argument(
        "--dataset",
        type=str,
        default="emotion",
        help="Dataset name (emotion, medical, legal, scientific, verb)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="1b",
        choices=["1b", "7b"],
        help="OLMo model size (1b or 7b)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: runs/olmo-{size}/{dataset}/)",
    )
    args = parser.parse_args()

    # Get model config
    model_config = OLMO_MODELS[args.model]
    model_id = model_config["model_id"]
    num_layers = model_config["num_layers"]
    layers = list(range(num_layers))

    # Resolve paths
    script_dir = Path(__file__).parent
    dataset_dir = script_dir / "dataset" / "filtered"

    # Find dataset file - prefer dedup_filtered (has sentence templates)
    dataset_file = dataset_dir / f"{args.dataset}_ngrams_dedup_filtered.jsonl"
    if not dataset_file.exists():
        dataset_file = dataset_dir / f"{args.dataset}_ngrams_filtered_pairs.jsonl"
    if not dataset_file.exists():
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
        output_dir = script_dir / "runs" / f"olmo-{args.model}" / args.dataset

    run_experiment(
        dataset_path=dataset_file,
        output_dir=output_dir,
        model_id=model_id,
        layers=layers,
        num_layers=num_layers,
        model_name=f"olmo-{args.model}",
    )


if __name__ == "__main__":
    main()
