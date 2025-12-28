#!/usr/bin/env python3
"""
Mean Subspace Ablation for Causal Intervention in Language Models

Ablation formula: x_new = x - (x · v) * v + μ * v
Where v is the direction and μ is the mean projection from calibration.

This preserves activation norm while removing information in direction v.
"""

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import warnings

from nnterp import StandardizedTransformer
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import torch
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)


@dataclass
class AblationConfig:
    """Configuration for mean subspace ablation experiments."""

    model_name: str = "allenai/OLMo-1B-hf"
    device: str = "cuda"
    torch_dtype: torch.dtype = torch.float16
    layer_idx: int = 5
    target_component: str = "residual"
    calibration_batch_size: int = 8
    eval_batch_size: int = 4


@dataclass
class SubspaceDirection:
    """A direction in activation space for ablation."""

    vector: torch.Tensor
    layer_idx: int
    name: str = "unnamed"
    discovery_method: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        norm = torch.norm(self.vector)
        if norm > 0:
            self.vector = self.vector / norm

    @property
    def dim(self) -> int:
        return self.vector.shape[0]

    def to(self, device: str) -> "SubspaceDirection":
        return SubspaceDirection(
            vector=self.vector.to(device),
            layer_idx=self.layer_idx,
            name=self.name,
            discovery_method=self.discovery_method,
            metadata=self.metadata,
        )


class DirectionDiscovery:
    """Methods for discovering meaningful directions in activation space."""

    @staticmethod
    def from_diff_means(
        activations_positive: np.ndarray,
        activations_negative: np.ndarray,
        layer_idx: int,
        name: str = "diff_means_direction",
    ) -> SubspaceDirection:
        """Direction from difference of means: v = mean(positive) - mean(negative)."""
        direction = np.mean(activations_positive, axis=0) - np.mean(activations_negative, axis=0)
        return SubspaceDirection(
            vector=torch.tensor(direction, dtype=torch.float32),
            layer_idx=layer_idx,
            name=name,
            discovery_method="diff_means",
            metadata={
                "n_positive": len(activations_positive),
                "n_negative": len(activations_negative),
                "raw_norm": float(np.linalg.norm(direction)),
            },
        )

    @staticmethod
    def from_pca(
        activations: np.ndarray, layer_idx: int, component: int = 0, name: Optional[str] = None
    ) -> SubspaceDirection:
        """Extract a principal component as the direction."""
        scaler = StandardScaler()
        pca = PCA()
        pca.fit(scaler.fit_transform(activations))

        return SubspaceDirection(
            vector=torch.tensor(pca.components_[component], dtype=torch.float32),
            layer_idx=layer_idx,
            name=name or f"pca_component_{component}",
            discovery_method="pca",
            metadata={
                "component": component,
                "explained_variance_ratio": float(pca.explained_variance_ratio_[component]),
                "n_samples": len(activations),
            },
        )

    @staticmethod
    def from_pca_on_diff(
        activations_positive: np.ndarray,
        activations_negative: np.ndarray,
        layer_idx: int,
        component: int = 0,
        name: Optional[str] = None,
    ) -> SubspaceDirection:
        """PCA on pairwise differences (requires paired data)."""
        assert len(activations_positive) == len(activations_negative), (
            "Need paired data with same number of samples"
        )

        differences = activations_positive - activations_negative
        scaler = StandardScaler()
        pca = PCA()
        pca.fit(scaler.fit_transform(differences))

        return SubspaceDirection(
            vector=torch.tensor(pca.components_[component], dtype=torch.float32),
            layer_idx=layer_idx,
            name=name or f"diff_pca_component_{component}",
            discovery_method="pca_on_diff",
            metadata={
                "component": component,
                "explained_variance_ratio": float(pca.explained_variance_ratio_[component]),
                "n_pairs": len(differences),
            },
        )


class MeanSubspaceAblation:
    """Main class for mean subspace ablation experiments.

    Uses nnterp's StandardizedTransformer for architecture-agnostic model access.
    Supports: Pythia, OLMo, Llama, Qwen, Gemma, and other transformer architectures.
    """

    def __init__(self, config: AblationConfig):
        self.config = config
        self.device = config.device
        self.model: Optional[StandardizedTransformer] = None

    def load_model(self, revision: Optional[str] = None) -> None:
        """Load model using nnterp's StandardizedTransformer.

        This provides architecture-agnostic access to layers and activations.
        """
        print(f"Loading model: {self.config.model_name}")
        kwargs: Dict[str, Any] = {
            "device_map": self.config.device,
            "torch_dtype": self.config.torch_dtype,
        }
        if revision:
            kwargs["revision"] = revision

        self.model = StandardizedTransformer(self.config.model_name, **kwargs)
        print(
            f"Loaded model with {self.model.num_layers} layers, hidden_size={self.model.hidden_size}"
        )

    def _get_activations_accessor(self, layer_idx: int) -> Any:
        """Get activation accessor based on target_component config.

        Uses nnterp's standardized accessors for architecture-agnostic access.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if self.config.target_component == "residual":
            return self.model.layers_input[layer_idx]
        elif self.config.target_component == "mlp":
            return self.model.mlps_output[layer_idx]
        elif self.config.target_component == "attention":
            return self.model.attentions_output[layer_idx]
        else:
            raise ValueError(f"Unknown target component: {self.config.target_component}")

    def _set_activations(self, layer_idx: int, value: Any) -> None:
        """Set activations at a layer based on target_component config."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if self.config.target_component == "residual":
            self.model.layers_input[layer_idx] = value
        elif self.config.target_component == "mlp":
            self.model.mlps_output[layer_idx] = value
        elif self.config.target_component == "attention":
            self.model.attentions_output[layer_idx] = value
        else:
            raise ValueError(f"Unknown target component: {self.config.target_component}")

    def _extract_at_position(self, act_tensor: torch.Tensor) -> torch.Tensor:
        """Extract activations at last position.
        Handles cases where sequence_length dimension might be squeezed.
        """
        if act_tensor.dim() == 3:
            return act_tensor[:, -1, :]
        elif act_tensor.dim() == 2:
            return act_tensor
        else:
            raise ValueError(f"Unexpected activation tensor dimension: {act_tensor.dim()}")

    def extract_activations(self, texts: List[str], layer_idx: Optional[int] = None) -> np.ndarray:
        """Extract activations from model for given texts."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        layer_idx = layer_idx if layer_idx is not None else self.config.layer_idx
        all_activations = []
        batch_size = self.config.calibration_batch_size

        for batch_start in tqdm(range(0, len(texts), batch_size), desc="Extracting activations"):
            batch_texts = texts[batch_start : batch_start + batch_size]
            with torch.no_grad():
                with self.model.trace(batch_texts):
                    activations = self._get_activations_accessor(layer_idx).save()
                    _ = self.model.output

            batch_acts = self._extract_at_position(activations)
            all_activations.append(batch_acts.cpu().numpy())

        return np.concatenate(all_activations, axis=0)

    def extract_paired_activations(
        self,
        positive_texts: List[str],
        negative_texts: List[str],
        layer_idx: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract activations for paired positive/negative examples."""
        print(f"Extracting positive activations ({len(positive_texts)} samples)...")
        pos_acts = self.extract_activations(positive_texts, layer_idx)
        print(f"Extracting negative activations ({len(negative_texts)} samples)...")
        neg_acts = self.extract_activations(negative_texts, layer_idx)
        return pos_acts, neg_acts

    def calibrate(self, direction: SubspaceDirection, calibration_texts: List[str]) -> float:
        """Compute mean projection onto direction for calibration."""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        print(f"Calibrating on {len(calibration_texts)} samples...")
        direction_vec = direction.vector.to(self.device)
        all_projections = []
        batch_size = self.config.calibration_batch_size

        for batch_start in tqdm(range(0, len(calibration_texts), batch_size), desc="Calibrating"):
            batch_texts = calibration_texts[batch_start : batch_start + batch_size]
            with torch.no_grad():
                with self.model.trace(batch_texts):
                    activations = self._get_activations_accessor(direction.layer_idx).save()
                    _ = self.model.output

            batch_acts = self._extract_at_position(activations)
            projections = torch.matmul(batch_acts.float(), direction_vec.float())
            all_projections.append(projections.cpu())

        all_projections_tensor = torch.cat(all_projections)
        mean_proj = float(all_projections_tensor.mean())
        print(f"Mean projection: {mean_proj:.4f} (std: {float(all_projections_tensor.std()):.4f})")
        return mean_proj

    def get_logprobs(
        self,
        text: str,
        direction: Optional[SubspaceDirection] = None,
        mean_proj: Optional[float] = 0.0,
    ) -> float:
        """Compute log probability of text, optionally with ablation.

        Ablation formula: x_new = x - (x · v) * v + μ * v
        Which simplifies to: x_new = x + (μ - x · v) * v
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        if direction is not None and mean_proj is None:
            raise ValueError("mean_proj required when direction is provided")

        with torch.no_grad():
            with self.model.trace(text):
                if direction is not None:
                    x = self._get_activations_accessor(direction.layer_idx)
                    # Match direction vector dtype to activation dtype (float16/bfloat16)
                    direction_vec = direction.vector.to(device=self.device, dtype=x.dtype)

                    # Ablation: x_new = x + (mean_proj - current_proj) * v
                    # x has shape [B, S, H], direction_vec has shape [H]
                    current_proj = torch.matmul(x, direction_vec)  # [B, S]
                    # Expand for broadcasting: [B, S, 1] * [H] -> [B, S, H]
                    ablation_delta = (mean_proj - current_proj).unsqueeze(-1) * direction_vec
                    self._set_activations(direction.layer_idx, x + ablation_delta)

                logits = self.model.output.logits.save()

        logits_val = logits
        tokens = self.model.tokenizer(text, return_tensors="pt")["input_ids"].to(self.device)
        log_probs = torch.nn.functional.log_softmax(logits_val, dim=-1)

        return sum(log_probs[0, i - 1, tokens[0, i]].item() for i in range(1, tokens.shape[1]))

    def evaluate_minimal_pairs(
        self,
        pairs: List[Tuple[str, str]],
        direction: Optional[SubspaceDirection] = None,
        mean_proj: Optional[float] = None,
        desc: str = "Evaluating",
    ) -> Dict[str, Any]:
        """Evaluate on minimal pairs. Correct if P(good) > P(bad)."""
        results = []
        for good_sent, bad_sent in tqdm(pairs, desc=desc):
            good_lp = self.get_logprobs(good_sent, direction, mean_proj)
            bad_lp = self.get_logprobs(bad_sent, direction, mean_proj)
            results.append(
                {
                    "good_sentence": good_sent,
                    "bad_sentence": bad_sent,
                    "good_logprob": good_lp,
                    "bad_logprob": bad_lp,
                    "correct": good_lp > bad_lp,
                    "margin": good_lp - bad_lp,
                }
            )

        correct = sum(r["correct"] for r in results)
        return {
            "accuracy": correct / len(results) if results else 0.0,
            "correct": correct,
            "total": len(results),
            "results": results,
            "ablation_applied": direction is not None,
        }

    def run_ablation_experiment(
        self,
        direction: SubspaceDirection,
        calibration_texts: List[str],
        eval_pairs: List[Tuple[str, str]],
    ) -> Dict[str, Any]:
        """Run complete ablation experiment with baseline."""
        results = {
            "direction_name": direction.name,
            "layer_idx": direction.layer_idx,
            "discovery_method": direction.discovery_method,
            "direction_metadata": direction.metadata,
        }

        # Baseline
        print("\n=== Baseline (No Ablation) ===")
        baseline = self.evaluate_minimal_pairs(eval_pairs, desc="Baseline")
        results["baseline"] = baseline
        print(f"Baseline accuracy: {baseline['accuracy']:.2%}")

        # Calibrate and ablate
        print("\n=== Calibrating Direction ===")
        mean_proj = self.calibrate(direction, calibration_texts)
        results["mean_projection"] = mean_proj

        print("\n=== Ablation Evaluation ===")
        ablation = self.evaluate_minimal_pairs(eval_pairs, direction, mean_proj, desc="Ablated")
        results["ablation"] = ablation
        effect = baseline["accuracy"] - ablation["accuracy"]
        results["ablation_effect"] = effect
        print(f"Ablated accuracy: {ablation['accuracy']:.2%} (effect: {effect:+.2%})")

        return results


def visualization(
    results: List[Dict[str, Any]],
    results_dir: Path,
    model_name: str,
):
    """Create comprehensive visualizations for ablation results.

    Args:
        results: List of result dictionaries, each containing:
            - layer: int
            - diff_means: dict with baseline_accuracy, ablation_accuracy, ablation_effect, etc.
        results_dir: Directory to save plots
        model_name: Name of the model (for plot titles and filenames)
    """
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    # Extract data
    layers = [r["layer"] for r in results]
    dm = [r["diff_means"] for r in results]

    baseline_acc = [d["baseline_accuracy"] for d in dm]
    ablation_acc = [d["ablation_accuracy"] for d in dm]
    ablation_effect = [d["ablation_effect"] for d in dm]
    mean_proj = [d["mean_projection"] for d in dm]

    # Clean model name for filename
    clean_name = model_name.replace("/", "_").replace("-", "_")

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f"Ablation Analysis: {model_name}", fontsize=16, fontweight="bold")

    # Plot 1: Ablation Effect by Layer (Main Result)
    ax1 = axes[0, 0]
    ax1.plot(
        layers,
        ablation_effect,
        marker="o",
        linewidth=2,
        markersize=8,
        color="#2E86AB",
        label="Ablation Effect",
    )
    ax1.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax1.fill_between(layers, ablation_effect, 0, alpha=0.3, color="#2E86AB")
    ax1.set_xlabel("Layer", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Ablation Effect (Δ Accuracy)", fontsize=12, fontweight="bold")
    ax1.set_title("Ablation Effect by Layer", fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)

    # Highlight top 3 layers
    top_3_indices = sorted(
        range(len(ablation_effect)), key=lambda i: abs(ablation_effect[i]), reverse=True
    )[:3]
    for idx in top_3_indices:
        ax1.plot(
            layers[idx], ablation_effect[idx], marker="*", markersize=15, color="red", zorder=5
        )

    # Plot 2: Accuracy Comparison (Baseline vs Ablated)
    ax2 = axes[0, 1]
    width = 0.35
    x = range(len(layers))
    ax2.bar(
        [i - width / 2 for i in x],
        baseline_acc,
        width,
        label="Baseline",
        color="#06A77D",
        alpha=0.8,
    )
    ax2.bar(
        [i + width / 2 for i in x],
        ablation_acc,
        width,
        label="Ablated",
        color="#D81E5B",
        alpha=0.8,
    )
    ax2.set_xlabel("Layer", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Accuracy", fontsize=12, fontweight="bold")
    ax2.set_title("Accuracy Comparison by Layer", fontsize=14, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(layers)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis="y")

    # Plot 3: Mean Projection Values
    ax3 = axes[1, 0]
    colors = ["#FF6B6B" if abs(e) > 0.05 else "#95E1D3" for e in ablation_effect]
    ax3.bar(layers, mean_proj, color=colors, alpha=0.7, edgecolor="black")
    ax3.set_xlabel("Layer", fontsize=12, fontweight="bold")
    ax3.set_ylabel("Mean Projection", fontsize=12, fontweight="bold")
    ax3.set_title("Mean Projection Values (Calibration)", fontsize=14, fontweight="bold")
    ax3.grid(True, alpha=0.3, axis="y")

    # Add legend for colors
    high_patch = mpatches.Patch(color="#FF6B6B", alpha=0.7, label="High Effect (|Δ| > 0.05)")
    low_patch = mpatches.Patch(color="#95E1D3", alpha=0.7, label="Low Effect (|Δ| ≤ 0.05)")
    ax3.legend(handles=[high_patch, low_patch], fontsize=10)

    # Plot 4: Effect Size Heatmap-style visualization
    ax4 = axes[1, 1]

    # Normalize effects for color mapping
    max_abs_effect = max(abs(e) for e in ablation_effect) if ablation_effect else 1
    normalized_effects = [e / max_abs_effect if max_abs_effect > 0 else 0 for e in ablation_effect]

    # Create color-coded bars
    colors_gradient = plt.cm.RdYlGn_r([0.5 + 0.5 * ne for ne in normalized_effects])
    bars = ax4.barh(
        layers,
        [abs(e) for e in ablation_effect],
        color=colors_gradient,
        edgecolor="black",
        alpha=0.8,
    )
    ax4.set_ylabel("Layer", fontsize=12, fontweight="bold")
    ax4.set_xlabel("|Ablation Effect|", fontsize=12, fontweight="bold")
    ax4.set_title("Effect Magnitude by Layer (Sorted by Layer)", fontsize=14, fontweight="bold")
    ax4.invert_yaxis()
    ax4.grid(True, alpha=0.3, axis="x")

    # Add value labels on bars
    for i, (layer, effect) in enumerate(zip(layers, ablation_effect)):
        ax4.text(
            abs(effect) + max_abs_effect * 0.01, layer, f"{effect:+.3f}", va="center", fontsize=9
        )

    plt.tight_layout()

    # Save plot
    plot_path = results_dir / f"ablation_analysis_{clean_name}.pdf"
    plt.savefig(plot_path, bbox_inches="tight")
    print(f"Saved visualization to {plot_path}")
    plt.close()

    # Create a second figure: Line plot with confidence-style bands
    fig2, ax = plt.subplots(figsize=(14, 6))

    ax.plot(
        layers,
        baseline_acc,
        marker="o",
        linewidth=2.5,
        markersize=8,
        color="#06A77D",
        label="Baseline (No Ablation)",
        zorder=3,
    )
    ax.plot(
        layers,
        ablation_acc,
        marker="s",
        linewidth=2.5,
        markersize=8,
        color="#D81E5B",
        label="Ablated (Diff-Means Direction)",
        zorder=3,
    )

    ax.set_xlabel("Layer", fontsize=14, fontweight="bold")
    ax.set_ylabel("Accuracy", fontsize=14, fontweight="bold")
    ax.set_title(f"Layer-wise Accuracy: {model_name}", fontsize=16, fontweight="bold")
    ax.legend(fontsize=12, loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(
        [
            min(min(baseline_acc), min(ablation_acc)) - 0.05,
            max(max(baseline_acc), max(ablation_acc)) + 0.05,
        ]
    )

    plt.tight_layout()
    plot_path2 = results_dir / f"accuracy_by_layer_{clean_name}.pdf"
    plt.savefig(plot_path2, bbox_inches="tight")
    print(f"Saved accuracy plot to {plot_path2}")
    plt.close()

    # Create a third figure: Statistical summary
    fig3, ax = plt.subplots(figsize=(10, 6))

    # Sort by effect size
    sorted_indices = sorted(
        range(len(ablation_effect)), key=lambda i: ablation_effect[i], reverse=True
    )
    sorted_layers = [layers[i] for i in sorted_indices]
    sorted_effects = [ablation_effect[i] for i in sorted_indices]

    colors_sorted = ["#D81E5B" if e > 0 else "#2E86AB" for e in sorted_effects]
    _ = ax.barh(
        range(len(sorted_layers)),
        sorted_effects,
        color=colors_sorted,
        alpha=0.8,
        edgecolor="black",
    )
    ax.set_yticks(range(len(sorted_layers)))
    ax.set_yticklabels([f"Layer {l}" for l in sorted_layers])
    ax.set_xlabel("Ablation Effect (Δ Accuracy)", fontsize=12, fontweight="bold")
    ax.set_title(f"Ablation Effect Ranked by Impact: {model_name}", fontsize=14, fontweight="bold")
    ax.axvline(x=0, color="black", linewidth=1)
    ax.grid(True, alpha=0.3, axis="x")

    # Add value labels
    for i, effect in enumerate(sorted_effects):
        ax.text(
            effect + (0.003 if effect > 0 else -0.003),
            i,
            f"{effect:+.3f}",
            va="center",
            ha="left" if effect > 0 else "right",
            fontsize=9,
        )

    plt.tight_layout()
    plot_path3 = results_dir / f"effect_ranked_{clean_name}.pdf"
    plt.savefig(plot_path3, bbox_inches="tight")
    print(f"Saved ranked effect plot to {plot_path3}")
    plt.close()


def load_blimp_minimal_pairs(
    subset: str = "anaphor_number_agreement", max_pairs: Optional[int] = None
) -> List[Tuple[str, str]]:
    """Load minimal pairs from BLIMP dataset."""
    from datasets import load_dataset

    ds: Any = load_dataset("nyu-mll/blimp", subset, split="train")
    pairs = [(row["sentence_good"], row["sentence_bad"]) for row in ds]
    return pairs[:max_pairs] if max_pairs else pairs


if __name__ == "__main__":
    import csv
    from datetime import datetime
    import random

    # Set random seeds for reproducibility
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    model_layers: dict[str, int] = {
        # "EleutherAI/pythia-70m-deduped": 6,
        # "EleutherAI/pythia-1b-deduped": 16,
        # "EleutherAI/pythia-2.8b-deduped": 32,
        # "allenai/OLMo-1B-hf": 16,
        # "allenai/OLMo-7B-hf": 32,
        # "Qwen/Qwen2.5-1.5B": 28,
        "google/gemma-2-2b": 26,
        "EleutherAI/pythia-6.9b-deduped": 32,
        # "meta-llama/Meta-Llama-3-8B": 32,
        "google/gemma-2-9b": 42,
        "mistralai/Mistral-7B-v0.3": 32,
    }
    blimp_subset = "anaphor_number_agreement"
    max_pairs = None

    print("Loading BLiMP dataset...")
    pairs = load_blimp_minimal_pairs(blimp_subset, max_pairs=max_pairs)

    # Split into non-overlapping sets to avoid data leakage:
    # - discovery_pairs: used to find the ablation direction (diff of means)
    # - calibration_pairs: used to compute mean projection for ablation
    # - eval_pairs: used to evaluate the effect of ablation
    if len(pairs) < 200:
        raise ValueError(
            f"Insufficient pairs in BLiMP subset '{blimp_subset}': {len(pairs)} < 200. "
            "Need at least 100 for discovery, 50 for calibration, and 50 for evaluation."
        )

    discovery_pairs = pairs[:100]
    calibration_pairs = pairs[100:150]
    eval_pairs = pairs[150:450]  # 300 samples - good balance of speed vs precision

    print(
        f"Data split: {len(discovery_pairs)} discovery, {len(calibration_pairs)} calibration, {len(eval_pairs)} eval"
    )

    positive_texts = [p[0] for p in discovery_pairs]
    negative_texts = [p[1] for p in discovery_pairs]
    # Calibration uses separate, non-overlapping data
    calibration_texts = [p[0] for p in calibration_pairs] + [p[1] for p in calibration_pairs]

    # Storage for all results
    all_results = []

    # Create output directory
    output_dir = Path("cache/ablation_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    import gc
    import shutil

    def clear_model_cache():
        """Clear HuggingFace cache to free disk space."""
        cache_dirs = [
            Path.home() / ".cache/huggingface/hub",
            Path("/workspace/.cache/huggingface/hub"),
        ]
        for cache_dir in cache_dirs:
            if cache_dir.exists():
                for item in cache_dir.glob("models--*"):
                    try:
                        shutil.rmtree(item)
                        print(f"Cleared cache: {item.name}")
                    except Exception as e:
                        print(f"Failed to clear {item}: {e}")

    for model_name, num_layers in model_layers.items():
        print(f"\n\n{'#' * 100}")
        print(f"RUNNING EXPERIMENT FOR MODEL: {model_name} with {num_layers} layers")
        print(f"{'#' * 100}\n")

        # Create ablator once (we'll reuse it across layers)
        config = AblationConfig(
            model_name=model_name,
            layer_idx=0,
            target_component="residual",
            calibration_batch_size=4,
        )

        ablator = MeanSubspaceAblation(config)
        ablator.load_model()

        # Run experiment for each layer
        for layer in range(num_layers):
            print(f"\n{'=' * 80}")
            print(f"LAYER {layer} / {num_layers - 1}")
            print(f"{'=' * 80}")

            # Extract activations for this layer
            print(f"\nExtracting activations at layer {layer}...")
            pos_acts, neg_acts = ablator.extract_paired_activations(
                positive_texts, negative_texts, layer_idx=layer
            )

            print("\n--- Diff-Means Direction ---")
            diff_direction = DirectionDiscovery.from_diff_means(
                pos_acts, neg_acts, layer, name=f"layer{layer}_diffmeans"
            )

            diff_results = ablator.run_ablation_experiment(
                diff_direction.to(config.device),
                calibration_texts,
                eval_pairs,
            )

            # Store compact results
            all_results.append(
                {
                    "model": model_name,
                    "layer": layer,
                    "diff_means": {
                        "direction_name": diff_results["direction_name"],
                        "discovery_method": diff_results["discovery_method"],
                        "baseline_accuracy": diff_results["baseline"]["accuracy"],
                        "ablation_accuracy": diff_results["ablation"]["accuracy"],
                        "ablation_effect": diff_results["ablation_effect"],
                        "mean_projection": diff_results["mean_projection"],
                        "metadata": diff_results["direction_metadata"],
                    },
                }
            )

            # Print layer summary
            print(f"\n{'=' * 80}")
            print(f"LAYER {layer} SUMMARY:")
            print(f"  Diff-Means: {diff_results['ablation_effect']:+.2%} effect")
            print(f"{'=' * 80}")

            # Clean up GPU memory after each layer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Save results and visualizations for this model
        print(f"\n{'#' * 100}")
        print(f"SAVING RESULTS FOR MODEL: {model_name}")
        print(f"{'#' * 100}\n")

        # Filter results for current model
        model_results = [r for r in all_results if r["model"] == model_name]

        # Generate timestamp for this model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        clean_model_name = model_name.replace("/", "_").replace("-", "_")

        # Save CSV for this model
        model_csv_path = output_dir / f"summary_{clean_model_name}_{blimp_subset}_{timestamp}.csv"
        with open(model_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "model",
                    "layer",
                    "method",
                    "baseline_acc",
                    "ablation_acc",
                    "effect",
                    "mean_proj",
                ]
            )

            for result in model_results:
                dm = result["diff_means"]
                writer.writerow(
                    [
                        result["model"],
                        result["layer"],
                        "diff_means",
                        f"{dm['baseline_accuracy']:.4f}",
                        f"{dm['ablation_accuracy']:.4f}",
                        f"{dm['ablation_effect']:.4f}",
                        f"{dm['mean_projection']:.4f}",
                    ]
                )

        print(f"Saved model CSV to {model_csv_path}")

        # Save JSON for this model
        model_json_path = (
            output_dir / f"results_{clean_model_name}_{blimp_subset}_{timestamp}.json"
        )
        with open(model_json_path, "w") as f:
            json.dump(
                {
                    "experiment_info": {
                        "model": model_name,
                        "num_layers": num_layers,
                        "blimp_subset": blimp_subset,
                        "discovery_pairs": len(discovery_pairs),
                        "eval_pairs": len(eval_pairs),
                        "calibration_samples": len(calibration_texts),
                        "timestamp": timestamp,
                    },
                    "results": model_results,
                },
                f,
                indent=2,
            )

        print(f"Saved model JSON to {model_json_path}")

        # Generate visualizations for this model
        print(f"Generating visualizations for {model_name}...")
        visualization(model_results, output_dir, model_name)

        # Clean up model from memory and disk after processing all layers
        print(f"\nCleaning up {model_name}...")
        del ablator.model
        del ablator
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        clear_model_cache()
        print(f"Cleanup complete for {model_name}\n")

    # Save summary CSV for easy analysis after all models are done
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = output_dir / f"summary_all_models_{blimp_subset}_{timestamp}.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "model",
                "layer",
                "method",
                "baseline_acc",
                "ablation_acc",
                "effect",
                "mean_proj",
            ]
        )

        for result in all_results:
            model = result["model"]
            layer = result["layer"]
            # Diff-means row
            dm = result["diff_means"]
            writer.writerow(
                [
                    model,
                    layer,
                    "diff_means",
                    f"{dm['baseline_accuracy']:.4f}",
                    f"{dm['ablation_accuracy']:.4f}",
                    f"{dm['ablation_effect']:.4f}",
                    f"{dm['mean_projection']:.4f}",
                ]
            )

    print(f"\nAll models summary CSV saved to {summary_path}")

    # Save full results as JSON
    full_results_path = output_dir / f"full_results_{blimp_subset}_{timestamp}.json"
    with open(full_results_path, "w") as f:
        json.dump(
            {
                "experiment_info": {
                    "models": list(model_layers.keys()),
                    "blimp_subset": blimp_subset,
                    "discovery_pairs": len(discovery_pairs),
                    "eval_pairs": len(eval_pairs),
                    "calibration_samples": len(calibration_texts),
                    "timestamp": timestamp,
                },
                "results": all_results,
            },
            f,
            indent=2,
        )
