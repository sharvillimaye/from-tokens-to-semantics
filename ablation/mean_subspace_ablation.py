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

from nnsight import LanguageModel
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

    @staticmethod
    def random_direction(
        hidden_dim: int, layer_idx: int, seed: int = 42, name: str = "random_direction"
    ) -> SubspaceDirection:
        """Random unit direction for baseline comparisons."""
        rng = np.random.default_rng(seed)
        return SubspaceDirection(
            vector=torch.tensor(rng.standard_normal(hidden_dim), dtype=torch.float32),
            layer_idx=layer_idx,
            name=name,
            discovery_method="random",
            metadata={"seed": seed},
        )


class MeanSubspaceAblation:
    """Main class for mean subspace ablation experiments."""

    def __init__(self, config: AblationConfig):
        self.config = config
        self.device = config.device
        self.model: Any = None
        self._arch_info: Optional[Dict[str, Any]] = None

    def load_model(self, revision: Optional[str] = None) -> None:
        """Load model using nnsight."""
        print(f"Loading model: {self.config.model_name}")
        kwargs = {"device_map": self.config.device, "torch_dtype": self.config.torch_dtype}
        if revision:
            kwargs["revision"] = revision

        self.model = LanguageModel(self.config.model_name, **kwargs)  # type: ignore
        self.model.eval()
        self._arch_info = self._detect_architecture()
        print(f"Detected architecture: {self._arch_info['type']}")

    def _detect_architecture(self) -> Dict[str, Any]:
        """Detect model architecture and return access patterns."""
        if hasattr(self.model, "gpt_neox"):  # Pythia
            return {
                "type": "pythia",
                "layer_access": lambda idx: self.model.gpt_neox.layers[idx],  # type: ignore
                "residual_stream": lambda layer: layer.input[0],
                "mlp_output": lambda layer: layer.mlp.output,
                "attn_output": lambda layer: layer.attention.output,
            }
        if hasattr(self.model, "transformer") and hasattr(
            self.model.transformer, "blocks"
        ):  # OLMo
            return {
                "type": "olmo",
                "layer_access": lambda idx: self.model.transformer.blocks[idx],  # type: ignore
                "residual_stream": lambda layer: layer.input[0],
                "mlp_output": lambda layer: layer.feed_forward.output,
                "attn_output": lambda layer: layer.attention.output,
            }
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):  # Llama
            return {
                "type": "llama",
                "layer_access": lambda idx: self.model.model.layers[idx],  # type: ignore
                "residual_stream": lambda layer: layer.input[0],
                "mlp_output": lambda layer: layer.mlp.output,
                "attn_output": lambda layer: layer.self_attn.output,
            }
        raise ValueError("Unsupported model architecture")

    def _get_activation_hook_point(self, layer_proxy) -> Any:
        """Get activation hook point based on target_component config."""
        if self._arch_info is None:
            raise RuntimeError("Architecture info not detected. Call load_model() first.")
        component_map = {
            "residual": "residual_stream",
            "mlp": "mlp_output",
            "attention": "attn_output",
        }
        if self.config.target_component not in component_map:
            raise ValueError(f"Unknown target component: {self.config.target_component}")
        return self._arch_info[component_map[self.config.target_component]](layer_proxy)  # type: ignore

    def _extract_at_position(self, act_tensor: torch.Tensor) -> torch.Tensor:
        """Extract activations at last position."""
        return act_tensor[:, -1, :]

    def extract_activations(self, texts: List[str], layer_idx: Optional[int] = None) -> np.ndarray:
        """Extract activations from model for given texts."""
        if self.model is None or self._arch_info is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        layer_idx = layer_idx or self.config.layer_idx
        all_activations = []
        batch_size = self.config.calibration_batch_size

        for batch_start in tqdm(range(0, len(texts), batch_size), desc="Extracting activations"):
            batch_texts = texts[batch_start : batch_start + batch_size]
            with torch.no_grad():
                with self.model.trace(batch_texts):  # type: ignore
                    layer_proxy = self._arch_info["layer_access"](layer_idx)  # type: ignore
                    activations = self._get_activation_hook_point(layer_proxy).save()
                    _ = self.model.output  # type: ignore

            batch_acts = self._extract_at_position(activations.value)
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
        if self.model is None or self._arch_info is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        print(f"Calibrating on {len(calibration_texts)} samples...")
        direction_vec = direction.vector.to(self.device)
        all_projections = []
        batch_size = self.config.calibration_batch_size

        for batch_start in tqdm(range(0, len(calibration_texts), batch_size), desc="Calibrating"):
            batch_texts = calibration_texts[batch_start : batch_start + batch_size]
            with torch.no_grad():
                with self.model.trace(batch_texts):  # type: ignore
                    layer_proxy = self._arch_info["layer_access"](direction.layer_idx)  # type: ignore
                    activations = self._get_activation_hook_point(layer_proxy).save()
                    _ = self.model.output  # type: ignore

            batch_acts = self._extract_at_position(activations.value)
            projections = torch.matmul(batch_acts.float(), direction_vec.float())
            all_projections.append(projections.cpu())

        all_projections = torch.cat(all_projections)
        mean_proj = float(all_projections.mean())
        print(f"Mean projection: {mean_proj:.4f} (std: {float(all_projections.std()):.4f})")
        return mean_proj

    def get_logprobs(
        self,
        text: str,
        direction: Optional[SubspaceDirection] = None,
        mean_proj: Optional[float] = 0.0,
    ) -> float:
        """Compute log probability of text, optionally with ablation."""
        if self.model is None or self._arch_info is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        if direction is not None and mean_proj is None:
            raise ValueError("mean_proj required when direction is provided")

        with torch.no_grad():
            with self.model.trace(text):  # type: ignore
                if direction is not None:
                    direction_vec = direction.vector.to(self.device).float()
                    layer_proxy = self._arch_info["layer_access"](direction.layer_idx)  # type: ignore
                    x = self._get_activation_hook_point(layer_proxy)

                    # Ablation: x_new = x + (mean_proj - current_proj) * v
                    current_proj = torch.matmul(x.float(), direction_vec).unsqueeze(-1)
                    x[:] = x + (mean_proj - current_proj) * direction_vec

                logits = self.model.output.logits.save()  # type: ignore

        logits_val = logits.value
        tokens = self.model.tokenizer(text, return_tensors="pt")["input_ids"].to(self.device)  # type: ignore
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
        include_random_baseline: bool = True,
    ) -> Dict[str, Any]:
        """Run complete ablation experiment with baseline and optional random control."""
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

        # Random baseline
        if include_random_baseline:
            print("\n=== Random Direction Baseline ===")
            random_dir = DirectionDiscovery.random_direction(
                direction.dim, direction.layer_idx, name="random_baseline"
            ).to(self.device)
            random_mean = self.calibrate(random_dir, calibration_texts)
            random_results = self.evaluate_minimal_pairs(
                eval_pairs, random_dir, random_mean, desc="Random"
            )
            results["random_baseline"] = random_results
            random_effect = baseline["accuracy"] - random_results["accuracy"]
            results["random_effect"] = random_effect
            print(
                f"Random accuracy: {random_results['accuracy']:.2%} (effect: {random_effect:+.2%})"
            )

        return results


def load_blimp_minimal_pairs(
    subset: str = "anaphor_number_agreement", max_pairs: Optional[int] = None
) -> List[Tuple[str, str]]:
    """Load minimal pairs from BLIMP dataset."""
    from datasets import load_dataset

    ds: Any = load_dataset("nyu-mll/blimp", subset, split="train")
    pairs = [(row["sentence_good"], row["sentence_bad"]) for row in ds]
    return pairs[:max_pairs] if max_pairs else pairs


if __name__ == "__main__":
    import datetime

    # Configuration
    model_name = "allenai/OLMo-1B-hf"
    num_layers = 16  # OLMo-1B has 16 layers
    blimp_subset = "anaphor_number_agreement"
    max_pairs = 200

    # Load data once
    print("Loading BLiMP dataset...")
    pairs = load_blimp_minimal_pairs(blimp_subset, max_pairs=max_pairs)
    discovery_pairs, eval_pairs = pairs[:100], pairs[100:]

    positive_texts = [p[0] for p in discovery_pairs]
    negative_texts = [p[1] for p in discovery_pairs]
    calibration_texts = positive_texts[:50] + negative_texts[:50]

    # Storage for all results
    all_results = []

    # Create ablator once (we'll reuse it across layers)
    config = AblationConfig(
        model_name=model_name,
        layer_idx=0,  # Will be overridden per layer
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

        # Direction 1: Diff-means
        print(f"\n--- Diff-Means Direction ---")
        diff_direction = DirectionDiscovery.from_diff_means(
            pos_acts, neg_acts, layer, name=f"layer{layer}_diffmeans"
        )

        diff_results = ablator.run_ablation_experiment(
            diff_direction.to(config.device),
            calibration_texts,
            eval_pairs,
            include_random_baseline=True,
        )

        # Direction 2: PCA on differences
        print(f"\n--- PCA on Differences Direction ---")
        pca_direction = DirectionDiscovery.from_pca_on_diff(
            pos_acts, neg_acts, layer, component=0, name=f"layer{layer}_pca"
        )

        pca_results = ablator.run_ablation_experiment(
            pca_direction.to(config.device),
            calibration_texts,
            eval_pairs,
            include_random_baseline=True,
        )

        # Store compact results
        all_results.append(
            {
                "layer": layer,
                "diff_means": {
                    "direction_name": diff_results["direction_name"],
                    "discovery_method": diff_results["discovery_method"],
                    "baseline_accuracy": diff_results["baseline"]["accuracy"],
                    "ablation_accuracy": diff_results["ablation"]["accuracy"],
                    "ablation_effect": diff_results["ablation_effect"],
                    "random_accuracy": diff_results.get("random_baseline", {}).get("accuracy"),
                    "random_effect": diff_results.get("random_effect"),
                    "mean_projection": diff_results["mean_projection"],
                    "metadata": diff_results["direction_metadata"],
                },
                "pca": {
                    "direction_name": pca_results["direction_name"],
                    "discovery_method": pca_results["discovery_method"],
                    "baseline_accuracy": pca_results["baseline"]["accuracy"],
                    "ablation_accuracy": pca_results["ablation"]["accuracy"],
                    "ablation_effect": pca_results["ablation_effect"],
                    "random_accuracy": pca_results.get("random_baseline", {}).get("accuracy"),
                    "random_effect": pca_results.get("random_effect"),
                    "mean_projection": pca_results["mean_projection"],
                    "metadata": pca_results["direction_metadata"],
                },
            }
        )

        # Print layer summary
        print(f"\n{'=' * 80}")
        print(f"LAYER {layer} SUMMARY:")
        print(f"  Diff-Means: {diff_results['ablation_effect']:+.2%} effect")
        print(f"  PCA:        {pca_results['ablation_effect']:+.2%} effect")
        print(f"{'=' * 80}")

    # Save all results
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("cache/ablation_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save full results
    output_path = output_dir / f"full_results_{blimp_subset}_{timestamp}.json"
    with open(output_path, "w") as f:
        json.dump(
            {
                "experiment_info": {
                    "model": model_name,
                    "blimp_subset": blimp_subset,
                    "num_layers": num_layers,
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
    print(f"\nFull results saved to {output_path}")

    # Save summary CSV for easy analysis
    import csv

    summary_path = output_dir / f"summary_{blimp_subset}_{timestamp}.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "layer",
                "method",
                "baseline_acc",
                "ablation_acc",
                "effect",
                "random_acc",
                "random_effect",
                "mean_proj",
            ]
        )

        for result in all_results:
            layer = result["layer"]
            # Diff-means row
            dm = result["diff_means"]
            writer.writerow(
                [
                    layer,
                    "diff_means",
                    f"{dm['baseline_accuracy']:.4f}",
                    f"{dm['ablation_accuracy']:.4f}",
                    f"{dm['ablation_effect']:.4f}",
                    f"{dm.get('random_accuracy', 0):.4f}",
                    f"{dm.get('random_effect', 0):.4f}",
                    f"{dm['mean_projection']:.4f}",
                ]
            )
            # PCA row
            pca = result["pca"]
            writer.writerow(
                [
                    layer,
                    "pca",
                    f"{pca['baseline_accuracy']:.4f}",
                    f"{pca['ablation_accuracy']:.4f}",
                    f"{pca['ablation_effect']:.4f}",
                    f"{pca.get('random_accuracy', 0):.4f}",
                    f"{pca.get('random_effect', 0):.4f}",
                    f"{pca['mean_projection']:.4f}",
                ]
            )

    print(f"Summary CSV saved to {summary_path}")

    # Print final summary
    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE - SUMMARY")
    print("=" * 80)
    print(f"\nModel: {model_name}")
    print(f"BLiMP subset: {blimp_subset}")
    print(f"Layers tested: {num_layers}")
    print("\nTop 3 layers by ablation effect (Diff-Means):")
    sorted_dm = sorted(
        all_results, key=lambda x: abs(x["diff_means"]["ablation_effect"]), reverse=True
    )
    for i, r in enumerate(sorted_dm[:3], 1):
        print(f"  {i}. Layer {r['layer']}: {r['diff_means']['ablation_effect']:+.2%}")

    print("\nTop 3 layers by ablation effect (PCA):")
    sorted_pca = sorted(all_results, key=lambda x: abs(x["pca"]["ablation_effect"]), reverse=True)
    for i, r in enumerate(sorted_pca[:3], 1):
        print(f"  {i}. Layer {r['layer']}: {r['pca']['ablation_effect']:+.2%}")

    print(f"\nResults saved to: {output_dir.absolute()}")
