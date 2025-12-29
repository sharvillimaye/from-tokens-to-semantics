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


@dataclass
class SubspaceDirections:
    """A subspace in activation space for multi-direction ablation.

    Holds multiple orthonormal directions (e.g., top-k PCA components) for
    simultaneous ablation. The ablation formula becomes:

        x_new = x - V @ V.T @ x + V @ μ

    Where V is the matrix of orthonormal directions and μ contains the mean
    projections for each direction.
    """

    vectors: torch.Tensor  # Shape: [hidden_dim, k] - orthonormal columns
    layer_idx: int
    name: str = "unnamed"
    discovery_method: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Ensure vectors are orthonormal (PCA components already are, but normalize for safety)
        # vectors shape: [hidden_dim, k]
        if self.vectors.dim() == 1:
            # Single vector passed, reshape to [hidden_dim, 1]
            self.vectors = self.vectors.unsqueeze(1)

        # Normalize each column to unit norm
        norms = torch.norm(self.vectors, dim=0, keepdim=True)
        self.vectors = self.vectors / norms.clamp(min=1e-8)

    @property
    def hidden_dim(self) -> int:
        return self.vectors.shape[0]

    @property
    def n_components(self) -> int:
        return self.vectors.shape[1]

    def to(self, device: str) -> "SubspaceDirections":
        return SubspaceDirections(
            vectors=self.vectors.to(device),
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
    def from_pca_on_diff_subspace(
        activations_positive: np.ndarray,
        activations_negative: np.ndarray,
        layer_idx: int,
        n_components: int = 5,
        variance_threshold: Optional[float] = None,
        name: Optional[str] = None,
    ) -> SubspaceDirections:
        """PCA on pairwise differences returning top-k subspace for multi-direction ablation.

        Args:
            activations_positive: Activations for positive examples, shape [n_samples, hidden_dim]
            activations_negative: Activations for negative examples, shape [n_samples, hidden_dim]
            layer_idx: Layer index these activations came from
            n_components: Number of top principal components to include (default: 5)
            variance_threshold: If provided, select components until cumulative explained
                variance reaches this threshold (e.g., 0.9 for 90%). Overrides n_components.
            name: Optional name for the subspace

        Returns:
            SubspaceDirections containing the top-k orthonormal directions
        """
        assert len(activations_positive) == len(activations_negative), (
            "Need paired data with same number of samples"
        )

        differences = activations_positive - activations_negative
        scaler = StandardScaler()
        pca = PCA()
        pca.fit(scaler.fit_transform(differences))

        # Determine number of components to use
        requested_n_components = n_components
        if variance_threshold is not None:
            cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
            n_components = int(np.searchsorted(cumulative_variance, variance_threshold) + 1)
            n_components = min(n_components, len(pca.components_))

        # Ensure we don't exceed available components (limited by min(n_samples, hidden_dim))
        max_components = len(pca.components_)
        n_components = min(n_components, max_components)

        if n_components < requested_n_components and variance_threshold is None:
            print(f"Warning: Requested {requested_n_components} components but only {max_components} available. Using {n_components}.")

        # Extract top-k components: pca.components_ has shape [n_components, hidden_dim]
        # We want [hidden_dim, k] for our convention
        top_k_components = pca.components_[:n_components].T  # [hidden_dim, k]

        cumulative_variance = float(np.sum(pca.explained_variance_ratio_[:n_components]))

        return SubspaceDirections(
            vectors=torch.tensor(top_k_components, dtype=torch.float32),
            layer_idx=layer_idx,
            name=name or f"diff_pca_subspace_k{n_components}",
            discovery_method="pca_on_diff_subspace",
            metadata={
                "n_components": n_components,
                "explained_variance_ratios": pca.explained_variance_ratio_[:n_components].tolist(),
                "cumulative_explained_variance": cumulative_variance,
                "n_pairs": len(differences),
                "variance_threshold": variance_threshold,
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

    def calibrate_subspace(
        self, subspace: SubspaceDirections, calibration_texts: List[str]
    ) -> torch.Tensor:
        """Compute mean projections onto each direction in the subspace.

        Args:
            subspace: SubspaceDirections containing k orthonormal directions
            calibration_texts: Texts to use for computing mean projections

        Returns:
            Tensor of shape [k] containing mean projection for each direction
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        print(f"Calibrating subspace ({subspace.n_components} components) on {len(calibration_texts)} samples...")
        # subspace.vectors has shape [hidden_dim, k]
        direction_matrix = subspace.vectors.to(self.device)
        all_projections = []
        batch_size = self.config.calibration_batch_size

        for batch_start in tqdm(range(0, len(calibration_texts), batch_size), desc="Calibrating subspace"):
            batch_texts = calibration_texts[batch_start : batch_start + batch_size]
            with torch.no_grad():
                with self.model.trace(batch_texts):
                    activations = self._get_activations_accessor(subspace.layer_idx).save()
                    _ = self.model.output

            batch_acts = self._extract_at_position(activations)
            # batch_acts: [B, H], direction_matrix: [H, k]
            # projections: [B, k] - projection onto each of the k directions
            projections = torch.matmul(batch_acts.float(), direction_matrix.float())
            all_projections.append(projections.cpu())

        all_projections_tensor = torch.cat(all_projections, dim=0)  # [N, k]
        mean_projs = all_projections_tensor.mean(dim=0)  # [k]
        stds = all_projections_tensor.std(dim=0)

        print(f"Mean projections per component: {mean_projs.tolist()}")
        print(f"Std per component: {stds.tolist()}")
        return mean_projs

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

    def get_logprobs_subspace(
        self,
        text: str,
        subspace: Optional[SubspaceDirections] = None,
        mean_projs: Optional[torch.Tensor] = None,
    ) -> float:
        """Compute log probability of text, optionally with multi-direction subspace ablation.

        Multi-direction ablation formula:
            x_new = x - V @ V.T @ x + V @ μ
        Which can be rewritten as:
            proj = x @ V           # [B, S, k] - project onto each direction
            delta = μ - proj       # [B, S, k] - difference from mean for each
            x_new = x + delta @ V.T  # [B, S, H] - apply correction

        Args:
            text: Input text to evaluate
            subspace: Optional SubspaceDirections for ablation
            mean_projs: Tensor of shape [k] with mean projection per direction

        Returns:
            Total log probability of the text
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        if subspace is not None and mean_projs is None:
            raise ValueError("mean_projs required when subspace is provided")

        with torch.no_grad():
            with self.model.trace(text):
                if subspace is not None:
                    x = self._get_activations_accessor(subspace.layer_idx)
                    # Match direction vectors dtype to activation dtype
                    # subspace.vectors: [H, k]
                    V = subspace.vectors.to(device=self.device, dtype=x.dtype)
                    mu = mean_projs.to(device=self.device, dtype=x.dtype)  # [k]

                    # x has shape [B, S, H], V has shape [H, k]
                    # Project onto each direction: [B, S, k]
                    current_proj = torch.matmul(x, V)

                    # Compute delta from mean for each direction: [k] - [B, S, k] -> [B, S, k]
                    delta = mu - current_proj

                    # Apply correction in original space: [B, S, k] @ [k, H] -> [B, S, H]
                    ablation_delta = torch.matmul(delta, V.T)
                    self._set_activations(subspace.layer_idx, x + ablation_delta)

                logits = self.model.output.logits.save()

        logits_val = logits
        tokens = self.model.tokenizer(text, return_tensors="pt")["input_ids"].to(self.device)
        log_probs = torch.nn.functional.log_softmax(logits_val, dim=-1)

        return sum(log_probs[0, i - 1, tokens[0, i]].item() for i in range(1, tokens.shape[1]))

    def evaluate_minimal_pairs_subspace(
        self,
        pairs: List[Tuple[str, str]],
        subspace: Optional[SubspaceDirections] = None,
        mean_projs: Optional[torch.Tensor] = None,
        desc: str = "Evaluating",
    ) -> Dict[str, Any]:
        """Evaluate on minimal pairs with multi-direction subspace ablation.

        Args:
            pairs: List of (good_sentence, bad_sentence) tuples
            subspace: Optional SubspaceDirections for ablation
            mean_projs: Tensor of shape [k] with mean projection per direction

        Returns:
            Dictionary with accuracy, correct count, and detailed results
        """
        results = []
        for good_sent, bad_sent in tqdm(pairs, desc=desc):
            good_lp = self.get_logprobs_subspace(good_sent, subspace, mean_projs)
            bad_lp = self.get_logprobs_subspace(bad_sent, subspace, mean_projs)
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
            "ablation_applied": subspace is not None,
            "n_components": subspace.n_components if subspace else 0,
        }

    def run_ablation_experiment(
        self,
        direction: SubspaceDirection,
        calibration_texts: List[str],
        eval_pairs: List[Tuple[str, str]],
        baseline: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run complete ablation experiment with baseline.
        
        Args:
            direction: The direction to ablate
            calibration_texts: Texts for computing mean projection
            eval_pairs: List of (good_sentence, bad_sentence) for evaluation
            baseline: Optional pre-computed baseline results to avoid redundant computation
        """
        results = {
            "direction_name": direction.name,
            "layer_idx": direction.layer_idx,
            "discovery_method": direction.discovery_method,
            "direction_metadata": direction.metadata,
        }

        # Baseline - use provided or compute
        if baseline is None:
            print("\n=== Baseline (No Ablation) ===")
            baseline = self.evaluate_minimal_pairs(eval_pairs, desc="Baseline")
            print(f"Baseline accuracy: {baseline['accuracy']:.2%}")
        results["baseline"] = baseline

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

    def run_ablation_experiment_subspace(
        self,
        subspace: SubspaceDirections,
        calibration_texts: List[str],
        eval_pairs: List[Tuple[str, str]],
        baseline: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run complete multi-direction subspace ablation experiment with baseline.

        Args:
            subspace: SubspaceDirections containing k orthonormal directions
            calibration_texts: Texts for computing mean projections
            eval_pairs: List of (good_sentence, bad_sentence) for evaluation
            baseline: Optional pre-computed baseline results to avoid redundant computation

        Returns:
            Dictionary with baseline, ablation results, and effect metrics
        """
        results = {
            "subspace_name": subspace.name,
            "layer_idx": subspace.layer_idx,
            "discovery_method": subspace.discovery_method,
            "n_components": subspace.n_components,
            "subspace_metadata": subspace.metadata,
        }

        # Baseline - use provided or compute
        if baseline is None:
            print("\n=== Baseline (No Ablation) ===")
            baseline = self.evaluate_minimal_pairs_subspace(eval_pairs, desc="Baseline")
            print(f"Baseline accuracy: {baseline['accuracy']:.2%}")
        results["baseline"] = baseline

        # Calibrate subspace
        print(f"\n=== Calibrating Subspace ({subspace.n_components} components) ===")
        mean_projs = self.calibrate_subspace(subspace, calibration_texts)
        results["mean_projections"] = mean_projs.tolist()

        # Ablation evaluation
        print("\n=== Subspace Ablation Evaluation ===")
        ablation = self.evaluate_minimal_pairs_subspace(
            eval_pairs, subspace, mean_projs, desc=f"Ablated (k={subspace.n_components})"
        )
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
    """Create visualization for ablation effect by layer.

    Args:
        results: List of result dictionaries, each containing:
            - layer: int
            - diff_means: dict with ablation_effect
            - pca_on_diff_subspace: dict with ablation_effect and n_components
        results_dir: Directory to save plots
        model_name: Name of the model (for plot titles and filenames)
    """
    import matplotlib.pyplot as plt

    # Extract data for both methods
    layers = [r["layer"] for r in results]
    dm = [r["diff_means"] for r in results]
    pca_subspace = [r["pca_on_diff_subspace"] for r in results]

    # Ablation effect data
    dm_ablation_effect = [d["ablation_effect"] for d in dm]
    pca_subspace_ablation_effect = [p["ablation_effect"] for p in pca_subspace]

    # Get n_components for label (should be same across layers)
    n_components = pca_subspace[0].get("n_components", "k")

    # Clean model name for filename
    clean_name = model_name.replace("/", "_").replace("-", "_")

    # Create single figure showing ablation effect by layer
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(
        layers,
        dm_ablation_effect,
        marker="o",
        linewidth=2,
        markersize=8,
        color="#2E86AB",
        label="Diff-Means (1D)",
    )
    ax.plot(
        layers,
        pca_subspace_ablation_effect,
        marker="s",
        linewidth=2,
        markersize=8,
        color="#E94F37",
        label=f"PCA-on-Diff Subspace (k={n_components})",
    )
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.fill_between(layers, dm_ablation_effect, 0, alpha=0.2, color="#2E86AB")
    ax.fill_between(layers, pca_subspace_ablation_effect, 0, alpha=0.2, color="#E94F37")
    ax.set_xlabel("Layer", fontsize=12, fontweight="bold")
    ax.set_ylabel("Ablation Effect (Δ Accuracy)", fontsize=12, fontweight="bold")
    ax.set_title(f"Ablation Effect by Layer: {model_name}", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    plt.tight_layout()

    # Save plot
    plot_path = results_dir / f"ablation_effect_{clean_name}.pdf"
    plt.savefig(plot_path, bbox_inches="tight")
    print(f"Saved visualization to {plot_path}")
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
        "EleutherAI/pythia-70m-deduped": 6,
        "EleutherAI/pythia-1b-deduped": 16,
        "EleutherAI/pythia-2.8b-deduped": 32,
        "EleutherAI/pythia-6.9b-deduped": 32,
        "allenai/OLMo-1B-hf": 16,
        "allenai/OLMo-7B-hf": 32,
        "google/gemma-2-2b": 26,
        "meta-llama/Meta-Llama-3-8B": 32,
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
    if len(pairs) < 700:
        raise ValueError(
            f"Insufficient pairs in BLiMP subset '{blimp_subset}': {len(pairs)} < 700. "
            "Need at least 100 for discovery, 50 for calibration, and 550 for evaluation."
        )

    discovery_pairs = pairs[:100]
    calibration_pairs = pairs[100:150]
    eval_pairs = pairs[150:700]  # 550 samples - can detect ~3.5% effects reliably

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
    import os
    import shutil
    import time

    # Set HF cache to local directory to avoid polluting shared cluster storage
    # and ensure we can actually clear it
    HF_CACHE_DIR = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"
    os.environ["HF_HOME"] = str(HF_CACHE_DIR.parent)
    os.environ["TRANSFORMERS_CACHE"] = str(HF_CACHE_DIR)

    def aggressive_cuda_cleanup():
        """Aggressively clean up CUDA resources to prevent device busy errors."""
        if torch.cuda.is_available():
            # Synchronize all CUDA streams
            torch.cuda.synchronize()
            # Empty the cache
            torch.cuda.empty_cache()
            # Reset peak memory stats
            torch.cuda.reset_peak_memory_stats()
            # Force garbage collection
            gc.collect()
            # Empty cache again after gc
            torch.cuda.empty_cache()
            # Small delay to allow GPU to fully release resources
            time.sleep(2)

    def clear_model_cache():
        """Clear HuggingFace cache to free disk space."""
        cache_dirs = [
            HF_CACHE_DIR,
            Path.home() / ".cache/huggingface/hub",
            Path("/workspace/.cache/huggingface/hub"),
        ]
        # Also check environment variables
        for env_var in ["HF_HOME", "TRANSFORMERS_CACHE", "HUGGINGFACE_HUB_CACHE"]:
            if env_var in os.environ:
                env_path = Path(os.environ[env_var])
                if env_path.name != "hub":
                    env_path = env_path / "hub"
                cache_dirs.append(env_path)

        # Deduplicate paths
        cache_dirs = list(set(cache_dirs))

        for cache_dir in cache_dirs:
            if cache_dir.exists():
                for item in cache_dir.glob("models--*"):
                    try:
                        shutil.rmtree(item)
                        print(f"Cleared cache: {item.name}")
                    except Exception as e:
                        print(f"Failed to clear {item}: {e}")

    failed_models = []

    MAX_LOAD_RETRIES = 3

    for model_name, num_layers in model_layers.items():
        print(f"\n\n{'#' * 100}")
        print(f"RUNNING EXPERIMENT FOR MODEL: {model_name} with {num_layers} layers")
        print(f"{'#' * 100}\n")

        ablator = None
        for attempt in range(MAX_LOAD_RETRIES):
            try:
                # Create ablator once (we'll reuse it across layers)
                config = AblationConfig(
                    model_name=model_name,
                    layer_idx=0,
                    target_component="residual",
                    calibration_batch_size=4,
                )

                ablator = MeanSubspaceAblation(config)
                ablator.load_model()
                break  # Success, exit retry loop
            except Exception as e:
                print(f"❌ FAILED to load model {model_name} (attempt {attempt + 1}/{MAX_LOAD_RETRIES}): {e}")
                gc.collect()
                aggressive_cuda_cleanup()
                
                if attempt < MAX_LOAD_RETRIES - 1:
                    wait_time = (attempt + 1) * 5  # Exponential backoff: 5s, 10s, 15s
                    print(f"Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                else:
                    print("Max retries reached. Skipping to next model...")
                    failed_models.append({"model": model_name, "error": str(e), "stage": "loading"})
                    clear_model_cache()

        if ablator is None or ablator.model is None:
            continue  # Skip to next model

        try:
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

                # Compute baseline once (no ablation) - shared across all methods
                print("\n=== Baseline (No Ablation) ===")
                baseline = ablator.evaluate_minimal_pairs(eval_pairs, desc="Baseline")
                print(f"Baseline accuracy: {baseline['accuracy']:.2%}")

                print("\n--- Diff-Means Direction ---")
                diff_direction = DirectionDiscovery.from_diff_means(
                    pos_acts, neg_acts, layer, name=f"layer{layer}_diffmeans"
                )

                diff_results = ablator.run_ablation_experiment(
                    diff_direction.to(config.device),
                    calibration_texts,
                    eval_pairs,
                    baseline=baseline,
                )

                print("\n--- PCA on Differences Subspace (Multi-Direction) ---")
                pca_diff_subspace = DirectionDiscovery.from_pca_on_diff_subspace(
                    pos_acts, neg_acts, layer, n_components=5, name=f"layer{layer}_pca_diff_subspace"
                )

                pca_subspace_results = ablator.run_ablation_experiment_subspace(
                    pca_diff_subspace.to(config.device),
                    calibration_texts,
                    eval_pairs,
                    baseline=baseline,
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
                        "pca_on_diff_subspace": {
                            "subspace_name": pca_subspace_results["subspace_name"],
                            "discovery_method": pca_subspace_results["discovery_method"],
                            "n_components": pca_subspace_results["n_components"],
                            "baseline_accuracy": pca_subspace_results["baseline"]["accuracy"],
                            "ablation_accuracy": pca_subspace_results["ablation"]["accuracy"],
                            "ablation_effect": pca_subspace_results["ablation_effect"],
                            "mean_projections": pca_subspace_results["mean_projections"],
                            "metadata": pca_subspace_results["subspace_metadata"],
                        },
                    }
                )

                # Print layer summary
                print(f"\n{'=' * 80}")
                print(f"LAYER {layer} SUMMARY:")
                print(f"  Diff-Means (1D): {diff_results['ablation_effect']:+.2%} effect")
                print(f"  PCA-on-Diff Subspace (k={pca_subspace_results['n_components']}): {pca_subspace_results['ablation_effect']:+.2%} effect")
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
                        "n_components",
                        "baseline_acc",
                        "ablation_acc",
                        "effect",
                        "cumulative_variance",
                    ]
                )

                for result in model_results:
                    # Diff-means row
                    dm = result["diff_means"]
                    writer.writerow(
                        [
                            result["model"],
                            result["layer"],
                            "diff_means",
                            1,  # Single direction
                            f"{dm['baseline_accuracy']:.4f}",
                            f"{dm['ablation_accuracy']:.4f}",
                            f"{dm['ablation_effect']:.4f}",
                            "N/A",
                        ]
                    )
                    # PCA-on-diff subspace row
                    pca = result["pca_on_diff_subspace"]
                    cumulative_var = pca["metadata"].get("cumulative_explained_variance", "N/A")
                    writer.writerow(
                        [
                            result["model"],
                            result["layer"],
                            "pca_on_diff_subspace",
                            pca["n_components"],
                            f"{pca['baseline_accuracy']:.4f}",
                            f"{pca['ablation_accuracy']:.4f}",
                            f"{pca['ablation_effect']:.4f}",
                            f"{cumulative_var:.4f}" if isinstance(cumulative_var, float) else cumulative_var,
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

            # Cleanup model after successful experiment to free GPU memory
            del ablator.model
            del ablator
            gc.collect()
            aggressive_cuda_cleanup()
            clear_model_cache()

        except Exception as e:
            print(f"❌ FAILED during experiment for {model_name}: {e}")
            print("Saving partial results and continuing to next model...")
            failed_models.append({"model": model_name, "error": str(e), "stage": "experiment"})
            # Still try to clean up
            try:
                del ablator.model
                del ablator
            except Exception:
                pass
            gc.collect()
            aggressive_cuda_cleanup()
            clear_model_cache()

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
                "n_components",
                "baseline_acc",
                "ablation_acc",
                "effect",
                "cumulative_variance",
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
                    1,  # Single direction
                    f"{dm['baseline_accuracy']:.4f}",
                    f"{dm['ablation_accuracy']:.4f}",
                    f"{dm['ablation_effect']:.4f}",
                    "N/A",
                ]
            )
            # PCA-on-diff subspace row
            pca = result["pca_on_diff_subspace"]
            cumulative_var = pca["metadata"].get("cumulative_explained_variance", "N/A")
            writer.writerow(
                [
                    model,
                    layer,
                    "pca_on_diff_subspace",
                    pca["n_components"],
                    f"{pca['baseline_accuracy']:.4f}",
                    f"{pca['ablation_accuracy']:.4f}",
                    f"{pca['ablation_effect']:.4f}",
                    f"{cumulative_var:.4f}" if isinstance(cumulative_var, float) else cumulative_var,
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
                "failed_models": failed_models,
            },
            f,
            indent=2,
        )

    # Print final summary
    print(f"\n{'=' * 80}")
    print("EXPERIMENT COMPLETE")
    print(f"{'=' * 80}")
    print(f"Results saved to: {output_dir}")
    print(f"Models processed successfully: {len(model_layers) - len(failed_models)}/{len(model_layers)}")
    if failed_models:
        print(f"\nFailed models ({len(failed_models)}):")
        for fm in failed_models:
            print(f"  - {fm['model']}: {fm['error'][:100]}...")
