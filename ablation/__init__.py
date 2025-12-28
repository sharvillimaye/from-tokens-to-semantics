"""
Ablation Module for Causal Intervention in Language Models

This module provides tools for testing causal hypotheses about neural network
representations using mean subspace ablation techniques.

Main Components:
================

1. MeanSubspaceAblation
   - Core class for running ablation experiments
   - Handles model loading, activation extraction, and intervention

2. DirectionDiscovery
   - Methods for finding meaningful directions in activation space
   - Supports diff-means, PCA, and custom directions

3. SubspaceDirection
   - Data class representing a single direction to ablate
   - Includes metadata about how the direction was discovered

4. SubspaceDirections
   - Data class for multi-direction ablation (top-k PCA components)
   - Enables simultaneous ablation of multiple orthonormal directions

5. PolytopeDirectionDiscovery
   - Discovers directions from polytope analysis data
   - Integrates with checkpoint_analysis.py records

6. PolytopeAblationExperiment
   - High-level experiment runner for polytope-based ablation
   - Compares effects across layers and checkpoints

Usage Example (Single Direction):
=================================

    from ablation import MeanSubspaceAblation, AblationConfig, DirectionDiscovery

    # Setup
    config = AblationConfig(model_name="allenai/OLMo-1B-hf", layer_idx=8)
    ablator = MeanSubspaceAblation(config)
    ablator.load_model()

    # Discover direction from activations
    pos_acts, neg_acts = ablator.extract_paired_activations(good_texts, bad_texts)
    direction = DirectionDiscovery.from_diff_means(pos_acts, neg_acts, layer_idx=8)

    # Calibrate and run experiment
    mean_proj = ablator.calibrate(direction, calibration_texts)
    results = ablator.evaluate_minimal_pairs(eval_pairs, direction, mean_proj)

Usage Example (Multi-Direction Subspace):
=========================================

    from ablation import MeanSubspaceAblation, AblationConfig, DirectionDiscovery

    # Setup
    config = AblationConfig(model_name="allenai/OLMo-1B-hf", layer_idx=8)
    ablator = MeanSubspaceAblation(config)
    ablator.load_model()

    # Discover subspace from activations (top-5 PCA components)
    pos_acts, neg_acts = ablator.extract_paired_activations(good_texts, bad_texts)
    subspace = DirectionDiscovery.from_pca_on_diff_subspace(
        pos_acts, neg_acts, layer_idx=8, n_components=5
    )

    # Calibrate and run experiment
    results = ablator.run_ablation_experiment_subspace(subspace, calibration_texts, eval_pairs)
"""

from .mean_subspace_ablation import (
    AblationConfig,
    DirectionDiscovery,
    MeanSubspaceAblation,
    SubspaceDirection,
    SubspaceDirections,
    load_blimp_minimal_pairs,
)

__all__ = [
    # Core ablation
    "MeanSubspaceAblation",
    "AblationConfig",
    "SubspaceDirection",
    "SubspaceDirections",
    "DirectionDiscovery",
    "load_blimp_minimal_pairs",
]
