#!/usr/bin/env python3
"""
Comprehensive Polytope Superposition Analyzer for LLM Activations

This module implements a robust four-phase analysis pipeline for studying superposition
phenomena in LLM activations, specifically designed to compare high vs low frequency
n-gram patterns and understand geometric structure differences.

PIPELINE PHASES:
===============

Phase 1: Preprocessing with Binary Pattern Extraction
- Converts activation vectors to binary patterns using thresholding
- Computes sparsity metrics and unique pattern analysis
- Handles high-dimensional data with optional sampling

Phase 2: Polytope Boundary Analysis with Density Calculations  
- Calculates polytope densities using Hamming vs Euclidean distance ratios
- Identifies boundary crossings and geometric transitions
- Uses efficient pairwise analysis with configurable limits

Phase 3: Superposition Detection and Measurement
- Measures vector overlaps and interference patterns
- Calculates superposition strength and phase classification
- Distinguishes between strong/weak superposition regimes

Phase 4: Advanced Clustering Analysis with HDBSCAN
- Applies semantic clustering to both binary patterns and activations
- Measures cluster consistency between pattern and activation spaces
- Includes t-SNE visualization and quality metrics

DEPENDENCIES:
============
pip install numpy scipy scikit-learn hdbscan pandas matplotlib seaborn

USAGE:
======
from polytope.polytope_analyzer import SuperpositionAnalyzer

# Initialize analyzer
analyzer = SuperpositionAnalyzer(min_cluster_size=5, random_seed=42)

# Run analysis comparing high vs low frequency activations
results = analyzer.analyze_polytope_superposition(
    activations_high_freq=high_freq_data,  # Shape: (n_samples_high, n_neurons)
    activations_low_freq=low_freq_data,    # Shape: (n_samples_low, n_neurons)
    semantic_category="n_grams"
)

# Generate comprehensive report
report = analyzer.generate_analysis_report(results)
print(report)

# Access detailed metrics
print(f"Superposition difference: {results['comparison']['superposition_strength_difference']}")
print(f"Phase transition: {results['comparison']['phase_transition']}")

COMPUTATIONAL EFFICIENCY:
========================
- Automatic sampling for large datasets (>10K pairs in polytope analysis)
- Sparse matrix operations for high-dimensional activations (>1K neurons)
- Configurable clustering parameters based on dataset size
- Memory-efficient pairwise distance calculations

STATISTICAL ROBUSTNESS:
======================
- Bootstrap confidence intervals for key metrics
- Effect size calculations (Cohen's d approximation)
- Multiple comparison handling
- Robust clustering with noise detection

EXPECTED OUTPUTS:
================
- Quantitative superposition strength differences between frequency groups
- Geometric polytope structure characterization
- Phase classification (strong vs weak superposition)
- Statistical validation with confidence intervals
- Mechanistic insights into n-gram frequency effects on neural representations

Original Polytope Analysis Pipeline Components:
- CETT-based spline code generation
- Advanced boundary detection algorithms  
- MASO framework integration
- Safety polytope management
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict
from pathlib import Path
import warnings
from loguru import logger
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Required imports for advanced polytope analysis
from hdbscan import HDBSCAN
from scipy.spatial.distance import cdist
import scipy.sparse as sparse
try:  # Optional dependency for dimensionality reduction before clustering
    from umap import UMAP  # type: ignore
    UMAP_AVAILABLE = True
except Exception:  # pragma: no cover - environment may not have umap
    UMAP_AVAILABLE = False

# Optional imports for visualization and timestamps
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    plt = None

# Do not silence warnings globally; suppress only noisy sklearn alias warning
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module=r"sklearn\.utils\.deprecation",
    message=r".*'force_all_finite' was renamed to 'ensure_all_finite'.*",
)

class SuperpositionAnalyzer:
    """
    Comprehensive superposition analysis pipeline for LLM activations based on n-gram frequency.
    
    Implements four-phase analysis:
    1. Preprocessing with binary pattern extraction
    2. Polytope boundary analysis with density calculations  
    3. Superposition detection and measurement
    4. Advanced clustering analysis with semantic region identification
    
    Usage:
        analyzer = SuperpositionAnalyzer()
        results = analyzer.analyze_polytope_superposition(high_freq_activations, low_freq_activations)
    """
    
    def __init__(self, min_cluster_size: int = 5, random_seed: Optional[int] = 42):
        """
        Initialize superposition analyzer.
        
        Args:
            min_cluster_size: Minimum cluster size for HDBSCAN
            random_seed: Random seed for reproducibility
        """
        self.min_cluster_size = min_cluster_size
        self.random_seed = random_seed
        if random_seed is not None:
            np.random.seed(random_seed)
    

    def compute_polytope_metrics(self, activations: np.ndarray, binary_patterns: np.ndarray, 
                                max_pairs: int = 10000) -> Dict[str, Any]:
        """
        Phase 1: Polytope boundary analysis with density calculations.
        
        Args:
            activations: Activation vectors
            binary_patterns: Binary polytope patterns
            max_pairs: Maximum pairs to compute for efficiency
            
        Returns:
            Dictionary with polytope metrics
        """
        n_samples = len(activations)
        if n_samples < 2:
            raise ValueError("Need at least two samples to compute polytope metrics")

        # Build unique random pairs up to max_pairs (or all pairs if fewer)
        total_pairs = n_samples * (n_samples - 1) // 2
        if total_pairs <= max_pairs:
            pairs = np.array([(i, j) for i in range(n_samples) for j in range(i + 1, n_samples)], dtype=np.int64)
        else:
            # Sample without replacement from the pair index space by generating candidate pairs and deduplicating
            rng = np.random.default_rng(self.random_seed)
            target = max_pairs
            pairs_set = set()
            # Over-generate to reduce loops
            while len(pairs_set) < target:
                i = rng.integers(0, n_samples, size=target * 2)
                j = rng.integers(0, n_samples, size=target * 2)
                mask = i < j
                cand = np.stack([i[mask], j[mask]], axis=1)
                for a, b in cand:
                    if len(pairs_set) >= target:
                        break
                    pairs_set.add((int(a), int(b)))
            pairs = np.array(list(pairs_set), dtype=np.int64)

        # Vectorized distances for sampled pairs
        diff = activations[pairs[:, 0]] - activations[pairs[:, 1]]
        euclidean_distances = np.linalg.norm(diff, axis=1)

        # Hamming distances on binary patterns
        xor = binary_patterns[pairs[:, 0]] != binary_patterns[pairs[:, 1]]
        hamming_distances = np.sum(xor, axis=1)

        # Density and boundary crossings
        with np.errstate(divide='ignore', invalid='ignore'):
            densities = np.where(euclidean_distances > 1e-12, hamming_distances / euclidean_distances, np.nan)
        valid_mask = ~np.isnan(densities) & np.isfinite(densities)
        densities = densities[valid_mask]
        euc_valid = euclidean_distances[valid_mask]
        ham_valid = hamming_distances[valid_mask]

        if densities.size == 0:
            raise ValueError("No valid polytope density calculations possible - check input data")
        
        boundary_threshold = np.percentile(euc_valid, 25)
        boundary_crossings = int(np.sum((ham_valid > 0) & (euc_valid < boundary_threshold)))
        
        return {
            'polytope_densities': densities,
            'mean_density': float(np.mean(densities)),
            'density_std': float(np.std(densities)),
            'boundary_crossings': boundary_crossings,
            'boundary_crossing_rate': float(boundary_crossings) / float(len(euc_valid)),
            'density_percentiles': {
                '25th': float(np.percentile(densities, 25)),
                '75th': float(np.percentile(densities, 75)),
                '90th': float(np.percentile(densities, 90))
            }
        }
    
    def measure_superposition(self, activations: np.ndarray, binary_patterns: np.ndarray,
                              max_neuron_pairs: int = 10000) -> Dict[str, Any]:
        """
        Phase 2: Superposition detection and measurement.
        
        Args:
            activations: Activation vectors
            binary_patterns: Binary polytope patterns
            
        Returns:
            Dictionary with superposition metrics
        """
        # Compute participation ratio (effective dimension) from PCA spectrum
        n_samples, layer_width = activations.shape
        if n_samples < 2:
            raise ValueError("Need at least two samples to measure superposition")

        # Standardize features for stable PCA
        scaler = StandardScaler(with_mean=True, with_std=True)
        X = scaler.fit_transform(activations)

        # PCA components must be <= min(n_samples, n_features)
        max_allowed_components = max(1, min(n_samples, layer_width) - 1)
        n_components = int(min(256, max_allowed_components))
        pca = PCA(n_components=n_components, random_state=self.random_seed)
        pca.fit(X)
        eigvals = pca.explained_variance_
        eigvals = eigvals[eigvals > 0]
        pr = float((np.sum(eigvals) ** 2) / np.sum(eigvals ** 2)) if eigvals.size > 0 else 0.0

        # Sample neuron-neuron overlaps to avoid O(d^2)
        activation_matrix = activations.T  # neurons x samples
        neuron_count = activation_matrix.shape[0]
        if neuron_count < 2:
            raise ValueError("Layer width must be >= 2 to compute overlaps")

        total_pairs = neuron_count * (neuron_count - 1) // 2
        target_pairs = min(max_neuron_pairs, total_pairs)
        rng = np.random.default_rng(self.random_seed)
        if total_pairs <= target_pairs:
            neuron_pairs = np.array([(i, j) for i in range(neuron_count) for j in range(i + 1, neuron_count)], dtype=np.int64)
        else:
            pairs_set = set()
            while len(pairs_set) < target_pairs:
                i = rng.integers(0, neuron_count, size=target_pairs * 2)
                j = rng.integers(0, neuron_count, size=target_pairs * 2)
                mask = i < j
                cand = np.stack([i[mask], j[mask]], axis=1)
                for a, b in cand:
                    if len(pairs_set) >= target_pairs:
                        break
                    pairs_set.add((int(a), int(b)))
            neuron_pairs = np.array(list(pairs_set), dtype=np.int64)

        norms = np.linalg.norm(activation_matrix, axis=1)
        norms[norms == 0] = 1e-12
        v0 = activation_matrix[neuron_pairs[:, 0]]
        v1 = activation_matrix[neuron_pairs[:, 1]]
        dots = np.sum(v0 * v1, axis=1)
        denom = norms[neuron_pairs[:, 0]] * norms[neuron_pairs[:, 1]]
        cos_sim = np.clip(dots / denom, -1.0, 1.0)
        interference_scores = cos_sim
        mean_sq_interference = float(np.mean(interference_scores ** 2))

        # Unique pattern count and compression
        n_unique_patterns = int(len(np.unique(binary_patterns, axis=0)))
        compression_ratio = float(n_unique_patterns) / float(layer_width)

        expected_strong_interference = 1.0 / float(layer_width)
        phase_classification = "strong" if mean_sq_interference > expected_strong_interference else "weak"
        
        return {
            'participation_ratio': pr,
            'effective_dimension': pr,
            'mean_squared_overlap_sampled': mean_sq_interference,
            'interference_per_dimension': mean_sq_interference,  # keep key for downstream comparators
            'compression_ratio': compression_ratio,
            'phase_classification': phase_classification,
            'n_unique_patterns': n_unique_patterns,
            'interference_stats': {
                'mean': float(np.mean(interference_scores)),
                'std': float(np.std(interference_scores)),
                'max': float(np.max(interference_scores)),
                'min': float(np.min(interference_scores)),
                'num_pairs': int(len(interference_scores))
            }
        }
    
    def clustering_analysis(self, binary_patterns: np.ndarray, activations: np.ndarray) -> Dict[str, Any]:
        """
        Phase 4: Advanced clustering analysis with HDBSCAN.
        
        Args:
            binary_patterns: Binary polytope patterns
            activations: Activation vectors
            
        Returns:
            Dictionary with clustering results
        """
        # Dimensionality reduction for high-dimensional inputs (UMAP) before clustering
        bin_for_cluster = binary_patterns
        act_for_cluster = activations
        # Use UMAP only when dimensionality is high and sample size supports it; otherwise PCA fallback
        n_samples_bin = len(binary_patterns)
        n_samples_act = len(activations)
        
        # For binary patterns: use UMAP only with sufficient samples and dimensions
        if binary_patterns.shape[1] > 100 and n_samples_bin > 15:
            if UMAP_AVAILABLE:
                try:
                    # More conservative UMAP settings for small sample sizes
                    n_components = min(20, max(2, min(binary_patterns.shape[1] // 20, n_samples_bin // 2)))
                    n_neighbors = min(5, max(2, n_samples_bin // 4))
                    
                    reducer_bin = UMAP(
                        n_components=n_components,
                        n_neighbors=n_neighbors,
                        metric='hamming',
                        random_state=self.random_seed,
                        low_memory=True,
                        verbose=False
                    )
                    bin_for_cluster = reducer_bin.fit_transform(binary_patterns)
                except Exception as e:
                    logger.warning(f"UMAP failed for binary patterns: {e}, falling back to PCA")
                    # PCA fallback for small N or UMAP issues
                    n_comp = max(2, min(20, n_samples_bin - 1, binary_patterns.shape[1] - 1))
                    pca_bin = PCA(n_components=n_comp, random_state=self.random_seed)
                    bin_for_cluster = pca_bin.fit_transform(binary_patterns.astype(float))
            else:
                # If UMAP not available, try PCA reduction
                try:
                    n_comp = max(2, min(20, n_samples_bin - 1, binary_patterns.shape[1] - 1))
                    pca_bin = PCA(n_components=n_comp, random_state=self.random_seed)
                    bin_for_cluster = pca_bin.fit_transform(binary_patterns.astype(float))
                except Exception:
                    bin_for_cluster = binary_patterns
        elif binary_patterns.shape[1] > 50:  # Medium dimensionality: try PCA
            try:
                n_comp = max(2, min(20, n_samples_bin - 1, binary_patterns.shape[1] - 1))
                pca_bin = PCA(n_components=n_comp, random_state=self.random_seed)
                bin_for_cluster = pca_bin.fit_transform(binary_patterns.astype(float))
            except Exception:
                bin_for_cluster = binary_patterns

        # For activation vectors: use UMAP only with sufficient samples and dimensions
        if activations.shape[1] > 100 and n_samples_act > 15:
            if UMAP_AVAILABLE:
                try:
                    # More conservative UMAP settings for small sample sizes
                    n_components = min(20, max(2, min(activations.shape[1] // 20, n_samples_act // 2)))
                    n_neighbors = min(5, max(2, n_samples_act // 4))
                    
                    reducer_act = UMAP(
                        n_components=n_components,
                        n_neighbors=n_neighbors,
                        metric='euclidean',
                        random_state=self.random_seed,
                        low_memory=True,
                        verbose=False
                    )
                    act_for_cluster = reducer_act.fit_transform(activations)
                except Exception as e:
                    logger.warning(f"UMAP failed for activations: {e}, falling back to PCA")
                    n_comp = max(2, min(20, n_samples_act - 1, activations.shape[1] - 1))
                    pca_act = PCA(n_components=n_comp, random_state=self.random_seed)
                    act_for_cluster = pca_act.fit_transform(activations)
            else:
                try:
                    n_comp = max(2, min(20, n_samples_act - 1, activations.shape[1] - 1))
                    pca_act = PCA(n_components=n_comp, random_state=self.random_seed)
                    act_for_cluster = pca_act.fit_transform(activations)
                except Exception:
                    act_for_cluster = activations
        elif activations.shape[1] > 50:  # Medium dimensionality: try PCA
            try:
                n_comp = max(2, min(20, n_samples_act - 1, activations.shape[1] - 1))
                pca_act = PCA(n_components=n_comp, random_state=self.random_seed)
                act_for_cluster = pca_act.fit_transform(activations)
            except Exception:
                act_for_cluster = activations

        # HDBSCAN on reduced (or original) spaces
        clusterer_binary = HDBSCAN(
            min_cluster_size=max(2, self.min_cluster_size, int(np.sqrt(len(bin_for_cluster))//2)),
            metric='euclidean',
            cluster_selection_method='eom'
        )
        binary_clusters = clusterer_binary.fit_predict(bin_for_cluster)
        
        # HDBSCAN on activation vectors
        clusterer_activations = HDBSCAN(
            min_cluster_size=max(2, self.min_cluster_size, int(np.sqrt(len(act_for_cluster))//2)),
            metric='euclidean',
            cluster_selection_method='eom'
        )
        activation_clusters = clusterer_activations.fit_predict(act_for_cluster)
        
        # Compare cluster consistency
        cluster_consistency = adjusted_rand_score(binary_clusters, activation_clusters)
        
        # Dimensionality reduction for visualization
        if len(binary_patterns) < 10:
            binary_embedding = None
        else:
            # Create distance matrix for binary patterns
            max_tsne_points = 2000
            if len(binary_patterns) > max_tsne_points:
                # Sample for TSNE to avoid O(N^2) memory/time
                rng = np.random.default_rng(self.random_seed)
                idx = rng.choice(len(binary_patterns), size=max_tsne_points, replace=False)
                bp_tsne = binary_patterns[idx]
            else:
                bp_tsne = binary_patterns

            binary_distances = squareform(pdist(bp_tsne, metric='hamming'))
            perplexity = min(30, max(5, len(bp_tsne)//4))
            tsne = TSNE(n_components=2, random_state=self.random_seed, metric='precomputed', 
                       perplexity=perplexity, init='random')
            binary_embedding = tsne.fit_transform(binary_distances)
        
        # Cluster quality metrics
        n_clusters_binary = len(set(binary_clusters)) - (1 if -1 in binary_clusters else 0)
        n_clusters_activation = len(set(activation_clusters)) - (1 if -1 in activation_clusters else 0)
        
        # Silhouette analysis
        from sklearn.metrics import silhouette_score
        # Filter out noise labels (-1)
        valid_bin = binary_clusters != -1
        if n_clusters_binary > 1 and np.sum(valid_bin) > 1 and len(np.unique(binary_clusters[valid_bin])) > 1:
            silhouette_binary = silhouette_score(binary_patterns[valid_bin], binary_clusters[valid_bin], metric='hamming')
        else:
            silhouette_binary = None
            
        valid_act = activation_clusters != -1
        if n_clusters_activation > 1 and np.sum(valid_act) > 1 and len(np.unique(activation_clusters[valid_act])) > 1:
            silhouette_activation = silhouette_score(activations[valid_act], activation_clusters[valid_act])
        else:
            silhouette_activation = None
        
        return {
            'binary_clusters': binary_clusters,
            'activation_clusters': activation_clusters,
            'cluster_consistency': cluster_consistency,
            'binary_embedding': binary_embedding,
            'n_clusters_binary': n_clusters_binary,
            'n_clusters_activation': n_clusters_activation,
            'silhouette_binary': silhouette_binary,
            'silhouette_activation': silhouette_activation,
            'noise_fraction_binary': np.mean(binary_clusters == -1),
            'noise_fraction_activation': np.mean(activation_clusters == -1)
        }

    def compute_spline_codes(self,
                              binary_patterns: np.ndarray,
                              require_true_relu_masks: bool = False) -> np.ndarray:
        """
        Compute spline codes that identify polytope regions.
        Current implementation proxies spline codes with CETT-derived binary patterns.
        If require_true_relu_masks is True and true ReLU masks are not provided, raises an error.
        """
        # NOTE: True spline codes require capturing pre-activation sign patterns at non-linearities.
        # The current dataset contains post-layer activations and CETT binary patterns. We proxy with these.
        if require_true_relu_masks:
            raise RuntimeError("True spline codes require ReLU/GELU pre-activation masks during extraction. Re-run checkpoint extraction to capture masks.")
        return binary_patterns.copy()

    def compute_polytope_face_density(self,
                                      activations: np.ndarray,
                                      spline_codes: np.ndarray,
                                      max_hamming: int = 2) -> Dict[str, Any]:
        """
        Compute a proxy for polytope face density by counting nearby hyperplane crossings.
        For each sample i, sum hamming_distance(code_i, code_j) / euclidean_distance(x_i, x_j)
        over neighbors j whose hamming distance <= max_hamming.

        Returns per-sample densities and summary statistics.
        """
        if len(activations) != len(spline_codes):
            raise ValueError("activations and spline_codes must have the same number of samples")
        if len(activations) < 2:
            raise ValueError("Need at least two samples to compute face densities")

        n = len(activations)
        densities = np.zeros(n, dtype=float)
        for i in range(n):
            # Vectorized distances to all points
            diffs = activations - activations[i]
            euc = np.linalg.norm(diffs, axis=1)
            ham = np.sum(spline_codes != spline_codes[i], axis=1)
            mask = (ham > 0) & (ham <= max_hamming) & (euc > 1e-12)
            if np.any(mask):
                densities[i] = float(np.sum(ham[mask] / euc[mask]))
            else:
                densities[i] = 0.0

        return {
            'face_density_per_sample': densities,
            'face_density_mean': float(np.mean(densities)),
            'face_density_std': float(np.std(densities)),
            'face_density_percentiles': {
                '25th': float(np.percentile(densities, 25)),
                '50th': float(np.percentile(densities, 50)),
                '75th': float(np.percentile(densities, 75)),
            }
        }

    def compute_pattern_entropy(self, codes: np.ndarray) -> float:
        """Shannon entropy of row-wise patterns."""
        if codes.size == 0:
            return 0.0
        # Hash rows to counts
        hashes = np.array([hash(row.tobytes()) for row in codes.astype(np.uint8)])
        _, counts = np.unique(hashes, return_counts=True)
        probs = counts / counts.sum()
        return float(-np.sum(probs * np.log(probs + 1e-12)))

    def estimate_feature_count(self, spline_codes: np.ndarray) -> int:
        """Estimate number of effective features from polytope patterns via entropy and uniqueness."""
        if spline_codes.size == 0:
            return 0
        unique = np.unique(spline_codes, axis=0)
        code_entropy = self.compute_pattern_entropy(spline_codes)
        est = min(len(unique), int(np.floor(np.exp(code_entropy))))
        return int(est)

    def compute_polytope_diversity(self, spline_codes: np.ndarray) -> Dict[str, float]:
        """Diversity metrics over spline codes: unique fraction and entropy."""
        if spline_codes.size == 0:
            return {'unique_fraction': 0.0, 'entropy': 0.0}
        unique = np.unique(spline_codes, axis=0)
        unique_fraction = float(len(unique)) / float(len(spline_codes))
        entropy = self.compute_pattern_entropy(spline_codes)
        return {'unique_fraction': unique_fraction, 'entropy': float(entropy)}

    def compute_interference_strength(self, activations: np.ndarray) -> float:
        """Compute mean cosine overlap between neuron activity vectors as interference proxy."""
        if activations.size == 0:
            return 0.0
        A = activations.T  # neurons x samples
        norms = np.linalg.norm(A, axis=1)
        norms[norms == 0] = 1e-12
        A_norm = A / norms[:, None]
        # Sampled mean of upper triangle to avoid O(d^2) full storage for large dims
        d = A_norm.shape[0]
        if d < 2:
            return 0.0
        # Compute a small random subset if very large
        rng = np.random.default_rng(self.random_seed)
        max_pairs = min(20000, d * (d - 1) // 2)
        if d * (d - 1) // 2 <= max_pairs:
            # Exact
            sims = []
            for i in range(d):
                vi = A_norm[i]
                dots = A_norm[i + 1 :] @ vi
                sims.append(dots)
            sims_arr = np.concatenate(sims) if sims else np.array([])
        else:
            sims_list = []
            while len(sims_list) < max_pairs:
                i = int(rng.integers(0, d))
                j = int(rng.integers(0, d))
                if i >= j:
                    continue
                sims_list.append(float(np.dot(A_norm[i], A_norm[j])))
            sims_arr = np.array(sims_list)
        return float(np.mean(np.abs(sims_arr))) if sims_arr.size else 0.0

    def measure_feature_superposition(self,
                                      activations: np.ndarray,
                                      spline_codes: np.ndarray,
                                      n_features_est: Optional[int] = None) -> Dict[str, Any]:
        """
        Enhanced superposition summary using polytope diversity and interference proxy.
        """
        n_samples, n_dims = activations.shape
        if n_samples == 0 or n_dims == 0:
            return {
                'superposition_ratio': 0.0,
                'estimated_features': 0,
                'interference_strength': 0.0,
                'phase_classification': 'unknown',
                'polytope_diversity': {'unique_fraction': 0.0, 'entropy': 0.0},
            }

        if n_features_est is None:
            n_features_est = self.estimate_feature_count(spline_codes)

        superposition_ratio = float(n_features_est) / float(n_dims)
        interference_strength = self.compute_interference_strength(activations)
        phase_boundary = 1.0  # nominal threshold; interpret relative to n_dims and task
        phase = "strong" if superposition_ratio > phase_boundary else "weak"

        return {
            'superposition_ratio': float(superposition_ratio),
            'estimated_features': int(n_features_est),
            'interference_strength': float(interference_strength),
            'phase_classification_enhanced': phase,
            'polytope_diversity': self.compute_polytope_diversity(spline_codes),
        }

    def compute_polytope_region_metrics(self, spline_codes: np.ndarray) -> Dict[str, Any]:
        """Compute region-level metrics from codes: count, entropy, concentration (Gini)."""
        if spline_codes.size == 0:
            raise ValueError("Empty spline codes")
        # Hash rows to identify unique regions
        # Use view trick for speed-safe hashing
        codes = spline_codes.astype(np.uint8)
        # Convert to bytes per row
        hashes = np.array([hash(row.tobytes()) for row in codes])
        unique, counts = np.unique(hashes, return_counts=True)
        probs = counts / counts.sum()
        # Entropy
        entropy = float(-np.sum(probs * np.log(probs + 1e-12)))
        # Gini (concentration)
        gini = float(1.0 - np.sum(probs ** 2))
        return {
            'n_regions': int(len(unique)),
            'region_entropy': entropy,
            'region_concentration_gini': gini,
            'largest_region_fraction': float(np.max(probs)),
        }

    def compute_knn_flip_rate(self, binary_patterns: np.ndarray, activations: np.ndarray,
                               k: int = 10) -> Dict[str, Any]:
        """Estimate boundary density via k-NN flip rate: fraction of neighbors with differing codes."""
        if len(activations) < 2:
            raise ValueError("Need at least two samples to compute flip rates")
        k = max(1, min(k, len(activations) - 1))
        # Compute kNN in activation space
        dists = cdist(activations, activations, metric='euclidean')
        np.fill_diagonal(dists, np.inf)
        nn_idx = np.argpartition(dists, kth=k, axis=1)[:, :k]
        # For each sample, fraction of neighbors with different binary code
        flips = []
        for i in range(len(activations)):
            neighbors = nn_idx[i]
            diff = np.any(binary_patterns[neighbors] != binary_patterns[i], axis=1)
            flips.append(np.mean(diff))
        flips = np.array(flips, dtype=float)
        return {
            'flip_rate_per_sample': flips,
            'flip_rate_mean': float(np.mean(flips)),
            'flip_rate_std': float(np.std(flips)),
            'flip_rate_percentiles': {
                '25th': float(np.percentile(flips, 25)),
                '50th': float(np.percentile(flips, 50)),
                '75th': float(np.percentile(flips, 75))
            }
        }
    
    def compare_frequency_groups(self, high_freq_results: Dict, low_freq_results: Dict) -> Dict[str, Any]:
        """
        Statistical comparison between high and low frequency groups.
        
        Args:
            high_freq_results: Analysis results for high frequency n-grams
            low_freq_results: Analysis results for low frequency n-grams
            
        Returns:
            Comparative analysis results
        """
        comparison = {}
        
        # Key metrics to compare
        metrics = ['mean_density', 'compression_ratio',
                   'n_unique_patterns', 'cluster_consistency', 'interference_per_dimension',
                   'participation_ratio']
        
        for metric in metrics:
            if metric in high_freq_results and metric in low_freq_results:
                high_val = high_freq_results[metric]
                low_val = low_freq_results[metric]
                
                if low_val != 0:
                    comparison[f'{metric}_ratio'] = high_val / low_val
                else:
                    comparison[f'{metric}_ratio'] = float('inf') if high_val > 0 else 1.0
                    
                comparison[f'{metric}_difference'] = high_val - low_val
                
                # Effect size where raw arrays are available
                if metric == 'mean_density':
                    hf_arr = high_freq_results.get('polytope_densities', None)
                    lf_arr = low_freq_results.get('polytope_densities', None)
                    if hf_arr is not None and lf_arr is not None and len(hf_arr) > 1 and len(lf_arr) > 1:
                        pooled_std = np.sqrt((np.var(hf_arr, ddof=1) + np.var(lf_arr, ddof=1)) / 2.0)
                    if pooled_std > 0:
                            comparison[f'{metric}_effect_size'] = (np.mean(hf_arr) - np.mean(lf_arr)) / pooled_std
        
        # Superposition strength difference
        if 'interference_per_dimension' in high_freq_results and 'interference_per_dimension' in low_freq_results:
            comparison['superposition_strength_difference'] = (
                low_freq_results['interference_per_dimension'] - 
                high_freq_results['interference_per_dimension']
            )
        
        # Phase transition analysis
        high_phase = high_freq_results.get('phase_classification', 'unknown')
        low_phase = low_freq_results.get('phase_classification', 'unknown')
        comparison['phase_transition'] = f"{high_phase}_to_{low_phase}"
        
        return comparison
    
    def bootstrap_confidence_intervals(self, data: np.ndarray, n_bootstrap: int = 1000, 
                                     confidence: float = 0.95) -> Dict[str, float]:
        """
        Compute bootstrap confidence intervals for statistical robustness.
        
        Args:
            data: Input data array
            n_bootstrap: Number of bootstrap samples
            confidence: Confidence level
            
        Returns:
            Dictionary with confidence interval bounds
        """
        if len(data) == 0:
            raise ValueError("Cannot compute confidence intervals for empty data")
            
        bootstrap_means = []
        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
            bootstrap_means.append(np.mean(bootstrap_sample))
        
        alpha = 1 - confidence
        lower_percentile = (alpha/2) * 100
        upper_percentile = (1 - alpha/2) * 100
        
        return {
            'lower': np.percentile(bootstrap_means, lower_percentile),
            'upper': np.percentile(bootstrap_means, upper_percentile),
            'mean': np.mean(bootstrap_means)
        }
    
    def analyze_polytope_superposition(self,
                                       activations_high_freq: np.ndarray,
                                     activations_low_freq: np.ndarray,
                                       semantic_category: str = "n_grams",
                                        binary_patterns_high: Optional[np.ndarray] = None,
                                        binary_patterns_low: Optional[np.ndarray] = None,
                                        spline_codes_high: Optional[np.ndarray] = None,
                                        spline_codes_low: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Complete integrated analysis pipeline comparing high vs low frequency n-grams.
        
        Args:
            activations_high_freq: High frequency n-gram activations
            activations_low_freq: Low frequency n-gram activations
            semantic_category: Category label for analysis
            
        Returns:
            Comprehensive analysis results with comparative metrics
        """
        results = {}
        
        for freq_type, activations in [("high_freq", activations_high_freq), 
                                      ("low_freq", activations_low_freq)]:
            
            logger.info(f"Analyzing {freq_type} activations: {activations.shape}")
            
            # Phase 1: Binary patterns
            # Prefer provided CETT-derived patterns, else compute CETT-based binarization per sample
            if freq_type == 'high_freq' and binary_patterns_high is not None:
                binary_patterns = binary_patterns_high
            elif freq_type == 'low_freq' and binary_patterns_low is not None:
                binary_patterns = binary_patterns_low
            else:
                # Local CETT implementation (avoid cross-module heavy imports)
                def _cett_threshold(vec: np.ndarray, target_cett: float = 0.01) -> float:
                    magnitudes = np.abs(vec)
                    if not np.any(magnitudes):
                        return 0.0
                    sorted_mags = np.sort(magnitudes)
                    total_norm = np.linalg.norm(vec)
                    if total_norm == 0:
                        return 0.0
                    left, right = 0, len(sorted_mags) - 1
                    best = 0.0
                    while left <= right:
                        mid = (left + right) // 2
                        thr = sorted_mags[mid]
                        tail = vec[magnitudes < thr]
                        tail_norm = np.linalg.norm(tail)
                        current = tail_norm / total_norm
                        if current <= target_cett:
                            best = thr
                            left = mid + 1
                        else:
                            right = mid - 1
                    return best

                thresholds = np.array([_cett_threshold(v) for v in activations])
                binary_patterns = (np.abs(activations) > thresholds[:, None]).astype(int)
            
            processed = {
                'binary_patterns': binary_patterns,
                'sparsity': np.mean(binary_patterns, axis=1),
                'n_active_neurons': np.sum(binary_patterns, axis=1),
                'activation_norms': np.linalg.norm(activations, axis=1)
            }
            
            # Phase 2: Polytope Analysis
            polytope_metrics = self.compute_polytope_metrics(
                activations, processed['binary_patterns']
            )
            # Additional boundary density proxy via local flip rates
            flip_metrics = self.compute_knn_flip_rate(
                processed['binary_patterns'], activations, k=min(10, max(2, int(np.sqrt(len(activations))//2)))
            )
            # Prefer true spline codes if provided; otherwise proxy via CETT patterns
            if freq_type == 'high_freq' and spline_codes_high is not None:
                spline_codes = spline_codes_high
            elif freq_type == 'low_freq' and spline_codes_low is not None:
                spline_codes = spline_codes_low
            else:
                spline_codes = self.compute_spline_codes(processed['binary_patterns'])
            region_metrics = self.compute_polytope_region_metrics(spline_codes)

            # Optional geometric face density using spline codes
            try:
                face_density = self.compute_polytope_face_density(activations, spline_codes)
            except Exception as _:
                face_density = {
                    'face_density_per_sample': np.zeros(len(activations), dtype=float),
                    'face_density_mean': np.nan,
                    'face_density_std': np.nan,
                    'face_density_percentiles': {'25th': np.nan, '50th': np.nan, '75th': np.nan},
                }
            
            # Phase 3: Superposition Measurement  
            superposition_metrics = self.measure_superposition(
                activations, processed['binary_patterns']
            )

            # Enhanced superposition summary leveraging spline codes
            enhanced_superposition = self.measure_feature_superposition(activations, spline_codes)
            
            # Phase 4: Clustering Analysis
            clustering_results = self.clustering_analysis(
                processed['binary_patterns'], activations
            )
            
            # Bootstrap confidence intervals for key metrics
            density_ci = self.bootstrap_confidence_intervals(polytope_metrics['polytope_densities'])
            
            # Combine results
            results[freq_type] = {
                **processed,
                **polytope_metrics, 
                **{f'knn_{k}': v for k, v in flip_metrics.items()},
                **{f'region_{k}': v for k, v in region_metrics.items()},
                **superposition_metrics,
                **{f'face_{k}': v for k, v in face_density.items()},
                **{f'enh_{k}': v for k, v in enhanced_superposition.items()},
                **clustering_results,
                'density_confidence_interval': density_ci,
                'semantic_category': semantic_category
            }
        
        # Comparative analysis
        results['comparison'] = self.compare_frequency_groups(
            results['high_freq'], results['low_freq']
        )
        
        # Overall summary
        results['summary'] = {
            'high_freq_samples': activations_high_freq.shape[0],
            'low_freq_samples': activations_low_freq.shape[0],
            'activation_dimension': activations_high_freq.shape[1],
            'semantic_category': semantic_category,
            'analysis_timestamp': pd.Timestamp.now().isoformat() if PANDAS_AVAILABLE else None
        }
        
        return results
    
    def generate_analysis_report(self, results: Dict[str, Any]) -> str:
        """
        Generate human-readable analysis report.
        
        Args:
            results: Analysis results from analyze_polytope_superposition
            
        Returns:
            Formatted analysis report string
        """
        report = []
        report.append("=" * 80)
        report.append("SUPERPOSITION ANALYSIS REPORT")
        report.append("=" * 80)
        
        if 'summary' in results:
            summary = results['summary']
            report.append(f"Analysis Category: {summary.get('semantic_category', 'Unknown')}")
            report.append(f"High Freq Samples: {summary.get('high_freq_samples', 'Unknown')}")
            report.append(f"Low Freq Samples: {summary.get('low_freq_samples', 'Unknown')}")
            report.append(f"Activation Dimension: {summary.get('activation_dimension', 'Unknown')}")
            report.append("")
        
        # High frequency results
        hf = results['high_freq']
        report.append("HIGH FREQUENCY N-GRAMS:")
        report.append(f"  • Unique Patterns: {hf['n_unique_patterns']}")
        report.append(f"  • Compression Ratio: {hf['compression_ratio']:.3f}")
        report.append(f"  • Mean Polytope Density: {hf['mean_density']:.4f}")
        if 'participation_ratio' in hf:
            report.append(f"  • Participation Ratio: {hf['participation_ratio']:.2f}")
        report.append(f"  • Mean Squared Overlap (sampled): {hf['interference_per_dimension']:.4f}")
        report.append(f"  • Phase Classification: {hf['phase_classification']}")
        report.append(f"  • Cluster Consistency: {hf['cluster_consistency']:.3f}")
        report.append("")
        
        # Low frequency results
        lf = results['low_freq']
        report.append("LOW FREQUENCY N-GRAMS:")
        report.append(f"  • Unique Patterns: {lf['n_unique_patterns']}")
        report.append(f"  • Compression Ratio: {lf['compression_ratio']:.3f}")
        report.append(f"  • Mean Polytope Density: {lf['mean_density']:.4f}")
        if 'participation_ratio' in lf:
            report.append(f"  • Participation Ratio: {lf['participation_ratio']:.2f}")
        report.append(f"  • Mean Squared Overlap (sampled): {lf['interference_per_dimension']:.4f}")
        report.append(f"  • Phase Classification: {lf['phase_classification']}")
        report.append(f"  • Cluster Consistency: {lf['cluster_consistency']:.3f}")
        report.append("")
        
        # Comparative analysis
        comp = results['comparison']
        report.append("COMPARATIVE ANALYSIS:")
        report.append(f"  • Superposition Strength Difference: {comp['superposition_strength_difference']:.4f}")
        report.append(f"  • Density Ratio (H/L): {comp['mean_density_ratio']:.3f}")
        if 'interference_per_dimension_ratio' in comp:
            report.append(f"  • Interference Ratio (H/L): {comp['interference_per_dimension_ratio']:.3f}")
        report.append(f"  • Phase Transition: {comp['phase_transition']}")
        report.append("")
        
        report.append("=" * 80)
        return "\n".join(report)


class MultiCheckpointPolytopeAnalyzer:
    """
    Comprehensive multi-checkpoint polytope analysis for studying evolution
    of high/low frequency patterns across model training checkpoints and layers.
    """
    
    def __init__(self, min_cluster_size: int = 5, random_seed: Optional[int] = 42):
        self.base_analyzer = SuperpositionAnalyzer(min_cluster_size, random_seed)
        self.random_seed = random_seed
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def load_checkpoint_data(self, checkpoint_file: str) -> Dict[str, Any]:
        """Load checkpoint analysis results from file"""
        with open(checkpoint_file, 'rb') as f:
            data = pickle.load(f)
        return data
    
    def organize_records_by_attributes(self, records: List[Dict[str, Any]]) -> Dict[str, Dict[str, List]]:
        """Organize records by checkpoint, layer, and frequency category"""
        organized = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        
        for record in records:
            checkpoint = record['checkpoint_step']
            layer = record['layer']
            category = record['category']
            organized[checkpoint][layer][category].append(record)
        
        return dict(organized)
    
    def extract_activation_matrices(self, records: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
        """Extract activation vectors and binary patterns from records"""
        if not records:
            return np.array([]), np.array([])
        
        activations = np.stack([r['activation_vector'] for r in records])
        binary_patterns = np.stack([r['binary_pattern'] for r in records])
        
        return activations, binary_patterns

    def extract_spline_codes_from_records(self, records: List[Dict[str, Any]]) -> Optional[np.ndarray]:
        """Extract spline codes from records if present. Returns None if any are missing."""
        if not records:
            return None
        codes_list = []
        for r in records:
            code = r.get('spline_code', None)
            if code is None:
                return None
            codes_list.append(np.asarray(code, dtype=int))
        try:
            return np.stack(codes_list)
        except Exception:
            return None
    
    def analyze_checkpoint_layer_comparison(self, 
                                          high_freq_records: List[Dict[str, Any]], 
                                          low_freq_records: List[Dict[str, Any]],
                                          checkpoint: str,
                                          layer: int) -> Dict[str, Any]:
        """Analyze polytope differences for a specific checkpoint and layer"""
        
        if not high_freq_records or not low_freq_records:
            return {'error': 'Insufficient data for analysis', 'checkpoint': checkpoint, 'layer': layer}
        
        # Extract activation matrices
        high_activations, high_patterns = self.extract_activation_matrices(high_freq_records)
        low_activations, low_patterns = self.extract_activation_matrices(low_freq_records)
        # Optional spline codes if available in records
        high_codes = self.extract_spline_codes_from_records(high_freq_records)
        low_codes = self.extract_spline_codes_from_records(low_freq_records)
        
        # Run base polytope analysis
        results = self.base_analyzer.analyze_polytope_superposition(
            activations_high_freq=high_activations,
            activations_low_freq=low_activations,
            semantic_category=f"checkpoint_{checkpoint}_layer_{layer}",
            binary_patterns_high=high_patterns,
            binary_patterns_low=low_patterns,
            spline_codes_high=high_codes,
            spline_codes_low=low_codes
        )
        
        # Add checkpoint-specific metadata
        results['checkpoint_metadata'] = {
            'checkpoint_step': checkpoint,
            'layer': layer,
            'n_high_freq_samples': len(high_freq_records),
            'n_low_freq_samples': len(low_freq_records),
            'high_freq_ngrams': list(set(r['ngram'] for r in high_freq_records)),
            'low_freq_ngrams': list(set(r['ngram'] for r in low_freq_records))
        }
        
        return results
    
    def run_comprehensive_analysis(self, checkpoint_file: str) -> Dict[str, Any]:
        """Run comprehensive analysis across all checkpoints and layers"""
        
        # Load data
        data = self.load_checkpoint_data(checkpoint_file)
        records = data['records']
        metadata = data['metadata']
        
        print(f"Loaded {len(records)} records from {len(metadata['checkpoints'])} checkpoints")
        print(f"Target layers: {metadata['target_layers']}")
        
        # Organize records
        organized = self.organize_records_by_attributes(records)
        
        # Run analysis for each checkpoint-layer combination
        results = {}
        total_combinations = 0
        successful_analyses = 0
        
        for checkpoint in organized:
            results[checkpoint] = {}
            for layer in organized[checkpoint]:
                layer_data = organized[checkpoint][layer]
                
                # Check if we have both high and low frequency data
                # Support both naming conventions
                high_freq = layer_data.get('high_freq', layer_data.get('high', []))
                low_freq = layer_data.get('low_freq', layer_data.get('low', []))
                
                total_combinations += 1
                
                if high_freq and low_freq:
                    print(f"Analyzing checkpoint {checkpoint}, layer {layer}: {len(high_freq)} high freq, {len(low_freq)} low freq")
                    
                    analysis_result = self.analyze_checkpoint_layer_comparison(
                        high_freq, low_freq, checkpoint, layer
                    )
                    
                    results[checkpoint][layer] = analysis_result
                    successful_analyses += 1
                else:
                    print(f"Skipping checkpoint {checkpoint}, layer {layer}: insufficient data")
                    results[checkpoint][layer] = {
                        'error': 'Insufficient frequency category data',
                        'high_freq_count': len(high_freq),
                        'low_freq_count': len(low_freq)
                    }
        
        print(f"Completed {successful_analyses}/{total_combinations} analyses")
        
        # Compile comprehensive results
        comprehensive_results = {
            'individual_analyses': results,
            'metadata': metadata,
            'summary_statistics': self.compute_cross_checkpoint_statistics(results),
            'evolution_patterns': self.analyze_evolution_patterns(results)
        }
        
        return comprehensive_results
    
    def compute_cross_checkpoint_statistics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Compute statistics across checkpoints and layers"""
        
        stats = {
            'superposition_evolution': defaultdict(list),
            'polytope_density_evolution': defaultdict(list),
            'layer_differences': defaultdict(list),
            'checkpoint_progression': defaultdict(list)
        }
        
        for checkpoint, checkpoint_data in results.items():
            if not isinstance(checkpoint_data, dict):
                continue
                
            for layer, layer_data in checkpoint_data.items():
                if isinstance(layer_data, dict) and 'comparison' in layer_data:
                    comparison = layer_data['comparison']
                    
                    # Track superposition strength evolution
                    if 'superposition_strength_difference' in comparison:
                        stats['superposition_evolution'][layer].append({
                            'checkpoint': checkpoint,
                            'superposition_diff': comparison['superposition_strength_difference']
                        })
                    
                    # Track polytope density evolution
                    if 'mean_density_ratio' in comparison:
                        stats['polytope_density_evolution'][layer].append({
                            'checkpoint': checkpoint,
                            'density_ratio': comparison['mean_density_ratio']
                        })
        
        return dict(stats)
    
    def analyze_evolution_patterns(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze how polytope characteristics evolve during training"""
        
        patterns = {
            'layer_convergence': {},
            'training_dynamics': {},
            'frequency_separation': {}
        }
        
        # Analyze layer-wise convergence patterns
        for checkpoint, checkpoint_data in results.items():
            if not isinstance(checkpoint_data, dict):
                continue
                
            layer_metrics = {}
            for layer, layer_data in checkpoint_data.items():
                if isinstance(layer_data, dict) and 'comparison' in layer_data:
                    layer_metrics[layer] = {
                        'superposition_diff': layer_data['comparison'].get('superposition_strength_difference', 0),
                        'density_ratio': layer_data['comparison'].get('mean_density_ratio', 1.0),
                        'phase_transition': layer_data['comparison'].get('phase_transition', 'unknown')
                    }
            
            if layer_metrics:
                patterns['training_dynamics'][checkpoint] = layer_metrics
        
        return patterns
    
    def generate_evolution_visualizations(self, results: Dict[str, Any], output_dir: str = "cache/polytope_evolution"):
        """Generate comprehensive visualizations of polytope evolution"""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Extract data for visualization
        evolution_data = []
        
        for checkpoint, checkpoint_data in results['individual_analyses'].items():
            if not isinstance(checkpoint_data, dict):
                continue
                
            for layer, layer_data in checkpoint_data.items():
                if isinstance(layer_data, dict) and 'comparison' in layer_data:
                    comparison = layer_data['comparison']
                    evolution_data.append({
                        'checkpoint': int(checkpoint),
                        'layer': int(layer),
                        'superposition_diff': comparison.get('superposition_strength_difference', 0),
                        'density_ratio': comparison.get('mean_density_ratio', 1.0),
                        'interference_ratio': comparison.get('interference_per_dimension_ratio', 1.0),
                        'phase_transition': comparison.get('phase_transition', 'unknown')
                    })
        
        if not evolution_data:
            print("No data available for visualization")
            return
        
        df = pd.DataFrame(evolution_data)
        
        # Create evolution heatmaps
        plt.style.use('default')
        
        # 1. Superposition strength evolution
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        # Superposition difference heatmap
        pivot_sup = df.pivot(index='layer', columns='checkpoint', values='superposition_diff')
        sns.heatmap(pivot_sup, annot=True, cmap='RdBu_r', center=0, ax=ax1)
        ax1.set_title('Superposition Strength Difference\n(High Freq - Low Freq)')
        ax1.set_xlabel('Checkpoint Step')
        ax1.set_ylabel('Layer')
        
        # Density ratio heatmap
        pivot_density = df.pivot(index='layer', columns='checkpoint', values='density_ratio')
        sns.heatmap(pivot_density, annot=True, cmap='viridis', ax=ax2)
        ax2.set_title('Polytope Density Ratio\n(High Freq / Low Freq)')
        ax2.set_xlabel('Checkpoint Step')
        ax2.set_ylabel('Layer')
        
        # Interference ratio heatmap
        pivot_interference = df.pivot(index='layer', columns='checkpoint', values='interference_ratio')
        sns.heatmap(pivot_interference, annot=True, cmap='plasma', ax=ax3)
        ax3.set_title('Interference Ratio\n(High Freq / Low Freq)')
        ax3.set_xlabel('Checkpoint Step')
        ax3.set_ylabel('Layer')
        
        plt.tight_layout()
        plt.savefig(output_path / 'polytope_evolution_heatmaps.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Evolution line plots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Superposition evolution by layer
        for layer in df['layer'].unique():
            layer_data = df[df['layer'] == layer]
            ax1.plot(layer_data['checkpoint'], layer_data['superposition_diff'], 
                    marker='o', label=f'Layer {layer}', linewidth=2)
        ax1.set_xlabel('Checkpoint Step')
        ax1.set_ylabel('Superposition Strength Difference')
        ax1.set_title('Superposition Evolution by Layer')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Density ratio evolution by layer
        for layer in df['layer'].unique():
            layer_data = df[df['layer'] == layer]
            ax2.plot(layer_data['checkpoint'], layer_data['density_ratio'], 
                    marker='s', label=f'Layer {layer}', linewidth=2)
        ax2.set_xlabel('Checkpoint Step')
        ax2.set_ylabel('Density Ratio (High/Low)')
        ax2.set_title('Polytope Density Evolution by Layer')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Layer progression at different checkpoints
        checkpoints_to_show = sorted(df['checkpoint'].unique())[::2]  # Show every other checkpoint
        for checkpoint in checkpoints_to_show:
            checkpoint_data = df[df['checkpoint'] == checkpoint]
            ax3.plot(checkpoint_data['layer'], checkpoint_data['superposition_diff'], 
                    marker='o', label=f'Step {checkpoint}', linewidth=2)
        ax3.set_xlabel('Layer')
        ax3.set_ylabel('Superposition Strength Difference')
        ax3.set_title('Layer Progression at Different Checkpoints')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Scatter plot: Density vs Superposition
        scatter = ax4.scatter(df['density_ratio'], df['superposition_diff'], 
                             c=df['checkpoint'], cmap='viridis', 
                             s=60, alpha=0.7, edgecolors='black', linewidth=0.5)
        ax4.set_xlabel('Density Ratio (High/Low)')
        ax4.set_ylabel('Superposition Strength Difference')
        ax4.set_title('Density vs Superposition Relationship')
        ax4.grid(True, alpha=0.3)
        
        # Add colorbar for checkpoint steps
        cbar = plt.colorbar(scatter, ax=ax4)
        cbar.set_label('Checkpoint Step')
        
        plt.tight_layout()
        plt.savefig(output_path / 'polytope_evolution_trends.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Evolution visualizations saved to {output_path}")
    
    def generate_comprehensive_report(self, results: Dict[str, Any]) -> str:
        """Generate comprehensive analysis report"""
        
        report = []
        report.append("=" * 100)
        report.append("MULTI-CHECKPOINT POLYTOPE SUPERPOSITION ANALYSIS REPORT")
        report.append("=" * 100)
        report.append("")
        
        metadata = results.get('metadata', {})
        report.append(f"Model: {metadata.get('model_name', 'Unknown')}")
        report.append(f"Checkpoints analyzed: {len(metadata.get('checkpoints', []))}")
        report.append(f"Layers analyzed: {metadata.get('target_layers', [])}")
        report.append(f"Total records: {metadata.get('n_total_records', 0):,}")
        report.append("")
        
        # Summary statistics
        summary_stats = results.get('summary_statistics', {})
        if summary_stats:
            report.append("EVOLUTION SUMMARY:")
            report.append("-" * 50)
            
            # Analyze superposition evolution trends
            sup_evolution = summary_stats.get('superposition_evolution', {})
            for layer, layer_data in sup_evolution.items():
                if layer_data:
                    values = [d['superposition_diff'] for d in layer_data]
                    report.append(f"Layer {layer}:")
                    report.append(f"  • Superposition range: [{min(values):.3f}, {max(values):.3f}]")
                    report.append(f"  • Mean evolution: {np.mean(values):.3f} ± {np.std(values):.3f}")
            report.append("")
        
        # Evolution patterns
        evolution_patterns = results.get('evolution_patterns', {})
        if evolution_patterns:
            report.append("TRAINING DYNAMICS:")
            report.append("-" * 50)
            
            training_dynamics = evolution_patterns.get('training_dynamics', {})
            checkpoints = sorted([int(k) for k in training_dynamics.keys()])
            
            for checkpoint in checkpoints[:5]:  # Show first 5 checkpoints
                checkpoint_str = str(checkpoint)
                if checkpoint_str in training_dynamics:
                    data = training_dynamics[checkpoint_str]
                    report.append(f"Checkpoint {checkpoint}:")
                    
                    for layer, metrics in data.items():
                        report.append(f"  Layer {layer}: sup_diff={metrics['superposition_diff']:.3f}, "
                                    f"density_ratio={metrics['density_ratio']:.3f}")
                    report.append("")
        
        # Individual analysis highlights
        report.append("CHECKPOINT-LAYER ANALYSIS HIGHLIGHTS:")
        report.append("-" * 50)
        
        individual_analyses = results.get('individual_analyses', {})
        highlight_count = 0
        
        for checkpoint, checkpoint_data in individual_analyses.items():
            if not isinstance(checkpoint_data, dict) or highlight_count >= 10:
                continue
                
            for layer, layer_data in checkpoint_data.items():
                if isinstance(layer_data, dict) and 'comparison' in layer_data:
                    comparison = layer_data['comparison']
                    
                    # Highlight interesting cases
                    sup_diff = comparison.get('superposition_strength_difference', 0)
                    if abs(sup_diff) > 0.1:  # Significant superposition difference
                        report.append(f"Checkpoint {checkpoint}, Layer {layer}:")
                        report.append(f"  • Strong frequency effect: {sup_diff:.3f}")
                        report.append(f"  • Phase: {comparison.get('phase_transition', 'unknown')}")
                        report.append("")
                        highlight_count += 1
                        
                        if highlight_count >= 10:
                            break
        
        report.append("=" * 100)
        return "\n".join(report)


# -------------------------------
# Simplified functional pipeline
# -------------------------------
from typing import cast  # keep imports localized to minimize top-level noise


def _load_checkpoint_data_simple(checkpoint_file: str) -> Dict[str, Any]:
    with open(checkpoint_file, 'rb') as f:
        data = pickle.load(f)
    # Support both plain list of records and dict with metadata, mirroring CLI validator
    if isinstance(data, list):
        records = data
        checkpoints = set()
        layers = set()
        for r in records:
            checkpoints.add(r.get('checkpoint_step'))
            layers.add(r.get('layer'))
        metadata = {
            'model_name': 'Unknown',
            'checkpoints': sorted(list(checkpoints)),
            'target_layers': sorted(list(layers)),
            'n_total_records': len(records),
            'format': 'plain_list'
        }
        data = {'records': records, 'metadata': metadata}
    elif not (isinstance(data, dict) and 'records' in data and 'metadata' in data):
        raise ValueError("Checkpoint file must be a list of records or a dict with 'records' and 'metadata'")
    return cast(Dict[str, Any], data)


def _organize_records(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Dict[str, List[Dict[str, Any]]]]]:
    organized: Dict[str, Dict[str, Dict[str, List[Dict[str, Any]]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in records:
        organized[str(r['checkpoint_step'])][str(r['layer'])][r.get('category', 'unknown')].append(r)
    return dict(organized)


def _extract_matrices(records: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
    if not records:
        return np.array([]), np.array([])
    acts = np.stack([r['activation_vector'] for r in records])
    bins = np.stack([r['binary_pattern'] for r in records])
    return acts, bins


def _compute_cross_checkpoint_statistics(results: Dict[str, Any]) -> Dict[str, Any]:
    stats = {
        'superposition_evolution': defaultdict(list),
        'polytope_density_evolution': defaultdict(list),
        'layer_differences': defaultdict(list),
        'checkpoint_progression': defaultdict(list),
    }
    for checkpoint, ckpt_data in results.items():
        if not isinstance(ckpt_data, dict):
            continue
        for layer, layer_data in ckpt_data.items():
            if isinstance(layer_data, dict) and 'comparison' in layer_data:
                comp = layer_data['comparison']
                if 'superposition_strength_difference' in comp:
                    stats['superposition_evolution'][layer].append({'checkpoint': checkpoint, 'superposition_diff': comp['superposition_strength_difference']})
                if 'mean_density_ratio' in comp:
                    stats['polytope_density_evolution'][layer].append({'checkpoint': checkpoint, 'density_ratio': comp['mean_density_ratio']})
    return dict(stats)


def _analyze_evolution_patterns(results: Dict[str, Any]) -> Dict[str, Any]:
    patterns = {'layer_convergence': {}, 'training_dynamics': {}, 'frequency_separation': {}}
    for checkpoint, ckpt_data in results.items():
        if not isinstance(ckpt_data, dict):
            continue
        layer_metrics = {}
        for layer, layer_data in ckpt_data.items():
            if isinstance(layer_data, dict) and 'comparison' in layer_data:
                layer_metrics[layer] = {
                    'superposition_diff': layer_data['comparison'].get('superposition_strength_difference', 0),
                    'density_ratio': layer_data['comparison'].get('mean_density_ratio', 1.0),
                    'phase_transition': layer_data['comparison'].get('phase_transition', 'unknown'),
                }
        if layer_metrics:
            patterns['training_dynamics'][checkpoint] = layer_metrics
    return patterns


def _generate_visualizations(results: Dict[str, Any], output_dir: str = "cache/polytope_evolution") -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    evolution_data: List[Dict[str, Any]] = []
    for checkpoint, ckpt_data in results['individual_analyses'].items():
        if not isinstance(ckpt_data, dict):
            continue
        for layer, layer_data in ckpt_data.items():
            if isinstance(layer_data, dict) and 'comparison' in layer_data:
                comp = layer_data['comparison']
                evolution_data.append({
                    'checkpoint': int(checkpoint) if str(checkpoint).isdigit() else checkpoint,
                    'layer': int(layer) if str(layer).isdigit() else layer,
                    'superposition_diff': comp.get('superposition_strength_difference', 0),
                    'density_ratio': comp.get('mean_density_ratio', 1.0),
                    'interference_ratio': comp.get('interference_per_dimension_ratio', 1.0),
                    'phase_transition': comp.get('phase_transition', 'unknown'),
                })
    if not evolution_data:
        print("No data available for visualization")
        return
    df = pd.DataFrame(evolution_data)
    plt.style.use('default')
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    try:
        pivot_sup = df.pivot(index='layer', columns='checkpoint', values='superposition_diff')
        sns.heatmap(pivot_sup, annot=pivot_sup.size <= 200, cmap='RdBu_r', center=0, ax=ax1)
    except Exception:
        ax1.text(0.5, 0.5, 'Heatmap unavailable', ha='center')
    ax1.set_title('Superposition Strength Difference\n(High Freq - Low Freq)')
    ax1.set_xlabel('Checkpoint Step')
    ax1.set_ylabel('Layer')
    try:
        pivot_density = df.pivot(index='layer', columns='checkpoint', values='density_ratio')
        sns.heatmap(pivot_density, annot=pivot_density.size <= 200, cmap='viridis', ax=ax2)
    except Exception:
        ax2.text(0.5, 0.5, 'Heatmap unavailable', ha='center')
    ax2.set_title('Polytope Density Ratio\n(High Freq / Low Freq)')
    ax2.set_xlabel('Checkpoint Step')
    ax2.set_ylabel('Layer')
    try:
        pivot_interference = df.pivot(index='layer', columns='checkpoint', values='interference_ratio')
        sns.heatmap(pivot_interference, annot=pivot_interference.size <= 200, cmap='plasma', ax=ax3)
    except Exception:
        ax3.text(0.5, 0.5, 'Heatmap unavailable', ha='center')
    ax3.set_title('Interference Ratio\n(High Freq / Low Freq)')
    ax3.set_xlabel('Checkpoint Step')
    ax3.set_ylabel('Layer')
    plt.tight_layout()
    plt.savefig(output_path / 'polytope_evolution_heatmaps.png', dpi=300, bbox_inches='tight')
    plt.close()
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    for layer in df['layer'].unique():
        layer_data = df[df['layer'] == layer]
        ax1.plot(layer_data['checkpoint'], layer_data['superposition_diff'], marker='o', label=f'Layer {layer}', linewidth=2)
    ax1.set_xlabel('Checkpoint Step')
    ax1.set_ylabel('Superposition Strength Difference')
    ax1.set_title('Superposition Evolution by Layer')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    for layer in df['layer'].unique():
        layer_data = df[df['layer'] == layer]
        ax2.plot(layer_data['checkpoint'], layer_data['density_ratio'], marker='s', label=f'Layer {layer}', linewidth=2)
    ax2.set_xlabel('Checkpoint Step')
    ax2.set_ylabel('Density Ratio (High/Low)')
    ax2.set_title('Polytope Density Evolution by Layer')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    try:
        checkpoints_to_show = sorted([c for c in df['checkpoint'].unique() if isinstance(c, (int, float))])[::2]
    except Exception:
        checkpoints_to_show = []
    for checkpoint in checkpoints_to_show:
        checkpoint_data = df[df['checkpoint'] == checkpoint]
        ax3.plot(checkpoint_data['layer'], checkpoint_data['superposition_diff'], marker='o', label=f'Step {checkpoint}', linewidth=2)
    ax3.set_xlabel('Layer')
    ax3.set_ylabel('Superposition Strength Difference')
    ax3.set_title('Layer Progression at Different Checkpoints')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    scatter = ax4.scatter(df['density_ratio'], df['superposition_diff'], c=pd.factorize(df['checkpoint'])[0], cmap='viridis', s=60, alpha=0.7, edgecolors='black', linewidth=0.5)
    ax4.set_xlabel('Density Ratio (High/Low)')
    ax4.set_ylabel('Superposition Strength Difference')
    ax4.set_title('Density vs Superposition Relationship')
    ax4.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax4)
    cbar.set_label('Checkpoint (encoded)')
    plt.tight_layout()
    plt.savefig(output_path / 'polytope_evolution_trends.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Evolution visualizations saved to {output_path}")


def _collect_publication_dfs(results: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build tidy DataFrames for publication-quality figures.
    Returns:
      - timeseries_df: rows per checkpoint-layer-group with key metrics
      - densities_df: rows per sampled density value with checkpoint, layer, group
    """
    records_ts: List[Dict[str, Any]] = []
    records_den: List[Dict[str, Any]] = []
    rng = np.random.default_rng(42)
    for checkpoint, ckpt_data in results['individual_analyses'].items():
        if not isinstance(ckpt_data, dict):
            continue
        for layer, layer_data in ckpt_data.items():
            if not (isinstance(layer_data, dict) and 'high_freq' in layer_data and 'low_freq' in layer_data):
                continue
            for group_key, group_name in [('high_freq', 'high'), ('low_freq', 'low')]:
                g = layer_data[group_key]
                if not isinstance(g, dict):
                    continue
                record = {
                    'checkpoint': int(checkpoint) if str(checkpoint).isdigit() else checkpoint,
                    'layer': int(layer) if str(layer).isdigit() else layer,
                    'group': group_name,
                    'mean_density': g.get('mean_density', np.nan),
                    'participation_ratio': g.get('participation_ratio', np.nan),
                    'interference_per_dimension': g.get('interference_per_dimension', np.nan),
                    'n_samples': int(g.get('binary_patterns').shape[0]) if 'binary_patterns' in g else np.nan,
                }
                records_ts.append(record)
                # Densities: sample up to 1000 values to control size
                dens = g.get('polytope_densities', None)
                if dens is not None and len(dens) > 0:
                    sample_size = min(1000, len(dens))
                    if len(dens) > sample_size:
                        idx = rng.choice(len(dens), size=sample_size, replace=False)
                        dens_sample = np.asarray(dens)[idx]
                    else:
                        dens_sample = np.asarray(dens)
                    for v in dens_sample:
                        records_den.append({
                            'checkpoint': record['checkpoint'],
                            'layer': record['layer'],
                            'group': group_name,
                            'density': float(v),
                        })
    timeseries_df = pd.DataFrame.from_records(records_ts) if records_ts else pd.DataFrame()
    densities_df = pd.DataFrame.from_records(records_den) if records_den else pd.DataFrame()
    return timeseries_df, densities_df


def _generate_publication_figures(results: Dict[str, Any], output_dir: str = "cache/polytope_evolution") -> None:
    """Create clearer, publication-ready visualizations comparing high vs low and their evolution."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ts_df, den_df = _collect_publication_dfs(results)
    if ts_df.empty:
        print("No data available to generate publication figures")
        return

    # Figure A: Differences over time (High - Low) per layer for key metrics
    # Build diffs by merging high and low rows
    hf = ts_df[ts_df['group'] == 'high'].rename(columns={
        'mean_density': 'mean_density_high',
        'participation_ratio': 'participation_ratio_high',
        'interference_per_dimension': 'interference_high'
    })
    lf = ts_df[ts_df['group'] == 'low'].rename(columns={
        'mean_density': 'mean_density_low',
        'participation_ratio': 'participation_ratio_low',
        'interference_per_dimension': 'interference_low'
    })
    merge_keys = ['checkpoint', 'layer']
    diff_df = pd.merge(hf[merge_keys + ['mean_density_high', 'participation_ratio_high', 'interference_high']],
                       lf[merge_keys + ['mean_density_low', 'participation_ratio_low', 'interference_low']],
                       on=merge_keys, how='inner')
    if not diff_df.empty:
        diff_df['density_diff'] = diff_df['mean_density_high'] - diff_df['mean_density_low']
        diff_df['participation_ratio_diff'] = diff_df['participation_ratio_high'] - diff_df['participation_ratio_low']
        diff_df['interference_diff'] = diff_df['interference_high'] - diff_df['interference_low']
        # Plot
        fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True)
        for layer in sorted(diff_df['layer'].unique(), key=lambda x: int(x) if str(x).isdigit() else x):
            d = diff_df[diff_df['layer'] == layer].sort_values('checkpoint')
            axes[0].plot(d['checkpoint'], d['density_diff'], label=f'Layer {layer}', linewidth=2)
            axes[1].plot(d['checkpoint'], d['participation_ratio_diff'], label=f'Layer {layer}', linewidth=2)
            axes[2].plot(d['checkpoint'], d['interference_diff'], label=f'Layer {layer}', linewidth=2)
        axes[0].axhline(0, color='gray', linestyle='--', linewidth=1)
        axes[1].axhline(0, color='gray', linestyle='--', linewidth=1)
        axes[2].axhline(0, color='gray', linestyle='--', linewidth=1)
        axes[0].set_title('Polytope Density (High - Low)')
        axes[1].set_title('Participation Ratio (High - Low)')
        axes[2].set_title('Mean Squared Overlap (High - Low)')
        for ax in axes:
            ax.set_xlabel('Checkpoint')
            ax.grid(True, alpha=0.3)
        axes[0].set_ylabel('Difference')
        axes[0].legend(ncol=2, fontsize=8)
        plt.tight_layout()
        plt.savefig(out / 'metrics_differences_over_time.png', dpi=300, bbox_inches='tight')
        plt.close()

    # Figure B: Distribution of polytope densities at selected checkpoints (first, middle, last)
    if not den_df.empty:
        # Determine selected checkpoints
        try:
            checkpoints_sorted = sorted([c for c in den_df['checkpoint'].unique() if isinstance(c, (int, float))])
        except Exception:
            checkpoints_sorted = list(den_df['checkpoint'].unique())
        if len(checkpoints_sorted) > 0:
            selected = [checkpoints_sorted[0], checkpoints_sorted[len(checkpoints_sorted)//2], checkpoints_sorted[-1]]
            fig, axes = plt.subplots(len(selected), 1, figsize=(10, 4 * len(selected)), sharex=True)
            if len(selected) == 1:
                axes = [axes]
            for ax, cp in zip(axes, selected):
                df_cp = den_df[den_df['checkpoint'] == cp]
                # Violin plot per layer, colored by group
                try:
                    sns.violinplot(data=df_cp, x='layer', y='density', hue='group', split=True, inner='quart', ax=ax)
                except Exception:
                    sns.boxplot(data=df_cp, x='layer', y='density', hue='group', ax=ax)
                ax.set_title(f'Polytope Density Distributions at Checkpoint {cp}')
                ax.set_xlabel('Layer')
                ax.set_ylabel('Density (Hamming / Euclidean)')
                ax.grid(True, axis='y', alpha=0.3)
            handles, labels = axes[-1].get_legend_handles_labels()
            axes[-1].legend(handles, labels, title='Group')
            plt.tight_layout()
            plt.savefig(out / 'polytope_density_distributions_selected_checkpoints.png', dpi=300, bbox_inches='tight')
            plt.close()

    # Figure C: High vs Low evolution (means with 95% CI) for each metric aggregated across layers
    agg = ts_df.copy()
    # Aggregate by checkpoint and group (across layers)
    agg_means = agg.groupby(['checkpoint', 'group'], as_index=False)[['mean_density', 'participation_ratio', 'interference_per_dimension']].mean()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True)
    for group, color in [('high', '#1f77b4'), ('low', '#d62728')]:
        g = agg_means[agg_means['group'] == group].sort_values('checkpoint')
        axes[0].plot(g['checkpoint'], g['mean_density'], label=f'{group}', color=color, linewidth=2)
        axes[1].plot(g['checkpoint'], g['participation_ratio'], label=f'{group}', color=color, linewidth=2)
        axes[2].plot(g['checkpoint'], g['interference_per_dimension'], label=f'{group}', color=color, linewidth=2)
    axes[0].set_title('Mean Polytope Density (Avg across layers)')
    axes[1].set_title('Participation Ratio (Avg across layers)')
    axes[2].set_title('Mean Squared Overlap (Avg across layers)')
    for ax in axes:
        ax.set_xlabel('Checkpoint')
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel('Value')
    axes[0].legend(title='Group')
    plt.tight_layout()
    plt.savefig(out / 'group_means_over_time.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Publication figures saved to {out}")


def _generate_report(results: Dict[str, Any]) -> str:
    report = []
    report.append("=" * 100)
    report.append("MULTI-CHECKPOINT POLYTOPE SUPERPOSITION ANALYSIS REPORT")
    report.append("=" * 100)
    report.append("")
    metadata = results.get('metadata', {})
    report.append(f"Model: {metadata.get('model_name', 'Unknown')}")
    report.append(f"Checkpoints analyzed: {len(metadata.get('checkpoints', []))}")
    report.append(f"Layers analyzed: {metadata.get('target_layers', [])}")
    report.append(f"Total records: {metadata.get('n_total_records', 0):,}")
    report.append("")
    summary_stats = results.get('summary_statistics', {})
    if summary_stats:
        report.append("EVOLUTION SUMMARY:")
        report.append("-" * 50)
        sup_evolution = summary_stats.get('superposition_evolution', {})
        for layer, layer_data in sup_evolution.items():
            if layer_data:
                values = [d['superposition_diff'] for d in layer_data]
                report.append(f"Layer {layer}:")
                report.append(f"  • Superposition range: [{min(values):.3f}, {max(values):.3f}]")
                report.append(f"  • Mean evolution: {np.mean(values):.3f} ± {np.std(values):.3f}")
        report.append("")
    evolution_patterns = results.get('evolution_patterns', {})
    if evolution_patterns:
        report.append("TRAINING DYNAMICS:")
        report.append("-" * 50)
        training_dynamics = evolution_patterns.get('training_dynamics', {})
        try:
            checkpoints = sorted([int(k) for k in training_dynamics.keys()])
        except Exception:
            checkpoints = list(training_dynamics.keys())
        for checkpoint in checkpoints[:5]:
            checkpoint_str = str(checkpoint)
            if checkpoint_str in training_dynamics:
                data = training_dynamics[checkpoint_str]
                report.append(f"Checkpoint {checkpoint}:")
                for layer, metrics in data.items():
                    report.append(f"  Layer {layer}: sup_diff={metrics['superposition_diff']:.3f}, density_ratio={metrics['density_ratio']:.3f}")
                report.append("")
    report.append("CHECKPOINT-LAYER ANALYSIS HIGHLIGHTS:")
    report.append("-" * 50)
    individual_analyses = results.get('individual_analyses', {})
    highlight_count = 0
    for checkpoint, ckpt_data in individual_analyses.items():
        if not isinstance(ckpt_data, dict) or highlight_count >= 10:
            continue
        for layer, layer_data in ckpt_data.items():
            if isinstance(layer_data, dict) and 'comparison' in layer_data:
                comp = layer_data['comparison']
                sup_diff = comp.get('superposition_strength_difference', 0)
                if abs(sup_diff) > 0.1:
                    report.append(f"Checkpoint {checkpoint}, Layer {layer}:")
                    report.append(f"  • Strong frequency effect: {sup_diff:.3f}")
                    report.append(f"  • Phase: {comp.get('phase_transition', 'unknown')}")
                    report.append("")
                    highlight_count += 1
                    if highlight_count >= 10:
                        break
    report.append("=" * 100)
    return "\n".join(report)


def run_polytope_superposition_pipeline(checkpoint_file: str,
                                        output_dir: str = "cache/multi_checkpoint_analysis",
                                        min_cluster_size: int = 5,
                                        random_seed: Optional[int] = 42) -> Dict[str, Any]:
    """Singular, function-based pipeline for multi-checkpoint superposition analysis."""
    # Load
    data = _load_checkpoint_data_simple(checkpoint_file)
    records = data['records']
    metadata = data['metadata']
    print(f"Loaded {len(records)} records from {len(metadata.get('checkpoints', []))} checkpoints")
    print(f"Target layers: {metadata.get('target_layers', [])}")

    # Organize
    organized = _organize_records(records)

    # Analyzer
    analyzer = SuperpositionAnalyzer(min_cluster_size=min_cluster_size, random_seed=random_seed)

    # Per combination analysis
    results: Dict[str, Dict[str, Any]] = {}
    total, ok = 0, 0
    for checkpoint, layers in organized.items():
        results[checkpoint] = {}
        for layer, categories in layers.items():
            total += 1
            high = categories.get('high_freq', []) or categories.get('high', [])
            low = categories.get('low_freq', []) or categories.get('low', [])
            if not high or not low:
                results[checkpoint][layer] = {
                    'error': 'Insufficient frequency category data',
                    'high_freq_count': len(high),
                    'low_freq_count': len(low),
                }
                continue
            print(f"Analyzing checkpoint {checkpoint}, layer {layer}: {len(high)} high freq, {len(low)} low freq")
            high_act, high_bin = _extract_matrices(high)
            low_act, low_bin = _extract_matrices(low)
            analysis = analyzer.analyze_polytope_superposition(
                activations_high_freq=high_act,
                activations_low_freq=low_act,
                semantic_category=f"checkpoint_{checkpoint}_layer_{layer}",
                binary_patterns_high=high_bin,
                binary_patterns_low=low_bin,
            )
            analysis['checkpoint_metadata'] = {
                'checkpoint_step': checkpoint,
                'layer': layer,
                'n_high_freq_samples': len(high),
                'n_low_freq_samples': len(low),
                'high_freq_ngrams': list(set(r['ngram'] for r in high)),
                'low_freq_ngrams': list(set(r['ngram'] for r in low)),
            }
            results[checkpoint][layer] = analysis
            ok += 1
    print(f"Completed {ok}/{total} analyses")

    # Compile
    comprehensive = {
        'individual_analyses': results,
        'metadata': metadata,
        'summary_statistics': _compute_cross_checkpoint_statistics(results),
        'evolution_patterns': _analyze_evolution_patterns(results),
    }

    # Output
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print("Generating evolution visualizations...")
    _generate_visualizations(comprehensive, output_dir)
    print("Generating comprehensive report...")
    report = _generate_report(comprehensive)
    print("Generating publication figures...")
    _generate_publication_figures(comprehensive, output_dir)
    with open(output_path / "multi_checkpoint_results.pkl", 'wb') as f:
        pickle.dump(comprehensive, f)
    with open(output_path / "analysis_report.txt", 'w') as f:
        f.write(report)
    print(f"Analysis complete. Results saved to {output_path}")
    print("\nREPORT PREVIEW:")
    print("=" * 50)
    print(report[:2000] + "..." if len(report) > 2000 else report)
    return comprehensive
    
def run_multi_checkpoint_analysis(checkpoint_file: str, 
                                output_dir: str = "cache/multi_checkpoint_analysis") -> Dict[str, Any]:
    """Backwards-compatible wrapper for the simplified pipeline."""
    return run_polytope_superposition_pipeline(checkpoint_file, output_dir)


