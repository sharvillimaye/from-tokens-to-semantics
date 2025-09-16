#!/usr/bin/env python3
"""
Refined Polytope Superposition Analyzer for LLM Mechanistic Interpretability

This module implements a streamlined analysis pipeline for studying superposition
phenomena in LLM activations using spline codes from pre-activation patterns.

CORE RESEARCH FOCUS:
===================
1. Spline code analysis from pre-activation vectors
2. Polytope structure comparison between frequency groups  
3. Superposition quantification via interference patterns
4. Evolution tracking across training checkpoints
5. Clean visualization pipeline for research insights

ARCHITECTURE:
=============
- Uses spline codes as primary binary patterns (from pre-activation vectors)
- CETT thresholding for post-activation binary patterns (backup)
- Focused metrics: polytope density, participation ratio, interference
- Integration with checkpoint_analysis.py data structures

RESEARCH PIPELINE:
=================
1. Load checkpoint analysis results 
2. Extract spline codes and activation patterns
3. Compute polytope metrics for high vs low frequency groups
4. Generate visualizations and statistical comparisons
5. Track evolution across training checkpoints
"""

from concurrent.futures import ProcessPoolExecutor
import gc
import json
import multiprocessing as mp
from pathlib import Path
import pickle
from typing import Any, Dict, List
import warnings

from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore', category=FutureWarning)


class RefinedPolytopeAnalyzer:
    """
    Streamlined polytope analyzer focused on mechanistic interpretability research.
    
    Key improvements over original:
    - Prioritizes spline codes over CETT binary patterns
    - Cleaner API with direct integration to checkpoint analysis
    - Built-in logging and visualization pipeline
    - Research-focused metrics and statistical tests
    """
    
    def __init__(self, random_seed: int = 42, cache_dir: str = "cache"):
        """Initialize the refined analyzer."""
        self.random_seed = random_seed
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Runtime/config defaults
        self.max_pairs: int = 2000  # lower by default for faster, more stable runs
        self.per_checkpoint_timeout: int = 900  # seconds; avoid premature timeouts on large checkpoints
        
        # Configure logging
        logger.add(
            self.cache_dir / "polytope_analysis.log",
            rotation="10 MB",
            level="INFO",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
        )
        
        # Set random seed for reproducibility
        np.random.seed(random_seed)
        self.rng = np.random.default_rng(random_seed)
        
        logger.info("Initialized RefinedPolytopeAnalyzer")

    # ---------------------------
    # Numeric stability helpers
    # ---------------------------
    def _zscore_for_distance(self, activations: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        """Z-score features to stabilize Euclidean distance computations with overflow protection.
        Does not mutate the original array. Returns float64 array.
        """
        # Handle extreme values more carefully to prevent overflow
        # First check for inf/nan and replace with manageable values
        activations_clean = np.where(np.isfinite(activations), activations, 0.0)
        
        # Use more conservative clipping bounds to prevent overflow in subsequent operations
        clip_bound = 1e3  # Much more conservative than 1e6
        X = np.clip(activations_clean, -clip_bound, clip_bound)
        
        # Convert to float64 after clipping to avoid cast overflow
        X = X.astype(np.float64, copy=False)
        
        # Compute mean and std with overflow protection
        with np.errstate(over='ignore', invalid='ignore'):
            mean = np.mean(X, axis=0)
            std = np.std(X, axis=0)
        
        # Replace inf/nan in stats with safe values
        mean = np.where(np.isfinite(mean), mean, 0.0)
        std = np.where(np.isfinite(std) & (std >= eps), std, 1.0)
        
        # Perform z-scoring with additional overflow protection
        with np.errstate(over='ignore', invalid='ignore'):
            z_scored = (X - mean) / std
        
        # Final clipping to prevent extreme standardized values and handle any remaining inf/nan
        z_scored = np.where(np.isfinite(z_scored), z_scored, 0.0)
        return np.clip(z_scored, -6, 6)  # More conservative than -10,10

    def _rowwise_norm(self, X: np.ndarray) -> np.ndarray:
        """Compute row-wise L2 norms with clipping to avoid overflow."""
        # Handle inf/nan values first
        X_clean = np.where(np.isfinite(X), X, 0.0)
        
        # Convert to float64 and use conservative clipping
        X64 = X_clean.astype(np.float64, copy=False)
        X64 = np.clip(X64, -1e3, 1e3)  # More conservative clipping
        
        # Compute norms with overflow protection
        with np.errstate(over='ignore', invalid='ignore'):
            norms_squared = np.sum(X64 * X64, axis=1)
            norms = np.sqrt(norms_squared)
        
        # Handle any remaining inf/nan in the result
        norms = np.where(np.isfinite(norms), norms, 0.0)
        return norms

    def load_checkpoint_data(self, checkpoint_path: str, stream_processing: bool = True, 
                            chunk_size: int = 1000) -> Dict[str, Any]:
        """Load checkpoint analysis results with optional streaming for memory efficiency."""
        logger.info(f"Loading checkpoint data from {checkpoint_path}")
        
        if stream_processing:
            return self._load_checkpoint_data_streaming(checkpoint_path, chunk_size)
        else:
            return self._load_checkpoint_data_full(checkpoint_path)
    
    def _load_checkpoint_data_full(self, checkpoint_path: str) -> Dict[str, Any]:
        """Load full checkpoint data into memory (legacy method)."""
        with open(checkpoint_path, 'rb') as f:
            data = pickle.load(f)
        
        if isinstance(data, dict) and 'records' in data:
            records = data['records']
            metadata = data.get('metadata', {})
        else:
            # Backward compatibility
            records = data
            metadata = {}
        
        logger.info(f"Loaded {len(records)} activation records")
        logger.info(f"Metadata: {metadata}")
        
        return {'records': records, 'metadata': metadata}
    
    def _load_checkpoint_data_streaming(self, checkpoint_path: str, chunk_size: int) -> Dict[str, Any]:
        """Load checkpoint data with streaming to manage memory usage."""
        # First pass: read metadata and count records
        with open(checkpoint_path, 'rb') as f:
            data = pickle.load(f)
        
        if isinstance(data, dict) and 'records' in data:
            records = data['records']
            metadata = data.get('metadata', {})
        else:
            records = data
            metadata = {}
        
        logger.info(f"Loaded {len(records)} activation records with streaming support (chunk_size={chunk_size})")
        logger.info(f"Metadata: {metadata}")
        
        # For now, still return full data but with garbage collection
        # In production, this could be enhanced with actual streaming processing
        # where records are processed in chunks of size 'chunk_size'
        gc.collect()
        
        return {'records': records, 'metadata': metadata, 'streaming_enabled': True, 'chunk_size': chunk_size}

    def organize_by_frequency(self, records: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Organize records by frequency category (legacy method - use organize_by_layer_and_frequency for layer-aware analysis)."""
        organized = {'high_freq': [], 'medium_freq': [], 'low_freq': []}
        
        for record in records:
            # Normalize category names
            cat = record.get('category', record.get('frequency_category', 'unknown')).lower()
            
            if 'high' in cat:
                organized['high_freq'].append(record)
            elif 'medium' in cat:
                organized['medium_freq'].append(record)
            elif 'low' in cat:
                organized['low_freq'].append(record)

        # Log distribution
        for cat, recs in organized.items():
            if recs:
                logger.info(f"{cat}: {len(recs)} records")
        
        return organized
    
    def organize_by_layer_and_frequency(self, records: List[Dict[str, Any]]) -> Dict[int, Dict[str, List[Dict[str, Any]]]]:
        """Organize records by layer and frequency category for proper layer-wise analysis."""
        organized = {}
        
        for record in records:
            layer = record.get('layer')
            if layer is None:
                logger.warning(f"Record missing layer information: {record.get('phrase', 'unknown')}")
                continue
                
            # Initialize layer if not seen before
            if layer not in organized:
                organized[layer] = {'high_freq': [], 'medium_freq': [], 'low_freq': []}
            
            # Normalize category names
            cat = record.get('category', record.get('frequency_category', 'unknown')).lower()
            
            if 'high' in cat:
                organized[layer]['high_freq'].append(record)
            elif 'medium' in cat:
                organized[layer]['medium_freq'].append(record)
            elif 'low' in cat:
                organized[layer]['low_freq'].append(record)
        
        # Log distribution per layer
        for layer in sorted(organized.keys()):
            layer_counts = {cat: len(recs) for cat, recs in organized[layer].items() if recs}
            if layer_counts:
                logger.info(f"Layer {layer}: {layer_counts}")
        
        return organized

    def extract_analysis_matrices(self, records: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
        """Extract matrices for analysis from records."""
        if not records:
            logger.warning("No records provided for matrix extraction")
            return {}
        
        # Extract activation vectors (post-activation)
        activations = np.stack([r['activation_vector'] for r in records])
        
        # Extract spline codes (preferred binary patterns from pre-activation)
        spline_codes = []
        binary_patterns = []
        
        for r in records:
            # Prefer spline codes
            if 'spline_code' in r and r['spline_code'] is not None:
                spline_codes.append(r['spline_code'])
            else:
                spline_codes.append(None)
            
            # Always extract binary patterns as backup
            binary_patterns.append(r['binary_pattern'])
        
        # Convert to arrays, handling missing spline codes
        spline_available_count = sum(1 for sc in spline_codes if sc is not None)
        logger.info(f"Spline code availability: {spline_available_count}/{len(spline_codes)} records have spline codes")
        
        if any(sc is not None for sc in spline_codes):
            # Use spline codes where available, pad where missing
            valid_spline = [sc for sc in spline_codes if sc is not None]
            if valid_spline:
                # Get dimensions from valid spline codes
                spline_matrix = []
                
                for i, sc in enumerate(spline_codes):
                    if sc is not None:
                        # Ensure compatibility with activation dimensions
                        if len(sc) > activations.shape[1]:
                            sc = sc[:activations.shape[1]]
                        elif len(sc) < activations.shape[1]:
                            # Pad with zeros
                            sc = np.pad(sc, (0, activations.shape[1] - len(sc)))
                        spline_matrix.append(sc)
                    else:
                        # Use binary pattern as fallback
                        bp = binary_patterns[i]  # Use correct index
                        if len(bp) > activations.shape[1]:
                            bp = bp[:activations.shape[1]]
                        elif len(bp) < activations.shape[1]:
                            bp = np.pad(bp, (0, activations.shape[1] - len(bp)))
                        spline_matrix.append(bp)
                
                spline_codes_array = np.stack(spline_matrix)
                logger.info(f"Created hybrid spline/binary matrix: {spline_codes_array.shape}")
            else:
                spline_codes_array = None
        else:
            logger.warning("No spline codes available in any records - check pre-activation extraction")
            # Debug: Check if records have pre_activation_vector
            pre_act_count = sum(1 for r in records if r.get('pre_activation_vector') is not None)
            logger.info(f"Records with pre_activation_vector: {pre_act_count}/{len(records)}")
            spline_codes_array = None
        
        # Binary patterns array
        binary_patterns_array = np.stack(binary_patterns)
        
        # Ensure dimensions match
        if spline_codes_array is not None and spline_codes_array.shape[1] > activations.shape[1]:
            spline_codes_array = spline_codes_array[:, :activations.shape[1]]
        
        if binary_patterns_array.shape[1] > activations.shape[1]:
            binary_patterns_array = binary_patterns_array[:, :activations.shape[1]]
        
        matrices = {
            'activations': activations,
            'binary_patterns': binary_patterns_array,
        }
        
        if spline_codes_array is not None:
            matrices['spline_codes'] = spline_codes_array
            logger.info(f"Extracted spline codes: {spline_codes_array.shape}")
        else:
            logger.warning("No spline codes available, using binary patterns only")
        
        logger.info(f"Extracted matrices - Activations: {activations.shape}, Binary patterns: {binary_patterns_array.shape}")
        
        return matrices

    def compute_polytope_density(self, activations: np.ndarray, patterns: np.ndarray, 
                                max_pairs: int = 5000, chunk_size: int = 1000) -> Dict[str, float]:
        """Enhanced polytope density computation with chunked processing to prevent memory crashes."""
        n_samples = len(activations)

        # Standardize activations to avoid overflow in distance computations
        activations_std = self._zscore_for_distance(activations)

        # Build unique random pairs up to max_pairs (or all pairs if fewer)
        total_pairs = n_samples * (n_samples - 1) // 2
        if total_pairs <= max_pairs:
            pairs = np.array([(i, j) for i in range(n_samples) for j in range(i + 1, n_samples)], dtype=np.int32)
        else:
            # Memory-efficient vectorized sampling
            target = max_pairs
            pairs = self._generate_random_pairs_vectorized(n_samples, target)

        if len(pairs) == 0:
            return {'density_mean': 0.0, 'density_std': 0.0, 'boundary_crossings': 0.0}

        # Process pairs in chunks to prevent memory overload
        return self._compute_density_chunked(activations_std, patterns, pairs, chunk_size)
    
    def _generate_random_pairs_vectorized(self, n_samples: int, target: int) -> np.ndarray:
        """Generate random pairs efficiently without excessive memory usage."""
        # Generate more samples than needed to account for duplicates
        oversample_factor = 1.5
        n_candidates = min(int(target * oversample_factor), n_samples * (n_samples - 1) // 2)
        
        # Vectorized pair generation
        i = self.rng.integers(0, n_samples, size=n_candidates, dtype=np.int32)
        j = self.rng.integers(0, n_samples, size=n_candidates, dtype=np.int32)
        
        # Filter to get i < j pairs
        mask = i < j
        valid_pairs = np.column_stack([i[mask], j[mask]])
        
        # Remove duplicates and take first 'target' pairs
        unique_pairs = np.unique(valid_pairs, axis=0)
        return unique_pairs[:target]
    
    def _compute_density_chunked(self, activations: np.ndarray, patterns: np.ndarray, 
                                pairs: np.ndarray, chunk_size: int) -> Dict[str, float]:
        """Compute polytope density in chunks to manage memory usage."""
        all_densities = []
        all_euclidean = []
        all_hamming = []
        
        n_chunks = (len(pairs) + chunk_size - 1) // chunk_size
        
        for chunk_idx in range(n_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, len(pairs))
            chunk_pairs = pairs[start_idx:end_idx]
            
            # Vectorized distance computation for chunk
            diff = activations[chunk_pairs[:, 0]] - activations[chunk_pairs[:, 1]]
            # Safe norm to avoid overflow
            euclidean_distances = self._rowwise_norm(diff)
            
            # Hamming distances on binary patterns
            xor = patterns[chunk_pairs[:, 0]] != patterns[chunk_pairs[:, 1]]
            hamming_distances = np.sum(xor, axis=1)
            
            # Store results for this chunk
            all_euclidean.append(euclidean_distances)
            all_hamming.append(hamming_distances)
            
            # Compute densities with robust error handling
            with np.errstate(divide='ignore', invalid='ignore'):
                chunk_densities = np.where(euclidean_distances > 1e-12, 
                                         hamming_distances / euclidean_distances, np.nan)
            
            valid_mask = ~np.isnan(chunk_densities) & np.isfinite(chunk_densities)
            if np.any(valid_mask):
                all_densities.append(chunk_densities[valid_mask])
        
        # Combine results from all chunks
        if not all_densities:
            logger.warning("No valid polytope density calculations possible")
            return {'density_mean': 0.0, 'density_std': 0.0, 'boundary_crossings': 0.0}
        
        densities = np.concatenate(all_densities)
        euclidean_all = np.concatenate(all_euclidean)
        hamming_all = np.concatenate(all_hamming)
        
        # Enhanced boundary analysis
        boundary_threshold = np.percentile(euclidean_all, 25)
        boundary_crossings = int(np.sum((hamming_all > 0) & (euclidean_all < boundary_threshold)))
        boundary_crossing_rate = float(boundary_crossings) / float(len(euclidean_all))
        
        return {
            'density_mean': float(np.mean(densities)),
            'density_std': float(np.std(densities)),
            'boundary_crossings': boundary_crossings,
            'boundary_crossing_rate': boundary_crossing_rate,
            'density_percentiles': {
                '25th': float(np.percentile(densities, 25)),
                '75th': float(np.percentile(densities, 75)),
                '90th': float(np.percentile(densities, 90))
            },
            'valid_pairs': len(densities)
        }

    def compute_participation_ratio(self, activations: np.ndarray) -> float:
        """Compute participation ratio (effective dimensionality) with robust numerical handling."""
        try:
            # Handle inf/nan values first
            activations_clean = np.where(np.isfinite(activations), activations, 0.0)
            
            # Use more conservative clipping to prevent overflow
            activations_clipped = np.clip(activations_clean, -1e3, 1e3)
            
            # Standardize activations with overflow protection
            with np.errstate(over='ignore', invalid='ignore'):
                scaler = StandardScaler()
                activations_std = scaler.fit_transform(activations_clipped)
            
            # Handle any inf/nan from standardization
            activations_std = np.where(np.isfinite(activations_std), activations_std, 0.0)
            activations_std = np.clip(activations_std, -6, 6)  # Conservative clipping
            
            # PCA with error handling
            pca = PCA()
            pca.fit(activations_std)
            
            # Participation ratio from eigenvalue spectrum with numerical safeguards
            eigenvals = pca.explained_variance_
            eigenvals = np.where(np.isfinite(eigenvals), eigenvals, 0.0)
            eigenvals = np.maximum(eigenvals, 1e-12)  # Avoid division by zero
            
            with np.errstate(over='ignore', invalid='ignore'):
                numerator = np.sum(eigenvals) ** 2
                denominator = np.sum(eigenvals ** 2)
            
            # Check for overflow/invalid results
            if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator < 1e-12:
                logger.warning("Participation ratio calculation unstable, returning default")
                return 1.0
                
            participation_ratio = numerator / denominator
            
            # Ensure result is reasonable and finite
            if not np.isfinite(participation_ratio):
                return 1.0
                
            participation_ratio = np.clip(participation_ratio, 1.0, float(len(eigenvals)))
            
            return float(participation_ratio)
            
        except (np.linalg.LinAlgError, ValueError, RuntimeWarning) as e:
            logger.warning(f"Participation ratio computation failed: {e}, returning default value")
            return 1.0

    def compute_interference_patterns(self, activations_high: np.ndarray, activations_low: np.ndarray) -> Dict[str, float]:
        """Enhanced interference pattern analysis between frequency groups with vectorized operations."""
        # Standardize for stability
        activations_high_std = self._zscore_for_distance(activations_high)
        activations_low_std = self._zscore_for_distance(activations_low)
        
        # Mean vectors for each group (in standardized space)
        mean_high = np.mean(activations_high_std, axis=0)
        mean_low = np.mean(activations_low_std, axis=0)
        
        # Cosine similarity between group means with robust norm computation
        norm_high = np.sqrt(np.sum(np.clip(mean_high, -1e6, 1e6) ** 2))
        norm_low = np.sqrt(np.sum(np.clip(mean_low, -1e6, 1e6) ** 2))
        denom = norm_high * norm_low
        cos_sim = (np.dot(mean_high, mean_low) / denom) if denom > 0 else 0.0
        
        # Activation norms for distribution analysis
        norms_high = self._rowwise_norm(activations_high_std)
        norms_low = self._rowwise_norm(activations_low_std)
        
        # Statistical tests
        statistic, p_value = mannwhitneyu(norms_high, norms_low, alternative='two-sided')
        
        # Vectorized neuron-wise interference computation
        overlap_score = self._compute_neuron_correlations_vectorized(activations_high_std, activations_low_std)
        
        return {
            'cosine_similarity': float(cos_sim),
            'norm_correlation': float(np.corrcoef(norms_high, norms_low)[0, 1]) if len(norms_high) == len(norms_low) else 0.0,
            'mann_whitney_statistic': float(statistic),
            'mann_whitney_pvalue': float(p_value),
            'neuron_overlap_score': overlap_score,
            'mean_norm_high': float(np.mean(norms_high)),
            'mean_norm_low': float(np.mean(norms_low)),
            'norm_ratio': float(np.mean(norms_high) / np.mean(norms_low)) if np.mean(norms_low) > 0 else 0.0
        }
    
    def _compute_neuron_correlations_vectorized(self, activations_high: np.ndarray, activations_low: np.ndarray) -> float:
        """Compute neuron-wise correlations using vectorized operations for better performance."""
        min_neurons = min(activations_high.shape[1], activations_low.shape[1])
        min_samples = min(activations_high.shape[0], activations_low.shape[0])
        
        # Extract matching neurons and samples to ensure compatible dimensions
        high_neurons = activations_high[:min_samples, :min_neurons]
        low_neurons = activations_low[:min_samples, :min_neurons]
        
        # Compute variances for all neurons at once
        high_vars = np.var(high_neurons, axis=0)
        low_vars = np.var(low_neurons, axis=0)
        
        # Find neurons with non-zero variance
        valid_mask = (high_vars > 0) & (low_vars > 0)
        
        if not np.any(valid_mask):
            return 0.0
        
        # Extract valid neurons
        high_valid = high_neurons[:, valid_mask]
        low_valid = low_neurons[:, valid_mask]
        
        # Need at least 2 samples for correlation
        if high_valid.shape[0] < 2:
            return 0.0
        
        # Compute correlations for all valid neuron pairs at once
        correlations = []
        for i in range(high_valid.shape[1]):
            high_neuron = high_valid[:, i]
            low_neuron = low_valid[:, i]
            
            # Check if both arrays have the same length and sufficient variance
            if len(high_neuron) == len(low_neuron) and len(high_neuron) > 1:
                corr = np.corrcoef(high_neuron, low_neuron)[0, 1]
                if np.isfinite(corr):
                    correlations.append(abs(corr))
        
        return float(np.mean(correlations)) if correlations else 0.0

    def compute_ngram_polytope_mapping_metrics(self, patterns: np.ndarray, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute n-gram to polytope mapping metrics from patterns (spline codes or binary patterns)."""
        if len(patterns) == 0:
            return {
                'polysemantic_fraction': 0.0,
                'ngram_reuse_fraction': 0.0,
                'avg_phrases_per_polytope': 0.0,
                'cross_frequency_sharing_fraction': 0.0
            }
        
        # Unique polytope analysis
        unique_patterns, inverse_indices, counts = np.unique(patterns, axis=0, return_inverse=True, return_counts=True)
        
        # Enhanced N-gram to polytope mapping
        polytope_to_ngrams = {}
        ngram_to_polytopes = {}
        
        for i, record in enumerate(records):
            if i >= len(inverse_indices):  # Safety check
                continue
                
            polytope_idx = inverse_indices[i]
            polytope_id = tuple(unique_patterns[polytope_idx].astype(int))
            
            phrase = record.get('phrase', f'sample_{i}')
            
            if phrase:  # Only process non-empty phrases
                # Bidirectional mapping: polytope -> n-grams and n-gram -> polytopes
                if polytope_id not in polytope_to_ngrams:
                    polytope_to_ngrams[polytope_id] = set()
                polytope_to_ngrams[polytope_id].add(phrase)
                
                if phrase not in ngram_to_polytopes:
                    ngram_to_polytopes[phrase] = set()
                ngram_to_polytopes[phrase].add(polytope_id)
        
        # Polysemantic analysis (polytopes with multiple n-grams)
        polysemantic_count = sum(1 for ngrams in polytope_to_ngrams.values() if len(ngrams) > 1)
        total_polytopes = len(polytope_to_ngrams)
        polysemantic_fraction = polysemantic_count / total_polytopes if total_polytopes > 0 else 0.0
        
        # N-gram reuse analysis (n-grams appearing in multiple polytopes)
        ngram_reuse_count = sum(1 for polytopes in ngram_to_polytopes.values() if len(polytopes) > 1)
        total_ngrams = len(ngram_to_polytopes)
        ngram_reuse_fraction = ngram_reuse_count / total_ngrams if total_ngrams > 0 else 0.0
        
        # Average phrases per polytope
        avg_phrases_per_polytope = np.mean([len(ngrams) for ngrams in polytope_to_ngrams.values()]) if polytope_to_ngrams else 0.0
        
        return {
            'polysemantic_fraction': float(polysemantic_fraction),
            'ngram_reuse_fraction': float(ngram_reuse_fraction),
            'avg_phrases_per_polytope': float(avg_phrases_per_polytope),
            'total_polytopes': total_polytopes,
            'total_unique_ngrams': total_ngrams,
            'polysemantic_count': polysemantic_count,
            'ngram_reuse_count': ngram_reuse_count
        }

    def compute_spline_code_metrics(self, spline_codes: np.ndarray, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute comprehensive metrics for spline code analysis with advanced n-gram to polytope mapping."""
        if len(spline_codes) == 0:
            return {}
        
        # Unique polytope analysis
        unique_codes, inverse_indices, counts = np.unique(spline_codes, axis=0, return_inverse=True, return_counts=True)
        
        # Enhanced N-gram to polytope mapping with frequency tracking
        polytope_to_ngrams = {}
        ngram_to_polytopes = {}
        frequency_based_mapping = {'high_freq': {}, 'low_freq': {}}
        
        for i, record in enumerate(records):
            if i >= len(inverse_indices):  # Safety check
                continue
                
            polytope_idx = inverse_indices[i]
            polytope_id = tuple(unique_codes[polytope_idx].astype(int))
            
            phrase = record.get('phrase', f'sample_{i}')
            freq_category = record.get('category', record.get('frequency_category', 'unknown')).lower()
            
            if phrase:  # Only process non-empty phrases
                # Bidirectional mapping: polytope -> n-grams and n-gram -> polytopes
                if polytope_id not in polytope_to_ngrams:
                    polytope_to_ngrams[polytope_id] = set()
                polytope_to_ngrams[polytope_id].add(phrase)
                
                if phrase not in ngram_to_polytopes:
                    ngram_to_polytopes[phrase] = set()
                ngram_to_polytopes[phrase].add(polytope_id)
                
                # Frequency-specific mapping
                freq_key = 'high_freq' if 'high' in freq_category else 'low_freq' if 'low' in freq_category else 'unknown'
                if freq_key != 'unknown':
                    if polytope_id not in frequency_based_mapping[freq_key]:
                        frequency_based_mapping[freq_key][polytope_id] = set()
                    frequency_based_mapping[freq_key][polytope_id].add(phrase)
        
        # Polysemantic analysis (polytopes with multiple n-grams)
        polysemantic_count = 0
        polytope_phrase_counts = []
        highly_polysemantic_threshold = 3  # Polytopes with 3+ n-grams
        highly_polysemantic_count = 0
        
        for polytope_id, ngrams in polytope_to_ngrams.items():
            unique_ngram_count = len(ngrams)
            polytope_phrase_counts.append(unique_ngram_count)
            if unique_ngram_count > 1:
                polysemantic_count += 1
            if unique_ngram_count >= highly_polysemantic_threshold:
                highly_polysemantic_count += 1
        
        # N-gram reuse analysis (n-grams appearing in multiple polytopes)  
        ngram_reuse_count = 0
        ngram_polytope_counts = []
        for phrase, polytope_set in ngram_to_polytopes.items():
            polytope_count = len(polytope_set)
            ngram_polytope_counts.append(polytope_count)
            if polytope_count > 1:
                ngram_reuse_count += 1
        
        # Cross-frequency polytope sharing
        high_freq_polytopes = set(frequency_based_mapping['high_freq'].keys())
        low_freq_polytopes = set(frequency_based_mapping['low_freq'].keys())
        shared_polytopes = high_freq_polytopes.intersection(low_freq_polytopes)
        
        # Comprehensive metrics
        total_polytopes = len(unique_codes)
        total_ngrams = len(ngram_to_polytopes)
        
        polysemantic_fraction = polysemantic_count / total_polytopes if total_polytopes > 0 else 0.0
        highly_polysemantic_fraction = highly_polysemantic_count / total_polytopes if total_polytopes > 0 else 0.0
        ngram_reuse_fraction = ngram_reuse_count / total_ngrams if total_ngrams > 0 else 0.0
        
        avg_phrases_per_polytope = np.mean(polytope_phrase_counts) if polytope_phrase_counts else 0.0
        max_phrases_per_polytope = max(polytope_phrase_counts) if polytope_phrase_counts else 0
        avg_polytopes_per_ngram = np.mean(ngram_polytope_counts) if ngram_polytope_counts else 0.0
        
        return {
            # Basic polytope metrics
            'total_polytopes': total_polytopes,
            'total_unique_ngrams': total_ngrams,
            'polytope_occupancy': counts.tolist(),
            'mean_occupancy': float(np.mean(counts)),
            'max_occupancy': int(np.max(counts)),
            'samples_per_polytope': float(len(spline_codes) / total_polytopes),
            
            # Polysemantic analysis (polytopes with multiple n-grams)
            'polysemantic_polytope_count': polysemantic_count,
            'polysemantic_fraction': float(polysemantic_fraction),
            'highly_polysemantic_count': highly_polysemantic_count,
            'highly_polysemantic_fraction': float(highly_polysemantic_fraction),
            'avg_phrases_per_polytope': float(avg_phrases_per_polytope),
            'max_phrases_per_polytope': int(max_phrases_per_polytope),
            
            # N-gram reuse analysis (n-grams in multiple polytopes)
            'ngram_reuse_count': ngram_reuse_count,
            'ngram_reuse_fraction': float(ngram_reuse_fraction),
            'avg_polytopes_per_ngram': float(avg_polytopes_per_ngram),
            'max_polytopes_per_ngram': int(max(ngram_polytope_counts)) if ngram_polytope_counts else 0,
            
            # Cross-frequency analysis
            'cross_frequency_shared_polytopes': len(shared_polytopes),
            'cross_frequency_sharing_fraction': len(shared_polytopes) / total_polytopes if total_polytopes > 0 else 0.0,
            'high_freq_only_polytopes': len(high_freq_polytopes - low_freq_polytopes),
            'low_freq_only_polytopes': len(low_freq_polytopes - high_freq_polytopes),
            
            # Detailed mappings (for research analysis)
            'polytope_to_phrase_mapping': {str(k): list(v) for k, v in polytope_to_ngrams.items()},
            'ngram_to_polytope_mapping': {k: [str(p) for p in v] for k, v in ngram_to_polytopes.items()},
            'frequency_based_mapping': {
                freq: {str(k): list(v) for k, v in mapping.items()} 
                for freq, mapping in frequency_based_mapping.items()
            }
        }
    
    def analyze_checkpoint(self, records: List[Dict[str, Any]], checkpoint_step: str, 
                         use_layer_wise_analysis: bool = True) -> Dict[str, Any]:
        """Enhanced checkpoint analysis with layer-wise metrics."""
        logger.info(f"Analyzing checkpoint {checkpoint_step} with {len(records)} records")
        
        if use_layer_wise_analysis:
            return self._analyze_checkpoint_layer_wise(records, checkpoint_step)
        else:
            return self._analyze_checkpoint_legacy(records, checkpoint_step)
    
    def _analyze_checkpoint_legacy(self, records: List[Dict[str, Any]], checkpoint_step: str) -> Dict[str, Any]:
        """Legacy analysis method that averages across layers (kept for backward compatibility)."""
        # Organize by frequency (ignoring layers)
        freq_groups = self.organize_by_frequency(records)
        
        # Focus on high vs low frequency comparison
        high_freq_records = freq_groups['high_freq']
        low_freq_records = freq_groups['low_freq']
        
        if not high_freq_records or not low_freq_records:
            logger.warning(f"Missing frequency groups in checkpoint {checkpoint_step}")
            return {'error': 'Missing frequency groups'}
        
        # Extract matrices
        high_matrices = self.extract_analysis_matrices(high_freq_records)
        low_matrices = self.extract_analysis_matrices(low_freq_records)
        
        results = {
            'checkpoint_step': checkpoint_step,
            'n_high_freq': len(high_freq_records),
            'n_low_freq': len(low_freq_records),
            'analysis_type': 'legacy_layer_averaged'
        }
        
        # Compute metrics for each frequency group (averaged across layers)
        for freq_type, matrices in [('high_freq', high_matrices), ('low_freq', low_matrices)]:
            if not matrices:
                continue
                
            activations = matrices['activations']
            patterns = matrices.get('spline_codes', matrices['binary_patterns'])
            
            density_metrics = self.compute_polytope_density(activations, patterns, max_pairs=self.max_pairs)
            results[f'{freq_type}_density'] = density_metrics
            
            # Fixed pattern reuse calculation - detects actual reuse
            unique_patterns, counts = np.unique(patterns, axis=0, return_counts=True)
            total_patterns = len(patterns)
            unique_pattern_count = len(unique_patterns)
            
            # Pattern reuse rate: fraction of patterns that are reused (appear >1 time)
            reused_patterns = np.sum(counts > 1)  # Count patterns appearing multiple times
            pattern_reuse_rate = float(reused_patterns / unique_pattern_count) if unique_pattern_count > 0 else 0.0
            
            # Additional metric: pattern compression ratio
            compression_ratio = float(unique_pattern_count / total_patterns) if total_patterns > 0 else 1.0
            
            results[f'{freq_type}_unique_polytopes'] = unique_pattern_count
            results[f'{freq_type}_compression_ratio'] = compression_ratio
            results[f'{freq_type}_pattern_reuse_rate'] = pattern_reuse_rate
            results[f'{freq_type}_participation_ratio'] = self.compute_participation_ratio(activations)
            results[f'{freq_type}_sparsity'] = float(np.mean(np.sum(patterns, axis=1) / patterns.shape[1]))
            results[f'{freq_type}_activation_norm'] = float(np.mean(self._rowwise_norm(activations)))
            
            # Compute n-gram to polytope mapping metrics (prefer spline codes, fallback to binary patterns)
            pattern_matrix = matrices.get('spline_codes', matrices['binary_patterns'])
            mapping_metrics = self.compute_ngram_polytope_mapping_metrics(pattern_matrix, 
                high_freq_records if freq_type == 'high_freq' else low_freq_records)
            results[f'{freq_type}_mapping_metrics'] = mapping_metrics
            
            # Keep legacy spline metrics for backward compatibility
            if 'spline_codes' in matrices:
                spline_metrics = self.compute_spline_code_metrics(matrices['spline_codes'], 
                    high_freq_records if freq_type == 'high_freq' else low_freq_records)
                results[f'{freq_type}_spline_metrics'] = spline_metrics
        
        # Cross-group analysis
        if high_matrices and low_matrices:
            interference = self.compute_interference_patterns(
                high_matrices['activations'], low_matrices['activations']
            )
            results['interference'] = interference
            
            high_patterns = high_matrices.get('spline_codes', high_matrices['binary_patterns'])
            low_patterns = low_matrices.get('spline_codes', low_matrices['binary_patterns'])
            
            high_polytopes = {tuple(pattern.astype(int)) for pattern in high_patterns}
            low_polytopes = {tuple(pattern.astype(int)) for pattern in low_patterns}
            
            shared_polytopes = high_polytopes.intersection(low_polytopes)
            total_unique_polytopes = len(high_polytopes.union(low_polytopes))
            
            results['polytope_sharing'] = {
                'shared_polytope_count': len(shared_polytopes),
                'total_unique_polytopes': total_unique_polytopes,
                'sharing_fraction': len(shared_polytopes) / total_unique_polytopes if total_unique_polytopes > 0 else 0.0,
                'high_only_polytopes': len(high_polytopes - low_polytopes),
                'low_only_polytopes': len(low_polytopes - high_polytopes)
            }
        
        return results
    
    def _analyze_checkpoint_layer_wise(self, records: List[Dict[str, Any]], checkpoint_step: str) -> Dict[str, Any]:
        """New layer-wise analysis method that computes metrics per layer."""
        # Organize by layer and frequency
        layer_groups = self.organize_by_layer_and_frequency(records)
        
        if not layer_groups:
            logger.warning(f"No valid layer groups found in checkpoint {checkpoint_step}")
            return {'error': 'No valid layer groups'}
        
        results = {
            'checkpoint_step': checkpoint_step,
            'analysis_type': 'layer_wise',
            'layers': sorted(layer_groups.keys()),
            'layer_results': {}
        }
        
        total_records = 0
        for layer, freq_groups in layer_groups.items():
            high_freq_records = freq_groups['high_freq']
            low_freq_records = freq_groups['low_freq']
            
            total_records += len(high_freq_records) + len(low_freq_records)
            
            if not high_freq_records or not low_freq_records:
                logger.warning(f"Missing frequency groups in layer {layer} of checkpoint {checkpoint_step}")
                results['layer_results'][layer] = {'error': 'Missing frequency groups'}
                continue
            
            # Analyze this specific layer
            layer_result = self._analyze_single_layer(layer, high_freq_records, low_freq_records, checkpoint_step)
            results['layer_results'][layer] = layer_result
        
        results['total_records'] = total_records
        
        # Add layer transition analysis
        transition_metrics = self._analyze_layer_transitions(layer_groups, checkpoint_step)
        results['layer_transitions'] = transition_metrics
        
        logger.info(f"Checkpoint {checkpoint_step} layer-wise analysis complete for {len(layer_groups)} layers")
        return results
    
    def _analyze_single_layer(self, layer: int, high_freq_records: List[Dict[str, Any]], 
                            low_freq_records: List[Dict[str, Any]], checkpoint_step: str) -> Dict[str, Any]:
        """Analyze a single layer with high/low frequency comparison."""
        # Extract matrices for this layer
        high_matrices = self.extract_analysis_matrices(high_freq_records)
        low_matrices = self.extract_analysis_matrices(low_freq_records)
        
        layer_result = {
            'layer': layer,
            'n_high_freq': len(high_freq_records),
            'n_low_freq': len(low_freq_records),
        }
        
        # Compute metrics for each frequency group in this layer
        for freq_type, matrices, records in [('high_freq', high_matrices, high_freq_records), 
                                           ('low_freq', low_matrices, low_freq_records)]:
            if not matrices:
                continue
                
            activations = matrices['activations']
            patterns = matrices.get('spline_codes', matrices['binary_patterns'])
            
            # All the same metrics, but now layer-specific
            density_metrics = self.compute_polytope_density(activations, patterns, max_pairs=self.max_pairs)
            layer_result[f'{freq_type}_density'] = density_metrics
            
            # Fixed pattern reuse calculation - detects actual reuse
            unique_patterns, counts = np.unique(patterns, axis=0, return_counts=True)
            total_patterns = len(patterns)
            unique_pattern_count = len(unique_patterns)
            
            # Pattern reuse rate: fraction of patterns that are reused (appear >1 time)
            reused_patterns = np.sum(counts > 1)  # Count patterns appearing multiple times
            pattern_reuse_rate = float(reused_patterns / unique_pattern_count) if unique_pattern_count > 0 else 0.0
            
            # Additional metric: pattern compression ratio
            compression_ratio = float(unique_pattern_count / total_patterns) if total_patterns > 0 else 1.0
            
            layer_result[f'{freq_type}_unique_polytopes'] = unique_pattern_count
            layer_result[f'{freq_type}_compression_ratio'] = compression_ratio
            layer_result[f'{freq_type}_pattern_reuse_rate'] = pattern_reuse_rate
            layer_result[f'{freq_type}_participation_ratio'] = self.compute_participation_ratio(activations)
            layer_result[f'{freq_type}_sparsity'] = float(np.mean(np.sum(patterns, axis=1) / patterns.shape[1]))
            layer_result[f'{freq_type}_activation_norm'] = float(np.mean(self._rowwise_norm(activations)))
            
            # Compute n-gram to polytope mapping metrics (prefer spline codes, fallback to binary patterns)
            pattern_matrix = matrices.get('spline_codes', matrices['binary_patterns'])
            mapping_metrics = self.compute_ngram_polytope_mapping_metrics(pattern_matrix, records)
            layer_result[f'{freq_type}_mapping_metrics'] = mapping_metrics
            
            # Keep legacy spline metrics for backward compatibility
            if 'spline_codes' in matrices:
                spline_metrics = self.compute_spline_code_metrics(matrices['spline_codes'], records)
                layer_result[f'{freq_type}_spline_metrics'] = spline_metrics
        
        # Cross-group analysis for this layer
        if high_matrices and low_matrices:
            interference = self.compute_interference_patterns(
                high_matrices['activations'], low_matrices['activations']
            )
            layer_result['interference'] = interference
            
            # Layer-specific polytope sharing
            high_patterns = high_matrices.get('spline_codes', high_matrices['binary_patterns'])
            low_patterns = low_matrices.get('spline_codes', low_matrices['binary_patterns'])
            
            high_polytopes = {tuple(pattern.astype(int)) for pattern in high_patterns}
            low_polytopes = {tuple(pattern.astype(int)) for pattern in low_patterns}
            
            shared_polytopes = high_polytopes.intersection(low_polytopes)
            total_unique_polytopes = len(high_polytopes.union(low_polytopes))
            
            layer_result['polytope_sharing'] = {
                'shared_polytope_count': len(shared_polytopes),
                'total_unique_polytopes': total_unique_polytopes,
                'sharing_fraction': len(shared_polytopes) / total_unique_polytopes if total_unique_polytopes > 0 else 0.0,
                'high_only_polytopes': len(high_polytopes - low_polytopes),
                'low_only_polytopes': len(low_polytopes - high_polytopes)
            }
        
        return layer_result
    
    def _analyze_layer_transitions(self, layer_groups: Dict[int, Dict[str, List[Dict[str, Any]]]], 
                                 checkpoint_step: str) -> Dict[str, Any]:
        """Analyze how polytopes transition between adjacent layers."""
        sorted_layers = sorted(layer_groups.keys())
        if len(sorted_layers) < 2:
            return {'warning': 'Need at least 2 layers for transition analysis'}
        
        transition_results = {
            'layer_pairs': [],
            'transition_metrics': {},
            'inheritance_patterns': {},
            'emergence_patterns': {}
        }
        
        # Analyze transitions between adjacent layers
        for i in range(len(sorted_layers) - 1):
            layer_a = sorted_layers[i]
            layer_b = sorted_layers[i + 1]
            
            transition_results['layer_pairs'].append((layer_a, layer_b))
            
            # Get patterns for both layers across frequency groups
            for freq_type in ['high_freq', 'low_freq']:
                records_a = layer_groups[layer_a][freq_type]
                records_b = layer_groups[layer_b][freq_type]
                
                if not records_a or not records_b:
                    continue
                
                # Extract pattern matrices
                matrices_a = self.extract_analysis_matrices(records_a)
                matrices_b = self.extract_analysis_matrices(records_b)
                
                if not matrices_a or not matrices_b:
                    continue
                
                patterns_a = matrices_a.get('spline_codes', matrices_a['binary_patterns'])
                patterns_b = matrices_b.get('spline_codes', matrices_b['binary_patterns'])
                
                # Compute transition metrics
                transition_key = f"{layer_a}_to_{layer_b}_{freq_type}"
                transition_metrics = self._compute_pattern_transitions(patterns_a, patterns_b, records_a, records_b)
                transition_results['transition_metrics'][transition_key] = transition_metrics
        
        # Compute global inheritance patterns across all layers
        inheritance_metrics = self._compute_inheritance_patterns(layer_groups, sorted_layers)
        transition_results['inheritance_patterns'] = inheritance_metrics
        
        return transition_results
    
    def _compute_pattern_transitions(self, patterns_a: np.ndarray, patterns_b: np.ndarray,
                                   records_a: List[Dict[str, Any]], records_b: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute detailed metrics for pattern transitions between two layers."""
        # Get unique patterns in each layer
        unique_a, indices_a, counts_a = np.unique(patterns_a, axis=0, return_inverse=True, return_counts=True)
        unique_b, indices_b, counts_b = np.unique(patterns_b, axis=0, return_inverse=True, return_counts=True)
        
        # Convert to sets of tuples for intersection analysis
        set_a = {tuple(pattern.astype(int)) for pattern in unique_a}
        set_b = {tuple(pattern.astype(int)) for pattern in unique_b}
        
        # Pattern inheritance: patterns that appear in both layers
        inherited_patterns = set_a.intersection(set_b)
        emerged_patterns = set_b - set_a  # New patterns in layer B
        lost_patterns = set_a - set_b     # Patterns that disappeared
        
        # Compute inheritance metrics
        inheritance_fraction = len(inherited_patterns) / len(set_a) if len(set_a) > 0 else 0.0
        emergence_fraction = len(emerged_patterns) / len(set_b) if len(set_b) > 0 else 0.0
        loss_fraction = len(lost_patterns) / len(set_a) if len(set_a) > 0 else 0.0
        
        # Pattern stability: measure how pattern usage changes
        stability_scores = []
        for pattern in inherited_patterns:
            pattern_array = np.array(pattern, dtype=patterns_a.dtype)
            
            # Count occurrences in each layer
            count_a = np.sum(np.all(patterns_a == pattern_array, axis=1))
            count_b = np.sum(np.all(patterns_b == pattern_array, axis=1))
            
            # Stability = minimum count / maximum count (0 to 1)
            if count_a > 0 and count_b > 0:
                stability = min(count_a, count_b) / max(count_a, count_b)
                stability_scores.append(stability)
        
        avg_pattern_stability = np.mean(stability_scores) if stability_scores else 0.0
        
        # N-gram inheritance analysis
        ngram_inheritance = self._analyze_ngram_inheritance(inherited_patterns, emerged_patterns, 
                                                          records_a, records_b, patterns_a, patterns_b)
        
        return {
            'total_patterns_a': len(set_a),
            'total_patterns_b': len(set_b),
            'inherited_patterns': len(inherited_patterns),
            'emerged_patterns': len(emerged_patterns),
            'lost_patterns': len(lost_patterns),
            'inheritance_fraction': float(inheritance_fraction),
            'emergence_fraction': float(emergence_fraction),
            'loss_fraction': float(loss_fraction),
            'pattern_stability': float(avg_pattern_stability),
            'pattern_novelty': float(len(emerged_patterns) / (len(set_a) + len(set_b)) if len(set_a) + len(set_b) > 0 else 0.0),
            'ngram_inheritance': ngram_inheritance
        }
    
    def _analyze_ngram_inheritance(self, inherited_patterns: set, emerged_patterns: set,
                                 records_a: List[Dict[str, Any]], records_b: List[Dict[str, Any]], 
                                 patterns_a: np.ndarray, patterns_b: np.ndarray) -> Dict[str, Any]:
        """Analyze how n-grams are inherited or emerge between layers."""
        # Map patterns to n-grams in each layer
        pattern_to_ngrams_a = {}
        pattern_to_ngrams_b = {}
        
        for i, record in enumerate(records_a):
            if i < len(patterns_a):
                pattern_key = tuple(patterns_a[i].astype(int))
                phrase = record.get('phrase', f'sample_{i}')
                if pattern_key not in pattern_to_ngrams_a:
                    pattern_to_ngrams_a[pattern_key] = set()
                pattern_to_ngrams_a[pattern_key].add(phrase)
        
        for i, record in enumerate(records_b):
            if i < len(patterns_b):
                pattern_key = tuple(patterns_b[i].astype(int))
                phrase = record.get('phrase', f'sample_{i}')
                if pattern_key not in pattern_to_ngrams_b:
                    pattern_to_ngrams_b[pattern_key] = set()
                pattern_to_ngrams_b[pattern_key].add(phrase)
        
        # Analyze n-gram conservation for inherited patterns
        conserved_ngrams = 0
        total_inherited_ngrams = 0
        new_ngram_mappings = 0
        
        for pattern in inherited_patterns:
            ngrams_a = pattern_to_ngrams_a.get(pattern, set())
            ngrams_b = pattern_to_ngrams_b.get(pattern, set())
            
            if ngrams_a and ngrams_b:
                conserved = len(ngrams_a.intersection(ngrams_b))
                new_mappings = len(ngrams_b - ngrams_a)
                
                conserved_ngrams += conserved
                total_inherited_ngrams += len(ngrams_a)
                new_ngram_mappings += new_mappings
        
        # Analyze emerging patterns and their n-grams
        emerged_ngram_count = sum(len(pattern_to_ngrams_b.get(pattern, set())) 
                                for pattern in emerged_patterns)
        
        return {
            'conserved_ngrams': conserved_ngrams,
            'total_inherited_ngrams': total_inherited_ngrams,
            'ngram_conservation_rate': conserved_ngrams / total_inherited_ngrams if total_inherited_ngrams > 0 else 0.0,
            'new_ngram_mappings': new_ngram_mappings,
            'emerged_ngram_count': emerged_ngram_count,
            'semantic_stability': conserved_ngrams / (conserved_ngrams + new_ngram_mappings) if (conserved_ngrams + new_ngram_mappings) > 0 else 0.0
        }
    
    def _compute_inheritance_patterns(self, layer_groups: Dict[int, Dict[str, List[Dict[str, Any]]]], 
                                    sorted_layers: List[int]) -> Dict[str, Any]:
        """Compute global inheritance patterns across all layers."""
        inheritance_summary = {
            'layer_sequence': sorted_layers,
            'global_pattern_flow': {},
            'frequency_specific_inheritance': {}
        }
        
        # Track patterns across the entire layer sequence
        for freq_type in ['high_freq', 'low_freq']:
            layer_patterns = {}
            
            # Extract patterns from each layer
            for layer in sorted_layers:
                records = layer_groups[layer][freq_type]
                if records:
                    matrices = self.extract_analysis_matrices(records)
                    if matrices:
                        patterns = matrices.get('spline_codes', matrices['binary_patterns'])
                        unique_patterns = {tuple(p.astype(int)) for p in np.unique(patterns, axis=0)}
                        layer_patterns[layer] = unique_patterns
            
            # Compute inheritance metrics across all layers
            if layer_patterns:
                first_layer = min(layer_patterns.keys())
                last_layer = max(layer_patterns.keys())
                
                # Patterns that persist from first to last layer
                persistent_patterns = layer_patterns[first_layer]
                for layer in sorted(layer_patterns.keys())[1:]:
                    persistent_patterns = persistent_patterns.intersection(layer_patterns[layer])
                
                # Patterns that appear in multiple layers (but not necessarily all)
                all_patterns = set()
                for patterns in layer_patterns.values():
                    all_patterns.update(patterns)
                
                multi_layer_patterns = set()
                for pattern in all_patterns:
                    layer_count = sum(1 for layer_pats in layer_patterns.values() if pattern in layer_pats)
                    if layer_count > 1:
                        multi_layer_patterns.add(pattern)
                
                inheritance_summary['frequency_specific_inheritance'][freq_type] = {
                    'total_unique_patterns': len(all_patterns),
                    'persistent_patterns': len(persistent_patterns),
                    'multi_layer_patterns': len(multi_layer_patterns),
                    'persistence_rate': len(persistent_patterns) / len(all_patterns) if all_patterns else 0.0,
                    'multi_layer_rate': len(multi_layer_patterns) / len(all_patterns) if all_patterns else 0.0,
                    'average_layer_span': np.mean([
                        sum(1 for layer_pats in layer_patterns.values() if pattern in layer_pats)
                        for pattern in all_patterns
                    ]) if all_patterns else 0.0
                }
        
        return inheritance_summary

    def create_evolution_visualization(self, results: List[Dict[str, Any]], output_dir: str) -> None:
        """Create evolution visualizations supporting both legacy and layer-wise analysis."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Check if we have layer-wise results
        has_layer_wise = any(r.get('analysis_type') == 'layer_wise' for r in results if 'error' not in r)
        
        if has_layer_wise:
            self._create_layer_wise_visualizations(results, output_path)
        else:
            self._create_legacy_visualizations(results, output_path)
    
    def _create_legacy_visualizations(self, results: List[Dict[str, Any]], output_path: Path) -> None:
        """Create legacy visualizations (averaged across layers)."""
        df_data = []
        for result in results:
            if 'error' in result:
                continue
                
            checkpoint = result['checkpoint_step']
            high_density = result.get('high_freq_density', {})
            low_density = result.get('low_freq_density', {})
            
            df_data.append({
                'checkpoint': int(checkpoint),
                'group': 'high_freq',
                'density_mean': high_density.get('density_mean', 0),
                'participation_ratio': result.get('high_freq_participation_ratio', 0),
                'sparsity': result.get('high_freq_sparsity', 0),
                'activation_norm': result.get('high_freq_activation_norm', 0)
            })
            
            df_data.append({
                'checkpoint': int(checkpoint),
                'group': 'low_freq', 
                'density_mean': low_density.get('density_mean', 0),
                'participation_ratio': result.get('low_freq_participation_ratio', 0),
                'sparsity': result.get('low_freq_sparsity', 0),
                'activation_norm': result.get('low_freq_activation_norm', 0)
            })
        
        if not df_data:
            logger.warning("No data for legacy visualization")
            return
            
        df = pd.DataFrame(df_data)
        self._plot_legacy_evolution(df, output_path)
    
    def _create_layer_wise_visualizations(self, results: List[Dict[str, Any]], output_path: Path) -> None:
        """Create layer-wise evolution visualizations."""
        df_data = []
        
        for result in results:
            if 'error' in result or result.get('analysis_type') != 'layer_wise':
                continue
                
            checkpoint = int(result['checkpoint_step'])
            layer_results = result.get('layer_results', {})
            
            for layer_num, layer_data in layer_results.items():
                if 'error' in layer_data:
                    continue
                    
                # High freq metrics for this layer
                high_density = layer_data.get('high_freq_density', {})
                low_density = layer_data.get('low_freq_density', {})
                # N-gram mapping metrics (new improved metrics)
                high_mapping = layer_data.get('high_freq_mapping_metrics', {})
                low_mapping = layer_data.get('low_freq_mapping_metrics', {})
                # Legacy spline mapping metrics (if computed)
                high_spline = layer_data.get('high_freq_spline_metrics', {})
                low_spline = layer_data.get('low_freq_spline_metrics', {})
                # Cross-frequency polytope sharing (layer-level)
                sharing = layer_data.get('polytope_sharing', {})
                sharing_fraction = sharing.get('sharing_fraction', None)
                
                df_data.append({
                    'checkpoint': checkpoint,
                    'layer': layer_num,
                    'group': 'high_freq',
                    'density_mean': high_density.get('density_mean', 0),
                    'participation_ratio': layer_data.get('high_freq_participation_ratio', 0),
                    'sparsity': layer_data.get('high_freq_sparsity', 0),
                    'activation_norm': layer_data.get('high_freq_activation_norm', 0),
                    'pattern_reuse_rate': layer_data.get('high_freq_pattern_reuse_rate', 0),
                    # N-gram mapping metrics (prefer new metrics, fallback to spline)
                    'mapping_polysemantic_fraction': high_mapping.get('polysemantic_fraction', high_spline.get('polysemantic_fraction')),
                    'mapping_ngram_reuse_fraction': high_mapping.get('ngram_reuse_fraction', high_spline.get('ngram_reuse_fraction')),
                    'mapping_avg_phrases_per_polytope': high_mapping.get('avg_phrases_per_polytope', high_spline.get('avg_phrases_per_polytope')),
                    # Legacy spline mapping metrics columns (may be None)
                    'spline_polysemantic_fraction': high_spline.get('polysemantic_fraction'),
                    'spline_ngram_reuse_fraction': high_spline.get('ngram_reuse_fraction'),
                    'spline_avg_phrases_per_polytope': high_spline.get('avg_phrases_per_polytope'),
                    # Duplicate sharing per group for easy aggregation later
                    'sharing_fraction': sharing_fraction
                })
                
                df_data.append({
                    'checkpoint': checkpoint,
                    'layer': layer_num,
                    'group': 'low_freq', 
                    'density_mean': low_density.get('density_mean', 0),
                    'participation_ratio': layer_data.get('low_freq_participation_ratio', 0),
                    'sparsity': layer_data.get('low_freq_sparsity', 0),
                    'activation_norm': layer_data.get('low_freq_activation_norm', 0),
                    'pattern_reuse_rate': layer_data.get('low_freq_pattern_reuse_rate', 0),
                    # N-gram mapping metrics (prefer new metrics, fallback to spline)
                    'mapping_polysemantic_fraction': low_mapping.get('polysemantic_fraction', low_spline.get('polysemantic_fraction')),
                    'mapping_ngram_reuse_fraction': low_mapping.get('ngram_reuse_fraction', low_spline.get('ngram_reuse_fraction')),
                    'mapping_avg_phrases_per_polytope': low_mapping.get('avg_phrases_per_polytope', low_spline.get('avg_phrases_per_polytope')),
                    # Legacy spline mapping metrics columns (may be None)
                    'spline_polysemantic_fraction': low_spline.get('polysemantic_fraction'),
                    'spline_ngram_reuse_fraction': low_spline.get('ngram_reuse_fraction'),
                    'spline_avg_phrases_per_polytope': low_spline.get('avg_phrases_per_polytope'),
                    'sharing_fraction': sharing_fraction
                })
        
        if not df_data:
            logger.warning("No data for layer-wise visualization")
            return
            
        df = pd.DataFrame(df_data)
        self._plot_layer_wise_evolution(df, output_path)
    
    def _plot_legacy_evolution(self, df: pd.DataFrame, output_path: Path) -> None:
        """Plot legacy evolution (averaged across layers)."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Polytope Evolution Across Training Checkpoints (Layer Averaged)', fontsize=16)
        
        metrics = [
            ('density_mean', 'Polytope Density'),
            ('participation_ratio', 'Participation Ratio'), 
            ('sparsity', 'Sparsity'),
            ('activation_norm', 'Activation Norm')
        ]
        
        for idx, (metric, title) in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]
            
            for group in ['high_freq', 'low_freq']:
                group_data = df[df['group'] == group]
                if not group_data.empty:
                    ax.plot(group_data['checkpoint'], group_data[metric], 
                           marker='o', label=group.replace('_', ' ').title())
            
            ax.set_xlabel('Training Checkpoint')
            ax.set_ylabel(title)
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'polytope_evolution_legacy.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_layer_wise_evolution(self, df: pd.DataFrame, output_path: Path) -> None:
        """Plot layer-wise evolution with separate plots per layer."""
        layers = sorted(df['layer'].unique())
        metrics = [
            ('density_mean', 'Polytope Density'),
            ('participation_ratio', 'Participation Ratio'), 
            ('sparsity', 'Sparsity'),
            ('activation_norm', 'Activation Norm'),
            ('pattern_reuse_rate', 'Pattern Reuse Rate')
        ]
        
        # Create layer-wise plots
        for layer in layers:
            layer_df = df[df['layer'] == layer]
            if layer_df.empty:
                continue
                
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle(f'Polytope Superposition Analysis - Layer {layer}', fontsize=16)
            
            for idx, (metric, title) in enumerate(metrics):
                if idx < 6:  # We have 6 subplots (2x3)
                    ax = axes[idx // 3, idx % 3]
                    
                    for group in ['high_freq', 'low_freq']:
                        group_data = layer_df[layer_df['group'] == group]
                        if not group_data.empty:
                            # Use different markers and colors for clarity
                            color = '#1f77b4' if group == 'high_freq' else '#ff7f0e'  # Blue for high, orange for low
                            marker = 'o' if group == 'high_freq' else 's'  # Circle for high, square for low
                            ax.plot(group_data['checkpoint'], group_data[metric], 
                                   marker=marker, color=color, linewidth=2, markersize=8,
                                   label=group.replace('_', ' ').title() + ' N-grams')
                    
                    ax.set_xlabel('Training Checkpoint', fontsize=12)
                    ax.set_ylabel(title, fontsize=12)
                    ax.set_title(f'{title} (Layer {layer})', fontsize=12, fontweight='bold')
                    ax.legend(fontsize=10)
                    ax.grid(True, alpha=0.3)
                    
                    # Add trend annotations for key metrics
                    if metric == 'density_mean':
                        ax.text(0.02, 0.98, 'Higher = More Polytope Structure', 
                               transform=ax.transAxes, fontsize=8, alpha=0.7, verticalalignment='top')
                    elif metric == 'pattern_reuse_rate':
                        ax.text(0.02, 0.98, 'Higher = More Superposition', 
                               transform=ax.transAxes, fontsize=8, alpha=0.7, verticalalignment='top')
            
            # Remove empty subplot
            if len(metrics) < 6:
                fig.delaxes(axes[1, 2])
            
            plt.tight_layout()
            plt.savefig(output_path / f'polytope_evolution_layer_{layer}.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # Create cross-layer comparison plots
        self._plot_cross_layer_comparison(df, output_path)
        
        # Create n-gram to polytope mapping analysis plots
        self._plot_ngram_polytope_mapping_analysis(df, output_path)
    
    def _plot_cross_layer_comparison(self, df: pd.DataFrame, output_path: Path) -> None:
        """Create plots comparing metrics across layers."""
        checkpoints = sorted(df['checkpoint'].unique())
        layers = sorted(df['layer'].unique())
        
        # Plot density evolution across layers for each checkpoint
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Cross-Layer Polytope Density Comparison', fontsize=16)
        
        for group_idx, group in enumerate(['high_freq', 'low_freq']):
            ax = axes[group_idx]
            
            for checkpoint in checkpoints:
                checkpoint_data = df[(df['checkpoint'] == checkpoint) & (df['group'] == group)]
                if not checkpoint_data.empty:
                    ax.plot(checkpoint_data['layer'], checkpoint_data['density_mean'], 
                           marker='o', label=f'Checkpoint {checkpoint}')
            
            ax.set_xlabel('Layer')
            ax.set_ylabel('Polytope Density')
            ax.set_title(f'{group.replace("_", " ").title()} N-grams')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'cross_layer_density_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Layer-wise visualizations saved to {output_path}")
    
    def _plot_ngram_polytope_mapping_analysis(self, df: pd.DataFrame, output_path: Path) -> None:
        """Create specialized plots for n-gram to polytope mapping analysis.
        Falls back to placeholders if metrics are unavailable."""

        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('N-gram to Polytope Mapping Analysis', fontsize=16)

        # Helper to know if a column is present with some non-null values
        def has_metric(column: str) -> bool:
            return (column in df.columns) and df[column].notna().any()

        # Panel 1: Polytopes with Multiple N-grams (polysemantic fraction)
        ax = axes[0, 0]
        # Prioritize new mapping metrics over legacy spline metrics
        polysemantic_column = 'mapping_polysemantic_fraction' if has_metric('mapping_polysemantic_fraction') else 'spline_polysemantic_fraction'
        if has_metric(polysemantic_column):
            for group in ['high_freq', 'low_freq']:
                group_df = df[(df['group'] == group) & df[polysemantic_column].notna()]
                if not group_df.empty:
                    series = group_df.groupby('checkpoint')[polysemantic_column].mean()
                    ax.plot(series.index, series.values, marker='o', label=group.replace('_', ' ').title())
            ax.set_xlabel('Training Checkpoint')
            ax.set_ylabel('Polysemantic Fraction')
            ax.set_title('Polytopes with Multiple N-grams')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'Polysemantic Polytope\nEvolution\n(Coming Soon)',
                    ha='center', va='center', fontsize=12, alpha=0.6)
            ax.set_title('Polytopes with Multiple N-grams')
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(True, alpha=0.2)

        # Panel 2: N-grams in Multiple Polytopes (reuse fraction)
        ax = axes[0, 1]
        ngram_reuse_column = 'mapping_ngram_reuse_fraction' if has_metric('mapping_ngram_reuse_fraction') else 'spline_ngram_reuse_fraction'
        if has_metric(ngram_reuse_column):
            for group in ['high_freq', 'low_freq']:
                group_df = df[(df['group'] == group) & df[ngram_reuse_column].notna()]
                if not group_df.empty:
                    series = group_df.groupby('checkpoint')[ngram_reuse_column].mean()
                    ax.plot(series.index, series.values, marker='s', label=group.replace('_', ' ').title())
            ax.set_xlabel('Training Checkpoint')
            ax.set_ylabel('N-gram Reuse Fraction')
            ax.set_title('N-grams in Multiple Polytopes')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'N-gram Reuse\nAnalysis\n(Coming Soon)',
                    ha='center', va='center', fontsize=12, alpha=0.6)
            ax.set_title('N-grams in Multiple Polytopes')
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(True, alpha=0.2)

        # Panel 3: Cross-Frequency Polytope Sharing
        ax = axes[1, 0]
        if has_metric('sharing_fraction'):
            sharing_series = df[df['sharing_fraction'].notna()].groupby('checkpoint')['sharing_fraction'].mean()
            if not sharing_series.empty:
                ax.plot(sharing_series.index, sharing_series.values, marker='^', color='green', label='Cross-group sharing')
            ax.set_xlabel('Training Checkpoint')
            ax.set_ylabel('Sharing Fraction')
            ax.set_title('High/Low Frequency Polytope Overlap')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'Cross-Frequency\nPolytope Sharing\n(Coming Soon)',
                    ha='center', va='center', fontsize=12, alpha=0.6)
            ax.set_title('High/Low Frequency Polytope Overlap')
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(True, alpha=0.2)

        # Panel 4: Mapping Efficiency (avg phrases per polytope)
        ax = axes[1, 1]
        avg_phrases_column = 'mapping_avg_phrases_per_polytope' if has_metric('mapping_avg_phrases_per_polytope') else 'spline_avg_phrases_per_polytope'
        if has_metric(avg_phrases_column):
            for group in ['high_freq', 'low_freq']:
                group_df = df[(df['group'] == group) & df[avg_phrases_column].notna()]
                if not group_df.empty:
                    series = group_df.groupby('checkpoint')[avg_phrases_column].mean()
                    ax.plot(series.index, series.values, marker='d', label=group.replace('_', ' ').title())
            ax.set_xlabel('Training Checkpoint')
            ax.set_ylabel('Avg Phrases per Polytope')
            ax.set_title('Mapping Efficiency Over Training')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'Mapping Quality\nMetrics\n(Coming Soon)',
                    ha='center', va='center', fontsize=12, alpha=0.6)
            ax.set_title('Mapping Efficiency Over Training')
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(True, alpha=0.2)

        plt.tight_layout()
        plt.savefig(output_path / 'ngram_polytope_mapping_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"N-gram polytope mapping visualization saved to {output_path}")
    
    def _plot_layer_wise_interference_and_sharing(self, results: List[Dict[str, Any]], output_path: Path) -> None:
        """Plot interference and sharing evolution for layer-wise analysis."""
        # Interference and polytope sharing evolution
        interference_data = []
        sharing_data = []
        
        for result in results:
            checkpoint = int(result['checkpoint_step'])
            
            if 'interference' in result:
                interference_data.append({
                    'checkpoint': checkpoint,
                    'cosine_similarity': result['interference']['cosine_similarity'],
                    'mann_whitney_pvalue': result['interference']['mann_whitney_pvalue'],
                    'neuron_overlap_score': result['interference'].get('neuron_overlap_score', 0.0)
                })
            
            if 'polytope_sharing' in result:
                sharing_data.append({
                    'checkpoint': checkpoint,
                    'sharing_fraction': result['polytope_sharing']['sharing_fraction'],
                    'shared_polytope_count': result['polytope_sharing']['shared_polytope_count']
                })
        
        if interference_data:
            df_interference = pd.DataFrame(interference_data)
            
            n_plots = 3 if sharing_data else 2
            fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 5))
            if n_plots == 1:
                axes = [axes]
            fig.suptitle('Interference and Polytope Sharing Evolution', fontsize=14)
            
            # Plot 1: Cosine similarity
            axes[0].plot(df_interference['checkpoint'], df_interference['cosine_similarity'], 'o-', color='blue')
            axes[0].set_xlabel('Training Checkpoint')
            axes[0].set_ylabel('Cosine Similarity')
            axes[0].set_title('Inter-group Mean Vector Similarity')
            axes[0].grid(True, alpha=0.3)
            
            # Plot 2: Statistical significance
            axes[1].semilogy(df_interference['checkpoint'], df_interference['mann_whitney_pvalue'], 'o-', color='orange')
            axes[1].axhline(y=0.05, color='r', linestyle='--', alpha=0.7, label='p=0.05')
            axes[1].set_xlabel('Training Checkpoint')
            axes[1].set_ylabel('Mann-Whitney p-value')
            axes[1].set_title('Statistical Difference Between Groups')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            # Plot 3: Polytope sharing (if available)
            if sharing_data and len(axes) > 2:
                df_sharing = pd.DataFrame(sharing_data)
                axes[2].plot(df_sharing['checkpoint'], df_sharing['sharing_fraction'], 'o-', color='green')
                axes[2].set_xlabel('Training Checkpoint')
                axes[2].set_ylabel('Polytope Sharing Fraction')
                axes[2].set_title('Cross-Group Polytope Sharing')
                axes[2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_path / 'interference_evolution.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info(f"Visualizations saved to {output_path}")

    def run_full_analysis(self, checkpoint_path: str, output_dir: str = "cache/polytope_results", 
                         use_multiprocessing: bool = True, max_workers: int = None) -> str:
        """Run complete polytope analysis pipeline with optional multiprocessing."""
        logger.info("Starting full polytope analysis pipeline")
        
        # Load data
        data = self.load_checkpoint_data(checkpoint_path)
        records = data['records']
        
        # Organize by checkpoint
        checkpoint_groups = {}
        for record in records:
            checkpoint = str(record['checkpoint_step'])
            if checkpoint not in checkpoint_groups:
                checkpoint_groups[checkpoint] = []
            checkpoint_groups[checkpoint].append(record)
        
        logger.info(f"Found {len(checkpoint_groups)} checkpoints")
        
        # Analyze each checkpoint (with optional multiprocessing)
        if use_multiprocessing and len(checkpoint_groups) > 1:
            results = self._analyze_checkpoints_parallel(checkpoint_groups, max_workers)
        else:
            results = self._analyze_checkpoints_sequential(checkpoint_groups)
        
        # Save results
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        results_file = output_path / 'polytope_analysis_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Create visualizations
        self.create_evolution_visualization(results, output_dir)
        
        # Generate summary report
        summary = self.generate_summary_report(results)
        report_file = output_path / 'analysis_summary.txt'
        with open(report_file, 'w') as f:
            f.write(summary)
        
        logger.info(f"Analysis complete. Results saved to {output_path}")
        return str(output_path)
    
    def _analyze_checkpoints_sequential(self, checkpoint_groups: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Analyze checkpoints sequentially (original method)."""
        results = []
        for checkpoint in sorted(checkpoint_groups.keys(), key=lambda x: int(x)):
            checkpoint_records = checkpoint_groups[checkpoint]
            result = self.analyze_checkpoint(checkpoint_records, checkpoint)
            results.append(result)
        return results
    
    def _analyze_checkpoints_parallel(self, checkpoint_groups: Dict[str, List[Dict[str, Any]]], 
                                    max_workers: int = None) -> List[Dict[str, Any]]:
        """Analyze checkpoints in parallel using multiprocessing."""
        if max_workers is None:
            # Avoid oversubscription which can cause memory pressure
            max_workers = min(mp.cpu_count(), len(checkpoint_groups), 8)
        
        logger.info(f"Using {max_workers} parallel workers for checkpoint analysis")
        
        # Prepare checkpoint data for parallel processing
        checkpoint_items = [(checkpoint, records) for checkpoint, records in checkpoint_groups.items()]
        checkpoint_items.sort(key=lambda x: int(x[0]))  # Sort by checkpoint number
        
        # Use multiprocessing to analyze checkpoints in parallel
        try:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Submit all checkpoint analysis tasks with layer-wise analysis
                futures = [executor.submit(analyze_checkpoint_worker, records, checkpoint, self.random_seed, True) 
                          for checkpoint, records in checkpoint_items]
                
                # Collect results in original order
                results = []
                for i, future in enumerate(futures):
                    try:
                        result = future.result(timeout=self.per_checkpoint_timeout)
                        results.append(result)
                        logger.info(f"Completed checkpoint {checkpoint_items[i][0]}")
                    except Exception as e:
                        # Log with traceback and error type for better diagnostics
                        logger.exception(f"Error processing checkpoint {checkpoint_items[i][0]}: {e.__class__.__name__}: {str(e)}")
                        err_msg = f"{e.__class__.__name__}: {str(e) or 'No error message provided'}"
                        results.append({'error': err_msg, 'checkpoint_step': checkpoint_items[i][0]})
                
                return results
                
        except Exception as e:
            logger.error(f"Multiprocessing failed, falling back to sequential: {str(e)}")
            return self._analyze_checkpoints_sequential(checkpoint_groups)

    def generate_summary_report(self, results: List[Dict[str, Any]]) -> str:
        """Generate a summary report supporting both legacy and layer-wise analysis."""
        valid_results = [r for r in results if 'error' not in r]
        
        if not valid_results:
            return "No valid results to summarize."
        
        # Check if we have layer-wise results
        has_layer_wise = any(r.get('analysis_type') == 'layer_wise' for r in valid_results)
        
        if has_layer_wise:
            return self._generate_layer_wise_report(valid_results)
        else:
            return self._generate_legacy_report(valid_results)
    
    def _generate_legacy_report(self, valid_results: List[Dict[str, Any]]) -> str:
        """Generate legacy report (layer-averaged analysis)."""
        report = "POLYTOPE SUPERPOSITION ANALYSIS SUMMARY (LEGACY - LAYER AVERAGED)\n"
        report += "=" * 70 + "\n\n"
        
        report += f"Analysis completed for {len(valid_results)} checkpoints\n"
        report += f"Checkpoints analyzed: {[r['checkpoint_step'] for r in valid_results]}\n"
        report += "⚠️  WARNING: This analysis averages metrics across layers, potentially masking layer-specific effects\n\n"
        
        # Rest of legacy report logic...
        high_densities = [r['high_freq_density']['density_mean'] for r in valid_results if 'high_freq_density' in r]
        low_densities = [r['low_freq_density']['density_mean'] for r in valid_results if 'low_freq_density' in r]
        
        if high_densities and low_densities:
            report += "POLYTOPE DENSITY COMPARISON:\n"
            report += f"High frequency - Mean: {np.mean(high_densities):.4f}, Std: {np.std(high_densities):.4f}\n"
            report += f"Low frequency - Mean: {np.mean(low_densities):.4f}, Std: {np.std(low_densities):.4f}\n"
            
            if len(high_densities) > 1 and len(low_densities) > 1:
                statistic, p_value = mannwhitneyu(high_densities, low_densities)
                report += f"Mann-Whitney U test: statistic={statistic:.2f}, p-value={p_value:.6f}\n"
        
        report += "\n" + "=" * 70 + "\n"
        report += "RECOMMENDATION: Use layer-wise analysis for better insights\n"
        
        return report
    
    def _generate_layer_wise_report(self, valid_results: List[Dict[str, Any]]) -> str:
        """Generate comprehensive layer-wise analysis report."""
        report = "POLYTOPE SUPERPOSITION ANALYSIS SUMMARY (LAYER-WISE)\n"
        report += "=" * 60 + "\n\n"
        
        # Collect all layers and checkpoints
        all_layers = set()
        all_checkpoints = set()
        
        for result in valid_results:
            all_checkpoints.add(result['checkpoint_step'])
            if 'layer_results' in result:
                all_layers.update(result['layer_results'].keys())
        
        sorted_layers = sorted(all_layers)
        sorted_checkpoints = sorted(all_checkpoints, key=int)
        
        report += f"Analysis completed for {len(valid_results)} checkpoints across {len(sorted_layers)} layers\n"
        report += f"Checkpoints: {sorted_checkpoints}\n"
        report += f"Layers: {sorted_layers}\n\n"
        
        # Layer-wise density analysis
        layer_density_stats = {}
        for layer in sorted_layers:
            high_densities = []
            low_densities = []
            pattern_reuse_high = []
            pattern_reuse_low = []
            
            for result in valid_results:
                layer_data = result.get('layer_results', {}).get(layer, {})
                if 'error' not in layer_data:
                    high_density = layer_data.get('high_freq_density', {})
                    low_density = layer_data.get('low_freq_density', {})
                    
                    if 'density_mean' in high_density:
                        high_densities.append(high_density['density_mean'])
                    if 'density_mean' in low_density:
                        low_densities.append(low_density['density_mean'])
                    
                    pattern_reuse_high.append(layer_data.get('high_freq_pattern_reuse_rate', 0))
                    pattern_reuse_low.append(layer_data.get('low_freq_pattern_reuse_rate', 0))
            
            if high_densities and low_densities:
                layer_density_stats[layer] = {
                    'high_mean': np.mean(high_densities),
                    'low_mean': np.mean(low_densities),
                    'high_std': np.std(high_densities),
                    'low_std': np.std(low_densities),
                    'pattern_reuse_high': np.mean(pattern_reuse_high),
                    'pattern_reuse_low': np.mean(pattern_reuse_low)
                }
        
        # Layer-by-layer analysis
        report += "LAYER-WISE POLYTOPE ANALYSIS:\n"
        report += "-" * 40 + "\n"
        
        for layer in sorted_layers:
            if layer in layer_density_stats:
                stats = layer_density_stats[layer]
                report += f"\nLayer {layer}:\n"
                report += f"  High freq density: {stats['high_mean']:.4f} ± {stats['high_std']:.4f}\n"
                report += f"  Low freq density:  {stats['low_mean']:.4f} ± {stats['low_std']:.4f}\n"
                report += f"  Pattern reuse (H/L): {stats['pattern_reuse_high']:.3f} / {stats['pattern_reuse_low']:.3f}\n"
                
                density_diff = stats['high_mean'] - stats['low_mean']
                if abs(density_diff) > 0.05:
                    trend = "Higher" if density_diff > 0 else "Lower"
                    report += f"  → {trend} density for high-freq n-grams (Δ = {density_diff:+.4f})\n"
        
        # Cross-layer patterns
        report += "\n\nCROSS-LAYER PATTERNS:\n"
        report += "-" * 40 + "\n"
        
        if len(sorted_layers) > 1:
            # Identify layers with highest/lowest density differences
            def layer_diff_fn(layer_id):
                return abs(layer_density_stats[layer_id]['high_mean'] - layer_density_stats[layer_id]['low_mean'])
            
            max_diff_layer = max(layer_density_stats.keys(), key=layer_diff_fn)
            min_diff_layer = min(layer_density_stats.keys(), key=layer_diff_fn)
            
            report += f"Largest frequency effect: Layer {max_diff_layer}\n"
            report += f"Smallest frequency effect: Layer {min_diff_layer}\n"
            
            # Trend analysis
            early_layer = min(sorted_layers)
            late_layer = max(sorted_layers)
            
            if early_layer in layer_density_stats and late_layer in layer_density_stats:
                early_diff = layer_density_stats[early_layer]['high_mean'] - layer_density_stats[early_layer]['low_mean']
                late_diff = layer_density_stats[late_layer]['high_mean'] - layer_density_stats[late_layer]['low_mean']
                
                if abs(late_diff) > abs(early_diff):
                    report += "Frequency effects INCREASE with layer depth\n"
                else:
                    report += "Frequency effects DECREASE with layer depth\n"
        
        # Pattern reuse insights
        avg_reuse_high = np.mean([stats['pattern_reuse_high'] for stats in layer_density_stats.values()])
        avg_reuse_low = np.mean([stats['pattern_reuse_low'] for stats in layer_density_stats.values()])
        
        report += "\nPATTERN REUSE ANALYSIS:\n"
        report += "-" * 40 + "\n"
        report += f"Average pattern reuse (high freq): {avg_reuse_high:.4f}\n"
        report += f"Average pattern reuse (low freq):  {avg_reuse_low:.4f}\n"
        
        if avg_reuse_high > 0.1 or avg_reuse_low > 0.1:
            report += "✓ Significant pattern reuse detected - evidence of polytope sharing\n"
        else:
            report += "⚠️ Low pattern reuse - may indicate insufficient data or specialized representations\n"
        
        # Research implications
        report += "\nRESEARCH IMPLICATIONS:\n"
        report += "-" * 40 + "\n"
        
        report += "• Layer-wise analysis reveals differential superposition across network depth\n"
        if len(sorted_layers) > 1:
            report += "• Frequency-based polytope differences vary by layer position\n"
        report += "• Spline code analysis provides cleaner polytope boundaries than CETT thresholding\n"
        
        report += "\n" + "=" * 60 + "\n"
        report += "LAYER-WISE POLYTOPE ANALYSIS COMPLETED\n"
        report += "Enhanced insights: per-layer metrics, cross-layer comparisons, pattern reuse tracking\n"
        
        return report


def analyze_checkpoint_worker(records: List[Dict[str, Any]], checkpoint_step: str, random_seed: int, 
                             use_layer_wise_analysis: bool = True) -> Dict[str, Any]:
    """
    Worker function for multiprocessing checkpoint analysis.
    
    This function creates a temporary analyzer instance to avoid pickling issues
    with the main analyzer object's logger and other non-serializable components.
    """
    try:
        # Create a minimal analyzer for this worker
        temp_analyzer = RefinedPolytopeAnalyzer(random_seed=random_seed, cache_dir="cache/temp_worker")
        
        # Analyze the checkpoint with layer-wise analysis by default
        result = temp_analyzer.analyze_checkpoint(records, checkpoint_step, use_layer_wise_analysis)
        
        # Clean up temp analyzer resources
        del temp_analyzer
        gc.collect()
        
        return result
        
    except Exception as e:
        return {
            'error': f"Worker failed for checkpoint {checkpoint_step}: {str(e)}",
            'checkpoint_step': checkpoint_step
        }


def run_polytope_analysis_pipeline(checkpoint_file: str, output_dir: str = "cache/refined_polytope_results") -> str:
    """
    Main entry point for running the refined polytope analysis pipeline.
    
    Args:
        checkpoint_file: Path to checkpoint analysis results (pickle file)
        output_dir: Output directory for results and visualizations
        
    Returns:
        Path to results directory
    """
    analyzer = RefinedPolytopeAnalyzer()
    return analyzer.run_full_analysis(checkpoint_file, output_dir)


def render_visuals_from_results_json(results_json: str, output_dir: str = "cache/refined_polytope_results_rerun_70m") -> str:
    """Render visualizations and summary from a previously saved results JSON.

    Args:
        results_json: Path to a JSON file produced by run_full_analysis
        output_dir: Output directory for figures and summary

    Returns:
        Path to the output directory containing the generated artifacts
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with open(results_json, 'r') as f:
        results = json.load(f)

    analyzer = RefinedPolytopeAnalyzer()
    # Reuse the same visualization/report pipeline without recomputing metrics
    analyzer.create_evolution_visualization(results, output_dir)
    summary = analyzer.generate_summary_report(results)
    report_file = output_path / 'analysis_summary_from_json.txt'
    with open(report_file, 'w') as f:
        f.write(summary)

    logger.info(f"Rendered visualizations from JSON to {output_path}")
    return str(output_path)

if __name__ == "__main__":
    # Example usage
    checkpoint_file = "/workspace/cache/activation_records_70m.pkl"
    results_dir = run_polytope_analysis_pipeline(checkpoint_file)
    print(f"Analysis complete. Results saved to: {results_dir}")

