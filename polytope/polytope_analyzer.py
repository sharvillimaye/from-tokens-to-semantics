#!/usr/bin/env python3
"""
Simplified Polytope Superposition Analyzer for LLM Activations

This module implements a focused analysis pipeline for studying superposition
phenomena in LLM activations, comparing high vs low frequency n-gram patterns.

CORE ANALYSIS:
==============
1. Binary Pattern Extraction using CETT thresholding
2. Polytope Density Analysis using Hamming vs Euclidean distance ratios
3. Superposition Measurement via interference patterns
4. Direct Frequency Group Comparison

DEPENDENCIES:
============
pip install numpy scipy scikit-learn pandas matplotlib

USAGE:
======
from polytope.polytope_analyzer import SuperpositionAnalyzer

# Initialize analyzer
analyzer = SuperpositionAnalyzer(random_seed=42)

# Run analysis comparing high vs low frequency activations
results = analyzer.analyze_polytope_superposition(
    activations_high_freq=high_freq_data,  # Shape: (n_samples_high, n_neurons)
    activations_low_freq=low_freq_data,    # Shape: (n_samples_low, n_neurons)
    semantic_category="n_grams"
)

# Generate simple report
report = analyzer.generate_analysis_report(results)
print(report)

EXPECTED OUTPUTS:
================
- Quantitative superposition strength differences between frequency groups
- Geometric polytope structure characterization
- Phase classification (strong vs weak superposition)
- Direct comparison metrics between high and low frequency patterns
"""

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import pickle
from typing import Any, Dict, List, Optional, Tuple, cast
import warnings

from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import mannwhitneyu
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

try:
    from hdbscan import HDBSCAN  # optional, used only if clustering function is called
except Exception:
    HDBSCAN = None  # type: ignore
try:
    from sklearn.metrics import silhouette_score  # optional
except Exception:
    silhouette_score = None  # type: ignore

# Suppress sklearn warnings
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module=r"sklearn\.utils\.deprecation"
)

class SuperpositionAnalyzer:
    """
    Simplified superposition analysis pipeline for LLM activations based on n-gram frequency.
    
    Implements focused analysis:
    1. Binary pattern extraction using CETT thresholding
    2. Polytope density calculations  
    3. Superposition interference measurement
    4. Direct frequency group comparison
    
    Usage:
        analyzer = SuperpositionAnalyzer()
        results = analyzer.analyze_polytope_superposition(high_freq_activations, low_freq_activations)
    """
    
    def __init__(self, random_seed: Optional[int] = 42):
        """
        Initialize simplified superposition analyzer.
        
        Args:
            random_seed: Random seed for reproducibility
        """
        self.random_seed = random_seed
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Initialize random number generator for consistent sampling
        self.rng = np.random.default_rng(random_seed)
    
    def _validate_inputs(self, activations: np.ndarray, binary_patterns: Optional[np.ndarray] = None, 
                        spline_codes: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """
        Comprehensive input validation with detailed error messages.
        
        Args:
            activations: Activation vectors
            binary_patterns: Binary patterns (optional)
            spline_codes: Spline codes (optional)
        
        Raises:
            ValueError: If inputs are invalid
        """
        # Validate activations
        if not isinstance(activations, np.ndarray):
            raise ValueError("Activations must be a numpy array")
        
        if activations.ndim != 2:
            raise ValueError(f"Activations must be 2D array, got shape {activations.shape}")
        
        if activations.shape[0] < 2:
            raise ValueError(f"Need at least 2 samples, got {activations.shape[0]}")
        
        # Check for NaN/infinite values
        if not np.isfinite(activations).all():
            raise ValueError("Activations contain NaN or infinite values")
        
        # Validate binary patterns if provided
        if binary_patterns is not None:
            if not isinstance(binary_patterns, np.ndarray):
                raise ValueError("Binary patterns must be a numpy array")
            
            # Handle dimension mismatch by truncating binary patterns to match activations
            if binary_patterns.shape != activations.shape:
                n_samples, n_features = activations.shape
                if binary_patterns.shape[0] != n_samples:
                    raise ValueError(f"Binary patterns must have same number of samples as activations ({n_samples})")
                
                # Truncate to match activation dimensions (common when spline_code has more dims)
                if binary_patterns.shape[1] > n_features:
                    logger.info(f"Truncating binary patterns from {binary_patterns.shape[1]} to {n_features} dimensions")
                    binary_patterns = binary_patterns[:, :n_features]
                elif binary_patterns.shape[1] < n_features:
                    raise ValueError(f"Binary patterns have too few dimensions: {binary_patterns.shape[1]} < {n_features}")
            
            # Check if binary patterns are actually binary
            unique_values = np.unique(binary_patterns)
            if not np.array_equal(unique_values, np.array([0, 1])) and not np.all(np.isin(unique_values, [0, 1])):
                logger.warning(f"Binary patterns contain non-binary values: {unique_values}")
        
        # Validate spline codes if provided
        if spline_codes is not None:
            if not isinstance(spline_codes, np.ndarray):
                raise ValueError("Spline codes must be a numpy array")
            
            if len(spline_codes) != len(activations):
                raise ValueError(f"Spline codes length {len(spline_codes)} must match activations length {len(activations)}")
            
            if spline_codes.ndim != 2:
                raise ValueError(f"Spline codes must be 2D array, got shape {spline_codes.shape}")
            
            # Check if spline codes are binary
            unique_values = np.unique(spline_codes)
            if not np.all(np.isin(unique_values, [0, 1])):
                logger.warning(f"Spline codes contain non-binary values: {unique_values}")
        
        # Return processed binary patterns (may have been truncated)
        return binary_patterns
    
    def _safe_correlation(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Safely compute correlation coefficient with proper error handling.
        
        Args:
            x, y: Input arrays
            
        Returns:
            Correlation coefficient or 0.0 if computation fails
        """
        try:
            if len(x) < 2 or len(y) < 2:
                return 0.0
            
            # Check for constant arrays (zero variance)
            if np.var(x) == 0 or np.var(y) == 0:
                return 0.0
            
            correlation_matrix = np.corrcoef(x, y)
            if np.isfinite(correlation_matrix).all():
                return float(correlation_matrix[0, 1])
            else:
                return 0.0
                
        except Exception as e:
            logger.warning(f"Correlation computation failed: {e}")
            return 0.0
    
    def _compute_simple_stats(self, data: np.ndarray) -> Dict[str, float]:
        """
        Compute simple descriptive statistics replacing bootstrap complexity.
        
        Args:
            data: Input data array
            
        Returns:
            Dict with mean and standard error
        """
        if len(data) < 2:
            return {'mean': 0.0, 'std_error': 0.0}
        
        mean_val = float(np.mean(data))
        std_error = float(np.std(data) / np.sqrt(len(data)))
        
        return {'mean': mean_val, 'std_error': std_error}

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
        # Validate inputs and get potentially truncated binary patterns
        binary_patterns = self._validate_inputs(activations, binary_patterns)
        
        n_samples = len(activations)

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
        
        # Compute simple descriptive statistics (removed bootstrap complexity)
        density_stats = self._compute_simple_stats(densities)
        
        return {
            'polytope_densities': densities,
            'mean_density': density_stats['mean'],
            'mean_density_std_error': density_stats['std_error'],
            'density_std': float(np.std(densities)),
            'boundary_crossings': boundary_crossings,
            'boundary_crossing_rate': float(boundary_crossings) / float(len(euc_valid)),
            'density_percentiles': {
                '25th': float(np.percentile(densities, 25)),
                '75th': float(np.percentile(densities, 75)),
                '90th': float(np.percentile(densities, 90))
            }
        }
    
    
    def analyze_ngram_polytope_mapping(self, records: List[Dict[str, Any]], 
                                     spline_codes: np.ndarray) -> Dict[str, Any]:
        """
        Map n-grams directly to polytopes to discover emergent patterns.
        
        This analysis reveals:
        1. Which n-grams share the same polytopes (potential polysemanticity)
        2. Frequency-based polytope occupation patterns
        3. N-gram diversity within polytopes
        
        Args:
            records: List of activation records with n-gram metadata
            spline_codes: Binary spline codes for polytope identification
            
        Returns:
            Dictionary with n-gram to polytope mapping analysis
        """
        if len(records) != len(spline_codes):
            raise ValueError("Records and spline codes must have same length")
        
        # Create polytope identifiers from spline codes
        polytope_to_ngrams = defaultdict(list)
        ngram_to_polytopes = defaultdict(set)
        frequency_bins = defaultdict(lambda: defaultdict(list))
        
        for idx, record in enumerate(records):
            # Get n-gram identifier
            ngram = record.get('phrase', record.get('ngram', record.get('text', f'sample_{idx}')))
            
            # Get frequency category
            freq_category = record.get('category', record.get('frequency_bin', 'unknown'))
            
            # Create polytope ID from spline code (fix collision bug)
            polytope_id = tuple(spline_codes[idx].astype(int))
            
            # Map n-gram to polytope
            polytope_to_ngrams[polytope_id].append({
                'ngram': ngram,
                'frequency_category': freq_category,
                'record_index': idx,
                'activation_vector': record.get('activation_vector'),
                'spline_code': spline_codes[idx]
            })
            
            ngram_to_polytopes[ngram].add(polytope_id)
            frequency_bins[freq_category][polytope_id].append(ngram)
        
        # Analyze polytope sharing patterns
        polysemantic_polytopes = {}  # Polytopes containing multiple different n-grams
        monosemantic_polytopes = {}  # Polytopes containing only one type of n-gram
        
        for polytope_id, items in polytope_to_ngrams.items():
            unique_ngrams = list(set(item['ngram'] for item in items))
            
            if len(unique_ngrams) == 1:
                monosemantic_polytopes[polytope_id] = {
                    'ngram': unique_ngrams[0],
                    'count': len(items),
                    'frequency_categories': list(set(item['frequency_category'] for item in items))
                }
            else:
                polysemantic_polytopes[polytope_id] = {
                    'ngrams': unique_ngrams,
                    'ngram_counts': {ngram: sum(1 for item in items if item['ngram'] == ngram) 
                                   for ngram in unique_ngrams},
                    'total_samples': len(items),
                    'frequency_mix': list(set(item['frequency_category'] for item in items))
                }
        
        # Analyze n-gram distribution across polytopes
        multi_polytope_ngrams = {ngram: list(polytopes) for ngram, polytopes in ngram_to_polytopes.items() 
                               if len(polytopes) > 1}
        
        # Frequency-based polytope occupation analysis
        frequency_polytope_stats = {}
        for freq_cat, polytope_dict in frequency_bins.items():
            frequency_polytope_stats[freq_cat] = {
                'unique_polytopes': len(polytope_dict),
                'total_samples': sum(len(ngrams) for ngrams in polytope_dict.values()),
                'polytope_sharing': sum(1 for ngrams in polytope_dict.values() if len(set(ngrams)) > 1),
                'avg_ngrams_per_polytope': float(np.mean([len(set(ngrams)) for ngrams in polytope_dict.values()]))
            }
        
        return {
            'polytope_to_ngrams': dict(polytope_to_ngrams),
            'ngram_to_polytopes': dict(ngram_to_polytopes),
            'polysemantic_polytopes': polysemantic_polytopes,
            'monosemantic_polytopes': monosemantic_polytopes,
            'multi_polytope_ngrams': multi_polytope_ngrams,
            'frequency_polytope_stats': frequency_polytope_stats,
            'summary_stats': {
                'total_polytopes': len(polytope_to_ngrams),
                'total_unique_ngrams': len(ngram_to_polytopes),
                'polysemantic_polytope_count': len(polysemantic_polytopes),
                'monosemantic_polytope_count': len(monosemantic_polytopes),
                'polysemantic_fraction': len(polysemantic_polytopes) / len(polytope_to_ngrams) if polytope_to_ngrams else 0.0,
                'ngrams_sharing_polytopes': len(multi_polytope_ngrams),
                'avg_polytopes_per_ngram': float(np.mean([len(polytopes) for polytopes in ngram_to_polytopes.values()])) if ngram_to_polytopes else 0.0
            }
        }
    
    def test_polytope_monosemanticity(self, records: List[Dict[str, Any]], 
                                    spline_codes: np.ndarray) -> Dict[str, Any]:
        """
        Test if samples with identical spline codes are semantically coherent.
        
        Core polytope lens prediction: Samples in the same polytope should
        undergo similar network transformations and represent similar concepts.
        
        Args:
            records: List of activation records with semantic metadata
            spline_codes: Binary spline codes for polytope identification
            
        Returns:
            Dictionary with monosemanticity analysis results
        """
        if len(records) != len(spline_codes):
            raise ValueError("Records and spline codes must have same length")
        
        # Group samples by identical spline codes (same polytope)
        polytope_groups = defaultdict(list)
        for idx, record in enumerate(records):
            code_hash = tuple(spline_codes[idx].astype(int))
            polytope_groups[code_hash].append({
                'index': idx,
                'record': record,
                'ngram': record.get('phrase', record.get('ngram', record.get('text', f'sample_{idx}'))),
                'frequency_category': record.get('category', record.get('frequency_bin', 'unknown')),
                'spline_code': spline_codes[idx]
            })
        
        monosemantic_polytopes = 0
        polysemantic_polytopes = 0
        polytope_analysis = {}
        
        for polytope_id, group in polytope_groups.items():
            if len(group) < 2:  # Skip single-sample polytopes
                continue
                
            # Extract n-grams in this polytope
            ngrams_in_polytope = [item['ngram'] for item in group]
            unique_ngrams = list(set(ngrams_in_polytope))
            
            # Extract frequency categories
            freq_categories = [item['frequency_category'] for item in group]
            unique_freq_categories = list(set(freq_categories))
            
            is_monosemantic = len(unique_ngrams) == 1
            
            polytope_analysis[polytope_id] = {
                'sample_count': len(group),
                'unique_ngrams': unique_ngrams,
                'unique_ngram_count': len(unique_ngrams),
                'frequency_categories': unique_freq_categories,
                'is_monosemantic': is_monosemantic,
                'ngram_distribution': {ngram: ngrams_in_polytope.count(ngram) for ngram in unique_ngrams},
                'frequency_distribution': {cat: freq_categories.count(cat) for cat in unique_freq_categories}
            }
            
            if is_monosemantic:
                monosemantic_polytopes += 1
            else:
                polysemantic_polytopes += 1
        
        total_multi_sample_polytopes = monosemantic_polytopes + polysemantic_polytopes
        
        # Analyze frequency-based polytope sharing patterns
        frequency_sharing_analysis = {}
        for freq_cat in ['high_freq', 'low_freq', 'mid_freq']:
            freq_polytopes = [pid for pid, analysis in polytope_analysis.items() 
                            if freq_cat in analysis['frequency_categories']]
            
            mixed_freq_polytopes = [pid for pid in freq_polytopes 
                                  if len(polytope_analysis[pid]['frequency_categories']) > 1]
            
            frequency_sharing_analysis[freq_cat] = {
                'total_polytopes': len(freq_polytopes),
                'mixed_frequency_polytopes': len(mixed_freq_polytopes),
                'pure_frequency_fraction': (len(freq_polytopes) - len(mixed_freq_polytopes)) / len(freq_polytopes) if freq_polytopes else 0.0
            }
        
        return {
            'polytope_analysis': polytope_analysis,
            'monosemantic_polytope_count': monosemantic_polytopes,
            'polysemantic_polytope_count': polysemantic_polytopes,
            'total_multi_sample_polytopes': total_multi_sample_polytopes,
            'monosemantic_fraction': monosemantic_polytopes / total_multi_sample_polytopes if total_multi_sample_polytopes > 0 else 0,
            'polysemantic_fraction': polysemantic_polytopes / total_multi_sample_polytopes if total_multi_sample_polytopes > 0 else 0,
            'frequency_sharing_analysis': frequency_sharing_analysis,
            'summary_stats': {
                'avg_samples_per_polytope': float(np.mean([len(group) for group in polytope_groups.values() if len(group) >= 2])) if polytope_groups else 0.0,
                'max_samples_per_polytope': max(len(group) for group in polytope_groups.values()) if polytope_groups else 0,
                'avg_ngrams_per_polytope': float(np.mean([analysis['unique_ngram_count'] for analysis in polytope_analysis.values()])) if polytope_analysis else 0.0
            }
        }
    
    def analyze_frequency_polytope_relationship(self, records: List[Dict[str, Any]], 
                                              spline_codes: np.ndarray) -> Dict[str, Any]:
        """
        Test the core hypothesis: frequency affects polytope structure and sharing patterns.
        
        Analyzes whether mid-frequency n-grams show higher polytope sharing compared
        to high/low frequency n-grams, supporting polysemanticity hypothesis.
        
        Args:
            records: List of activation records with frequency metadata
            spline_codes: Binary spline codes for polytope identification
            
        Returns:
            Dictionary with frequency-polytope relationship analysis
        """
        if len(records) != len(spline_codes):
            raise ValueError("Records and spline codes must have same length")
        
        # Organize by frequency bins
        frequency_bins = {'low_freq': [], 'mid_freq': [], 'high_freq': []}
        
        for idx, record in enumerate(records):
            category = record.get('category', record.get('frequency_bin', 'unknown'))
            # Normalize category names
            if category.lower() in ['low_freq', 'low_frequency', 'low']:
                bin_name = 'low_freq'
            elif category.lower() in ['mid_freq', 'mid_frequency', 'medium', 'mid']:
                bin_name = 'mid_freq'  
            elif category.lower() in ['high_freq', 'high_frequency', 'high']:
                bin_name = 'high_freq'
            else:
                continue  # Skip unknown categories
                
            frequency_bins[bin_name].append({
                'index': idx,
                'record': record,
                'spline_code': spline_codes[idx],
                'ngram': record.get('phrase', record.get('ngram', record.get('text', f'sample_{idx}')))
            })
        
        results = {}
        for bin_name, bin_records in frequency_bins.items():
            if not bin_records:
                continue
                
            bin_spline_codes = np.array([item['spline_code'] for item in bin_records])
            
            # Unique polytope count
            unique_codes = np.unique(bin_spline_codes, axis=0)
            unique_polytope_count = len(unique_codes)
            
            # Polytope sharing analysis
            polytope_counts = defaultdict(int)
            polytope_to_ngrams = defaultdict(set)
            
            for item in bin_records:
                code_hash = tuple(item['spline_code'].astype(int))
                polytope_counts[code_hash] += 1
                polytope_to_ngrams[code_hash].add(item['ngram'])
            
            # Calculate sharing metrics
            shared_polytopes = sum(1 for count in polytope_counts.values() if count > 1)
            total_samples = len(bin_records)
            polytope_reuse_rate = shared_polytopes / total_samples if total_samples > 0 else 0
            samples_per_polytope = total_samples / unique_polytope_count if unique_polytope_count > 0 else 0
            
            # N-gram diversity within polytopes
            polysemantic_polytopes = sum(1 for ngrams in polytope_to_ngrams.values() if len(ngrams) > 1)
            ngram_diversity_score = polysemantic_polytopes / len(polytope_to_ngrams) if polytope_to_ngrams else 0
            
            # Boundary crossing analysis for this frequency bin
            boundary_crossings = []
            if len(bin_spline_codes) >= 2:
                for i in range(len(bin_spline_codes)):
                    for j in range(i + 1, min(i + 50, len(bin_spline_codes))):  # Sample to avoid O(n^2)
                        hamming_dist = int(np.sum(bin_spline_codes[i] != bin_spline_codes[j]))
                        if hamming_dist > 0:  # Different polytopes
                            boundary_crossings.append(hamming_dist)
            
            results[bin_name] = {
                'sample_count': total_samples,
                'unique_polytope_count': unique_polytope_count,
                'shared_polytope_count': shared_polytopes,
                'polytope_reuse_rate': polytope_reuse_rate,
                'samples_per_polytope': samples_per_polytope,
                'polysemantic_polytope_count': polysemantic_polytopes,
                'ngram_diversity_score': ngram_diversity_score,
                'avg_boundary_crossings': float(np.mean(boundary_crossings)) if boundary_crossings else 0.0,
                'unique_ngram_count': len(set(item['ngram'] for item in bin_records))
            }
        
        # Test core hypothesis with statistical significance
        hypothesis_tests = {}
        statistical_tests = {}
        
        if all(bin_name in results for bin_name in ['low_freq', 'mid_freq', 'high_freq']):
            # Extract raw data for statistical testing
            sharing_data = {}
            diversity_data = {}
            
            for bin_name in ['low_freq', 'mid_freq', 'high_freq']:
                bin_records = frequency_bins[bin_name]
                if not bin_records:
                    continue
                
                # Compute individual polytope sharing rates for statistical testing
                bin_polytope_counts = defaultdict(int)
                for item in bin_records:
                    code_hash = tuple(item['spline_code'].astype(int))
                    bin_polytope_counts[code_hash] += 1
                
                # Individual sharing indicators (0/1 for each sample)
                sharing_indicators = []
                diversity_scores = []
                
                for item in bin_records:
                    code_hash = tuple(item['spline_code'].astype(int))
                    # Binary: is this sample in a shared polytope?
                    sharing_indicators.append(1 if bin_polytope_counts[code_hash] > 1 else 0)
                    
                    # Add some noise for diversity (placeholder - could be improved with actual diversity metric)
                    diversity_scores.append(np.random.normal(results[bin_name]['ngram_diversity_score'], 0.1))
                
                sharing_data[bin_name] = sharing_indicators
                diversity_data[bin_name] = diversity_scores
            
            # Statistical hypothesis testing
            try:
                # Test 1: Mid-frequency has highest sharing rate
                if all(bin_name in sharing_data for bin_name in ['low_freq', 'mid_freq', 'high_freq']):
                    # Direct pairwise Mann-Whitney U tests (simplified from complex statistical testing)
                    mid_vs_low_stat, mid_vs_low_p = mannwhitneyu(
                        sharing_data['mid_freq'], sharing_data['low_freq'], alternative='greater'
                    )
                    mid_vs_high_stat, mid_vs_high_p = mannwhitneyu(
                        sharing_data['mid_freq'], sharing_data['high_freq'], alternative='greater'
                    )
                    
                    statistical_tests['mid_vs_low_sharing_p'] = float(mid_vs_low_p)
                    statistical_tests['mid_vs_high_sharing_p'] = float(mid_vs_high_p)
                    
                    # Bonferroni correction for multiple comparisons
                    corrected_alpha = 0.05 / 2
                    statistical_tests['bonferroni_corrected_alpha'] = corrected_alpha
                    statistical_tests['mid_freq_significantly_highest'] = (
                        mid_vs_low_p < corrected_alpha and mid_vs_high_p < corrected_alpha
                    )
            
            except Exception as e:
                logger.warning(f"Statistical testing failed: {e}")
                statistical_tests['error'] = str(e)
            
            # Original deterministic tests (now with statistical backing)
            mid_sharing = results['mid_freq']['polytope_reuse_rate']
            low_sharing = results['low_freq']['polytope_reuse_rate']
            high_sharing = results['high_freq']['polytope_reuse_rate']
            
            hypothesis_tests['mid_freq_highest_sharing'] = (
                mid_sharing > low_sharing and mid_sharing > high_sharing
            )
            
            hypothesis_tests['mid_freq_highest_diversity'] = (
                results['mid_freq']['ngram_diversity_score'] > results['low_freq']['ngram_diversity_score'] and
                results['mid_freq']['ngram_diversity_score'] > results['high_freq']['ngram_diversity_score']
            )
            
            # Expected pattern: mid > low > high for polytope sharing
            hypothesis_tests['expected_sharing_pattern'] = (
                results['mid_freq']['samples_per_polytope'] > results['low_freq']['samples_per_polytope'] > results['high_freq']['samples_per_polytope']
            )
        
        return {
            'frequency_bin_results': results,
            'hypothesis_tests': hypothesis_tests,
            'statistical_tests': statistical_tests,
            'cross_frequency_comparison': {
                'polytope_reuse_ranking': sorted(results.keys(), 
                                               key=lambda x: results[x]['polytope_reuse_rate'], 
                                               reverse=True) if results else [],
                'diversity_ranking': sorted(results.keys(),
                                          key=lambda x: results[x]['ngram_diversity_score'],
                                          reverse=True) if results else []
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
    
    def analyze_spline_code_clustering(groups):
        """take in spline codes of each group and cluster"""
        # cluster the spline codes
        if HDBSCAN is None or silhouette_score is None:
            raise ImportError("Clustering requires hdbscan and sklearn.metrics.silhouette_score to be installed.")
        results = {}
        for frequency_group, spline_codes in groups.items():
            clusterer = HDBSCAN(min_cluster_size=5, metric='hamming')
            cluster_labels = clusterer.fit_predict(spline_codes)
            # store the cluster labels
            groups[frequency_group]['cluster_labels'] = cluster_labels
            # Measure clustering quality
            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            
            if n_clusters > 1:
                # Silhouette score for cluster separation
                valid_mask = cluster_labels != -1
                if np.sum(valid_mask) > 1:
                    silhouette = silhouette_score(
                        spline_codes[valid_mask], 
                        cluster_labels[valid_mask], 
                        metric='hamming'
                    )
                else:
                    silhouette = 0
            else:
                silhouette = 0
            
            results[frequency_group] = {
                'n_clusters': n_clusters,
                'silhouette_score': silhouette,
                'noise_fraction': np.mean(cluster_labels == -1),
                'cluster_purity': silhouette  # Higher = better separated
            }
        
        return results
    
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
                   'n_unique_patterns', 'pattern_diversity', 'interference_per_dimension',
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
class MultiCheckpointPolytopeAnalyzer:
    """
    Simplified multi-checkpoint polytope analysis for studying 
    high/low frequency patterns across model training checkpoints.
    """
    
    def __init__(self, random_seed: Optional[int] = 42):
        self.base_analyzer = SuperpositionAnalyzer(random_seed)
        self.random_seed = random_seed
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def load_checkpoint_data(self, checkpoint_file: str) -> Dict[str, Any]:
        """Load checkpoint analysis results from file"""
        with open(checkpoint_file, 'rb') as f:
            data = pickle.load(f)
        # Normalize frequency category field for downstream grouping
        try:
            records = data['records'] if isinstance(data, dict) and 'records' in data else data
            for r in records:
                raw_cat = r.get('category', None)
                if not raw_cat or raw_cat == 'unknown':
                    raw_cat = r.get('frequency_category', r.get('frequencyCategory', None))
                if isinstance(raw_cat, str) and raw_cat:
                    norm = raw_cat.strip().lower()
                    if norm in {'high_frequency', 'highfreq', 'high-freq', 'high'}:
                        r['category'] = 'high_freq'
                    elif norm in {'low_frequency', 'lowfreq', 'low-freq', 'low'}:
                        r['category'] = 'low_freq'
                    elif norm in {'medium_frequency', 'mid_frequency', 'mid', 'medium'}:
                        r['category'] = 'mid_freq'
                    else:
                        r['category'] = raw_cat
        except Exception:
            pass
        return data
    
    def organize_records_by_attributes(self, records: List[Dict[str, Any]]) -> Dict[str, Dict[str, List]]:
        """Organize records by checkpoint, layer, and frequency category"""
        organized = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        
        for record in records:
            checkpoint = record['checkpoint_step']
            layer = record['layer']
            raw_cat = record.get('category', record.get('frequency_category', 'unknown'))
            norm = str(raw_cat).strip().lower()
            if norm in {'high_frequency', 'highfreq', 'high-freq', 'high'}:
                category = 'high_freq'
            elif norm in {'low_frequency', 'lowfreq', 'low-freq', 'low'}:
                category = 'low_freq'
            elif norm in {'medium_frequency', 'mid_frequency', 'mid', 'medium'}:
                category = 'mid_freq'
            else:
                category = raw_cat
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

    def strict_extract_spline_codes(self, records: List[Dict[str, Any]]) -> np.ndarray:
        """Strictly extract spline codes; raise if any record is missing codes."""
        codes = self.extract_spline_codes_from_records(records)
        if codes is None:
            raise ValueError("Strict analysis requires 'spline_code' present for all records.")
        return codes
    
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
        # Strict spline codes from records
        high_codes = self.strict_extract_spline_codes(high_freq_records)
        low_codes = self.strict_extract_spline_codes(low_freq_records)
        
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
            'high_freq_phrases': list(set(r.get('phrase', r.get('phrase')) for r in high_freq_records)),
            'low_freq_phrases': list(set(r.get('phrase', r.get('phrase')) for r in low_freq_records))
        }
        
        return results
    
  

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
        # Attempt to infer model name from record metadata if present
        inferred_model: str = 'Unknown'
        if records:
            r0 = records[0]
            meta = r0.get('metadata') if isinstance(r0, dict) else None
            if isinstance(meta, dict):
                inferred_model = meta.get('model_name', inferred_model)
            # Also try top-level just in case
            if inferred_model == 'Unknown':
                inferred_model = r0.get('model_name', inferred_model)
        metadata = {
            'model_name': inferred_model,
            'checkpoints': sorted(list(checkpoints)),
            'target_layers': sorted(list(layers)),
            'n_total_records': len(records),
            'format': 'plain_list'
        }
        # Normalize frequency category labels on records
        for r in records:
            raw_cat = r.get('category', None)
            if not raw_cat or raw_cat == 'unknown':
                raw_cat = r.get('frequency_category', r.get('frequencyCategory', None))
            if isinstance(raw_cat, str) and raw_cat:
                norm = raw_cat.strip().lower()
                if norm in {'high_frequency', 'highfreq', 'high-freq', 'high'}:
                    r['category'] = 'high_freq'
                elif norm in {'low_frequency', 'lowfreq', 'low-freq', 'low'}:
                    r['category'] = 'low_freq'
                else:
                    r['category'] = raw_cat
        data = {'records': records, 'metadata': metadata}
    elif not (isinstance(data, dict) and 'records' in data and 'metadata' in data):
        raise ValueError("Checkpoint file must be a list of records or a dict with 'records' and 'metadata'")
    else:
        # Normalize categories in dict form as well
        try:
            for r in data['records']:
                raw_cat = r.get('category', None)
                if not raw_cat or raw_cat == 'unknown':
                    raw_cat = r.get('frequency_category', r.get('frequencyCategory', None))
                if isinstance(raw_cat, str) and raw_cat:
                    norm = raw_cat.strip().lower()
                    if norm in {'high_frequency', 'highfreq', 'high-freq', 'high'}:
                        r['category'] = 'high_freq'
                    elif norm in {'low_frequency', 'lowfreq', 'low-freq', 'low'}:
                        r['category'] = 'low_freq'
                    else:
                        r['category'] = raw_cat
        except Exception:
            pass
    return cast(Dict[str, Any], data)


def _organize_records(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Dict[str, List[Dict[str, Any]]]]]:
    organized: Dict[str, Dict[str, Dict[str, List[Dict[str, Any]]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in records:
        # Map various frequency labels into canonical keys used downstream
        raw_cat = r.get('category', r.get('frequency_category', 'unknown'))
        norm = str(raw_cat).strip().lower()
        if norm in {'high_frequency', 'highfreq', 'high-freq', 'high'}:
            cat = 'high_freq'
        elif norm in {'low_frequency', 'lowfreq', 'low-freq', 'low'}:
            cat = 'low_freq'
        elif norm in {'medium_frequency', 'mid_frequency', 'mid', 'medium'}:
            cat = 'mid_freq'
        else:
            cat = raw_cat
        organized[str(r['checkpoint_step'])][str(r['layer'])][cat].append(r)
    return dict(organized)


def _extract_matrices(records: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
    if not records:
        return np.array([]), np.array([])
    acts = np.stack([r['activation_vector'] for r in records])
    bins = np.stack([r['binary_pattern'] for r in records])
    return acts, bins


def _compute_cross_checkpoint_statistics(results: Dict[str, Any]) -> Dict[str, Any]:
    """Compute simple summary statistics across checkpoints"""
    summary = {
        'total_comparisons': 0,
        'mean_superposition_difference': 0.0,
        'mean_density_ratio': 0.0,
        'successful_analyses': 0
    }
    
    superposition_diffs = []
    density_ratios = []
    
    for checkpoint, ckpt_data in results.items():
        if not isinstance(ckpt_data, dict):
            continue
        for layer, layer_data in ckpt_data.items():
            if isinstance(layer_data, dict) and 'comparison' in layer_data:
                comp = layer_data['comparison']
                summary['total_comparisons'] += 1
                summary['successful_analyses'] += 1
                
                if 'superposition_strength_difference' in comp:
                    superposition_diffs.append(comp['superposition_strength_difference'])
                
                if 'mean_density_ratio' in comp:
                    density_ratios.append(comp['mean_density_ratio'])
    
    if superposition_diffs:
        summary['mean_superposition_difference'] = float(np.mean(superposition_diffs))
        summary['std_superposition_difference'] = float(np.std(superposition_diffs))
    
    if density_ratios:
        summary['mean_density_ratio'] = float(np.mean(density_ratios))
        summary['std_density_ratio'] = float(np.std(density_ratios))
    
    return summary


def _analyze_evolution_patterns(results: Dict[str, Any]) -> Dict[str, Any]:
    """Simple analysis of patterns across checkpoints"""
    patterns = {
        'strongest_effects': [],
        'weak_effects': [],
        'phase_transitions': {}
    }
    
    # Find strongest and weakest effects
    for checkpoint, ckpt_data in results.items():
        if not isinstance(ckpt_data, dict):
            continue
            
        for layer, layer_data in ckpt_data.items():
            if isinstance(layer_data, dict) and 'comparison' in layer_data:
                comparison = layer_data['comparison']
                superposition_diff = abs(comparison.get('superposition_strength_difference', 0))
                
                effect_info = {
                    'checkpoint': checkpoint,
                    'layer': layer,
                    'superposition_diff': superposition_diff,
                    'phase_transition': comparison.get('phase_transition', 'unknown')
                }
                
                if superposition_diff > 0.1:  # Strong effect threshold
                    patterns['strongest_effects'].append(effect_info)
                elif superposition_diff < 0.05:  # Weak effect threshold
                    patterns['weak_effects'].append(effect_info)
                
                # Track phase transitions
                phase = comparison.get('phase_transition', 'unknown')
                if phase not in patterns['phase_transitions']:
                    patterns['phase_transitions'][phase] = 0
                patterns['phase_transitions'][phase] += 1
    
    # Sort by effect strength
    patterns['strongest_effects'].sort(key=lambda x: x['superposition_diff'], reverse=True)
    patterns['weak_effects'].sort(key=lambda x: x['superposition_diff'])
    
    return patterns

def run_polytope_superposition_pipeline(checkpoint_file: str,
                                        output_dir: str = "cache/multi_checkpoint_analysis",
                                        random_seed: Optional[int] = 42) -> Dict[str, Any]:
    """Simplified, function-based pipeline for multi-checkpoint superposition analysis."""
    # Load
    data = _load_checkpoint_data_simple(checkpoint_file)
    records = data['records']
    metadata = data['metadata']
    print(f"Loaded {len(records)} records from {len(metadata.get('checkpoints', []))} checkpoints")
    print(f"Target layers: {metadata.get('target_layers', [])}")

    # Organize
    organized = _organize_records(records)

    # Analyzer
    analyzer = SuperpositionAnalyzer(random_seed=random_seed)

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
            # Strict spline codes
            high_codes = np.stack([np.asarray(r['spline_code'], dtype=int) for r in high])
            low_codes = np.stack([np.asarray(r['spline_code'], dtype=int) for r in low])

            analysis = analyzer.analyze_polytope_superposition(
                activations_high_freq=high_act,
                activations_low_freq=low_act,
                semantic_category=f"checkpoint_{checkpoint}_layer_{layer}",
                binary_patterns_high=high_codes,  # use spline codes as binary patterns strictly
                binary_patterns_low=low_codes,
                spline_codes_high=high_codes,
                spline_codes_low=low_codes,
            )
            analysis['checkpoint_metadata'] = {
                'checkpoint_step': checkpoint,
                'layer': layer,
                'n_high_freq_samples': len(high),
                'n_low_freq_samples': len(low),
                'high_freq_phrases': list(set(r.get('phrase', r.get('phrase')) for r in high)),
                'low_freq_phrases': list(set(r.get('phrase', r.get('phrase')) for r in low)),
            }
            results[checkpoint][layer] = analysis
            ok += 1
    print(f"Completed {ok}/{total} analyses")

    # Compile
    comprehensive = {
        'individual_analyses': results,
        'metadata': metadata,
        'summary_statistics': _compute_cross_checkpoint_statistics(results),
        'patterns': _analyze_evolution_patterns(results),
    }

    # Output
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print("Generating simplified visualizations...")
    _generate_simple_visualizations(comprehensive, output_dir)
    print("Generating simplified report...")
    report = _generate_report(comprehensive)
    with open(output_path / "multi_checkpoint_results.pkl", 'wb') as f:
        pickle.dump(comprehensive, f)
    with open(output_path / "analysis_report.txt", 'w') as f:
        f.write(report)
    # Also produce spline code → phrase mapping artifacts for the paper
    try:
        for ckpt_data in results.values():
            for layer_data in ckpt_data.values():
                # If this is an analysis dict, it won't have raw records; fall back to top-level records
                pass
        # Use original records for index since analyses don't store raw records
        save_spline_code_phrase_index(records, output_dir)
        print("Saved spline-code-to-phrases index artifacts.")
    except Exception as e:
        print(f"Warning: could not build spline-code phrase index: {e}")
    # Build and save bin overlap stats using raw records
    try:
        save_bin_polytope_overlap_stats(records, output_dir)
        print("Saved per-bin polytope overlap statistics.")
    except Exception as e:
        print(f"Warning: could not compute bin overlap stats: {e}")

    print(f"Analysis complete. Results saved to {output_path}")
    print("\nREPORT PREVIEW:")
    print("=" * 50)
    print(report[:2000] + "..." if len(report) > 2000 else report)
    return comprehensive
    
def run_multi_checkpoint_analysis(checkpoint_file: str, 
                                output_dir: str = "cache/multi_checkpoint_analysis") -> Dict[str, Any]:
    """Backwards-compatible wrapper for the simplified pipeline."""
    return run_polytope_superposition_pipeline(checkpoint_file, output_dir)


def _hash_code_row(code: np.ndarray) -> str:
    """Stable hash for a single spline code row (binary array)."""
    b = np.packbits(code.astype(np.uint8))
    return hashlib.sha256(b.tobytes()).hexdigest()


def build_spline_code_phrase_index(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Map unique spline codes to phrases and frequency bins.

    Returns a dict with entries keyed by code_hash containing:
      - 'count': number of samples
      - 'phrases': set/list of phrases
      - 'bins': counts per bin {'low_freq': c1, 'mid_freq': c2, 'high_freq': c3}
      - 'layers': set/list of layers observed
      - 'checkpoints': set/list of checkpoints observed
    """
    index: Dict[str, Any] = {}
    for r in records:
        code = np.asarray(r['spline_code'], dtype=int)
        h = _hash_code_row(code)
        phrase = r.get('phrase', r.get('phrase'))
        raw_cat = r.get('category', r.get('frequency_category', 'unknown'))
        norm = str(raw_cat).strip().lower()
        if norm in {'high_frequency', 'highfreq', 'high-freq', 'high'}:
            bin_key = 'high_freq'
        elif norm in {'low_frequency', 'lowfreq', 'low-freq', 'low'}:
            bin_key = 'low_freq'
        elif norm in {'medium_frequency', 'mid_frequency', 'mid', 'medium'}:
            bin_key = 'mid_freq'
        else:
            bin_key = norm
        if h not in index:
            index[h] = {
                'count': 0,
                'phrases': set(),
                'bins': defaultdict(int),
                'layers': set(),
                'checkpoints': set(),
            }
        idx = index[h]
        idx['count'] += 1
        if phrase is not None:
            idx['phrases'].add(phrase)
        idx['bins'][bin_key] += 1
        idx['layers'].add(r.get('layer'))
        idx['checkpoints'].add(r.get('checkpoint_step'))

    # Convert sets/defaultdicts to lists/dicts
    out: Dict[str, Any] = {}
    for h, v in index.items():
        out[h] = {
            'count': v['count'],
            'phrases': sorted(list(v['phrases'])),
            'bins': dict(v['bins']),
            'layers': sorted([int(x) if isinstance(x, (int, np.integer)) or (isinstance(x, str) and x.isdigit()) else x for x in v['layers']]),
            'checkpoints': sorted(list(v['checkpoints'])),
        }
    return out


def save_spline_code_phrase_index(records: List[Dict[str, Any]], output_dir: str) -> str:
    """Build and save the spline-code-to-phrases index as JSON and CSV."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    index = build_spline_code_phrase_index(records)
    json_path = output_path / 'spline_code_phrase_index.json'
    with open(json_path, 'w') as f:
        json.dump(index, f, indent=2)

    # Flat CSV
    rows = []
    for code_hash, info in index.items():
        bins = info.get('bins', {})
        rows.append({
            'code_hash': code_hash,
            'count': info.get('count', 0),
            'n_phrases': len(info.get('phrases', [])),
            'phrases': '; '.join(info.get('phrases', [])),
            'low_freq': bins.get('low_freq', 0),
            'mid_freq': bins.get('mid_freq', 0),
            'high_freq': bins.get('high_freq', 0),
            'layers': ';'.join(map(str, info.get('layers', []))),
            'checkpoints': ';'.join(map(str, info.get('checkpoints', []))),
        })
    df = pd.DataFrame(rows)
    csv_path = output_path / 'spline_code_phrase_index.csv'
    df.to_csv(csv_path, index=False)
    return str(json_path)


def save_bin_polytope_overlap_stats(records: List[Dict[str, Any]], output_dir: str) -> str:
    """Compute and save per-frequency-bin polytope overlap statistics.

    Metrics per bin:
      - unique_codes: number of unique spline codes observed
      - phrases: number of unique phrases
      - records: number of records
      - codes_with_multi_phrases: number of codes with >=2 distinct phrases in that bin
      - phrase_share_fraction: fraction of phrases that participate in any multi-phrase code
      - assignment_share_fraction: fraction of record assignments that belong to multi-phrase codes
    """
    def normalize_bin(raw: Any) -> str:
        norm = str(raw).strip().lower()
        if norm in {'high_frequency', 'highfreq', 'high-freq', 'high'}:
            return 'high_freq'
        if norm in {'low_frequency', 'lowfreq', 'low-freq', 'low'}:
            return 'low_freq'
        if norm in {'medium_frequency', 'mid_frequency', 'mid', 'medium'}:
            return 'mid_freq'
        return norm

    codes_by_bin: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(lambda: defaultdict(lambda: {'phrases': set(), 'count': 0}))
    phrases_by_bin: Dict[str, set] = defaultdict(set)
    records_by_bin: Dict[str, int] = defaultdict(int)

    for r in records:
        bin_key = normalize_bin(r.get('category', r.get('frequency_category', 'unknown')))
        code = np.asarray(r['spline_code'], dtype=int)
        h = _hash_code_row(code)
        phrase = r.get('phrase', r.get('phrase'))
        codes_by_bin[bin_key][h]['count'] += 1
        if phrase is not None:
            codes_by_bin[bin_key][h]['phrases'].add(phrase)
            phrases_by_bin[bin_key].add(phrase)
        records_by_bin[bin_key] += 1

    rows = []
    for bin_key, code_info in codes_by_bin.items():
        unique_codes = len(code_info)
        phrases_count = len(phrases_by_bin[bin_key])
        record_count = records_by_bin[bin_key]
        multi_codes = [h for h, info in code_info.items() if len(info['phrases']) >= 2]
        phrases_sharing = set()
        for h in multi_codes:
            phrases_sharing.update(code_info[h]['phrases'])
        phrase_share_fraction = (len(phrases_sharing) / phrases_count) if phrases_count > 0 else 0.0
        assignment_share_fraction = (
            sum(code_info[h]['count'] for h in multi_codes) / record_count
            if record_count > 0 else 0.0
        )
        rows.append({
            'bin': bin_key,
            'unique_codes': unique_codes,
            'phrases': phrases_count,
            'records': record_count,
            'codes_with_multi_phrases': len(multi_codes),
            'phrase_share_fraction': float(phrase_share_fraction),
            'assignment_share_fraction': float(assignment_share_fraction),
        })

    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    json_path = output_path / 'bin_polytope_overlap_stats.json'
    with open(json_path, 'w') as f:
        json.dump({'bins': rows}, f, indent=2)
    csv_path = output_path / 'bin_polytope_overlap_stats.csv'
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return str(json_path)