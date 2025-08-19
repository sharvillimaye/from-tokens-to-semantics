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
        """Organize records by frequency category."""
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
        if any(sc is not None for sc in spline_codes):
            # Use spline codes where available, pad where missing
            valid_spline = [sc for sc in spline_codes if sc is not None]
            if valid_spline:
                # Get dimensions from valid spline codes
                spline_matrix = []
                
                for sc in spline_codes:
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
                        bp = binary_patterns[len(spline_matrix)]
                        if len(bp) > activations.shape[1]:
                            bp = bp[:activations.shape[1]]
                        elif len(bp) < activations.shape[1]:
                            bp = np.pad(bp, (0, activations.shape[1] - len(bp)))
                        spline_matrix.append(bp)
                
                spline_codes_array = np.stack(spline_matrix)
            else:
                spline_codes_array = None
        else:
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
        return self._compute_density_chunked(activations, patterns, pairs, chunk_size)
    
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
            euclidean_distances = np.linalg.norm(diff, axis=1)
            
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
        """Compute participation ratio (effective dimensionality)."""
        # Standardize activations
        scaler = StandardScaler()
        activations_std = scaler.fit_transform(activations)
        
        # PCA
        pca = PCA()
        pca.fit(activations_std)
        
        # Participation ratio from eigenvalue spectrum
        eigenvals = pca.explained_variance_
        participation_ratio = (np.sum(eigenvals) ** 2) / np.sum(eigenvals ** 2)
        
        return float(participation_ratio)

    def compute_interference_patterns(self, activations_high: np.ndarray, activations_low: np.ndarray) -> Dict[str, float]:
        """Enhanced interference pattern analysis between frequency groups with vectorized operations."""
        # Mean vectors for each group
        mean_high = np.mean(activations_high, axis=0)
        mean_low = np.mean(activations_low, axis=0)
        
        # Cosine similarity between group means
        cos_sim = np.dot(mean_high, mean_low) / (np.linalg.norm(mean_high) * np.linalg.norm(mean_low))
        
        # Activation norms for distribution analysis
        norms_high = np.linalg.norm(activations_high, axis=1)
        norms_low = np.linalg.norm(activations_low, axis=1)
        
        # Statistical tests
        statistic, p_value = mannwhitneyu(norms_high, norms_low, alternative='two-sided')
        
        # Vectorized neuron-wise interference computation
        overlap_score = self._compute_neuron_correlations_vectorized(activations_high, activations_low)
        
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

    def compute_spline_code_metrics(self, spline_codes: np.ndarray, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute metrics specific to spline code analysis."""
        if len(spline_codes) == 0:
            return {}
        
        # Unique polytope analysis
        unique_codes, inverse_indices, counts = np.unique(spline_codes, axis=0, return_inverse=True, return_counts=True)
        
        # N-gram to polytope mapping
        polytope_to_ngrams = {}
        for i, record in enumerate(records):
            polytope_idx = inverse_indices[i]
            polytope_id = tuple(unique_codes[polytope_idx].astype(int))
            
            if polytope_id not in polytope_to_ngrams:
                polytope_to_ngrams[polytope_id] = []
            
            polytope_to_ngrams[polytope_id].append(record.get('phrase', f'sample_{i}'))
        
        # Polysemantic polytope detection (multiple n-grams per polytope)
        polysemantic_count = 0
        for ngrams in polytope_to_ngrams.values():
            unique_ngrams = set(ngrams)
            if len(unique_ngrams) > 1:
                polysemantic_count += 1
        
        polysemantic_fraction = polysemantic_count / len(polytope_to_ngrams) if polytope_to_ngrams else 0.0
        
        return {
            'total_polytopes': len(unique_codes),
            'polytope_occupancy': counts.tolist(),
            'mean_occupancy': float(np.mean(counts)),
            'max_occupancy': int(np.max(counts)),
            'polysemantic_polytope_count': polysemantic_count,
            'polysemantic_fraction': polysemantic_fraction,
            'samples_per_polytope': float(len(spline_codes) / len(unique_codes))
        }
    
    def analyze_checkpoint(self, records: List[Dict[str, Any]], checkpoint_step: str) -> Dict[str, Any]:
        """Enhanced checkpoint analysis with spline code metrics."""
        logger.info(f"Analyzing checkpoint {checkpoint_step} with {len(records)} records")
        
        # Organize by frequency
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
        }
        
        # Compute metrics for each frequency group
        for freq_type, matrices in [('high_freq', high_matrices), ('low_freq', low_matrices)]:
            if not matrices:
                continue
                
            activations = matrices['activations']
            
            # Use spline codes if available, otherwise binary patterns
            patterns = matrices.get('spline_codes', matrices['binary_patterns'])
            
            # Polytope density with enhanced metrics
            density_metrics = self.compute_polytope_density(activations, patterns)
            results[f'{freq_type}_density'] = density_metrics
            
            # Polytope uniqueness analysis
            unique_patterns = len(np.unique(patterns, axis=0))
            pattern_reuse_rate = 1.0 - (unique_patterns / len(patterns)) if len(patterns) > 0 else 0.0
            
            results[f'{freq_type}_unique_polytopes'] = unique_patterns
            results[f'{freq_type}_pattern_reuse_rate'] = pattern_reuse_rate
            
            # Participation ratio
            participation_ratio = self.compute_participation_ratio(activations)
            results[f'{freq_type}_participation_ratio'] = participation_ratio
            
            # Basic statistics
            results[f'{freq_type}_sparsity'] = float(np.mean(np.sum(patterns, axis=1) / patterns.shape[1]))
            results[f'{freq_type}_activation_norm'] = float(np.mean(np.linalg.norm(activations, axis=1)))
            
            # Spline code specific metrics (if available)
            if 'spline_codes' in matrices:
                spline_metrics = self.compute_spline_code_metrics(matrices['spline_codes'], 
                    high_freq_records if freq_type == 'high_freq' else low_freq_records)
                results[f'{freq_type}_spline_metrics'] = spline_metrics
        
        # Cross-group interference and polytope sharing analysis
        if high_matrices and low_matrices:
            interference = self.compute_interference_patterns(
                high_matrices['activations'], low_matrices['activations']
            )
            results['interference'] = interference
            
            # Polytope sharing between groups
            high_patterns = high_matrices.get('spline_codes', high_matrices['binary_patterns'])
            low_patterns = low_matrices.get('spline_codes', low_matrices['binary_patterns'])
            
            # Convert to hashable tuples for set operations
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
        
        logger.info(f"Checkpoint {checkpoint_step} analysis complete")
        return results

    def create_evolution_visualization(self, results: List[Dict[str, Any]], output_dir: str) -> None:
        """Create evolution visualizations."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Convert to DataFrame for easier plotting
        df_data = []
        for result in results:
            if 'error' in result:
                continue
                
            checkpoint = result['checkpoint_step']
            
            # High freq metrics
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
            logger.warning("No data for visualization")
            return
            
        df = pd.DataFrame(df_data)
        
        # Create multi-panel evolution plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Polytope Evolution Across Training Checkpoints', fontsize=16)
        
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
        plt.savefig(output_path / 'polytope_evolution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
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
            max_workers = min(mp.cpu_count(), len(checkpoint_groups))
        
        logger.info(f"Using {max_workers} parallel workers for checkpoint analysis")
        
        # Prepare checkpoint data for parallel processing
        checkpoint_items = [(checkpoint, records) for checkpoint, records in checkpoint_groups.items()]
        checkpoint_items.sort(key=lambda x: int(x[0]))  # Sort by checkpoint number
        
        # Use multiprocessing to analyze checkpoints in parallel
        try:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                # Submit all checkpoint analysis tasks
                futures = [executor.submit(analyze_checkpoint_worker, records, checkpoint, self.random_seed) 
                          for checkpoint, records in checkpoint_items]
                
                # Collect results in original order
                results = []
                for i, future in enumerate(futures):
                    try:
                        result = future.result(timeout=300)  # 5 minute timeout per checkpoint
                        results.append(result)
                        logger.info(f"Completed checkpoint {checkpoint_items[i][0]}")
                    except Exception as e:
                        logger.error(f"Error processing checkpoint {checkpoint_items[i][0]}: {str(e)}")
                        results.append({'error': str(e), 'checkpoint_step': checkpoint_items[i][0]})
                
                return results
                
        except Exception as e:
            logger.error(f"Multiprocessing failed, falling back to sequential: {str(e)}")
            return self._analyze_checkpoints_sequential(checkpoint_groups)

    def generate_summary_report(self, results: List[Dict[str, Any]]) -> str:
        """Generate a summary report of the analysis."""
        valid_results = [r for r in results if 'error' not in r]
        
        if not valid_results:
            return "No valid results to summarize."
        
        report = "POLYTOPE SUPERPOSITION ANALYSIS SUMMARY\n"
        report += "=" * 50 + "\n\n"
        
        report += f"Analysis completed for {len(valid_results)} checkpoints\n"
        report += f"Checkpoints analyzed: {[r['checkpoint_step'] for r in valid_results]}\n\n"
        
        # Enhanced aggregate statistics
        high_densities = [r['high_freq_density']['density_mean'] for r in valid_results if 'high_freq_density' in r]
        low_densities = [r['low_freq_density']['density_mean'] for r in valid_results if 'low_freq_density' in r]
        
        high_sharing = [r.get('high_freq_pattern_reuse_rate', 0) for r in valid_results]
        low_sharing = [r.get('low_freq_pattern_reuse_rate', 0) for r in valid_results]
        
        polytope_sharing_fractions = [r['polytope_sharing']['sharing_fraction'] for r in valid_results if 'polytope_sharing' in r]
        
        if high_densities and low_densities:
            report += "POLYTOPE DENSITY COMPARISON:\n"
            report += f"High frequency - Mean: {np.mean(high_densities):.4f}, Std: {np.std(high_densities):.4f}\n"
            report += f"Low frequency - Mean: {np.mean(low_densities):.4f}, Std: {np.std(low_densities):.4f}\n"
            
            # Statistical test
            if len(high_densities) > 1 and len(low_densities) > 1:
                statistic, p_value = mannwhitneyu(high_densities, low_densities)
                report += f"Mann-Whitney U test: statistic={statistic:.2f}, p-value={p_value:.6f}\n"
        
        # Participation ratios
        high_pr = [r['high_freq_participation_ratio'] for r in valid_results if 'high_freq_participation_ratio' in r]
        low_pr = [r['low_freq_participation_ratio'] for r in valid_results if 'low_freq_participation_ratio' in r]
        
        if high_pr and low_pr:
            report += "\nPARTICIPATION RATIO COMPARISON:\n"
            report += f"High frequency - Mean: {np.mean(high_pr):.2f}, Std: {np.std(high_pr):.2f}\n"
            report += f"Low frequency - Mean: {np.mean(low_pr):.2f}, Std: {np.std(low_pr):.2f}\n"
        
        # Pattern reuse analysis
        if high_sharing and low_sharing:
            report += "\nPOLYTOPE SHARING ANALYSIS:\n"
            report += f"High frequency pattern reuse - Mean: {np.mean(high_sharing):.4f}, Std: {np.std(high_sharing):.4f}\n"
            report += f"Low frequency pattern reuse - Mean: {np.mean(low_sharing):.4f}, Std: {np.std(low_sharing):.4f}\n"
            
            if len(high_sharing) > 1 and len(low_sharing) > 1:
                sharing_stat, sharing_p = mannwhitneyu(high_sharing, low_sharing)
                report += f"Pattern reuse comparison: statistic={sharing_stat:.2f}, p-value={sharing_p:.6f}\n"
        
        # Interference patterns
        cos_similarities = [r['interference']['cosine_similarity'] for r in valid_results if 'interference' in r]
        if cos_similarities:
            report += "\nINTERFERENCE PATTERNS:\n"
            report += f"Mean cosine similarity between groups: {np.mean(cos_similarities):.4f}\n"
            report += f"Range: [{np.min(cos_similarities):.4f}, {np.max(cos_similarities):.4f}]\n"
        
        # Cross-group polytope sharing
        if polytope_sharing_fractions:
            report += "\nCROSS-GROUP POLYTOPE SHARING:\n"
            report += f"Mean sharing fraction: {np.mean(polytope_sharing_fractions):.4f}\n"
            report += f"Range: [{np.min(polytope_sharing_fractions):.4f}, {np.max(polytope_sharing_fractions):.4f}]\n"
            
            if np.mean(polytope_sharing_fractions) > 0.1:
                report += "⚠️  High polytope sharing detected - potential superposition\n"
            else:
                report += "✅ Low polytope sharing - likely specialized representations\n"
        
        # Research implications
        report += "\nRESEARCH IMPLICATIONS:\n"
        
        if high_densities and low_densities:
            density_diff = np.mean(high_densities) - np.mean(low_densities)
            if density_diff > 0.1:
                report += "• High-frequency n-grams show higher polytope density\n"
                report += "• Suggests more complex geometric structure for frequent patterns\n"
            elif density_diff < -0.1:
                report += "• Low-frequency n-grams show higher polytope density\n"
                report += "• Suggests more distributed representations for rare patterns\n"
            else:
                report += "• Similar polytope densities across frequency groups\n"
        
        if polytope_sharing_fractions and np.mean(polytope_sharing_fractions) > 0.05:
            report += "• Evidence of polytope sharing between frequency groups\n"
            report += "• Supports superposition hypothesis in neural representations\n"
        
        report += "\n" + "=" * 50 + "\n"
        report += "REFINED POLYTOPE ANALYSIS COMPLETED\n"
        report += "Key innovations: spline codes, enhanced metrics, statistical testing\n"
        
        return report


def analyze_checkpoint_worker(records: List[Dict[str, Any]], checkpoint_step: str, random_seed: int) -> Dict[str, Any]:
    """
    Worker function for multiprocessing checkpoint analysis.
    
    This function creates a temporary analyzer instance to avoid pickling issues
    with the main analyzer object's logger and other non-serializable components.
    """
    try:
        # Create a minimal analyzer for this worker
        temp_analyzer = RefinedPolytopeAnalyzer(random_seed=random_seed, cache_dir="cache/temp_worker")
        
        # Analyze the checkpoint
        result = temp_analyzer.analyze_checkpoint(records, checkpoint_step)
        
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


if __name__ == "__main__":
    # Example usage
    checkpoint_file = "cache/checkpoint_analysis/checkpoint_analysis_20241219_143022.pkl"
    results_dir = run_polytope_analysis_pipeline(checkpoint_file)
    print(f"Analysis complete. Results saved to: {results_dir}")