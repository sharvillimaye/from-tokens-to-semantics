from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import gc
import json
import multiprocessing as mp
from pathlib import Path
import pickle
from typing import Any, Dict, List, Optional, Tuple
import warnings
from collections import defaultdict
from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from hdbscan import HDBSCAN  

warnings.filterwarnings('ignore', category=FutureWarning)

class PolytopeAnalyzer:
    """
    Streamlined polytope analyzer focused on mechanistic interpretability research.
    
    Key improvements over original:
    - Prioritizes spline codes over CETT binary patterns
    - Cleaner API with direct integration to checkpoint analysis
    - Built-in logging and visualization pipeline
    - Research-focused metrics and statistical tests
    """
    
    def __init__(self, random_seed: int = 42, cache_dir: str = "cache"):
        """Initialize analyzer with sane defaults and logging."""
        self.random_seed = random_seed
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Runtime/config defaults
        self.max_pairs: int = 2000  # lower by default for faster, more stable runs
        self.per_checkpoint_timeout: int = 900  # seconds; avoid premature timeouts on large checkpoints
        self.max_workers: int = max(1, mp.cpu_count() - 1)
        # Feature toggles
        self.enable_interference: bool = True
        
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

        # Configure plotting defaults for ICLR figures
        try:
            sns.set_theme(context="paper", style="whitegrid")
        except Exception:
            pass

    # ---------------------------
    # Pipeline Orchestration
    # ---------------------------
    def run_checkpoint_file(
        self,
        checkpoint_path: str,
        output_dir: Optional[str] = None,
        parallel_layers: bool = True,
        max_workers: Optional[int] = None,
        concurrency_backend: str = "thread",
    ) -> Dict[str, Any]:
        """High-level entrypoint: load → analyze → save for one checkpoint.

        - parallel_layers: run each layer's analysis concurrently
        - max_workers: override default worker pool size
        - concurrency_backend: "process" or "thread"
        """
        data = self.load_checkpoint_data(checkpoint_path)
        records = data.get('records', [])
        # Group by checkpoints

        checkpoint_groups = {}
        for record in records:
            checkpoint = str(record['checkpoint_step'])
            if checkpoint not in checkpoint_groups:
                checkpoint_groups[checkpoint] = []
            checkpoint_groups[checkpoint].append(record)
        
        all_results: Dict[str, Any] = {}
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        for checkpoint, ckpt_records in checkpoint_groups.items():
            analysis = self.analyze_checkpoint(
                ckpt_records,
                checkpoint_step=checkpoint,
                use_layer_wise_analysis=True,
                parallel_layers=parallel_layers,
                max_workers=max_workers,
                concurrency_backend=concurrency_backend,
            )
            all_results[checkpoint] = analysis
            if output_dir:
                self._save_analysis_results(analysis, checkpoint, output_dir)

        return all_results

    def get_checkpoint_evolution_summary(self, multi_checkpoint_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a summary of how metrics evolve across checkpoints.
        
        Returns:
            Summary statistics and trends across checkpoints
        """
        flattened = self.flatten_multi_checkpoint_results(multi_checkpoint_results)
        
        summary = {
            'checkpoints': sorted(multi_checkpoint_results.keys()),
            'layers': sorted(flattened.keys()),
            'evolution_by_layer': {}
        }
        
        for layer, layer_history in flattened.items():
            layer_summary = {
                'checkpoints_analyzed': len(layer_history),
                'metrics_over_time': {}
            }
            
            # Extract key metrics across checkpoints for this layer
            for metric in ['high_freq_participation_ratio', 'low_freq_participation_ratio', 
                          'high_freq_sparsity', 'low_freq_sparsity']:
                values = [entry.get(metric, 0.0) for entry in layer_history if isinstance(entry, dict)]
                if values:
                    layer_summary['metrics_over_time'][metric] = {
                        'values': values,
                        'trend': 'increasing' if len(values) > 1 and values[-1] > values[0] else 'decreasing',
                        'min': float(min(values)),
                        'max': float(max(values)),
                        'mean': float(sum(values) / len(values))
                    }
            
            summary['evolution_by_layer'][layer] = layer_summary
        
        return summary

    def run_multiple_checkpoints(
        self,
        checkpoint_paths: List[str],
        output_dir: Optional[str] = None,
        parallel_checkpoints: bool = True,
        parallel_layers: bool = True,
        max_workers: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Analyze multiple checkpoints and return a dict keyed by checkpoint step."""
        results: Dict[str, Any] = {}
        worker_count = max_workers or self.max_workers
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)

        def _analyze_path(path: str) -> Tuple[str, Dict[str, Any]]:
            data = self._load_checkpoint_data_full(path)
            records = data.get('records', [])
            # Infer checkpoint step from filename or metadata if present
            step = None
            try:
                step = str(Path(path).stem).split("_")[-1]
            except Exception:
                step = "unknown"
            analysis = self.analyze_checkpoint(
                records,
                checkpoint_step=step,
                use_layer_wise_analysis=True,
                parallel_layers=parallel_layers,
                max_workers=max_workers,
                concurrency_backend="thread",
            )
            if output_dir:
                self._save_analysis_results(analysis, step, output_dir)
            return step, analysis

        if parallel_checkpoints and len(checkpoint_paths) > 1:
            try:
                with ThreadPoolExecutor(max_workers=worker_count) as ex:
                    futures = {ex.submit(_analyze_path, p): p for p in checkpoint_paths}
                    for fut in as_completed(futures):
                        step, analysis = fut.result()
                        results[step] = analysis
            except Exception:
                for p in checkpoint_paths:
                    step, analysis = _analyze_path(p)
                    results[step] = analysis
        else:
            for p in checkpoint_paths:
                step, analysis = _analyze_path(p)
                results[step] = analysis

        return results

    def flatten_multi_checkpoint_results(self, multi_checkpoint_results: Dict[str, Any]) -> Dict[int, List[Dict[str, Any]]]:
        """Flatten results into per-layer time series entries with checkpoint annotations.

        Returns mapping: layer -> [ { 'checkpoint_step': str, <metrics...> }, ... ]
        """
        flattened: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for checkpoint_step, analysis in multi_checkpoint_results.items():
            layer_results = analysis.get('layer_results', {}) if isinstance(analysis, dict) else {}
            for layer, layer_result in layer_results.items():
                if not isinstance(layer, int):
                    try:
                        layer_int = int(layer)
                    except Exception:
                        continue
                else:
                    layer_int = layer
                entry: Dict[str, Any] = {'checkpoint_step': str(checkpoint_step)}
                # Pull selected metrics if available
                for prefix in ['high_freq', 'low_freq']:
                    if f'{prefix}_participation_ratio' in layer_result:
                        entry[f'{prefix}_participation_ratio'] = layer_result.get(f'{prefix}_participation_ratio', 0.0)
                    density = layer_result.get(f'{prefix}_density', {})
                    if density:
                        entry[f'{prefix}_boundary_crossing_rate'] = density.get('boundary_crossing_rate', 0.0)
                        entry[f'{prefix}_density_mean'] = density.get('density_mean', 0.0)
                    entry[f'{prefix}_compression_superposition'] = (
                        layer_result.get(f'{prefix}_compression_superposition', {}).get('compression_superposition', 0.0)
                    )
                    entry[f'{prefix}_boundary_superposition'] = (
                        layer_result.get(f'{prefix}_boundary_superposition', {}).get('boundary_superposition', 0.0)
                    )
                # Polytope sharing & cross-frequency analysis
                sharing = layer_result.get('polytope_sharing', {})
                if sharing:
                    entry['polytope_sharing_fraction'] = sharing.get('sharing_fraction', 0.0)
                cross = layer_result.get('clustering', {}).get('cross_frequency', {})
                if cross:
                    entry['cross_freq_jaccard'] = cross.get('jaccard_overlap', 0.0)
                    entry['cross_freq_boundary_density_ratio'] = cross.get('boundary_density_ratio', 0.0)
                # Subspace analysis
                subspace = layer_result.get('subspace_analysis', {})
                if subspace:
                    entry['subspace_principal_angle_deg'] = subspace.get('principal_angle_deg', 0.0)
                    entry['subspace_alignment'] = subspace.get('subspace_alignment', 0.0)
                flattened[layer_int].append(entry)
        # Sort by checkpoint order if numeric
        for layer, entries in flattened.items():
            def _key(e: Dict[str, Any]):
                step = e.get('checkpoint_step', '')
                try:
                    return int(step)
                except Exception:
                    return step
            flattened[layer] = sorted(entries, key=_key)
        return flattened

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


    def organize_by_frequency(self, records: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Organize records by frequency category (legacy method - use organize_by_layer_and_frequency for layer-aware analysis)."""
        organized = {'high_freq': [], 'medium_freq': [], 'low_freq': []}
        
        for record in records:
            # Normalize category names
            cat = record.get('frequency_category')
            
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
        """Extract aligned matrices for analysis (activations and codes).

        Ensures shape compatibility and robust handling of missing/variable-length codes.
        """
        if not records:
            logger.warning("No records provided for matrix extraction")
            return {}
        
        # Extract activation vectors (post-activation)
        activations = np.stack([np.asarray(r['activation_vector']) for r in records])

        # Collect raw code vectors; allow None
        raw_codes: List[Optional[np.ndarray]] = [
            (np.asarray(r.get('binary_pattern')) if r.get('binary_pattern') is not None else None)
            for r in records
        ]
        available = [c for c in raw_codes if c is not None]
        logger.info(f"Spline code availability: {len(available)}/{len(raw_codes)} records have codes")

        if available:
            target_dim = min(activations.shape[1], int(max(len(c) for c in available)))
            def _fit_dim(c: Optional[np.ndarray]) -> np.ndarray:
                if c is None:
                    return np.zeros(target_dim, dtype=np.uint8)
                arr = np.asarray(c)
                if arr.ndim != 1:
                    arr = arr.ravel()
                if len(arr) > target_dim:
                    arr = arr[:target_dim]
                elif len(arr) < target_dim:
                    arr = np.pad(arr, (0, target_dim - len(arr)))
                return (arr > 0).astype(np.uint8)
            spline_codes_array = np.stack([_fit_dim(c) for c in raw_codes])
        else:
            logger.warning("No binary/spline codes present; defaulting to zeros")
            spline_codes_array = np.zeros((len(records), activations.shape[1]), dtype=np.uint8)

        matrices = {
            'activations': activations,
            'binary_patterns': spline_codes_array,
        }

        logger.info(
            f"Extracted matrices - Activations: {activations.shape}, Codes: {spline_codes_array.shape}"
        )

        return matrices

    def compute_polytope_density(self, activations: np.ndarray, spline_codes: np.ndarray, 
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
        return self._compute_density_chunked(activations_std, spline_codes, pairs, chunk_size)
    
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
    
    def _compute_density_chunked(self, activations: np.ndarray, spline_codes: np.ndarray, 
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
            xor = spline_codes[chunk_pairs[:, 0]] != spline_codes[chunk_pairs[:, 1]]
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
            'valid_pairs': len(densities)
        }

    def compute_polytope_compression_superposition(self, spline_codes: np.ndarray, 
                                                 records: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Core superposition metric: How many unique n-grams map to the same polytopes?
        High compression = high superposition
        """
        if spline_codes is None or len(spline_codes) == 0 or len(records) == 0:
            return {'compression_superposition': 0.0}
        polytope_to_ngrams: Dict[Tuple[int, ...], set] = defaultdict(set)
        ngram_to_polytope: Dict[str, Tuple[int, ...]] = {}
        for i, record in enumerate(records):
            if i >= len(spline_codes):
                continue
            polytope_id = tuple(np.asarray(spline_codes[i]).astype(int).tolist())
            ngram = record.get('phrase', f'sample_{i}')
            polytope_to_ngrams[polytope_id].add(ngram)
            ngram_to_polytope[ngram] = polytope_id
        total_unique_ngrams = len(set(ngram_to_polytope.keys()))
        total_polytopes = len(polytope_to_ngrams)
        if total_polytopes == 0:
            return {'compression_superposition': 0.0}
        compression_ratio = float(total_unique_ngrams) / float(total_polytopes)
        superposition_intensity = max(0.0, (compression_ratio - 1.0) / compression_ratio)
        polysemantic_polytopes = sum(1 for ngrams in polytope_to_ngrams.values() if len(ngrams) > 1)
        polysemantic_fraction = float(polysemantic_polytopes) / float(total_polytopes) if total_polytopes > 0 else 0.0
        avg_ngrams_per_polytope = float(total_unique_ngrams) / float(total_polytopes) if total_polytopes > 0 else 0.0
        # Interference patterns between frequency groups are added outside per-group metric
        return {
            'compression_superposition': float(superposition_intensity),
        }

    def compute_feature_disentanglement(
        self,
        patterns: np.ndarray,
        records: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """Measure feature disentanglement within a group using TF-IDF semantics vs polytopes.

        Lower mean intra-polytope semantic diversity implies better disentanglement.
        """
        if patterns is None or len(patterns) == 0 or not records:
            return {'semantic_dispersal': 0.0, 'polytope_purity': 0.0}
        # Cap sample size for TF-IDF to avoid excessive memory/CPU
        max_docs = 5000
        if len(records) > max_docs:
            idx = self.rng.choice(len(records), size=max_docs, replace=False)
            texts = [records[i].get('phrase', '') for i in idx]
            patterns = patterns[idx]
        else:
            texts = [r.get('phrase', '') for r in records]
        try:
            tfidf = TfidfVectorizer(max_features=2048).fit_transform(texts)
            # Group by polytope id
            ids = [tuple((row > 0).astype(int).tolist()) for row in patterns]
            groups: Dict[Tuple[int, ...], List[int]] = defaultdict(list)
            for idx, pid in enumerate(ids):
                groups[pid].append(idx)
            intra_sims = []
            sizes = []
            for _, idxs in groups.items():
                if len(idxs) < 2:
                    continue
                sub = tfidf[idxs]
                sim = cosine_similarity(sub)
                # Exclude diagonal
                mask = ~np.eye(sim.shape[0], dtype=bool)
                intra = float(np.mean(sim[mask])) if sim.size > 0 else 0.0
                intra_sims.append(intra)
                sizes.append(len(idxs))
            if intra_sims:
                # Size-weighted mean intra-similarity (higher is more coherent/pure)
                weights = np.array(sizes, dtype=float)
                purity = float(np.average(np.array(intra_sims, dtype=float), weights=weights))
                dispersal = float(1.0 - purity)
            else:
                purity = 0.0
                dispersal = 0.0
        except Exception:
            purity = 0.0
            dispersal = 0.0
        return {'semantic_dispersal': dispersal, 'polytope_purity': purity}

    def compute_boundary_density_superposition(self, density_metrics: Dict[str, float]) -> Dict[str, float]:
        """
        Use polytope density calculations for superposition.
        High boundary crossings = high superposition (features interfering)
        """
        if not density_metrics:
            return {'boundary_superposition': 0.0}
        boundary_crossing_rate = float(density_metrics.get('boundary_crossing_rate', 0.0))
        density_mean = float(density_metrics.get('density_mean', 0.0))
        density_std = float(density_metrics.get('density_std', 0.0))
        boundary_superposition = min(1.0, boundary_crossing_rate * 2.0)
        density_variability = min(1.0, density_std / density_mean) if density_mean > 0 else 0.0
        combined_score = 0.7 * boundary_superposition + 0.3 * density_variability
        return {
            'boundary_superposition': float(boundary_superposition),
        }

    def compute_participation_ratio(self, activations: np.ndarray) -> float:
        """Compute participation ratio (effective dimensionality) efficiently and robustly.

        Uses randomized PCA with a capped number of components for speed on large matrices.
        """
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
            
            # PCA with error handling (randomized SVD, limited components)
            n_features = activations_std.shape[1]
            n_samples = activations_std.shape[0]
            n_components = int(min(128, n_features, n_samples))
            pca = PCA(n_components=n_components, svd_solver='randomized', random_state=self.random_seed)
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

   
    def analyze_checkpoint(
        self,
        records: List[Dict[str, Any]],
        checkpoint_step: str,
        use_layer_wise_analysis: bool = True,
        parallel_layers: bool = True,
        max_workers: Optional[int] = None,
        concurrency_backend: str = "process",
    ) -> Dict[str, Any]:
        """Enhanced checkpoint analysis with optional per-layer parallelization."""
        logger.info(f"Analyzing checkpoint {checkpoint_step} with {len(records)} records")
        
        if use_layer_wise_analysis:
            return self._analyze_checkpoint_layer_wise(
                records,
                checkpoint_step,
                parallel_layers=parallel_layers,
                max_workers=max_workers,
                concurrency_backend=concurrency_backend,
            )
            
    def _analyze_checkpoint_layer_wise(
        self,
        records: List[Dict[str, Any]],
        checkpoint_step: str,
        parallel_layers: bool = True,
        max_workers: Optional[int] = None,
        concurrency_backend: str = "process",
    ) -> Dict[str, Any]:
        """Layer-wise analysis with optional concurrency across layers."""
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
        tasks: List[Tuple[int, List[Dict[str, Any]], List[Dict[str, Any]]]] = []
        for layer, freq_groups in layer_groups.items():
            hf = freq_groups['high_freq']
            lf = freq_groups['low_freq']
            total_records += len(hf) + len(lf)
            if not hf or not lf:
                logger.warning(f"Missing frequency groups in layer {layer} of checkpoint {checkpoint_step}")
                results['layer_results'][layer] = {'error': 'Missing frequency groups'}
            else:
                tasks.append((layer, hf, lf))

        # Execute per-layer analyses
        worker_count = max_workers or self.max_workers
        if parallel_layers and len(tasks) > 1:
            try:
                if concurrency_backend == "thread":
                    exec_context = ThreadPoolExecutor(max_workers=worker_count)
                else:
                    exec_context = ProcessPoolExecutor(max_workers=worker_count)
                with exec_context as ex:
                    future_to_layer = {
                        ex.submit(self._analyze_single_layer, layer, hf, lf, checkpoint_step): layer
                        for (layer, hf, lf) in tasks
                    }
                    for fut in as_completed(future_to_layer):
                        layer_id = future_to_layer[fut]
                        try:
                            results['layer_results'][layer_id] = fut.result()
                        except Exception as e:
                            logger.exception(f"Layer {layer_id} analysis failed: {e}")
                            results['layer_results'][layer_id] = {'error': str(e)}
            except Exception as e:
                logger.warning(f"Parallel layer analysis failed ({e}); falling back to sequential")
                for layer, hf, lf in tasks:
                    results['layer_results'][layer] = self._analyze_single_layer(layer, hf, lf, checkpoint_step)
        else:
            for layer, hf, lf in tasks:
                results['layer_results'][layer] = self._analyze_single_layer(layer, hf, lf, checkpoint_step)
        
        results['total_records'] = total_records
        
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
            'checkpoint_step': checkpoint_step,
            'n_high_freq': len(high_freq_records),
            'n_low_freq': len(low_freq_records),
        }
        
        # Compute metrics for each frequency group in this layer
        for freq_type, matrices, records in [('high_freq', high_matrices, high_freq_records), 
                                           ('low_freq', low_matrices, low_freq_records)]:
            if not matrices:
                continue
                
            activations = matrices['activations']
            patterns = matrices.get('binary_patterns', matrices['binary_patterns'])
            
            # All the same metrics, but now layer-specific
            logger.info(f"Layer {layer} {freq_type}: computing polytope density on activations {activations.shape}")
            density_metrics = self.compute_polytope_density(activations, patterns, max_pairs=self.max_pairs)
            layer_result[f'{freq_type}_density'] = density_metrics
            
            logger.info(f"Layer {layer} {freq_type}: computing participation ratio")
            layer_result[f'{freq_type}_participation_ratio'] = self.compute_participation_ratio(activations)
            logger.info(f"Layer {layer} {freq_type}: computing boundary superposition")
            layer_result[f'{freq_type}_boundary_superposition'] = self.compute_boundary_density_superposition(density_metrics)
            logger.info(f"Layer {layer} {freq_type}: computing compression superposition")
            layer_result[f'{freq_type}_compression_superposition'] = self.compute_polytope_compression_superposition(patterns, records)
            # Feature disentanglement per group
            logger.info(f"Layer {layer} {freq_type}: computing feature disentanglement")
            layer_result[f'{freq_type}_disentanglement'] = self.compute_feature_disentanglement(patterns, records)
        
        # Cross-group analysis for this layer
        if high_matrices and low_matrices:
            
            # Layer-specific polytope sharing
            high_patterns = high_matrices.get('binary_patterns', high_matrices['binary_patterns'])
            low_patterns = low_matrices.get('binary_patterns', low_matrices['binary_patterns'])
            
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
            logger.info(f"Layer {layer}: clustering spline codes (high/low) with downsampling")
            layer_result['clustering'] = self.enhanced_spline_code_clustering(
                high_patterns, low_patterns, high_freq_records, low_freq_records
            )
            logger.info(f"Layer {layer}: computing activation subspace analysis")
            subspace = self.compute_activation_subspace_analysis(
                high_matrices['activations'],
                low_matrices['activations'],
                high_patterns,
                low_patterns
            )
            layer_result['subspace_analysis'] = subspace
        return layer_result

    def compute_activation_subspace_analysis(
        self,
        activations_high: np.ndarray,
        activations_low: np.ndarray,
        patterns_high: np.ndarray,
        patterns_low: np.ndarray,
        max_components: int = 32,
    ) -> Dict[str, Any]:
        """Compare subspaces between high/low-frequency activations with stability checks.

        - Computes PCA subspaces and principal angle (via canonical correlations).
        - Returns alignment metrics that can be tracked over training.
        """
        try:
            # Standardize and clip
            A_h = self._zscore_for_distance(activations_high)
            A_l = self._zscore_for_distance(activations_low)

            # Determine components
            n_comp = int(min(max_components, A_h.shape[1], A_h.shape[0], A_l.shape[1], A_l.shape[0]))
            if n_comp < 2:
                return {
                    'principal_angle_deg': 90.0,
                    'subspace_alignment': 0.0,
                    'subspace_validation': {'hull_computation_recommended': False}
                }

            pca_h = PCA(n_components=n_comp)
            pca_l = PCA(n_components=n_comp)
            H = pca_h.fit_transform(A_h)
            L = pca_l.fit_transform(A_l)

            # Compute singular values of cross-covariance between orthonormal bases
            U_h = pca_h.components_.T
            U_l = pca_l.components_.T
            # Guard
            if U_h.shape[1] != U_l.shape[1]:
                m = int(min(U_h.shape[1], U_l.shape[1]))
                U_h = U_h[:, :m]
                U_l = U_l[:, :m]
            # Cosines of principal angles are singular values of U_h^T U_l
            M = U_h.T @ U_l
            sv = np.linalg.svd(M, compute_uv=False)
            sv = np.clip(sv, 0.0, 1.0)
            # Principal angle is arccos of min singular value (largest angle)
            principal_angle = float(np.degrees(np.arccos(max(1e-9, float(np.min(sv))))))
            alignment = float(np.mean(sv))

            # Semantic disentanglement via ARI between clusterings on both groups (proxy)
            try:
                clus_h = HDBSCAN(min_cluster_size=5, min_samples=3).fit_predict((patterns_high > 0).astype(np.uint8))
                clus_l = HDBSCAN(min_cluster_size=5, min_samples=3).fit_predict((patterns_low > 0).astype(np.uint8))
                min_len = min(len(clus_h), len(clus_l))
                ari = float(adjusted_rand_score(clus_h[:min_len], clus_l[:min_len])) if min_len > 10 else 0.0
            except Exception:
                ari = 0.0

            return {
                'principal_angle_deg': principal_angle,
                'subspace_alignment': alignment,
                'cross_group_ari': ari,
                'subspace_validation': {
                    'hull_computation_recommended': alignment < 0.95 or principal_angle > 5.0
                }
            }
        except Exception as e:
            logger.warning(f"Subspace analysis failed: {e}")
            return {
                'principal_angle_deg': 90.0,
                'subspace_alignment': 0.0,
                'subspace_validation': {'hull_computation_recommended': False}
            }
    
    def enhanced_spline_code_clustering(self, spline_codes_high: np.ndarray, 
                                       spline_codes_low: np.ndarray, 
                                       records_high: List[Dict[str, Any]], 
                                       records_low: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Cluster spline codes for each frequency group and analyze fragmentation/coherence.
        Uses HDBSCAN (preferred for binary polytope regions)
        """
        def _cluster_group(spline_codes: np.ndarray, records: List[Dict[str, Any]], freq_type: str) -> Dict[str, Any]:
            if spline_codes is None or len(spline_codes) == 0:
                return {
                    'n_clusters': 0,
                    'n_noise_points': 0,
                    'cluster_sizes': [],
                    'semantic_coherence': {'mean_coherence': 0.0, 'coherence_std': 0.0, 'highly_coherent_clusters': 0},
                    'fragmentation_score': 0.0,
                    'noise_ratio': 0.0,
                    'labels': []
                }
            # Ensure boolean/binary array for Hamming and downsample for speed
            X_full = (spline_codes > 0).astype(np.uint8)
            n = len(X_full)
            sample_size = min(4000, n)
            if n > sample_size:
                idx = self.rng.choice(n, size=sample_size, replace=False)
                X = X_full[idx]
                recs = [records[i] for i in idx]
            else:
                X = X_full
                recs = records
            labels = None
            clusterer = HDBSCAN(min_cluster_size=10, min_samples=5, metric='hamming',
            cluster_selection_epsilon=0.05)
            labels = clusterer.fit_predict(X)
            labels_list = list(labels)
            unique_labels = sorted([l for l in set(labels_list) if l != -1])
            n_clusters = len(unique_labels)
            n_noise = labels_list.count(-1)
            cluster_sizes = [labels_list.count(i) for i in unique_labels]
            coherence = self._analyze_cluster_semantics(labels, recs)
            fragmentation = float(n_clusters) / float(len(X)) if len(X) > 0 else 0.0
            noise_ratio = float(n_noise) / float(len(X)) if len(X) > 0 else 0.0
            return {
                'n_clusters': n_clusters,
                'n_noise_points': n_noise,
                'cluster_sizes': cluster_sizes,
                'semantic_coherence': coherence,
                'fragmentation_score': fragmentation,
                'noise_ratio': noise_ratio,
                'labels': labels_list
            }
        results: Dict[str, Any] = {}
        results['high_freq_clustering'] = _cluster_group(spline_codes_high, records_high, 'high_freq')
        results['low_freq_clustering'] = _cluster_group(spline_codes_low, records_low, 'low_freq')
        results['cross_frequency'] = self._analyze_cross_frequency_polytopes(
            spline_codes_high, spline_codes_low, records_high, records_low
        )
        return results

    def _analyze_cluster_semantics(self, cluster_labels: np.ndarray, 
                                   records: List[Dict[str, Any]]) -> Dict[str, float]:
        """Analyze semantic coherence within each cluster using phrases as proxy semantics.
        Lower scores indicate tighter, more monosemantic clusters.
        """
        cluster_to_indices: Dict[int, List[int]] = defaultdict(list)
        for idx, label in enumerate(cluster_labels):
            if label != -1:
                cluster_to_indices[int(label)].append(idx)
        coherence_scores: List[float] = []
        highly_coherent = 0
        for cluster_id, indices in cluster_to_indices.items():
            if not indices:
                continue
            phrases = []
            for i in indices:
                if i < len(records):
                    phrase = records[i].get('phrase', '')
                    if phrase:
                        phrases.append(phrase)
            unique_phrase_count = len(set(phrases)) if phrases else 0
            cluster_size = len(indices)
            coherence = float(unique_phrase_count) / float(cluster_size) if cluster_size > 0 else 0.0
            coherence_scores.append(coherence)

            if coherence < 0.5:
                highly_coherent += 1
        return {
            'mean_coherence': float(np.mean(coherence_scores)) if coherence_scores else 0.0,
            'coherence_std': float(np.std(coherence_scores)) if coherence_scores else 0.0,
            'highly_coherent_clusters': int(highly_coherent)
        }

    def _analyze_cross_frequency_polytopes(self, spline_codes_high: np.ndarray, spline_codes_low: np.ndarray,
                                           records_high: List[Dict[str, Any]], records_low: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Cross-frequency polytope overlap and boundary density using Hamming geometry."""
        results: Dict[str, Any] = {}
        if spline_codes_high is None or spline_codes_low is None or len(spline_codes_high) == 0 or len(spline_codes_low) == 0:
            return {'jaccard_overlap': 0.0, 'low_in_high_fraction': 0.0, 'boundary_density_ratio': 0.0}
        H = (spline_codes_high > 0).astype(np.uint8)
        L = (spline_codes_low > 0).astype(np.uint8)
        high_set = {tuple(row.tolist()) for row in H}
        low_set = {tuple(row.tolist()) for row in L}
        inter = len(high_set.intersection(low_set))
        union = len(high_set.union(low_set))
        jaccard = float(inter) / float(union) if union > 0 else 0.0
        low_in_high = float(inter) / float(len(low_set)) if len(low_set) > 0 else 0.0
        # Boundary density ratio: across-group vs within-group average Hamming
        rng = self.rng
        sample_high = min(len(H), 1000)
        sample_low = min(len(L), 1000)
        idx_h = rng.choice(len(H), size=sample_high, replace=False)
        idx_l = rng.choice(len(L), size=sample_low, replace=False)
        H_s = H[idx_h]
        L_s = L[idx_l]
        # Across-group distances
        # Compute pairwise on samples in a vectorized but memory-conscious way
        across = []
        for h in H_s:
            xor = (L_s != h)
            across.append(np.sum(xor, axis=1))
        across = np.concatenate(across) if across else np.array([], dtype=np.float64)
        # Within-group distances (high)
        within_h = []
        for i in range(len(H_s)):
            xor = (H_s[i+1:] != H_s[i])
            if xor.size:
                within_h.append(np.sum(xor, axis=1))
        within_h = np.concatenate(within_h) if within_h else np.array([], dtype=np.float64)
        # Within-group distances (low)
        within_l = []
        for i in range(len(L_s)):
            xor = (L_s[i+1:] != L_s[i])
            if xor.size:
                within_l.append(np.sum(xor, axis=1))
        within_l = np.concatenate(within_l) if within_l else np.array([], dtype=np.float64)
        mean_across = float(np.mean(across)) if across.size else 0.0
        mean_within = float(np.mean(np.concatenate([within_h, within_l]))) if within_h.size or within_l.size else 1.0
        boundary_density_ratio = float(mean_across / mean_within) if mean_within > 0 else 0.0
        results.update({
            'jaccard_overlap': jaccard,
            'low_in_high_fraction': low_in_high,
            'boundary_density_ratio': boundary_density_ratio
        })
        return results

    def _save_analysis_results(self, analysis_results: Dict[str, Any], checkpoint_step: str, path: str) -> None:
        """Save analysis results to a file."""
        with open(f'{path}/analysis_results_{checkpoint_step}.json', 'w') as f:
            json.dump(analysis_results, f)

    def _visualization_results(self, analysis_results: Dict[str, Any], checkpoint_step: str, path: str) -> None:
        """Visualize analysis results."""
        with open(f'{path}/analysis_results_{checkpoint_step}.json', 'r') as f:
            analysis_results = json.load(f)
        return analysis_results

# Visualizations
def participation_ratio_heatmap(results: Dict[str, Any], output_dir: str):
    """Create a participation ratio heatmap over layers vs checkpoints (high/low panels)."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    plt.style.use('seaborn-paper')
    analyzer = PolytopeAnalyzer()
    flattened = analyzer.flatten_multi_checkpoint_results(results)
    # Build matrices
    layers = sorted(flattened.keys())
    # Determine checkpoint ordering
    all_steps = sorted({e['checkpoint_step'] for v in flattened.values() for e in v}, key=lambda s: int(s) if str(s).isdigit() else s)
    step_to_idx = {s: i for i, s in enumerate(all_steps)}
    H = np.full((len(layers), len(all_steps)), np.nan, dtype=float)
    L = np.full((len(layers), len(all_steps)), np.nan, dtype=float)
    for li, layer in enumerate(layers):
        for entry in flattened[layer]:
            si = step_to_idx[entry['checkpoint_step']]
            if 'high_freq_participation_ratio' in entry:
                H[li, si] = entry['high_freq_participation_ratio']
            if 'low_freq_participation_ratio' in entry:
                L[li, si] = entry['low_freq_participation_ratio']
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300, constrained_layout=True)
    for ax, mat, title in zip(axes, [H, L], ["High freq PR", "Low freq PR"]):
        sns.heatmap(mat, ax=ax, cmap="viridis", cbar=True, vmin=np.nanmin(mat), vmax=np.nanmax(mat))
        ax.set_title(title)
        ax.set_xlabel("Checkpoint step")
        ax.set_ylabel("Layer")
        ax.set_xticks(np.arange(len(all_steps)) + 0.5)
        ax.set_xticklabels(all_steps, rotation=45, ha='right')
        ax.set_yticks(np.arange(len(layers)) + 0.5)
        ax.set_yticklabels(layers)
    fig.suptitle("Participation Ratio Evolution")
    fig.savefig(Path(output_dir) / "participation_ratio_heatmap.pdf", format="pdf")
    plt.close(fig)

def polytope_density_heatmap(results: Dict[str, Any], output_dir: str):
    """Create boundary crossing rate heatmaps for high/low frequency groups."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    plt.style.use('seaborn-paper')
    analyzer = PolytopeAnalyzer()
    flattened = analyzer.flatten_multi_checkpoint_results(results)
    layers = sorted(flattened.keys())
    all_steps = sorted({e['checkpoint_step'] for v in flattened.values() for e in v}, key=lambda s: int(s) if str(s).isdigit() else s)
    step_to_idx = {s: i for i, s in enumerate(all_steps)}
    H = np.full((len(layers), len(all_steps)), np.nan, dtype=float)
    L = np.full((len(layers), len(all_steps)), np.nan, dtype=float)
    for li, layer in enumerate(layers):
        for entry in flattened[layer]:
            si = step_to_idx[entry['checkpoint_step']]
            H[li, si] = entry.get('high_freq_boundary_crossing_rate', np.nan)
            L[li, si] = entry.get('low_freq_boundary_crossing_rate', np.nan)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300, constrained_layout=True)
    for ax, mat, title in zip(axes, [H, L], ["High freq boundary rate", "Low freq boundary rate"]):
        sns.heatmap(mat, ax=ax, cmap="plasma", cbar=True, vmin=np.nanmin(mat), vmax=np.nanmax(mat))
        ax.set_title(title)
        ax.set_xlabel("Checkpoint step")
        ax.set_ylabel("Layer")
        ax.set_xticks(np.arange(len(all_steps)) + 0.5)
        ax.set_xticklabels(all_steps, rotation=45, ha='right')
        ax.set_yticks(np.arange(len(layers)) + 0.5)
        ax.set_yticklabels(layers)
    fig.suptitle("Polytope Boundary Density Evolution")
    fig.savefig(Path(output_dir) / "polytope_boundary_density_heatmap.pdf", format="pdf")
    plt.close(fig)

def create_checkpoint_evolution_heatmap(results: Dict[str, Any], output_dir: str) -> None:
    """Visualize evolution across checkpoints and layers for core metrics.

    Multiple panels: superposition (compression, boundary), participation ratio (high/low).
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    analyzer = PolytopeAnalyzer()
    flattened = analyzer.flatten_multi_checkpoint_results(results)
    layers = sorted(flattened.keys())
    all_steps = sorted({e['checkpoint_step'] for v in flattened.values() for e in v}, key=lambda s: int(s) if str(s).isdigit() else s)
    step_to_idx = {s: i for i, s in enumerate(all_steps)}
    mats = {
        'high_comp_sup': np.full((len(layers), len(all_steps)), np.nan, dtype=float),
        'low_comp_sup': np.full((len(layers), len(all_steps)), np.nan, dtype=float),
        'high_bound_sup': np.full((len(layers), len(all_steps)), np.nan, dtype=float),
        'low_bound_sup': np.full((len(layers), len(all_steps)), np.nan, dtype=float),
    }
    for li, layer in enumerate(layers):
        for entry in flattened[layer]:
            si = step_to_idx[entry['checkpoint_step']]
            mats['high_comp_sup'][li, si] = entry.get('high_freq_compression_superposition', np.nan)
            mats['low_comp_sup'][li, si] = entry.get('low_freq_compression_superposition', np.nan)
            mats['high_bound_sup'][li, si] = entry.get('high_freq_boundary_superposition', np.nan)
            mats['low_bound_sup'][li, si] = entry.get('low_freq_boundary_superposition', np.nan)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=300, constrained_layout=True)
    titles = [
        (0, 0, 'High freq compression superposition'),
        (0, 1, 'Low freq compression superposition'),
        (1, 0, 'High freq boundary superposition'),
        (1, 1, 'Low freq boundary superposition'),
    ]
    for i, j, title in titles:
        mat = mats['high_comp_sup'] if title.startswith('High freq compression') else \
              mats['low_comp_sup'] if title.startswith('Low freq compression') else \
              mats['high_bound_sup'] if title.startswith('High freq boundary') else \
              mats['low_bound_sup']
        ax = axes[i][j]
        sns.heatmap(mat, ax=ax, cmap="viridis", cbar=True, vmin=np.nanmin(mat), vmax=np.nanmax(mat))
        ax.set_title(title)
        ax.set_xlabel("Checkpoint step")
        ax.set_ylabel("Layer")
        ax.set_xticks(np.arange(len(all_steps)) + 0.5)
        ax.set_xticklabels(all_steps, rotation=45, ha='right')
        ax.set_yticks(np.arange(len(layers)) + 0.5)
        ax.set_yticklabels(layers)
    fig.suptitle("Checkpoint Evolution Heatmaps")
    fig.savefig(Path(output_dir) / "checkpoint_evolution_heatmaps.pdf", format="pdf")
    plt.close(fig)

def polytope_boundary_density_plots(results: Dict[str, Any], output_dir: str) -> None:
    """2D density plots of boundary gradients and crossing patterns with high/low panels."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    analyzer = PolytopeAnalyzer()
    flattened = analyzer.flatten_multi_checkpoint_results(results)
    # Build a long-form dataframe for seaborn
    rows: List[Dict[str, Any]] = []
    for layer, entries in flattened.items():
        for e in entries:
            rows.append({
                'layer': layer,
                'checkpoint': e['checkpoint_step'],
                'group': 'high',
                'boundary_rate': e.get('high_freq_boundary_crossing_rate', np.nan),
                'density_mean': e.get('high_freq_density_mean', np.nan),
            })
            rows.append({
                'layer': layer,
                'checkpoint': e['checkpoint_step'],
                'group': 'low',
                'boundary_rate': e.get('low_freq_boundary_crossing_rate', np.nan),
                'density_mean': e.get('low_freq_density_mean', np.nan),
            })
    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300, constrained_layout=True)
    for ax, grp, title in zip(axes, ['high', 'low'], ["High freq", "Low freq"]):
        sub = df[df['group'] == grp]
        sns.kdeplot(
            data=sub,
            x='density_mean', y='boundary_rate',
            cmap='RdBu', fill=True, thresh=0.05, levels=50, ax=ax
        )
        ax.set_title(f"Boundary density gradients — {title}")
        ax.set_xlabel("Mean density (Hamming/Euclidean)")
        ax.set_ylabel("Boundary crossing rate")
    fig.suptitle("Polytope Boundary Density Plots")
    fig.savefig(Path(output_dir) / "polytope_boundary_density_plots.pdf", format="pdf")
    plt.close(fig)

def superposition_phase_diagrams(results: Dict[str, Any], output_dir: str) -> None:
    """Phase diagrams: compression ratio vs boundary density, highlighting phases."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    analyzer = PolytopeAnalyzer()
    flattened = analyzer.flatten_multi_checkpoint_results(results)
    rows: List[Dict[str, Any]] = []
    for layer, entries in flattened.items():
        for e in entries:
            rows.append({
                'layer': layer,
                'checkpoint': e['checkpoint_step'],
                'group': 'high',
                'compression': e.get('high_freq_compression_superposition', np.nan),
                'boundary_rate': e.get('high_freq_boundary_crossing_rate', np.nan),
            })
            rows.append({
                'layer': layer,
                'checkpoint': e['checkpoint_step'],
                'group': 'low',
                'compression': e.get('low_freq_compression_superposition', np.nan),
                'boundary_rate': e.get('low_freq_boundary_crossing_rate', np.nan),
            })
    df = pd.DataFrame(rows).dropna()
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300, constrained_layout=True)
    sns.scatterplot(data=df, x='compression', y='boundary_rate', hue='group', style='layer', ax=ax)
    ax.set_xlabel("Compression superposition")
    ax.set_ylabel("Boundary crossing rate")
    ax.set_title("Superposition Phase Diagram")
    fig.savefig(Path(output_dir) / "superposition_phase_diagram.pdf", format="pdf")
    plt.close(fig)

def layerwise_metric_distributions(results: Dict[str, Any], output_dir: str) -> None:
    """Layer-wise distributions: PR and superposition with violin/box plots and CIs."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    analyzer = PolytopeAnalyzer()
    flattened = analyzer.flatten_multi_checkpoint_results(results)
    rows: List[Dict[str, Any]] = []
    for layer, entries in flattened.items():
        for e in entries:
            for g in ['high', 'low']:
                rows.append({
                    'layer': layer,
                    'group': g,
                    'checkpoint': e['checkpoint_step'],
                    'participation_ratio': e.get(f'{g}_freq_participation_ratio', np.nan),
                    'compression': e.get(f'{g}_freq_compression_superposition', np.nan)
                })
    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300, constrained_layout=True)
    sns.violinplot(data=df, x='layer', y='participation_ratio', hue='group', split=True, ax=axes[0])
    axes[0].set_title("Participation ratio by layer")
    sns.boxplot(data=df, x='layer', y='compression', hue='group', ax=axes[1])
    axes[1].set_title("Superposition intensity by layer")
    for ax in axes:
        ax.legend(loc='best', fontsize=8)
    fig.savefig(Path(output_dir) / "layerwise_metric_distributions.pdf", format="pdf")
    plt.close(fig)

    
if __name__ == "__main__":
    # run pipeline and get results
    analyzer = PolytopeAnalyzer()
    # Example: load multiple checkpoints if available; fallback to single
    # Disable interference for faster end-to-end runs by default; enable later for deep dives
    analyzer.enable_interference = False
    try:
        results = analyzer.run_multiple_checkpoints(
            checkpoint_paths=[
                "cache/checkpoint_analysis/activation_records_1000.pkl",
                "cache/checkpoint_analysis/activation_records_2000.pkl",
                "cache/checkpoint_analysis/activation_records_3000.pkl",
            ],
            output_dir="cache/polytope_analysis/results",
            parallel_checkpoints=True,
            parallel_layers=True,
        )
    except Exception:
        results = analyzer.run_checkpoint_file(
            checkpoint_path="cache/checkpoint_analysis/activation_records.pkl",
            output_dir="cache/polytope_analysis/results",
            parallel_layers=True,
        )

    # Create ICLR-ready figures
    out_dir = "cache/polytope_analysis/results"
    try:
        participation_ratio_heatmap(results, out_dir)
        polytope_density_heatmap(results, out_dir)
        create_checkpoint_evolution_heatmap(results, out_dir)
        polytope_boundary_density_plots(results, out_dir)
        superposition_phase_diagrams(results, out_dir)
        layerwise_metric_distributions(results, out_dir)
    except Exception as e:
        logger.warning(f"Visualization generation failed: {e}")

