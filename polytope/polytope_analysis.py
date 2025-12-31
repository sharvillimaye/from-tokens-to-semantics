from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from pathlib import Path
import re
import pickle
from typing import Any, Dict, List, Optional
import warnings
from collections import defaultdict
from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib.ticker import MaxNLocator

warnings.filterwarnings('ignore', category=FutureWarning)

def diagnose_activation_data(activations: np.ndarray, label: str = "") -> Dict[str, Any]:
    """Diagnose potential numerical issues in activation data for debugging.
    
    Args:
        activations: Activation matrix to diagnose
        label: Optional label for logging
        
    Returns:
        Dictionary with diagnostic statistics
    """
    finite_mask = np.isfinite(activations)
    finite_data = activations[finite_mask] if np.any(finite_mask) else np.array([])
    
    diagnostics = {
        'label': label,
        'shape': activations.shape,
        'n_samples': activations.shape[0],
        'n_features': activations.shape[1] if len(activations.shape) > 1 else 0,
        'has_nan': bool(np.any(np.isnan(activations))),
        'has_inf': bool(np.any(np.isinf(activations))),
        'pct_nonfinite': float(100.0 * (1.0 - np.sum(finite_mask) / activations.size)),
    }
    
    if len(finite_data) > 0:
        diagnostics.update({
            'min': float(np.min(finite_data)),
            'max': float(np.max(finite_data)),
            'mean': float(np.mean(finite_data)),
            'std': float(np.std(finite_data)),
            'median': float(np.median(finite_data)),
        })
        
        if len(activations.shape) == 2 and activations.shape[0] >= activations.shape[1]:
            try:
                diagnostics['matrix_rank'] = int(np.linalg.matrix_rank(activations))
                diagnostics['condition_number'] = float(np.linalg.cond(activations))
            except:
                diagnostics['matrix_rank'] = 'error'
                diagnostics['condition_number'] = 'error'
    
    return diagnostics

def get_participation_ratio(high_activation_space: Dict[int, np.ndarray], low_activation_space: Dict[int, np.ndarray], 
                           strict_mode: bool = True) -> Dict[str, float]:
    """Compute participation ratio using eigen values from covariance matrix for each layer and frequency category.
    
    Args:
        high_activation_space: Dictionary mapping layer to high-frequency activations
        low_activation_space: Dictionary mapping layer to low-frequency activations
        strict_mode: If True, raise exception on computation failure. If False, return NaN.
    """
    high_participation_ratio = {}
    low_participation_ratio = {}

    for layer, activations in high_activation_space.items():
        high_participation_ratio[layer] = compute_participation_ratio(activations, strict_mode=strict_mode)

    for layer, activations in low_activation_space.items():
        low_participation_ratio[layer] = compute_participation_ratio(activations, strict_mode=strict_mode)

    return high_participation_ratio, low_participation_ratio

def get_polytope_density(high_activation_space: Dict[int, np.ndarray], low_activation_space: Dict[int, np.ndarray], high_binary_pattern_space: Dict[int, np.ndarray], low_binary_pattern_space: Dict[int, np.ndarray]) -> Dict[str, float]:
    """Compute polytope density for each layer and frequency category."""
    high_polytope_density = {}
    low_polytope_density = {}

    for layer, activations in high_activation_space.items():
        high_polytope_density[layer] = compute_polytope_density(activations, high_binary_pattern_space[layer])
    
    for layer, activations in low_activation_space.items():
        low_polytope_density[layer] = compute_polytope_density(activations, low_binary_pattern_space[layer])

    return high_polytope_density, low_polytope_density

def compute_polytope_density(activations: np.ndarray, patterns: np.ndarray, 
                                max_pairs: int = 5000, pair_chunk_size: int = 50000) -> Dict[str, float]:
    """Polytope density computation with chunked processing to prevent memory crashes.
    
    Calculates euclidean distance and hamming distance between pairs of activations and computes density.
    
        
    """
    # Input validation
    if len(activations) == 0 or len(patterns) == 0:
        raise ValueError("Empty activations or patterns provided")
    
    if len(activations) != len(patterns):
        raise ValueError(f"Activations and patterns must have same length, got {len(activations)} and {len(patterns)}")
    
    n_samples = len(activations)
    
    # Check for NaN/Inf in activations
    if not np.all(np.isfinite(activations)):
        logger.warning("Non-finite values detected in activations, cleaning...")
        activations = np.where(np.isfinite(activations), activations, 0.0)

    activations_std = zscore_for_distance(activations)


    n_possible_pairs = n_samples * (n_samples - 1) // 2

    actual_max_pairs = int(min(max_pairs, n_possible_pairs))

    # Sample pairs without constructing all O(n^2) pairs (ensure i<j and uniqueness)
    i_idx = np.random.randint(0, n_samples, size=actual_max_pairs, dtype=np.int32)
    j_idx = np.random.randint(0, n_samples, size=actual_max_pairs, dtype=np.int32)
    same_mask = (i_idx == j_idx)
    while np.any(same_mask):
        j_idx[same_mask] = np.random.randint(0, n_samples, size=int(np.sum(same_mask)), dtype=np.int32)
        same_mask = (i_idx == j_idx)
    swap_mask = i_idx > j_idx
    if np.any(swap_mask):
        tmp = i_idx[swap_mask].copy()
        i_idx[swap_mask] = j_idx[swap_mask]
        j_idx[swap_mask] = tmp
    pairs = np.stack([i_idx, j_idx], axis=1)
    if len(pairs) > 0:
        pairs = np.unique(pairs, axis=0)

    # Ensure patterns are boolean for efficient Hamming computation
    patterns_bool = patterns.astype(bool, copy=False)

    # Compute pairwise distances in chunks (vectorized)
    euclidean_distances = []
    hamming_distances = []
    total_pairs = len(pairs)
    for start in range(0, total_pairs, max(1, pair_chunk_size)):
        end = min(start + max(1, pair_chunk_size), total_pairs)
        ii = pairs[start:end, 0]
        jj = pairs[start:end, 1]
        diffs = activations_std[ii] - activations_std[jj]
        euclid = _rowwise_norm(diffs)
        ham = np.count_nonzero(patterns_bool[ii] != patterns_bool[jj], axis=1)
        euclidean_distances.append(euclid)
        hamming_distances.append(ham)

    euclidean_distances = np.concatenate(euclidean_distances) if len(euclidean_distances) > 1 else euclidean_distances[0]
    hamming_distances = np.concatenate(hamming_distances) if len(hamming_distances) > 1 else hamming_distances[0]
    
    # Compute density (hamming / euclidean)
    densities = hamming_distances / (euclidean_distances + 1e-12)
    


    return {
        'density_mean': float(np.mean(densities)),
        'density_std': float(np.std(densities)),
    }

def compute_participation_ratio(activations: np.ndarray, strict_mode: bool = True) -> float:
    """Compute participation ratio with research-grade error handling.
    
    Args:
        activations: Activation matrix (n_samples, n_features)
        strict_mode: If True, raise exception on failure. If False, return NaN with error logging.
        
    Returns:
        Participation ratio (PR >= 1.0)
        
    Raises:
        ValueError: If computation is numerically unstable (strict_mode=True)
        np.linalg.LinAlgError: If PCA fails (strict_mode=True)
    """
    try:
        activations_clean = np.where(np.isfinite(activations), activations, 0.0)
        activations_clipped = np.clip(activations_clean, -1e3, 1e3)
        
        # Manual standardization with overflow protection (more robust than StandardScaler)
        with np.errstate(over='ignore', invalid='ignore'):
            mean = np.mean(activations_clipped, axis=0)
            # Compute std with intermediate clipping to prevent overflow
            centered = activations_clipped - mean
            centered_clipped = np.clip(centered, -100, 100)  # Clip before squaring
            variance = np.mean(centered_clipped ** 2, axis=0)
            std = np.sqrt(variance)
            std = np.where(std > 1e-8, std, 1.0)  # Avoid division by zero
            activations_std = centered / std
            
        activations_std = np.where(np.isfinite(activations_std), activations_std, 0.0)
        activations_std = np.clip(activations_std, -6, 6)
        
        # Use standard PCA, but with explicit error handling
        # For very large or unstable datasets, could switch to IncrementalPCA
        pca = PCA()
        try:
            pca.fit(activations_std)
        except np.linalg.LinAlgError as e:
            error_msg = (
                f"PCA decomposition failed: {e}\n"
                f"  This suggests the activation matrix is singular or ill-conditioned\n"
                f"  Consider: 1) Checking for duplicate samples, 2) Adding regularization, 3) Using more data"
            )
            logger.error(error_msg)
            if strict_mode:
                raise ValueError(error_msg)
            return np.nan
        eigenvals = pca.explained_variance_
        eigenvals = np.where(np.isfinite(eigenvals), eigenvals, 0.0)
        eigenvals = np.maximum(eigenvals, 1e-12)
        
        with np.errstate(over='ignore', invalid='ignore'):
            numerator = np.sum(eigenvals) ** 2
            denominator = np.sum(eigenvals ** 2)
        
        # CRITICAL: No silent fallback - diagnose and fail explicitly
        if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator < 1e-12:
            error_msg = (
                f"Participation ratio numerically unstable:\n"
                f"  Shape: {activations.shape}\n"
                f"  Numerator: {numerator:.6e}\n"
                f"  Denominator: {denominator:.6e}\n"
                f"  Eigenvalue range: [{np.min(eigenvals):.6e}, {np.max(eigenvals):.6e}]\n"
                f"  Activation stats: min={np.min(activations_clean):.3f}, max={np.max(activations_clean):.3f}, "
                f"mean={np.mean(activations_clean):.3f}, std={np.std(activations_clean):.3f}\n"
                f"  Matrix rank: {np.linalg.matrix_rank(activations_std)}/{min(activations.shape)}"
            )
            logger.error(error_msg)
            if strict_mode:
                raise ValueError(error_msg)
            return np.nan
        
        participation_ratio = numerator / denominator
        
        if not np.isfinite(participation_ratio):
            error_msg = (
                f"Participation ratio result is non-finite: {participation_ratio}\n"
                f"  Numerator: {numerator:.6e}, Denominator: {denominator:.6e}\n"
                f"  Shape: {activations.shape}"
            )
            logger.error(error_msg)
            if strict_mode:
                raise ValueError(error_msg)
            return np.nan
        
        participation_ratio = np.clip(participation_ratio, 1.0, float(len(eigenvals)))
        return float(participation_ratio)
        
    except (np.linalg.LinAlgError, ValueError, RuntimeWarning) as e:
        error_msg = (
            f"Participation ratio computation failed with {type(e).__name__}: {str(e)}\n"
            f"  Shape: {activations.shape}\n"
            f"  Has NaN: {np.any(np.isnan(activations))}\n"
            f"  Has Inf: {np.any(np.isinf(activations))}"
        )
        logger.error(error_msg)
        if strict_mode:
            raise
        return np.nan
    
def zscore_for_distance(activations: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Z-score features to stabilize Euclidean distance computations with overflow protection."""
    activations_clean = np.where(np.isfinite(activations), activations, 0.0)
    clip_bound = 1e3  
    X = np.clip(activations_clean, -clip_bound, clip_bound)
    
    X = X.astype(np.float64, copy=False)
    
    with np.errstate(over='ignore', invalid='ignore'):
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
    
    mean = np.where(np.isfinite(mean), mean, 0.0)
    std = np.where(np.isfinite(std) & (std >= eps), std, 1.0)
    
    with np.errstate(over='ignore', invalid='ignore'):
        z_scored = (X - mean) / std
    
    z_scored = np.where(np.isfinite(z_scored), z_scored, 0.0)
    return np.clip(z_scored, -6, 6)

def _rowwise_norm(X: np.ndarray) -> np.ndarray:
    """Compute row-wise L2 norms with clipping to avoid overflow."""
    X_clean = np.where(np.isfinite(X), X, 0.0)
    X64 = X_clean.astype(np.float64, copy=False)
    X64 = np.clip(X64, -1e3, 1e3)
    with np.errstate(over='ignore', invalid='ignore'):
        norms_sq = np.sum(X64 * X64, axis=1)
        norms = np.sqrt(norms_sq)
    norms = np.where(np.isfinite(norms), norms, 0.0)
    return norms

def load_checkpoint_data(checkpoint_path: str) -> Dict[str, Any]:
    with open(checkpoint_path, 'rb') as f:
        data = pickle.load(f)
    
    # Handle both dict and list formats
    if isinstance(data, dict):
        records = data.get('records', data)
        metadata = data.get('metadata', {})
    else:
        records = data
        metadata = {}
    return {'records': records, 'metadata': metadata}

def organize_by_frequency(records: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Organize records by frequency category (legacy method - use organize_by_layer_and_frequency for layer-aware analysis)."""
    organized = {'high_freq': [], 'medium_freq': [], 'low_freq': []}

    for record in records:
        cat = record.get('category', record.get('frequency_category', 'unknown')).lower()
        
        if 'high' in cat:
            organized['high_freq'].append(record)
        elif 'medium' in cat:
            organized['medium_freq'].append(record)
        elif 'low' in cat:
            organized['low_freq'].append(record)

    for cat, recs in organized.items():
        if recs:
            logger.info(f"{cat}: {len(recs)} records")
    
    return organized

def organize_by_layer_and_frequency(records: List[Dict[str, Any]]) -> Dict[int, Dict[str, List[Dict[str, Any]]]]:
    """Organize records by layer and frequency category for proper layer-wise analysis."""
    organized = {}
    
    for record in records:
        layer = record.get('layer')
            
        if layer not in organized:
            organized[layer] = {'high_freq': [], 'medium_freq': [], 'low_freq': []}
        
        cat = record.get('frequency_category').lower()
        
        if 'high' in cat:
            organized[layer]['high_freq'].append(record)
        elif 'medium' in cat:
            organized[layer]['medium_freq'].append(record)
        elif 'low' in cat:
            organized[layer]['low_freq'].append(record)
    
    for layer in sorted(organized.keys()):
        layer_counts = {cat: len(recs) for cat, recs in organized[layer].items() if recs}
        if layer_counts:
            logger.info(f"Layer {layer}: {layer_counts}")
    
    return organized

def organize_by_checkpoint_layer_frequency(records: List[Dict[str, Any]]) -> Dict[str, Dict[int, Dict[str, List[Dict[str, Any]]]]]:
    """Organize records by checkpoint, then layer, then frequency category.
    
    This is the primary organization function for multi-checkpoint analysis.
    
    Args:
        records: List of activation records with checkpoint_step, layer, and frequency_category fields
        
    Returns:
        Nested dictionary: {checkpoint_step: {layer: {'high_freq': [...], 'low_freq': [...]}}}
    """
    organized = defaultdict(lambda: defaultdict(lambda: {'high_freq': [], 'low_freq': []}))
    
    for record in records:
        checkpoint = str(record.get('checkpoint_step', 'unknown'))
        layer = record.get('layer')
        cat = record.get('frequency_category', '').lower()
        if 'high' in cat:
            organized[checkpoint][layer]['high_freq'].append(record)
        elif 'low' in cat:
            organized[checkpoint][layer]['low_freq'].append(record)
    
    organized = {k: dict(v) for k, v in organized.items()}
    
    logger.info(f"Organized {len(records)} records across {len(organized)} checkpoints")
    for checkpoint in sorted(organized.keys()):
        n_layers = len(organized[checkpoint])
        total_records = sum(
            len(organized[checkpoint][layer]['high_freq']) + len(organized[checkpoint][layer]['low_freq'])
            for layer in organized[checkpoint]
        )
        logger.info(f"  Checkpoint {checkpoint}: {n_layers} layers, {total_records} records")
    
    return organized

def get_activation_spaces(records: List[Dict[str, Any]]) -> Dict[int, np.ndarray]:
    """Get activation, and binary pattern spaces for each layer and frequency category."""
    organized = organize_by_layer_and_frequency(records) # gives me layers with high, medium, and low frequency categories
    high_activation_space = {}
    low_activation_space = {}
    high_binary_pattern_space = {}
    low_binary_pattern_space = {}

    for layer, freq_groups in organized.items():
        for freq_type, recs in freq_groups.items():
            activations = np.stack([r['activation_vector'] for r in recs])
            binary_patterns = np.stack([r['binary_pattern'] for r in recs])
            if freq_type == 'high_freq':
                high_activation_space[layer] = activations
                high_binary_pattern_space[layer] = binary_patterns
            elif freq_type == 'low_freq':
                low_activation_space[layer] = activations
                low_binary_pattern_space[layer] = binary_patterns
    
    return high_activation_space, low_activation_space, high_binary_pattern_space, low_binary_pattern_space

def bootstrap_resample(records: List[Dict[str, Any]], n_bootstrap: int = 100, 
                       sample_fraction: float = 0.8, random_seed: Optional[int] = None) -> List[List[Dict[str, Any]]]:
    """Generate bootstrap samples for confidence interval estimation.
    
    Args:
        records: List of activation records to resample
        n_bootstrap: Number of bootstrap samples to generate
        sample_fraction: Fraction of data to sample in each bootstrap (with replacement)
        random_seed: Random seed for reproducibility
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    n_samples = len(records)
    sample_size = max(1, int(n_samples * sample_fraction))
    
    bootstrap_samples = []
    for _ in range(n_bootstrap):
        indices = np.random.choice(n_samples, size=sample_size, replace=True)
        bootstrap_sample = [records[i] for i in indices]
        bootstrap_samples.append(bootstrap_sample)
    
    return bootstrap_samples

def _bootstrap_iteration(args):
    """Single bootstrap iteration for parallel processing.
    
    Args:
        args: Tuple of (activations, patterns, indices, max_pairs)
    
    Returns:
        Tuple of (density_estimate, pr_estimate) or (None, None) on failure
    """
    activations, patterns, indices, max_pairs = args
    try:
        boot_activations = activations[indices]
        boot_patterns = patterns[indices]
        
        # Compute density
        density_result = compute_polytope_density(boot_activations, boot_patterns, max_pairs=max_pairs)
        density_estimate = density_result['density_mean']
        
        # Compute participation ratio
        pr_estimate = compute_participation_ratio(boot_activations)
        
        return (density_estimate, pr_estimate)
    except Exception as e:
        logger.debug(f"Bootstrap iteration failed: {e}")
        return (None, None)


def compute_metrics_with_confidence_intervals(
    activations: np.ndarray, 
    patterns: np.ndarray,
    n_bootstrap: int = 5,
    confidence_level: float = 0.95,
    max_pairs: int = 500,
    random_seed: Optional[int] = None,
    ci_method: str = 'bootstrap',
    n_batches: int = 5,
    n_workers: Optional[int] = None
) -> Dict[str, Any]:
    """Compute polytope and participation metrics with confidence intervals.
    
    Supports two CI methods:
      - 'bootstrap' (default): standard bootstrap resampling with replacement
      - 'batch_means': split samples into batches without replacement and use batch means
    
    Args:
        activations: Activation matrix of shape (n_samples, n_features)
        patterns: Binary pattern matrix of shape (n_samples, n_features)
        n_bootstrap: Number of bootstrap iterations (when ci_method='bootstrap')
        confidence_level: Confidence level for intervals (e.g., 0.95 for 95% CI)
        max_pairs: Maximum pairs for density computation per resample/batch
        random_seed: Random seed for reproducibility
        ci_method: 'bootstrap' or 'batch_means'
        n_batches: Number of batches (when ci_method='batch_means')
        n_workers: Number of parallel workers for resampling (bootstrap only)
        
    Returns:
        Dictionary containing:
            - density_mean, density_std, density_ci_lower, density_ci_upper
            - participation_ratio_mean (single value, no CI since deterministic)
            - participation_ratio_std (set to 0.0)
            - n_samples: number of samples used
            - n_bootstrap_runs: number of resamples for density (bootstrap iters or batches)
            - ci_method: method used for CI
    """
    if ci_method not in {'bootstrap', 'batch_means'}:
        raise ValueError(f"Unsupported ci_method: {ci_method}")

    if random_seed is not None:
        np.random.seed(random_seed)

    n_samples = len(activations)
    
    # Compute participation ratio ONCE on full dataset (deterministic, no resampling needed)
    participation_ratio = compute_participation_ratio(activations, strict_mode=False)
    
    # Validate and diagnose PR computation
    if np.isnan(participation_ratio):
        logger.error(f"Participation ratio failed for sample with {n_samples} samples")
        diagnostics = diagnose_activation_data(activations, label=f"PR_failure_{n_samples}_samples")
        logger.error(f"Diagnostics: {diagnostics}")
    else:
        logger.debug(f"Computed participation ratio on full dataset: {participation_ratio:.4f}")

    if ci_method == 'batch_means':
        # Batch-means CI: shuffle and split into n_batches (without replacement)
        # Only for density (participation ratio already computed on full dataset)
        batches = max(2, int(min(n_batches, n_samples // 2)))
        if batches < n_batches:
            logger.debug(f"Reducing n_batches from {n_batches} to {batches} due to small sample size ({n_samples})")
        rng = np.random.default_rng(random_seed)
        perm = rng.permutation(n_samples)
        batch_sizes = [n_samples // batches] * batches
        for i in range(n_samples % batches):
            batch_sizes[i] += 1
        offsets = np.cumsum([0] + batch_sizes)

        density_means = []

        logger.debug(f"Running batch-means with {batches} batches for density CI (confidence={confidence_level})")
        for b in range(batches):
            start = offsets[b]
            end = offsets[b + 1]
            if end - start < 2:
                continue
            idx = perm[start:end]
            batch_act = activations[idx]
            batch_pat = patterns[idx]
            density_result = compute_polytope_density(batch_act, batch_pat, max_pairs=max_pairs)
            density_means.append(density_result['density_mean'])

        density_means = np.array(density_means, dtype=float)

        # Filter non-finite
        density_means = density_means[np.isfinite(density_means)]

        alpha = 1 - confidence_level
        if len(density_means) > 0:
            density_mean = float(np.mean(density_means))
            density_std = float(np.std(density_means, ddof=1) if len(density_means) > 1 else 0.0)
            density_ci_lower = float(np.percentile(density_means, 100 * alpha / 2))
            density_ci_upper = float(np.percentile(density_means, 100 * (1 - alpha / 2)))
        else:
            density_mean = density_std = density_ci_lower = density_ci_upper = 0.0

        return {
            'density_mean': density_mean,
            'density_std': density_std,
            'density_ci_lower': density_ci_lower,
            'density_ci_upper': density_ci_upper,
            'participation_ratio_mean': participation_ratio,
            'participation_ratio_std': 0.0,  # No CI for participation ratio (deterministic)
            'participation_ratio_ci_lower': participation_ratio,
            'participation_ratio_ci_upper': participation_ratio,
            'n_samples': n_samples,
            'n_bootstrap_runs': int(len(density_means)),
            'ci_method': 'batch_means',
        }

    # Bootstrap branch - only for density (participation ratio already computed)
    if random_seed is not None:
        np.random.seed(random_seed)
    
    sample_size = max(10, int(n_samples * 0.8))  # Use 80% of data for each bootstrap
    
    # Generate all bootstrap indices upfront for reproducibility
    bootstrap_indices = [
        np.random.choice(n_samples, size=sample_size, replace=True)
        for _ in range(n_bootstrap)
    ]
    
    # Prepare arguments for parallel processing (only density, not PR)
    args_list = [
        (activations, patterns, indices, max_pairs)
        for indices in bootstrap_indices
    ]
    
    # Determine number of workers - cap at CPU count for efficiency
    if n_workers is None:
        n_workers = mp.cpu_count()
    else:
        # Cap at 2x CPU count to avoid overhead
        n_workers = min(n_workers, mp.cpu_count() * 2)
    
    # Run bootstrap iterations in parallel for density only
    density_estimates = []
    
    logger.debug(f"Running {n_bootstrap} bootstrap iterations for density CI with {n_workers} workers (confidence={confidence_level})")
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(_bootstrap_iteration, args) for args in args_list]
        
        for future in as_completed(futures):
            density_est, _ = future.result()  # Ignore PR estimate
            if density_est is not None:
                density_estimates.append(density_est)
    
    # Convert to array
    density_estimates = np.array(density_estimates)
    
    # Filter out non-finite values
    density_estimates = density_estimates[np.isfinite(density_estimates)]
    
    if len(density_estimates) < n_bootstrap * 0.5:
        logger.warning(f"Only {len(density_estimates)}/{n_bootstrap} bootstrap samples succeeded")
    
    # Compute statistics
    alpha = 1 - confidence_level
    
    # Density statistics
    if len(density_estimates) > 0:
        density_mean = float(np.mean(density_estimates))
        density_std = float(np.std(density_estimates))
        density_ci_lower = float(np.percentile(density_estimates, 100 * alpha / 2))
        density_ci_upper = float(np.percentile(density_estimates, 100 * (1 - alpha / 2)))
    else:
        density_mean = density_std = density_ci_lower = density_ci_upper = 0.0
    
    return {
        'density_mean': density_mean,
        'density_std': density_std,
        'density_ci_lower': density_ci_lower,
        'density_ci_upper': density_ci_upper,
        'participation_ratio_mean': participation_ratio,
        'participation_ratio_std': 0.0,  # No CI for participation ratio (deterministic)
        'participation_ratio_ci_lower': participation_ratio,
        'participation_ratio_ci_upper': participation_ratio,
        'n_samples': n_samples,
        'n_bootstrap_runs': len(density_estimates),
        'ci_method': 'bootstrap',
    }

def validate_layer_metrics(metrics: Dict[str, Any], checkpoint: str, layer: int, 
                          freq_type: str, raise_on_invalid: bool = True) -> bool:
    """Validate computed metrics for research integrity.
    
    Args:
        metrics: Metrics dictionary from compute_metrics_with_confidence_intervals
        checkpoint: Checkpoint identifier
        layer: Layer number
        freq_type: 'high_freq' or 'low_freq'
        raise_on_invalid: If True, raise exception on invalid data
        
    Returns:
        True if valid, False otherwise
        
    Raises:
        ValueError: If metrics are invalid and raise_on_invalid=True
    """
    issues = []
    
    pr_mean = metrics.get('participation_ratio_mean', np.nan)
    density_mean = metrics.get('density_mean', np.nan)
    
    if np.isnan(pr_mean):
        issues.append(f"Participation ratio is NaN")
    elif pr_mean < 1.0:
        issues.append(f"Participation ratio {pr_mean:.3f} < 1.0 (impossible)")
    
    if np.isnan(density_mean):
        issues.append(f"Density mean is NaN")
    elif density_mean < 0:
        issues.append(f"Density {density_mean:.3f} < 0 (impossible)")
    
    if metrics.get('n_samples', 0) < 10:
        issues.append(f"Too few samples: {metrics.get('n_samples', 0)}")
    
    if issues:
        error_msg = (
            f"VALIDATION FAILED: Checkpoint {checkpoint}, Layer {layer}, {freq_type}\n"
            + "\n".join(f"  - {issue}" for issue in issues)
        )
        logger.error(error_msg)
        if raise_on_invalid:
            raise ValueError(error_msg)
        return False
    
    return True
    
def polytope_analysis(
    path_to_records: str, 
    n_bootstrap: int = 5,
    confidence_level: float = 0.95,
    max_pairs: int = 500,
    random_seed: Optional[int] = None,
    n_workers: Optional[int] = None,
    parallel_checkpoints: bool = False,
    ci_method: str = 'batch_means',
    n_batches: int = 5,
    sample_cap: Optional[int] = None,
    strict_mode: bool = False,
    validate_metrics: bool = True
) -> Dict[str, Any]:
    """Run polytope analysis with statistical robustness on checkpoint data.
    
    This is the main analysis function that organizes records by checkpoint→layer→frequency
    and computes polytope metrics with confidence intervals for research-grade analysis.
    
    Args:
        path_to_records: Path to pickle file containing checkpoint records
        n_bootstrap: Number of bootstrap iterations for CI estimation
        confidence_level: Confidence level for intervals (default 0.95 for 95% CI)
        max_pairs: Maximum number of pairs for density computation
        random_seed: Random seed for reproducibility
        n_workers: Number of parallel workers for bootstrap (default: auto-detect)
        parallel_checkpoints: If True, process checkpoints in parallel (default: False)
        ci_method: 'bootstrap' or 'batch_means' (default: 'batch_means' for speed)
        n_batches: Number of batches when ci_method='batch_means'
        strict_mode: If True, raise exceptions on computational failures (default: False for robustness)
        validate_metrics: If True, validate all computed metrics for research integrity (default: True)
        
    Returns:
        Dictionary with structure:
        {
            'results': {
                checkpoint_step: {
                    layer: {
                        'high_freq': {
                            'density_mean': float,
                            'density_std': float,
                            'density_ci_lower': float,
                            'density_ci_upper': float,
                            'participation_ratio_mean': float,
                            'participation_ratio_std': float,
                            'participation_ratio_ci_lower': float,
                            'participation_ratio_ci_upper': float,
                            'n_samples': int,
                            'n_bootstrap_runs': int
                        },
                        'low_freq': {...}
                    }
                }
            },
            'metadata': {
                'n_bootstrap': int,
                'confidence_level': float,
                'total_records': int,
                'n_checkpoints': int,
                'random_seed': int,
                'n_workers': int
            }
        }
    """
    if ci_method == 'bootstrap':
        logger.info(f"Starting polytope analysis with {n_bootstrap} bootstrap iterations")
    else:
        logger.info(f"Starting polytope analysis using batch-means (n_batches={n_batches})")
    logger.info(f"Confidence level: {confidence_level}, max_pairs: {max_pairs}, n_runs: {n_batches if ci_method=='batch_means' else n_bootstrap}")
    
    # Set workers to CPU count if not specified, capped at 2x CPU count
    if n_workers is None:
        n_workers = mp.cpu_count()
    else:
        n_workers = min(n_workers, mp.cpu_count() * 2)
    logger.info(f"Using {n_workers} parallel workers (CPU count: {mp.cpu_count()})")
    
    # Load data
    try:
        data = load_checkpoint_data(path_to_records)
        records = data['records']
        input_metadata = data.get('metadata', {})
    except Exception as e:
        logger.error(f"Failed to load checkpoint data: {e}")
        raise
    
    logger.info(f"Loaded {len(records)} total records")
    
    # Organize by checkpoint → layer → frequency
    organized = organize_by_checkpoint_layer_frequency(records)
    
    # Initialize results structure
    results = {}
    
    # Process each checkpoint
    for checkpoint in sorted(organized.keys()):
        logger.info(f"Processing checkpoint {checkpoint}")
        results[checkpoint] = {}
        
        # Process each layer
        for layer in sorted(organized[checkpoint].keys()):
            results[checkpoint][layer] = {}
            
            # Process high and low frequency groups
            for freq_type in ['high_freq', 'low_freq']:
                freq_records = organized[checkpoint][layer][freq_type]

                # Optional subsampling for faster runs
                if sample_cap is not None and len(freq_records) > sample_cap:
                    rng = np.random.default_rng(random_seed)
                    idx = rng.choice(len(freq_records), size=sample_cap, replace=False)
                    freq_records = [freq_records[i] for i in idx]
            
                # Extract activations and patterns
                activations = np.stack([r['activation_vector'] for r in freq_records])
                patterns = np.stack([r['binary_pattern'] for r in freq_records])
                
                # Compute metrics with confidence intervals
                metrics = compute_metrics_with_confidence_intervals(
                    activations=activations,
                    patterns=patterns,
                    n_bootstrap=n_bootstrap,
                    confidence_level=confidence_level,
                    max_pairs=max_pairs,
                    random_seed=random_seed,
                    ci_method=ci_method,
                    n_batches=n_batches,
                    n_workers=n_workers
                )
                
                # Validate metrics for research integrity
                if validate_metrics:
                    validate_layer_metrics(metrics, checkpoint, layer, freq_type, raise_on_invalid=strict_mode)
                
                results[checkpoint][layer][freq_type] = metrics
                
                logger.info(f"  Layer {layer} {freq_type}: "
                            f"density={metrics['density_mean']:.3f}±{metrics['density_std']:.3f}, "
                            f"PR={metrics['participation_ratio_mean']:.2f}±{metrics['participation_ratio_std']:.2f}")
                    
    metadata = {
        'n_bootstrap': n_bootstrap,
        'confidence_level': confidence_level,
        'max_pairs': max_pairs,
        'total_records': len(records),
        'n_checkpoints': len(results),
        'checkpoints': sorted(results.keys()),
        'random_seed': random_seed,
        'n_workers': n_workers,
        'ci_method': ci_method,
        'n_batches': n_batches,
        'parallel_checkpoints': parallel_checkpoints,
        'input_metadata': input_metadata,
        'sample_cap': sample_cap,
        'strict_mode': strict_mode,
        'validate_metrics': validate_metrics
    }
    
    logger.info(f"Analysis complete: {len(results)} checkpoints processed")
    
    return {
        'results': results,
        'metadata': metadata
    }
    

def _setup_presentation_style():
    """Configure matplotlib and seaborn for PowerPoint presentation-quality plots."""
    palette = sns.color_palette("Paired", n_colors=20)
    sns.set_theme(style="whitegrid", font_scale=1.35, palette="tab10")
    
    # Configure matplotlib style for presentations
    plt.rcParams["lines.solid_capstyle"] = "projecting"
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.family"] = "sans-serif"  # Fallback if XCharter not available
    plt.rcParams["font.size"] = 14
    plt.rcParams["font.weight"] = "semibold"
    plt.rcParams["axes.titleweight"] = "semibold"
    plt.rcParams["axes.labelweight"] = "semibold"
    plt.rcParams["axes.linewidth"] = 2.2
    plt.rcParams["xtick.major.width"] = 2.0
    plt.rcParams["ytick.major.width"] = 2.0
    plt.rcParams["xtick.minor.width"] = 1.6
    plt.rcParams["ytick.minor.width"] = 1.6
    
    return palette


def _style_ax(ax, xlabel="X Axis", ylabel="Y Axis", title=None):
    """Apply consistent styling to axes for presentation-quality plots."""
    ax.grid(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    
    # Darker, thicker spines for projection
    for s in ax.spines.values():
        s.set_color("black")
        s.set_linewidth(2.2)
    
    # Larger ticks for PowerPoint
    ax.tick_params(axis="both", which="major", pad=0, colors="0.4",
                   labelsize=14, length=8, width=2.0)
    ax.tick_params(axis="both", which="minor", length=5, width=1.6)
    
    # Axis labels
    ax.set_xlabel(xlabel, fontsize=16, labelpad=8, weight="semibold")
    ax.set_ylabel(ylabel, fontsize=16, labelpad=8, weight="semibold")
    
    # Optional title
    if title:
        ax.set_title(title, fontsize=18, weight="semibold", pad=15)


def polytope_graphs(results: Dict[str, Any], output_dir: str = "polytope_graphs") -> None:
    """Generate graphs for polytope analysis.
    Graphs:
    Difference heatmap (including error bars, CI, etc.)
    layerwise charts comparing participation ratio vs checkpoints, density vs checkpoints (including error bars, CI, etc.)
    """
    from pathlib import Path
    
    # Setup presentation styling
    palette = _setup_presentation_style()
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Extract data from results
    analysis_results = results.get('results', {})
    if not analysis_results:
        logger.error("No analysis results found")
        return
    
    # Helpers for natural numeric sorting of checkpoints (e.g., "1000", "step_2000")
    def _parse_checkpoint_numeric(label: Any) -> int:
        try:
            return int(label)
        except (TypeError, ValueError):
            digits = re.findall(r"\d+", str(label))
            return int(digits[-1]) if digits else float("inf")

    # Parse results into structured format with numerically ordered checkpoints
    checkpoint_labels = sorted(analysis_results.keys(), key=_parse_checkpoint_numeric)
    checkpoint_numeric = [
        _parse_checkpoint_numeric(lbl) for lbl in checkpoint_labels
    ]

    # Keep layers ordered (assumes layers are numeric already)
    layers = sorted(next(iter(analysis_results.values())).keys())
    
    logger.info(f"Creating visualizations for {len(checkpoint_labels)} checkpoints and {len(layers)} layers")
    
    # Create visualizations
    _plot_difference_heatmaps(analysis_results, checkpoint_labels, layers, output_path, palette)
    _plot_layerwise_metrics(analysis_results, checkpoint_labels, checkpoint_numeric, layers, output_path, palette)
    
    logger.info(f"Visualizations saved to {output_path}")


def _plot_difference_heatmaps(analysis_results: Dict, checkpoints: List, layers: List, 
                              output_path: Path, palette) -> None:
    """Create heatmaps showing difference between high_freq and low_freq groups."""
    metrics = [
        ('density_mean', 'Polytope Density Difference (High - Low)'),
        ('participation_ratio_mean', 'Participation Ratio Difference (High - Low)')
    ]
    
    for metric_key, title in metrics:
        # Build difference matrix: layers x checkpoints
        diff_matrix = np.zeros((len(layers), len(checkpoints)))
        
        for i, layer in enumerate(layers):
            for j, checkpoint in enumerate(checkpoints):
                high_val = analysis_results[checkpoint][layer]['high_freq'].get(metric_key, 0)
                low_val = analysis_results[checkpoint][layer]['low_freq'].get(metric_key, 0)
                diff_matrix[i, j] = high_val - low_val
        
        # Dynamic figure sizing for clarity with many checkpoints/layers
        width = min(20.0, max(8.0, 0.35 * len(checkpoints) + 6.0))
        height = min(16.0, max(6.0, 0.25 * len(layers) + 4.0))
        fig, ax = plt.subplots(figsize=(width, height))
        
        # Use diverging colormap centered at 0
        vmax = np.abs(diff_matrix).max()
        vmin = -vmax
        
        im = ax.imshow(
            diff_matrix,
            aspect='auto',
            cmap='coolwarm',
            vmin=vmin,
            vmax=vmax,
            origin='lower',
            interpolation='nearest'
        )
        
        # Apply presentation styling
        _style_ax(ax, xlabel='Checkpoint', ylabel='Layer', title=title)
        
        # Set ticks with presentation-friendly sizes
        if len(checkpoints) > 10:
            step = max(1, len(checkpoints) // 10)
            xticks_idx = list(range(0, len(checkpoints), step))
        else:
            xticks_idx = list(range(len(checkpoints)))
        ax.set_xticks(xticks_idx)
        ax.set_xticklabels([str(checkpoints[i]) for i in xticks_idx], 
                          rotation=45, ha='right', fontsize=14, weight='semibold')
        
        if len(layers) > 15:
            step = max(1, len(layers) // 15)
            yticks_idx = list(range(0, len(layers), step))
        else:
            yticks_idx = list(range(len(layers)))
        ax.set_yticks(yticks_idx)
        ax.set_yticklabels([str(layers[i]) for i in yticks_idx], fontsize=14, weight='semibold')
        
        # Add colorbar with larger font
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Difference', fontsize=16, weight='semibold')
        cbar.ax.tick_params(labelsize=14)
        
        plt.tight_layout()
        filename = metric_key.replace('_mean', '') + '_difference_heatmap.pdf'
        plt.savefig(output_path / filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved {filename}")


def _plot_layerwise_metrics(analysis_results: Dict, checkpoints: List, checkpoint_numeric: List[float], layers: List, 
                            output_path: Path, palette) -> None:
    """Create layerwise line charts comparing metrics across checkpoints with error bars."""
    metrics = [
        ('density_mean', 'density_std', 'Polytope Density'),
        ('participation_ratio_mean', 'participation_ratio_std', 'Participation Ratio')
    ]
    
    # Use presentation-friendly colors from palette
    colors = {'high_freq': palette[0], 'low_freq': palette[2]}
    markers = {'high_freq': 'o', 'low_freq': 's'}
    
    # Select representative layers to avoid clutter
    max_layers_to_plot = 6
    if len(layers) > max_layers_to_plot:
        layer_indices = np.linspace(0, len(layers) - 1, max_layers_to_plot, dtype=int)
        selected_layers = [layers[i] for i in layer_indices]
    else:
        selected_layers = layers
    
    for metric_mean, metric_std, title in metrics:
        # Create subplot for each layer
        n_layers = len(selected_layers)
        ncols = min(3, n_layers)
        nrows = int(np.ceil(n_layers / ncols))
        
        # Scale figure size with 16:9 aspect ratio consideration
        fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 3.5 * nrows))
        if n_layers == 1:
            axes = np.array([axes])
        axes = axes.flatten() if n_layers > 1 else axes
        
        for idx, layer in enumerate(selected_layers):
            ax = axes[idx] if n_layers > 1 else axes[0]
            
            # Track data quality per frequency type
            data_quality = {'high_freq': 0, 'low_freq': 0}
            
            for freq_type in ['high_freq', 'low_freq']:
                means = []
                stds = []
                valid_x_vals = []
                
                for i, checkpoint in enumerate(checkpoints):
                    metrics_data = analysis_results[checkpoint][layer][freq_type]
                    mean_val = metrics_data.get(metric_mean, 0)
                    std_val = metrics_data.get(metric_std, 0)
                    
                    # Skip NaN values with warning
                    if np.isnan(mean_val):
                        logger.warning(f"NaN value detected: checkpoint={checkpoint}, layer={layer}, "
                                      f"freq={freq_type}, metric={metric_mean}")
                        continue
                    
                    means.append(mean_val)
                    stds.append(std_val)
                    valid_x_vals.append(checkpoint_numeric[i])
                
                # Track valid data points for this frequency type
                data_quality[freq_type] = len(means)
                
                means = np.array(means)
                stds = np.array(stds)
                x_vals = np.array(valid_x_vals, dtype=float)
                
                # Plot with presentation-quality thick lines (only if we have data)
                if len(x_vals) > 0:
                    color = colors[freq_type]
                    marker = markers[freq_type]
                    label = freq_type.replace('_', ' ').title()
                    
                    marker_step = max(1, len(x_vals) // 10)
                    ax.plot(
                        x_vals,
                        means,
                        marker=marker,
                        markevery=marker_step,
                        color=color,
                        linewidth=3.2,
                        markersize=6,
                        label=label,
                        alpha=0.98,
                    )
                    ax.fill_between(x_vals, means - stds, means + stds, color=color, alpha=0.22)
            
            # Apply presentation styling to each subplot
            _style_ax(ax, xlabel='Checkpoint', ylabel=title, title=f'Layer {layer}')
            
            # Add data quality annotation
            n_total = len(checkpoints)
            n_valid_high = data_quality['high_freq']
            n_valid_low = data_quality['low_freq']
            if n_valid_high < n_total or n_valid_low < n_total:
                ax.text(0.02, 0.02, f'Valid H/L: {n_valid_high}/{n_valid_low} of {n_total}', 
                       transform=ax.transAxes, fontsize=10, alpha=0.7,
                       bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
            
            # Ensure consistent numeric x-axis and sparse ticks
            ax.set_xlim(float(checkpoint_numeric[0]), float(checkpoint_numeric[-1]))
            if len(checkpoints) > 10:
                step = max(1, len(checkpoints) // 8)
                tick_idx = list(range(0, len(checkpoints), step))
            else:
                tick_idx = list(range(len(checkpoints)))
            tick_positions = [checkpoint_numeric[i] for i in tick_idx]
            tick_labels = [str(checkpoints[i]) for i in tick_idx]
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=14, weight='semibold')
            ax.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=False, prune=None))

            # Legend with larger font for presentations
            leg = ax.legend(frameon=False, fontsize=14, loc='best', 
                          handlelength=3.8, borderaxespad=1.0)
        
        # Remove unused subplots
        for idx in range(n_layers, len(axes)):
            fig.delaxes(axes[idx])
        
        plt.tight_layout()
        filename = metric_mean.replace('_mean', '') + '_layerwise_comparison.pdf'
        plt.savefig(output_path / filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved {filename}")
    
    