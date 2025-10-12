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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from hdbscan import HDBSCAN  

warnings.filterwarnings('ignore', category=FutureWarning)

def get_participation_ratio(high_activation_space: Dict[int, np.ndarray], low_activation_space: Dict[int, np.ndarray]) -> Dict[str, float]:
    """Compute participation ratio using eigen values from covariance matrix for each layer and frequency category."""
    high_participation_ratio = {}
    low_participation_ratio = {}

    for layer, activations in high_activation_space.items():
        high_participation_ratio[layer] = compute_participation_ratio(activations)

    for layer, activations in low_activation_space.items():
        low_participation_ratio[layer] = compute_participation_ratio(activations)

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
                                max_pairs: int = 5000) -> Dict[str, float]:
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

    actual_max_pairs = min(max_pairs, n_possible_pairs)
    pairs = np.array([(i, j) for i in range(n_samples) for j in range(i + 1, n_samples)], dtype=np.int32)
    
    if len(pairs) > actual_max_pairs:
        pairs = pairs[np.random.choice(len(pairs), actual_max_pairs, replace=False)]
    
    # Compute pairwise distances
    euclidean_distances = []
    hamming_distances = []

    for pair in pairs:
        euclidean_distances.append(np.linalg.norm(activations_std[pair[0]] - activations_std[pair[1]]))
        hamming_distances.append(np.sum(patterns[pair[0]] != patterns[pair[1]]))

    euclidean_distances = np.array(euclidean_distances)
    hamming_distances = np.array(hamming_distances)
    
    # Compute density (hamming / euclidean)
    densities = hamming_distances / (euclidean_distances + 1e-12)
    


    return {
        'density_mean': float(np.mean(densities)),
        'density_std': float(np.std(densities)),
    }

def compute_participation_ratio(activations: np.ndarray) -> float:
    """Compute participation ratio using eigen values from covariance matrix."""
    activations_clean = np.where(np.isfinite(activations), activations, 0.0)
    activations_clipped = np.clip(activations_clean, -1e3, 1e3)
    activations_std = zscore_for_distance(activations_clipped)
    pca = PCA()
    pca.fit(activations_std)
    eigenvals = pca.explained_variance_
    eigenvals = np.where(np.isfinite(eigenvals), eigenvals, 0.0)
    eigenvals = np.maximum(eigenvals, 1e-12)
    participation_ratio = np.sum(eigenvals) ** 2 / np.sum(eigenvals ** 2)
    
    participation_ratio = np.clip(participation_ratio, 1.0, float(len(eigenvals)))

    return float(participation_ratio)
    
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

def load_checkpoint_data(checkpoint_path: str) -> Dict[str, Any]:
    with open(checkpoint_path, 'rb') as f:
        data = pickle.load(f)

    # organize by checkpoints
    checkpoint_groups = {}
    for record in records:
        checkpoint = str(record['checkpoint_step'])
        if checkpoint not in checkpoint_groups:
            checkpoint_groups[checkpoint] = []
        checkpoint_groups[checkpoint].append(record)

    records = checkpoint_groups
    metadata = data['metadata']
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

def compute_metrics_with_confidence_intervals(
    activations: np.ndarray, 
    patterns: np.ndarray,
    n_bootstrap: int = 100,
    confidence_level: float = 0.95,
    max_pairs: int = 5000,
    random_seed: Optional[int] = None
) -> Dict[str, Any]:
    """Compute polytope and participation metrics with confidence intervals using bootstrap.
    
    Args:
        activations: Activation matrix of shape (n_samples, n_features)
        patterns: Binary pattern matrix of shape (n_samples, n_features)
        n_bootstrap: Number of bootstrap iterations
        confidence_level: Confidence level for intervals (e.g., 0.95 for 95% CI)
        max_pairs: Maximum pairs for density computation
        random_seed: Random seed for reproducibility
        
    Returns:
        Dictionary containing:
            - density_mean, density_std, density_ci_lower, density_ci_upper
            - participation_ratio_mean, participation_ratio_std, pr_ci_lower, pr_ci_upper
            - n_samples: number of samples used
            - n_bootstrap_runs: number of bootstrap iterations
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    n_samples = len(activations)
    
    # Collect bootstrap estimates
    density_estimates = []
    pr_estimates = []
    
    sample_size = max(10, int(n_samples * 0.8))  # Use 80% of data for each bootstrap
    
    for i in range(n_bootstrap):            # Resample with replacement
        indices = np.random.choice(n_samples, size=sample_size, replace=True)
        boot_activations = activations[indices]
        boot_patterns = patterns[indices]
        
        # Compute density
        density_result = compute_polytope_density(boot_activations, boot_patterns, max_pairs=max_pairs)
        density_estimates.append(density_result['density_mean'])
        
        # Compute participation ratio
        pr = compute_participation_ratio(boot_activations)
        pr_estimates.append(pr)
            
    
    # Convert to arrays
    density_estimates = np.array(density_estimates)
    pr_estimates = np.array(pr_estimates)
    
    # Filter out non-finite values
    density_estimates = density_estimates[np.isfinite(density_estimates)]
    pr_estimates = pr_estimates[np.isfinite(pr_estimates)]
    
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
    
    # Participation ratio statistics
    if len(pr_estimates) > 0:
        pr_mean = float(np.mean(pr_estimates))
        pr_std = float(np.std(pr_estimates))
        pr_ci_lower = float(np.percentile(pr_estimates, 100 * alpha / 2))
        pr_ci_upper = float(np.percentile(pr_estimates, 100 * (1 - alpha / 2)))
    else:
        pr_mean = pr_std = pr_ci_lower = pr_ci_upper = 0.0
    
    return {
        'density_mean': density_mean,
        'density_std': density_std,
        'density_ci_lower': density_ci_lower,
        'density_ci_upper': density_ci_upper,
        'participation_ratio_mean': pr_mean,
        'participation_ratio_std': pr_std,
        'participation_ratio_ci_lower': pr_ci_lower,
        'participation_ratio_ci_upper': pr_ci_upper,
        'n_samples': n_samples,
        'n_bootstrap_runs': len(density_estimates),
    }
    
def polytope_analysis(
    path_to_records: str, 
    n_bootstrap: int = 100,
    confidence_level: float = 0.95,
    max_pairs: int = 5000,
    random_seed: Optional[int] = None
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
                'random_seed': int
            }
        }
    """
    logger.info(f"Starting polytope analysis with {n_bootstrap} bootstrap iterations")
    logger.info(f"Confidence level: {confidence_level}, max_pairs: {max_pairs}")
    
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
                    random_seed=random_seed
                )
                
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
        'input_metadata': input_metadata
    }
    
    logger.info(f"Analysis complete: {len(results)} checkpoints processed")
    
    return {
        'results': results,
        'metadata': metadata
    }
    