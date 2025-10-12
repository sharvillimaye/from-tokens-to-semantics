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
    n_bootstrap: int = 100,
    confidence_level: float = 0.95,
    max_pairs: int = 5000,
    random_seed: Optional[int] = None,
    n_workers: Optional[int] = None
) -> Dict[str, Any]:
    """Compute polytope and participation metrics with confidence intervals using bootstrap.
    
    Args:
        activations: Activation matrix of shape (n_samples, n_features)
        patterns: Binary pattern matrix of shape (n_samples, n_features)
        n_bootstrap: Number of bootstrap iterations
        confidence_level: Confidence level for intervals (e.g., 0.95 for 95% CI)
        max_pairs: Maximum pairs for density computation
        random_seed: Random seed for reproducibility
        n_workers: Number of parallel workers (default: min(n_bootstrap, cpu_count))
        
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
    sample_size = max(10, int(n_samples * 0.8))  # Use 80% of data for each bootstrap
    
    # Generate all bootstrap indices upfront for reproducibility
    bootstrap_indices = [
        np.random.choice(n_samples, size=sample_size, replace=True)
        for _ in range(n_bootstrap)
    ]
    
    # Prepare arguments for parallel processing
    args_list = [
        (activations, patterns, indices, max_pairs)
        for indices in bootstrap_indices
    ]
    
    # Determine number of workers
    if n_workers is None:
        n_workers = min(n_bootstrap, mp.cpu_count())
    
    # Run bootstrap iterations in parallel using ThreadPoolExecutor
    density_estimates = []
    pr_estimates = []
    
    logger.debug(f"Running {n_bootstrap} bootstrap iterations with {n_workers} workers")
    
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(_bootstrap_iteration, args) for args in args_list]
        
        for future in as_completed(futures):
            density_est, pr_est = future.result()
            if density_est is not None and pr_est is not None:
                density_estimates.append(density_est)
                pr_estimates.append(pr_est)
    
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
    random_seed: Optional[int] = None,
    n_workers: Optional[int] = None,
    parallel_checkpoints: bool = False
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
    logger.info(f"Starting polytope analysis with {n_bootstrap} bootstrap iterations")
    logger.info(f"Confidence level: {confidence_level}, max_pairs: {max_pairs}")
    
    if n_workers is None:
        n_workers = min(n_bootstrap, mp.cpu_count())
    logger.info(f"Using {n_workers} parallel workers for bootstrap iterations")
    
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
                    random_seed=random_seed,
                    n_workers=n_workers
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
        'n_workers': n_workers,
        'parallel_checkpoints': parallel_checkpoints,
        'input_metadata': input_metadata
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
    
    # Parse results into structured format
    checkpoints = sorted(analysis_results.keys())
    layers = sorted(next(iter(analysis_results.values())).keys())
    
    logger.info(f"Creating visualizations for {len(checkpoints)} checkpoints and {len(layers)} layers")
    
    # Create visualizations
    _plot_difference_heatmaps(analysis_results, checkpoints, layers, output_path, palette)
    _plot_layerwise_metrics(analysis_results, checkpoints, layers, output_path, palette)
    
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
        
        # Create heatmap with 16:9 aspect ratio for presentations
        fig, ax = plt.subplots(figsize=(11.5, 6.46875))
        
        # Use diverging colormap centered at 0
        vmax = np.abs(diff_matrix).max()
        vmin = -vmax
        
        im = ax.imshow(diff_matrix, aspect='auto', cmap='coolwarm', 
                      vmin=vmin, vmax=vmax, origin='lower')
        
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


def _plot_layerwise_metrics(analysis_results: Dict, checkpoints: List, layers: List, 
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
            
            for freq_type in ['high_freq', 'low_freq']:
                means = []
                stds = []
                valid_checkpoints = []
                
                for checkpoint in checkpoints:
                    metrics_data = analysis_results[checkpoint][layer][freq_type]
                    means.append(metrics_data.get(metric_mean, 0))
                    stds.append(metrics_data.get(metric_std, 0))
                    valid_checkpoints.append(checkpoint)
                
                means = np.array(means)
                stds = np.array(stds)
                
                # Plot with presentation-quality thick lines
                color = colors[freq_type]
                marker = markers[freq_type]
                label = freq_type.replace('_', ' ').title()
                
                ax.plot(valid_checkpoints, means, marker=marker, color=color, 
                       linewidth=4.8, markersize=10, label=label, alpha=0.98)
                ax.fill_between(valid_checkpoints, means - stds, means + stds, 
                               color=color, alpha=0.25)
            
            # Apply presentation styling to each subplot
            _style_ax(ax, xlabel='Checkpoint', ylabel=title, title=f'Layer {layer}')
            
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
    
    