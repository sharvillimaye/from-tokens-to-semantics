#!/usr/bin/env python3
"""
Enhanced Polytope Metrics for Multi-Dimensional Superposition Analysis
Implements three-pronged visualization strategy for NeurIPS submission:
1. Temporal Evolution Analysis - track layers across training checkpoints
2. Layer-wise Progression Analysis - heatmaps across network depth 
3. Correlation Analysis - n-gram frequency vs polytope structure
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional, Union
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as patches
from collections import defaultdict
import warnings
from pathlib import Path

# Optional seaborn import
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    sns = None

warnings.filterwarnings('ignore')


def find_extreme_points(points: np.ndarray, n_components: int = 20) -> np.ndarray:
    """
    Find extreme points using PCA for high-dimensional initialization
    
    Args:
        points: Input points (n_samples, n_features)
        n_components: Number of PCA components to use for finding extremes
        
    Returns:
        Indices of extreme points
    """
    n_points, n_dims = points.shape
    
    if n_points < 3:
        return np.arange(n_points)
    
    # Always use PCA for high-dimensional data
    n_components = min(n_components, n_points - 1, n_dims)
    
    scaler = StandardScaler()
    points_scaled = scaler.fit_transform(points)
    
    pca = PCA(n_components=n_components)
    points_pca = pca.fit_transform(points_scaled)
    
    extreme_indices = set()
    
    # Find min/max in each PCA dimension
    for dim in range(points_pca.shape[1]):
        min_idx = np.argmin(points_pca[:, dim])
        max_idx = np.argmax(points_pca[:, dim])
        extreme_indices.add(min_idx)
        extreme_indices.add(max_idx)
    
    # Add points with highest norms in PCA space
    pca_norms = np.linalg.norm(points_pca, axis=1)
    top_norm_indices = np.argsort(pca_norms)[-min(10, n_points):]
    extreme_indices.update(top_norm_indices)
    
    # Add some random points for diversity in high dimensions
    if len(extreme_indices) < min(50, n_points // 20):
        remaining_indices = set(range(n_points)) - extreme_indices
        if remaining_indices:
            n_random = min(10, len(remaining_indices))
            random_indices = np.random.choice(list(remaining_indices), size=n_random, replace=False)
            extreme_indices.update(random_indices)
    
    return np.array(list(extreme_indices))


def distance_to_approximate_hull(point: np.ndarray, hull_points: np.ndarray) -> float:
    """
    Calculate distance from point to approximate convex hull using optimization
    Designed for high-dimensional settings
    
    Args:
        point: Query point
        hull_points: Points defining the approximate hull
        
    Returns:
        Distance to approximate hull
    """
    n_hull_points = len(hull_points)
    
    # For single point hull
    if n_hull_points == 1:
        return np.linalg.norm(point - hull_points[0])
    
    try:
        # Minimize ||point - sum(alpha_i * hull_points_i)||^2
        # subject to sum(alpha_i) = 1, alpha_i >= 0
        def objective(alpha):
            weighted_sum = np.sum(alpha[:, np.newaxis] * hull_points, axis=0)
            return np.sum((point - weighted_sum) ** 2)
        
        # Constraints and bounds
        constraints = [{'type': 'eq', 'fun': lambda alpha: np.sum(alpha) - 1}]
        bounds = [(0, None) for _ in range(n_hull_points)]
        
        # Better initialization: start with closest point
        distances_to_vertices = np.linalg.norm(hull_points - point, axis=1)
        closest_idx = np.argmin(distances_to_vertices)
        alpha0 = np.zeros(n_hull_points)
        alpha0[closest_idx] = 1.0
        
        # Solve optimization
        result = minimize(
            objective, alpha0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-9}
        )
        
        if result.success:
            return np.sqrt(result.fun)
        else:
            # If optimization fails, return distance to closest vertex
            return np.min(distances_to_vertices)
            
    except Exception:
        # Final fallback: distance to closest hull point
        distances = np.linalg.norm(hull_points - point, axis=1)
        return np.min(distances)


def greedy_hull_approximation(points: np.ndarray, epsilon: float = 0.05, max_iter: int = 1000, 
                             sample_size: int = 2000, verbose: bool = True) -> np.ndarray:
    """
    Greedy approximation of convex hull for high-dimensional data
    
    Args:
        points: Input points (n_samples, n_features)
        epsilon: Convergence tolerance
        max_iter: Maximum iterations
        sample_size: Maximum points to check per iteration for efficiency
        verbose: Print progress
        
    Returns:
        Indices of approximate hull vertices
    """
    n_points, n_dims = points.shape
    
    if verbose:
        print(f"Greedy hull approximation: {n_points} points in {n_dims}D, epsilon={epsilon}")
    
    if n_points <= 3:
        return np.arange(n_points)
    
    # Initialize with extreme points from PCA
    hull_indices = set(find_extreme_points_pca(points, n_components=min(30, n_dims)))
    remaining_indices = set(range(n_points)) - hull_indices
    
    if verbose:
        print(f"Initialized with {len(hull_indices)} extreme points")
    
    # Greedy expansion
    for iteration in range(max_iter):
        if not remaining_indices:
            if verbose:
                print("No more points to add")
            break
        
        hull_points = points[list(hull_indices)]
        
        # Sample remaining points for efficiency in high dimensions
        check_indices = list(remaining_indices)
        if len(check_indices) > sample_size:
            check_indices = np.random.choice(check_indices, size=sample_size, replace=False)
        
        # Find point farthest from current hull
        max_distance = 0.0
        farthest_idx = None
        
        for idx in check_indices:
            distance = distance_to_approximate_hull(points[idx], hull_points)
            if distance > max_distance:
                max_distance = distance
                farthest_idx = idx
        
        # Check convergence
        if max_distance <= epsilon:
            if verbose:
                print(f"Converged at iteration {iteration}: max_distance = {max_distance:.6f}")
            break
        
        # Add farthest point to hull
        if farthest_idx is not None:
            hull_indices.add(farthest_idx)
            remaining_indices.discard(farthest_idx)
        
        # Progress reporting
        if verbose and (iteration + 1) % 100 == 0:
            compression_ratio = len(hull_indices) / n_points
            print(f"Iteration {iteration + 1}: {len(hull_indices)} vertices "
                  f"(compression: {compression_ratio:.4f}), max_dist: {max_distance:.6f}")
    
    hull_vertices = np.array(list(hull_indices))
    final_compression = len(hull_vertices) / n_points
    
    if verbose:
        print(f"Final approximation: {len(hull_vertices)} vertices "
              f"(compression ratio: {final_compression:.4f})")
    
    return hull_vertices


def compute_approximate_volume(hull_points: np.ndarray, n_samples: int = 10000) -> float:
    """
    Approximate volume using Monte Carlo sampling within bounding box
    
    Args:
        hull_points: Points defining the approximate hull
        n_samples: Number of Monte Carlo samples
        
    Returns:
        Approximate volume
    """
    if len(hull_points) < 2:
        return 0.0
    
    # Find bounding box
    min_coords = np.min(hull_points, axis=0)
    max_coords = np.max(hull_points, axis=0)
    
    # Volume of bounding box
    box_volume = np.prod(max_coords - min_coords)
    
    if box_volume == 0:
        return 0.0
    
    # Monte Carlo sampling
    n_inside = 0
    
    for _ in range(n_samples):
        # Random point in bounding box
        random_point = np.random.uniform(min_coords, max_coords)
        
        # Check if inside hull (distance ≈ 0)
        distance = distance_to_approximate_hull(random_point, hull_points)
        if distance <= 1e-6:  # Very small tolerance for "inside"
            n_inside += 1
    
    # Estimate volume
    volume_ratio = n_inside / n_samples
    return box_volume * volume_ratio


def compute_approximate_surface_area(hull_points: np.ndarray, n_samples: int = 5000) -> float:
    """
    Approximate surface area using sampling near the hull boundary
    
    Args:
        hull_points: Points defining the approximate hull
        n_samples: Number of samples for estimation
        
    Returns:
        Approximate surface area
    """
    if len(hull_points) < 2:
        return 0.0
    
    # Sample points around hull vertices
    surface_points = []
    
    for hull_point in hull_points:
        # Add small perturbations around each hull vertex
        for _ in range(max(1, n_samples // len(hull_points))):
            noise = np.random.normal(0, 0.01, size=hull_point.shape)
            perturbed_point = hull_point + noise
            surface_points.append(perturbed_point)
    
    surface_points = np.array(surface_points)
    
    # Estimate surface area using pairwise distances
    if len(surface_points) > 1:
        distances = pdist(surface_points)
        # Surface area approximation based on point density
        mean_distance = np.mean(distances)
        return len(hull_points) * (mean_distance ** (hull_points.shape[1] - 1))
    else:
        return 0.0


def compute_polytope_metrics(points: np.ndarray, use_approximation: bool = True, epsilon: float = 0.1) -> Dict[str, float]:
    """
    Compute polytope metrics for a set of points
    
    Args:
        points: Input points (n_samples, n_features)
        use_approximation: Use hull approximation for large sets
        epsilon: Approximation tolerance
        
    Returns:
        Dictionary of metrics
    """
    n_points, n_dims = points.shape
    
    if n_points < 2:
        return {
            'volume': 0.0, 'surface_area': 0.0, 'n_vertices': n_points,
            'n_facets': 0, 'hull_valid': False, 'approximation_used': False,
            'mean_distance': 0.0, 'std_distance': 0.0, 'centroid_variance': 0.0,
            'effective_dimension': 0.0
        }
    
    # Use approximation for large datasets
    approximation_used = False
    hull_points = points

    hull_indices = greedy_hull_approximation(points, epsilon=epsilon)
    hull_points = points[hull_indices]
    approximation_used = True

    # Compute volume and surface area
    volume = compute_approximate_volume(hull_points)
    surface_area = compute_approximate_surface_area(hull_points)
    n_vertices = len(hull_points)
    n_facets = 0
    hull_valid = True
    
    # Additional metrics on original points
    sample_points = points
    
    distances = pdist(sample_points)
    mean_distance = np.mean(distances)
    std_distance = np.std(distances)
    
    # Centroid variance
    centroid = np.mean(points, axis=0)
    centroid_distances = np.linalg.norm(points - centroid, axis=1)
    centroid_variance = np.var(centroid_distances)
    
    # Effective dimension (participation ratio)
    norms = np.linalg.norm(points, axis=1)
    if np.sum(norms) > 0:
        normalized_norms = norms / np.sum(norms)
        effective_dim = 1.0 / np.sum(normalized_norms**2)
    else:
        effective_dim = 0.0
    
    return {
        'volume': volume,
        'surface_area': surface_area,
        'n_vertices': n_vertices,
        'n_facets': n_facets,
        'hull_valid': hull_valid,
        'approximation_used': approximation_used,
        'mean_distance': mean_distance,
        'std_distance': std_distance,
        'centroid_variance': centroid_variance,
        'effective_dimension': effective_dim
    }


def reduce_dimensions(points: np.ndarray, n_components: int = 20) -> Tuple[np.ndarray, dict]:
    """
    Reduce dimensionality using PCA
    
    Args:
        points: Input points (n_samples, n_features)
        n_components: Number of components
        
    Returns:
        (reduced_points, transformer_info)
    """
    if points.shape[0] < 2:
        return points, {}
    
    n_components = min(n_components, points.shape[0] - 1, points.shape[1])
    
    scaler = StandardScaler()
    points_scaled = scaler.fit_transform(points)
    
    pca = PCA(n_components=n_components)
    points_reduced = pca.fit_transform(points_scaled)
    
    variance_explained = np.sum(pca.explained_variance_ratio_)
    
    return points_reduced, {
        'pca': pca, 'scaler': scaler, 
        'variance_explained': variance_explained,
        'explained_variance_ratio': pca.explained_variance_ratio_
    }


def filter_by_activation_threshold(records: List[Dict], threshold_percentile: float = 75) -> List[Dict]:
    """
    Filter records by activation norm threshold
    
    Args:
        records: List of activation records
        threshold_percentile: Percentile threshold (0-100)
        
    Returns:
        Filtered records
    """
    if not records:
        return []
    
    norms = [r['activation_norm'] for r in records]
    threshold = np.percentile(norms, threshold_percentile)
    
    filtered = [r for r in records if r['activation_norm'] >= threshold]
    
    return filtered


def group_by_frequency(records: List[Dict], n_bins: int = 5) -> Dict[str, List[Dict]]:
    """
    Group records by n-gram frequency bins
    
    Args:
        records: List of activation records
        n_bins: Number of frequency bins
        
    Returns:
        Dictionary mapping bin_name -> records
    """
    frequencies = [r.get('ngram_frequency', 0) for r in records]
    
    if not any(frequencies):
        return {'all': records}
    
    freq_min, freq_max = min(frequencies), max(frequencies)
    bin_edges = np.linspace(freq_min, freq_max + 0.001, n_bins + 1)
    
    bins = defaultdict(list)
    
    for record in records:
        freq = record.get('ngram_frequency', 0)
        bin_idx = np.digitize(freq, bin_edges) - 1
        bin_idx = max(0, min(bin_idx, n_bins - 1))
        
        bin_name = f"freq_{bin_edges[bin_idx]:.2f}_{bin_edges[bin_idx+1]:.2f}"
        bins[bin_name].append(record)
    
    return dict(bins)


def analyze_layer_polytopes(records: List[Dict], target_layers: List[int] = None, 
                          n_components: int = 20, use_approximation: bool = True) -> Dict[int, Dict]:
    """
    Analyze polytope structure for each layer
    
    Args:
        records: List of activation records
        target_layers: Layers to analyze (None for all)
        n_components: PCA components
        use_approximation: Use hull approximation
        
    Returns:
        Dictionary mapping layer -> analysis results
    """
    # Group by layer
    layer_records = defaultdict(list)
    for record in records:
        layer_records[record['layer']].append(record)
    
    # Filter layers
    if target_layers:
        layer_records = {l: recs for l, recs in layer_records.items() if l in target_layers}
    
    results = {}
    
    for layer, recs in layer_records.items():
        # Filter by activation threshold
        filtered_recs = filter_by_activation_threshold(recs, threshold_percentile=75)
        
        if len(filtered_recs) < 3:
            continue
        
        # Extract activation vectors
        activations = np.stack([r['activation_vector'] for r in filtered_recs])
        
        # Reduce dimensions
        reduced_activations, pca_info = reduce_dimensions(activations, n_components)
        
        # Compute polytope metrics
        metrics = compute_polytope_metrics(reduced_activations, use_approximation=use_approximation)
        
        results[layer] = {
            'n_records': len(filtered_recs),
            'original_dim': activations.shape[1],
            'reduced_dim': reduced_activations.shape[1],
            'pca_info': pca_info,
            'metrics': metrics,
            'activations': reduced_activations
        }
        
    return results


def analyze_frequency_polytope_relationship(records: List[Dict], target_layers: List[int] = None) -> pd.DataFrame:
    """
    Analyze relationship between n-gram frequency and polytope metrics
    
    Args:
        records: List of activation records
        target_layers: Layers to analyze
        
    Returns:
        DataFrame with analysis results
    """
    results = []
    
    # Group by frequency
    freq_groups = group_by_frequency(records, n_bins=5)
    
    for freq_bin, freq_records in freq_groups.items():
        # Analyze polytopes for this frequency bin
        layer_results = analyze_layer_polytopes(
            freq_records, 
            target_layers=target_layers,
            use_approximation=True
        )
        
        # Extract results
        for layer, analysis in layer_results.items():
            metrics = analysis['metrics']
            
            result = {
                'frequency_bin': freq_bin,
                'layer': layer,
                'n_samples': analysis['n_records'],
                'mean_frequency': np.mean([r.get('ngram_frequency', 0) for r in freq_records]),
                **metrics  # Unpack all metrics
            }
            results.append(result)
    
    return pd.DataFrame(results)


def plot_temporal_evolution(evolution_results: Dict[str, Any], 
                          save_path: str = None, 
                          figsize: Tuple[int, int] = (15, 5)) -> None:
    """
    FIGURE 1: Temporal Evolution Plot for NeurIPS submission
    
    Args:
        evolution_results: Results from temporal_evolution_analysis
        save_path: Optional save path
        figsize: Figure size
    """
    if 'error' in evolution_results:
        print(f"Error in evolution results: {evolution_results['error']}")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    layer = evolution_results['layer']
    
    # Primary metrics to plot
    metrics = ['volume', 'n_vertices', 'effective_dimension']
    metric_labels = ['Polytope Volume', 'Vertex Count', 'Effective Dimension']
    
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[i]
        
        # Plot frequency evolution lines
        for freq_bin, freq_data in evolution_results['frequency_evolution'].items():
            if not freq_data:
                continue
                
            df = pd.DataFrame(freq_data)
            
            # Clean frequency bin name for legend
            clean_bin_name = freq_bin.replace('freq_', '').replace('_', '-')
            
            ax.plot(df['checkpoint_num'], df[metric], 
                   marker='o', linewidth=2, markersize=6,
                   label=f'Freq {clean_bin_name}', alpha=0.8)
        
        # Plot overall evolution as thick black line
        if evolution_results['overall_evolution']:
            overall_df = pd.DataFrame(evolution_results['overall_evolution'])
            ax.plot(overall_df['checkpoint_num'], overall_df[metric],
                   color='black', linewidth=3, marker='s', markersize=8,
                   label='Overall', alpha=0.9)
        
        ax.set_xlabel('Training Checkpoint')
        ax.set_ylabel(label)
        ax.set_title(f'{label} Evolution')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add trend annotation
        if evolution_results['statistics']:
            trend_key = f"{metric.split('_')[0]}_trend"
            if trend_key in evolution_results['statistics']:
                trend = evolution_results['statistics'][trend_key]
                ax.text(0.05, 0.95, f'Trend: {trend:.3f}', 
                       transform=ax.transAxes, fontsize=10,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.suptitle(f'Temporal Evolution Analysis - Layer {layer}', fontsize=16, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_layer_progression_heatmap(progression_results: Dict[str, Any],
                                 save_path: str = None,
                                 figsize: Tuple[int, int] = (12, 8)) -> None:
    """
    FIGURE 2: Layer Progression Heatmap for NeurIPS submission
    
    Args:
        progression_results: Results from layer_progression_analysis
        save_path: Optional save path
        figsize: Figure size
    """
    if 'matrices' not in progression_results:
        print("Error: No matrices found in progression results")
        return
    
    matrices = progression_results['matrices']
    n_metrics = len(matrices)
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    
    # Custom colormap for better visualization
    colors = ['#2166ac', '#f7f7f7', '#762a83']
    n_bins = 100
    custom_cmap = LinearSegmentedColormap.from_list('custom', colors, N=n_bins)
    
    for i, (metric_name, matrix) in enumerate(matrices.items()):
        if i >= 4:  # Limit to 4 subplots
            break
            
        ax = axes[i]
        
        # Create heatmap
        im = ax.imshow(matrix.values, cmap=custom_cmap, aspect='auto', interpolation='nearest')
        
        # Set ticks and labels
        ax.set_xticks(range(len(matrix.columns)))
        ax.set_xticklabels(matrix.columns, rotation=45, ha='right')
        ax.set_yticks(range(len(matrix.index)))
        ax.set_yticklabels(matrix.index)
        
        ax.set_xlabel('Training Checkpoint')
        ax.set_ylabel('Layer')
        ax.set_title(f'{metric_name.replace("_", " ").title()}')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label(f'{metric_name.replace("_", " ").title()}', rotation=270, labelpad=20)
        
        # Add value annotations for small matrices
        if matrix.shape[0] <= 8 and matrix.shape[1] <= 5:
            for row in range(matrix.shape[0]):
                for col in range(matrix.shape[1]):
                    value = matrix.iloc[row, col]
                    ax.text(col, row, f'{value:.2f}', ha='center', va='center',
                           color='white' if value > matrix.values.max() * 0.5 else 'black',
                           fontsize=8)
    
    plt.suptitle('Layer-wise Progression Analysis', fontsize=16, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Layer progression heatmap saved to {save_path}")
    
    plt.show()


def plot_correlation_analysis(correlation_results: Dict[str, Any],
                            save_path: str = None,
                            figsize: Tuple[int, int] = (15, 10)) -> None:
    """
    FIGURE 3: Enhanced Correlation Analysis for NeurIPS submission
    
    Args:
        correlation_results: Results from enhanced_correlation_analysis
        save_path: Optional save path
        figsize: Figure size
    """
    if 'error' in correlation_results:
        print(f"Error in correlation results: {correlation_results['error']}")
        return
    
    df = correlation_results['correlation_data']
    stages = correlation_results['stages']
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    
    # Metrics to analyze
    metrics = ['activation_norm', 'sparsity', 'n_active_neurons', 'effective_dimension']
    metric_labels = ['Activation Norm', 'Sparsity', 'Active Neurons', 'Effective Dimension']
    
    # Color palette for stages
    stage_colors = plt.cm.viridis(np.linspace(0, 1, len(stages)))
    
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        if i >= 4:
            break
            
        ax = axes[i]
        
        # Plot scatter points for each stage
        for j, stage in enumerate(stages):
            stage_df = df[df['checkpoint_stage'] == stage]
            
            if len(stage_df) == 0:
                continue
            
            # Scatter plot
            ax.scatter(stage_df['ngram_frequency'], stage_df[metric],
                      c=[stage_colors[j]], alpha=0.6, s=30, label=f'Step {stage}')
            
            # Add regression line with confidence interval
            if len(stage_df) > 3:
                x = stage_df['ngram_frequency'].values
                y = stage_df[metric].values
                
                # Compute regression
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                
                if p_value < 0.05:  # Only show significant regressions
                    x_line = np.linspace(x.min(), x.max(), 100)
                    y_line = slope * x_line + intercept
                    
                    ax.plot(x_line, y_line, color=stage_colors[j], 
                           linewidth=2, alpha=0.8)
                    
                    # Add confidence interval (approximate)
                    y_err = std_err * x_line
                    ax.fill_between(x_line, y_line - y_err, y_line + y_err,
                                   color=stage_colors[j], alpha=0.2)
                    
                    # Add correlation annotation
                    ax.text(0.05, 0.95 - j*0.1, f'Step {stage}: r={r_value:.3f}',
                           transform=ax.transAxes, fontsize=9,
                           color=stage_colors[j], weight='bold')
        
        ax.set_xlabel('N-gram Frequency')
        ax.set_ylabel(label)
        ax.set_title(f'{label} vs N-gram Frequency')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('N-gram Frequency vs Polytope Structure Correlation', fontsize=16, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Correlation analysis plot saved to {save_path}")
    
    plt.show()


def create_neurips_figures(records: List[Dict],
                         target_layer: int = 6,
                         checkpoint_order: List[str] = None,
                         output_dir: str = "reports/figures") -> Dict[str, str]:
    """
    Generate all three NeurIPS figures with consistent styling
    
    Args:
        records: List of activation records
        target_layer: Layer for temporal analysis
        checkpoint_order: Ordered checkpoints
        output_dir: Output directory for figures
        
    Returns:
        Dictionary mapping figure name to save path
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    saved_figures = {}
    
    print("=== Generating NeurIPS Figures ===")
    
    # Figure 1: Temporal Evolution
    print("\nGenerating Figure 1: Temporal Evolution...")
    temporal_results = temporal_evolution_analysis(records, target_layer, checkpoint_order)
    if 'error' not in temporal_results:
        fig1_path = output_path / "figure1_temporal_evolution.png"
        plot_temporal_evolution(temporal_results, save_path=str(fig1_path))
        saved_figures['figure1'] = str(fig1_path)
    
    # Figure 2: Layer Progression
    print("\nGenerating Figure 2: Layer Progression...")
    progression_results = layer_progression_analysis(records)
    if progression_results['progression_data'] is not None and not progression_results['progression_data'].empty:
        fig2_path = output_path / "figure2_layer_progression.png"
        plot_layer_progression_heatmap(progression_results, save_path=str(fig2_path))
        saved_figures['figure2'] = str(fig2_path)
    
    # Figure 3: Correlation Analysis
    print("\nGenerating Figure 3: Correlation Analysis...")
    correlation_results = enhanced_correlation_analysis(records)
    if 'error' not in correlation_results:
        fig3_path = output_path / "figure3_correlation_analysis.png"
        plot_correlation_analysis(correlation_results, save_path=str(fig3_path))
        saved_figures['figure3'] = str(fig3_path)
    
    print(f"\nGenerated {len(saved_figures)} figures in {output_dir}")
    return saved_figures


def plot_polytope_analysis(df: pd.DataFrame, save_path: str = None):
    """
    Legacy polytope analysis plotting (maintained for compatibility)
    
    Args:
        df: Results dataframe from analyze_frequency_polytope_relationship
        save_path: Optional save path
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    metrics = ['volume', 'effective_dimension', 'mean_distance', 'centroid_variance']
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        for layer in sorted(df['layer'].unique()):
            layer_data = df[df['layer'] == layer]
            
            ax.scatter(layer_data['mean_frequency'], layer_data[metric], 
                      label=f'Layer {layer}', alpha=0.7, s=60)
        
        ax.set_xlabel('Mean N-gram Frequency')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'{metric.replace("_", " ").title()} vs Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.suptitle('Polytope Metrics vs N-gram Frequency', fontsize=14, y=1.02)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def monte_carlo_volume_estimation(points: np.ndarray, n_samples: int = 100000) -> float:
    """
    Monte Carlo volume estimation for degenerate polytopes
    
    Args:
        points: Input points defining the polytope
        n_samples: Number of Monte Carlo samples
        
    Returns:
        Estimated volume
    """
    if len(points) < 2:
        return 0.0
    
    # Get bounding box
    min_bounds = np.min(points, axis=0)
    max_bounds = np.max(points, axis=0)
    
    # Generate random samples in bounding box
    random_samples = np.random.uniform(
        min_bounds, max_bounds, size=(n_samples, points.shape[1])
    )
    
    # Check if samples are inside convex hull
    try:
        hull = ConvexHull(points)
        inside_count = 0
        
        for sample in random_samples:
            # Simple point-in-hull test using hull equations
            equations = hull.equations
            if np.all(equations[:, :-1] @ sample + equations[:, -1] <= 1e-12):
                inside_count += 1
        
        # Estimate volume
        bounding_volume = np.prod(max_bounds - min_bounds)
        estimated_volume = bounding_volume * (inside_count / n_samples)
        
        return max(estimated_volume, 0.0)
        
    except:
        return 0.0


def temporal_evolution_analysis(records: List[Dict], 
                              target_layer: int,
                              checkpoint_order: List[str] = None,
                              frequency_bins: int = 3) -> Dict[str, Any]:
    """
    PRIMARY ANALYSIS: Track single layer across training checkpoints
    
    Args:
        records: List of activation records
        target_layer: Layer to analyze across time
        checkpoint_order: Ordered list of checkpoint steps
        frequency_bins: Number of n-gram frequency bins
        
    Returns:
        Temporal evolution results for visualization
    """
    print(f"=== Temporal Evolution Analysis: Layer {target_layer} ===")
    
    # Filter records for target layer
    layer_records = [r for r in records if r['layer'] == target_layer]
    
    if not layer_records:
        return {'error': f'No records found for layer {target_layer}'}
    
    # Group by checkpoint
    checkpoint_groups = defaultdict(list)
    for record in layer_records:
        checkpoint_groups[record['checkpoint_step']].append(record)
    
    # Order checkpoints if provided
    if checkpoint_order is None:
        checkpoint_order = sorted(checkpoint_groups.keys(), key=lambda x: int(x))
    
    # Group by frequency for multiple lines
    freq_groups = group_by_frequency(layer_records, n_bins=frequency_bins)
    
    results = {
        'layer': target_layer,
        'checkpoints': checkpoint_order,
        'frequency_evolution': {},
        'overall_evolution': [],
        'statistics': {}
    }
    
    # Analyze each frequency group across checkpoints
    for freq_bin, freq_records in freq_groups.items():
        print(f"\nAnalyzing frequency bin: {freq_bin}")
        
        freq_checkpoint_groups = defaultdict(list)
        for record in freq_records:
            freq_checkpoint_groups[record['checkpoint_step']].append(record)
        
        evolution_data = []
        
        for checkpoint in checkpoint_order:
            if checkpoint not in freq_checkpoint_groups:
                continue
                
            checkpoint_records = freq_checkpoint_groups[checkpoint]
            
            # Extract activations and compute metrics
            activations = np.stack([r['activation_vector'] for r in checkpoint_records])
            
            # Reduce dimensions
            reduced_activations, pca_info = reduce_dimensions(activations, n_components=20)
            
            # Compute polytope metrics with fallback
            metrics = compute_polytope_metrics(reduced_activations, use_approximation=True)
            
            # Add Monte Carlo fallback if needed
            if metrics['volume'] == 0.0 and metrics['hull_valid'] is False:
                mc_volume = monte_carlo_volume_estimation(reduced_activations)
                metrics['mc_volume'] = mc_volume
            
            evolution_point = {
                'checkpoint': checkpoint,
                'checkpoint_num': int(checkpoint),
                'n_samples': len(checkpoint_records),
                'frequency_bin': freq_bin,
                'mean_frequency': np.mean([r.get('ngram_frequency', 0) for r in checkpoint_records]),
                **metrics,
                'variance_explained': pca_info.get('variance_explained', 0.0)
            }
            evolution_data.append(evolution_point)
        
        results['frequency_evolution'][freq_bin] = evolution_data
    
    # Overall evolution (all frequency bins combined)
    for checkpoint in checkpoint_order:
        if checkpoint not in checkpoint_groups:
            continue
            
        checkpoint_records = checkpoint_groups[checkpoint]
        activations = np.stack([r['activation_vector'] for r in checkpoint_records])
        reduced_activations, pca_info = reduce_dimensions(activations, n_components=20)
        metrics = compute_polytope_metrics(reduced_activations, use_approximation=True)
        
        # Add Monte Carlo fallback if needed
        if metrics['volume'] == 0.0 and metrics['hull_valid'] is False:
            mc_volume = monte_carlo_volume_estimation(reduced_activations)
            metrics['mc_volume'] = mc_volume
        
        overall_point = {
            'checkpoint': checkpoint,
            'checkpoint_num': int(checkpoint),
            'n_samples': len(checkpoint_records),
            **metrics,
            'variance_explained': pca_info.get('variance_explained', 0.0)
        }
        results['overall_evolution'].append(overall_point)
    
    # Compute evolution statistics
    if results['overall_evolution']:
        df = pd.DataFrame(results['overall_evolution'])
        results['statistics'] = {
            'volume_trend': stats.pearsonr(df['checkpoint_num'], df['volume'])[0] if len(df) > 1 else 0,
            'vertex_trend': stats.pearsonr(df['checkpoint_num'], df['n_vertices'])[0] if len(df) > 1 else 0,
            'complexity_trend': stats.pearsonr(df['checkpoint_num'], df['effective_dimension'])[0] if len(df) > 1 else 0,
            'final_volume': df['volume'].iloc[-1] if len(df) > 0 else 0,
            'volume_change': df['volume'].iloc[-1] - df['volume'].iloc[0] if len(df) > 1 else 0
        }
    
    print(f"Completed temporal analysis for layer {target_layer}")
    print(f"Analyzed {len(checkpoint_order)} checkpoints")
    print(f"Volume trend correlation: {results['statistics'].get('volume_trend', 0):.3f}")
    
    return results


def layer_progression_analysis(records: List[Dict],
                             target_checkpoints: List[str] = None,
                             target_layers: List[int] = None) -> Dict[str, Any]:
    """
    SECONDARY ANALYSIS: Compare metrics across network layers at key checkpoints
    
    Args:
        records: List of activation records
        target_checkpoints: Key checkpoints to analyze (early, middle, late)
        target_layers: Layers to analyze
        
    Returns:
        Layer progression results for heatmap visualization
    """
    print("=== Layer-wise Progression Analysis ===")
    
    # Get available layers and checkpoints
    available_layers = sorted(set(r['layer'] for r in records))
    available_checkpoints = sorted(set(r['checkpoint_step'] for r in records), key=lambda x: int(x))
    
    if target_layers is None:
        target_layers = available_layers
    if target_checkpoints is None:
        # Select key checkpoints: early, middle, late
        n_checkpoints = len(available_checkpoints)
        indices = [0, n_checkpoints//2, n_checkpoints-1]
        target_checkpoints = [available_checkpoints[i] for i in indices if i < len(available_checkpoints)]
    
    print(f"Analyzing {len(target_layers)} layers across {len(target_checkpoints)} checkpoints")
    
    # Create progression matrix
    progression_data = []
    
    for checkpoint in target_checkpoints:
        for layer in target_layers:
            # Filter records
            layer_checkpoint_records = [
                r for r in records 
                if r['layer'] == layer and r['checkpoint_step'] == checkpoint
            ]
            
            if len(layer_checkpoint_records) < 3:
                continue
            
            # Extract activations and analyze
            activations = np.stack([r['activation_vector'] for r in layer_checkpoint_records])
            reduced_activations, pca_info = reduce_dimensions(activations, n_components=20)
            metrics = compute_polytope_metrics(reduced_activations, use_approximation=True)
            
            progression_point = {
                'checkpoint': checkpoint,
                'checkpoint_num': int(checkpoint),
                'layer': layer,
                'n_samples': len(layer_checkpoint_records),
                **metrics,
                'variance_explained': pca_info.get('variance_explained', 0.0)
            }
            progression_data.append(progression_point)
    
    df = pd.DataFrame(progression_data)
    
    # Create matrices for heatmap visualization
    metrics_matrices = {}
    key_metrics = ['volume', 'n_vertices', 'effective_dimension', 'mean_distance']
    
    for metric in key_metrics:
        matrix = df.pivot(index='layer', columns='checkpoint', values=metric)
        metrics_matrices[metric] = matrix.fillna(0)
    
    return {
        'progression_data': df,
        'matrices': metrics_matrices,
        'checkpoints': target_checkpoints,
        'layers': target_layers,
        'summary': {
            'n_data_points': len(progression_data),
            'checkpoint_range': f"{target_checkpoints[0]}-{target_checkpoints[-1]}",
            'layer_range': f"{min(target_layers)}-{max(target_layers)}"
        }
    }


def enhanced_correlation_analysis(records: List[Dict],
                                target_layers: List[int] = None,
                                checkpoint_stages: List[str] = None) -> Dict[str, Any]:
    """
    SUPPORTING ANALYSIS: N-gram frequency vs polytope structure with statistics
    
    Args:
        records: List of activation records with frequency information
        target_layers: Layers to analyze
        checkpoint_stages: Training stages to compare (early, middle, late)
        
    Returns:
        Correlation analysis results with regression and confidence intervals
    """
    print("=== Enhanced Correlation Analysis ===")
    
    # Filter records with frequency information
    freq_records = [r for r in records if 'ngram_frequency' in r and r.get('ngram_frequency', 0) > 0]
    
    if not freq_records:
        return {'error': 'No records with frequency information found'}
    
    # Get checkpoint stages
    available_checkpoints = sorted(set(r['checkpoint_step'] for r in freq_records), key=lambda x: int(x))
    if checkpoint_stages is None:
        n_checkpoints = len(available_checkpoints)
        stage_indices = [0, n_checkpoints//2, n_checkpoints-1]
        checkpoint_stages = [available_checkpoints[i] for i in stage_indices if i < len(available_checkpoints)]
    
    # Get target layers
    if target_layers is None:
        target_layers = sorted(set(r['layer'] for r in freq_records))
    
    correlation_data = []
    
    for stage in checkpoint_stages:
        stage_records = [r for r in freq_records if r['checkpoint_step'] == stage]
        
        for layer in target_layers:
            layer_stage_records = [r for r in stage_records if r['layer'] == layer]
            
            if len(layer_stage_records) < 5:
                continue
            
            # Group by frequency ranges for better analysis
            frequencies = [r['ngram_frequency'] for r in layer_stage_records]
            freq_percentiles = np.percentile(frequencies, [25, 50, 75])
            
            for i, record in enumerate(layer_stage_records):
                # Compute polytope metrics for individual record's context
                # (In practice, you'd batch similar records together)
                freq = record['ngram_frequency']
                
                # Determine frequency category
                if freq <= freq_percentiles[0]:
                    freq_category = 'low'
                elif freq <= freq_percentiles[1]:
                    freq_category = 'medium-low'
                elif freq <= freq_percentiles[2]:
                    freq_category = 'medium-high'
                else:
                    freq_category = 'high'
                
                correlation_point = {
                    'checkpoint_stage': stage,
                    'checkpoint_num': int(stage),
                    'layer': layer,
                    'ngram_frequency': freq,
                    'frequency_category': freq_category,
                    'activation_norm': record['activation_norm'],
                    'sparsity': record.get('sparsity', 0),
                    'n_active_neurons': record.get('n_active_neurons', 0),
                    'effective_dimension': np.sum((record['activation_vector'] > 0).astype(float))
                }
                correlation_data.append(correlation_point)
    
    df = pd.DataFrame(correlation_data)
    
    # Compute correlations for each stage and layer
    correlation_results = {}
    
    for stage in checkpoint_stages:
        stage_df = df[df['checkpoint_stage'] == stage]
        stage_correlations = {}
        
        for layer in target_layers:
            layer_df = stage_df[stage_df['layer'] == layer]
            
            if len(layer_df) < 5:
                continue
            
            # Compute correlations
            metrics = ['activation_norm', 'sparsity', 'n_active_neurons', 'effective_dimension']
            layer_corr = {}
            
            for metric in metrics:
                if len(layer_df) > 2:
                    corr, p_value = stats.pearsonr(layer_df['ngram_frequency'], layer_df[metric])
                    layer_corr[metric] = {'correlation': corr, 'p_value': p_value}
            
            stage_correlations[layer] = layer_corr
        
        correlation_results[stage] = stage_correlations
    
    return {
        'correlation_data': df,
        'correlation_results': correlation_results,
        'stages': checkpoint_stages,
        'layers': target_layers,
        'summary': {
            'n_data_points': len(correlation_data),
            'frequency_range': (df['ngram_frequency'].min(), df['ngram_frequency'].max()),
            'stages_analyzed': len(checkpoint_stages)
        }
    }


def run_full_analysis(records: List[Dict], target_layers: List[int] = None) -> Dict:
    """
    Run complete polytope analysis pipeline
    
    Args:
        records: List of activation records with keys:
                 - 'layer': int
                 - 'activation_vector': np.ndarray
                 - 'activation_norm': float
                 - 'ngram_frequency': float (optional)
        target_layers: Layers to analyze (None for all)
        
    Returns:
        Dictionary with all results
    """
    print("=== Running Full Polytope Analysis ===")
    print(f"Total records: {len(records)}")
    
    if target_layers:
        print(f"Target layers: {target_layers}")
    
    # 1. Layer-wise polytope analysis
    print("\n1. Layer-wise polytope analysis...")
    layer_analysis = analyze_layer_polytopes(records, target_layers, use_approximation=True)
    
    # 2. Frequency-polytope relationship
    print("\n2. Frequency-polytope relationship analysis...")
    freq_analysis = analyze_frequency_polytope_relationship(records, target_layers)
    
    # 3. Plot results
    if not freq_analysis.empty:
        print("\n3. Plotting results...")
        plot_polytope_analysis(freq_analysis)
    
    return {
        'layer_analysis': layer_analysis,
        'frequency_analysis': freq_analysis,
        'summary': {
            'n_layers': len(layer_analysis),
            'n_records': len(records),
            'layers_analyzed': list(layer_analysis.keys())
        }
    }


def adaptive_checkpoint_selection(available_checkpoints: List[str], 
                                max_checkpoints: int = 15) -> List[str]:
    """
    Adaptive checkpoint selection strategy based on training dynamics
    Dense early training, sparse late training
    
    Args:
        available_checkpoints: All available checkpoint steps
        max_checkpoints: Maximum checkpoints to select
        
    Returns:
        Selected checkpoints following adaptive strategy
    """
    if len(available_checkpoints) <= max_checkpoints:
        return available_checkpoints
    
    # Sort checkpoints by step number
    sorted_checkpoints = sorted(available_checkpoints, key=lambda x: int(x))
    n_total = len(sorted_checkpoints)
    
    selected = []
    
    # Early training (first 20%): every checkpoint or every 2nd
    early_end = min(int(n_total * 0.2), n_total)
    early_stride = max(1, early_end // (max_checkpoints // 3))
    selected.extend(sorted_checkpoints[0:early_end:early_stride])
    
    # Middle training (20%-60%): every 3-5 checkpoints
    middle_start = early_end
    middle_end = min(int(n_total * 0.6), n_total)
    middle_stride = max(3, (middle_end - middle_start) // (max_checkpoints // 3))
    selected.extend(sorted_checkpoints[middle_start:middle_end:middle_stride])
    
    # Late training (60%-100%): every 5-10 checkpoints
    late_start = middle_end
    late_stride = max(5, (n_total - late_start) // (max_checkpoints // 3))
    selected.extend(sorted_checkpoints[late_start::late_stride])
    
    # Always include the last checkpoint
    if sorted_checkpoints[-1] not in selected:
        selected.append(sorted_checkpoints[-1])
    
    # Remove duplicates and sort
    selected = sorted(list(set(selected)), key=lambda x: int(x))
    
    # Trim to max_checkpoints if still too many
    if len(selected) > max_checkpoints:
        indices = np.linspace(0, len(selected)-1, max_checkpoints, dtype=int)
        selected = [selected[i] for i in indices]
    
    return selected


def bootstrap_confidence_intervals(data: np.ndarray, 
                                 statistic_func: callable,
                                 n_bootstrap: int = 1000,
                                 confidence_level: float = 0.95) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence intervals for a statistic
    
    Args:
        data: Input data array
        statistic_func: Function to compute statistic (e.g., np.mean)
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level (0-1)
        
    Returns:
        (statistic_value, lower_bound, upper_bound)
    """
    n_samples = len(data)
    if n_samples < 2:
        stat_val = statistic_func(data) if len(data) > 0 else 0
        return stat_val, stat_val, stat_val
        
    bootstrap_stats = []
    
    for _ in range(n_bootstrap):
        # Bootstrap resample
        bootstrap_sample = np.random.choice(data, size=n_samples, replace=True)
        bootstrap_stat = statistic_func(bootstrap_sample)
        bootstrap_stats.append(bootstrap_stat)
    
    bootstrap_stats = np.array(bootstrap_stats)
    
    # Original statistic
    original_stat = statistic_func(data)
    
    # Confidence interval
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    lower_bound = np.percentile(bootstrap_stats, lower_percentile)
    upper_bound = np.percentile(bootstrap_stats, upper_percentile)
    
    return original_stat, lower_bound, upper_bound


def enhanced_run_analysis(records: List[Dict], 
                        target_layers: List[int] = None,
                        checkpoint_selection: str = "adaptive",
                        max_checkpoints: int = 15) -> Dict[str, Any]:
    """
    Enhanced analysis pipeline with all three visualization strategies
    
    Args:
        records: List of activation records
        target_layers: Layers to analyze (None for all)
        checkpoint_selection: "adaptive", "uniform", or "all"
        max_checkpoints: Maximum checkpoints for temporal analysis
        
    Returns:
        Complete analysis results
    """
    print("=== Enhanced Multi-Dimensional Polytope Analysis ===")
    print(f"Total records: {len(records)}")
    
    # Get available checkpoints and layers
    available_checkpoints = sorted(set(r['checkpoint_step'] for r in records), key=lambda x: int(x))
    available_layers = sorted(set(r['layer'] for r in records))
    
    if target_layers is None:
        target_layers = available_layers
    
    print(f"Available checkpoints: {len(available_checkpoints)}")
    print(f"Available layers: {available_layers}")
    print(f"Target layers: {target_layers}")
    
    # Select checkpoints based on strategy
    if checkpoint_selection == "adaptive":
        selected_checkpoints = adaptive_checkpoint_selection(available_checkpoints, max_checkpoints)
    elif checkpoint_selection == "uniform":
        if len(available_checkpoints) > max_checkpoints:
            indices = np.linspace(0, len(available_checkpoints)-1, max_checkpoints, dtype=int)
            selected_checkpoints = [available_checkpoints[i] for i in indices]
        else:
            selected_checkpoints = available_checkpoints
    else:  # "all"
        selected_checkpoints = available_checkpoints
    
    print(f"Selected {len(selected_checkpoints)} checkpoints: {selected_checkpoints}")
    
    results = {}
    
    # 1. Temporal Evolution Analysis (Primary)
    print("\n=== 1. Temporal Evolution Analysis ===")
    middle_layer = target_layers[len(target_layers)//2] if target_layers else 6
    temporal_results = temporal_evolution_analysis(
        records, target_layer=middle_layer, checkpoint_order=selected_checkpoints
    )
    results['temporal_evolution'] = temporal_results
    
    # 2. Layer Progression Analysis (Secondary)
    print("\n=== 2. Layer-wise Progression Analysis ===")
    # Select key checkpoints for layer analysis
    n_checkpoints = len(selected_checkpoints)
    key_checkpoint_indices = [0, n_checkpoints//2, n_checkpoints-1]
    key_checkpoints = [selected_checkpoints[i] for i in key_checkpoint_indices if i < len(selected_checkpoints)]
    
    progression_results = layer_progression_analysis(
        records, target_checkpoints=key_checkpoints, target_layers=target_layers
    )
    results['layer_progression'] = progression_results
    
    # 3. Enhanced Correlation Analysis (Supporting)
    print("\n=== 3. Enhanced Correlation Analysis ===")
    correlation_results = enhanced_correlation_analysis(
        records, target_layers=target_layers, checkpoint_stages=key_checkpoints
    )
    results['correlation_analysis'] = correlation_results
    
    # 4. Generate publication-ready figures
    print("\n=== 4. Generating NeurIPS Figures ===")
    saved_figures = create_neurips_figures(
        records, target_layer=middle_layer, checkpoint_order=selected_checkpoints
    )
    results['figures'] = saved_figures
    
    # 5. Summary statistics with confidence intervals
    print("\n=== 5. Computing Summary Statistics ===")
    summary_stats = {}
    
    # Volume evolution statistics
    if 'overall_evolution' in temporal_results and temporal_results['overall_evolution']:
        volumes = [point['volume'] for point in temporal_results['overall_evolution']]
        if len(volumes) > 1:
            vol_mean, vol_lower, vol_upper = bootstrap_confidence_intervals(
                np.array(volumes), np.mean, n_bootstrap=1000
            )
            summary_stats['volume_evolution'] = {
                'mean': vol_mean,
                'ci_lower': vol_lower,
                'ci_upper': vol_upper,
                'trend_correlation': temporal_results['statistics'].get('volume_trend', 0)
            }
    
    # Layer complexity statistics
    if not progression_results['progression_data'].empty:
        complexity_by_layer = progression_results['progression_data'].groupby('layer')['effective_dimension'].mean()
        summary_stats['layer_complexity'] = {
            'mean_by_layer': complexity_by_layer.to_dict(),
            'overall_mean': complexity_by_layer.mean(),
            'complexity_variance': complexity_by_layer.var()
        }
    
    results['summary_statistics'] = summary_stats
    
    # 6. Overall summary
    results['analysis_summary'] = {
        'n_records': len(records),
        'n_checkpoints': len(selected_checkpoints),
        'n_layers': len(target_layers),
        'checkpoint_range': f"{selected_checkpoints[0]}-{selected_checkpoints[-1]}",
        'primary_layer': middle_layer,
        'figures_generated': list(saved_figures.keys())
    }
    
    print("\n=== Analysis Complete ===")
    print(f"Analyzed {len(records):,} records across {len(selected_checkpoints)} checkpoints")
    print(f"Generated {len(saved_figures)} publication figures")
    
    return results


# Example usage
if __name__ == "__main__":
    # Create synthetic data for testing the enhanced analysis
    print("Creating synthetic test data for enhanced analysis...")
    
    np.random.seed(42)
    test_records = []
    
    # Simulate multiple checkpoints and layers
    checkpoints = ['100', '500', '1000', '2000', '3000', '5000', '8000', '10000']
    layers = [2, 4, 6, 8, 10, 12]
    
    for checkpoint in checkpoints:
        checkpoint_num = int(checkpoint)
        
        for layer in layers:
            # Layer and checkpoint dependent patterns
            layer_scale = 0.5 + layer * 0.1
            checkpoint_scale = 0.8 + (checkpoint_num / 10000) * 0.4
            
            n_samples = 50  # Records per layer per checkpoint
            
            for i in range(n_samples):
                # Create activation with realistic patterns
                activation = np.random.randn(768) * layer_scale * checkpoint_scale
                
                # Add some structure to activations
                if layer > 6:  # Later layers more structured
                    activation[:100] *= 2.0  # Amplify some dimensions
                
                # Frequency patterns (higher frequency = more structured)
                freq = np.random.exponential(1.0) * (layer / 6.0)
                
                record = {
                    'checkpoint_step': checkpoint,
                    'layer': layer,
                    'activation_vector': activation,
                    'activation_norm': np.linalg.norm(activation),
                    'ngram_frequency': freq,
                    'sparsity': 1.0 - (np.count_nonzero(activation) / len(activation)),
                    'n_active_neurons': int(np.count_nonzero(activation))
                }
                test_records.append(record)
    
    print(f"Created {len(test_records):,} synthetic records")
    print(f"Checkpoints: {checkpoints}")
    print(f"Layers: {layers}")
    
    # Run enhanced analysis
    results = enhanced_run_analysis(
        test_records, 
        target_layers=[4, 6, 8, 10], 
        checkpoint_selection="adaptive",
        max_checkpoints=10
    )
    
    print("\n=== Final Analysis Summary ===")
    summary = results['analysis_summary']
    print(f"Records analyzed: {summary['n_records']:,}")
    print(f"Checkpoints: {summary['n_checkpoints']}")
    print(f"Layers: {summary['n_layers']}")
    print(f"Primary layer: {summary['primary_layer']}")
    print(f"Figures: {summary['figures_generated']}")
    
    if 'volume_evolution' in results['summary_statistics']:
        vol_stats = results['summary_statistics']['volume_evolution']
        print(f"\nVolume evolution trend: {vol_stats['trend_correlation']:.3f}")
        print(f"Mean volume: {vol_stats['mean']:.6f} [{vol_stats['ci_lower']:.6f}, {vol_stats['ci_upper']:.6f}]")