#!/usr/bin/env python3
"""
Simplified Polytope Metrics for Activation Analysis
Simple functions for geometric analysis of neural activation polytopes
"""

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, distance_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any, List, Tuple, Optional
import matplotlib.pyplot as plt
import warnings
from tqdm import tqdm


def preprocess_activations(activations: np.ndarray, 
                         normalize: bool = True,
                         remove_duplicates: bool = True) -> np.ndarray:
    """
    Clean and preprocess activation vectors
    
    Args:
        activations: Array of shape (n_samples, n_features)
        normalize: Whether to standardize features
        remove_duplicates: Whether to remove duplicate vectors
        
    Returns:
        Cleaned activation array
    """
    # Remove zero vectors
    non_zero_mask = np.any(activations != 0, axis=1)
    if not np.any(non_zero_mask):
        raise ValueError("All activation vectors are zero")
    
    cleaned = activations[non_zero_mask]
    
    # Remove duplicates if requested
    if remove_duplicates:
        cleaned = np.unique(cleaned, axis=0)
    
    # Normalize if requested
    if normalize:
        scaler = StandardScaler()
        cleaned = scaler.fit_transform(cleaned)
    
    return cleaned


def reduce_dimensions(activations: np.ndarray, n_components: int = 10) -> Tuple[np.ndarray, PCA]:
    """
    Reduce dimensionality using PCA
    
    Args:
        activations: Input activation vectors
        n_components: Number of PCA components
        
    Returns:
        Tuple of (reduced_activations, fitted_pca)
    """
    if activations.shape[1] <= n_components:
        return activations, None
    
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(activations)
    return reduced, pca


def compute_convex_hull_metrics(points: np.ndarray, min_points: int = 4) -> Dict[str, Any]:
    """
    Compute basic convex hull metrics
    
    Args:
        points: Array of points to analyze
        min_points: Minimum points needed for hull computation
        
    Returns:
        Dictionary with hull metrics
    """
    if len(points) < min_points:
        return {
            'volume': 0.0,
            'surface_area': 0.0,
            'n_vertices': 0,
            'n_faces': 0,
            'hull_valid': False,
            'error': f"Insufficient points: {len(points)} < {min_points}"
        }
    
    try:
        hull = ConvexHull(points)
        
        # Compute volume (handle different dimensions)
        if points.shape[1] == 1:
            volume = np.max(points) - np.min(points)
        else:
            volume = hull.volume
        
        return {
            'volume': float(volume),
            'surface_area': float(hull.area),
            'n_vertices': len(hull.vertices),
            'n_faces': len(hull.simplices),
            'hull_valid': True,
            'hull': hull
        }
        
    except Exception as e:
        # Fallback: use bounding box volume
        ranges = np.max(points, axis=0) - np.min(points, axis=0)
        volume = np.prod(ranges)
        
        return {
            'volume': float(volume),
            'surface_area': 0.0,
            'n_vertices': len(points),
            'n_faces': 0,
            'hull_valid': False,
            'error': str(e)
        }


def compute_geometric_properties(points: np.ndarray) -> Dict[str, float]:
    """
    Compute geometric properties of point set
    
    Args:
        points: Array of points
        
    Returns:
        Dictionary with geometric properties
    """
    if len(points) < 2:
        return {
            'diameter': 0.0,
            'radius': 0.0,
            'aspect_ratio': 1.0
        }
    
    # Diameter: maximum distance between points
    distances = distance_matrix(points, points)
    diameter = np.max(distances)
    
    # Radius: max distance from centroid
    center = np.mean(points, axis=0)
    radii = np.linalg.norm(points - center, axis=1)
    radius = np.max(radii)
    
    # Aspect ratio: ratio of largest to smallest principal axis
    centered = points - center
    if len(centered) > points.shape[1]:  # Only if we have enough points
        try:
            _, s, _ = np.linalg.svd(centered, full_matrices=False)
            aspect_ratio = s[0] / s[-1] if s[-1] > 0 else float('inf')
        except:
            aspect_ratio = 1.0
    else:
        aspect_ratio = 1.0
    
    return {
        'diameter': float(diameter),
        'radius': float(radius),
        'aspect_ratio': float(aspect_ratio)
    }


def compute_effective_dimension(points: np.ndarray, variance_threshold: float = 0.95) -> int:
    """
    Compute effective dimensionality using PCA
    
    Args:
        points: Input points
        variance_threshold: Cumulative variance threshold
        
    Returns:
        Effective dimension
    """
    if points.shape[1] <= 1:
        return points.shape[1]
    
    try:
        pca = PCA()
        pca.fit(points)
        
        cumsum_var = np.cumsum(pca.explained_variance_ratio_)
        effective_dim = np.argmax(cumsum_var >= variance_threshold) + 1
        
        return min(effective_dim, len(pca.explained_variance_ratio_))
    except:
        return points.shape[1]


def compute_complexity_score(hull_metrics: Dict[str, Any], points: np.ndarray) -> float:
    """
    Compute polytope complexity score
    
    Args:
        hull_metrics: Results from compute_convex_hull_metrics
        points: Original points
        
    Returns:
        Complexity score between 0 and 1
    """
    if not hull_metrics['hull_valid']:
        return 0.0
    
    n_points = len(points)
    n_vertices = hull_metrics['n_vertices']
    n_faces = hull_metrics['n_faces']
    dimension = points.shape[1]
    
    # Vertex efficiency: ratio of vertices to total points
    vertex_efficiency = n_vertices / n_points if n_points > 0 else 0
    
    # Face density: faces per vertex
    face_density = n_faces / n_vertices if n_vertices > 0 else 0
    
    # Dimensional complexity
    theoretical_max_vertices = min(2 ** dimension, n_points)
    vertex_complexity = n_vertices / theoretical_max_vertices if theoretical_max_vertices > 0 else 0
    
    # Combine metrics
    complexity = (vertex_efficiency + face_density + vertex_complexity) / 3
    return min(complexity, 1.0)


def compute_stability_score(points: np.ndarray, 
                          n_samples: int = 50,
                          subsample_ratio: float = 0.7) -> float:
    """
    Compute stability score using bootstrap sampling
    
    Args:
        points: Input points
        n_samples: Number of bootstrap samples
        subsample_ratio: Fraction of points to sample
        
    Returns:
        Stability score (higher = more stable)
    """
    n_points = len(points)
    subsample_size = max(4, int(n_points * subsample_ratio))
    
    if subsample_size >= n_points or n_points < 8:
        return 0.0
    
    volumes = []
    
    for _ in range(n_samples):
        try:
            # Random subsample
            indices = np.random.choice(n_points, subsample_size, replace=False)
            subset_points = points[indices]
            
            # Compute volume
            hull_metrics = compute_convex_hull_metrics(subset_points)
            if hull_metrics['hull_valid']:
                volumes.append(hull_metrics['volume'])
        except:
            continue
    
    if len(volumes) < 2:
        return 0.0
    
    # Stability as inverse of coefficient of variation
    mean_volume = np.mean(volumes)
    std_volume = np.std(volumes)
    
    if mean_volume == 0:
        return 0.0
    
    cv = std_volume / mean_volume
    return 1.0 / (1.0 + cv)


def analyze_activation_polytope(activation_vectors: np.ndarray,
                              normalize: bool = True,
                              pca_components: int = 10,
                              compute_stability: bool = True) -> Dict[str, Any]:
    """
    Complete polytope analysis of activation vectors
    
    Args:
        activation_vectors: Array of shape (n_samples, n_features)
        normalize: Whether to normalize activations
        pca_components: Number of PCA components for dimensionality reduction
        compute_stability: Whether to compute stability metrics
        
    Returns:
        Dictionary with all polytope metrics
    """
    
    # Preprocess
    try:
        processed = preprocess_activations(activation_vectors, normalize=normalize)
    except ValueError as e:
        return {'error': str(e), 'valid': False}
    
    # Reduce dimensionality
    reduced, pca = reduce_dimensions(processed, pca_components)
    
    # Compute hull metrics
    hull_metrics = compute_convex_hull_metrics(reduced)
    
    # Compute geometric properties
    geometric_props = compute_geometric_properties(reduced)
    
    # Compute complexity
    complexity = compute_complexity_score(hull_metrics, reduced)
    
    # Compute effective dimension
    effective_dim = compute_effective_dimension(processed)
    
    # Compute stability if requested
    stability = 0.0
    if compute_stability and len(processed) >= 8:
        stability = compute_stability_score(reduced)
    
    # Dimensional spread
    dimensional_spread = np.var(reduced, axis=0).tolist()
    
    # Combine all metrics
    result = {
        'volume': hull_metrics['volume'],
        'surface_area': hull_metrics['surface_area'],
        'n_vertices': hull_metrics['n_vertices'],
        'n_faces': hull_metrics['n_faces'],
        'diameter': geometric_props['diameter'],
        'radius': geometric_props['radius'],
        'aspect_ratio': geometric_props['aspect_ratio'],
        'complexity_score': complexity,
        'effective_dimension': effective_dim,
        'stability_score': stability,
        'dimensional_spread': dimensional_spread,
        'valid': hull_metrics['hull_valid'],
        'n_original_points': len(activation_vectors),
        'n_processed_points': len(processed),
        'original_dimension': activation_vectors.shape[1],
        'reduced_dimension': reduced.shape[1]
    }
    
    # Add PCA info if available
    if pca is not None:
        result['pca_variance_explained'] = pca.explained_variance_ratio_.sum()
    
    return result


def analyze_polytope_evolution(activation_data: Dict[str, np.ndarray],
                             **kwargs) -> pd.DataFrame:
    """
    Analyze polytope evolution across checkpoints
    
    Args:
        activation_data: Dict mapping checkpoint -> activation matrix
        **kwargs: Additional arguments for analyze_activation_polytope
        
    Returns:
        DataFrame with evolution metrics
    """
    results = []
    
    for checkpoint, activations in tqdm(activation_data.items(), desc="Analyzing evolution"):
        metrics = analyze_activation_polytope(activations, **kwargs)
        
        row = {'checkpoint': checkpoint}
        row.update(metrics)
        results.append(row)
    
    return pd.DataFrame(results)


def compare_polytopes(activation_groups: List[np.ndarray],
                     group_names: List[str] = None,
                     **kwargs) -> Dict[str, Any]:
    """
    Compare polytopes across different activation groups
    
    Args:
        activation_groups: List of activation matrices
        group_names: Names for each group
        **kwargs: Additional arguments for analyze_activation_polytope
        
    Returns:
        Comparison results
    """
    if group_names is None:
        group_names = [f"Group_{i}" for i in range(len(activation_groups))]
    
    # Analyze each group
    metrics_list = []
    for activations in activation_groups:
        metrics = analyze_activation_polytope(activations, **kwargs)
        metrics_list.append(metrics)
    
    # Extract key metrics for comparison
    volumes = [m.get('volume', 0) for m in metrics_list]
    complexities = [m.get('complexity_score', 0) for m in metrics_list]
    dimensions = [m.get('effective_dimension', 0) for m in metrics_list]
    
    # Compute relative metrics
    base_volume = volumes[0] if volumes[0] > 0 else 1
    volume_ratios = [v / base_volume for v in volumes]
    
    return {
        'group_names': group_names,
        'volumes': volumes,
        'volume_ratios': volume_ratios,
        'complexities': complexities,
        'effective_dimensions': dimensions,
        'detailed_metrics': metrics_list,
        'summary': {
            'volume_mean': np.mean(volumes),
            'volume_std': np.std(volumes),
            'complexity_mean': np.mean(complexities),
            'complexity_std': np.std(complexities)
        }
    }


def plot_polytope_evolution(evolution_df: pd.DataFrame,
                           metrics: List[str] = None,
                           figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
    """
    Plot polytope evolution over checkpoints
    
    Args:
        evolution_df: DataFrame from analyze_polytope_evolution
        metrics: List of metrics to plot
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    if metrics is None:
        metrics = ['volume', 'complexity_score', 'stability_score', 'effective_dimension']
    
    # Filter available metrics
    available_metrics = [m for m in metrics if m in evolution_df.columns]
    n_metrics = len(available_metrics)
    
    if n_metrics == 0:
        raise ValueError("No valid metrics found in DataFrame")
    
    # Create subplots
    n_cols = 2
    n_rows = (n_metrics + 1) // 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    if n_metrics == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    
    axes_flat = axes.flatten()
    
    # Plot each metric
    for i, metric in enumerate(available_metrics):
        ax = axes_flat[i]
        
        # Handle checkpoint column
        if 'checkpoint' in evolution_df.columns:
            x_data = pd.to_numeric(evolution_df['checkpoint'], errors='coerce')
            x_label = 'Checkpoint'
        else:
            x_data = evolution_df.index
            x_label = 'Index'
        
        ax.plot(x_data, evolution_df[metric], marker='o', linewidth=2, markersize=6)
        ax.set_xlabel(x_label)
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'{metric.replace("_", " ").title()} Evolution')
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_metrics, len(axes_flat)):
        axes_flat[i].set_visible(False)
    
    plt.tight_layout()
    return fig


# Convenience functions for common analyses
def quick_polytope_analysis(activations: np.ndarray) -> Dict[str, Any]:
    """Quick analysis with default settings"""
    return analyze_activation_polytope(activations)


def compare_ngram_polytopes(high_freq_activations: np.ndarray,
                           low_freq_activations: np.ndarray) -> Dict[str, Any]:
    """Compare high vs low frequency n-gram polytopes"""
    return compare_polytopes(
        [high_freq_activations, low_freq_activations],
        ['High-frequency', 'Low-frequency']
    )


# Example usage
def main():
    """Example usage of simplified polytope analysis"""
    # Generate example data
    np.random.seed(42)
    n_samples, n_features = 100, 512
    
    # High-frequency n-gram: more structured
    high_freq = np.random.normal(0, 1, (n_samples, n_features))
    high_freq[:, :10] *= 5
    
    # Low-frequency n-gram: more distributed  
    low_freq = np.random.normal(0, 1, (n_samples, n_features))
    
    # Analyze
    high_metrics = quick_polytope_analysis(high_freq)
    low_metrics = quick_polytope_analysis(low_freq)
    
    print("High-frequency n-gram:")
    print(f"  Volume: {high_metrics['volume']:.4f}")
    print(f"  Complexity: {high_metrics['complexity_score']:.4f}")
    print(f"  Effective dimension: {high_metrics['effective_dimension']}")
    
    print("\nLow-frequency n-gram:")
    print(f"  Volume: {low_metrics['volume']:.4f}")
    print(f"  Complexity: {low_metrics['complexity_score']:.4f}")
    print(f"  Effective dimension: {low_metrics['effective_dimension']}")
    
    # Compare
    comparison = compare_ngram_polytopes(high_freq, low_freq)
    print(f"\nVolume ratio (low/high): {comparison['volume_ratios'][1]:.4f}")


if __name__ == "__main__":
    main()