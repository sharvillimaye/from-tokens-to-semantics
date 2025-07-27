#!/usr/bin/env python3
"""
Comprehensive Polytope Metrics for Activation Analysis
Advanced geometric analysis of neural activation polytopes for polysemanticity research
"""

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, distance_matrix
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Any, Tuple, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
from dataclasses import dataclass
import pickle
from pathlib import Path


@dataclass
class PolytopeMetrics:
    """Container for polytope analysis results"""
    volume: float
    surface_area: float
    n_vertices: int
    n_faces: int
    diameter: float
    radius: float
    aspect_ratio: float
    complexity_score: float
    dimensional_spread: List[float]
    effective_dimension: int
    stability_score: float
    metadata: Dict[str, Any]


class AdvancedPolytopeAnalyzer:
    """Advanced analyzer for neural activation polytopes"""
    
    def __init__(self, 
                 pca_components: int = 10,
                 min_points_for_hull: int = 4,
                 stability_samples: int = 100):
        """
        Initialize the polytope analyzer
        
        Args:
            pca_components: Number of PCA components for dimensionality reduction
            min_points_for_hull: Minimum points needed to compute convex hull
            stability_samples: Number of bootstrap samples for stability analysis
        """
        self.pca_components = pca_components
        self.min_points_for_hull = min_points_for_hull
        self.stability_samples = stability_samples
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=pca_components)
        
    def analyze_activation_polytope(self, 
                                  activation_vectors: np.ndarray,
                                  normalize: bool = True,
                                  compute_stability: bool = True) -> PolytopeMetrics:
        """
        Comprehensive polytope analysis of activation vectors
        
        Args:
            activation_vectors: Array of shape (n_samples, n_features)
            normalize: Whether to normalize activation vectors
            compute_stability: Whether to compute stability metrics
            
        Returns:
            PolytopeMetrics object with all computed metrics
        """
        
        if len(activation_vectors) < self.min_points_for_hull:
            return self._create_empty_metrics(
                f"Insufficient points: {len(activation_vectors)} < {self.min_points_for_hull}"
            )
        
        try:
            # Preprocess activations
            processed_activations = self._preprocess_activations(activation_vectors, normalize)
            
            # Reduce dimensionality for polytope computation
            reduced_activations = self._reduce_dimensionality(processed_activations)
            
            # Compute convex hull
            hull = ConvexHull(reduced_activations)
            
            # Calculate basic metrics
            volume = self._compute_volume(hull, reduced_activations)
            surface_area = self._compute_surface_area(hull)
            n_vertices = len(hull.vertices)
            n_faces = len(hull.simplices)
            
            # Calculate geometric properties
            diameter = self._compute_diameter(reduced_activations[hull.vertices])
            radius = self._compute_circumradius(reduced_activations[hull.vertices])
            aspect_ratio = self._compute_aspect_ratio(reduced_activations[hull.vertices])
            
            # Calculate complexity and dimensional metrics
            complexity_score = self._compute_complexity_score(hull, reduced_activations)
            dimensional_spread = self._compute_dimensional_spread(reduced_activations)
            effective_dimension = self._compute_effective_dimension(processed_activations)
            
            # Calculate stability if requested
            stability_score = 0.0
            if compute_stability and len(activation_vectors) >= 2 * self.min_points_for_hull:
                stability_score = self._compute_stability_score(reduced_activations)
            
            # Create metadata
            metadata = {
                'n_original_points': len(activation_vectors),
                'original_dimension': activation_vectors.shape[1],
                'reduced_dimension': reduced_activations.shape[1],
                'hull_valid': True,
                'preprocessing_applied': normalize,
                'pca_explained_variance_ratio': self.pca.explained_variance_ratio_.tolist(),
                'total_variance_explained': self.pca.explained_variance_ratio_.sum()
            }
            
            return PolytopeMetrics(
                volume=volume,
                surface_area=surface_area,
                n_vertices=n_vertices,
                n_faces=n_faces,
                diameter=diameter,
                radius=radius,
                aspect_ratio=aspect_ratio,
                complexity_score=complexity_score,
                dimensional_spread=dimensional_spread,
                effective_dimension=effective_dimension,
                stability_score=stability_score,
                metadata=metadata
            )
            
        except Exception as e:
            warnings.warn(f"Error in polytope analysis: {e}")
            return self._create_empty_metrics(f"Analysis error: {str(e)}")
    
    def _preprocess_activations(self, activations: np.ndarray, normalize: bool) -> np.ndarray:
        """Preprocess activation vectors"""
        
        # Remove zero vectors and duplicates
        non_zero_mask = np.any(activations != 0, axis=1)
        if not np.any(non_zero_mask):
            raise ValueError("All activation vectors are zero")
        
        cleaned_activations = activations[non_zero_mask]
        
        # Remove duplicate vectors
        unique_activations = np.unique(cleaned_activations, axis=0)
        
        if normalize:
            # Standardize features
            unique_activations = self.scaler.fit_transform(unique_activations)
        
        return unique_activations
    
    def _reduce_dimensionality(self, activations: np.ndarray) -> np.ndarray:
        """Reduce dimensionality using PCA"""
        
        if activations.shape[1] <= self.pca_components:
            return activations
        
        # Fit PCA and transform
        reduced = self.pca.fit_transform(activations)
        return reduced
    
    def _compute_volume(self, hull: ConvexHull, points: np.ndarray) -> float:
        """Compute polytope volume"""
        try:
            if points.shape[1] == 1:
                # 1D case: length
                return np.max(points) - np.min(points)
            elif points.shape[1] == 2:
                # 2D case: area
                return hull.volume  # In 2D, this is area
            else:
                # nD case: hypervolume
                return hull.volume
        except:
            # Fallback: approximate volume using bounding box
            ranges = np.max(points, axis=0) - np.min(points, axis=0)
            return np.prod(ranges)
    
    def _compute_surface_area(self, hull: ConvexHull) -> float:
        """Compute polytope surface area"""
        try:
            return hull.area
        except:
            # Fallback for cases where area computation fails
            return 0.0
    
    def _compute_diameter(self, vertices: np.ndarray) -> float:
        """Compute maximum distance between vertices"""
        if len(vertices) < 2:
            return 0.0
        
        distances = distance_matrix(vertices, vertices)
        return np.max(distances)
    
    def _compute_circumradius(self, vertices: np.ndarray) -> float:
        """Compute circumradius (radius of smallest enclosing sphere)"""
        if len(vertices) < 2:
            return 0.0
        
        # Approximate as radius of bounding sphere
        center = np.mean(vertices, axis=0)
        distances = np.linalg.norm(vertices - center, axis=1)
        return np.max(distances)
    
    def _compute_aspect_ratio(self, vertices: np.ndarray) -> float:
        """Compute aspect ratio (ratio of largest to smallest principal axis)"""
        if len(vertices) < 2:
            return 1.0
        
        # Use SVD to find principal axes
        centered = vertices - np.mean(vertices, axis=0)
        _, s, _ = np.linalg.svd(centered, full_matrices=False)
        
        if s[-1] == 0:
            return float('inf')
        
        return s[0] / s[-1]
    
    def _compute_complexity_score(self, hull: ConvexHull, points: np.ndarray) -> float:
        """Compute a complexity score based on multiple factors"""
        
        # Normalize metrics to [0, 1] range
        n_points = len(points)
        n_vertices = len(hull.vertices)
        n_faces = len(hull.simplices)
        dimension = points.shape[1]
        
        # Vertex efficiency: ratio of vertices to total points
        vertex_efficiency = n_vertices / n_points if n_points > 0 else 0
        
        # Face density: faces per vertex
        face_density = n_faces / n_vertices if n_vertices > 0 else 0
        
        # Dimensional complexity: compare to theoretical maximum
        theoretical_max_vertices = 2 ** dimension  # Hypercube vertices
        vertex_complexity = n_vertices / theoretical_max_vertices if theoretical_max_vertices > 0 else 0
        
        # Combine metrics
        complexity = (vertex_efficiency + face_density + vertex_complexity) / 3
        return min(complexity, 1.0)  # Cap at 1.0
    
    def _compute_dimensional_spread(self, points: np.ndarray) -> List[float]:
        """Compute spread (variance) in each dimension"""
        return np.var(points, axis=0).tolist()
    
    def _compute_effective_dimension(self, points: np.ndarray) -> int:
        """Compute effective dimensionality using PCA variance threshold"""
        
        if points.shape[1] <= 1:
            return points.shape[1]
        
        # Use current PCA fit or create new one
        if hasattr(self.pca, 'explained_variance_ratio_'):
            explained_var = self.pca.explained_variance_ratio_
        else:
            temp_pca = PCA()
            temp_pca.fit(points)
            explained_var = temp_pca.explained_variance_ratio_
        
        # Find number of components needed for 95% variance
        cumsum_var = np.cumsum(explained_var)
        effective_dim = np.argmax(cumsum_var >= 0.95) + 1
        
        return min(effective_dim, len(explained_var))
    
    def _compute_stability_score(self, points: np.ndarray) -> float:
        """Compute stability score using bootstrap sampling"""
        
        if len(points) < 2 * self.min_points_for_hull:
            return 0.0
        
        n_points = len(points)
        subsample_size = max(self.min_points_for_hull, n_points // 2)
        volumes = []
        
        for _ in range(self.stability_samples):
            try:
                # Random subsample
                indices = np.random.choice(n_points, subsample_size, replace=False)
                subset_points = points[indices]
                
                # Compute hull and volume
                subset_hull = ConvexHull(subset_points)
                volume = self._compute_volume(subset_hull, subset_points)
                volumes.append(volume)
                
            except:
                continue  # Skip failed samples
        
        if len(volumes) < 2:
            return 0.0
        
        # Stability as inverse of coefficient of variation
        mean_volume = np.mean(volumes)
        std_volume = np.std(volumes)
        
        if mean_volume == 0:
            return 0.0
        
        cv = std_volume / mean_volume
        stability = 1.0 / (1.0 + cv)  # Higher stability = lower variation
        
        return stability
    
    def _create_empty_metrics(self, reason: str) -> PolytopeMetrics:
        """Create empty metrics object for failed analysis"""
        return PolytopeMetrics(
            volume=0.0,
            surface_area=0.0,
            n_vertices=0,
            n_faces=0,
            diameter=0.0,
            radius=0.0,
            aspect_ratio=1.0,
            complexity_score=0.0,
            dimensional_spread=[],
            effective_dimension=0,
            stability_score=0.0,
            metadata={'error': reason, 'hull_valid': False}
        )
    
    def analyze_polytope_evolution(self, 
                                 activation_data: Dict[str, np.ndarray],
                                 ngram: str = None,
                                 layer: int = None) -> pd.DataFrame:
        """
        Analyze how polytope metrics evolve across checkpoints
        
        Args:
            activation_data: Dict mapping checkpoint -> activation matrix
            ngram: Specific n-gram to analyze (optional)
            layer: Specific layer to analyze (optional)
            
        Returns:
            DataFrame with polytope metrics across checkpoints
        """
        
        results = []
        
        for checkpoint, activations in tqdm(activation_data.items(), desc="Analyzing evolution"):
            metrics = self.analyze_activation_polytope(activations)
            
            result_row = {
                'checkpoint': checkpoint,
                'ngram': ngram,
                'layer': layer,
                'volume': metrics.volume,
                'surface_area': metrics.surface_area,
                'n_vertices': metrics.n_vertices,
                'n_faces': metrics.n_faces,
                'diameter': metrics.diameter,
                'radius': metrics.radius,
                'aspect_ratio': metrics.aspect_ratio,
                'complexity_score': metrics.complexity_score,
                'effective_dimension': metrics.effective_dimension,
                'stability_score': metrics.stability_score,
                'n_points': metrics.metadata.get('n_original_points', 0),
                'variance_explained': metrics.metadata.get('total_variance_explained', 0)
            }
            
            results.append(result_row)
        
        return pd.DataFrame(results)
    
    def compare_polytopes(self, 
                         polytope_metrics_list: List[PolytopeMetrics],
                         labels: List[str] = None) -> Dict[str, Any]:
        """Compare multiple polytopes across various metrics"""
        
        if labels is None:
            labels = [f"Polytope_{i}" for i in range(len(polytope_metrics_list))]
        
        comparison_data = {
            'labels': labels,
            'volumes': [m.volume for m in polytope_metrics_list],
            'complexities': [m.complexity_score for m in polytope_metrics_list],
            'stabilities': [m.stability_score for m in polytope_metrics_list],
            'dimensions': [m.effective_dimension for m in polytope_metrics_list],
            'n_vertices': [m.n_vertices for m in polytope_metrics_list]
        }
        
        # Compute relative metrics
        base_volume = comparison_data['volumes'][0] if comparison_data['volumes'][0] > 0 else 1
        comparison_data['volume_ratios'] = [v / base_volume for v in comparison_data['volumes']]
        
        # Statistical summaries
        comparison_data['volume_stats'] = {
            'mean': np.mean(comparison_data['volumes']),
            'std': np.std(comparison_data['volumes']),
            'range': np.max(comparison_data['volumes']) - np.min(comparison_data['volumes'])
        }
        
        return comparison_data
    
    def visualize_polytope_evolution(self, 
                                   evolution_df: pd.DataFrame,
                                   metrics: List[str] = None,
                                   figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """Visualize polytope evolution over checkpoints"""
        
        if metrics is None:
            metrics = ['volume', 'complexity_score', 'stability_score', 'effective_dimension']
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics[:4]):
            if metric in evolution_df.columns:
                ax = axes[i]
                
                if 'checkpoint' in evolution_df.columns:
                    # Convert checkpoint to numeric for plotting
                    checkpoints = pd.to_numeric(evolution_df['checkpoint'], errors='coerce')
                    ax.plot(checkpoints, evolution_df[metric], marker='o', linewidth=2)
                    ax.set_xlabel('Checkpoint')
                else:
                    ax.plot(evolution_df.index, evolution_df[metric], marker='o', linewidth=2)
                    ax.set_xlabel('Index')
                
                ax.set_ylabel(metric.replace('_', ' ').title())
                ax.set_title(f'{metric.replace("_", " ").title()} Evolution')
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig


# Convenience functions for common use cases
def get_polytope_metrics(activation_vectors: np.ndarray, **kwargs) -> PolytopeMetrics:
    """
    Convenience function to calculate polytope metrics
    
    Args:
        activation_vectors: Array of activation vectors
        **kwargs: Additional arguments for AdvancedPolytopeAnalyzer
        
    Returns:
        PolytopeMetrics object
    """
    analyzer = AdvancedPolytopeAnalyzer(**kwargs)
    return analyzer.analyze_activation_polytope(activation_vectors)


def compute_polytope_complexity_evolution(checkpoint_activations: Dict[str, np.ndarray]) -> pd.DataFrame:
    """
    Compute polytope complexity evolution across checkpoints
    
    Args:
        checkpoint_activations: Dict mapping checkpoint -> activation matrix
        
    Returns:
        DataFrame with complexity metrics
    """
    analyzer = AdvancedPolytopeAnalyzer()
    return analyzer.analyze_polytope_evolution(checkpoint_activations)


def compare_activation_polytopes(activation_groups: List[np.ndarray], 
                               group_names: List[str] = None) -> Dict[str, Any]:
    """
    Compare polytopes across different activation groups
    
    Args:
        activation_groups: List of activation matrices to compare
        group_names: Names for each group
        
    Returns:
        Comparison results dictionary
    """
    analyzer = AdvancedPolytopeAnalyzer()
    
    # Compute metrics for each group
    metrics_list = []
    for activations in activation_groups:
        metrics = analyzer.analyze_activation_polytope(activations)
        metrics_list.append(metrics)
    
    # Compare polytopes
    return analyzer.compare_polytopes(metrics_list, group_names)


# Example usage
def main():
    """Example usage of polytope analysis"""
    
    # Generate example activation data
    np.random.seed(42)
    n_samples = 100
    n_features = 512
    
    # Simulate activations with different polytope structures
    # High-frequency n-gram: more structured (lower complexity)
    high_freq_activations = np.random.normal(0, 1, (n_samples, n_features))
    high_freq_activations[:, :10] *= 5  # Emphasize first 10 dimensions
    
    # Low-frequency n-gram: more distributed (higher complexity)
    low_freq_activations = np.random.normal(0, 1, (n_samples, n_features))
    
    # Analyze polytopes
    analyzer = AdvancedPolytopeAnalyzer()
    
    high_freq_metrics = analyzer.analyze_activation_polytope(high_freq_activations)
    low_freq_metrics = analyzer.analyze_activation_polytope(low_freq_activations)
    
    print("High-frequency n-gram polytope:")
    print(f"  Volume: {high_freq_metrics.volume:.4f}")
    print(f"  Complexity: {high_freq_metrics.complexity_score:.4f}")
    print(f"  Effective dimension: {high_freq_metrics.effective_dimension}")
    
    print("\nLow-frequency n-gram polytope:")
    print(f"  Volume: {low_freq_metrics.volume:.4f}")
    print(f"  Complexity: {low_freq_metrics.complexity_score:.4f}")
    print(f"  Effective dimension: {low_freq_metrics.effective_dimension}")
    
    # Compare polytopes
    comparison = analyzer.compare_polytopes(
        [high_freq_metrics, low_freq_metrics],
        ['High-frequency', 'Low-frequency']
    )
    
    print(f"\nVolume ratio (low/high): {comparison['volume_ratios'][1]:.4f}")
    
    return analyzer, high_freq_metrics, low_freq_metrics


if __name__ == "__main__":
    analyzer, high_freq, low_freq = main()