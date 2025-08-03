#!/usr/bin/env python3
"""
Consolidated Polytope Analyzer for Activation Records
Simple, robust analysis of high vs low frequency activation data across checkpoints.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict
from pathlib import Path
import logging
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PolytopeAnalyzer:
    """
    Simple, robust polytope analyzer for activation records.
    
    Usage:
        analyzer = PolytopeAnalyzer()
        results = analyzer.analyze_records(activation_records)
    """
    
    def __init__(self, n_pca_components: int = 20, approximation_epsilon: float = 0.05):
        """
        Initialize the analyzer.
        
        Args:
            n_pca_components: Number of PCA components for dimensionality reduction
            approximation_epsilon: Tolerance for polytope approximation
        """
        self.n_pca_components = n_pca_components
        self.approximation_epsilon = approximation_epsilon
    
    def validate_records(self, records: List[Dict]) -> List[Dict]:
        """
        Validate activation records and ensure required fields exist.
        
        Args:
            records: List of activation records
            
        Returns:
            List of validated records
        """
        validated = []
        required_fields = ['activation_vector', 'frequency']
        
        for i, record in enumerate(records):
            # Check required fields
            if not all(field in record for field in required_fields):
                logger.warning(f"Record {i} missing required fields, skipping")
                continue
            
            # Validate activation vector
            try:
                activation = np.array(record['activation_vector'])
                if np.any(np.isnan(activation)) or np.any(np.isinf(activation)):
                    logger.warning(f"Record {i} has invalid activation values, skipping")
                    continue
                record['activation_vector'] = activation
            except:
                logger.warning(f"Record {i} has invalid activation vector format, skipping")
                continue
            
            # Add default fields if missing
            if 'layer' not in record:
                record['layer'] = 0
            if 'checkpoint' not in record:
                record['checkpoint'] = '0'
            if 'ngram' not in record:
                record['ngram'] = 'unknown'
            if 'binary_pattern' not in record:
                record['binary_pattern'] = np.zeros_like(activation, dtype=bool)
            
            # Compute derived metrics
            record['activation_norm'] = np.linalg.norm(activation)
            record['sparsity'] = 1.0 - (np.count_nonzero(activation) / len(activation))
            record['n_active_neurons'] = int(np.count_nonzero(activation))
            
            validated.append(record)
        
        logger.info(f"Validated {len(validated)}/{len(records)} records")
        return validated
    
    def reduce_dimensions(self, points: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Reduce dimensionality using PCA for computational efficiency.
        
        Args:
            points: Input points (n_samples, n_features)
            
        Returns:
            (reduced_points, pca_info)
        """
        if points.shape[0] < 2:
            return points, {}
        
        n_components = min(self.n_pca_components, points.shape[0] - 1, points.shape[1])
        
        # Standardize data
        scaler = StandardScaler()
        points_scaled = scaler.fit_transform(points)
        
        # Apply PCA
        pca = PCA(n_components=n_components)
        points_reduced = pca.fit_transform(points_scaled)
        
        variance_explained = np.sum(pca.explained_variance_ratio_)
        
        return points_reduced, {
            'variance_explained': variance_explained,
            'n_components': n_components,
            'original_dims': points.shape[1]
        }
    
    def find_hull_vertices(self, points: np.ndarray) -> np.ndarray:
        """
        Find convex hull vertices using greedy approximation for high-dimensional data.
        
        Args:
            points: Input points (n_samples, n_features)
            
        Returns:
            Indices of hull vertices
        """
        n_points, n_dims = points.shape
        
        if n_points <= 3:
            return np.arange(n_points)
        
        # Find extreme points using PCA
        try:
            # Use PCA to find initial extreme points
            pca = PCA(n_components=min(10, n_dims, n_points-1))
            points_pca = pca.fit_transform(StandardScaler().fit_transform(points))
            
            extreme_indices = set()
            
            # Find min/max in each PCA dimension
            for dim in range(points_pca.shape[1]):
                min_idx = np.argmin(points_pca[:, dim])
                max_idx = np.argmax(points_pca[:, dim])
                extreme_indices.add(min_idx)
                extreme_indices.add(max_idx)
            
            # Add points with highest norms
            norms = np.linalg.norm(points_pca, axis=1)
            top_indices = np.argsort(norms)[-5:]
            extreme_indices.update(top_indices)
            
            return np.array(list(extreme_indices))
            
        except Exception as e:
            logger.warning(f"Hull vertex finding failed: {e}, using all points")
            return np.arange(min(50, n_points))  # Limit to 50 points for efficiency
    
    def compute_volume(self, points: np.ndarray) -> float:
        """
        Compute polytope volume using multiple robust methods.
        
        Args:
            points: Input points defining the polytope
            
        Returns:
            Estimated volume
        """
        if len(points) < 2:
            return 0.0
        
        try:
            # Method 1: Try scipy ConvexHull for small point sets
            if len(points) <= 100:
                try:
                    hull = ConvexHull(points)
                    if hasattr(hull, 'volume'):
                        return float(hull.volume)
                except Exception as e:
                    logger.debug(f"ConvexHull volume failed: {e}")
            
            # Method 2: Monte Carlo volume estimation
            return self._monte_carlo_volume(points)
            
        except Exception as e:
            logger.error(f"Volume computation failed: {e}")
            return 0.0
    
    def _monte_carlo_volume(self, points: np.ndarray, n_samples: int = 10000) -> float:
        """Monte Carlo volume estimation."""
        # Find bounding box
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)
        
        # Check for degenerate dimensions
        coord_ranges = max_coords - min_coords
        if np.any(coord_ranges <= 1e-12):
            non_degenerate = np.sum(coord_ranges > 1e-12)
            if non_degenerate <= 1:
                return np.max(coord_ranges)
        
        # Volume of bounding box
        box_volume = np.prod(coord_ranges)
        
        if box_volume <= 1e-20:
            return 0.0
        
        # Generate random points in bounding box
        random_points = np.random.uniform(min_coords, max_coords, size=(n_samples, len(min_coords)))
        
        # Count points inside convex hull (simplified check)
        inside_count = 0
        
        # Use a simple distance-based approximation
        for point in random_points:
            # Check if point is close to the convex hull of input points
            distances = np.linalg.norm(points - point, axis=1)
            min_distance = np.min(distances)
            
            # Simple heuristic: if close to existing points, consider "inside"
            if min_distance <= np.percentile(pdist(points), 25):
                inside_count += 1
        
        volume_ratio = inside_count / n_samples
        return box_volume * volume_ratio
    
    def compute_surface_area(self, points: np.ndarray) -> float:
        """
        Estimate surface area of the polytope.
        
        Args:
            points: Input points defining the polytope
            
        Returns:
            Estimated surface area
        """
        if len(points) < 2:
            return 0.0
        
        try:
            if len(points) <= 100:
                hull = ConvexHull(points)
                if hasattr(hull, 'area'):
                    return float(hull.area)
        except:
            pass
        
        # Fallback: estimate based on point spread and dimensionality
        distances = pdist(points)
        mean_distance = np.mean(distances)
        n_dims = points.shape[1]
        
        # Rough surface area estimate
        return len(points) * (mean_distance ** (n_dims - 1))
    
    def compute_polytope_metrics(self, points: np.ndarray) -> Dict[str, float]:
        """
        Compute comprehensive polytope metrics.
        
        Args:
            points: Input points (n_samples, n_features)
            
        Returns:
            Dictionary of polytope metrics
        """
        n_points, n_dims = points.shape
        
        if n_points < 2:
            return {
                'volume': 0.0,
                'surface_area': 0.0,
                'n_vertices': n_points,
                'n_facets': 0,
                'mean_distance': 0.0,
                'std_distance': 0.0,
                'centroid_variance': 0.0,
                'effective_dimension': 0.0,
                'hull_valid': False
            }
        
        try:
            # Find hull vertices
            hull_indices = self.find_hull_vertices(points)
            hull_points = points[hull_indices]
            
            # Compute volume and surface area
            volume = self.compute_volume(hull_points)
            surface_area = self.compute_surface_area(hull_points)
            
            # Compute distance metrics
            distances = pdist(points)
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
                'volume': float(volume),
                'surface_area': float(surface_area),
                'n_vertices': len(hull_indices),
                'n_facets': 0,  # Computing facets is expensive for high-dim
                'mean_distance': float(mean_distance),
                'std_distance': float(std_distance),
                'centroid_variance': float(centroid_variance),
                'effective_dimension': float(effective_dim),
                'hull_valid': True
            }
            
        except Exception as e:
            logger.error(f"Polytope metrics computation failed: {e}")
            raise RuntimeError(f"Polytope analysis failed: {str(e)}")
    
    def stratify_by_frequency(self, records: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Stratify records into high and low frequency groups.
        
        Args:
            records: List of activation records
            
        Returns:
            Dictionary with 'high' and 'low' frequency groups
        """
        frequencies = [r['frequency'] for r in records]
        
        if not frequencies:
            return {'high': [], 'low': []}
        
        # Use median as threshold for simple binary split
        median_freq = np.median(frequencies)
        
        high_freq = [r for r in records if r['frequency'] >= median_freq]
        low_freq = [r for r in records if r['frequency'] < median_freq]
        
        logger.info(f"Stratified into {len(high_freq)} high-freq and {len(low_freq)} low-freq records")
        return {'high': high_freq, 'low': low_freq}
    
    def analyze_frequency_group(self, records: List[Dict], group_name: str) -> Dict[str, Any]:
        """
        Analyze polytope metrics for a frequency group.
        
        Args:
            records: List of activation records
            group_name: Name of the frequency group
            
        Returns:
            Analysis results for the group
        """
        if len(records) < 2:
            logger.warning(f"Insufficient records for {group_name} group: {len(records)}")
            return {'error': 'Insufficient records', 'n_records': len(records)}
        
        # Extract activation vectors
        activations = np.stack([r['activation_vector'] for r in records])
        
        # Reduce dimensions for computational efficiency
        reduced_activations, pca_info = self.reduce_dimensions(activations)
        
        # Compute polytope metrics
        polytope_metrics = self.compute_polytope_metrics(reduced_activations)
        
        # Compute activation statistics
        activation_stats = {
            'mean_frequency': np.mean([r['frequency'] for r in records]),
            'std_frequency': np.std([r['frequency'] for r in records]),
            'mean_sparsity': np.mean([r['sparsity'] for r in records]),
            'mean_norm': np.mean([r['activation_norm'] for r in records]),
            'mean_active_neurons': np.mean([r['n_active_neurons'] for r in records])
        }
        
        return {
            'n_records': len(records),
            'polytope_metrics': polytope_metrics,
            'activation_stats': activation_stats,
            'pca_info': pca_info,
            'group_name': group_name
        }
    
    def compare_across_checkpoints(self, records: List[Dict]) -> Dict[str, Any]:
        """
        Compare high vs low frequency groups across checkpoints.
        
        Args:
            records: List of activation records
            
        Returns:
            Comparison results across checkpoints
        """
        # Group by checkpoint
        checkpoint_groups = defaultdict(list)
        for record in records:
            checkpoint_groups[record['checkpoint']].append(record)
        
        results = {
            'checkpoint_analysis': {},
            'comparison_summary': {},
            'evolution_metrics': []
        }
        
        for checkpoint, checkpoint_records in checkpoint_groups.items():
            logger.info(f"Analyzing checkpoint {checkpoint} with {len(checkpoint_records)} records")
            
            # Stratify by frequency
            freq_groups = self.stratify_by_frequency(checkpoint_records)
            
            checkpoint_results = {}
            
            # Analyze each frequency group
            for group_name, group_records in freq_groups.items():
                group_analysis = self.analyze_frequency_group(group_records, group_name)
                checkpoint_results[group_name] = group_analysis
            
            # Compare groups if both exist
            if 'high' in checkpoint_results and 'low' in checkpoint_results:
                high_vol = checkpoint_results['high']['polytope_metrics']['volume']
                low_vol = checkpoint_results['low']['polytope_metrics']['volume']
                
                comparison = {
                    'volume_ratio_high_to_low': high_vol / low_vol if low_vol > 1e-10 else float('inf'),
                    'volume_difference': high_vol - low_vol,
                    'sparsity_difference': (
                        checkpoint_results['low']['activation_stats']['mean_sparsity'] -
                        checkpoint_results['high']['activation_stats']['mean_sparsity']
                    ),
                    'norm_ratio': (
                        checkpoint_results['high']['activation_stats']['mean_norm'] /
                        checkpoint_results['low']['activation_stats']['mean_norm']
                        if checkpoint_results['low']['activation_stats']['mean_norm'] > 0 else 1.0
                    )
                }
                checkpoint_results['comparison'] = comparison
            
            results['checkpoint_analysis'][checkpoint] = checkpoint_results
        
        # Compute evolution metrics
        evolution_data = []
        for checkpoint, data in results['checkpoint_analysis'].items():
            if 'comparison' in data:
                evolution_point = {
                    'checkpoint': checkpoint,
                    'volume_ratio': data['comparison']['volume_ratio_high_to_low'],
                    'sparsity_difference': data['comparison']['sparsity_difference'],
                    'norm_ratio': data['comparison']['norm_ratio']
                }
                evolution_data.append(evolution_point)
        
        results['evolution_metrics'] = evolution_data
        
        return results
    
    def create_visualizations(self, analysis_results: Dict[str, Any], save_dir: str = "./polytope_plots") -> Dict[str, plt.Figure]:
        """
        Create visualizations of the analysis results.
        
        Args:
            analysis_results: Results from compare_across_checkpoints
            save_dir: Directory to save plots
            
        Returns:
            Dictionary of matplotlib figures
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        
        figures = {}
        
        # Plot 1: Volume comparison across checkpoints
        if analysis_results['evolution_metrics']:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            evolution_df = pd.DataFrame(analysis_results['evolution_metrics'])
            
            # Volume ratio evolution
            ax1.plot(evolution_df['checkpoint'], evolution_df['volume_ratio'], 'o-', linewidth=2, markersize=8)
            ax1.set_title('Volume Ratio (High/Low Frequency)', fontweight='bold')
            ax1.set_xlabel('Checkpoint')
            ax1.set_ylabel('Volume Ratio')
            ax1.grid(True, alpha=0.3)
            ax1.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Equal volumes')
            ax1.legend()
            
            # Sparsity difference evolution
            ax2.plot(evolution_df['checkpoint'], evolution_df['sparsity_difference'], 's-', linewidth=2, markersize=8, color='orange')
            ax2.set_title('Sparsity Difference (Low - High)', fontweight='bold')
            ax2.set_xlabel('Checkpoint')
            ax2.set_ylabel('Sparsity Difference')
            ax2.grid(True, alpha=0.3)
            ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Equal sparsity')
            ax2.legend()
            
            # Norm ratio evolution
            ax3.plot(evolution_df['checkpoint'], evolution_df['norm_ratio'], '^-', linewidth=2, markersize=8, color='green')
            ax3.set_title('Norm Ratio (High/Low Frequency)', fontweight='bold')
            ax3.set_xlabel('Checkpoint')
            ax3.set_ylabel('Norm Ratio')
            ax3.grid(True, alpha=0.3)
            ax3.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Equal norms')
            ax3.legend()
            
            # Summary metrics heatmap
            checkpoint_names = [str(cp) for cp in evolution_df['checkpoint']]
            metrics_matrix = np.array([
                evolution_df['volume_ratio'].values,
                evolution_df['sparsity_difference'].values,
                evolution_df['norm_ratio'].values
            ])
            
            im = ax4.imshow(metrics_matrix, aspect='auto', cmap='viridis')
            ax4.set_xticks(range(len(checkpoint_names)))
            ax4.set_xticklabels(checkpoint_names)
            ax4.set_yticks(range(3))
            ax4.set_yticklabels(['Volume Ratio', 'Sparsity Diff', 'Norm Ratio'])
            ax4.set_title('Metrics Evolution Heatmap', fontweight='bold')
            plt.colorbar(im, ax=ax4)
            
            plt.tight_layout()
            figures['evolution_summary'] = fig
            fig.savefig(save_dir / 'evolution_summary.png', dpi=300, bbox_inches='tight')
        
        # Plot 2: Detailed checkpoint comparison
        checkpoint_data = []
        for checkpoint, data in analysis_results['checkpoint_analysis'].items():
            for group in ['high', 'low']:
                if group in data and 'polytope_metrics' in data[group]:
                    metrics = data[group]['polytope_metrics']
                    stats = data[group]['activation_stats']
                    
                    checkpoint_data.append({
                        'checkpoint': checkpoint,
                        'frequency_group': group,
                        'volume': metrics['volume'],
                        'surface_area': metrics['surface_area'],
                        'effective_dimension': metrics['effective_dimension'],
                        'mean_sparsity': stats['mean_sparsity'],
                        'mean_norm': stats['mean_norm']
                    })
        
        if checkpoint_data:
            df = pd.DataFrame(checkpoint_data)
            
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            axes = axes.flatten()
            
            metrics = ['volume', 'surface_area', 'effective_dimension', 'mean_sparsity', 'mean_norm']
            
            for i, metric in enumerate(metrics):
                if i < len(axes):
                    for group in ['high', 'low']:
                        group_data = df[df['frequency_group'] == group]
                        if not group_data.empty:
                            axes[i].plot(group_data['checkpoint'], group_data[metric], 
                                       'o-', label=f'{group.title()} Frequency', linewidth=2, markersize=8)
                    
                    axes[i].set_title(f'{metric.replace("_", " ").title()}', fontweight='bold')
                    axes[i].set_xlabel('Checkpoint')
                    axes[i].set_ylabel(metric.replace("_", " ").title())
                    axes[i].legend()
                    axes[i].grid(True, alpha=0.3)
            
            # Remove empty subplot
            if len(metrics) < len(axes):
                fig.delaxes(axes[-1])
            
            plt.tight_layout()
            figures['detailed_comparison'] = fig
            fig.savefig(save_dir / 'detailed_comparison.png', dpi=300, bbox_inches='tight')
        
        logger.info(f"Created {len(figures)} visualization plots in {save_dir}")
        return figures
    
    def analyze_records(self, records: List[Dict]) -> Dict[str, Any]:
        """
        Main analysis function - analyze activation records for frequency differences.
        
        Args:
            records: List of activation records with keys:
                    - activation_vector: np.ndarray
                    - frequency: float 
                    - layer: int (optional)
                    - checkpoint: str (optional)
                    - ngram: str (optional)
                    - binary_pattern: np.ndarray (optional)
                    
        Returns:
            Complete analysis results
        """
        logger.info(f"Starting analysis of {len(records)} activation records")
        
        # Validate records
        validated_records = self.validate_records(records)
        
        if len(validated_records) < 4:
            raise ValueError("Need at least 4 valid records for analysis")
        
        # Run comparison across checkpoints
        analysis_results = self.compare_across_checkpoints(validated_records)
        
        # Create visualizations
        figures = self.create_visualizations(analysis_results)
        
        # Generate summary
        summary = self._generate_summary(analysis_results, validated_records)
        
        return {
            'analysis_results': analysis_results,
            'figures': figures,
            'summary': summary,
            'n_records_processed': len(validated_records),
            'n_records_input': len(records)
        }
    
    def _generate_summary(self, analysis_results: Dict[str, Any], records: List[Dict]) -> Dict[str, Any]:
        """Generate analysis summary."""
        # Basic statistics
        frequencies = [r['frequency'] for r in records]
        sparsities = [r['sparsity'] for r in records]
        norms = [r['activation_norm'] for r in records]
        
        # Frequency-based statistics
        median_freq = np.median(frequencies)
        high_freq_records = [r for r in records if r['frequency'] >= median_freq]
        low_freq_records = [r for r in records if r['frequency'] < median_freq]
        
        high_sparsity = np.mean([r['sparsity'] for r in high_freq_records])
        low_sparsity = np.mean([r['sparsity'] for r in low_freq_records])
        
        high_norm = np.mean([r['activation_norm'] for r in high_freq_records])
        low_norm = np.mean([r['activation_norm'] for r in low_freq_records])
        
        return {
            'total_records': len(records),
            'n_checkpoints': len(set(r['checkpoint'] for r in records)),
            'n_layers': len(set(r['layer'] for r in records)),
            'frequency_stats': {
                'min': np.min(frequencies),
                'max': np.max(frequencies),
                'median': median_freq,
                'mean': np.mean(frequencies)
            },
            'high_frequency_group': {
                'n_records': len(high_freq_records),
                'mean_sparsity': high_sparsity,
                'mean_norm': high_norm
            },
            'low_frequency_group': {
                'n_records': len(low_freq_records),
                'mean_sparsity': low_sparsity,
                'mean_norm': low_norm
            },
            'key_findings': {
                'sparsity_difference': low_sparsity - high_sparsity,
                'norm_ratio': high_norm / low_norm if low_norm > 0 else 1.0,
                'frequency_range': np.max(frequencies) - np.min(frequencies)
            }
        }


# Example usage function
def example_usage():
    """
    Example of how to use the PolytopeAnalyzer with activation records.
    """
    print("Polytope Analyzer - Example Usage")
    print("=" * 50)
    
    # Example activation records structure
    example_records = [
        {
            'activation_vector': np.random.randn(512),  # 512-dim activation
            'frequency': 150.0,  # High frequency
            'layer': 6,
            'checkpoint': 'step_1000',
            'ngram': 'the cat',
            'binary_pattern': np.random.rand(512) > 0.7
        },
        {
            'activation_vector': np.random.randn(512),
            'frequency': 5.0,  # Low frequency  
            'layer': 6,
            'checkpoint': 'step_1000',
            'ngram': 'obscure word',
            'binary_pattern': np.random.rand(512) > 0.8
        }
        # ... more records
    ]
    
    # Initialize analyzer
    analyzer = PolytopeAnalyzer()
    
    # Run analysis
    # results = analyzer.analyze_records(example_records)
    
    print("Example record structure:")
    for key, value in example_records[0].items():
        if isinstance(value, np.ndarray):
            print(f"  {key}: np.ndarray shape {value.shape}")
        else:
            print(f"  {key}: {value}")
    
    print("\nTo run analysis:")
    print("  analyzer = PolytopeAnalyzer()")
    print("  results = analyzer.analyze_records(your_activation_records)")
    print("  # Results include analysis_results, figures, and summary")


if __name__ == "__main__":
    example_usage()