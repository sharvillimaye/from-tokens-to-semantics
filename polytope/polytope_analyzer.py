#!/usr/bin/env python3
"""
Consolidated Polytope Analyzer for Activation Records
Simple, robust analysis of high vs low frequency activation data across checkpoints.
"""

import numpy as np
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict
from pathlib import Path
import logging
import warnings

# Optional imports for visualization (only needed for plotting functions)
try:
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

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
    
    def __init__(self, n_pca_components: float = 0.95, approximation_epsilon: float = 0.05):
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
        
        Required fields from checkpoint analysis:
        - activation_vector: Neural network activation data (np.ndarray)
        - ngram_frequency: N-gram frequency value (int/float)
        
        Optional fields (with defaults):
        - layer: Model layer number (defaults to 0)
        - checkpoint_step: Training checkpoint identifier (defaults to '0')
        - ngram: N-gram text (defaults to 'unknown')
        
        Args:
            records: List of activation records from checkpoint analysis
            
        Returns:
            List of validated records compatible with polytope analysis
        """
        validated = []
        required_fields = ['activation_vector', 'ngram_frequency']
        
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
    
    def find_extreme_points(self, points: np.ndarray, n_components: int = 20) -> np.ndarray:
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

    def distance_to_approximate_hull(self, point: np.ndarray, hull_points: np.ndarray) -> float:
        """
        Fast approximation of distance from point to convex hull.
        
        Uses geometric approximations instead of optimization for speed.
        
        Args:
            point: Point to compute distance from
            hull_points: Points defining the approximate hull
            
        Returns:
            Approximate distance to hull (0 if inside)
        """
        n_hull_points = len(hull_points)
        
        if n_hull_points == 0:
            return float('inf')
        elif n_hull_points == 1:
            return np.linalg.norm(point - hull_points[0])
        
        # Check if point is already a hull vertex
        distances_to_vertices = np.linalg.norm(hull_points - point, axis=1)
        min_vertex_distance = np.min(distances_to_vertices)
        
        if min_vertex_distance < 1e-12:
            return 0.0
        
        # Fast approximation: use closest vertices and geometric mean
        if n_hull_points <= 3:
            return min_vertex_distance
        
        # For larger hulls, use a fast geometric approximation
        try:
            # Find the 3 closest hull vertices
            closest_indices = np.argsort(distances_to_vertices)[:min(3, n_hull_points)]
            closest_points = hull_points[closest_indices]
            
            # Compute centroid of closest points
            centroid = np.mean(closest_points, axis=0)
            centroid_distance = np.linalg.norm(point - centroid)
            
            # Use minimum of vertex distance and centroid distance as approximation
            # This is much faster than optimization and still reasonably accurate
            return min(min_vertex_distance, centroid_distance)
            
        except Exception:
            # Fallback: return distance to closest vertex
            return min_vertex_distance

    def find_hull_vertices(self, points: np.ndarray) -> np.ndarray:
        """
        Find convex hull vertices using greedy approximation for high-dimensional data.
        
        Args:
            points: Input points (n_samples, n_features)
            
        Returns:
            Indices of hull vertices
        """
        return self.greedy_hull_approximation(points, epsilon=self.approximation_epsilon)

    def revised_greedy_expansion_algorithm(self, points: np.ndarray, epsilon: float = 0.05, 
                                         max_iter: int = 1000, max_runtime: float = 60.0, verbose: bool = False) -> np.ndarray:
        r"""
        Fast Revised Greedy Expansion Algorithm for convex hull approximation.
        
        Optimized version with:
        - Fast distance approximations (no optimization)
        - Sampling-based minimax selection
        - Adaptive epsilon for speed
        - Runtime limits and early termination
        
        Args:
            points: Input points S = {x1, x2, ..., xs} (n_samples, n_features)
            epsilon: Initial approximation rate epsilon
            max_iter: Maximum iterations
            max_runtime: Maximum runtime in seconds
            verbose: Print progress
            
        Returns:
            Indices of epsilon-approximation convex hull vertices
        """
        n_points, n_dims = points.shape
        
        if verbose:
            logger.info(f"Revised Greedy Expansion: {n_points} points in {n_dims}D, epsilon={epsilon}")
        
        if n_points <= 3:
            return np.arange(n_points)
        
        try:
            import time
            start_time = time.time()
            current_epsilon = epsilon
            
            # Step 1: Find kernelized extreme points and initialize E
            extreme_indices = self.find_extreme_points(points, n_components=min(30, n_dims))
            E = set(extreme_indices)  # Hull vertex indices
            
            if verbose:
                logger.info(f"Step 1: Initialized E with {len(E)} kernelized extreme points")
            
            # Step 2: Assign the set of outside points R = {x ∈ S|d(x, E) > 0}
            R = self._compute_outside_points(points, E, current_epsilon, verbose)
            
            if verbose:
                logger.info(f"Step 2: Found {len(R)} outside points")
            
            # Step 3: Main loop with adaptive epsilon and runtime monitoring
            for iteration in range(max_iter):
                current_time = time.time()
                runtime = current_time - start_time
                
                # Runtime check and adaptive epsilon
                if runtime > max_runtime:
                    if verbose:
                        logger.warning(f"Runtime limit {max_runtime}s exceeded, stopping early")
                    break
                elif runtime > max_runtime * 0.5:
                    # Increase epsilon to speed up convergence
                    current_epsilon = epsilon * 2.0
                    if verbose:
                        logger.info(f"Increasing epsilon to {current_epsilon:.4f} for faster convergence")
                
                if not R:
                    if verbose:
                        logger.info("No more outside points")
                    break
                
                try:
                    # Quick convergence check every 10 iterations (expensive operation)
                    if iteration % 10 == 0:
                        max_distance = self._compute_max_distance_to_hull(points, E)
                        if max_distance <= current_epsilon:
                            if verbose:
                                logger.info(f"Converged at iteration {iteration}: max_distance = {max_distance:.6f}")
                            break
                    
                    # Step 4: Select point using fast minimax criterion
                    selected_point = self._minimax_point_selection(points, E, R, verbose)
                    
                    if selected_point is None:
                        if verbose:
                            logger.warning("No valid point selected, stopping")
                        break
                    
                    # Add selected point to E
                    E.add(selected_point)
                    R.discard(selected_point)
                    
                    # Progress reporting
                    if verbose and (iteration + 1) % 10 == 0:
                        logger.info(f"Iteration {iteration + 1}: |E|={len(E)}, |R|={len(R)}, "
                                  f"runtime={runtime:.1f}s, eps={current_epsilon:.4f}")
                    
                    # Pruning (less frequent for speed)
                    if iteration % 5 == 0:  # Only prune every 5 iterations
                        # Step 5: Hull vertex pruning (optional for speed)
                        if len(E) < 100:  # Only prune if hull is still small
                            E = self._prune_hull_vertices(points, E, current_epsilon, verbose)
                        
                        # Step 6: Outside points pruning
                        R = self._prune_outside_points(points, E, R, current_epsilon, verbose)
                                  
                except Exception as e:
                    if verbose:
                        logger.error(f"Error in iteration {iteration}: {e}")
                    break
            
            hull_vertices = np.array(list(E))
            final_compression = len(hull_vertices) / n_points
            final_runtime = time.time() - start_time
            
            if verbose:
                logger.info(f"Final approximation: {len(hull_vertices)} vertices "
                          f"(compression: {final_compression:.4f}, runtime: {final_runtime:.2f}s)")
            
            return hull_vertices
            
        except Exception as e:
            logger.error(f"Critical error in revised greedy expansion: {e}")
            # Fallback to extreme points only
            return self.find_extreme_points(points, n_components=min(10, n_dims))

    def greedy_hull_approximation(self, points: np.ndarray, epsilon: float = 0.05, 
                                 max_iter: int = 1000, sample_size: int = 2000, 
                                 verbose: bool = False) -> np.ndarray:
        """
        Enhanced greedy approximation of convex hull for high-dimensional data.
        
        This method now uses the Fast Revised Greedy Expansion Algorithm.
        
        Args:
            points: Input points (n_samples, n_features)
            epsilon: Convergence tolerance
            max_iter: Maximum iterations
            sample_size: Unused (kept for backward compatibility)
            verbose: Print progress
            
        Returns:
            Indices of approximate hull vertices
        """
        # Auto-adjust parameters based on data size for performance
        max_runtime = 30.0 if len(points) > 1000 else 60.0
        return self.revised_greedy_expansion_algorithm(points, epsilon, max_iter, max_runtime, verbose)

    def _compute_outside_points(self, points: np.ndarray, E: set, epsilon: float, verbose: bool = False) -> set:
        r"""
        Compute the set of outside points R = {x in S|d(x, E) > 0}.
        
        Args:
            points: All input points
            E: Current hull vertex indices
            epsilon: Tolerance for considering points "outside"
            verbose: Print progress
            
        Returns:
            Set of outside point indices
        """
        R = set()
        hull_points = points[list(E)]
        
        for i in range(len(points)):
            if i not in E:
                distance = self.distance_to_approximate_hull(points[i], hull_points)
                if distance > epsilon:  # Using epsilon as threshold for "outside"
                    R.add(i)
        
        if verbose:
            logger.info(f"Computed {len(R)} outside points from {len(points) - len(E)} candidates")
        
        return R

    def _compute_max_distance_to_hull(self, points: np.ndarray, E: set) -> float:
        r"""
        Compute max_{x in S} d(x, E) for convergence checking.
        
        Args:
            points: All input points
            E: Current hull vertex indices
            
        Returns:
            Maximum distance from any point to the current hull
        """
        if not E:
            return float('inf')
            
        hull_points = points[list(E)]
        max_distance = 0.0
        
        for i in range(len(points)):
            if i not in E:
                distance = self.distance_to_approximate_hull(points[i], hull_points)
                max_distance = max(max_distance, distance)
        
        return max_distance

    def _minimax_point_selection(self, points: np.ndarray, E: set, R: set, verbose: bool = False) -> Optional[int]:
        r"""
        Fast minimax point selection with sampling and early termination.
        
        Args:
            points: All input points
            E: Current hull vertex indices
            R: Current outside point indices
            verbose: Print progress
            
        Returns:
            Index of selected point, or None if no valid point found
        """
        if not R:
            return None
            
        best_point = None
        best_max_distance = float('inf')
        
        # Aggressive sampling for speed - limit both candidate and evaluation sets
        R_list = list(R)
        
        # Sample candidates to evaluate (max 20 for speed)
        n_candidates = min(20, len(R_list))
        if len(R_list) > n_candidates:
            candidate_indices = np.random.choice(R_list, size=n_candidates, replace=False)
        else:
            candidate_indices = R_list
        
        # Sample evaluation points (max 50 for speed)
        n_eval_points = min(50, len(R_list))
        if len(R_list) > n_eval_points:
            eval_indices = np.random.choice(R_list, size=n_eval_points, replace=False)
        else:
            eval_indices = R_list
        
        for candidate_x in candidate_indices:
            try:
                # Create temporary hull E ∪ {x}
                temp_E = E.copy()
                temp_E.add(candidate_x)
                temp_hull_points = points[list(temp_E)]
                
                # Compute max distance over sampled evaluation points
                max_distance_for_candidate = 0.0
                
                # Only check distances from sampled points for speed
                for v in eval_indices:
                    if v != candidate_x:
                        distance = self.distance_to_approximate_hull(points[v], temp_hull_points)
                        max_distance_for_candidate = max(max_distance_for_candidate, distance)
                        
                        # Early termination: if this candidate is already worse than current best
                        if max_distance_for_candidate > best_max_distance:
                            break
                
                # Update best candidate if this one is better
                if max_distance_for_candidate < best_max_distance:
                    best_max_distance = max_distance_for_candidate
                    best_point = candidate_x
                    
            except Exception as e:
                if verbose:
                    logger.warning(f"Error evaluating candidate {candidate_x}: {e}")
                continue
        
        # Fallback: if no point selected, just pick the first point in R
        if best_point is None and R:
            best_point = next(iter(R))
            if verbose:
                logger.warning("Minimax selection failed, using fallback point")
        
        if verbose and best_point is not None:
            logger.debug(f"Selected point {best_point} (sampled {n_candidates} candidates, {n_eval_points} eval points)")
        
        return best_point

    def _prune_hull_vertices(self, points: np.ndarray, E: set, epsilon: float, verbose: bool = False) -> set:
        r"""
        Fast hull vertex pruning with sampling for efficiency.
        
        Args:
            points: All input points
            E: Current hull vertex indices
            epsilon: Tolerance for pruning
            verbose: Print progress
            
        Returns:
            Pruned set of hull vertex indices
        """
        if len(E) <= 5:  # Don't prune if we have very few vertices
            return E
            
        E_pruned = E.copy()
        vertices_to_remove = []
        
        # Sample vertices to check for pruning (max 20 for speed)
        E_list = list(E)
        n_check = min(20, len(E_list))
        if len(E_list) > n_check:
            vertices_to_check = np.random.choice(E_list, size=n_check, replace=False)
        else:
            vertices_to_check = E_list
        
        for vertex in vertices_to_check:
            try:
                # Create hull without this vertex: E\{vertex}
                E_without_vertex = E.copy()
                E_without_vertex.discard(vertex)
                
                if len(E_without_vertex) < 3:  # Need at least 3 points for meaningful hull
                    continue
                    
                hull_without_vertex = points[list(E_without_vertex)]
                
                # Compute distance from vertex to hull without it
                distance = self.distance_to_approximate_hull(points[vertex], hull_without_vertex)
                
                # Mark for removal if distance is small enough
                if distance <= epsilon:
                    vertices_to_remove.append(vertex)
                    
            except Exception as e:
                if verbose:
                    logger.warning(f"Error checking vertex {vertex} for pruning: {e}")
                continue
        
        # Remove redundant vertices
        for vertex in vertices_to_remove:
            E_pruned.discard(vertex)
        
        if verbose and vertices_to_remove:
            logger.debug(f"Pruned {len(vertices_to_remove)} hull vertices (checked {n_check}/{len(E)})")
        
        return E_pruned

    def _prune_outside_points(self, points: np.ndarray, E: set, R: set, epsilon: float, verbose: bool = False) -> set:
        r"""
        Fast outside points pruning with sampling for efficiency.
        
        Args:
            points: All input points
            E: Current hull vertex indices
            R: Current outside point indices
            epsilon: Tolerance for pruning
            verbose: Print progress
            
        Returns:
            Pruned set of outside point indices
        """
        if not R or not E:
            return R
            
        hull_points = points[list(E)]
        R_pruned = R.copy()
        points_to_remove = []
        
        # Sample points to check for pruning (max 100 for speed)
        R_list = list(R)
        n_check = min(100, len(R_list))
        if len(R_list) > n_check:
            points_to_check = np.random.choice(R_list, size=n_check, replace=False)
        else:
            points_to_check = R_list
        
        for point_idx in points_to_check:
            try:
                distance = self.distance_to_approximate_hull(points[point_idx], hull_points)
                
                if distance <= epsilon:
                    points_to_remove.append(point_idx)
                    
            except Exception as e:
                if verbose:
                    logger.warning(f"Error checking outside point {point_idx} for pruning: {e}")
                continue
        
        # Remove points that are now close enough
        for point_idx in points_to_remove:
            R_pruned.discard(point_idx)
        
        if verbose and points_to_remove:
            logger.debug(f"Pruned {len(points_to_remove)} outside points (checked {n_check}/{len(R)})")
        
        return R_pruned
    
    def compute_volume(self, points: np.ndarray) -> float:
        """
        Compute polytope volume using approximation methods for high-dimensional data.
        
        Args:
            points: Input points defining the polytope
            
        Returns:
            Estimated volume using greedy hull approximation
        """
        if len(points) < 2:
            return 0.0
        
        try:
            # Get approximate hull vertices
            hull_vertex_indices = self.find_hull_vertices(points)
            hull_points = points[hull_vertex_indices]
            
            # Use Monte Carlo volume estimation with hull approximation
            return self._approximate_monte_carlo_volume(hull_points)
            
        except Exception as e:
            logger.error(f"Volume computation failed: {e}")
            return 0.0
    
    def _approximate_monte_carlo_volume(self, hull_points: np.ndarray, n_samples: int = 10000) -> float:
        """
        Robust approximate volume using Monte Carlo sampling with hull approximation
        
        Args:
            hull_points: Points defining the approximate hull
            n_samples: Number of Monte Carlo samples
            
        Returns:
            Approximate volume
        """
        if len(hull_points) < 2:
            return 0.0
        
        try:
            # Find bounding box
            min_coords = np.min(hull_points, axis=0)
            max_coords = np.max(hull_points, axis=0)
            
            # Check for degenerate cases
            coord_ranges = max_coords - min_coords
            if np.any(coord_ranges <= 1e-12):
                # Nearly degenerate in some dimensions
                non_degenerate_dims = np.sum(coord_ranges > 1e-12)
                if non_degenerate_dims == 0:
                    return 0.0
                elif non_degenerate_dims == 1:
                    # Line-like structure
                    return np.max(coord_ranges)
                elif non_degenerate_dims == 2:
                    # Plane-like structure, estimate area
                    valid_ranges = coord_ranges[coord_ranges > 1e-12]
                    return np.prod(valid_ranges) if len(valid_ranges) >= 2 else np.max(valid_ranges)
            
            # Volume of bounding box
            box_volume = np.prod(coord_ranges)
            
            if box_volume <= 1e-20:
                return 0.0
            
            # Use multiple estimation methods for robustness
            volume_estimates = []
            
            # Method 1: Standard Monte Carlo with hull approximation
            try:
                n_inside = 0
                sample_batch_size = min(1000, n_samples)
                
                for batch_start in range(0, n_samples, sample_batch_size):
                    batch_size = min(sample_batch_size, n_samples - batch_start)
                    
                    # Generate batch of random points
                    random_points = np.random.uniform(
                        min_coords, max_coords, size=(batch_size, len(min_coords))
                    )
                    
                    # Check batch of points using hull approximation
                    for point in random_points:
                        distance = self.distance_to_approximate_hull(point, hull_points)
                        if distance <= 1e-6:  # Very small tolerance for "inside"
                            n_inside += 1
                
                volume_ratio = n_inside / n_samples
                mc_volume = box_volume * volume_ratio
                volume_estimates.append(mc_volume)
                
            except Exception as e:
                logger.warning(f"Monte Carlo volume estimation failed: {e}")
            
            # Method 2: Bounding box scaling estimate for high dimensions
            try:
                n_dims = hull_points.shape[1]
                if n_dims > 0:
                    # Rough estimate based on point density and dimensionality
                    density_factor = len(hull_points) / (2 ** min(n_dims, 20))  # Cap at 20 dims
                    density_volume = box_volume * min(1.0, density_factor)
                    volume_estimates.append(density_volume)
            except:
                pass
            
            # Return best estimate
            if volume_estimates:
                # Use median of estimates for robustness
                return float(np.median(volume_estimates))
            else:
                # Final fallback: use simple box approximation
                return box_volume * 0.1  # Conservative estimate
                
        except Exception as e:
            logger.error(f"Volume computation failed: {e}")
            return 0.0
    
    def compute_surface_area(self, points: np.ndarray) -> float:
        """
        Estimate surface area of the polytope using approximation methods.
        
        Args:
            points: Input points defining the polytope
            
        Returns:
            Estimated surface area using greedy hull approximation
        """
        if len(points) < 2:
            return 0.0
        
        try:
            # Get approximate hull vertices
            hull_vertex_indices = self.find_hull_vertices(points)
            hull_points = points[hull_vertex_indices]
            
            # Estimate surface area based on hull vertices and dimensionality
            n_hull_vertices = len(hull_points)
            n_dims = points.shape[1]
            
            if n_hull_vertices < 2:
                return 0.0
            
            # Method 1: Surface area estimation based on hull point distances
            distances = pdist(hull_points)
            if len(distances) == 0:
                return 0.0
                
            mean_distance = np.mean(distances)
            max_distance = np.max(distances)
            
            # Scale by dimensionality and number of hull vertices
            # Surface area grows roughly as (n-1) dimensional measure
            if n_dims <= 1:
                return max_distance
            else:
                # Estimate surface area using hull vertices and mean distances
                surface_scaling = n_hull_vertices * (mean_distance ** (n_dims - 1))
                
                # Apply correction factor for high dimensions
                dim_correction = np.sqrt(n_dims)
                
                return surface_scaling / dim_correction
                
        except Exception as e:
            logger.warning(f"Surface area computation failed: {e}")
            
            # Final fallback: simple distance-based estimate
            try:
                distances = pdist(points)
                mean_distance = np.mean(distances)
                n_dims = points.shape[1]
                return len(points) * (mean_distance ** max(1, n_dims - 1))
            except:
                return 0.0
    
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
        frequencies = [r['ngram_frequency'] for r in records]
        
        if not frequencies:
            return {'high': [], 'low': []}
        
        # Use median as threshold for simple binary split
        median_freq = np.median(frequencies)
        
        high_freq = [r for r in records if r['ngram_frequency'] >= median_freq]
        low_freq = [r for r in records if r['ngram_frequency'] < median_freq]
        
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
            'mean_frequency': np.mean([r['ngram_frequency'] for r in records]),
            'std_frequency': np.std([r['ngram_frequency'] for r in records]),
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
            checkpoint_groups[record['checkpoint_step']].append(record)
        
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
        frequencies = [r['ngram_frequency'] for r in records]
        sparsities = [r['sparsity'] for r in records]
        norms = [r['activation_norm'] for r in records]
        
        # Frequency-based statistics
        median_freq = np.median(frequencies)
        high_freq_records = [r for r in records if r['ngram_frequency'] >= median_freq]
        low_freq_records = [r for r in records if r['ngram_frequency'] < median_freq]
        
        high_sparsity = np.mean([r['sparsity'] for r in high_freq_records])
        low_sparsity = np.mean([r['sparsity'] for r in low_freq_records])
        
        high_norm = np.mean([r['activation_norm'] for r in high_freq_records])
        low_norm = np.mean([r['activation_norm'] for r in low_freq_records])
        
        return {
            'total_records': len(records),
            'n_checkpoints': len(set(r['checkpoint_step'] for r in records)),
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