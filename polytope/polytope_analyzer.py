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

# Required imports for advanced polytope analysis
from hdbscan import HDBSCAN
from scipy.spatial import ConvexHull, Delaunay
from scipy.spatial.distance import cdist
from scipy.optimize import linprog

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SplineCodeGenerator:
    """
    Generates spline codes using CETT (Cumulative Error Tail Threshold) methodology.
    
    Spline codes are binary vectors indicating which neurons activate above threshold:
    - 1 where polytope causes neuron to activate above CETT threshold
    - 0 otherwise
    """
    
    def __init__(self, cett_target: float = 0.01):
        """
        Initialize spline code generator.
        
        Args:
            cett_target: Target CETT value (0.01 = 1% error tolerance)
        """
        self.cett_target = cett_target
    
    def compute_cett_threshold(self, activation_vector: np.ndarray) -> float:
        """
        Compute CETT-based threshold for activation vector.
        
        Uses binary search to find optimal threshold achieving target CETT.
        CETT = ||tail_activations|| / ||total_activations||
        
        Args:
            activation_vector: Neural activation vector
            
        Returns:
            Optimal threshold value
        """
        magnitudes = np.abs(activation_vector)
        total_norm = np.linalg.norm(activation_vector)
        
        if total_norm == 0:
            return 0.0
        
        # Binary search for optimal threshold
        sorted_magnitudes = np.sort(magnitudes)
        left, right = 0, len(sorted_magnitudes) - 1
        best_threshold = 0.0
        
        while left <= right:
            mid = (left + right) // 2
            threshold = sorted_magnitudes[mid]
            
            # Compute CETT for this threshold
            below_threshold_mask = magnitudes < threshold
            tail_norm = np.linalg.norm(activation_vector * below_threshold_mask)
            current_cett = tail_norm / total_norm
            
            if current_cett <= self.cett_target:
                best_threshold = threshold
                left = mid + 1
            else:
                right = mid - 1
        
        return best_threshold
    
    def generate_spline_code(self, activation_vector: np.ndarray) -> np.ndarray:
        """
        Generate spline code from activation vector using CETT threshold.
        
        Args:
            activation_vector: Neural activation vector
            
        Returns:
            Binary spline code with 1 where neuron activates above CETT threshold
        """
        threshold = self.compute_cett_threshold(activation_vector)
        return (np.abs(activation_vector) > threshold).astype(int)
    
    def batch_generate_spline_codes(self, preactivations_batch: np.ndarray) -> np.ndarray:
        """
        Generate spline codes for batch of inputs.
        
        Args:
            preactivations_batch: Batch of preactivations (N, M)
            
        Returns:
            Batch of spline codes (N, M)
        """
        return (preactivations_batch > self.activation_threshold).astype(int)
    
    def compute_spline_code_distance(self, code1: np.ndarray, code2: np.ndarray) -> int:
        """
        Compute Hamming distance between two spline codes.
        
        Args:
            code1, code2: Binary spline codes
            
        Returns:
            Hamming distance (number of differing bits)
        """
        return np.sum(code1 != code2)
    
    def find_unique_spline_codes(self, spline_codes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find unique spline codes and their indices.
        
        Args:
            spline_codes: Batch of spline codes (N, M)
            
        Returns:
            (unique_codes, unique_indices)
        """
        unique_codes, unique_indices = np.unique(spline_codes, axis=0, return_index=True)
        return unique_codes, unique_indices


class HyperplaneIntersection:
    """
    Implements hyperplane intersection algorithms for polytope boundary detection.
    """
    
    def __init__(self, epsilon: float = 1e-12):
        self.epsilon = epsilon
    
    def hyperbox_hyperplane_intersection(self, vertices: np.ndarray, hyperplane_normal: np.ndarray, 
                                       hyperplane_bias: float) -> np.ndarray:
        """
        Compute intersection of hyperbox with hyperplane using border node method.
        
        Args:
            vertices: Vertices of hyperbox (N, D)
            hyperplane_normal: Normal vector of hyperplane (D,)
            hyperplane_bias: Bias term b in hyperplane equation N·x = b
            
        Returns:
            Intersection vertices (K, D)
        """
        if len(vertices) == 0:
            return np.array([])
        
        # Compute signed distances to hyperplane
        distances = np.dot(vertices, hyperplane_normal) - hyperplane_bias
        
        # Find vertices on opposite sides
        positive_mask = distances > self.epsilon
        negative_mask = distances < -self.epsilon
        on_plane_mask = np.abs(distances) <= self.epsilon
        
        intersection_vertices = []
        
        # Add vertices already on the hyperplane
        intersection_vertices.extend(vertices[on_plane_mask])
        
        # Find intersection points along edges
        for i in range(len(vertices)):
            for j in range(i + 1, len(vertices)):
                # Check if edge crosses hyperplane
                if (positive_mask[i] and negative_mask[j]) or (negative_mask[i] and positive_mask[j]):
                    # Compute intersection point
                    t = distances[i] / (distances[i] - distances[j])
                    intersection_point = vertices[i] + t * (vertices[j] - vertices[i])
                    intersection_vertices.append(intersection_point)
        
        if len(intersection_vertices) == 0:
            return np.array([])
        
        return np.array(intersection_vertices)
    
    def split_polytope_by_hyperplane(self, vertices: np.ndarray, hyperplane_normal: np.ndarray,
                                   hyperplane_bias: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split polytope into two parts using hyperplane (SplitPlane algorithm).
        
        Args:
            vertices: Polytope vertices (N, D)
            hyperplane_normal: Normal vector (D,)
            hyperplane_bias: Bias term
            
        Returns:
            (vertices_positive_side, vertices_negative_side)
        """
        distances = np.dot(vertices, hyperplane_normal) - hyperplane_bias
        
        # Find intersection points
        intersection_points = []
        
        # Add existing vertices on hyperplane
        on_plane_mask = np.abs(distances) <= self.epsilon
        intersection_points.extend(vertices[on_plane_mask])
        
        # Find edge intersections
        for i in range(len(vertices)):
            for j in range(i + 1, len(vertices)):
                if distances[i] * distances[j] < 0:  # Edge crosses plane
                    t = distances[i] / (distances[i] - distances[j])
                    intersection_point = vertices[i] + t * (vertices[j] - vertices[i])
                    intersection_points.append(intersection_point)
        
        intersection_array = np.array(intersection_points) if intersection_points else np.array([])
        
        # Split vertices by side
        positive_vertices = vertices[distances > self.epsilon]
        negative_vertices = vertices[distances < -self.epsilon]
        
        # Combine with intersection points
        if len(intersection_array) > 0:
            positive_side = np.vstack([positive_vertices, intersection_array]) if len(positive_vertices) > 0 else intersection_array
            negative_side = np.vstack([negative_vertices, intersection_array]) if len(negative_vertices) > 0 else intersection_array
        else:
            positive_side = positive_vertices
            negative_side = negative_vertices
        
        return positive_side, negative_side


class VRepresentationManager:
    """
    Manages vertex representation (V-representation) of polytopes.
    """
    
    def __init__(self, epsilon: float = 1e-12):
        self.epsilon = epsilon
    
    def compute_convex_hull(self, points: np.ndarray) -> np.ndarray:
        """
        Compute convex hull vertices using scipy.
        
        Args:
            points: Input points (N, D)
            
        Returns:
            Hull vertices (K, D)
        """
        if len(points) < 3:
            return points
        
        hull = ConvexHull(points)
        return points[hull.vertices]
    

    
    def get_vertices_counter_clockwise(self, vertices: np.ndarray) -> np.ndarray:
        """
        Order 2D vertices in counter-clockwise order.
        
        Args:
            vertices: 2D vertices (N, 2)
            
        Returns:
            Counter-clockwise ordered vertices
        """
        if vertices.shape[1] != 2:
            return vertices  # Only works for 2D
        
        # Compute centroid
        centroid = np.mean(vertices, axis=0)
        
        # Compute angles from centroid
        angles = np.arctan2(vertices[:, 1] - centroid[1], vertices[:, 0] - centroid[0])
        
        # Sort by angle
        sorted_indices = np.argsort(angles)
        return vertices[sorted_indices]
    
    def compute_edges(self, vertices: np.ndarray) -> List[Tuple[int, int]]:
        """
        Compute edges of 2D polytope.
        
        Args:
            vertices: 2D vertices in counter-clockwise order
            
        Returns:
            List of edge pairs (vertex indices)
        """
        if vertices.shape[1] != 2:
            logger.warning("Edge computation only implemented for 2D")
            return []
        
        n_vertices = len(vertices)
        edges = [(i, (i + 1) % n_vertices) for i in range(n_vertices)]
        return edges
    
    def compute_facets(self, vertices: np.ndarray) -> List[np.ndarray]:
        """
        Compute facets (faces) of polytope.
        
        Args:
            vertices: Polytope vertices
            
        Returns:
            List of facet vertex arrays
        """
        try:
            hull = ConvexHull(vertices)
            facets = []
            for simplex in hull.simplices:
                facets.append(vertices[simplex])
            return facets
        except Exception:
            return []


class SyReNNBoundaryDetector:
    """
    Implements SyReNN algorithms for exact polytope boundary detection.
    Based on ExtendPWL and SplitPlane algorithms.
    """
    
    def __init__(self, epsilon: float = 1e-12):
        self.epsilon = epsilon
        self.hyperplane_intersection = HyperplaneIntersection(epsilon)
        self.v_rep = VRepresentationManager(epsilon)
    
    def extend_pwl(self, input_polytopes: List[np.ndarray], 
                   hyperplanes: List[Tuple[np.ndarray, float]]) -> List[np.ndarray]:
        """
        ExtendPWL Algorithm: Extend piecewise linear function through layer.
        
        Args:
            input_polytopes: List of polytope vertex arrays from previous layer
            hyperplanes: List of (normal, bias) pairs defining activation boundaries
            
        Returns:
            Refined polytope partition as list of vertex arrays
        """
        # Initialize work queue with input polytopes
        work_queue = input_polytopes.copy()
        result_polytopes = []
        
        while work_queue:
            current_polytope = work_queue.pop(0)
            
            if len(current_polytope) == 0:
                continue
            
            # Check if any hyperplane splits this polytope
            split_occurred = False
            
            for hyperplane_normal, hyperplane_bias in hyperplanes:
                # Check if polytope vertices lie on opposite sides of hyperplane
                distances = np.dot(current_polytope, hyperplane_normal) - hyperplane_bias
                
                has_positive = np.any(distances > self.epsilon)
                has_negative = np.any(distances < -self.epsilon)
                
                if has_positive and has_negative:
                    # Split the polytope
                    positive_side, negative_side = self.hyperplane_intersection.split_polytope_by_hyperplane(
                        current_polytope, hyperplane_normal, hyperplane_bias
                    )
                    
                    # Add split polytopes back to work queue
                    if len(positive_side) > 0:
                        work_queue.append(positive_side)
                    if len(negative_side) > 0:
                        work_queue.append(negative_side)
                    
                    split_occurred = True
                    break
            
            # If no hyperplane splits this polytope, add to results
            if not split_occurred:
                result_polytopes.append(current_polytope)
        
        return result_polytopes
    
    def split_plane_2d(self, vertices: np.ndarray, hyperplane_normal: np.ndarray, 
                      hyperplane_bias: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        SplitPlane algorithm for 2D polytopes.
        
        Args:
            vertices: 2D polytope vertices (N, 2)
            hyperplane_normal: Normal vector (2,)
            hyperplane_bias: Bias term
            
        Returns:
            (vertices_A, vertices_B) for split polytopes
        """
        if vertices.shape[1] != 2:
            raise ValueError("SplitPlane2D only works for 2D vertices")
        
        # Order vertices counter-clockwise
        vertices_ccw = self.v_rep.get_vertices_counter_clockwise(vertices)
        distances = np.dot(vertices_ccw, hyperplane_normal) - hyperplane_bias
        
        # Find intersection points
        intersection_points = []
        n_vertices = len(vertices_ccw)
        
        for i in range(n_vertices):
            j = (i + 1) % n_vertices
            
            # Check if edge crosses hyperplane
            if distances[i] * distances[j] < 0:
                # Compute intersection
                t = distances[i] / (distances[i] - distances[j])
                intersection_point = vertices_ccw[i] + t * (vertices_ccw[j] - vertices_ccw[i])
                intersection_points.append(intersection_point)
        
        if len(intersection_points) < 2:
            # No proper split possible
            return vertices_ccw, np.array([])
        
        # Take first two intersection points
        p1, p2 = intersection_points[0], intersection_points[1]
        
        # Create split polytopes
        vertices_A = []
        vertices_B = []
        
        # Add intersection points to both
        vertices_A.extend([p1, p2])
        vertices_B.extend([p1, p2])
        
        # Distribute original vertices by side
        for i, vertex in enumerate(vertices_ccw):
            if distances[i] > self.epsilon:
                vertices_A.append(vertex)
            elif distances[i] < -self.epsilon:
                vertices_B.append(vertex)
            # Vertices on hyperplane are already included via intersections
        
        # Compute convex hulls
        hull_A = self.v_rep.compute_convex_hull(np.array(vertices_A)) if vertices_A else np.array([])
        hull_B = self.v_rep.compute_convex_hull(np.array(vertices_B)) if vertices_B else np.array([])
        
        return hull_A, hull_B
    
    def split_hyperplane_kd(self, vertices: np.ndarray, hyperplane_normal: np.ndarray,
                           hyperplane_bias: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        SplitHyperPlane algorithm for k-dimensional polytopes (k > 2).
        
        Args:
            vertices: k-D polytope vertices
            hyperplane_normal: Normal vector
            hyperplane_bias: Bias term
            
        Returns:
            Split polytopes
        """
        if vertices.shape[1] == 2:
            return self.split_plane_2d(vertices, hyperplane_normal, hyperplane_bias)
        
        # For higher dimensions, use general splitting approach
        return self.hyperplane_intersection.split_polytope_by_hyperplane(
            vertices, hyperplane_normal, hyperplane_bias
        )


class SemanticRegionAnalyzer:
    """
    Implements HDBSCAN clustering for semantic polytope region identification.
    """
    
    def __init__(self, min_cluster_size: int = 5, min_samples: int = 3):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
    
    def cluster_spline_codes(self, spline_codes: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Cluster spline codes to identify semantic regions using HDBSCAN.
        
        Implementation of HDBSCAN methodology:
        1. Compute mutual reachability distances
        2. Build minimum spanning tree  
        3. Build cluster hierarchy
        4. Extract stable clusters
        
        Args:
            spline_codes: Binary spline codes (N, M)
            
        Returns:
            (cluster_labels, clustering_info)
        """
        if len(spline_codes) == 0:
            raise ValueError("No spline codes provided for clustering")
        
        # Use Hamming distance for binary spline codes
        clusterer = HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric='hamming',
            cluster_selection_method='eom'  # Excess of Mass for stability
        )
        
        cluster_labels = clusterer.fit_predict(spline_codes)
        
        # Compute clustering quality metrics
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = np.sum(cluster_labels == -1)
        
        clustering_info = {
            'n_clusters': n_clusters,
            'n_noise_points': n_noise,
            'cluster_sizes': [np.sum(cluster_labels == i) for i in range(n_clusters)],
            'clustering_efficiency': 1.0 - (n_noise / len(cluster_labels)) if len(cluster_labels) > 0 else 0.0,
            'probabilities': getattr(clusterer, 'probabilities_', None)
        }
        
        return cluster_labels, clustering_info
    

    
    def identify_monosemantic_regions(self, spline_codes: np.ndarray, 
                                    cluster_labels: np.ndarray) -> Dict[int, Dict]:
        """
        Identify regions where single semantic concepts are represented.
        
        Args:
            spline_codes: Binary spline codes (N, M)
            cluster_labels: Cluster assignments from HDBSCAN
            
        Returns:
            Dictionary mapping cluster_id to region properties
        """
        regions = {}
        
        for cluster_id in set(cluster_labels):
            if cluster_id == -1:  # Skip noise
                continue
            
            cluster_mask = cluster_labels == cluster_id
            cluster_codes = spline_codes[cluster_mask]
            
            if len(cluster_codes) == 0:
                continue
            
            # Compute region properties
            region_center = np.mean(cluster_codes, axis=0)
            region_variance = np.var(cluster_codes, axis=0)
            region_sparsity = np.mean(cluster_codes)
            
            # Measure semantic coherence (lower variance = more coherent)
            coherence = 1.0 / (1.0 + np.mean(region_variance))
            
            # Identify active neuron patterns
            active_neurons = np.where(region_center > 0.5)[0]
            
            regions[cluster_id] = {
                'size': len(cluster_codes),
                'center': region_center,
                'variance': region_variance,
                'sparsity': region_sparsity,
                'coherence': coherence,
                'active_neurons': active_neurons,
                'is_monosemantic': coherence > 0.7 and len(active_neurons) < 10
            }
        
        return regions
    
    def compute_boundary_density(self, spline_codes: np.ndarray, 
                                cluster_labels: np.ndarray) -> np.ndarray:
        """
        Compute density of polytope boundaries for semantic transition detection.
        
        Args:
            spline_codes: Binary spline codes
            cluster_labels: Cluster assignments
            
        Returns:
            Boundary density for each point
        """
        n_points = len(spline_codes)
        boundary_density = np.zeros(n_points)
        
        for i in range(n_points):
            if cluster_labels[i] == -1:  # Noise points have high boundary density
                boundary_density[i] = 1.0
                continue
            
            # Count neighbors from different clusters
            different_cluster_neighbors = 0
            total_neighbors = 0
            
            for j in range(n_points):
                if i == j:
                    continue
                
                # Compute Hamming distance
                hamming_dist = np.sum(spline_codes[i] != spline_codes[j])
                
                if hamming_dist <= 3:  # Consider as neighbor
                    total_neighbors += 1
                    if cluster_labels[j] != cluster_labels[i]:
                        different_cluster_neighbors += 1
            
            if total_neighbors > 0:
                boundary_density[i] = different_cluster_neighbors / total_neighbors
        
        return boundary_density


class MASOFramework:
    """
    Max-Affine Spline Operator (MASO) framework for template matching and feature analysis.
    """
    
    def __init__(self, epsilon: float = 1e-8):
        self.epsilon = epsilon
    
    def extract_template(self, network_gradient: np.ndarray, input_point: np.ndarray,
                        network_output: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract template from network gradient for given input.
        
        Args:
            network_gradient: Gradient d[network_output]/dx (C, D) for C classes, D dimensions
            input_point: Input point (D,)
            network_output: Network output (C,)
            
        Returns:
            Template information
        """
        templates = {}
        
        for class_idx in range(len(network_output)):
            template_c = network_gradient[class_idx]  # A[x]_c
            bias_c = network_output[class_idx] - np.dot(template_c, input_point)  # b[x]_c
            
            templates[f'class_{class_idx}'] = {
                'template': template_c,
                'bias': bias_c,
                'activation': network_output[class_idx]
            }
        
        return templates
    
    def compute_maso_operator(self, input_point: np.ndarray, templates: Dict[str, Dict]) -> Dict:
        """
        Compute MASO operator S[A, B](x) = A[x]·x + B[x].
        
        Args:
            input_point: Input point x
            templates: Template dictionary from extract_template
            
        Returns:
            MASO computation results
        """
        class_activations = {}
        
        for class_name, template_info in templates.items():
            A_x = template_info['template']  # Signal-dependent transformation
            B_x = template_info['bias']      # Signal-dependent bias
            
            activation = np.dot(A_x, input_point) + B_x
            class_activations[class_name] = activation
        
        # Find winning region (max activation)
        winning_class = max(class_activations.keys(), key=lambda k: class_activations[k])
        
        return {
            'activations': class_activations,
            'winning_class': winning_class,
            'winning_activation': class_activations[winning_class]
        }
    
    def analyze_semantic_content(self, templates: Dict[str, Dict], 
                               feature_names: Optional[List[str]] = None) -> Dict:
        """
        Analyze semantic content of templates through visualization analysis.
        
        Args:
            templates: Template dictionary
            feature_names: Optional names for features
            
        Returns:
            Semantic analysis results
        """
        analysis = {}
        
        for class_name, template_info in templates.items():
            template = template_info['template']
            
            # Find most important features (highest absolute values)
            feature_importance = np.abs(template)
            top_features = np.argsort(feature_importance)[-10:][::-1]  # Top 10 features
            
            # Compute template statistics
            template_stats = {
                'norm': np.linalg.norm(template),
                'sparsity': np.mean(np.abs(template) < self.epsilon),
                'max_magnitude': np.max(np.abs(template)),
                'top_features': top_features.tolist(),
                'top_feature_values': template[top_features].tolist()
            }
            
            if feature_names:
                template_stats['top_feature_names'] = [feature_names[i] for i in top_features]
            
            analysis[class_name] = template_stats
        
        return analysis


class SafetyPolytopeManager:
    """
    Safety Polytope (SaP) implementation for LLM safety enforcement.
    """
    
    def __init__(self, epsilon: float = 1e-8):
        self.epsilon = epsilon
        self.safety_constraints = []  # List of (A, b) pairs for Ax <= b
    
    def learn_safety_polytope(self, safe_activations: np.ndarray, 
                             unsafe_activations: np.ndarray) -> Dict:
        """
        Learn polytope boundaries that separate safe from unsafe regions.
        
        Args:
            safe_activations: Safe activation patterns (N_safe, D)
            unsafe_activations: Unsafe activation patterns (N_unsafe, D)
            
        Returns:
            Learned constraint parameters
        """

        
        # Use linear programming to find separating hyperplane
        from scipy.optimize import linprog
        
        n_safe, n_dims = safe_activations.shape
        n_unsafe = len(unsafe_activations)
        
        # Formulate as linear program: find w, b such that
        # w^T * x_safe + b <= -1 (safe side)
        # w^T * x_unsafe + b >= 1 (unsafe side)
        
        # Variables: [w (n_dims), b (1), slack_variables (n_safe + n_unsafe)]
        c = np.zeros(n_dims + 1 + n_safe + n_unsafe)
        c[n_dims + 1:] = 1.0  # Minimize slack variables
        
        # Inequality constraints: -w^T * x_safe - b - slack_safe <= -1
        #                        w^T * x_unsafe + b - slack_unsafe >= 1
        A_ineq = []
        b_ineq = []
        
        # Safe constraints: -w^T * x_safe - b - slack <= -1
        for i, x_safe in enumerate(safe_activations):
            constraint = np.zeros(n_dims + 1 + n_safe + n_unsafe)
            constraint[:n_dims] = -x_safe
            constraint[n_dims] = -1  # -b
            constraint[n_dims + 1 + i] = -1  # -slack_safe
            A_ineq.append(constraint)
            b_ineq.append(-1)
        
        # Unsafe constraints: -w^T * x_unsafe - b + slack >= -1 (flipped to <=)
        for i, x_unsafe in enumerate(unsafe_activations):
            constraint = np.zeros(n_dims + 1 + n_safe + n_unsafe)
            constraint[:n_dims] = x_unsafe
            constraint[n_dims] = 1  # b
            constraint[n_dims + 1 + n_safe + i] = -1  # -slack_unsafe
            A_ineq.append(constraint)
            b_ineq.append(-1)
        
        A_ineq = np.array(A_ineq)
        b_ineq = np.array(b_ineq)
        
        try:
            result = linprog(c, A_ub=A_ineq, b_ub=b_ineq, method='highs')
            
            if result.success:
                w = result.x[:n_dims]
                b = result.x[n_dims]
                
                self.safety_constraints.append((w.reshape(1, -1), np.array([b])))
                
                return {
                    'hyperplane_normal': w,
                    'hyperplane_bias': b,
                    'optimization_success': True,
                    'slack_violation': np.sum(result.x[n_dims + 1:])
                }
            else:
                raise RuntimeError("Safety polytope optimization failed")
                
        except Exception as e:
            raise RuntimeError(f"Safety polytope learning failed: {e}")
    

    
    def project_to_safe_region(self, unsafe_activation: np.ndarray) -> np.ndarray:
        """
        Project unsafe activation to nearest safe region boundary.
        
        Args:
            unsafe_activation: Unsafe activation pattern
            
        Returns:
            Projected safe activation
        """
        current_activation = unsafe_activation.copy()
        
        for A, b in self.safety_constraints:
            # Check if point violates constraint Ax <= b
            constraint_value = np.dot(A, current_activation.reshape(-1, 1)).flatten()
            
            if np.any(constraint_value > b):
                # Project to constraint boundary
                for i, (a_i, b_i) in enumerate(zip(A, b)):
                    if constraint_value[i] > b_i:
                        # Project along normal direction
                        violation = constraint_value[i] - b_i
                        projection = violation / (np.dot(a_i, a_i) + self.epsilon)
                        current_activation -= projection * a_i
        
        return current_activation
    
    def is_safe(self, activation: np.ndarray) -> bool:
        """
        Check if activation satisfies all safety constraints.
        
        Args:
            activation: Activation pattern to check
            
        Returns:
            True if safe, False otherwise
        """
        for A, b in self.safety_constraints:
            constraint_value = np.dot(A, activation.reshape(-1, 1)).flatten()
            if np.any(constraint_value > b + self.epsilon):
                return False
        return True


class PolytopeAnalyzer:
    """
    Advanced polytope analyzer for LLM activation records with spline codes and boundary detection.
    
    Features:
    - Spline code generation for polytope identification
    - SyReNN boundary detection algorithms (ExtendPWL, SplitPlane)
    - HDBSCAN clustering for semantic region identification
    - MASO framework for template analysis
    - Safety polytope management
    - Proper V-representation management
    
    Usage:
        analyzer = PolytopeAnalyzer()
        results = analyzer.analyze_records(activation_records)
    """
    
    def __init__(self, n_pca_components: float = 0.95, approximation_epsilon: float = 1e-12, 
                 random_seed: Optional[int] = None, dimensionality_reduction: str = "pca",
                 cett_target: float = 0.01, min_cluster_size: int = 5, use_advanced_methods: bool = True):
        """
        Initialize the advanced polytope analyzer.
        
        Args:
            n_pca_components: Number of PCA components for dimensionality reduction
            approximation_epsilon: Numerical precision for geometric computations
            random_seed: Random seed for reproducibility (None for random behavior)
            dimensionality_reduction: Method for dimensionality reduction ('pca', 'none', 'truncate')
            cett_target: Target CETT value for spline code thresholding (0.01 = 1% error tolerance)
            min_cluster_size: Minimum cluster size for HDBSCAN clustering
            use_advanced_methods: Whether to use advanced methods (SyReNN, HDBSCAN, etc.)
        """
        self.n_pca_components = n_pca_components
        self.approximation_epsilon = approximation_epsilon
        self.random_seed = random_seed
        self.dimensionality_reduction = dimensionality_reduction
        self.use_advanced_methods = use_advanced_methods
        
        # Initialize all components
        self.spline_generator = SplineCodeGenerator(cett_target)
        self.hyperplane_intersection = HyperplaneIntersection(approximation_epsilon)
        self.v_rep = VRepresentationManager(approximation_epsilon)
        self.boundary_detector = SyReNNBoundaryDetector(approximation_epsilon)
        self.semantic_analyzer = SemanticRegionAnalyzer(min_cluster_size)
        self.maso_framework = MASOFramework()
        self.safety_manager = SafetyPolytopeManager()
        
        # Set random seed if provided
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def _ensure_random_seed(self):
        """Ensure random seed is set for reproducible results."""
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
    
    def validate_records(self, records: List[Dict]) -> List[Dict]:
        """
        Validate activation records and ensure required fields exist.
        
        Required fields from checkpoint analysis:
        - activation_vector: Neural network activation data (np.ndarray)
        - category: 'high' or 'low' frequency category
        - sparsity: Sparsity measure
        - activation_norm: Activation norm
        - n_active_neurons: Number of active neurons
        
        Args:
            records: List of activation records from checkpoint analysis
            
        Returns:
            List of validated records compatible with polytope analysis
        """
        validated = []
        required_fields = ['activation_vector', 'category', 'sparsity', 'activation_norm', 'n_active_neurons']
        
        for i, record in enumerate(records):
            # Check required fields
            if not all(field in record for field in required_fields):
                missing = [f for f in required_fields if f not in record]
                logger.warning(f"Record {i} missing required fields {missing}, skipping")
                continue
            
            # Validate category
            if record['category'] not in ['high', 'low']:
                logger.warning(f"Record {i} has invalid category '{record['category']}', must be 'high' or 'low'")
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
            
            # Validate numeric fields
            try:
                float(record['sparsity'])
                float(record['activation_norm'])
                int(record['n_active_neurons'])
            except (ValueError, TypeError):
                logger.warning(f"Record {i} has invalid numeric values, skipping")
                continue
            
            validated.append(record)
        
        logger.info(f"Validated {len(validated)}/{len(records)} records")
        return validated
    
    def reduce_dimensions(self, points: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Reduce dimensionality using configurable methods for computational efficiency.
        
        Args:
            points: Input points (n_samples, n_features)
            
        Returns:
            (reduced_points, reduction_info)
        """
        if points.shape[0] < 2:
            return points, {}
        
        if self.dimensionality_reduction == "none":
            # No dimensionality reduction
            return points, {
                'method': 'none',
                'n_components': points.shape[1],
                'original_dims': points.shape[1],
                'variance_explained': 1.0
            }
        
        elif self.dimensionality_reduction == "truncate":
            # Simple truncation to first N dimensions
            max_dims = min(50, points.shape[1])  # Limit to 50 dimensions
            points_truncated = points[:, :max_dims]
            
            return points_truncated, {
                'method': 'truncate',
                'n_components': max_dims,
                'original_dims': points.shape[1],
                'variance_explained': None
            }
        
        else:  # "pca" (default)
            n_components = min(self.n_pca_components, points.shape[0] - 1, points.shape[1])
            
            # Standardize data
            scaler = StandardScaler()
            points_scaled = scaler.fit_transform(points)
            
            # Apply PCA
            pca = PCA(n_components=n_components)
            points_reduced = pca.fit_transform(points_scaled)
            
            variance_explained = np.sum(pca.explained_variance_ratio_)
            
            return points_reduced, {
                'method': 'pca',
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
        
        # Add more random points for diversity and to prevent convergence to same vertices
        if len(extreme_indices) < min(50, n_points // 20):
            remaining_indices = set(range(n_points)) - extreme_indices
            if remaining_indices:
                self._ensure_random_seed()
                # Increase random sampling to add more diversity
                n_random = min(20, len(remaining_indices))  # Increased from 10 to 20
                random_indices = np.random.choice(list(remaining_indices), size=n_random, replace=False)
                extreme_indices.update(random_indices)
        
        # Add additional diversity by including points with high variance in different subspaces
        if len(extreme_indices) < min(60, n_points // 10):  # Increased target size
            remaining_indices = set(range(n_points)) - extreme_indices
            if remaining_indices:
                # Find points with high variance in different coordinate directions
                remaining_points = points[list(remaining_indices)]
                variances = np.var(remaining_points, axis=0)
                high_var_indices = np.argsort(variances)[-min(10, len(remaining_indices)):]
                extreme_indices.update([list(remaining_indices)[i] for i in high_var_indices])
        
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

    def _compute_adaptive_epsilon(self, points: np.ndarray) -> float:
        """
        Compute adaptive epsilon based on data scale and distribution.
        
        Args:
            points: Input points
            
        Returns:
            Adaptive epsilon value
        """
        if len(points) < 2:
            return self.approximation_epsilon
        
        # Use pairwise distances to estimate data scale
        try:
            distances = pdist(points)
            if len(distances) == 0:
                return self.approximation_epsilon
            
            # Use percentile-based approach for robustness
            distance_scale = np.percentile(distances, 10)  # 10th percentile for robustness
            
            # Adaptive epsilon: smaller for tighter clusters, larger for spread out data
            base_epsilon = min(self.approximation_epsilon, 0.01)
            
            # Make epsilon more sensitive to data changes by using a wider range
            # and incorporating data variance
            data_variance = np.var(points)
            variance_factor = min(2.0, max(0.5, np.sqrt(data_variance) / 10))
            
            adaptive_epsilon = max(base_epsilon * 0.05, min(base_epsilon * 3, distance_scale * 0.1 * variance_factor))
            
            # Add small random component to prevent deterministic convergence
            self._ensure_random_seed()
            random_factor = 1.0 + 0.1 * np.random.random()
            adaptive_epsilon *= random_factor
            
            return float(adaptive_epsilon)
            
        except Exception:
            return self.approximation_epsilon

    def find_hull_vertices(self, points: np.ndarray) -> np.ndarray:
        """
        Find convex hull vertices using advanced or fallback methods.
        
        Args:
            points: Input points (n_samples, n_features)
            
        Returns:
            Indices of hull vertices
        """
        if self.use_advanced_methods:
            # Use proper convex hull computation
            try:
                hull_vertices = self.v_rep.compute_convex_hull(points)
                # Return indices by finding matches in original points
                indices = []
                for vertex in hull_vertices:
                    # Find closest match in original points
                    distances = np.linalg.norm(points - vertex, axis=1)
                    indices.append(np.argmin(distances))
                return np.array(indices)
            except Exception as e:
                logger.warning(f"Advanced hull computation failed: {e}, using fallback")
        
        # Fallback to original greedy approximation
        adaptive_epsilon = self._compute_adaptive_epsilon(points)
        return self.greedy_hull_approximation(points, epsilon=adaptive_epsilon)

    def revised_greedy_expansion_algorithm(self, points: np.ndarray, epsilon: float = 0.05, 
                                         max_iter: int = 1000, max_runtime: float = 60.0, verbose: bool = False,
                                         max_hull_vertices: int = None) -> np.ndarray:
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
                    
                    # Check if we've reached the maximum hull vertices limit
                    if max_hull_vertices is not None and len(E) >= max_hull_vertices:
                        if verbose:
                            logger.info(f"Reached maximum hull vertices limit ({max_hull_vertices}), stopping")
                        break
                    
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
        
        # Use more conservative epsilon to prevent including all points
        conservative_epsilon = max(epsilon * 2.0, 0.1)  # Increase epsilon for more conservative hull
        
        # Limit maximum hull vertices to prevent degenerate cases
        max_hull_vertices = min(len(points) // 4, 50)  # At most 25% of points or 50 vertices
        
        return self.revised_greedy_expansion_algorithm(
            points, conservative_epsilon, max_iter, max_runtime, verbose, max_hull_vertices
        )

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
            self._ensure_random_seed()
            candidate_indices = np.random.choice(R_list, size=n_candidates, replace=False)
        else:
            candidate_indices = R_list
        
        # Sample evaluation points (max 50 for speed)
        n_eval_points = min(50, len(R_list))
        if len(R_list) > n_eval_points:
            self._ensure_random_seed()
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
            self._ensure_random_seed()
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
            self._ensure_random_seed()
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
        Compute polytope volume using robust approximation methods for high-dimensional data.
        
        Args:
            points: Input points defining the polytope
            
        Returns:
            Estimated volume using improved hull approximation
        """
        if len(points) < 2:
            return 0.0
        
        n_points, n_dims = points.shape
        
        # For very small point sets, use simple volume estimates
        if n_points < 4:
            return self._compute_simplex_volume(points)
        
        try:
            # For high-dimensional sparse data, use bounding box volume as more stable estimate
            bounding_volume = self._compute_bounding_box_volume(points)
            
            # If bounding volume is too small, the polytope is essentially degenerate
            if bounding_volume < 1e-15:
                logger.debug("Degenerate polytope detected, using effective volume")
                return self._compute_effective_volume(points)
            
            # Get approximate hull vertices
            hull_vertex_indices = self.find_hull_vertices(points)
            hull_points = points[hull_vertex_indices]
            
            # Use improved Monte Carlo volume estimation with fallback
            mc_volume = self._robust_monte_carlo_volume(hull_points)
            
            # Return the more conservative estimate
            return min(mc_volume, bounding_volume) if mc_volume > 0 else bounding_volume
            
        except Exception as e:
            logger.warning(f"Volume computation failed, using fallback: {e}")
            return self._compute_bounding_box_volume(points)
    
    def _compute_simplex_volume(self, points: np.ndarray) -> float:
        """
        Compute volume for small point sets using simplex approximation.
        
        Args:
            points: Input points (n_samples <= 3, n_features)
            
        Returns:
            Simplex volume
        """
        n_points, n_dims = points.shape
        
        if n_points == 1:
            return 0.0
        elif n_points == 2:
            # Line segment length
            return np.linalg.norm(points[1] - points[0])
        elif n_points == 3:
            # Triangle area in any dimension
            v1 = points[1] - points[0]
            v2 = points[2] - points[0]
            # Area = 0.5 * |v1 × v2| (cross product magnitude)
            cross_prod = np.cross(v1, v2)
            if np.isscalar(cross_prod):
                return 0.5 * abs(cross_prod)
            else:
                return 0.5 * np.linalg.norm(cross_prod)
        else:
            # For more points, use bounding box approximation
            return self._compute_bounding_box_volume(points)
    
    def _compute_bounding_box_volume(self, points: np.ndarray) -> float:
        """
        Compute bounding box volume as a fallback method.
        
        Args:
            points: Input points
            
        Returns:
            Bounding box volume
        """
        if len(points) < 2:
            return 0.0
        
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)
        coord_ranges = max_coords - min_coords
        
        # Handle degenerate dimensions
        non_zero_ranges = coord_ranges[coord_ranges > 1e-12]
        if len(non_zero_ranges) == 0:
            return 0.0
        
        return np.prod(non_zero_ranges)
    
    def _robust_monte_carlo_volume(self, hull_points: np.ndarray, n_samples: int = 10000) -> float:
        """
        Improved Monte Carlo volume estimation with better numerical stability.
        
        Args:
            hull_points: Points defining the approximate hull
            n_samples: Number of Monte Carlo samples
            
        Returns:
            Estimated volume
        """
        if len(hull_points) < 2:
            return 0.0
        
        n_dims = hull_points.shape[1]
        
        # Find bounding box
        min_coords = np.min(hull_points, axis=0)
        max_coords = np.max(hull_points, axis=0)
        coord_ranges = max_coords - min_coords
        
        # Check for degenerate cases
        non_degenerate_dims = coord_ranges > 1e-12
        if not np.any(non_degenerate_dims):
            return 0.0
        
        # Volume of effective bounding box
        effective_ranges = coord_ranges[non_degenerate_dims]
        if len(effective_ranges) == 0:
            return 0.0
        
        box_volume = np.prod(effective_ranges)
        
        if box_volume <= 1e-20:
            return 0.0
        
        # Adaptive sampling based on dimensionality
        adaptive_samples = min(n_samples, max(1000, 100 * n_dims))
        
        # Monte Carlo estimation with batching for memory efficiency
        n_inside = 0
        batch_size = min(1000, adaptive_samples)
        
        # Adaptive threshold based on data scale
        adaptive_threshold = max(1e-6, 0.01 * np.mean(effective_ranges))
        
        for batch_start in range(0, adaptive_samples, batch_size):
            current_batch_size = min(batch_size, adaptive_samples - batch_start)
            
            # Generate random points
            self._ensure_random_seed()
            random_points = np.random.uniform(
                min_coords, max_coords, size=(current_batch_size, n_dims)
            )
            
            # Check points against hull using vectorized operations where possible
            for point in random_points:
                distance = self.distance_to_approximate_hull(point, hull_points)
                if distance <= adaptive_threshold:  # Adaptive tolerance for "inside"
                    n_inside += 1
        
        # Estimate volume
        volume_ratio = n_inside / adaptive_samples
        estimated_volume = box_volume * volume_ratio
        
        return float(estimated_volume)
    
    def _compute_effective_volume(self, points: np.ndarray) -> float:
        """
        Compute effective volume for degenerate polytopes based on point spread.
        
        Args:
            points: Input points
            
        Returns:
            Effective volume measure
        """
        n_points, n_dims = points.shape
        
        # Compute pairwise distances
        try:
            distances = pdist(points)
            if len(distances) == 0:
                return 0.0
            
            # Use geometric mean of distances as volume proxy
            mean_distance = np.mean(distances)
            std_distance = np.std(distances)
            
            # Volume proxy: mean distance raised to effective dimension
            # Use coefficient of variation to estimate effective dimension
            if mean_distance > 0:
                cv = std_distance / mean_distance
                effective_dim = min(n_dims, max(1.0, 3.0 * cv))  # Heuristic
                return mean_distance ** effective_dim
            else:
                return 0.0
                
        except Exception:
            return 0.0
    
    def _approximate_monte_carlo_volume(self, hull_points: np.ndarray, n_samples: int = 10000) -> float:
        """
        Legacy method - redirects to robust Monte Carlo volume estimation.
        
        Args:
            hull_points: Points defining the approximate hull
            n_samples: Number of Monte Carlo samples
            
        Returns:
            Approximate volume using improved method
        """
        return self._robust_monte_carlo_volume(hull_points, n_samples)
    
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
                # Stable surface area estimation using logarithmic scaling
                # Avoid exponential explosion while preserving relative differences
                base_area = n_hull_vertices * mean_distance
                
                # Use log scaling for dimensional effects to prevent numerical explosion
                if n_dims > 3:
                    # For high dimensions, use logarithmic scaling
                    dim_factor = 1 + np.log(n_dims) * np.log(max_distance / mean_distance + 1)
                else:
                    # For low dimensions, use conservative power scaling
                    dim_factor = (mean_distance / max(mean_distance/10, 1)) ** min(2, n_dims - 1)
                
                surface_scaling = base_area * dim_factor
                
                # Apply correction factor for high dimensions
                dim_correction = np.sqrt(max(1, n_dims / 10))
                
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
    
    def _get_default_metrics(self) -> Dict[str, float]:
        """
        Get default polytope metrics for degenerate cases.
        
        Returns:
            Dictionary with default metric values
        """
        return {
            'volume': 0.0,
            'surface_area': 0.0,
            'n_vertices': 0,
            'n_facets': 0,
            'mean_distance': 0.0,
            'std_distance': 0.0,
            'centroid_variance': 0.0,
            'effective_dimension': 0.0,
            'boundary_density': 0.0,
            'inter_polytope_distance': 0.0,
            'geometric_regularity': 0.0,
            'hull_valid': False
        }
    
    def compute_polytope_metrics(self, points: np.ndarray) -> Dict[str, float]:
        """
        Compute comprehensive polytope metrics.
        
        Args:
            points: Input points (n_samples, n_features)
            
        Returns:
            Dictionary of polytope metrics
        """
        try:
            # Validation checks
            if len(points) < 2:
                logger.warning(f"Insufficient points for polytope analysis: {len(points)}")
                return self._get_default_metrics()
            
            # Find hull vertices
            hull_indices = self.find_hull_vertices(points)
            hull_points = points[hull_indices]
            
            # Validation: ensure hull vertices are reasonable
            hull_ratio = len(hull_indices) / len(points)
            if hull_ratio > 0.95:  # Increased threshold - high ratios are normal for sparse neural data
                logger.warning(f"Very high hull vertex ratio {hull_ratio:.3f} - may indicate degenerate polytope")
            elif hull_ratio > 0.8:
                logger.debug(f"High hull vertex ratio {hull_ratio:.3f} - normal for sparse high-dimensional data")
            
            # Debug: Log hull vertex count for tracking changes
            logger.debug(f"Hull vertices: {len(hull_indices)} out of {len(points)} points (ratio: {hull_ratio:.3f})")
            
            # Compute volume and surface area
            volume = self.compute_volume(hull_points)
            surface_area = self.compute_surface_area(hull_points)
            
            # Debug information
            logger.debug(f"Polytope metrics: n_points={len(points)}, n_hull={len(hull_indices)}, "
                        f"volume={volume:.2e}, surface_area={surface_area:.2e}")
            
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
            
            # Boundary density (ratio of hull vertices to total points)
            boundary_density = len(hull_indices) / len(points)
            
            # Inter-polytope distance (distance from centroid to hull boundary)
            inter_polytope_distance = self.distance_to_approximate_hull(centroid, hull_points)
            
            # Geometric regularity (coefficient of variation of hull vertex distances)
            if len(hull_points) > 1:
                hull_distances = pdist(hull_points)
                if len(hull_distances) > 0 and np.mean(hull_distances) > 0:
                    geometric_regularity = np.std(hull_distances) / np.mean(hull_distances)
                else:
                    geometric_regularity = 0.0
            else:
                geometric_regularity = 0.0
            
            return {
                'volume': float(volume),
                'surface_area': float(surface_area),
                'n_vertices': len(hull_indices),
                'n_facets': 0,  # Computing facets is expensive for high-dim
                'mean_distance': float(mean_distance),
                'std_distance': float(std_distance),
                'centroid_variance': float(centroid_variance),
                'effective_dimension': float(effective_dim),
                'boundary_density': float(boundary_density),
                'inter_polytope_distance': float(inter_polytope_distance),
                'geometric_regularity': float(geometric_regularity),
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
        frequency_categories = [r['category'] for r in records]
        # get distinct frequency categories
        distinct_categories = list(set(frequency_categories))
        # stratify records by frequency category
        stratified_records = {cat: [r for r in records if r['category'] == cat] for cat in distinct_categories}
        # should be {'high': [*ngrams], 'low': [*ngrams]}
        return stratified_records
        
    
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
            'mean_sparsity': np.mean([r['sparsity'] for r in records]),
            'std_sparsity': np.std([r['sparsity'] for r in records]),
            'mean_norm': np.mean([r['activation_norm'] for r in records]),
            'std_norm': np.std([r['activation_norm'] for r in records]),
            'mean_active_neurons': np.mean([r['n_active_neurons'] for r in records]),
            'std_active_neurons': np.std([r['n_active_neurons'] for r in records])
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
        Compare high vs low frequency groups across checkpoints and layers.
        
        Args:
            records: List of activation records
            
        Returns:
            Comparison results across checkpoints and layers
        """
        # Group by (checkpoint, layer) for proper comparison
        checkpoint_layer_groups = defaultdict(list)
        for record in records:
            key = (record['checkpoint_step'], record.get('layer', 0))
            checkpoint_layer_groups[key].append(record)
        
        results = {
            'checkpoint_layer_analysis': {},
            'comparison_summary': {},
            'evolution_metrics': []
        }
        
        for (checkpoint, layer), checkpoint_layer_records in checkpoint_layer_groups.items():
            logger.info(f"Analyzing checkpoint {checkpoint}, layer {layer} with {len(checkpoint_layer_records)} records")
            
            # Skip if insufficient data for this checkpoint-layer combination
            if len(checkpoint_layer_records) < 4:
                logger.warning(f"Insufficient records for checkpoint {checkpoint}, layer {layer}: {len(checkpoint_layer_records)}")
                continue
            
            # Stratify by frequency within this checkpoint-layer group
            freq_groups = self.stratify_by_frequency(checkpoint_layer_records)
            
            # Verify we have both frequency groups
            if 'high' not in freq_groups or 'low' not in freq_groups:
                logger.warning(f"Missing frequency groups for checkpoint {checkpoint}, layer {layer}")
                continue
            
            # Check minimum group sizes
            if len(freq_groups['high']) < 2 or len(freq_groups['low']) < 2:
                logger.warning(f"Insufficient records in frequency groups for checkpoint {checkpoint}, layer {layer}")
                continue
            
            checkpoint_results = {}
            
            # Analyze each frequency group
            for group_name, group_records in freq_groups.items():
                group_analysis = self.analyze_frequency_group(group_records, group_name)
                checkpoint_results[group_name] = group_analysis
            
            # Compare groups if both exist
            if 'high' in checkpoint_results and 'low' in checkpoint_results:
                high_metrics = checkpoint_results['high']['polytope_metrics']
                low_metrics = checkpoint_results['low']['polytope_metrics']
                high_stats = checkpoint_results['high']['activation_stats']
                low_stats = checkpoint_results['low']['activation_stats']
                
                def safe_ratio(a, b, fallback=1.0):
                    """Compute ratio safely, handling zero denominators and extreme values."""
                    if abs(b) < 1e-15:  # Denominator is essentially zero
                        return fallback if abs(a) < 1e-15 else (1e6 if a > 0 else -1e6)
                    ratio = a / b
                    # Cap extreme ratios for numerical stability
                    return max(-1e6, min(1e6, ratio))
                
                comparison = {
                    # Volume metrics
                    'volume_ratio_high_to_low': safe_ratio(high_metrics['volume'], low_metrics['volume']),
                    'volume_difference': high_metrics['volume'] - low_metrics['volume'],
                    
                    # Surface area metrics  
                    'surface_area_ratio_high_to_low': safe_ratio(high_metrics['surface_area'], low_metrics['surface_area']),
                    'surface_area_difference': high_metrics['surface_area'] - low_metrics['surface_area'],
                    
                    # Boundary metrics
                    'boundary_density_ratio': safe_ratio(high_metrics['boundary_density'], low_metrics['boundary_density']),
                    'boundary_density_difference': high_metrics['boundary_density'] - low_metrics['boundary_density'],
                    
                    # Distance metrics
                    'inter_polytope_distance_ratio': safe_ratio(high_metrics['inter_polytope_distance'], low_metrics['inter_polytope_distance']),
                    'inter_polytope_distance_difference': high_metrics['inter_polytope_distance'] - low_metrics['inter_polytope_distance'],
                    
                    # Regularity metrics
                    'geometric_regularity_ratio': safe_ratio(high_metrics['geometric_regularity'], low_metrics['geometric_regularity']),
                    'geometric_regularity_difference': high_metrics['geometric_regularity'] - low_metrics['geometric_regularity'],
                    
                    # Dimension metrics
                    'effective_dimension_ratio': safe_ratio(high_metrics['effective_dimension'], low_metrics['effective_dimension']),
                    'effective_dimension_difference': high_metrics['effective_dimension'] - low_metrics['effective_dimension'],
                    
                    # Vertex count metrics
                    'n_vertices_ratio': safe_ratio(high_metrics['n_vertices'], low_metrics['n_vertices']),
                    'n_vertices_difference': high_metrics['n_vertices'] - low_metrics['n_vertices'],
                    
                    # Distance spread metrics
                    'mean_distance_ratio': safe_ratio(high_metrics['mean_distance'], low_metrics['mean_distance']),
                    'std_distance_ratio': safe_ratio(high_metrics['std_distance'], low_metrics['std_distance']),
                    
                    # Centroid metrics
                    'centroid_variance_ratio': safe_ratio(high_metrics['centroid_variance'], low_metrics['centroid_variance']),
                    'centroid_variance_difference': high_metrics['centroid_variance'] - low_metrics['centroid_variance'],
                    
                    # Activation stats (original metrics)
                    'sparsity_difference': low_stats['mean_sparsity'] - high_stats['mean_sparsity'],
                    'norm_ratio': safe_ratio(high_stats['mean_norm'], low_stats['mean_norm'])
                }
                checkpoint_results['comparison'] = comparison
            
            # Use compound key for checkpoint-layer analysis
            results['checkpoint_layer_analysis'][f"{checkpoint}_layer{layer}"] = checkpoint_results
        
        # Compute evolution metrics for all polytope metrics
        evolution_data = []
        for key, data in results['checkpoint_layer_analysis'].items():
            if 'comparison' in data:
                comp = data['comparison']
                # Extract checkpoint and layer from key
                checkpoint_str, layer_str = key.split('_layer')
                evolution_point = {
                    'checkpoint_layer_key': key,
                    'checkpoint': checkpoint_str,
                    'layer': int(layer_str),
                    # Volume evolution
                    'volume_ratio': comp['volume_ratio_high_to_low'],
                    'volume_difference': comp['volume_difference'],
                    # Surface area evolution  
                    'surface_area_ratio': comp['surface_area_ratio_high_to_low'],
                    'surface_area_difference': comp['surface_area_difference'],
                    # Boundary evolution
                    'boundary_density_ratio': comp['boundary_density_ratio'],
                    'boundary_density_difference': comp['boundary_density_difference'],
                    # Distance evolution
                    'inter_polytope_distance_ratio': comp['inter_polytope_distance_ratio'],
                    'inter_polytope_distance_difference': comp['inter_polytope_distance_difference'],
                    # Regularity evolution
                    'geometric_regularity_ratio': comp['geometric_regularity_ratio'],
                    'geometric_regularity_difference': comp['geometric_regularity_difference'],
                    # Dimension evolution
                    'effective_dimension_ratio': comp['effective_dimension_ratio'],
                    'effective_dimension_difference': comp['effective_dimension_difference'],
                    # Vertex evolution
                    'n_vertices_ratio': comp['n_vertices_ratio'],
                    'n_vertices_difference': comp['n_vertices_difference'],
                    # Distance spread evolution
                    'mean_distance_ratio': comp['mean_distance_ratio'],
                    'std_distance_ratio': comp['std_distance_ratio'],
                    # Centroid evolution
                    'centroid_variance_ratio': comp['centroid_variance_ratio'],
                    'centroid_variance_difference': comp['centroid_variance_difference'],
                    # Original activation stats
                    'sparsity_difference': comp['sparsity_difference'],
                    'norm_ratio': comp['norm_ratio']
                }
                evolution_data.append(evolution_point)
        
        results['evolution_metrics'] = evolution_data
        
        return results
    
    def create_visualizations(self, analysis_results: Dict[str, Any], save_dir: str = "./polytope_plots") -> Dict[str, plt.Figure]:
        """
        Create comprehensive visualizations of all polytope analysis results.
        
        Args:
            analysis_results: Results from compare_across_checkpoints
            save_dir: Directory to save plots
            
        Returns:
            Dictionary of matplotlib figures
        """
        if not PLOTTING_AVAILABLE:
            logger.warning("Plotting libraries not available, skipping visualizations")
            return {}
            
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        
        figures = {}
        
        if not analysis_results['evolution_metrics']:
            logger.warning("No evolution metrics found, skipping visualizations")
            return figures
            
        evolution_df = pd.DataFrame(analysis_results['evolution_metrics'])
        
        # Plot 1: Core Polytope Metrics Evolution
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Volume evolution
        ax1.plot(evolution_df['checkpoint'], evolution_df['volume_ratio'], 'o-', linewidth=2, markersize=8, color='blue')
        ax1.set_title('Volume Ratio Evolution (High/Low Frequency)', fontweight='bold')
        ax1.set_xlabel('Checkpoint')
        ax1.set_ylabel('Volume Ratio')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Equal volumes')
        ax1.legend()
        ax1.set_yscale('log')
        
        # Surface area evolution
        ax2.plot(evolution_df['checkpoint'], evolution_df['surface_area_ratio'], 's-', linewidth=2, markersize=8, color='orange')
        ax2.set_title('Surface Area Ratio Evolution (High/Low)', fontweight='bold')
        ax2.set_xlabel('Checkpoint')
        ax2.set_ylabel('Surface Area Ratio')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Equal surface areas')
        ax2.legend()
        ax2.set_yscale('log')
        
        # Boundary density evolution
        ax3.plot(evolution_df['checkpoint'], evolution_df['boundary_density_ratio'], '^-', linewidth=2, markersize=8, color='green')
        ax3.set_title('Boundary Density Ratio Evolution (High/Low)', fontweight='bold')
        ax3.set_xlabel('Checkpoint')
        ax3.set_ylabel('Boundary Density Ratio')
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Equal density')
        ax3.legend()
        
        # Geometric regularity evolution
        ax4.plot(evolution_df['checkpoint'], evolution_df['geometric_regularity_ratio'], 'd-', linewidth=2, markersize=8, color='purple')
        ax4.set_title('Geometric Regularity Ratio Evolution (High/Low)', fontweight='bold')
        ax4.set_xlabel('Checkpoint')
        ax4.set_ylabel('Regularity Ratio')
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Equal regularity')
        ax4.legend()
        
        plt.tight_layout()
        figures['core_metrics_evolution'] = fig
        fig.savefig(save_dir / 'core_metrics_evolution.png', dpi=300, bbox_inches='tight')
        
        # Plot 2: Distance and Dimension Metrics
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Inter-polytope distance evolution
        ax1.plot(evolution_df['checkpoint'], evolution_df['inter_polytope_distance_ratio'], 'o-', linewidth=2, markersize=8, color='brown')
        ax1.set_title('Inter-Polytope Distance Ratio (High/Low)', fontweight='bold')
        ax1.set_xlabel('Checkpoint')
        ax1.set_ylabel('Distance Ratio')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Equal distances')
        ax1.legend()
        
        # Effective dimension evolution
        ax2.plot(evolution_df['checkpoint'], evolution_df['effective_dimension_ratio'], 's-', linewidth=2, markersize=8, color='teal')
        ax2.set_title('Effective Dimension Ratio (High/Low)', fontweight='bold')
        ax2.set_xlabel('Checkpoint')
        ax2.set_ylabel('Dimension Ratio')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Equal dimensions')
        ax2.legend()
        
        # Number of vertices evolution
        ax3.plot(evolution_df['checkpoint'], evolution_df['n_vertices_ratio'], '^-', linewidth=2, markersize=8, color='navy')
        ax3.set_title('Number of Vertices Ratio (High/Low)', fontweight='bold')
        ax3.set_xlabel('Checkpoint')
        ax3.set_ylabel('Vertices Ratio')
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Equal vertices')
        ax3.legend()
        
        # Centroid variance evolution
        ax4.plot(evolution_df['checkpoint'], evolution_df['centroid_variance_ratio'], 'd-', linewidth=2, markersize=8, color='crimson')
        ax4.set_title('Centroid Variance Ratio (High/Low)', fontweight='bold')
        ax4.set_xlabel('Checkpoint')
        ax4.set_ylabel('Variance Ratio')
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Equal variance')
        ax4.legend()
        
        plt.tight_layout()
        figures['distance_dimension_metrics'] = fig
        fig.savefig(save_dir / 'distance_dimension_metrics.png', dpi=300, bbox_inches='tight')
        
        # Plot 3: Comprehensive Heatmap of All Metrics
        ratio_metrics = [
            'volume_ratio', 'surface_area_ratio', 'boundary_density_ratio',
            'inter_polytope_distance_ratio', 'geometric_regularity_ratio',
            'effective_dimension_ratio', 'n_vertices_ratio', 'centroid_variance_ratio',
            'mean_distance_ratio', 'std_distance_ratio', 'norm_ratio'
        ]
        
        ratio_labels = [
            'Volume', 'Surface Area', 'Boundary Density',
            'Inter-Polytope Dist', 'Geometric Regularity',
            'Effective Dimension', 'Vertices Count', 'Centroid Variance',
            'Mean Distance', 'Std Distance', 'Activation Norm'
        ]
        
        # Create heatmap data
        heatmap_data = []
        for metric in ratio_metrics:
            if metric in evolution_df.columns:
                heatmap_data.append(evolution_df[metric].values)
            else:
                heatmap_data.append(np.ones(len(evolution_df)))  # fallback
                
        heatmap_matrix = np.array(heatmap_data)
        
        fig, ax = plt.subplots(1, 1, figsize=(14, 10))
        im = ax.imshow(np.log10(np.maximum(heatmap_matrix, 1e-6)), aspect='auto', cmap='RdBu_r', vmin=-2, vmax=2)
        
        ax.set_xticks(range(len(evolution_df)))
        ax.set_xticklabels([f"CP{cp}" for cp in evolution_df['checkpoint']], rotation=45)
        ax.set_yticks(range(len(ratio_labels)))
        ax.set_yticklabels(ratio_labels)
        ax.set_title('Polytope Metrics Evolution Heatmap\n(Log10 of High/Low Ratios)', fontweight='bold', pad=20)
        
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Log10(High/Low Ratio)', rotation=270, labelpad=20)
        
        # Add text annotations for significant values
        for i in range(len(ratio_labels)):
            for j in range(len(evolution_df)):
                if abs(np.log10(heatmap_matrix[i, j])) > 0.3:  # Only annotate significant differences
                    text = f"{heatmap_matrix[i, j]:.2f}"
                    ax.text(j, i, text, ha="center", va="center", color="white", fontweight='bold')
        
        plt.tight_layout()
        figures['comprehensive_heatmap'] = fig
        fig.savefig(save_dir / 'comprehensive_heatmap.png', dpi=300, bbox_inches='tight')
        
        # Plot 4: Detailed Comparison by Group
        checkpoint_data = []
        for key, data in analysis_results.get('checkpoint_layer_analysis', {}).items():
            for group in ['high', 'low']:
                if group in data and 'polytope_metrics' in data[group]:
                    metrics = data[group]['polytope_metrics']
                    stats = data[group]['activation_stats']
                    
                    # Extract checkpoint and layer from key
                    checkpoint_str, layer_str = key.split('_layer')
                    checkpoint_data.append({
                        'checkpoint_layer_key': key,
                        'checkpoint': checkpoint_str,
                        'layer': int(layer_str),
                        'frequency_group': group,
                        'volume': metrics['volume'],
                        'surface_area': metrics['surface_area'],
                        'boundary_density': metrics['boundary_density'],
                        'inter_polytope_distance': metrics['inter_polytope_distance'],
                        'geometric_regularity': metrics['geometric_regularity'],
                        'effective_dimension': metrics['effective_dimension'],
                        'n_vertices': metrics['n_vertices'],
                        'centroid_variance': metrics['centroid_variance'],
                        'mean_sparsity': stats['mean_sparsity'],
                        'mean_norm': stats['mean_norm']
                    })
        
        if checkpoint_data:
            df = pd.DataFrame(checkpoint_data)
            
            # Create subplot grid for all metrics
            fig, axes = plt.subplots(3, 4, figsize=(20, 15))
            axes = axes.flatten()
            
            detailed_metrics = [
                'volume', 'surface_area', 'boundary_density', 'inter_polytope_distance',
                'geometric_regularity', 'effective_dimension', 'n_vertices', 'centroid_variance',
                'mean_sparsity', 'mean_norm'
            ]
            
            colors = {'high': 'red', 'low': 'blue'}
            markers = {'high': 'o', 'low': 's'}
            
            for i, metric in enumerate(detailed_metrics):
                if i < len(axes):
                    for group in ['high', 'low']:
                        group_data = df[df['frequency_group'] == group]
                        if not group_data.empty:
                            axes[i].plot(group_data['checkpoint'], group_data[metric], 
                                       marker=markers[group], color=colors[group],
                                       label=f'{group.title()} Frequency', linewidth=2, markersize=6)
                    
                    axes[i].set_title(f'{metric.replace("_", " ").title()}', fontweight='bold')
                    axes[i].set_xlabel('Checkpoint')
                    axes[i].set_ylabel(metric.replace("_", " ").title())
                    axes[i].legend()
                    axes[i].grid(True, alpha=0.3)
                    
                    # Use log scale for metrics with large ranges
                    if metric in ['volume', 'surface_area', 'n_vertices']:
                        axes[i].set_yscale('log')
            
            # Remove empty subplots
            for i in range(len(detailed_metrics), len(axes)):
                fig.delaxes(axes[i])
            
            plt.tight_layout()
            figures['detailed_metrics_comparison'] = fig
            fig.savefig(save_dir / 'detailed_metrics_comparison.png', dpi=300, bbox_inches='tight')
        
        logger.info(f"Created {len(figures)} comprehensive visualization plots in {save_dir}")
        return figures
    
    def analyze_polytope_structure(self, records: List[Dict]) -> Dict[str, Any]:
        """
        Comprehensive polytope analysis using advanced methodology.
        
        Analysis pipeline:
        1. Generate spline codes with CETT thresholding
        2. Cluster spline codes using HDBSCAN  
        3. Identify polytope regions and compute metrics
        4. Extract templates using MASO framework
        5. Perform safety analysis if applicable
        
        Args:
            records: List of validated activation records
            
        Returns:
            Complete polytope analysis results
        """
        logger.info(f"Starting polytope structure analysis of {len(records)} records")
        
        # Step 1: Generate spline codes
        activation_vectors = np.array([r['activation_vector'] for r in records])
        spline_codes = []
        for activation in activation_vectors:
            spline_code = self.spline_generator.generate_spline_code(activation)
            spline_codes.append(spline_code)
        
        logger.info(f"Generated {len(spline_codes)} spline codes")
        
        # Step 2: Cluster spline codes for semantic regions
        spline_code_matrix = np.array(spline_codes)
        cluster_labels, clustering_info = self.semantic_analyzer.cluster_spline_codes(spline_code_matrix)
        
        logger.info(f"Identified {clustering_info['n_clusters']} semantic clusters")
        
        # Step 3: Create polytope regions
        polytope_regions = self._create_polytope_regions(
            activation_vectors, spline_codes, cluster_labels
        )
        
        # Step 4: Extract templates (simplified approach)
        templates = {}
        for i, region in enumerate(polytope_regions):
            if len(region['vertices']) > 0:
                # Use centroid as representative template
                centroid = np.mean(region['vertices'], axis=0)
                template = centroid / (np.linalg.norm(centroid) + 1e-8)
                templates[f'region_{region["cluster_id"]}'] = template
        
        # Step 5: Safety analysis (using high-frequency as safe examples)
        safety_analysis = self._perform_safety_analysis(records, activation_vectors)
        
        # Step 6: Compute comprehensive metrics
        analysis_metrics = self._compute_analysis_metrics(polytope_regions, clustering_info)
        
        return {
            'spline_codes': spline_codes,
            'polytope_regions': polytope_regions,
            'cluster_info': clustering_info,
            'templates': templates,
            'safety_analysis': safety_analysis,
            'metrics': analysis_metrics,
            'n_records': len(records),
            'n_regions': len(polytope_regions)
        }
    
    def _create_polytope_regions(self, activation_vectors: np.ndarray, 
                               spline_codes: List[np.ndarray],
                               cluster_labels: np.ndarray) -> List[Dict[str, Any]]:
        """Create polytope regions from clustering results."""
        regions = []
        
        for cluster_id in set(cluster_labels):
            if cluster_id == -1:  # Skip noise cluster
                continue
            
            # Get spline codes and activations for this cluster
            cluster_mask = cluster_labels == cluster_id
            cluster_splines = [spline_codes[i] for i in range(len(spline_codes)) if cluster_mask[i]]
            cluster_activations = activation_vectors[cluster_mask]
            
            if len(cluster_activations) == 0:
                continue
            
            # Create polytope region
            region = {
                'vertices': cluster_activations,
                'spline_codes': cluster_splines,
                'cluster_id': cluster_id,
                'semantic_coherence': self._compute_semantic_coherence(cluster_splines),
                'boundary_density': self._compute_boundary_density(cluster_activations, activation_vectors)
            }
            
            regions.append(region)
        
        return regions
    
    def _compute_semantic_coherence(self, spline_codes: List[np.ndarray]) -> float:
        """Compute semantic coherence for a group of spline codes."""
        if len(spline_codes) < 2:
            return 1.0
        
        similarities = []
        for i in range(len(spline_codes)):
            for j in range(i + 1, len(spline_codes)):
                # Compute similarity (1 - normalized Hamming distance)
                hamming_dist = np.sum(spline_codes[i] != spline_codes[j])
                similarity = 1.0 - hamming_dist / len(spline_codes[i])
                similarities.append(similarity)
        
        return np.mean(similarities) if similarities else 0.0
    
    def _compute_boundary_density(self, cluster_vertices: np.ndarray, 
                                all_points: np.ndarray) -> float:
        """Compute boundary density metric for polytope region."""
        if len(cluster_vertices) < 3:
            return 0.0
        
        try:
            # Compute convex hull of cluster
            hull = ConvexHull(cluster_vertices)
            hull_vertices = cluster_vertices[hull.vertices]
            
            # Compute distances from all points to hull boundary (approximation)
            distances = cdist(all_points, hull_vertices)
            min_distances = np.min(distances, axis=1)
            
            # Points near boundary (within 20th percentile of distances)
            threshold = np.percentile(min_distances, 20)
            near_boundary = np.sum(min_distances <= threshold)
            
            return near_boundary / len(all_points)
            
        except Exception:
            return 0.0
    
    def _perform_safety_analysis(self, records: List[Dict], 
                               activation_vectors: np.ndarray) -> Dict[str, Any]:
        """Perform safety polytope analysis using high-frequency activations as safe examples."""
        try:
            # Use high-frequency activations as "safe" examples
            high_freq_indices = [i for i, r in enumerate(records) if r['category'] == 'high']
            low_freq_indices = [i for i, r in enumerate(records) if r['category'] == 'low']
            
            if len(high_freq_indices) < 4 or len(low_freq_indices) < 4:
                return {'error': 'Insufficient data for safety analysis'}
            
            safe_activations = activation_vectors[high_freq_indices]
            
            # Create safety polytope (need to provide unsafe examples too)
            unsafe_activations = activation_vectors[low_freq_indices]
            safety_polytope = self.safety_manager.learn_safety_polytope(safe_activations, unsafe_activations)
            
            return {
                'safety_polytope': safety_polytope,
                'n_safe_examples': len(safe_activations),
                'analysis_successful': True
            }
            
        except Exception as e:
            logger.warning(f"Safety analysis failed: {e}")
            return {'error': str(e), 'analysis_successful': False}
    
    def _compute_analysis_metrics(self, regions: List[Dict[str, Any]], 
                                clustering_info: Dict) -> Dict[str, Any]:
        """Compute comprehensive analysis metrics."""
        if not regions:
            return {'error': 'No polytope regions found'}
        
        coherence_scores = [r['semantic_coherence'] for r in regions]
        boundary_densities = [r['boundary_density'] for r in regions]
        region_sizes = [len(r['vertices']) for r in regions]
        
        return {
            'semantic_coherence': {
                'mean': np.mean(coherence_scores),
                'std': np.std(coherence_scores),
                'min': np.min(coherence_scores),
                'max': np.max(coherence_scores)
            },
            'boundary_density': {
                'mean': np.mean(boundary_densities),
                'std': np.std(boundary_densities),
                'min': np.min(boundary_densities),
                'max': np.max(boundary_densities)
            },
            'region_statistics': {
                'n_regions': len(regions),
                'mean_region_size': np.mean(region_sizes),
                'total_clustered_points': sum(region_sizes),
                'clustering_efficiency': clustering_info['clustering_efficiency']
            }
        }

    def analyze_records(self, records: List[Dict]) -> Dict[str, Any]:
        """
        Main analysis function - analyze activation records for frequency differences.
        
        Args:
            records: List of activation records with keys:
                    - activation_vector: np.ndarray (required)
                    - category: str ('high' or 'low') (required)
                    - sparsity: float (required)
                    - activation_norm: float (required)
                    - n_active_neurons: int (required)
                    - checkpoint_step: int (optional)
                    - layer: int (optional)
                    - ngram: str (optional)
                    
        Returns:
            Complete analysis results
        """
        logger.info(f"Starting analysis of {len(records)} activation records")
        
        # Validate records
        validated_records = self.validate_records(records)
        
        if len(validated_records) < 4:
            raise ValueError("Need at least 4 valid records for analysis")
        
        # Run advanced polytope structure analysis
        polytope_analysis = self.analyze_polytope_structure(validated_records)
        
        # Run comparison across checkpoints (backward compatibility)
        analysis_results = self.compare_across_checkpoints(validated_records)
        
        # Create visualizations
        figures = self.create_visualizations(analysis_results)
        
        # Generate summary
        summary = self._generate_summary(analysis_results, validated_records)
        
        return {
            'polytope_analysis': polytope_analysis,
            'analysis_results': analysis_results,
            'figures': figures,
            'summary': summary,
            'n_records_processed': len(validated_records),
            'n_records_input': len(records)
        }
    
    def _generate_summary(self, analysis_results: Dict[str, Any], records: List[Dict]) -> Dict[str, Any]:
        """Generate analysis summary."""
        # Category-based statistics (using existing 'category' field)
        high_freq_records = [r for r in records if r['category'] == 'high']
        low_freq_records = [r for r in records if r['category'] == 'low']
        
        # Compute group statistics if both groups exist
        if high_freq_records and low_freq_records:
            high_sparsity = np.mean([r['sparsity'] for r in high_freq_records])
            low_sparsity = np.mean([r['sparsity'] for r in low_freq_records])
            
            high_norm = np.mean([r['activation_norm'] for r in high_freq_records])
            low_norm = np.mean([r['activation_norm'] for r in low_freq_records])
            
            high_active_neurons = np.mean([r['n_active_neurons'] for r in high_freq_records])
            low_active_neurons = np.mean([r['n_active_neurons'] for r in low_freq_records])
        else:
            high_sparsity = low_sparsity = 0.0
            high_norm = low_norm = 0.0
            high_active_neurons = low_active_neurons = 0.0
        
        return {
            'total_records': len(records),
            'n_checkpoints': len(set(r['checkpoint_step'] for r in records)),
            'n_layers': len(set(r['layer'] for r in records)),
            'category_distribution': {
                'high_frequency': len(high_freq_records),
                'low_frequency': len(low_freq_records)
            },
            'high_frequency_group': {
                'n_records': len(high_freq_records),
                'mean_sparsity': high_sparsity,
                'mean_norm': high_norm,
                'mean_active_neurons': high_active_neurons
            },
            'low_frequency_group': {
                'n_records': len(low_freq_records),
                'mean_sparsity': low_sparsity,
                'mean_norm': low_norm,
                'mean_active_neurons': low_active_neurons
            },
            'key_findings': {
                'sparsity_difference': low_sparsity - high_sparsity,
                'norm_ratio': high_norm / low_norm if low_norm > 0 else 1.0,
                'active_neurons_difference': high_active_neurons - low_active_neurons
            }
        }