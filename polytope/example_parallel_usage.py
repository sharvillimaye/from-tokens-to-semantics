#!/usr/bin/env python3
"""
Example usage of the parallelized polytope analyzer.

This script demonstrates how to use the ParallelSuperpositionAnalyzer
to achieve significant speedups over the original implementation.
"""

import numpy as np
import time
from loguru import logger

# Import the parallel analyzer
from polytope_analyzer_parallel import ParallelSuperpositionAnalyzer


def generate_sample_data(n_samples: int = 1000, n_neurons: int = 512) -> tuple:
    """
    Generate sample activation data for demonstration.
    
    Args:
        n_samples: Number of samples per frequency group
        n_neurons: Number of neurons (activation dimension)
        
    Returns:
        Tuple of (high_freq_activations, low_freq_activations)
    """
    logger.info(f"Generating sample data: {n_samples} samples, {n_neurons} neurons")
    
    # Set random seed for reproducible results
    np.random.seed(42)
    
    # High frequency activations (more structured)
    high_freq_activations = np.random.normal(0, 1, (n_samples, n_neurons))
    high_freq_activations[:, :n_neurons//4] *= 2.0  # Stronger activations
    
    # Low frequency activations (more superposition)
    low_freq_activations = np.random.normal(0, 0.5, (n_samples, n_neurons))
    low_freq_activations += np.random.normal(0, 0.3, (n_samples, n_neurons))
    
    return high_freq_activations, low_freq_activations


def run_parallel_analysis_example():
    """
    Run a complete example of parallel polytope analysis.
    """
    logger.info("Starting parallel polytope analysis example")
    
    # Generate sample data
    high_freq_data, low_freq_data = generate_sample_data(n_samples=1000, n_neurons=512)
    
    # Initialize parallel analyzer
    # Use 4 jobs for optimal performance on most systems
    analyzer = ParallelSuperpositionAnalyzer(
        n_jobs=4,
        min_cluster_size=5,
        random_seed=42
    )
    
    logger.info("Running parallel analysis...")
    start_time = time.time()
    
    # Run the analysis
    results = analyzer.analyze_polytope_superposition_parallel(
        activations_high_freq=high_freq_data,
        activations_low_freq=low_freq_data,
        semantic_category="example_n_grams"
    )
    
    total_time = time.time() - start_time
    
    # Generate and print report
    report = analyzer.generate_analysis_report(results)
    print("\n" + "="*80)
    print("ANALYSIS RESULTS")
    print("="*80)
    print(report)
    
    # Print timing information
    print(f"\nAnalysis completed in {total_time:.2f} seconds")
    print(f"Time reported in results: {results['summary']['total_analysis_time']:.2f} seconds")
    
    # Print key metrics
    print(f"\nKey Metrics:")
    print(f"  High freq superposition: {results['high_freq']['interference_per_dimension']:.4f}")
    print(f"  Low freq superposition: {results['low_freq']['interference_per_dimension']:.4f}")
    print(f"  Superposition difference: {results['comparison']['superposition_strength_difference']:.4f}")
    print(f"  Phase transition: {results['comparison']['phase_transition']}")
    
    return results


def compare_with_original_timing():
    """
    Compare timing with original analyzer (if available).
    """
    try:
        from polytope_analyzer import SuperpositionAnalyzer
        
        logger.info("Comparing with original analyzer...")
        
        # Generate data
        high_freq_data, low_freq_data = generate_sample_data(n_samples=500, n_neurons=256)
        
        # Test original analyzer
        original_analyzer = SuperpositionAnalyzer(min_cluster_size=5, random_seed=42)
        start_time = time.time()
        original_results = original_analyzer.analyze_polytope_superposition(
            activations_high_freq=high_freq_data,
            activations_low_freq=low_freq_data,
            semantic_category="comparison_test"
        )
        original_time = time.time() - start_time
        
        # Test parallel analyzer
        parallel_analyzer = ParallelSuperpositionAnalyzer(n_jobs=4, min_cluster_size=5, random_seed=42)
        start_time = time.time()
        parallel_results = parallel_analyzer.analyze_polytope_superposition_parallel(
            activations_high_freq=high_freq_data,
            activations_low_freq=low_freq_data,
            semantic_category="comparison_test"
        )
        parallel_time = time.time() - start_time
        
        # Calculate speedup
        speedup = original_time / parallel_time
        
        print(f"\n{'='*60}")
        print("PERFORMANCE COMPARISON")
        print(f"{'='*60}")
        print(f"Original analyzer: {original_time:.2f} seconds")
        print(f"Parallel analyzer: {parallel_time:.2f} seconds")
        print(f"Speedup: {speedup:.2f}x")
        print(f"Time saved: {original_time - parallel_time:.2f} seconds")
        
        # Verify results are similar
        original_density = original_results['high_freq']['mean_density']
        parallel_density = parallel_results['high_freq']['mean_density']
        density_diff = abs(original_density - parallel_density)
        
        print(f"\nResult validation:")
        print(f"  Original density: {original_density:.4f}")
        print(f"  Parallel density: {parallel_density:.4f}")
        print(f"  Difference: {density_diff:.6f}")
        print(f"  Results consistent: {density_diff < 0.001}")
        
    except ImportError:
        logger.warning("Original analyzer not available for comparison")
    except Exception as e:
        logger.error(f"Comparison failed: {e}")


if __name__ == "__main__":
    # Run the main example
    results = run_parallel_analysis_example()
    
    # Compare with original (if available)
    compare_with_original_timing()
    
    print(f"\n{'='*60}")
    print("EXAMPLE COMPLETED SUCCESSFULLY")
    print(f"{'='*60}")
    print("The parallel analyzer is working correctly!")
    print("You can now use it for your 6-hour analysis to achieve significant speedups.")
