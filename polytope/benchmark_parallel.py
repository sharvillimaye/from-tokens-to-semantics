#!/usr/bin/env python3
"""
Benchmark script to compare original vs parallelized polytope analyzer performance.
"""

import numpy as np
import time
from loguru import logger
import sys
import os

# Add the polytope directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from polytope_analyzer import SuperpositionAnalyzer
    from polytope_analyzer_parallel import ParallelSuperpositionAnalyzer
except ImportError as e:
    logger.error(f"Import error: {e}")
    logger.info("Make sure both polytope_analyzer.py and polytope_analyzer_parallel.py are in the same directory")
    sys.exit(1)


def generate_test_data(n_samples_high: int = 1000, n_samples_low: int = 1000, 
                      n_neurons: int = 512) -> tuple:
    """
    Generate synthetic test data for benchmarking.
    
    Args:
        n_samples_high: Number of high frequency samples
        n_samples_low: Number of low frequency samples  
        n_neurons: Number of neurons (activation dimension)
        
    Returns:
        Tuple of (high_freq_activations, low_freq_activations)
    """
    logger.info(f"Generating test data: {n_samples_high} high freq, {n_samples_low} low freq, {n_neurons} neurons")
    
    # Set random seed for reproducible results
    np.random.seed(42)
    
    # Generate high frequency activations (more structured, less superposition)
    high_freq_activations = np.random.normal(0, 1, (n_samples_high, n_neurons))
    # Add some structure to high frequency
    high_freq_activations[:, :n_neurons//4] *= 2.0  # Stronger activations in first quarter
    
    # Generate low frequency activations (more superposition, less structure)
    low_freq_activations = np.random.normal(0, 0.5, (n_samples_low, n_neurons))
    # Add more superposition to low frequency
    low_freq_activations += np.random.normal(0, 0.3, (n_samples_low, n_neurons))
    
    return high_freq_activations, low_freq_activations


def benchmark_original_analyzer(high_freq_data: np.ndarray, low_freq_data: np.ndarray) -> float:
    """
    Benchmark the original polytope analyzer.
    
    Args:
        high_freq_data: High frequency activation data
        low_freq_data: Low frequency activation data
        
    Returns:
        Execution time in seconds
    """
    logger.info("Running original analyzer benchmark...")
    
    analyzer = SuperpositionAnalyzer(min_cluster_size=5, random_seed=42)
    
    start_time = time.time()
    
    try:
        results = analyzer.analyze_polytope_superposition(
            activations_high_freq=high_freq_data,
            activations_low_freq=low_freq_data,
            semantic_category="benchmark_test"
        )
        execution_time = time.time() - start_time
        
        logger.info(f"Original analyzer completed in {execution_time:.2f} seconds")
        logger.info(f"Results summary: {results['summary']}")
        
        return execution_time
        
    except Exception as e:
        logger.error(f"Original analyzer failed: {e}")
        return float('inf')


def benchmark_parallel_analyzer(high_freq_data: np.ndarray, low_freq_data: np.ndarray, 
                               n_jobs: int = 4) -> float:
    """
    Benchmark the parallelized polytope analyzer.
    
    Args:
        high_freq_data: High frequency activation data
        low_freq_data: Low frequency activation data
        n_jobs: Number of parallel jobs
        
    Returns:
        Execution time in seconds
    """
    logger.info(f"Running parallel analyzer benchmark with {n_jobs} jobs...")
    
    analyzer = ParallelSuperpositionAnalyzer(n_jobs=n_jobs, min_cluster_size=5, random_seed=42)
    
    start_time = time.time()
    
    try:
        results = analyzer.analyze_polytope_superposition_parallel(
            activations_high_freq=high_freq_data,
            activations_low_freq=low_freq_data,
            semantic_category="benchmark_test"
        )
        execution_time = time.time() - start_time
        
        logger.info(f"Parallel analyzer completed in {execution_time:.2f} seconds")
        logger.info(f"Results summary: {results['summary']}")
        
        return execution_time
        
    except Exception as e:
        logger.error(f"Parallel analyzer failed: {e}")
        return float('inf')


def run_benchmark_suite():
    """
    Run comprehensive benchmark suite with different data sizes.
    """
    logger.info("Starting polytope analyzer benchmark suite")
    
    # Test configurations
    test_configs = [
        {"n_samples_high": 500, "n_samples_low": 500, "n_neurons": 256, "name": "Small"},
        {"n_samples_high": 1000, "n_samples_low": 1000, "n_neurons": 512, "name": "Medium"},
        {"n_samples_high": 2000, "n_samples_low": 2000, "n_neurons": 1024, "name": "Large"},
    ]
    
    # Job configurations to test
    job_configs = [1, 2, 4, 8]
    
    results = []
    
    for config in test_configs:
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing {config['name']} dataset: {config['n_samples_high']} samples, {config['n_neurons']} neurons")
        logger.info(f"{'='*60}")
        
        # Generate test data
        high_freq_data, low_freq_data = generate_test_data(
            config['n_samples_high'], config['n_samples_low'], config['n_neurons']
        )
        
        # Test original analyzer
        original_time = benchmark_original_analyzer(high_freq_data, low_freq_data)
        
        # Test parallel analyzer with different job counts
        parallel_times = {}
        for n_jobs in job_configs:
            if n_jobs <= mp.cpu_count():
                parallel_time = benchmark_parallel_analyzer(high_freq_data, low_freq_data, n_jobs)
                parallel_times[n_jobs] = parallel_time
                
                if original_time != float('inf') and parallel_time != float('inf'):
                    speedup = original_time / parallel_time
                    logger.info(f"Speedup with {n_jobs} jobs: {speedup:.2f}x")
        
        results.append({
            'config': config,
            'original_time': original_time,
            'parallel_times': parallel_times
        })
    
    # Print summary
    logger.info(f"\n{'='*80}")
    logger.info("BENCHMARK SUMMARY")
    logger.info(f"{'='*80}")
    
    for result in results:
        config = result['config']
        original_time = result['original_time']
        parallel_times = result['parallel_times']
        
        logger.info(f"\n{config['name']} Dataset Results:")
        logger.info(f"  Original analyzer: {original_time:.2f}s")
        
        for n_jobs, parallel_time in parallel_times.items():
            if original_time != float('inf') and parallel_time != float('inf'):
                speedup = original_time / parallel_time
                efficiency = (speedup / n_jobs) * 100
                logger.info(f"  Parallel ({n_jobs} jobs): {parallel_time:.2f}s (speedup: {speedup:.2f}x, efficiency: {efficiency:.1f}%)")
            else:
                logger.info(f"  Parallel ({n_jobs} jobs): Failed")


if __name__ == "__main__":
    import multiprocessing as mp
    
    logger.info(f"System info: {mp.cpu_count()} CPUs available")
    
    try:
        run_benchmark_suite()
    except KeyboardInterrupt:
        logger.info("Benchmark interrupted by user")
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
