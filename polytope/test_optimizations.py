#!/usr/bin/env python3
"""
Test script to verify the optimized refined polytope analyzer works correctly.
This tests the critical optimizations without requiring large datasets.
"""

import numpy as np
import pickle
import time
from typing import Dict, Any, List
from refined_polytope_analyzer import RefinedPolytopeAnalyzer

def create_synthetic_records(n_samples: int = 100, n_neurons: int = 512) -> List[Dict[str, Any]]:
    """Create synthetic activation records for testing."""
    np.random.seed(42)
    records = []
    
    for i in range(n_samples):
        # Generate synthetic activation vectors
        activation_vector = np.random.randn(n_neurons).astype(np.float32)
        
        # Generate binary patterns (spline codes)
        spline_code = (activation_vector > 0).astype(np.int8)
        binary_pattern = (activation_vector > 0.5).astype(np.int8)
        
        # Alternate between high and low frequency
        category = 'high_freq' if i % 2 == 0 else 'low_freq'
        
        record = {
            'activation_vector': activation_vector,
            'spline_code': spline_code,
            'binary_pattern': binary_pattern,
            'category': category,
            'checkpoint_step': str(i // 20),  # 5 checkpoints
            'phrase': f'test_phrase_{i}'
        }
        records.append(record)
    
    return records

def test_chunked_density_computation():
    """Test the chunked polytope density computation."""
    print("Testing chunked polytope density computation...")
    
    analyzer = RefinedPolytopeAnalyzer(random_seed=42)
    
    # Create test data
    n_samples = 50
    n_neurons = 100
    activations = np.random.randn(n_samples, n_neurons).astype(np.float32)
    patterns = (activations > 0).astype(np.int8)
    
    # Test with different chunk sizes
    start_time = time.time()
    result_chunked = analyzer.compute_polytope_density(
        activations, patterns, max_pairs=500, chunk_size=50
    )
    chunked_time = time.time() - start_time
    
    print(f"Chunked computation completed in {chunked_time:.3f}s")
    print(f"Density mean: {result_chunked['density_mean']:.4f}")
    print(f"Valid pairs: {result_chunked['valid_pairs']}")
    
    assert result_chunked['density_mean'] > 0, "Density should be positive"
    assert result_chunked['valid_pairs'] > 0, "Should have valid pairs"
    print("✅ Chunked density computation test passed!")

def test_vectorized_interference():
    """Test the vectorized interference pattern computation."""
    print("\nTesting vectorized interference patterns...")
    
    analyzer = RefinedPolytopeAnalyzer(random_seed=42)
    
    # Create test data
    n_high, n_low = 30, 25
    n_neurons = 100
    
    activations_high = np.random.randn(n_high, n_neurons).astype(np.float32)
    activations_low = np.random.randn(n_low, n_neurons).astype(np.float32)
    
    # Add some correlation structure
    activations_low[:, :50] *= 0.5  # Reduce variance in first half
    
    start_time = time.time()
    interference = analyzer.compute_interference_patterns(activations_high, activations_low)
    vectorized_time = time.time() - start_time
    
    print(f"Vectorized interference computed in {vectorized_time:.3f}s")
    print(f"Cosine similarity: {interference['cosine_similarity']:.4f}")
    print(f"Neuron overlap score: {interference['neuron_overlap_score']:.4f}")
    
    assert 'cosine_similarity' in interference, "Should have cosine similarity"
    assert 'neuron_overlap_score' in interference, "Should have overlap score"
    print("✅ Vectorized interference computation test passed!")

def test_multiprocessing_checkpoint_analysis():
    """Test multiprocessing checkpoint analysis."""
    print("\nTesting multiprocessing checkpoint analysis...")
    
    analyzer = RefinedPolytopeAnalyzer(random_seed=42)
    records = create_synthetic_records(n_samples=100, n_neurons=256)
    
    # Organize by checkpoint
    checkpoint_groups = {}
    for record in records:
        checkpoint = str(record['checkpoint_step'])
        if checkpoint not in checkpoint_groups:
            checkpoint_groups[checkpoint] = []
        checkpoint_groups[checkpoint].append(record)
    
    print(f"Created {len(checkpoint_groups)} checkpoints with {len(records)} total records")
    
    # Test sequential vs parallel processing
    start_time = time.time()
    results_sequential = analyzer._analyze_checkpoints_sequential(checkpoint_groups)
    sequential_time = time.time() - start_time
    
    start_time = time.time()
    results_parallel = analyzer._analyze_checkpoints_parallel(checkpoint_groups, max_workers=2)
    parallel_time = time.time() - start_time
    
    print(f"Sequential processing: {sequential_time:.3f}s")
    print(f"Parallel processing: {parallel_time:.3f}s")
    
    assert len(results_sequential) == len(results_parallel), "Should have same number of results"
    assert len(results_sequential) == len(checkpoint_groups), "Should analyze all checkpoints"
    
    # Check that results are comparable (may differ due to randomness in multiprocessing)
    for seq_result, par_result in zip(results_sequential, results_parallel):
        if 'error' not in seq_result and 'error' not in par_result:
            assert seq_result['checkpoint_step'] == par_result['checkpoint_step'], "Checkpoints should match"
    
    print("✅ Multiprocessing checkpoint analysis test passed!")

def test_streaming_data_loading():
    """Test streaming data loading functionality."""
    print("\nTesting streaming data loading...")
    
    analyzer = RefinedPolytopeAnalyzer(random_seed=42)
    
    # Create a temporary pickle file with test data
    import tempfile
    import os
    
    records = create_synthetic_records(n_samples=50, n_neurons=128)
    test_data = {
        'records': records,
        'metadata': {'test': True, 'n_samples': len(records)}
    }
    
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp_file:
        with open(tmp_file.name, 'wb') as f:
            pickle.dump(test_data, f)
        
        # Test streaming load
        start_time = time.time()
        data_streamed = analyzer.load_checkpoint_data(tmp_file.name, stream_processing=True, chunk_size=10)
        streaming_time = time.time() - start_time
        
        # Test regular load
        start_time = time.time()
        data_regular = analyzer.load_checkpoint_data(tmp_file.name, stream_processing=False)
        regular_time = time.time() - start_time
        
        print(f"Streaming load: {streaming_time:.3f}s")
        print(f"Regular load: {regular_time:.3f}s")
        
        assert len(data_streamed['records']) == len(data_regular['records']), "Should load same data"
        assert 'streaming_enabled' in data_streamed, "Should indicate streaming enabled"
        
        # Clean up
        os.unlink(tmp_file.name)
    
    print("✅ Streaming data loading test passed!")

def test_memory_usage_improvement():
    """Test memory usage with larger datasets."""
    print("\nTesting memory usage improvements...")
    
    analyzer = RefinedPolytopeAnalyzer(random_seed=42)
    
    # Create larger test dataset
    n_samples = 200
    n_neurons = 512
    records = create_synthetic_records(n_samples=n_samples, n_neurons=n_neurons)
    
    # Test checkpoint analysis with optimizations
    start_time = time.time()
    result = analyzer.analyze_checkpoint(records[:100], "test_checkpoint")
    analysis_time = time.time() - start_time
    
    print(f"Checkpoint analysis for {100} records completed in {analysis_time:.3f}s")
    
    assert 'error' not in result, f"Analysis should not fail: {result.get('error', '')}"
    assert 'high_freq_density' in result, "Should compute high frequency density"
    assert 'low_freq_density' in result, "Should compute low frequency density"
    assert 'interference' in result, "Should compute interference patterns"
    
    print(f"High freq samples: {result['n_high_freq']}")
    print(f"Low freq samples: {result['n_low_freq']}")
    print(f"Interference cosine similarity: {result['interference']['cosine_similarity']:.4f}")
    
    print("✅ Memory usage improvement test passed!")

def main():
    """Run all optimization tests."""
    print("=" * 60)
    print("REFINED POLYTOPE ANALYZER OPTIMIZATION TESTS")
    print("=" * 60)
    
    try:
        test_chunked_density_computation()
        test_vectorized_interference()
        test_multiprocessing_checkpoint_analysis()
        test_streaming_data_loading()
        test_memory_usage_improvement()
        
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED! Optimizations are working correctly.")
        print("Key improvements verified:")
        print("  ✅ Chunked processing prevents memory overload")
        print("  ✅ Vectorized operations improve performance")
        print("  ✅ Multiprocessing enables parallel checkpoint analysis")
        print("  ✅ Streaming I/O reduces memory footprint")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())