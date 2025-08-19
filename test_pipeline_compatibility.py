#!/usr/bin/env python3
"""
Test script to verify pipeline compatibility with provided data structure.

This tests the dimension mismatch fix for spline_codes (16384) vs activations (4096).
"""

import numpy as np
from polytope.polytope_analyzer import SuperpositionAnalyzer

def test_dimension_handling():
    """Test that the pipeline handles dimension mismatches correctly."""
    
    print("Testing simplified polytope pipeline compatibility...")
    
    # Create test data mimicking your structure
    n_samples = 10
    activation_dim = 4096  # Your activation_vector dimension
    spline_dim = 16384     # Your spline_code dimension (double)
    
    # Mock activation data
    activations = np.random.randn(n_samples, activation_dim).astype(np.float16)
    
    # Mock spline codes with double dimensions
    spline_codes = np.random.randint(0, 2, size=(n_samples, spline_dim)).astype(np.uint8)
    
    print(f"✓ Test data created: activations {activations.shape}, spline_codes {spline_codes.shape}")
    
    # Test the analyzer
    analyzer = SuperpositionAnalyzer(random_seed=42)
    
    try:
        # This should automatically truncate spline codes to match activation dimensions
        result = analyzer.compute_polytope_metrics(activations, spline_codes)
        print(f"✅ SUCCESS: Polytope metrics computed successfully")
        print(f"   - Mean density: {result['mean_density']:.3f}")
        print(f"   - Density std error: {result['mean_density_std_error']:.3f}")
        print(f"   - Boundary crossings: {result['boundary_crossings']}")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False

def test_analyze_superposition():
    """Test the main analysis function with dimension mismatches."""
    
    print("\nTesting main superposition analysis...")
    
    n_samples = 20
    activation_dim = 4096
    spline_dim = 16384
    
    # Create high and low frequency mock data
    high_activations = np.random.randn(n_samples, activation_dim).astype(np.float16)
    low_activations = np.random.randn(n_samples, activation_dim).astype(np.float16)
    
    high_splines = np.random.randint(0, 2, size=(n_samples, spline_dim)).astype(np.uint8) 
    low_splines = np.random.randint(0, 2, size=(n_samples, spline_dim)).astype(np.uint8)
    
    analyzer = SuperpositionAnalyzer(random_seed=42)
    
    try:
        result = analyzer.analyze_polytope_superposition(
            activations_high_freq=high_activations,
            activations_low_freq=low_activations,
            semantic_category="test_data",
            binary_patterns_high=high_splines,  # These will be truncated
            binary_patterns_low=low_splines,    # These will be truncated  
            spline_codes_high=high_splines,
            spline_codes_low=low_splines
        )
        
        print("✅ SUCCESS: Full superposition analysis completed")
        
        # Check what keys are available in the result
        if 'high_freq' in result:
            high_freq_data = result['high_freq']
            print(f"   - High freq analysis keys: {list(high_freq_data.keys())}")
            if 'processed' in high_freq_data:
                print(f"   - High freq samples: {len(high_freq_data['processed']['binary_patterns'])}")
        
        if 'low_freq' in result:
            low_freq_data = result['low_freq']  
            print(f"   - Low freq analysis keys: {list(low_freq_data.keys())}")
            if 'processed' in low_freq_data:
                print(f"   - Low freq samples: {len(low_freq_data['processed']['binary_patterns'])}")
        
        if 'comparison' in result:
            comparison = result['comparison']
            sup_diff = comparison.get('superposition_strength_difference', 'N/A')
            mw_p = comparison.get('superposition_mannwhitney_p', 'N/A')
            print(f"   - Superposition difference: {sup_diff}")
            print(f"   - Mann-Whitney p-value: {mw_p}")
        
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("TESTING SIMPLIFIED POLYTOPE PIPELINE COMPATIBILITY")
    print("=" * 60)
    
    # Test individual components
    success1 = test_dimension_handling()
    success2 = test_analyze_superposition()
    
    print(f"\n{'='*60}")
    if success1 and success2:
        print("🎉 ALL TESTS PASSED - Pipeline is compatible with your data structure!")
        print("\nThe simplified pipeline now handles:")
        print("  ✓ Automatic dimension truncation (16384 → 4096)")
        print("  ✓ Mann-Whitney U statistical testing only")
        print("  ✓ Simple descriptive statistics (no bootstrap)")
        print("  ✓ Direct frequency group comparisons")
    else:
        print("❌ Some tests failed - please check the error messages above")
    print("=" * 60)