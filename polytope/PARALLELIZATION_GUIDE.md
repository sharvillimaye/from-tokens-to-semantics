# Polytope Analyzer Parallelization Guide

## Overview

The original polytope analyzer took 6 hours to complete due to several computational bottlenecks. This guide explains the parallelization improvements and how to achieve significant speedups.

## Key Bottlenecks Identified

### 1. Sequential Processing
- **Problem**: High and low frequency groups processed sequentially
- **Solution**: Parallel processing using `ProcessPoolExecutor`

### 2. Inefficient Pair Generation
- **Problem**: Nested loops generating pairs with `while len(pairs_set) < target`
- **Solution**: Vectorized pair generation with efficient sampling

### 3. Sequential CETT Threshold Computation
- **Problem**: `thresholds = np.array([_cett_threshold(v) for v in activations])`
- **Solution**: Vectorized binary search implementation

### 4. Sequential Distance Calculations
- **Problem**: Loops computing distances between pairs
- **Solution**: Vectorized distance computations using NumPy operations

### 5. Redundant Computations
- **Problem**: Same calculations repeated for similar operations
- **Solution**: Optimized algorithms with reduced redundant work

## Performance Improvements

### Expected Speedups
- **Small datasets** (500-1000 samples): 2-4x speedup
- **Medium datasets** (1000-2000 samples): 3-6x speedup  
- **Large datasets** (2000+ samples): 4-8x speedup

### Memory Efficiency
- Batch processing to reduce memory footprint
- Optimized data structures
- Reduced redundant storage

## Usage

### Basic Usage

```python
from polytope.polytope_analyzer_parallel import ParallelSuperpositionAnalyzer

# Initialize with optimal settings for your system
analyzer = ParallelSuperpositionAnalyzer(
    n_jobs=4,  # Use 4 CPU cores (adjust based on your system)
    min_cluster_size=5,
    random_seed=42
)

# Run parallel analysis
results = analyzer.analyze_polytope_superposition_parallel(
    activations_high_freq=high_freq_data,
    activations_low_freq=low_freq_data,
    semantic_category="n_grams"
)

# Generate report
report = analyzer.generate_analysis_report(results)
print(report)
```

### Advanced Usage with Pre-computed Data

```python
# If you have pre-computed binary patterns or spline codes
results = analyzer.analyze_polytope_superposition_parallel(
    activations_high_freq=high_freq_data,
    activations_low_freq=low_freq_data,
    semantic_category="n_grams",
    binary_patterns_high=precomputed_high_patterns,
    binary_patterns_low=precomputed_low_patterns,
    spline_codes_high=precomputed_high_splines,
    spline_codes_low=precomputed_low_splines
)
```

## Configuration Options

### n_jobs Parameter
- **-1**: Use all available CPU cores
- **1**: Single-threaded (for debugging)
- **2-8**: Optimal for most systems
- **>8**: May not provide additional benefit due to overhead

### batch_size Parameter
- **Default**: 1000
- **Small datasets**: 500-1000
- **Large datasets**: 1000-2000
- **Memory-constrained**: 500

## Benchmarking

Run the benchmark script to test performance on your system:

```bash
cd polytope
python benchmark_parallel.py
```

This will test different dataset sizes and job configurations to find optimal settings.

## Migration from Original Analyzer

### Minimal Changes Required

```python
# Original code
from polytope.polytope_analyzer import SuperpositionAnalyzer
analyzer = SuperpositionAnalyzer(min_cluster_size=5, random_seed=42)
results = analyzer.analyze_polytope_superposition(high_data, low_data)

# Parallel code (minimal changes)
from polytope.polytope_analyzer_parallel import ParallelSuperpositionAnalyzer
analyzer = ParallelSuperpositionAnalyzer(n_jobs=4, min_cluster_size=5, random_seed=42)
results = analyzer.analyze_polytope_superposition_parallel(high_data, low_data)
```

### API Compatibility
- Same input/output format
- Same result structure
- Same report generation
- Additional timing information in results

## Optimization Tips

### 1. Choose Optimal n_jobs
```python
import multiprocessing as mp
optimal_jobs = min(8, mp.cpu_count())  # Don't exceed 8 for most cases
analyzer = ParallelSuperpositionAnalyzer(n_jobs=optimal_jobs)
```

### 2. Pre-compute Binary Patterns
If you're running multiple analyses, pre-compute binary patterns:

```python
# Compute once, reuse
thresholds_high = analyzer._vectorized_cett_threshold(high_freq_data)
binary_patterns_high = (np.abs(high_freq_data) > thresholds_high[:, None]).astype(int)

# Use in multiple analyses
results1 = analyzer.analyze_polytope_superposition_parallel(
    high_freq_data, low_freq_data,
    binary_patterns_high=binary_patterns_high
)
```

### 3. Memory Management
For very large datasets:

```python
# Reduce batch size for memory-constrained systems
analyzer = ParallelSuperpositionAnalyzer(n_jobs=4, batch_size=500)

# Or process in chunks
chunk_size = 1000
for i in range(0, len(large_dataset), chunk_size):
    chunk = large_dataset[i:i+chunk_size]
    # Process chunk
```

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Ensure all dependencies are installed
   pip install numpy scipy scikit-learn hdbscan pandas matplotlib seaborn
   ```

2. **Memory Issues**
   - Reduce `batch_size`
   - Reduce `n_jobs`
   - Process data in smaller chunks

3. **Performance Not Improving**
   - Check CPU usage during execution
   - Ensure `n_jobs` doesn't exceed CPU cores
   - Verify dataset size is large enough to benefit from parallelization

### Debug Mode
```python
# Single-threaded for debugging
analyzer = ParallelSuperpositionAnalyzer(n_jobs=1)
```

## Expected Results

### Time Savings
- **6-hour analysis**: Should complete in 1-2 hours
- **1-hour analysis**: Should complete in 10-20 minutes
- **Small analyses**: May see 2-4x speedup

### Resource Usage
- **CPU**: Will utilize all specified cores
- **Memory**: Slightly higher due to parallel processing
- **Disk**: Same as original (no additional I/O)

## Future Improvements

1. **GPU Acceleration**: For very large datasets
2. **Distributed Processing**: For multi-machine setups
3. **Streaming Processing**: For real-time analysis
4. **Caching**: For repeated analyses

## Support

For issues or questions:
1. Check the benchmark script for performance validation
2. Verify system resources (CPU cores, memory)
3. Test with smaller datasets first
4. Review the troubleshooting section above
