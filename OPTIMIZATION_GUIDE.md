# N-Gram Mining Optimization Guide

This guide provides comprehensive strategies for optimizing n-gram mining performance, including GPU acceleration, parallelization, and memory management.

## 🚀 Performance Optimizations

### 1. GPU Acceleration

The optimized miner automatically detects and uses GPU acceleration when available:

```python
from ngrams.ngram_frequency import FastNgramMiner

# GPU-accelerated miner (automatic detection)
miner = FastNgramMiner(
    use_gpu=True,  # Automatically uses CUDA/MPS if available
    chunk_tokens=16_000_000,  # Optimized for GPU memory
    n_workers=4  # Fewer workers when using GPU
)
```

**GPU Benefits:**
- **10-50x speedup** for pattern matching operations
- **Parallel tensor operations** using PyTorch
- **Memory-efficient** sliding window operations
- **Automatic fallback** to CPU if GPU fails

### 2. Memory Management

#### Chunk Size Optimization
```python
# For 8GB GPU memory
miner = FastNgramMiner(
    chunk_tokens=8_000_000,  # ~32MB chunks
    max_memory_gb=6.0  # Leave 2GB for system
)

# For 24GB GPU memory  
miner = FastNgramMiner(
    chunk_tokens=32_000_000,  # ~128MB chunks
    max_memory_gb=20.0  # Leave 4GB for system
)
```

#### Memory-Efficient Processing
```python
from ngrams.gpu_optimizations import MemoryOptimizedProcessor

processor = MemoryOptimizedProcessor(
    max_memory_gb=8.0,
    chunk_overlap=1000  # Overlap to avoid missing patterns
)
```

### 3. Parallelization Strategies

#### Multi-Level Parallelization
```python
# CPU-only mode: Use multiple processes
miner = FastNgramMiner(
    use_gpu=False,
    n_workers=mp.cpu_count() - 1,  # Use all CPU cores
    chunk_tokens=32_000_000
)

# GPU mode: Use threads for I/O, GPU for computation
miner = FastNgramMiner(
    use_gpu=True,
    n_workers=4,  # Fewer workers, GPU handles computation
    chunk_tokens=16_000_000
)
```

#### Batch Processing
```python
# Process multiple shards in batches
results = miner.mine_and_count(
    shard_paths=shard_paths,
    probe_phrases=phrases,
    batch_size=4  # Process 4 chunks simultaneously
)
```

## 🔧 Advanced Optimizations

### 1. Mixed Precision (FP16)
```python
from ngrams.gpu_optimizations import GPUOptimizedNgramMiner

gpu_miner = GPUOptimizedNgramMiner(
    use_mixed_precision=True,  # Use FP16 for memory efficiency
    batch_size=1024,
    memory_fraction=0.8
)
```

### 2. Pattern Bank Optimization
```python
from ngrams.gpu_optimizations import create_memory_efficient_banks

# Limit patterns per bank to reduce memory usage
banks = create_memory_efficient_banks(
    phrases=probe_phrases,
    tokenizer=tokenizer,
    max_patterns_per_bank=10000
)
```

### 3. Dynamic Chunk Sizing
```python
from ngrams.gpu_optimizations import optimize_chunk_size

# Calculate optimal chunk size based on GPU memory
optimal_size = optimize_chunk_size(
    gpu_memory_gb=8.0,
    pattern_length=5,
    num_patterns=1000
)
```

## 📊 Performance Benchmarking

### Benchmark GPU vs CPU
```python
from ngrams.gpu_optimizations import benchmark_gpu_vs_cpu

# Create test data
test_data = np.random.randint(0, 50000, size=1000000, dtype=np.uint16)
phrases = ["test phrase", "another phrase", "third phrase"]

# Run benchmark
results = benchmark_gpu_vs_cpu(phrases, test_data, tokenizer, num_runs=5)

print(f"CPU mean time: {results['cpu']['mean_time']:.3f}s")
print(f"GPU mean time: {results['gpu']['mean_time']:.3f}s")
print(f"Speedup: {results['speedup']:.1f}x")
```

### Expected Performance Gains
- **GPU vs CPU**: 10-50x speedup for pattern matching
- **Optimized chunking**: 2-5x memory efficiency
- **Mixed precision**: 1.5-2x memory reduction
- **Parallel processing**: Linear scaling with CPU cores

## 🛠️ Troubleshooting

### 1. BrokenProcessPool Error

**Cause**: Worker processes crashing due to memory issues or GPU conflicts.

**Solutions**:
```python
# Reduce chunk size and workers
miner = FastNgramMiner(
    chunk_tokens=4_000_000,  # Smaller chunks
    n_workers=2,  # Fewer workers
    use_gpu=True
)

# Use conservative memory settings
miner = FastNgramMiner(
    max_memory_gb=4.0,  # Limit memory usage
    chunk_tokens=2_000_000
)
```

### 2. GPU Out of Memory

**Solutions**:
```python
# Reduce batch size and chunk size
gpu_miner = GPUOptimizedNgramMiner(
    batch_size=512,  # Smaller batches
    memory_fraction=0.6  # Use less GPU memory
)

# Enable mixed precision
gpu_miner = GPUOptimizedNgramMiner(
    use_mixed_precision=True,
    batch_size=1024
)
```

### 3. Slow Performance

**Diagnostics**:
```python
# Check device utilization
import torch
print(f"GPU available: {torch.cuda.is_available()}")
print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

# Monitor memory usage
print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.1f}GB")
print(f"GPU memory cached: {torch.cuda.memory_reserved() / 1e9:.1f}GB")
```

## 🎯 Best Practices

### 1. Configuration Guidelines

#### For Small Datasets (< 1GB)
```python
miner = FastNgramMiner(
    chunk_tokens=8_000_000,
    n_workers=4,
    use_gpu=True
)
```

#### For Medium Datasets (1-10GB)
```python
miner = FastNgramMiner(
    chunk_tokens=16_000_000,
    n_workers=6,
    use_gpu=True,
    max_memory_gb=6.0
)
```

#### For Large Datasets (> 10GB)
```python
miner = FastNgramMiner(
    chunk_tokens=32_000_000,
    n_workers=8,
    use_gpu=True,
    max_memory_gb=12.0
)
```

### 2. Memory Management
- **Monitor GPU memory** during processing
- **Use smaller chunks** for large pattern banks
- **Enable mixed precision** for memory efficiency
- **Clean up memory** between batches

### 3. Error Handling
```python
try:
    results = miner.mine_and_count(shard_paths, phrases)
except Exception as e:
    logger.error(f"Mining failed: {e}")
    # Fall back to CPU-only mode
    miner.use_gpu = False
    results = miner.mine_and_count(shard_paths, phrases)
```

## 📈 Performance Monitoring

### Memory Usage Tracking
```python
import psutil
import torch

def monitor_resources():
    # CPU memory
    cpu_memory = psutil.virtual_memory()
    print(f"CPU memory: {cpu_memory.percent}% used")
    
    # GPU memory
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / 1e9
        print(f"GPU memory: {gpu_memory:.1f}GB allocated")
```

### Progress Tracking
```python
from tqdm import tqdm

# Add progress bars to long-running operations
with tqdm(total=len(shard_paths), desc="Processing shards") as pbar:
    for shard in shard_paths:
        # Process shard
        pbar.update(1)
```

## 🔍 Debugging Tips

### 1. Enable Detailed Logging
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Monitor specific operations
logger.debug(f"Processing chunk {chunk_id}/{total_chunks}")
logger.debug(f"GPU memory usage: {torch.cuda.memory_allocated() / 1e9:.1f}GB")
```

### 2. Test with Small Data
```python
# Start with small test data
test_phrases = ["test", "small", "data"]
test_shards = [small_test_shard]

results = miner.mine_and_count(test_shards, test_phrases)
print("Test completed successfully")
```

### 3. Profile Performance
```python
import cProfile
import pstats

# Profile the mining operation
profiler = cProfile.Profile()
profiler.enable()

results = miner.mine_and_count(shard_paths, phrases)

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)  # Top 20 functions
```

## 🚀 Advanced Techniques

### 1. Multi-GPU Processing
```python
# For systems with multiple GPUs
import torch.distributed as dist

# Initialize distributed processing
dist.init_process_group(backend='nccl')
```

### 2. Streaming Processing
```python
# Process data in streaming fashion
def stream_process(shard_paths, phrases):
    for shard in shard_paths:
        # Process one shard at a time
        results = miner.mine_and_count([shard], phrases)
        yield results
```

### 3. Caching Results
```python
import pickle
from pathlib import Path

# Cache pattern banks
cache_path = Path("pattern_banks_cache.pkl")
if cache_path.exists():
    with open(cache_path, 'rb') as f:
        banks = pickle.load(f)
else:
    banks = build_pattern_banks_gpu(tokenizer, phrases)
    with open(cache_path, 'wb') as f:
        pickle.dump(banks, f)
```

This optimization guide should help you achieve maximum performance for your n-gram mining operations. Start with the basic optimizations and gradually implement advanced techniques based on your specific requirements and hardware constraints.




