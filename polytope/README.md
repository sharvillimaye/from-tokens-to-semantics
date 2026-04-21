# Polytope Superposition Analysis Pipeline (Anonymized)

A comprehensive research pipeline for analyzing superposition phenomena in Large Language Models (LLMs) through polytope density analysis and spline code extraction.

## Overview

This pipeline implements mechanistic interpretability techniques to study how neural networks represent multiple concepts within the same activation space (superposition). It focuses on analyzing the polytope structure of neural activations and how it evolves during training.

### Key Research Contributions

- **Spline Code Analysis**: Extract binary patterns from pre-activation vectors to identify polytope regions
- **Layer-wise Analysis**: Track superposition evolution across network depth
- **Frequency-based Comparison**: Compare high vs low frequency n-gram representations
- **Training Evolution**: Monitor polytope changes across training checkpoints
- **Publication-ready Visualizations**: Generate NeurIPS/ICML quality figures in PDF format

## Repository Structure

```
team-aasa/
├── polytope/                      # Core polytope analysis modules
│   ├── checkpoint_analysis.py     # Extract activations from model checkpoints
│   ├── refined_polytope_analyzer.py # Main analysis pipeline
│   ├── visualization_utils.py     # Publication-quality visualizations
│   ├── dataset_from_index.py      # N-gram dataset creation from tokengrams
│   └── ngram_dataset.py          # Dataset utilities

```

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended for model analysis)
- ~50GB free disk space for model checkpoints and cache

### Dependencies

```bash
# Core dependencies
pip install torch numpy pandas matplotlib seaborn
pip install transformers nnsight tokengrams
pip install loguru tqdm pathlib

# For visualization
pip install matplotlib scipy scikit-learn

# For notebook support
pip install jupyter ipykernel
```

### Memory Requirements

- **Small models (70M-410M)**: 8-16GB RAM
- **Medium models (1B-2.8B)**: 16-32GB RAM  
- **Large models (6.9B+)**: 32GB+ RAM recommended

## Quick Start

### 1. Basic Polytope Analysis

```python
from polytope.refined_polytope_analyzer import run_polytope_analysis_pipeline

# Run analysis on pre-computed activation records
results_dir = run_polytope_analysis_pipeline(
    checkpoint_file="cache/checkpoint_analysis_results.pkl",
    output_dir="results/polytope_analysis"
)
```

### 2. Extract Activations from Model Checkpoints

```python
from polytope.checkpoint_analysis import run_checkpoint_analysis_pipeline

# Extract activations from Pythia model checkpoints
results_file = run_checkpoint_analysis_pipeline(
    model_name="EleutherAI/pythia-6.9b",
    checkpoints=["1000", "5000", "10000", "50000", "143000"],
    dataset_path="polytope/dataset.json",
    output_dir="cache/checkpoint_analysis"
)
```

### 3. Create Publication Figures

```python
from polytope.visualization_utils import create_visualizations_from_json

# Generate PDF figures for papers
create_visualizations_from_json(
    json_path="results/polytope_analysis_results.json",
    output_dir="neurips-2026/figures/main",
    output_format="pdf"  # or "png"
)
```

## Core Components

### 1. Checkpoint Analysis (`checkpoint_analysis.py`)

Extracts neural activations from model checkpoints at specific token positions.

**Key Features:**
- Supports Pythia, GPT-2, LLaMA model families
- Extracts both pre and post-activation vectors
- Automatic layer selection for different model sizes
- Memory-efficient batch processing
- Network storage for large-scale analyses

**Usage:**
```python
from polytope.checkpoint_analysis import extract_activations_from_dataset

records = extract_activations_from_dataset(
    model_name="EleutherAI/pythia-6.9b",
    checkpoints=["1000", "10000", "50000"],
    dataset=dataset,
    target_layers=[1, 8, 16, 24, 31],  # Auto-selected if None
    batch_size=32
)
```

### 2. Polytope Analyzer (`refined_polytope_analyzer.py`)

Core analysis engine for computing polytope metrics and superposition quantification.

**Key Metrics:**
- **Polytope Density**: Hamming distance / Euclidean distance ratio
- **Participation Ratio**: Effective dimensionality of representations
- **Pattern Reuse**: Frequency of shared polytope patterns
- **Interference Patterns**: Cross-frequency group interactions
- **N-gram Mapping**: Polysemantic polytope analysis

**Usage:**
```python
from polytope.refined_polytope_analyzer import RefinedPolytopeAnalyzer

analyzer = RefinedPolytopeAnalyzer()
results = analyzer.run_full_analysis(
    checkpoint_path="cache/activations.pkl",
    output_dir="analysis_results"
)
```

### 3. Visualization Utils (`visualization_utils.py`)

Creates publication-quality figures optimized for academic papers.

**Features:**
- PDF output with proper font embedding for Overleaf
- Layer-wise heatmaps and evolution plots
- Cross-layer comparison visualizations
- Automatic LaTeX code generation
- Color-blind friendly palettes

### 4. N-gram Processing (`ngrams/`)

Handles n-gram frequency analysis and dataset creation.

**Components:**
- `interface.py`: Main n-gram processing interface
- `backends/`: Storage backends (in-memory, tokengrams)
- `parsers/`: Model-specific checkpoint cutoff parsers

## Analysis Pipeline

### Step 1: Prepare N-gram Dataset

```python
# Create dataset with high/low frequency n-grams
from polytope.ngram_dataset import create_ngram_polytope_dataset

dataset = create_ngram_polytope_dataset(
    high_freq_phrases=["the cat", "New York", "machine learning"],
    low_freq_phrases=["quantum entanglement", "archaeological evidence"],
    template="The capital of {country} is {capital}",
    model_name="EleutherAI/pythia-6.9b"
)
```

### Step 2: Extract Activations

```python
# Extract activations across training checkpoints
from polytope.checkpoint_analysis import run_checkpoint_analysis_pipeline

checkpoint_results = run_checkpoint_analysis_pipeline(
    model_name="EleutherAI/pythia-6.9b",
    checkpoints=["1000", "10000", "50000", "143000"],
    dataset_path="dataset.json",
    target_layers=None,  # Auto-select optimal layers
    batch_size=16
)
```

### Step 3: Analyze Polytopes

```python
# Run polytope analysis
from polytope.refined_polytope_analyzer import run_polytope_analysis_pipeline

results_dir = run_polytope_analysis_pipeline(
    checkpoint_file=checkpoint_results,
    output_dir="polytope_results"
)
```

### Step 4: Generate Visualizations

```python
# Create publication figures
from polytope.visualization_utils import create_visualizations_from_json

create_visualizations_from_json(
    json_path=f"{results_dir}/polytope_analysis_results.json",
    output_dir="neurips-2026/figures/main",
    output_format="pdf"
)
```

## Configuration Options

### Model Support

| Model Family | Layers Path | Activation Function | Status |
|--------------|-------------|-------------------|---------|
| Pythia | `gpt_neox.layers` | GELU | ✅ Full Support |
| GPT-2 | `transformer.h` | GELU | ✅ Full Support |
| LLaMA/Mistral | `model.layers` | SiLU | ✅ Beta Support |

### Layer Selection

The pipeline automatically selects optimal layers based on model size:

- **Small models (≤12 layers)**: Early, middle, late layers
- **Medium models (13-24 layers)**: Strategic sampling across depth
- **Large models (25+ layers)**: Focus on transition points (early: 1,2,4,6; middle: 16; late: 24,28,31)

### Memory Management

For large-scale analysis:

```python
# Use network storage for checkpoint progress
records = extract_activations_from_dataset_with_network_storage(
    model_name="EleutherAI/pythia-6.9b",
    checkpoints=checkpoints,
    dataset=dataset,
    network_storage_path="/shared/polytope_cache",
    resume=True  # Resume from previous runs
)
```

## Research Applications

### 1. Superposition Evolution Study

Track how superposition emerges during training:

```python
# Analyze progression from step 1 to 143000
checkpoints = [str(2**i) for i in range(10)]  # 1, 2, 4, ..., 512
checkpoints.extend([str(i) for i in range(1000, 144000, 1000)])
```

### 2. Layer-wise Mechanistic Analysis

Compare polytope structure across network depth:

```python
analyzer = RefinedPolytopeAnalyzer()
results = analyzer.analyze_checkpoint(records, checkpoint_step="50000", 
                                    use_layer_wise_analysis=True)
```

### 3. Frequency-based Representation Study

Analyze how frequency affects neural representations:

```python
# Create datasets with controlled frequency differences
high_freq = ["the", "and", "of", "to", "a"]  # Top 5 most frequent
low_freq = ["quixotic", "ephemeral", "serendipity"]  # Rare words
```

## Performance Optimization

### GPU Memory Management

```python
# Batch size recommendations by GPU memory
GPU_MEMORY_BATCH_SIZES = {
    "8GB": 4,
    "16GB": 8, 
    "24GB": 16,
    "40GB": 32,
    "80GB": 64
}
```

### Parallel Processing

```python
# Use multiprocessing for checkpoint analysis
analyzer.run_full_analysis(
    checkpoint_path="data.pkl",
    use_multiprocessing=True,
    max_workers=4  # Adjust based on available CPU/memory
)
```

## Output Formats

### Analysis Results

- `polytope_analysis_results.json`: Complete analysis results
- `analysis_summary.txt`: Human-readable summary report
- `checkpoint_progress/`: Individual checkpoint files for resume capability

### Visualizations

- **Heatmaps**: Layer × Checkpoint polytope density/participation ratio
- **Evolution Plots**: Metric changes over training
- **Cross-layer Analysis**: Comparative layer analysis
- **Small Multiples**: Per-layer time series

### Publication Support

- PDF figures with proper font embedding for Overleaf
- LaTeX code generation for figure inclusion
- Color-blind friendly palettes
- High-resolution output (300 DPI)

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```python
   # Reduce batch size
   batch_size = 4  # or lower
   
   # Use gradient checkpointing
   torch.cuda.empty_cache()
   ```

2. **Model Loading Errors**
   ```python
   # Clear HuggingFace cache
   from polytope.checkpoint_analysis import clear_hf_cache_for_revision
   clear_hf_cache_for_revision(model_name, revision)
   ```

3. **Missing Spline Codes**
   ```
   WARNING: No spline codes available, using binary patterns only
   ```
   - This is expected for some model architectures
   - Analysis will fall back to CETT-based binary patterns

### Performance Tips

- Use SSD storage for faster I/O
- Enable mixed precision (FP16) for larger batch sizes
- Use network storage for multi-node analysis
- Monitor GPU utilization with `nvidia-smi`

## Citation

To preserve anonymity for double-blind review, we omit citation and repository links here. The camera-ready will include full references.

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Follow the coding standards in `OPTIMIZATION_GUIDE.md`
4. Add tests for new functionality
5. Submit a pull request

## License

MIT License - see `LICENSE` file for details.


---

**Research Focus**: This pipeline is designed for mechanistic interpretability research, specifically studying superposition phenomena in neural language models. It prioritizes research reproducibility and publication-quality analysis over production deployment.
