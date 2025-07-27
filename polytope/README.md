# Refactored Polytope Analysis Pipeline

A comprehensive system for analyzing the evolution of polysemanticity in Large Language Models through geometric polytope analysis of neural activations.

## Overview

This refactored pipeline provides a robust framework for studying how neural representations evolve during training by:

1. **N-gram Dataset Generation**: Creating stratified datasets of n-grams across frequency bins
2. **Activation Extraction**: Extracting position-based activations from model checkpoints
3. **Polytope Analysis**: Computing geometric metrics of activation polytopes
4. **Evolution Tracking**: Analyzing how polytope complexity changes over training

## Key Features

✅ **Robust N-gram Processing**: Improved n-gram extraction with multiple matching strategies  
✅ **Efficient Data Structures**: Organized activation records with indexed access  
✅ **Comprehensive Polytope Metrics**: Volume, complexity, stability, and dimensional analysis  
✅ **Caching System**: Persistent caching for both n-grams and activations  
✅ **Unified Pipeline**: Complete analysis workflow with minimal setup  
✅ **Visualization Tools**: Built-in plotting for evolution and comparison analysis  
✅ **Cross-platform**: Uses `pathlib` for consistent file operations  

## Project Structure

```
polytope/
├── ngram_dataset.py              # Improved n-gram dataset builder
├── checkpoint_analysis.py        # Enhanced activation extraction
├── polytope_metrics.py          # Comprehensive polytope analysis
├── unified_polytope_analysis.py # Complete pipeline integration
├── example_usage.py             # Usage examples and demonstrations
└── README.md                    # This file
```

## Quick Start

### 1. Basic Usage

```python
from unified_polytope_analysis import UnifiedPolytopeAnalyzer

# Initialize the analyzer
analyzer = UnifiedPolytopeAnalyzer(
    working_dir="./results",
    cache_activations=True,
    cache_ngrams=True
)

# Run complete analysis
results = analyzer.run_complete_analysis(
    model_name="EleutherAI/pythia-70m",
    checkpoints=["1000", "2000", "3000"],
    n_gram_size=2,
    pile_samples=2000,
    frequency_analysis=True
)

# Generate visualizations
figures = analyzer.visualize_results(save_plots=True)
```

### 2. Component-by-Component Usage

#### N-gram Dataset Creation

```python
from ngram_dataset import ImprovedNGramDatasetBuilder

builder = ImprovedNGramDatasetBuilder()
dataset = builder.build_stratified_ngram_dataset(
    n_gram_size=2,
    pile_samples=1000
)
builder.save_dataset(dataset, "./my_dataset")
```

#### Activation Extraction

```python
from checkpoint_analysis import ImprovedCheckpointAnalyzer

analyzer = ImprovedCheckpointAnalyzer()
data_manager = analyzer.extract_activations_from_checkpoints(
    model_name="EleutherAI/pythia-70m",
    checkpoints=["1000", "2000"],
    dataset=dataset,
    matching_strategy="comprehensive"
)
```

#### Polytope Analysis

```python
from polytope_metrics import get_polytope_metrics
import numpy as np

# Analyze activation polytope
activations = np.random.normal(0, 1, (100, 512))
metrics = get_polytope_metrics(activations)

print(f"Volume: {metrics.volume}")
print(f"Complexity: {metrics.complexity_score}")
print(f"Effective Dimension: {metrics.effective_dimension}")
```

## Core Components

### 1. ImprovedNGramDatasetBuilder

**Key Improvements:**
- Robust text processing with position tracking
- Multiple dataset loading strategies with fallbacks
- Frequency caching with Infinigram API integration
- Stratified sampling across frequency bins
- Comprehensive error handling

**Features:**
- Extracts n-grams with precise character positions
- Categorizes by frequency (very_rare to very_high)
- Supports both local and global frequency analysis
- Exports to multiple formats (JSON, HuggingFace)

### 2. ImprovedCheckpointAnalyzer

**Key Improvements:**
- Enhanced n-gram matching with multiple strategies
- Structured data management with indexed access
- Robust token-position mapping
- Support for different model architectures
- Comprehensive error handling and logging

**Features:**
- Multiple matching strategies (exact, flexible, token-based)
- Efficient activation record storage
- Layer and checkpoint indexing
- Cross-architecture MLP access

### 3. AdvancedPolytopeAnalyzer

**Key Improvements:**
- Comprehensive geometric analysis
- Stability assessment through bootstrap sampling
- Dimensionality reduction with PCA
- Multi-metric complexity scoring
- Robust error handling for edge cases

**Features:**
- Volume, surface area, diameter calculations
- Complexity scoring based on multiple factors
- Effective dimensionality estimation
- Stability analysis via resampling
- Evolution tracking across checkpoints

### 4. UnifiedPolytopeAnalyzer

**Key Improvements:**
- Complete pipeline integration
- Automatic caching and result management
- Multi-format export capabilities
- Built-in visualization generation
- Comprehensive reporting

**Features:**
- End-to-end analysis workflow
- Automatic layer selection
- Frequency-based comparative analysis
- Rich visualization suite
- Export to CSV, JSON, and markdown

## Data Structures

### ActivationRecord

```python
@dataclass
class ActivationRecord:
    checkpoint_step: str
    text_idx: int
    text: str
    ngram: str
    matched_text: str
    match_confidence: float
    category: str
    layer: int
    token_position: int
    activation_vector: np.ndarray
    sparsity: float
    metadata: Dict[str, Any]
```

### PolytopeMetrics

```python
@dataclass
class PolytopeMetrics:
    volume: float
    surface_area: float
    n_vertices: int
    n_faces: int
    complexity_score: float
    effective_dimension: int
    stability_score: float
    metadata: Dict[str, Any]
```

## Configuration Options

### N-gram Dataset Builder

```python
builder = ImprovedNGramDatasetBuilder(
    api_base_url='https://api.infini-gram.io/',
    cache_dir="./ngram_cache",
    max_samples=10000
)
```

### Checkpoint Analyzer

```python
analyzer = ImprovedCheckpointAnalyzer(
    cache_dir="./activation_cache"
)
```

### Polytope Analyzer

```python
polytope_analyzer = AdvancedPolytopeAnalyzer(
    pca_components=10,
    min_points_for_hull=4,
    stability_samples=100
)
```

## Output Files

The pipeline generates several output files:

- `activation_data.csv` - Complete activation records
- `layer_polytope_metrics.csv` - Layer-wise polytope metrics
- `analysis_summary.md` - Comprehensive analysis report
- `layer_comparison.png` - Layer comparison visualization
- `evolution_layer_X.png` - Evolution plots for each layer
- `category_comparison.png` - Frequency category comparison

## Performance Considerations

### Memory Management
- Uses streaming for large datasets
- Implements activation caching
- Samples n-grams to manage memory

### Computational Efficiency
- PCA dimensionality reduction for polytope computation
- Parallel processing where possible
- Efficient indexing for data access

### Storage Optimization
- JSON caching for n-gram frequencies
- Pickle serialization for activation data
- Optional compression for large datasets

## Error Handling

The pipeline includes comprehensive error handling:

- **Dataset Loading**: Multiple fallback strategies
- **Model Loading**: Graceful checkpoint failures
- **N-gram Matching**: Multiple matching strategies
- **Polytope Computation**: Fallback metrics for edge cases
- **API Calls**: Rate limiting and caching

## Research Applications

This pipeline supports research into:

1. **Polysemanticity Evolution**: How neurons transition from polysemantic to monosemantic
2. **Frequency Effects**: How n-gram frequency affects representation geometry
3. **Training Dynamics**: Temporal changes in activation structure
4. **Architecture Comparison**: Cross-model geometric analysis
5. **Interpretability**: Understanding feature emergence patterns

## Contributing

When contributing to this project:

1. Use `pathlib.Path` for all file operations
2. Include comprehensive error handling
3. Add type hints for all functions
4. Write docstrings with examples
5. Test with multiple model architectures

## Dependencies

```
torch>=1.9.0
numpy>=1.21.0
pandas>=1.3.0
scipy>=1.7.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
seaborn>=0.11.0
tqdm>=4.62.0
datasets>=2.0.0
transformers>=4.20.0
nnsight>=0.2.0
requests>=2.28.0
```

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{polytope_analysis_pipeline,
  title={Refactored Polytope Analysis Pipeline for LLM Polysemanticity Research},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo/polytope-analysis}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Troubleshooting

### Common Issues

**N-gram dataset creation fails:**
- Check internet connection for dataset downloads
- Verify Infinigram API accessibility
- Try reducing `pile_samples` parameter

**Activation extraction fails:**
- Ensure model checkpoints are available
- Check GPU memory availability
- Verify nnsight compatibility with model

**Polytope computation fails:**
- Ensure sufficient activation samples (>4)
- Check for NaN values in activations
- Try reducing PCA components

**Memory issues:**
- Reduce batch sizes and sample counts
- Enable caching to avoid recomputation
- Use streaming for large datasets

### Getting Help

For issues and questions:
1. Check the example usage scripts
2. Review error messages for specific guidance
3. Open an issue with minimal reproduction example
4. Include system and dependency information

## Roadmap

Future improvements planned:

- [ ] Support for additional model architectures
- [ ] Distributed computation for large-scale analysis
- [ ] Interactive visualization dashboard
- [ ] Integration with interpretability tools
- [ ] Automated hyperparameter tuning
- [ ] GPU acceleration for polytope computation 