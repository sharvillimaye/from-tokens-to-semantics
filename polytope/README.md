# Multi-Dimensional Superposition Analysis

Enhanced polytope analysis implementation for studying superposition evolution across neural network training checkpoints.

## Overview

This module implements a **three-pronged visualization strategy** for analyzing how n-gram representations evolve during training:

1. **Temporal Evolution Analysis** (Primary) - Track single layer across training checkpoints
2. **Layer-wise Progression Analysis** (Secondary) - Compare metrics across network layers  
3. **Correlation Analysis** (Supporting) - N-gram frequency vs polytope structure

## Key Features

### Advanced Polytope Metrics
- **Custom convex hull approximation** for high-dimensional activations
- **Monte Carlo volume estimation** fallback for degenerate cases
- **Robust dimension reduction** with PCA and 95% variance retention
- **Statistical significance testing** with bootstrap confidence intervals

### Intelligent Checkpoint Selection
- **Adaptive sampling strategy**: Dense early training, sparse late training
- **Configurable checkpoint limits** to manage computational costs
- **Training dynamics awareness** for optimal checkpoint selection

### Publication-Ready Visualizations
- **Figure 1**: Temporal evolution plots with multiple frequency bins
- **Figure 2**: Layer progression heatmaps with custom colormaps
- **Figure 3**: Correlation scatter plots with regression lines and confidence intervals

## Quick Start

### Basic Usage

```python
from polytope_metrics import enhanced_run_analysis
from checkpoint_analysis import load_semantic_frequency_datasets

# Load your activation records
records = load_activation_records("path/to/records.pkl")

# Run complete analysis
results = enhanced_run_analysis(
    records=records,
    target_layers=[4, 6, 8, 10],
    checkpoint_selection="adaptive",
    max_checkpoints=15
)

# Results include all three analyses + publication figures
```

### Individual Analysis Components

```python
from polytope_metrics import (
    temporal_evolution_analysis,
    layer_progression_analysis, 
    enhanced_correlation_analysis
)

# 1. Temporal Evolution (Primary Analysis)
temporal_results = temporal_evolution_analysis(
    records=records,
    target_layer=6,  # Middle layer
    frequency_bins=3
)

# 2. Layer Progression (Secondary Analysis)  
progression_results = layer_progression_analysis(
    records=records,
    target_layers=[4, 6, 8, 10]
)

# 3. Correlation Analysis (Supporting Analysis)
correlation_results = enhanced_correlation_analysis(
    records=records,
    target_layers=[6, 8, 10]
)
```

### Generate NeurIPS Figures

```python
from polytope_metrics import create_neurips_figures

# Generate all three publication-ready figures
saved_figures = create_neurips_figures(
    records=records,
    target_layer=6,
    output_dir="reports/figures"
)

# Returns: {'figure1': 'path/to/fig1.png', 'figure2': '...', 'figure3': '...'}
```

## Data Pipeline

### Extract Activations from Checkpoints

```python
from checkpoint_analysis import (
    run_checkpoint_analysis_pipeline,
    load_semantic_frequency_datasets
)

# Complete pipeline from datasets to analysis
results_file = run_checkpoint_analysis_pipeline(
    model_name="EleutherAI/pythia-70m",
    checkpoints=["1000", "5000", "10000", "20000"],
    target_layers=[2, 4, 6, 8, 10, 12]
)
```

### Load and Process Datasets

```python
# Load semantic frequency datasets
datasets = load_semantic_frequency_datasets("datasets/semantic_frequency")

# Available datasets:
# - technology_high_freq_dataset.json
# - technology_low_freq_dataset.json  
# - science_high_freq_dataset.json
# - education_high_freq_dataset.json
# - general_high_freq_dataset.json
# - general_low_freq_dataset.json
```

## Analysis Strategy

### Temporal Evolution Analysis
**Purpose**: Demonstrate how superposition structure evolves during training

**Key Metrics**:
- Polytope volume evolution
- Vertex count changes  
- Effective dimension progression
- Frequency-dependent patterns

**Outputs**:
- Multi-line evolution plots
- Statistical trend correlations
- Bootstrap confidence intervals

### Layer Progression Analysis  
**Purpose**: Show representational complexity changes through network depth

**Key Metrics**:
- Volume heatmaps across layers × checkpoints
- Complexity progression patterns
- Critical layer identification

**Outputs**:
- Heatmap visualizations
- Layer-wise statistics
- Progression matrices

### Correlation Analysis
**Purpose**: Relate n-gram frequency to polytope structure

**Key Metrics**:
- Frequency vs activation norm
- Frequency vs sparsity patterns
- Training stage comparisons

**Outputs**:
- Scatter plots with regression lines
- Significance testing (p-values)
- Confidence intervals

## Advanced Features

### Adaptive Checkpoint Selection

Implements training dynamics-aware checkpoint selection:
- **Early training** (0-20%): Dense sampling (every 1-2 checkpoints)
- **Middle training** (20-60%): Moderate sampling (every 3-5 checkpoints)  
- **Late training** (60-100%): Sparse sampling (every 5-10 checkpoints)

```python
from polytope_metrics import adaptive_checkpoint_selection

selected = adaptive_checkpoint_selection(
    available_checkpoints=["100", "200", ..., "10000"],
    max_checkpoints=15
)
```

### Statistical Framework

Bootstrap confidence intervals and significance testing:

```python
from polytope_metrics import bootstrap_confidence_intervals

mean, lower, upper = bootstrap_confidence_intervals(
    data=volume_data,
    statistic_func=np.mean,
    n_bootstrap=1000,
    confidence_level=0.95
)
```

### Custom Convex Hull Implementation

Handles high-dimensional polytopes with robust approximation:

```python
from polytope_metrics import approximate_convex_hull, monte_carlo_volume_estimation

# Greedy hull approximation
hull_vertices = approximate_convex_hull(points, epsilon=0.1, max_iter=500)

# Monte Carlo volume fallback
volume = monte_carlo_volume_estimation(points, n_samples=100000)
```

## Implementation Principles

Following the **AI Research Codebase Principles**:

1. **Simplicity First**: Minimal, readable functions without over-engineering
2. **Task-Focused**: Direct implementation of research objectives
3. **Robust Implementation**: Input validation and error handling throughout
4. **Research-Specific**: Clear experimental purpose for each component
5. **Modular Design**: Reusable components for different analyses

## File Structure

```
polytope/
├── polytope_metrics.py          # Core analysis functions
├── checkpoint_analysis.py       # Data extraction pipeline  
├── neurips_analysis_example.py  # Complete demo
└── README.md                    # This file
```

## Example Output

### Analysis Summary
```
=== Analysis Complete ===
Analyzed 24,000 records across 8 checkpoints
Generated 3 publication figures

Volume Evolution Findings:
  Trend correlation: 0.734
  Mean volume: 0.000847
  95% CI: [0.000623, 0.001071]
  → FINDING: Strong positive volume growth during training

Layer Complexity Findings:
  Overall mean complexity: 12.47
  Complexity variance: 8.23
  Lowest complexity: Layer 4 (8.32)
  Highest complexity: Layer 10 (18.91)
```

### Generated Figures
- `figure1_temporal_evolution.png` - Evolution across checkpoints
- `figure2_layer_progression.png` - Heatmap across layers  
- `figure3_correlation_analysis.png` - Frequency correlations

## Requirements

- `numpy`, `pandas`, `matplotlib`, `seaborn`
- `scikit-learn` (PCA, preprocessing)
- `scipy` (ConvexHull, statistics)
- `nnsight` (model activation extraction)
- `torch` (PyTorch for model handling)

## Citation

For research using this analysis framework:

```
Multi-Dimensional Superposition Analysis: Polytope Evolution 
Across Neural Network Training Checkpoints
```