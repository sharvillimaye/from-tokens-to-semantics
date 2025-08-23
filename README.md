# From Tokens to Semantics: The Emergence and Stabilization of Polysemanticity in Language Models

This repository contains research code and analysis pipelines for studying neural network behavior, particularly focusing on polysemanticity, superposition, and temporal dynamics in language models.

## Project Overview

The repository consists of several components:

- **Neuron Embedding Pipeline**: Analysis of neuron activations and clustering behavior
- **N-gram Analysis**: Token sequence analysis and processing
- **JSD Calculations**: Jensen-Shannon Divergence analysis of frequency-affinity relationships
- **Polytope Metrics**: Geometric analysis of neural representations

## Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager (recommended)

### Installation

1. Clone the repository:
```bash
git clone <anonymized>
cd team-aasa
```

2. Install dependencies using uv:
```bash
uv sync
```

3. Activate the virtual environment:
```bash
uv shell
```

### Alternative Installation (pip)

If you prefer pip:
```bash
pip install -e .
```

## Project Structure

```
team-aasa/
├── neuron_embeddings/          # Neuron activation analysis
├── jsd_polysemanticity/        # JSD and polysemanticity pipeline
├── polytope/                   # Polytope analysis tools
├── ngrams/                     # N-gram processing and analysis
├── tests/                      # Test suite
└── pyproject.toml              # Project configuration and dependencies
```

## Key Dependencies

- **PyTorch**: Deep learning framework
- **Transformers**: Hugging Face transformers library
- **NNSight**: Neural network analysis tools
- **Matplotlib/Seaborn**: Visualization
- **Pandas/NumPy**: Data processing
- **Scikit-learn**: Machine learning utilities

## Usage Examples

### Running the JSD Pipeline

```python
from jsd_polysemanticity.pipeline import main

# Run the main analysis pipeline
main()
```

## Polytope Module

The `polytope/` package implements the polytope-based superposition analysis used in the paper.

```
polytope/
├── checkpoint_analysis.py      # Extract activations across checkpoints
├── refined_polytope_analyzer.py# Core analysis and pipeline entrypoints
├── visualization_utils.py      # Publication-grade figures (PDF/PNG)
├── dataset_from_index.py       # Build datasets from n-gram indices
└── ngram_dataset.py            # Utilities for templated n-gram datasets
```

Example end-to-end usage:

```python
from polytope.checkpoint_analysis import run_checkpoint_analysis_pipeline
from polytope.refined_polytope_analyzer import run_polytope_analysis_pipeline

checkpoint_file = run_checkpoint_analysis_pipeline(
    model_name="EleutherAI/pythia-410m",
    checkpoints=["1000", "10000", "50000"],
    dataset_path="dataset.json",
    output_dir="cache/checkpoint_analysis",
)

results_dir = run_polytope_analysis_pipeline(
    checkpoint_file=checkpoint_file,
    output_dir="results/polytope_analysis",
)
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
