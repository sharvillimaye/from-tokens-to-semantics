# From Tokens to Semantics: The Emergence and Stabilization of Polysemanticity in Language Models

This repository contains research code and analysis pipelines for studying neural network behavior, particularly focusing on polysemanticity, superposition, and temporal dynamics in language models.

## Project Overview

The repository consists of several research components:

- **Neuron Embeddings Analysis**: Analysis of neuron activations and clustering behavior
- **JSD Polysemanticity Pipeline**: Jensen-Shannon Divergence analysis of frequency-affinity relationships
- **Polytope Analysis**: Geometric analysis of neural representations
- **N-grams Analysis**: Token sequence analysis and processing

## Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager (recommended)

### Installation

1. Clone the repository:
```bash
git clone <team-aasa>
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
├── jsd_polysemanticity/       # JSD and polysemanticity pipeline
├── polytope/                  # Polytope analysis tools
├── ngrams/                    # N-gram processing and analysis
├── tests/                     # Test suite
└── pyproject.toml            # Project configuration and dependencies
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

### Neuron Embeddings Analysis

```python
from neuron_embeddings.demo import run_analysis

# Run neuron embedding analysis
run_analysis()
```

## Development

### Running Tests

```bash
uv run pytest
```

### Code Quality

The project uses [Ruff](https://docs.astral.sh/ruff/) for linting and formatting:

```bash
uv run ruff check .
uv run ruff format .
```

### Adding Dependencies

To add new dependencies, edit `pyproject.toml` and run:

```bash
uv sync
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or collaboration, please open an issue or contact the research team.
