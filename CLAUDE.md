# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is `superposition_timeline`, a research project for temporal analysis of polysemantic neurons in Large Language Models. The project studies how cumulative n-gram frequency affects neuron specialization and superposition phenomena across different model checkpoints during training.

## Essential Commands

### Environment and Dependencies
```bash
# Setup (requires Python 3.11)
make create_environment    # Create conda environment 
make requirements         # Install dependencies
source .venv/bin/activate # Or activate conda environment

# Code quality
make lint                 # Check code with ruff (line length 99)
make format               # Format code with ruff
ruff check --fix         # Auto-fix linting issues

# Testing
make test                 # Run pytest test suite
python -m pytest tests/  # Direct pytest execution
```

### Running Analysis Pipelines
```bash
# Main comprehensive polytope analysis
python run_comprehensive_polytope_analysis.py

# Country-capital frequency dataset generation  
python -m polytope.generate_country_capital_frequency_dataset \
  --output-path datasets/semantic_frequency/dataset.json \
  --target-examples-per-group 1000 --max-stream-samples 500000

# Improved analysis pipeline
python run_improved_pipeline.py
```

## Core Architecture

### 4-Phase Polytope Analysis Pipeline

The centerpiece is `polytope/polytope_analyzer.py` with the `SuperpositionAnalyzer` class implementing:

1. **Phase 1**: Preprocessing with binary pattern extraction using CETT thresholding
2. **Phase 2**: Polytope boundary analysis with density calculations  
3. **Phase 3**: Superposition detection and measurement
4. **Phase 4**: Advanced clustering analysis with HDBSCAN

Key metrics computed:
- **Polytope densities**: Hamming vs Euclidean distance ratios
- **Participation ratio**: Effective dimensionality from PCA spectrum
- **Interference patterns**: Neuron-neuron overlaps and cosine similarities
- **Cluster consistency**: Agreement between binary patterns and activation clusters

### Multi-Checkpoint Tracking

`MultiCheckpointPolytopeAnalyzer` tracks evolution across training checkpoints:
- Compares high-frequency vs low-frequency n-gram patterns
- Generates evolution visualizations and statistical summaries
- Produces publication-quality figures (300 DPI)

### N-gram Processing System

Abstract `NGramIndex` interface (`ngrams/interface.py`) with pluggable backends:
- **In-memory backend** (`ngrams/backends/inmemory.py`)
- **Tokengrams backend** (`ngrams/backends/tokengrams.py`)
- **Model parsers** for Pythia and OLMo cutoffs

## Key Module Structure

```
polytope/
├── polytope_analyzer.py        # Core analysis pipeline (1,700+ lines)
├── checkpoint_analysis.py      # Checkpoint data processing
└── ngram_dataset.py            # N-gram dataset utilities

ngrams/
├── interface.py                # Abstract n-gram interface
├── backends/                   # Storage backends
└── parsers/                    # Model-specific parsers

superposition_timeline/
├── config.py                   # Centralized configuration & paths
├── features.py                 # Feature engineering
└── plots.py                    # Visualization utilities
```

## Configuration and Data Organization

### Paths (from `superposition_timeline/config.py`)
```python
PROJ_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
```

### Caching Strategy
- Analysis results cached in `/cache/` directory
- Checkpoint data stored as pickle files
- Visualizations saved as PNG files with 300 DPI
- Use `cache_results=True` parameter in analysis functions

## Development Workflow

### Typical Research Pipeline
1. **Data Collection**: Generate n-gram datasets using dataset scripts
2. **Checkpoint Analysis**: Run activation extraction across model checkpoints
3. **Polytope Analysis**: Execute comprehensive analysis pipeline
4. **Visualization**: Generate evolution plots and publication figures
5. **Reporting**: Create analysis reports and statistical summaries

### Testing Strategy
- Main tests in `tests/` directory
- N-gram specific tests in `tests/ngrams_tests/`
- Use `pytest` with coverage for interface and backend testing
- Test edge cases (empty positions, single n-grams)

## Important Implementation Details

### Model Support
- **Pythia models**: 70M-2.8B parameters with cutoff parsers
- **OLMo-2 models**: 1B, 7B parameters with cutoff parsers
- Uses `transformer-lens` and `nnsight` for activation extraction

### Analysis Parameters
- **CETT thresholding**: For binary pattern extraction
- **HDBSCAN clustering**: For advanced clustering analysis
- **Frequency thresholds**: Configurable high/low frequency cutoffs
- **Context windows**: 60-320 characters for n-gram extraction

### Dataset Types Available
- Country-capital pairs (high/low frequency)
- Domain-specific datasets (education, science, technology)
- General n-gram frequency datasets
- Custom pattern-based datasets

## Git Repository Status
- Current branch: `feature/n-grams`
- Main development branch: `main`
- Always run lint checks before committing
- Use meaningful commit messages describing analysis changes

## Dependencies Notes
- Requires Python 3.11 (strict requirement)
- Built with `flit_core` for packaging
- Heavy scientific Python stack (torch, transformers, sklearn, etc.)
- Uses `ruff` for linting with 99-character line limit
- Logging configured with `loguru` and `tqdm` integration

## ALWAYS USE uv run
## ALWAYS unalias python