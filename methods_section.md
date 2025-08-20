# Methods

## Overview

Our analysis pipeline consists of four interconnected components designed to study superposition and polysemanticity in large language models through the geometric lens of activation polytopes. The pipeline tracks how n-gram frequency influences neural specialization during training by analyzing spline codes derived from pre-activation patterns across multiple model checkpoints.

## 1. Data Preparation and N-gram Dataset Construction

### 1.1 N-gram Dataset Generation

We constructed a balanced dataset of country-capital n-grams stratified by frequency categories. The dataset contains phrases from two main frequency groups:
- **High-frequency n-grams**: Common country-capital pairs appearing frequently in training data
- **Low-frequency n-grams**: Rare country-capital pairs with minimal training exposure

Each sample in our dataset includes:
- **phrase**: The target n-gram (e.g., "Abuja", "Berlin")
- **sentence**: Contextualized template (e.g., "{phrase} is located in")
- **frequency_category**: Binary classification (high_frequency/low_frequency)
- **token_ids_sentence**: Pre-computed tokenization using the target model's tokenizer
- **activation_target_idx**: Token position for activation extraction
- **phrase_start_idx** and **phrase_end_idx**: Boundaries of the target phrase within the sentence

This pre-computation eliminates tokenization overhead during analysis and ensures consistent token positions across all model checkpoints.

### 1.2 Template-based Context Control

We employed a standardized template "{phrase} is located in" to control for contextual effects, ensuring that differences in activation patterns reflect frequency-dependent specialization rather than syntactic variations. This approach isolates the impact of n-gram frequency on neural representations.

## 2. Checkpoint-based Activation Extraction

### 2.1 Multi-checkpoint Analysis Pipeline

Our checkpoint analysis pipeline (`checkpoint_analysis.py`) extracts both pre-activation and post-activation vectors from transformer models at multiple training stages. The pipeline supports:

- **Streaming data processing** to handle large checkpoint files without memory overflow
- **Layer-wise extraction** from strategically selected layers (early, middle, and late layers)
- **Batch processing** with configurable batch sizes for memory efficiency
- **Network storage integration** with automatic resumption capabilities

### 2.2 Dual Activation Extraction

For each sample and checkpoint, we extract two types of activation vectors:

#### Post-activation Vectors
Standard layer outputs after applying the activation function (ReLU, GELU, etc.) and residual connections. These represent the final neural activations passed to subsequent layers.

#### Pre-activation Vectors
MLP intermediate outputs before the activation function, capturing the raw linear combinations. These are crucial for computing spline codes that define polytope regions.

### 2.3 Layer Selection Strategy

We employ an adaptive layer selection algorithm that automatically determines optimal layers based on model architecture:
- **Small models** (≤6 layers): [0, 2, 4, -1]
- **Medium models** (7-24 layers): Early (25%), middle (50%), late (75%), and final layers
- **Large models** (>24 layers): Distributed sampling across depth with emphasis on transformation regions

### 2.4 Activation Function Detection

Our pipeline automatically detects the activation function used in each layer's MLP module:
1. **Configuration parsing**: Primary method using model configuration files
2. **Module inspection**: Fallback method examining layer architecture
3. **Family heuristics**: Model-specific defaults (e.g., Pythia → GELU)

This detection is essential for accurate spline code computation.

## 3. Binary Pattern and Spline Code Analysis

### 3.1 CETT-based Binary Patterns

We compute binary activation patterns using the Coordinate-wise Error Tail Thresholding (CETT) method:

```
threshold = argmin_θ ||tail(x, θ)||₂ / ||x||₂ ≤ ε
binary_pattern = |activation_vector| > threshold
```

Where ε = 0.01 (1% error tolerance) represents the maximum fractional error in tail approximation.

### 3.2 Spline Code Computation

Spline codes represent the fundamental binary patterns defining polytope regions, computed from pre-activation vectors based on activation function type:

#### ReLU Networks
```
spline_code = pre_activation_vector > 0
```

#### GELU Networks  
```
spline_code = pre_activation_vector > -0.44  # approximate GELU inflection point
```

#### SwiSH/SiLU Networks
```  
spline_code = pre_activation_vector > 0  # similar to ReLU for SwiSH
```

Spline codes capture the piecewise-linear structure of the network and serve as the primary binary patterns for polytope analysis.

### 3.3 Activation Record Structure

Each activation record contains comprehensive metadata and analysis vectors:
- **Temporal information**: checkpoint_step, sample_idx
- **Linguistic metadata**: phrase, sentence, frequency_category
- **Spatial information**: layer, activation_target_idx, phrase boundaries
- **Post-activation data**: activation_vector, binary_pattern, sparsity metrics
- **Pre-activation data**: pre_activation_vector, spline_code
- **Derived metrics**: activation norms, active neuron counts, CETT thresholds

## 4. Geometric Polytope Analysis

### 4.1 Core Polytope Metrics

Our refined polytope analyzer (`refined_polytope_analyzer.py`) computes three primary geometric measures:

#### Polytope Density
Measures the relationship between activation space distances and binary pattern differences:
```
density = hamming_distance(patterns_i, patterns_j) / euclidean_distance(activations_i, activations_j)
```

Computed using chunked processing with random sampling (max 2,000 pairs) to prevent memory crashes while maintaining statistical validity.

#### Participation Ratio  
Quantifies effective dimensionality of activation patterns:
```
PR = (Σᵢ λᵢ)² / Σᵢ λᵢ²
```
Where λᵢ are eigenvalues of the activation covariance matrix. Higher participation ratios indicate distributed representations, while lower ratios suggest concentrated, specialized features.

#### Interference Patterns
Measures representational overlap between frequency groups:
- **Cosine similarity** between group mean activation vectors
- **Mann-Whitney U test** comparing activation norm distributions  
- **Neuron-wise correlation analysis** across frequency groups

### 4.2 N-gram to Polytope Mapping Analysis

We quantify polysemantic structure through:

#### Polysemantic Fraction
Proportion of polytopes (unique spline codes) associated with multiple n-grams:
```
polysemantic_fraction = |{p : |ngrams(p)| > 1}| / |total_polytopes|
```

#### N-gram Reuse Analysis
Proportion of n-grams appearing in multiple polytopes:
```
ngram_reuse_fraction = |{n : |polytopes(n)| > 1}| / |total_ngrams|
```

#### Cross-frequency Sharing
Analysis of polytopes shared between high-frequency and low-frequency n-grams, indicating representational interference.

### 4.3 Layer-wise Evolution Tracking

Our analyzer supports both legacy (layer-averaged) and layer-wise analysis modes:

#### Layer-wise Analysis
- Computes metrics separately for each layer
- Tracks cross-layer pattern inheritance
- Analyzes polytope emergence and dissolution across depth

#### Transition Analysis
- Identifies pattern inheritance between adjacent layers
- Quantifies emergence of new polytopes
- Measures pattern stability across layer boundaries

### 4.4 Numerical Stability Enhancements

All computations include robust numerical handling:
- **Z-score normalization** with overflow protection for distance calculations
- **Extreme value clipping** (±10⁶) to prevent numerical instabilities
- **Vectorized operations** for improved performance and precision
- **Chunked processing** for large-scale distance computations

## 5. Statistical Analysis and Visualization

### 5.1 Temporal Evolution Tracking

The pipeline tracks metrics across training checkpoints to identify:
- **Critical frequency thresholds** where specialization occurs
- **Temporal dynamics** of polysemantic structure evolution  
- **Layer-specific development** patterns
- **Cross-model consistency** in specialization timing

### 5.2 Comparative Analysis

Statistical comparisons between frequency groups using:
- **Mann-Whitney U tests** for distribution differences
- **Effect size measurements** (Cohen's d) for practical significance
- **Correlation analyses** between frequency and geometric measures
- **Regression modeling** for threshold identification

### 5.3 Multiprocessing and Scalability

The analyzer supports parallel processing:
- **Checkpoint-level parallelization** using ProcessPoolExecutor
- **Worker function isolation** to avoid serialization issues
- **Memory-efficient streaming** for large datasets
- **Network storage integration** for incremental analysis resumption

### 5.4 Visualization Pipeline

Comprehensive visualization suite including:
- **Evolution time series** for key metrics across checkpoints
- **Layer-wise heat maps** showing depth-dependent patterns
- **Frequency-dependent distributions** comparing high vs. low frequency groups
- **Cross-layer inheritance diagrams** tracking pattern propagation
- **Statistical significance overlays** highlighting critical transitions

## 6. Quality Assurance and Validation

### 6.1 Data Validation
- **Dataset structure validation** ensuring required fields and data types
- **Token alignment verification** between phrases and sentences  
- **Frequency category balance** checks across samples

### 6.2 Model Compatibility
- **Architecture detection** for optimal layer selection
- **Activation function identification** for accurate spline code computation
- **Memory profiling** and resource optimization for different model sizes

### 6.3 Reproducibility Measures
- **Fixed random seeds** across all stochastic operations
- **Deterministic sampling** for pair-wise distance computations
- **Comprehensive logging** with rotation and level controls
- **Version-controlled configurations** for experimental parameters

This methodology provides a robust, scalable framework for analyzing the geometric evolution of neural representations across training, with particular focus on frequency-dependent specialization in large language models.
