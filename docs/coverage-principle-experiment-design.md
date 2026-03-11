# Coverage Principle Experiment Design

## Context

This experiment re-evaluates the "coverage principle" from the paper "From Tokens to Semantics"
(Limaye, Ramesh, Rohweder et al., NeurIPS 2025 MechInterp Workshop) using fixed coverage metrics.

The paper claims: "Coverage emerges as the central organizing principle: specialization is not about
frequency preference but how often a neuron engages with diverse contexts." Specifically, after
controlling for coverage, mean affinity has |rho| <= 0.05 — largely uninformative.

## Problem: Original Coverage Metric Is Flawed

The paper defines participation coverage as C(h) = sum_p P(h|p), where P(h|p) is L1-normalized:

    P(h|p) = ReLU(a_h) / sum_h' ReLU(a_h')

This has three issues:

1. **Inter-neuron competition**: L1 normalization makes neurons compete — if one neuron dominates
   for a phrase, all others' P(h|p) are suppressed even if they fire meaningfully.
2. **Share vs breadth conflation**: A neuron firing enormously for 3 phrases can have the same
   coverage as one firing weakly for 50 phrases. The metric measures activation share, not breadth.
3. **Bucket size imbalance**: Coverage sums across all buckets without normalizing by bucket size.
   Neurons preferring the larger bucket get mechanically inflated coverage.

These issues may cause coverage to absorb variance that legitimately belongs to affinity,
making affinity appear uninformative when it may not be.

## Fixed Metrics (coverage_affinity_experiment.py)

### Binary Coverage (replaces L1-normalized firing mass)
    coverage(h) = fraction of phrases where activation > 0

For post-GELU MLP intermediate (4H-dim), >0 is the natural firing threshold.
No normalization, no inter-neuron competition. Computed per frequency group and total.

### Raw Mass (unnormalized)
    raw_mass(h) = mean(ReLU(activation_h)) across phrases in group

Per-group averaged (not summed), so group size doesn't bias the metric.

### Frequency Affinity (balanced ratio)
    affinity(h) = raw_mass_high / (raw_mass_high + raw_mass_low)

In [0,1]. 0.5 = no preference. Group-size balanced because each group is independently averaged.

### Log-Frequency Affinity (continuous)
    logfreq_affinity(h) = sum_p ReLU(a_h(p)) * log f(p) / sum_p ReLU(a_h(p))

Uses actual corpus frequencies from the ScaleJSD dataset (not cumulative.parquet lookup).
Unnormalized activations — no inter-neuron competition.

## Activation Level

The experiment captures the **post-activation MLP intermediate (4H-dim)** — the input to the
down-projection layer. This is the canonical "neuron activation" for the MLP, matching the
dimension of the neuron embeddings n_clusters metric. Captured via PyTorch hooks on the
down-projection module (generic across Pythia/LLaMA/OLMo architectures).

## Statistical Analysis

For each (checkpoint, layer):

1. **Raw Spearman**: coverage vs JSD contribution, affinity vs JSD contribution
2. **Partial Spearman**: affinity vs JSD | coverage, and coverage vs JSD | affinity
3. **Coverage-matched comparison**: bin neurons into coverage deciles, within each decile
   compare JSD contributions of high vs low affinity neurons (Cliff's delta + Mann-Whitney)
4. **Optional**: merge with n_clusters from neuron embeddings pipeline, repeat all above
   with n_clusters as outcome

## Preliminary Results (Pythia-70M, emotion, step143000)

| Layer | Spearman(cov,jsd) | Spearman(aff,jsd) | Partial(aff,jsd|cov) | Coverage-matched delta |
|-------|-------------------|-------------------|----------------------|------------------------|
| 0     | +0.147            | +0.023            | +0.014               | +0.014                 |
| 1     | +0.313            | +0.049            | -0.037               | -0.123                 |
| 2     | +0.325            | +0.135            | +0.081*              | -0.029                 |
| 3     | +0.446            | +0.277            | +0.210*              | +0.158                 |
| 4     | +0.420            | +0.259            | +0.198*              | +0.117                 |
| 5     | +0.394            | +0.287            | +0.213*              | +0.137                 |

Key findings:
- Coverage is still the stronger predictor (r=0.3-0.4 vs 0.1-0.3 for affinity)
- **Affinity has significant independent signal in layers 2-5** (partial r~0.2, p < 1e-14)
- This contradicts the paper's |rho| <= 0.05 — the broken metric was masking affinity's signal
- Early layers show no affinity effect (frequency specialization hasn't emerged)

## Scaling Plan

### Models
- EleutherAI/pythia-70m-deduped (6 layers, 4H=2048)
- EleutherAI/pythia-160m-deduped (12 layers, 4H=3072)
- EleutherAI/pythia-410m-deduped (24 layers, 4H=4096)

### Checkpoints
15 checkpoints: step3000, step13000, step23000, ..., step143000

### Datasets
All 5 ScaleJSD categories: emotion (75 pairs), medical (35), legal (41), scientific (73), verb (67)
Total: 291 synonym pairs = 582 samples per checkpoint

### n_clusters Integration
Merge with pre-computed neuron embedding clusters from the neuron embeddings pipeline.
Match on (layer, neuron) where neuron is the 4H MLP neuron index.

### Execution
```bash
# Per model:
python coverage_affinity_experiment.py \
    --all-datasets \
    --dataset-dir ~/dev/personal/mechinterp/scaleJSD/dataset/legacy/filtered \
    --model EleutherAI/pythia-70m-deduped \
    --revisions step3000,step13000,...,step143000 \
    --output-dir results/coverage_affinity/pythia-70m/ \
    --save-activations \
    --clusters <neuron-embeddings-clusters.csv>
```

For GPU runs (pythia-160m, 410m), use the Lambda cluster K8s setup.

### Questions to Answer
1. Does the partial affinity signal (r~0.2) replicate across model scales?
2. Does it grow or shrink with model size?
3. Does it emerge over training (like polysemantic coupling in Fig 7)?
4. Should the coverage principle be restated as "coverage is primary, affinity is secondary
   and layer-dependent" rather than "affinity adds little"?
5. Does the fixed metric change the polytope density / participation ratio story?

## File Location
coverage_affinity_experiment.py at project root. Self-contained — no nnsight dependency,
uses standard transformers + PyTorch hooks. Supports --validate for smoke testing.
