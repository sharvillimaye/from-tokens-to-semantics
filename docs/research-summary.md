# From Tokens to Semantics: Research Summary

## Paper Context

**Title**: From Tokens to Semantics: The Emergence and Stabilization of Polysemanticity in Language Models **Venue**: NeurIPS 2025 Mechanistic Interpretability Workshop (poster) **Authors**: Limaye\*, Ramesh\*, Rohweder, Zhou\*, Panda, Bhaskar, Sharma **Models**: Pythia 70M–6.9B (15 training checkpoints each), OLMo 1B/7B **Data**: ScaleJSD dataset — 291 synonym pairs across 5 domains (emotion, medical, legal, scientific, verb), each pair consisting of a high-frequency and low-frequency token with identical meaning (e.g. "happy" vs "elated")

## Central Question

How do transformer language models internally represent tokens that share meaning but differ in corpus frequency? Specifically: does the network treat "happy" and "elated" the same way, and how does this change across layers, training time, and model scale?

---

## The Four Analyses

### 1. Neuron Embedding Clusters (Polysemanticity Measurement)

**What it measures**: How many distinct semantic roles each MLP neuron plays.

**Method**: For each neuron, collect its top-activating input examples, compute neuron embeddings (Hadamard product of input weights and pre-MLP activations), then apply hierarchical agglomerative clustering. The resulting cluster count (n_clusters) quantifies polysemanticity — a neuron with n_clusters=1 responds to one coherent concept, while n_clusters=5 means it activates for 5 unrelated contexts.

**Finding**: Polysemanticity follows a developmental trajectory across layers. Early layers show elevated n_clusters (high polysemanticity, fragmented representations). Deeper layers show consolidation — n_clusters decreases as neurons specialize into coherent semantic roles. This trajectory also evolves over training: early checkpoints show more fragmentation everywhere, while later checkpoints show clearer layer-wise structure.

### 2. Jensen-Shannon Divergence (JSD)

**What it measures**: How differently the model's neurons collectively respond to high-frequency vs low-frequency tokens, per layer.

**Method**: For each layer, compute the activation distribution across all MLP neurons for all high-frequency inputs, and separately for all low-frequency inputs. The JSD between these two distributions quantifies how much the layer's internal state discriminates based on frequency.

**Finding**: Early layers have high JSD — they represent high and low frequency tokens very differently. JSD decreases monotonically through layers. By the final layers, high and low frequency synonyms produce much more similar activation patterns. This means early layers are **frequency-sensitive** (they route information based on how common a token is) while late layers are **frequency-invariant** (they represent meaning regardless of frequency).

**Figure — JSD Heatmap** (`figures/jsd_heatmap.png`): Pythia-6.9B group JSD across layers (y-axis) and training checkpoints (x-axis) for all 5 domains. JSD is concentrated in layers 0–7 (dark = high JSD) and fades to near-zero in later layers, universally across domains and training stages.

### 3. Polytope Boundary Density

**What it measures**: The geometric divergence between representations of synonym pairs in activation space.

**Method**: For each synonym pair (high-freq, low-freq), at each layer, compute:

- **Hamming distance**: between the binary spline codes (which ReLU neurons are active). This counts how many polytope boundaries separate the two representations.
- **Euclidean distance**: between the MLP input vectors (residual stream). This measures raw distance in activation space.
- **Polytope density**: Hamming / Euclidean. This normalizes boundary count by distance — high density means many boundaries are crossed per unit of distance, indicating the representations sit in different computational regions.

**Finding**: Polytope density decreases through layers. Early layers place synonym pairs in geometrically distinct regions (many boundaries between them), while late layers bring them into the same geometric neighborhood. This independently confirms the JSD finding: early layers separate by frequency, late layers converge toward shared semantic representations. The density drop scales with model size — larger models show more dramatic convergence in late layers.

**Figures**:
- `figures/fig1_density_per_model.png` — Polytope density across layers for all 5 models and 5 domains. Universal decrease from early to late layers.
- `figures/fig3_cross_model_fractional.png` — Cross-model comparison at matched fractional depth (layer/total_layers). Shows the pattern holds across architectures (Pythia vs OLMo) and scales.
- `figures/fig4_density_decomposition.png` — Hamming (numerator) and Euclidean (denominator) components separately. The density drop is primarily driven by Hamming distance decreasing (fewer boundaries crossed) while Euclidean distance increases (representations spread in raw distance).
- `figures/fig5_scaling.png` — Scaling trends: how first-layer density, last-layer density, and the density drop change with model size across both model families.
- `figures/fig7_key_result.png` — Main result: density and normalized Hamming by fractional depth across all models.

### 4. Coverage and Frequency Affinity (Neuron-Level Analysis)

**What it measures**: What properties of an individual neuron predict how much it contributes to separating high-frequency from low-frequency tokens.

This analysis operates at the neuron level rather than the layer level. For each MLP neuron, it asks: does this neuron treat high and low frequency tokens differently, and what predicts whether it does?

#### The Two Neuron Properties

**Coverage**: The fraction of all input phrases for which the neuron fires (activation &gt; 0). This measures breadth of participation.

- High coverage neuron (e.g. 0.9): fires for almost everything — it encodes a broadly applicable feature.
- Low coverage neuron (e.g. 0.05): fires rarely — it encodes something narrow or rare.

**Frequency Affinity**: The ratio of a neuron's mean activation for high-frequency tokens vs its total activation across both groups.

- Affinity ≈ 0.5: fires equally for both groups — no frequency preference.
- Affinity &gt; 0.5: preferentially fires for high-frequency tokens.
- Affinity &lt; 0.5: preferentially fires for low-frequency tokens.

#### The Target Variable: JSD Contribution

Each neuron has a JSD contribution — how much it individually contributes to the overall divergence between high and low frequency activation distributions at its layer. A neuron with high JSD contribution fires very differently for "happy" vs "elated." A neuron with zero JSD contribution treats them identically.

#### The Question

**What predicts a neuron's JSD contribution: its coverage (how broadly it fires), its frequency affinity (what it prefers), or both?**

This question matters because it tells us how neurons organize to handle frequency. If only coverage matters, neurons are just "broad vs narrow" and frequency handling is an emergent property of the collective. If affinity also matters, individual neurons have developed explicit frequency-selective roles.

#### The Coverage Principle (Original, Flawed)

The workshop paper defined coverage using L1-normalized activation mass:

$$P\_{s,L}(h|p) = \\frac{\\text{ReLU}(a_h)}{\\sum\_{h'} \\text{ReLU}(a\_{h'})}$$

$$\\tilde{C}*{s,L}(h) = \\sum*{p} P\_{s,L}(h|p)$$

Under this metric, coverage fully explained JSD contribution. After controlling for coverage via partial correlation, frequency affinity had |rho| &lt;= 0.05 — effectively zero. The conclusion: "Coverage emerges as the central organizing principle. Specialization is not about frequency preference but how often a neuron engages with diverse contexts."

**The problem**: L1 normalization divides each neuron's activation by the sum across all neurons at that position. This creates inter-neuron competition — if one neuron dominates for a given input, all other neurons' coverage scores are suppressed regardless of whether they fire meaningfully. This has three consequences:

1. **Artificial competition**: Neurons compete for coverage share rather than being measured independently.
2. **Breadth vs magnitude conflation**: A neuron firing enormously for 3 phrases can score the same coverage as one firing weakly for 50 phrases. The metric confuses activation magnitude with activation breadth.
3. **Bucket size bias**: Coverage sums across all inputs without normalizing by group size, mechanically inflating scores for neurons that prefer the larger frequency group.

These issues cause coverage to absorb variance that legitimately belongs to affinity, making affinity appear uninformative.

#### The Corrected Metrics

**Binary coverage**: fraction of phrases where activation &gt; 0. No normalization, no competition between neurons. A neuron either fires or it doesn't for each input.

**Raw mass**: mean(ReLU(activation)) per frequency group, not summed. Group-size-balanced.

**Frequency affinity**: raw_mass_high / (raw_mass_high + raw_mass_low). Balanced ratio in \[0,1\].

#### Corrected Results

With the fixed metrics, across Pythia 70M–6.9B and OLMo-7B, 5 domains, 15 training checkpoints:

**Coverage and affinity are both significant predictors of a neuron's JSD contribution**, but their relative importance depends on layer depth:

- **Early layers**: Coverage predicts JSD contribution (Spearman r = 0.15–0.31). Affinity adds no independent signal (partial r ≈ 0, not significant). The original coverage principle holds here.
- **Late layers**: Coverage remains the stronger predictor (r = 0.32–0.45). But affinity has **significant independent signal** after controlling for coverage (partial r = 0.08–0.29, p &lt; 1e-14). Among neurons with identical coverage, those with stronger frequency preferences contribute more to frequency divergence.

This means neuron specialization is **two-dimensional** — neurons vary in both breadth (coverage) and frequency selectivity (affinity), and both dimensions matter for how the network handles frequency in late layers.

Training dynamics: The affinity signal appears by step 3000 and remains stable thereafter. This is not a gradually learned property — it locks in early.

Domain dependence: Medical shows the strongest affinity signal (\~0.29), verb the weakest (\~0.12). This likely reflects that medical terminology has the sharpest frequency-meaning correlations (rare medical terms are semantically very different from their common equivalents), while verbs are more frequency-homogeneous.

**Figures**:
- `figures/partial_corr_by_layer.png` — Partial correlation (affinity -> JSD | coverage) by layer for Pythia-6.9B and OLMo-7B. Affinity signal near-zero in early layers, rises in later layers. Shaded bands show stability across checkpoints.
- `figures/summary_bars.png` — Mean partial Spearman r across all layers for each model and domain, with the paper's original |rho| <= 0.05 threshold marked in red. All Pythia-6.9B values well above threshold.
- `figures/training_dynamics.png` — Affinity signal over training in Pythia-6.9B, grouped by early/mid/late layers. Present from step 3k, stable throughout.
- `figures/coverage_vs_affinity.png` — Scatter of coverage-JSD correlation vs affinity-JSD correlation per (checkpoint, layer). Nearly all Pythia-6.9B points exceed the ±0.05 threshold.

#### What "Indicators" Means Precisely

When we say coverage and affinity are "indicators," we mean: **they predict how much each neuron contributes to the network's internal distinction between high and low frequency synonym representations (JSD contribution)**. Coverage tells you that broadly-firing neurons separate frequency groups more. Affinity tells you — additionally, in late layers — that neurons with stronger frequency preferences also separate frequency groups more, independent of their breadth.

---

## Summary of the Frequency-Semantic Transition

All four analyses converge on the same picture:

| Layer Depth | JSD | Polytope Density | Polysemanticity | Coverage vs Affinity |
| --- | --- | --- | --- | --- |
| Early | High (frequency-sensitive) | High (geometrically separated) | High (fragmented) | Coverage only |
| Late | Low (frequency-invariant) | Low (geometrically converged) | Low (consolidated) | Coverage + Affinity |

Early layers act as frequency-sensitive routers — they separate tokens based on corpus frequency, representations are geometrically distant, neurons are polysemantic and broadly tuned. Late layers build frequency-invariant semantic representations — synonym pairs converge geometrically, neurons consolidate into coherent roles, and a subset of neurons maintain explicit frequency-selective function.

---

## Open Mechanistic Questions

### What do high-coverage vs low-coverage neurons actually encode?

The coverage principle says high-coverage neurons are monosemantic generalists and low-coverage neurons are polysemantic specialists. But we haven't directly verified what features each group encodes. Possible approaches:

- **Top-activating example analysis**: For high-coverage (&gt;0.8) and low-coverage (&lt;0.1) neurons, collect the inputs that maximally activate them and categorize the features (syntactic, semantic, frequency-related, positional).
- **Sparse autoencoder features**: Train SAEs on MLP activations and map SAE features to coverage/affinity scores of the underlying neurons.
- **Logit attribution**: Trace high-coverage and low-coverage neurons through W_down and the unembedding to see which output tokens they promote.

### Why does affinity emerge as a predictor only in late layers?

Two possible mechanisms, not yet tested:

1. **Residual stream accumulation**: Early layers write frequency-correlated signals into the residual stream. Late-layer neurons read from this enriched stream, and some develop input weights aligned with frequency-correlated directions. The affinity signal would then reflect weight-space structure, not a learned computation.
2. **Functional requirement**: Next-token prediction is inherently frequency-dependent (common words are more probable). Late layers must preserve some frequency information for the output, requiring dedicated frequency-selective neurons. Early layers don't need this because they're still building the representation.

### Can we causally validate the layer-depth dependence?

The current evidence is correlational. Possible causal tests:

- **Activation patching**: Swap activations between high/low frequency inputs at specific layers. If early layers are frequency-sensitive, patching early layers should affect frequency-dependent outputs but not semantic ones.
- **Ablation of high-affinity neurons**: Zero out neurons with extreme affinity in late layers. Does this selectively disrupt the model's frequency-dependent behavior without affecting semantic discrimination?
- **Weight-space analysis**: Compare input weight vectors (W_up rows) of high-affinity vs low-affinity neurons, matched on coverage. Do high-affinity neurons have weights geometrically aligned with frequency-correlated embedding directions?

---

## Experimental Status

| Experiment | Models Completed | Status |
| --- | --- | --- |
| Polytope density | Pythia 70M/1B/6.9B, OLMo 1B/7B | Complete |
| JSD pipeline | Pythia 70M–2.8B | Complete |
| Neuron embeddings | Pythia 70M | Partial |
| Coverage-affinity (corrected) | Pythia 70M/6.9B, OLMo 1B/7B | Complete |
| Weight-space analysis | — | Not started |
| Feature characterization | — | Not started |
| Causal validation | — | Not started |
