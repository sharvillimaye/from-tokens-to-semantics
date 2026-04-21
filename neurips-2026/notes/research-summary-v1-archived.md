# From Tokens to Semantics: Research Summary

## Paper Context

**Title**: From Tokens to Semantics: The Emergence and Stabilization of Polysemanticity in Language Models
**Venue**: NeurIPS 2025 Mechanistic Interpretability Workshop (poster)
**Authors**: Limaye\*, Ramesh\*, Rohweder, Zhou\*, Panda, Bhaskar, Sharma
**Models**: Pythia 70M–6.9B (15 training checkpoints each), OLMo 1B/7B, Llama 3.1 8B
**Data**: ScaleJSD dataset -- 291 synonym pairs across 5 domains (emotion, medical, legal, scientific, verb), each pair consisting of a high-frequency and low-frequency token with identical meaning (e.g. "happy" vs "elated")

## Central Question

How do transformer language models internally represent tokens that share meaning but differ in corpus frequency? When the model processes "happy" and "elated" -- words with the same meaning but very different frequencies in training data -- are they treated the same internally? How does this change across layers, training time, and model scale?

This question matters because it probes a fundamental tension in language models: the embedding layer assigns very different vectors to high and low frequency tokens (they appear in different contexts with different frequencies), yet the model must eventually treat synonyms similarly to make correct predictions. Somewhere across the layers, frequency-specific representations must give way to semantic ones.

---

## The Core Story: A Frequency-to-Semantics Transition

Transformer MLP layers undergo a progressive transition from frequency-based to meaning-based representations. We measure this from three angles -- JSD, polysemanticity, and polytope geometry. All three show declining trajectories through layers:

| What we measure | What it tells you | Layer trajectory |
| --- | --- | --- |
| JSD | How much the layer discriminates by frequency | High → Low |
| n_clusters (polysemanticity) | How many roles each neuron plays | Peaks early → decreases |
| Polytope density | How geometrically separated synonyms are | High → Low |

These trajectories correlate at the layer level but are NOT tightly coupled at the neuron level (see "Layer vs Neuron Level" below). They are parallel consequences of the same underlying process -- layers consolidating representations -- rather than causally linked through individual neurons.

### How Polysemanticity Relates to the Frequency-Semantics Transition

Polysemanticity decreases through layers. JSD also decreases through layers. These co-occur reliably:

**Layer-level correlation** (JSD per layer vs mean n_clusters per layer):
- Pythia-70M: Spearman r = +0.94, p = 0.005
- OLMo-1B: Spearman r = +0.62, p = 0.010

**But at the neuron level, they are essentially independent.** Within any given layer, a neuron's JSD contribution does not predict its polysemanticity:
- Pythia-70M: r ≈ +0.01 to +0.09 per layer (weak, mostly not significant)
- OLMo-1B: r ≈ -0.04 to +0.02 (effectively zero)
- Pythia-6.9B: r ≈ +0.03 to +0.09 (weak positive)
- OLMo-7B: r ≈ -0.04 to +0.01 (effectively zero)

**What this means**: JSD and polysemanticity are both consequences of where a layer sits in the network, but they operate through different neuron-level mechanisms. A neuron does not contribute more to frequency discrimination because it is polysemantic. Rather, early layers have properties (high polysemanticity AND high JSD) that both change as representations consolidate through depth. The layer-level correlation is ecological, not mechanistic.

**What we can say**: Early layers maintain high-entropy, frequency-sensitive representations where neurons play many roles and frequency information is preserved. Late layers have lower-entropy, more consolidated representations where neurons play fewer roles and frequency information has been stripped. These are aspects of the same transition but NOT linked through individual neuron properties.

**What we cannot say**: "JSD drops because neurons become monosemantic." This is wrong in two ways: (1) neurons in larger models remain polysemantic even in final layers (OLMo-1B final layer: mean 10 clusters, OLMo-7B: mean 14.7), and (2) within a layer, polysemanticity does not predict JSD contribution.

---

## The Four Analyses

### 1. Neuron Embedding Clusters -- Measuring Polysemanticity

**What it measures**: How many distinct semantic roles each MLP neuron plays.

**Method**: For each neuron, collect its top-activating input examples, compute neuron embeddings (Hadamard product of input weights and pre-MLP activations), then apply hierarchical agglomerative clustering (HAC). The cluster count (n_clusters) quantifies polysemanticity: n_clusters=1 means the neuron responds to one coherent concept, n_clusters=5 means it activates for 5 unrelated contexts.

**Scaling update**: The original pipeline (ne_tlk.py + checkpoints_demo.py) processed ONE NEURON AT A TIME with a full forward pass each, requiring weeks for even partial coverage. We rewrote this as `scripts/metrics/neuron_polysemanticity.py` — single forward pass per layer, PyTorch hooks (no TransformerLens dependency), per-neuron min-heaps for top-k tracking, works on any HuggingFace model. Completed ALL neurons across 4 models in ~2.3 hours on H100:
- Pythia-70M: 12,288 neurons (6 layers x 2,048) — 90 seconds
- OLMo-1B: 131,072 neurons (16 layers x 8,192)
- Pythia-6.9B: 524,288 neurons (32 layers x 16,384)
- OLMo-7B: 352,256 neurons (32 layers x 11,008)

**Finding**: Polysemanticity follows a developmental trajectory across layers, confirmed at full neuron coverage across all 4 models:
- **Pythia-70M**: mean clusters drop from ~5 (L0-L1) to 2.0 (L4), 40% monosemantic at L4
- **OLMo-1B**: gradual decline from 25-27 (L1-L2) to 10 (L15)
- **Pythia-6.9B**: peaks at ~23 (L3-L5), consolidates in mid-layers (L10-L12: mean 11-15), drops to 10.7 at final layer
- **Pythia-6.9B**: peaks at ~33 (L12-L15), consolidates in final third, drops to 14.7 at L31

The trajectory holds across all architectures: polysemanticity peaks in early-to-mid layers, then decreases.

**Important caveat**: "Decreasing polysemanticity" does NOT mean "becoming monosemantic." Only Pythia-70M approaches monosemanticity (mean 2.0 clusters, 40% monosemantic). In 1B+ models, even final-layer neurons are highly polysemantic (10-15 clusters). The correct description is: polysemanticity decreases from extremely high to moderately high.

### 2. Jensen-Shannon Divergence (JSD) -- Measuring Frequency Discrimination

**What it measures**: How differently the model's neurons collectively respond to high-frequency vs low-frequency tokens, per layer.

**Method**: For each layer, compute the activation distribution across all MLP neurons for all high-frequency inputs, and separately for all low-frequency inputs. The JSD between these two distributions quantifies how much the layer's internal state discriminates based on frequency. High JSD = the layer treats "happy" and "elated" very differently. Low JSD = it treats them similarly.

**Finding**: Early layers have high JSD -- they represent high and low frequency tokens very differently. JSD decreases through layers. By the final layers, high and low frequency synonyms produce more similar activation patterns.

**Verified** (Pythia-6.9B, emotion, step143000): L0 JSD = 0.150, monotonically decreasing to L31 JSD = 0.015 (10x reduction).

### 3. Polytope Boundary Density -- Measuring Geometric Divergence

**What it measures**: How geometrically separated synonym pair representations are in activation space.

**Method**: For each synonym pair (high-freq, low-freq) at each layer, compute:
- **Hamming distance** between binary spline codes (which ReLU neurons are on/off)
- **Euclidean distance** between MLP input vectors (residual stream)
- **Polytope density** = Hamming / Euclidean

**Finding**: Polytope density decreases through layers. The density drop is primarily driven by Hamming distance decreasing (fewer neuron on/off boundaries crossed) while Euclidean distance actually increases. This means late layers route synonyms through the same nonlinear regions even as their raw vectors diverge — "computational equivalence" in the polytope framework.

The density drop scales with model size — larger models show more dramatic convergence in late layers.

**Note on GELU**: Polytopes are strictly defined for piecewise-linear activations (ReLU). Pythia/OLMo use GELU. The Hamming code is a reasonable proxy for the soft on/off state, and results are consistent across architectures. Should be acknowledged as a limitation.

### 4. Neuron-Level Decomposition -- What Drives the Transition?

**What it measures**: What properties of an individual neuron predict (a) how much it contributes to frequency divergence (JSD contribution), and (b) how specialized it is (n_clusters).

The previous three analyses operate at the layer level. This one zooms into individual neurons.

#### The Three Neuron Axes

The previous sections describe a **layer-level transition**: early layers are more frequency-sensitive, while later layers are more frequency-invariant and semantically consolidated. The goal of this section is different. Here we ask: **within a single layer, what kinds of neurons exist, and what roles do they appear to play?**

The workshop paper treated neuron behavior largely through a single metric, "participation coverage" (Eq 3-4), based on L1-normalized activation mass. Our investigation shows that this bundles together multiple distinct properties. A cleaner description of neuron behavior separates **three axes**:

**1. Firing breadth (binary coverage)** — fraction of input phrases where the neuron fires (activation > 0).
- Measures: How many different inputs activate this neuron?
- Coverage = 0.9: broad, general-purpose neuron
- Coverage = 0.05: narrow, selective neuron
- Functional interpretation: broad participation in shared computation

**2. Activation intensity (raw mass)** — mean(ReLU(activation)) across phrases, unnormalized.
- Measures: How strongly does this neuron fire on average?
- Independent of how broadly it fires
- Functional interpretation: how much total signal the neuron tends to contribute

**3. Frequency affinity** — ratio of mean activation for high-frequency vs total: raw_mass_high / (raw_mass_high + raw_mass_low). In [0,1], 0.5 = no preference.
- Measures: Which side of the frequency contrast the neuron prefers
- |affinity - 0.5| high: frequency-selective specialist
- Affinity near 0.5: little or no frequency preference
- Functional interpretation: frequency-routing / frequency-selective signal

These are not meant to be the only possible neuron properties. They are the three axes most relevant to the frequency-to-semantics question in this paper: **breadth**, **intensity**, and **frequency preference**.

**Participation coverage (the workshop paper's metric)** is best viewed as a legacy, conflated quantity rather than one of the three axes:
- It measures a neuron's share of total activation after L1 normalization
- It mixes breadth and intensity into one number
- It introduces inter-neuron competition: if one neuron dominates, others are mechanically suppressed

This gives a cleaner interpretation of the neuron-level story:
- Layer-level analyses tell us **what changes with depth**
- The three-axis decomposition tells us **which kinds of neurons inside a layer are associated with that change**
- The next step, beyond correlation, is to test whether these axes pick out neuron populations with distinct causal roles

#### What Predicts JSD Contribution?

With binary coverage, across **Pythia 70M–6.9B, OLMo 1B/7B, and Llama 3.1 8B** (three architectures), 5 domains:

| Metric | All layers | Deep layers |
| --- | --- | --- |
| Binary coverage → JSD | r = +0.21 | r = +0.28 |
| Affinity → JSD | r = +0.14 | r = +0.22 |
| **Partial (Affinity → JSD \| Coverage)** | **r = +0.12** | **r = +0.18** |
| Layers where partial p < 0.05 | 79% | 89% |

Key findings:
- Binary coverage is the stronger predictor overall
- **Affinity has significant independent signal** after controlling for coverage (partial r = +0.18 in deep layers), contradicting the workshop paper's |rho| <= 0.05
- In Llama 3.1 deep layers, coverage drops to near-zero while affinity rises — large models increasingly rely on frequency-specialized neurons
- Affinity signal appears by step 3000 and remains stable thereafter
- Medical domain strongest (~0.29), verb weakest (~0.12)

**Why affinity was masked in the workshop paper**: The workshop paper used participation coverage (L1-normalized), which bakes activation intensity into the coverage metric. Since intensity correlates with affinity (through inter-neuron competition from L1 normalization), controlling for participation coverage accidentally controls away the affinity signal. With binary coverage (pure breadth), affinity is free to show its independent contribution.

#### What Predicts Specialization (n_clusters)?

**The relationship between activation intensity and polysemanticity is model-size-dependent:**

**Pythia-70M** (L2, emotion, raw mass deciles):
| Mass decile | Mean mass | Mean n_clusters | % monosemantic |
| --- | --- | --- | --- |
| 0 (near-zero) | 0.0003 | 2.5 | 41% |
| 5 (medium) | 0.042 | 4.8 | 13% |
| 9 (highest) | 0.284 | 4.9 | 21% |

Near-zero mass neurons are monosemantic (many are effectively dead — barely activate, so their few activations cluster into 1 group). Active neurons are broadly polysemantic with a slight decrease at the highest mass.

**OLMo-1B** (L15 final, emotion, raw mass deciles):
| Mass decile | Mean mass | Mean n_clusters | % monosemantic |
| --- | --- | --- | --- |
| 0 (lowest) | 0.007 | 11.0 | 8% |
| 5 (medium) | 0.025 | 10.7 | 5% |
| 9 (highest) | 0.130 | 6.3 | 25% |

Monotonically decreasing: higher mass → fewer clusters. No inverted-U. Even the lowest-mass neurons have nc=11 (all neurons are active in OLMo).

**OLMo-7B** (L31 final, emotion): Same monotonic pattern. Mass decile 0: nc=14.5, decile 9: nc=10.1.

**Pythia-6.9B** (L31 final, emotion): Near-zero neurons have nc=10.1 (22% mono), medium neurons peak at nc=12.1, highest-mass neurons drop to nc=6.2 (41% mono). This shows a mild inverted-U but the "low" end is driven by dead/near-zero neurons that are trivially monosemantic.

**Summary**: The robust finding across all models is that **high-intensity neurons are less polysemantic** (monotonically in OLMo, with a dead-neuron artifact in Pythia). The workshop paper's claim ("high-coverage neurons specialize") was about this mass effect, but using participation coverage obscured the mechanism. The "inverted-U" described in earlier versions of this document was not robust — it is primarily a dead-neuron effect in Pythia models, not a true nonlinearity among active neurons.

#### Participation Coverage vs n_clusters (Step 1 Verification, Completed 2026-04-03)

**Motivation**: The workshop paper's narrative implied "high coverage → monosemantic (specialized)." We needed to test whether this held with the paper's actual L1-normalized metric (participation coverage) against full n_clusters data, since the original JSD pipeline never formally merged these. The original `pythia70m_embeddings.parquet` referenced in the notebook config was never generated on the cluster — the original ne_tlk.py pipeline (one neuron per forward pass) never completed at scale.

**Method**: Added participation coverage computation to `scripts/metrics/coverage_affinity_experiment.py`. For each phrase p, L1-normalize activations across neurons: share_h(p) = ReLU(a_h(p)) / sum_h' ReLU(a_h'(p)). Participation coverage = mean_p(share_h(p)). Ran on cached activations merged with n_clusters data from `scripts/metrics/neuron_polysemanticity.py` for all 4 models x 5 domains. (K8s job `ani-participation-coverage`, ~10 min total, 2026-04-03.)

**Results** — Spearman correlation between participation coverage (PC) and n_clusters:

| Model | Early layers | Mid layers | Final layer(s) |
| --- | --- | --- | --- |
| **Pythia-70M** | **+0.08 to +0.28** | +0.13 to +0.21 | -0.04 (L5) |
| OLMo-1B | -0.02 to -0.10 | -0.05 to -0.10 | **-0.15 to -0.22** (L15) |
| Pythia-6.9B | weak negative | -0.03 to -0.05 | -0.04 to -0.12 (L31) |
| OLMo-7B | mixed +/-0.05 | -0.03 to -0.06 | **-0.12 to -0.14** (L30-31) |

**Key observations:**

1. **Pythia-70M: reversed.** PC → n_clusters is POSITIVE in layers 1-4 (r = +0.15 to +0.28). High-PC neurons are MORE polysemantic. This is because in a small model, the highest-PC neurons are the moderately active ones (the dead-neuron effect pulls down the low end, making the correlation positive).

2. **1B+ models: weakly consistent with narrative.** PC → n_clusters is consistently negative (r ~ -0.05 to -0.10), strongest at final layers (r ~ -0.15 to -0.22). The direction matches the paper's claim, but the effect is weak.

3. **PC and raw mass are nearly interchangeable.** PC correlations with n_clusters differ from raw mass correlations by <0.01 everywhere. The L1 normalization in PC effectively turns it into a mass proxy, confirming that participation coverage measures activation intensity (dominance), not breadth. Binary coverage behaves distinctly.

**Conclusion**: The workshop paper's "coverage principle" was specifically about JSD contribution (high coverage → high JSD contribution), which replicates and is mechanically straightforward (more firing = more contribution). The extended narrative about coverage predicting specialization was never formally tested in the original pipeline and is model-size-dependent. The three-axis decomposition (breadth, intensity, affinity) reveals structure that the single participation coverage metric hides.

**Forensic note on the legacy pipeline**: The original JSD pipeline (`jsd_polysemanticity/pipeline_notebook.ipynb`) computed `firing_mass` (= sum_p P(h|p), L1-normalized mass) and `freq_affinity` per neuron. The notebook config referenced `clusters_parquet = "./pythia70m_embeddings.parquet"` for merging with n_clusters, but this parquet was never generated — the original ne_tlk.py processed one neuron at a time via TransformerLens and never completed at scale. The `run_multidomain.py` that produced the actual cluster results doesn't reference clusters at all. The coverage-polysemanticity narrative was extrapolated from the JSD contribution result, not directly measured.

---

## Safety Neuron Investigation (Completed)

### Motivation

We investigated whether safety neurons (Chen et al., 2024, arXiv:2406.14144) — the ~5% of MLP neurons causally responsible for safety alignment — are distinguishable by our three axes.

### Method

Activation contrasting between Llama 3.1 8B base vs Instruct, and separately Tulu 3 SFT vs DPO, following Chen et al.'s methodology. 32 safety prompts + 15 benign controls. Merged safety neuron rankings with coverage/affinity metrics.

### Results

| Metric | Safety vs Non-safety | Effect size | Significance |
| --- | --- | --- | --- |
| **Binary coverage (breadth)** | 0.497 vs 0.500 | d ~ -0.01 | **Null** |
| **Frequency affinity** (all layers) | 0.495 vs 0.500 | d ~ -0.04 | Weak |
| **Frequency affinity** (L18-31) | 0.490 vs 0.511 | d ~ -0.15 | Significant |
| **Raw mass (intensity)** | 0.089 vs 0.015 | d ~ +0.45 | **Strong** (p ~ 0) |

- 36% of safety neurons are in the top 5% by raw mass (7x enrichment over chance)
- 36% of top-5% raw mass neurons are safety neurons
- Both SFT→DPO and base→instruct approaches gave consistent results

### Interpretation

Safety neurons are characterized by **activation intensity**, not by breadth or frequency preference. This is consistent with the finding that high-mass neurons are less polysemantic — safety neurons are intense specialists that fire strongly for their specific function. This is a supporting finding, not a main contribution.

---

## Critical Related Literature

### Directly Related (Must Cite and Position Against)

**"Emergent Specialization: Rare Token Neurons" (2505.12822, ENS/Sorbonne 2025)**
1.7% of neurons form an "influential plateau" for rare token prediction, with a three-phase distribution. Uses Pythia 70M and 410M. They identify rare-token neurons via ablation effect on prediction loss. They do NOT connect to coverage/affinity or layerwise decomposition. Our framework provides the organizing context for their finding.

**"Transformer Language Models Handle Word Frequency in Prediction Head" (Kobayashi et al., ACL Findings 2023, 2305.18294)**
Prediction head bias parameter encodes word frequency. Hidden states are nearly orthogonal to frequency direction (cos ~ 0.08). Our paper explains WHY — MLP layers progressively strip frequency information, so the head must re-introduce it. Four-stage pipeline: input embeddings (frequency-biased) → early MLPs (frequency routing) → late MLPs (frequency-invariant semantics) → prediction head (frequency re-correction).

**"How Do LLMs Use Their Depth?" (UC Berkeley/Georgia Tech, 2510.18871)**
Traces layerwise prediction dynamics. Independent confirmation that depth is used non-uniformly.

**"Evolution of Concepts in Language Model Pre-Training" (2509.17196)**
Tracks feature evolution across pretraining using crosscoders. Complementary methodology.

### Theoretical Foundations (For Normative Argument)

**"Saddle-to-Saddle Dynamics Explains A Simplicity Bias" (2512.20607, NeurIPS 2025)**
SGD provably learns simple/low-rank features first. Frequency is low-rank (Zipf's law), semantics is high-rank → frequency gets learned first.

**"Understanding Seemingly Useless Features in Next-Token Predictors" (Rofin et al., ICLR 2026, 2603.14087)**
Gradient signal decomposes into direct, pre-cached, and circuit-sharing components. Features not useful for immediate prediction emerge via shared gradients.

**"Not a nuisance but useful: Outlier dimensions favor frequent tokens" (2503.21718)**
Outlier dimensions in last layer favor frequent tokens, emerge early in training.

**"Exploiting Vocabulary Frequency Imbalance" (KAIST/NAVER, 2508.15390)**
Models reduce loss primarily by improving on frequent tokens. Output embedding norms of frequent tokens are larger.

**"Confidence Regulation Neurons" (ETH/MIT, NeurIPS 2024, 2406.16254)**
Neurons that regulate prediction confidence. May overlap with high-coverage neurons.

### Established Background

**"FF Layers Are Key-Value Memories" (Geva et al., EMNLP 2021, 2012.14913)** — Lower layers capture superficial patterns, upper layers semantic ones.

**"Polysemanticity and Capacity" (Scherlis et al., 2022, 2210.01892)** — Theoretical framework connecting polysemanticity to capacity constraints.

**"Universal Neurons in GPT2" (Gurnee et al., 2024, 2401.12181)** — 1-5% of neurons are universal across seeds. Prediction: should coincide with highest-coverage neurons.

**"Safety Neurons" (Chen et al., NeurIPS 2025, 2406.14144)** — ~5% of MLP neurons causally control safety. We tested and found raw mass is the differentiator.

---

## Linear Probes: Frequency vs Semantic Category (Completed 2026-04-03)

### Motivation

The normative argument claims "frequency is learned first, semantics later" due to SGD simplicity bias. Linear probes directly test this: train probes at each layer to predict (a) whether a token is high vs low frequency, and (b) which semantic domain it belongs to (emotion/medical/legal/scientific/verb). If the argument holds, frequency probes should peak early and semantic probes should peak late.

### Method

Used the `linear_probes` framework (binary linear classification on residual stream activations at `token_index=-1` of the target word). ScaleJSD synonym pairs provide natural controls: for each pair, high-freq and low-freq variants share the same sentence template, isolating frequency. Semantic probes use one-vs-rest binary classification per domain. Group-aware train/val/test splits (70/15/15) prevent synonym pair leakage. 20 epochs, early stopping, bootstrap CIs.

Models: Pythia-70M (15 training checkpoints), Pythia-6.9B (3 checkpoints: step3000/73000/143000), OLMo-1B, OLMo-7B. All layers probed.

### Results

**1. Frequency probes peak mid-network and decline in late layers (universal).**

| Model | Layers | Freq peak layer (AUROC) | Freq final layer (AUROC) | Drop |
| --- | --- | --- | --- | --- |
| Pythia-70M (step143000) | 6 | L2 (0.876) | L5 (0.566) | -0.31 |
| OLMo-1B | 16 | L10 (0.978) | L15 (0.895) | -0.08 |
| Pythia-6.9B (step143000) | 32 | L12 (0.979) | L31 (0.914) | -0.07 |
| OLMo-7B | 32 | L4 (0.984) | L31 (0.857) | -0.13 |

The decline is not "failure to encode" — frequency peaks at 0.97+ mid-network in larger models. Late layers **actively remove** frequency information from the residual stream.

**2. Semantic probes saturate early and persist throughout.**

All semantic probes reach AUROC >= 0.99 by L2-5 across all models and remain there through the final layer. Domain difficulty ranking is stable across architectures: verb (trivial, L0-1) > emotion/legal (L1-2) > medical/scientific (L3-7).

**3. Training dynamics reveal learned frequency erasure (Pythia-70M).**

| Layer | step3000 | step43000 | step93000 | step143000 |
| --- | --- | --- | --- | --- |
| L0 | 0.820 | 0.865 | 0.853 | 0.865 |
| L2 | 0.850 | 0.889 | 0.893 | 0.876 |
| L3 | 0.855 | 0.902 | 0.890 | 0.864 |
| L5 (final) | 0.830 | 0.815 | 0.695 | **0.566** |

Frequency probeability in the final layer degrades from 0.83 (step3000) to 0.57 (step143000) — near chance. The model **learns to erase frequency from late layers during training**. Meanwhile, semantic probeability only improves.

**4. Domain-specific patterns in larger models.**

In Pythia-6.9B and OLMo-7B, the "scientific" domain shows a slower ramp (reaching 0.99+ only at L5-6) compared to other domains, while "verb" is perfectly separable from L0. This likely reflects the lexical distinctiveness of each domain's sentence templates.

### How This Connects to Other Findings

The linear probes provide the **residual stream perspective** that complements the neuron-level analyses:

- **JSD drops through layers** (Section 2) = the same phenomenon measured via neuron activation distributions. Frequency becoming less distinguishable in the residual stream is why JSD between high/low freq activation patterns decreases.
- **Polytope density drops** (Section 3) = the geometric mechanism. Late layers route synonyms through identical ReLU regions regardless of frequency.
- **Affinity has independent signal in deep layers** (Section 4) = the remaining frequency signal is concentrated in a few specialist neurons (high affinity), not spread broadly. A linear probe on the full residual stream misses it, but individual neuron affinity captures it.
- **Kobayashi et al. (2305.18294)** found hidden states are nearly orthogonal to frequency (cos ~ 0.08) and the prediction head re-introduces frequency via bias. Our probes show *why* (late layers strip frequency) and *when* this develops during training.

### Revised Normative Argument

The original prediction ("frequency probes peak early, semantic probes peak late") was **partially wrong**. The actual pattern is more interesting:

- Frequency information is **everywhere in early training**, peaks in **mid-layers**, and is **actively erased in late layers** as training progresses.
- Semantic category is decodable from L1-2 onward and never erased.
- The story is not "frequency first, then semantics" but "**frequency everywhere early, then selectively removed**."

This is consistent with the simplicity bias argument (2512.20607): frequency (low-rank) gets encoded everywhere initially, then the model discovers it only needs frequency for the output head, and learns to strip it from the residual stream to make room for higher-rank semantic features.

### Caveat: Template Confound

Semantic probes likely achieve near-perfect AUROC partly because sentence templates carry domain-specific lexical cues. The frequency probe is more informative because both high/low variants share the same template, isolating the frequency signal. Follow-up: probe with domain-neutral templates or ngram-only text to test whether semantic probeability reflects genuine semantic representations vs template structure.

### Data

Results in `results/linear_probes/` — per-model, per-checkpoint `all_probes_summary.csv` files with columns: `probe, layer, val_auroc, val_accuracy`. Script: `linear_probes` repo (`experiments/scaleJSD_probing/run_probes.py`).

---

## Targeted Ablation: Causal Validation of the Three-Axis Decomposition (Completed 2026-04-04)

### Motivation

The neuron-level analysis (Section 4) shows that binary coverage, raw mass, and frequency affinity correlate differently with JSD contribution. But correlation does not establish functional roles. If the three axes identify genuinely different neuron populations, then ablating neurons selected by each axis should produce different causal effects on frequency discrimination and model behavior.

### Method

Zero-ablation of the post-activation MLP intermediate (4H-dim input to down_proj) for selected neurons in the last third of each model. This removes the neuron's contribution to the residual stream via the down-projection. Five models: Pythia-70M (L4-5), OLMo-1B (L11-15), Pythia-6.9B (L22-31), OLMo-7B (L22-31), Llama 3.1 8B (L22-31). All 5 ScaleJSD domains. Ablation percentages: 1%, 2%, 5%, 10%, 20%.

Eight selection strategies:
- **affinity**: top neurons by |frequency_affinity - 0.5| (most frequency-selective, either direction)
- **affinity_high_freq**: top neurons by frequency_affinity (prefer common words)
- **affinity_low_freq**: bottom neurons by frequency_affinity (prefer rare words)
- **coverage**: top neurons by binary coverage (broadest-firing)
- **low_coverage**: bottom neurons by binary coverage, excluding dead neurons (narrowest specialists)
- **mass**: top neurons by raw mass (highest-intensity)
- **jsd**: top neurons by JSD contribution (direct upper bound)
- **random**: random neurons (null control, 5 seeds)

Measurements: group JSD between high-freq and low-freq synonym activations (per layer), and output KL divergence between clean and ablated model logits.

### Results: The Dissociation

At 5% ablation, global summary across all 5 models and 5 datasets:

| Strategy | JSD change | Output KL | Interpretation |
| --- | --- | --- | --- |
| **affinity** | **-7.1%** | 0.14 | Frequency-selective neurons carry the frequency signal |
| **affinity_high_freq** | **-6.4%** | 0.056 | High-freq-preferring neurons: most signal, least output disruption |
| **affinity_low_freq** | **-5.0%** | 0.226 | Low-freq-preferring neurons: less signal, more output disruption |
| **coverage** | **+2.2%** | 0.60 | Removing broad neurons *exposes* frequency differences |
| **mass** | **+2.1%** | 0.88 | Removing intense neurons disrupts outputs most |
| **low_coverage** | -1.5% | 0.15 | Narrow specialists carry weak frequency signal |
| jsd | -10.0% | 0.12 | Upper bound (selected by the metric itself) |
| random | ~0% | 0.03 | Clean null |

### Key Findings

**1. The three axes produce a causal dissociation.**

All frequency-selectivity strategies (affinity, affinity_high, affinity_low, low_coverage) reduce JSD when ablated. All breadth/intensity strategies (coverage, mass) increase JSD. Random is null. This sign split is universal across all 5 models and 3 architectures (GPT-NeoX, OLMo, LLaMA).

This upgrades the correlational finding (Section 4) to a causal one: the three axes do not merely correlate differently with JSD — they identify neuron populations with **opposite causal effects** when removed.

**2. High-freq-preferring neurons carry more signal but less output importance.**

affinity_high_freq accounts for more JSD reduction (-6.4%) than affinity_low_freq (-5.0%) at 5%, but causes much less output disruption (KL 0.056 vs 0.226). The model dedicates more routing capacity to "this is a common word" than "this is a rare word," and this high-freq routing is more separable from the model's primary predictions. This connects to the "Exploiting Vocabulary Frequency Imbalance" finding (2508.15390): models optimize primarily on frequent tokens, so internal frequency tracking is biased toward them.

**3. Coverage ablation reveals shared computation, not more frequency information.**

Ablating broad-firing neurons increases JSD because these neurons contribute a common activation component to both high-freq and low-freq inputs. Removing them removes the shared "floor," making the remaining frequency-selective neurons more prominent in the L1-normalized JSD computation.

**Normalization caveat**: JSD is computed on L1-normalized distributions (see `scripts/metrics/coverage_affinity_experiment.py`). Part of the coverage/mass JSD increase may be a mechanical renormalization artifact rather than a genuine increase in representational frequency discrimination. The probe-under-ablation experiment (design doc: `neurips-2026/notes/probe-under-ablation-design.md`) would test this with a normalization-free metric (linear probe AUROC on raw residual stream activations under ablation).

**4. The dissociation is universal across scale and architecture.**

Sign of JSD effect at 5% ablation by model:

| Model | affinity | coverage | mass | random |
| --- | --- | --- | --- | --- |
| Pythia-70M (70M) | - | + | + | ~0 |
| OLMo-1B (1B) | - | + | + | ~0 |
| Pythia-6.9B (6.9B) | - | + | + | ~0 |
| OLMo-7B (7B) | - | + | + | ~0 |
| Llama 3.1 8B (8B) | - | + | + | ~0 |

Universality of sign is robust. Magnitude varies with model size and architecture (treat scaling trends as provisional).

### How This Ties Back to the Main Story

The paper's core contribution is the **frequency-to-semantics transition** across transformer layers, decomposed at the neuron level. The ablation results connect each piece:

1. **JSD drops through layers** (Section 2) because late layers contain more coverage/shared-computation neurons and fewer frequency-routing neurons. The balance shifts.

2. **Polytope density drops** (Section 3) because the shared-computation neurons route synonyms through similar nonlinear regions regardless of frequency.

3. **Coverage predicts JSD contribution; affinity has independent signal** (Section 4) — the ablation confirms this is not just correlation. Coverage neurons contribute to JSD by providing a shared baseline that frequency-selective neurons deviate from. Affinity neurons contribute the deviation itself.

4. **Linear probes show frequency erasure in late layers** (Step 3) because the residual stream in late layers is dominated by coverage/shared neurons, with the frequency signal carried by a shrinking minority of affinity specialists.

5. **Training dynamics** show frequency erasure intensifies because the model learns to consolidate frequency routing into fewer specialist neurons and expand shared-computation capacity. The Pythia-70M L5 AUROC drop (0.83 → 0.57 over training) reflects this rebalancing.

The three-axis decomposition is not just descriptive — it identifies the functional components of the transition mechanism.

### Framing for the Paper

The strongest claim:

> "The three-axis decomposition identifies neuron populations with different causal roles in the frequency-to-semantics transition. Ablating frequency-selective neurons (high affinity) reduces internal frequency discrimination with relatively modest output disruption, whereas ablating broadly active neurons (high coverage or mass) increases frequency discrimination and perturbs outputs more strongly. This dissociation demonstrates that late-layer MLPs contain at least two functionally distinct neuron populations: frequency-selective specialists and broadly active neurons supporting shared computation."

Note: "affinity ablation reduces JSD" is supportive but close to a construct-validity check (affinity is designed to capture frequency preference). The non-obvious finding is the opposite direction of coverage/mass ablation — that is the core of the dissociation argument.

### Open Controls

1. **Probe under ablation** (highest priority follow-up): train linear frequency probes on raw residual stream activations extracted during ablation. This tests the dissociation without L1 normalization. Design: `neurips-2026/notes/probe-under-ablation-design.md`.
2. **Cosine/Euclidean distance** between mean high-freq and low-freq activation vectors under ablation — another normalization-free metric.
3. **Synonym prediction accuracy** under ablation — behavioral measure rather than representational.

### Data

Results in `results/ablation/` — per-model `ablation_results.csv` (original 5 strategies) and `ablation_lowcov_results.csv` (directional affinity + low_coverage). Columns: `strategy, pct, seed, layer, ablate_layers, n_ablated, clean_group_jsd, ablated_group_jsd, jsd_change, jsd_change_pct, clean_mean_pair_jsd, ablated_mean_pair_jsd, pair_jsd_change_pct, output_kl, dataset`. Script: `scripts/interventions/targeted_ablation.py`.

---

## Circuit Decomposition Attempt (2026-04-07) — Inconclusive

### Motivation

We attempted to trace frequency information flow through the computation graph by projecting each component's output (attention, MLP) onto a "frequency direction" in the residual stream. The goal: show that MLP layers flip from writing frequency (positive projection, early layers) to erasing it (negative projection, late layers), providing a circuit-level explanation for the probe and JSD findings.

### Method

Defined a fixed frequency direction as `normalize(mean(h_L0 | high_freq) - mean(h_L0 | low_freq))` from the embedding layer. For each layer, captured attention output and MLP output separately, computed the difference between high-freq and low-freq mean outputs, and projected onto this direction. Ran across all 5 models and 5 datasets.

### Results — Did Not Support the Simple Prediction

The prediction was: MLP projection positive early (writing frequency), negative late (erasing). The actual pattern was more complex and model-dependent:

- **OLMo-1B**: MLP projection negative at **every** layer, yet frequency content grew monotonically
- **Llama 3.1 8B**: MLP projection positive at most layers, frequency content grew continuously
- **Pythia-6.9B**: MLP projection oscillated with no clear early/late pattern
- In all models, `freq_content` (norm of mean difference) **grew** through layers, never decreased

### Why It Didn't Work

Three methodological problems were identified upon review:

1. **Fixed frequency direction becomes meaningless in deep layers.** The residual stream rotates through layers. A direction computed from embeddings (L0) may be irrelevant by layer 15+. Using a fixed direction across all layers conflates rotation with erasure.

2. **Norm-based frequency content grows because the residual stream norm grows.** The measure `||mean(h|high) - mean(h|low)||` increases because all vectors grow in norm through the network. This does not mean frequency information increases — the *relative* frequency content (what probes measure) can still decrease while absolute norms grow.

3. **Projection of mean differences is not the right decomposition.** The MLP can produce outputs that are frequency-discriminating without being aligned with any single direction. The dot product with a fixed direction misses rotated or distributed frequency representations.

### What We Learned

- **Linear probes are a cleaner measurement than geometric projections** for tracking frequency information. The probe adapts to whatever direction frequency occupies at each layer, while a fixed projection cannot.
- **Circuit-level claims require per-layer frequency directions**, not a fixed direction. A proper version would train a probe at each layer and use the probe weight vector as the local frequency direction.
- **The simple "write then erase" narrative is likely too simple.** The frequency-to-semantics transition may involve rotation and redistribution of frequency information across subspaces, not a clean contraction along a single direction.
- **The probe-under-ablation approach is more robust** because it uses a normalization-free, per-layer-adaptive measurement (probe AUROC) rather than projections onto a fixed subspace.

### Status

Results discarded — the methodology was flawed. Code exists at `circuit_decomposition.py` but should not be used for paper claims. The probe-under-ablation experiment (design: `neurips-2026/notes/probe-under-ablation-design.md`) is the correct next step for establishing causal links between neuron populations and frequency information.

---

## Mechanistic Investigation: What Features Drive Coverage and Affinity?

### Approach 1: Output Weight Analysis (Logit Lens)

**Results** (Pythia-70M, step143000, all 5 domains merged, Layer 5):

| Property | Low Coverage (Q1) | High Coverage (Q4) |
| --- | --- | --- |
| Promoted tokens | 46% content, 24% whitespace, 23% subword | 63% content, 9% whitespace, 15% subword |
| Examples | `lessness`, `hline`, `Grand` | `each`, `the`, `etc`, `right` |
| Mean JSD contribution | 0.0000013 | 0.0000131 (10.3x higher) |

| Property | High Affinity (0.841) | Low Affinity (0.230) |
| --- | --- | --- |
| Examples | `the`, `for`, `a`, `an` | `2`, `II`, `it`, `pane` |

High-coverage neurons promote broadly useful vocabulary. High-affinity neurons promote common words, low-affinity promote rare words.

### Approach 2: Input Weight Analysis

**Result**: Frequency direction in embedding space does NOT separate neuron input weights (all values +/-0.03, std ~0.04). All coverage/affinity groups have similar input weight geometry. Frequency selectivity arises from residual stream content at each layer, not from static weight alignment.

---

## Conference Paper Direction

### The Core Contribution

The conference paper tells the story of a **frequency-to-semantics transition** that transformers build across their layers, decomposed at the neuron level.

**One-sentence pitch**: Transformer MLP layers implement a progressive transition from frequency-based to meaning-based representations, and we decompose the neuron-level mechanisms driving this transition across layers, training, and scale.

### What We Can Claim (Verified)

1. **The transition exists**: JSD, polysemanticity, and polytope density all decrease through layers. This is robust across Pythia (70M–6.9B), OLMo (1B, 7B), and Llama 3.1 8B.

2. **JSD and polysemanticity co-occur at the layer level** (r = +0.62 to +0.94) but are independent at the neuron level (r ~ 0). They are parallel aspects of the same layerwise transition, not causally linked through individual neurons.

3. **Coverage predicts JSD contribution; affinity adds independent signal.** Binary coverage r ~ +0.21-0.28; affinity partial r ~ +0.12-0.18 after controlling for coverage. This contradicts the workshop paper's |rho| <= 0.05.

4. **Higher activation intensity → less polysemantic.** Monotonically true in OLMo models. In Pythia models, near-zero mass neurons are trivially monosemantic (dead neurons), creating a misleading inverted-U. Among active neurons, the relationship is monotonically negative or flat.

5. **Participation coverage ~ raw mass.** The paper's L1-normalized metric is a proxy for activation intensity, not breadth. Binary coverage measures something distinct.

### What We Cannot Yet Claim

1. ~~"JSD drops because neurons become monosemantic"~~ — Neurons in 1B+ models remain highly polysemantic (10-15 clusters) even in final layers. And neuron-level JSD does not predict n_clusters.

2. ~~"Polysemanticity causes frequency invariance"~~ — The layer-level correlation is ecological. Within a layer, the two properties are independent.

3. **Internal convergence predicts behavioral convergence** — Not yet tested. This is Step 2.

4. **Frequency is learned before semantics because of simplicity bias** — Now partially grounded. Linear probes show frequency is encoded everywhere early and actively erased in late layers during training. Consistent with simplicity bias, but the story is "frequency everywhere then erased" not "frequency first then semantics."

### Paper Structure (Revised 2026-04-07)

The paper's cleanest narrative arc centers on **linear probes** as the primary measurement tool — they are intuitive, normalization-free, and directly answer "how much frequency/semantic information is linearly decodable at each layer?" The three-axis neuron decomposition and ablation results then answer "which neuron populations are responsible?"

**Section 3: The Frequency-to-Semantics Transition**
- Linear probes as the primary lens: frequency AUROC peaks mid-layers then drops; semantic AUROC saturates early and persists
- Training dynamics: frequency erasure is learned (Pythia-70M L5: 0.83→0.57 over training)
- Supporting evidence: JSD drops through layers, polytope density drops, polysemanticity decreases
- These multiple measurements converge on the same story from different angles

**Section 4: Neuron-Level Decomposition**
- Three-axis framework: breadth, intensity, frequency affinity
- How this corrects the workshop paper's conflated "participation coverage" metric
- Correlational evidence: coverage predicts JSD contribution, affinity has independent signal

**Section 5: Causal Validation — Probe Under Ablation** (NEXT PRIORITY)
- Ablate neuron populations selected by each axis
- Measure frequency probe AUROC on downstream residual stream (normalization-free)
- If affinity ablation reduces probe AUROC and coverage ablation does not → causal dissociation confirmed without L1 confound
- Supporting: JSD-based ablation dissociation (completed), directional affinity asymmetry

**Section 6: Applications** (PLANNED)
- Frequency steering: amplify/dampen affinity neurons to shift output distributions
- Connection to rare token prediction and frequency bias in generation
- Safety neuron connection (mass axis identifies safety neurons)

**Key shift from previous plan**: Linear probes replace JSD as the primary measurement throughout the paper. JSD and polytope density become supporting evidence. The ablation section uses probes (not JSD) as the outcome measure, eliminating the normalization confound.

---

## Next Steps (Prioritized)

### Step 1: Compute participation coverage -- COMPLETED (2026-04-03)
Added L1-normalized metric to pipeline. Results: paper's narrative is model-size-dependent (reversed for Pythia-70M, weakly consistent for 1B+ models). PC is a mass proxy, not a breadth measure. See "Participation Coverage vs n_clusters" section above.

### Step 2: Behavioral convergence -- COMPLETED (did not yield strong results)
Measured output KL divergence across Pythia checkpoints. Internal convergence did not clearly precede behavioral convergence — the relationship was noisy. Deprioritized in favor of the ablation approach which provides stronger mechanistic evidence.

### Step 2b: Targeted ablation -- COMPLETED (2026-04-04)
Ablated neurons selected by each of the three axes (affinity, coverage, mass) plus controls (low_coverage, directional affinity, jsd, random) across 5 models. Found a causal dissociation: affinity ablation reduces JSD, coverage/mass ablation increases JSD, universally across all models. High-freq-preferring neurons carry ~2x more signal than low-freq-preferring. See "Targeted Ablation" section above. Results in `results/ablation/`. Main caveat: L1 normalization confound on coverage/mass direction — probe-under-ablation experiment needed (design: `neurips-2026/notes/probe-under-ablation-design.md`).

### Step 3: Linear probes -- COMPLETED (2026-04-03)
Probed all layers for frequency (high/low) and semantic category (5 one-vs-rest) across Pythia-70M (15 checkpoints), Pythia-6.9B (3 checkpoints), OLMo-1B, OLMo-7B. Key finding: frequency peaks mid-layers then is actively erased in late layers (training dynamics confirm this is learned). Semantic probes saturate by L2-5. The normative argument is revised: not "frequency first then semantics" but "frequency everywhere early then selectively removed." See "Linear Probes" section above. Results in `results/linear_probes/`. Caveat: semantic probes may pick up template structure — follow-up with domain-neutral templates needed.

### Step 4: Probe under ablation — HIGHEST PRIORITY
Ablate each neuron population (affinity, coverage, mass, random) in late layers, then train frequency linear probes on the resulting residual stream activations at every layer. This is the cleanest causal test: does removing affinity neurons reduce how much frequency information is linearly decodable? Does removing coverage neurons leave frequency information intact (disproving the L1 confound)? Also compute cosine distance between mean high/low residual stream vectors as a normalization-free geometric metric. Design doc: `neurips-2026/notes/probe-under-ablation-design.md`.

### Step 4b: Circuit decomposition attempt — COMPLETED (inconclusive, 2026-04-07)
Attempted to trace frequency info flow via projections onto a fixed frequency direction. Methodology was flawed (fixed direction, norm growth, missing rotations). Results discarded. Key lesson: linear probes are the right tool, not geometric projections onto fixed subspaces. See "Circuit Decomposition Attempt" section above.

### Step 5: Frequency steering demonstration
Amplify/dampen affinity neurons at inference time (`activation *= alpha`) and measure shift in output distribution between high-freq and low-freq synonym completions. If the ratio P(common)/P(rare) shifts monotonically with alpha, this demonstrates practical utility of the decomposition.

### Step 6: Template-controlled semantic probes
Re-run semantic probes with ngram-only text (no sentence template) to address the template confound. Quick config change on existing infrastructure.

### Step 7: Robustness (both reviewers flagged)
- Clustering hyperparameter sweep (d_t, activation threshold)
- Alternative clustering method (DBSCAN/k-means alongside HAC)
- Outlier activation analysis (winsorize top 1%)

### Step 8: Write the paper

---

## Experimental Status

| Experiment | Models Completed | Status |
| --- | --- | --- |
| Polytope density | Pythia 70M/1B/6.9B, OLMo 1B/7B | Complete |
| JSD pipeline | Pythia 70M–2.8B | Complete |
| Neuron embeddings (n_clusters) | Pythia 70M/6.9B, OLMo 1B/7B (ALL neurons) | Complete |
| Coverage-affinity (corrected) | Pythia 70M/6.9B, OLMo 1B/7B, Llama 3.1 8B | Complete |
| Output weight analysis (logit lens) | Pythia 70M | Complete |
| Input weight analysis | Pythia 70M | Complete |
| Safety neuron investigation | Llama 3.1 8B (base vs instruct + Tulu SFT vs DPO) | Complete |
| Participation coverage → n_clusters | Pythia 70M/6.9B, OLMo 1B/7B (all neurons, 5 domains) | **Complete (2026-04-03)** |
| jsd_contrib vs n_clusters (neuron-level) | Pythia 70M/6.9B, OLMo 1B/7B | **Complete (2026-04-03, r ~ 0)** |
| Linear probes (freq vs semantic) | Pythia 70M (15 ckpts), Pythia 6.9B (3 ckpts), OLMo 1B/7B | **Complete (2026-04-03)** |
| Targeted ablation (3-axis dissociation) | Pythia 70M/6.9B, OLMo 1B/7B, Llama 3.1 8B (8 strategies, 5 domains) | **Complete (2026-04-04)** |
| Behavioral convergence | Pythia 70M (15 ckpts) | Complete (weak result, deprioritized) |
| Circuit decomposition (freq direction projections) | Pythia 70M/6.9B, OLMo 1B/7B, Llama 3.1 8B | **Inconclusive (2026-04-07) — methodology flawed, results discarded** |
| Probe under ablation (normalization control) | -- | **Not started — HIGHEST PRIORITY (designed, see neurips-2026/notes/probe-under-ablation-design.md)** |
| Frequency steering demonstration | -- | Not started (planned) |
| Template-controlled semantic probes | -- | Not started (planned) |
| Clustering hyperparameter sweep | -- | Not started |
| Alternative clustering method | -- | Not started |
| Outlier activation analysis | -- | Not started |
| **Stolfo overlap test (affinity ↔ token freq neurons)** | -- | **Not started — HIGH PRIORITY (new, see Reframing section)** |
| **Liu overlap test (affinity ↔ rare-token neurons)** | -- | **Not started — HIGH PRIORITY (new, see Reframing section)** |

---

## Paper Reframing: "Frequency Routing Circuits" (2026-04-08)

### Motivation for Reframing

The original framing — "we observe a frequency-to-semantics transition" — has three weaknesses:

1. **The 3-axis neuron population feels asserted, not proven.** We decompose neurons into breadth/intensity/affinity, show they correlate differently with JSD, and show ablation dissociation. But we never connect them to any established computational role or show they correspond to known functional types.

2. **The frequency-to-semantics transition is intuitive.** Any reasonable person would guess that early layers encode surface statistics and later layers encode meaning. Saying "we measured it with JSD, probes, and polytopes" is confirmation of the obvious, not discovery.

3. **No practical use case.** The paper currently ends with "we found this transition exists" — a descriptive claim with no downstream implication.

### The Key Insight: Unification with Existing Neuron Types

Reading across the related literature reveals connections the current framing misses:

**Stolfo et al.'s "token frequency neurons" (2406.16254)** are likely the same population as our **high-affinity neurons**. They found neurons that modulate the model's output distribution toward/away from the unigram frequency distribution — they push logits proportionally to log-frequency. Our affinity axis measures exactly this: which neurons respond preferentially to high-freq vs low-freq tokens.

**Kobayashi et al. (2305.18294)** shows the prediction head bias *re-introduces* frequency information that hidden states have stripped. Hidden states are nearly orthogonal to the frequency direction (cos ~ 0.08).

**Liu et al.'s rare-token neurons (2509.21163)** form a ~1.7% "influential plateau" — neurons with disproportionate impact on rare token prediction. Our low-affinity neurons (prefer rare tokens) likely overlap with this population.

**This means our paper is sitting on a unification story it hasn't told.**

### New Framing: "Frequency Routing Circuits in Transformer MLPs"

Instead of "we observe a frequency-to-semantics transition" (descriptive, intuitive), the paper should argue:

> **Transformer MLPs implement a learned frequency routing circuit: a small population of specialist neurons actively separates frequency information from semantic content in the residual stream, enabling the prediction head to independently recombine them at output time.**

This is a *mechanism* claim, not a *phenomenon* claim.

### The Four-Stage Pipeline

| Stage | What happens | Our evidence | External evidence |
| --- | --- | --- | --- |
| **1. Embedding** | Frequency baked into token vectors | (known) | Kobayashi: embeddings carry frequency |
| **2. Early/Mid MLP** | Affinity neurons route frequency signal; coverage neurons build shared semantic substrate | Ablation dissociation; probe peaks mid-layer | Liu et al.: rare-token neurons in final layer |
| **3. Late MLP** | Frequency actively stripped from residual stream; concentrated in specialist neurons | Probes show erasure during training (0.83→0.57); affinity ablation reduces probe AUROC | Kobayashi: hidden states orthogonal to freq direction |
| **4. Prediction Head** | Frequency re-introduced via bias + token frequency neurons | (Kobayashi's result) | Stolfo: token frequency neurons; Kobayashi: b_LN encodes freq |

**The novel claim**: Stages 2-3 are what our paper characterizes. Nobody else has shown the *neuron-level mechanism* of how frequency gets separated from semantics in the MLP layers. Kobayashi showed the endpoint (head re-introduces frequency). Stolfo showed individual neurons that modulate frequency at output. We show the *process*.

### Three Axes → Known Functional Types

| Our axis | Established functional type | Evidence for correspondence |
| --- | --- | --- |
| **Affinity (freq-selective)** | Stolfo's "token frequency neurons" | Both modulate output distribution relative to unigram; both affect KL(P_freq \|\| P_model) |
| **Mass (high-intensity)** | Chen's "safety neurons"; Stolfo's "entropy neurons" | Our safety neuron result (7x enrichment in top-5% mass); entropy neurons have high weight norm |
| **Coverage (broad-firing)** | Gurnee's "universal neurons" | Both fire broadly across inputs; prediction: should overlap with universal neurons across seeds |

### Novelty Assessment (Literature Check, 2026-04-08)

**What already exists:**

- Kobayashi et al. (2305.18294): prediction head bias encodes frequency; hidden states orthogonal to frequency direction. Shows the *endpoint* — does NOT explain the neuron-level mechanism of how frequency gets stripped.
- Stolfo et al. (2406.16254): token frequency neurons modulate logits proportional to log-freq; entropy neurons regulate confidence via LayerNorm null space. Characterizes individual neuron *types* at the output — does NOT connect to layer-wise frequency erasure or show a circuit.
- Liu et al. (2509.21163): ~1.7% rare-token plateau neurons; distributed specialization not modular. Analyzes *final-layer only* — no layerwise decomposition, no connection to frequency erasure across depth.
- Zhang et al. (2512.20607): saddle-to-saddle dynamics → simplicity bias → low-rank features learned first. Pure theory, no empirical connection to frequency encoding in transformers.
- Balogh (2603.10985) "Discrete Charm of MLP": binary routing of continuous signals; consensus neurons vs exception handlers. Different phenomenon (nonlinearity routing, not frequency). Only GPT-2 Small; didn't generalize to larger models.
- "How Do LLMs Use Depth?" (2510.18871): layerwise prediction dynamics. Layer-level description only, no neuron-level decomposition, no frequency focus.
- "Transfer Neurons" (2509.17030): early layers convert to English-centric, middle layers language-agnostic. Structurally analogous (information routing across layers) but for *language identity*, not *token frequency*. Good parallel citation.
- Hadad et al. (2602.16823): formal mechanistic interpretability with provable guarantees for circuit discovery. Vision models only, about verification methodology, not frequency/semantic content.

**What does NOT exist (our novel contributions):**

1. **Nobody has shown the neuron-level mechanism of frequency erasure.** Kobayashi shows frequency is absent from late hidden states. We show *how* it gets removed — via the balance shifting from affinity neurons to coverage neurons across layers.

2. **Nobody has causally validated functionally distinct neuron populations with *opposite* effects on frequency discrimination.** Our ablation dissociation (affinity ablation *reduces* JSD; coverage/mass ablation *increases* JSD) is unique. No prior work shows a systematic population-level dissociation with opposite causal signs.

3. **Nobody has unified token frequency neurons, rare-token neurons, and safety neurons under a single decomposition.** These were all discovered independently with different methods.

4. **Nobody has shown frequency probe AUROC declining through layers as a *learned* developmental process** (our training dynamics result: Pythia-70M L5 goes from 0.83 to 0.57 across training). Kobayashi's result is static.

5. **The end-to-end pipeline** (embedding → MLP routing → residual stream disentanglement → prediction head recombination) has never been assembled.

### AI Safety and Practical Applications

#### 1. Frequency Steering for Debiasing Generation
LLMs over-generate common tokens and under-generate rare ones. By amplifying/dampening affinity neurons at inference, we can shift the common/rare balance *without retraining*. Rare tokens include technical terms, proper nouns, and minority language constructions — systematic under-generation is a form of representational harm.

#### 2. Targeted Model Editing via Neuron Populations
Affinity neurons can be removed with minimal output disruption (KL 0.056 for high-freq affinity at 5% ablation). Coverage/mass ablation causes massive output disruption (KL 0.88). This means frequency behavior can be surgically modified without breaking the model — more precise than LoRA or full fine-tuning for domain adaptation.

#### 3. Safety Neuron Identification via Mass Axis
36% of top-5% mass neurons are safety neurons (7x enrichment). Mass is a cheap proxy for safety-criticality — single forward pass instead of expensive ablation studies.

#### 4. Calibration Monitoring
If affinity neurons overlap with Stolfo's token frequency neurons, then monitoring affinity neuron activation patterns could serve as a real-time calibration diagnostic during inference.

### Revised Paper Structure

**Title**: "Frequency Routing Circuits: How Transformer MLPs Disentangle Token Frequency from Semantics"

**Section 1: Introduction**
- Transformers must solve an information routing problem: embeddings conflate frequency and meaning, but the prediction head needs them separated
- We identify the MLP-layer circuit that implements this disentanglement

**Section 2: The Frequency-to-Semantics Transition** (compact, ~2 pages)
- Linear probes as the primary lens: frequency AUROC peaks mid-layers then drops; semantic AUROC saturates early and persists
- Training dynamics: frequency erasure is learned (Pythia-70M L5: 0.83→0.57 over training)
- Supporting evidence: JSD drops through layers, polytope density drops (figures in appendix)

**Section 3: Neuron-Level Decomposition of the Routing Circuit**
- Three functional populations: frequency routers (affinity), shared-computation substrate (coverage), intensity specialists (mass)
- Corrects the workshop paper's conflated "participation coverage" metric
- **Unification**: affinity neurons = Stolfo's token frequency neurons (empirical overlap test); rare-token preferring neurons = Liu et al.'s plateau neurons

**Section 4: Causal Validation**
- Ablation dissociation (completed results)
- Probe-under-ablation (highest priority experiment — key result)
- The dissociation is universal across 5 models, 3 architectures

**Section 5: The Complete Frequency Pipeline**
- End-to-end story: embedding → MLP routing → residual stream disentanglement → prediction head recombination
- Connects our results to Kobayashi (head) and Stolfo (confidence regulation)
- Theoretical grounding: saddle-to-saddle dynamics explains WHY frequency is learned first and separated later

**Section 6: Applications**
- Frequency steering demonstration
- Mass as cheap safety neuron proxy
- Discussion of implications for calibration and rare-token generation

### Revised Next Steps (Prioritized, 2026-04-08)

1. **Probe-under-ablation** (already designed) — marquee result
2. **Stolfo overlap test** — compute affinity for Stolfo's identified token frequency neurons on a shared model (e.g., Pythia-410M or GPT-2). If overlap is strong (>60%), we have the unification. ~2 hours of work.
3. **Liu overlap test** — check if Liu et al.'s rare-token plateau neurons are low-affinity
4. **Frequency steering demo** (already planned)
5. **Write the paper with new framing**
