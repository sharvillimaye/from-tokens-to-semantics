# Frequency Routing Circuits: How Transformer MLPs Disentangle Token Frequency from Semantics

**Rough Draft / Readable Breakdown — April 2026**

*Limaye\*, Ramesh\*, Rohweder, Zhou\*, Panda, Bhaskar, Sharma*

---

## Abstract (Draft)

Transformer language models receive token embeddings that conflate lexical frequency with semantic content, yet their prediction heads require these signals separated — frequency re-enters through bias parameters while hidden states remain nearly orthogonal to the frequency direction. We identify the MLP-layer circuit that implements this disentanglement. Through linear probes, neuron-level decomposition, and targeted ablation across five models (Pythia 70M–6.9B, OLMo 1B/7B, Llama 3.1 8B), we show that MLP layers contain three functionally distinct neuron populations: **frequency-selective specialists** (high affinity) that carry frequency information, **shared-computation neurons** (high coverage) that build frequency-invariant semantic representations, and **high-intensity neurons** (high mass) that regulate output confidence. Ablating these populations produces opposite causal effects on frequency discrimination — a dissociation universal across three architectures. Frequency information peaks mid-network and is actively erased from the residual stream during training, with the erasure intensifying over the course of pretraining. Our decomposition unifies several independently discovered neuron types — token frequency neurons, rare-token neurons, and safety neurons — under a single framework, and enables practical applications including surgical frequency steering for debiasing generation.

---

## 1. Introduction

### The Problem

When a transformer processes "happy" and "elated" — words with the same meaning but very different corpus frequencies — the embedding layer assigns them very different vectors. Yet by the final layers, the model must treat them similarly to make correct predictions. Somewhere across the layers, frequency-specific representations must give way to semantic ones.

This is not just a curiosity. Kobayashi et al. (2023) showed that the prediction head's bias parameter (`b_LN`) actively re-introduces frequency information into output logits, and that hidden states before the prediction head are nearly orthogonal to the frequency direction (cosine similarity ~ 0.08). This means the model has *already stripped frequency from its internal representations* by the time it reaches the prediction head, and then the head puts it back.

**The question nobody has answered: how do the MLP layers actually implement this stripping?**

### What We Find

We decompose this process at the neuron level across five models and three architectures:

1. **Linear probes reveal active frequency erasure.** Frequency information peaks in mid-layers (AUROC 0.97+) then declines sharply in late layers. In Pythia-70M, final-layer frequency probe AUROC drops from 0.83 to 0.57 over the course of training — the model *learns* to erase frequency. Meanwhile, semantic category remains perfectly decodable throughout.

2. **Three functionally distinct neuron populations drive the transition.** We decompose MLP neurons along three axes — firing breadth (binary coverage), activation intensity (raw mass), and frequency preference (affinity) — and show via targeted ablation that they have *opposite causal effects* on frequency discrimination. Ablating frequency-selective neurons reduces frequency discrimination; ablating broadly-active neurons increases it. This sign split is universal across all five models.

3. **The decomposition unifies independently discovered neuron types.** Our affinity axis corresponds to Stolfo et al.'s "token frequency neurons" (modulate output toward unigram distribution). Our mass axis identifies Chen et al.'s "safety neurons" (7x enrichment). Our coverage axis corresponds to Gurnee et al.'s "universal neurons." Three separate discoveries, one organizing framework.

4. **The full pipeline: embed → route → strip → recombine.** We assemble the first end-to-end account of frequency information flow: embeddings carry frequency, early MLPs begin routing it through specialist neurons, late MLPs strip it from the residual stream (concentrating it in a minority of affinity specialists), and the prediction head recombines frequency via its bias parameter.

### Why This Matters

- **For mechanistic interpretability**: We provide the first neuron-level mechanism for a macro-level phenomenon (frequency erasure) that was previously only described at the layer level.
- **For AI safety**: The mass axis cheaply identifies safety-critical neurons (single forward pass vs. expensive ablation).
- **For practical NLP**: Frequency-selective neurons can be surgically amplified/dampened to debias generation toward rare tokens without retraining.

---

## 2. The Frequency-to-Semantics Transition

This section establishes the macro-level phenomenon before we decompose its mechanism.

### 2.1 Linear Probes: Frequency Peaks Mid-Network Then Is Erased

We train binary linear probes on residual stream activations at each layer to predict whether a token is high-frequency or low-frequency (using the ScaleJSD synonym pair dataset: 291 pairs across 5 domains where each pair shares a sentence template, isolating the frequency signal).

**Key results:**

| Model | Layers | Freq peak layer (AUROC) | Freq final layer (AUROC) | Drop |
| --- | --- | --- | --- | --- |
| Pythia-70M | 6 | L2 (0.876) | L5 (0.566) | -0.31 |
| OLMo-1B | 16 | L10 (0.978) | L15 (0.895) | -0.08 |
| Pythia-6.9B | 32 | L12 (0.979) | L31 (0.914) | -0.07 |
| OLMo-7B | 32 | L4 (0.984) | L31 (0.857) | -0.13 |

The decline is not "failure to encode" — frequency peaks at 0.97+ in mid-layers of larger models. Late layers **actively remove** frequency information from the residual stream.

Semantic category probes, by contrast, reach AUROC >= 0.99 by layers 2-5 across all models and remain there through the final layer. The model builds semantic representations early and preserves them; it builds frequency representations early and then *destroys* them.

### 2.2 Training Dynamics: Frequency Erasure Is Learned

Tracking Pythia-70M across 15 training checkpoints reveals that frequency erasure intensifies during training:

| Layer | step3000 | step43000 | step93000 | step143000 |
| --- | --- | --- | --- | --- |
| L0 | 0.820 | 0.865 | 0.853 | 0.865 |
| L2 | 0.850 | 0.889 | 0.893 | 0.876 |
| L5 (final) | 0.830 | 0.815 | 0.695 | **0.566** |

Final-layer frequency probe AUROC drops from 0.83 (step 3000, near-random) to 0.57 (step 143000, barely above chance). The model doesn't start with frequency information missing — it *learns to remove it*.

This is consistent with the simplicity bias (Zhang et al., 2025): frequency is a low-rank feature (Zipf's law) that gets encoded everywhere initially. As training progresses, the model discovers it only needs frequency for the output head and strips it from the residual stream to make room for higher-rank semantic features.

### 2.3 Supporting Evidence: JSD and Polytope Density

Two additional metrics confirm the same story from different angles:

**JSD (Jensen-Shannon Divergence):** Per-layer JSD between activation distributions for high-freq vs low-freq tokens decreases monotonically through layers. Pythia-6.9B: L0 JSD = 0.150, L31 JSD = 0.015 (10x reduction).

**Polytope Density:** Hamming distance between synonym pair spline codes / Euclidean distance between their residual stream vectors. Decreases through layers, primarily because late layers route synonyms through the same ReLU regions (fewer boundary crossings) even as raw vector norms diverge. This means late layers achieve "computational equivalence" for synonyms regardless of frequency.

**Polysemanticity:** Mean number of semantic clusters per neuron peaks in early-to-mid layers and decreases through depth. Layer-level correlation with JSD is strong (Spearman r = 0.62–0.94), but neuron-level correlation is near zero (r ~ 0.01–0.09). JSD and polysemanticity are parallel consequences of the same layerwise transition, not causally linked through individual neurons.

---

## 3. Neuron-Level Decomposition of the Routing Circuit

The layer-level story (Section 2) tells us *what* changes with depth. This section asks: **which neurons inside each layer are responsible?**

### 3.1 The Three Axes

We decompose MLP neuron behavior along three axes:

**1. Firing breadth (binary coverage)** — fraction of input phrases where the neuron fires (activation > 0).
- Coverage = 0.9: broad, general-purpose neuron
- Coverage = 0.05: narrow, selective neuron

**2. Activation intensity (raw mass)** — mean(ReLU(activation)) across phrases, unnormalized.
- How strongly does this neuron fire on average?
- Independent of how broadly it fires.

**3. Frequency affinity** — ratio of mean activation for high-frequency vs total: `raw_mass_high / (raw_mass_high + raw_mass_low)`.
- Affinity near 1.0: prefers common words
- Affinity near 0.0: prefers rare words
- Affinity near 0.5: no frequency preference

### 3.2 Why Not the Workshop Paper's "Participation Coverage"?

The workshop paper (Limaye et al., NeurIPS 2025 MI Workshop) used **participation coverage** — an L1-normalized activation share — as its primary neuron metric. We show this metric is fundamentally flawed:

- Participation coverage ≈ raw mass (correlations differ by <0.01 everywhere). The L1 normalization turns it into an intensity proxy, not a breadth measure.
- L1 normalization introduces inter-neuron competition: if one neuron dominates, others are mechanically suppressed, inflating coverage and masking the affinity signal.
- With binary coverage (pure breadth), affinity shows significant independent signal (partial r = 0.12–0.18 after controlling for coverage) — contradicting the workshop paper's |rho| <= 0.05.

### 3.3 What Predicts Frequency Discrimination?

Across **Pythia 70M–6.9B, OLMo 1B/7B, and Llama 3.1 8B** (three architectures, 5 domains):

| Metric | All layers | Deep layers |
| --- | --- | --- |
| Binary coverage → JSD contribution | r = +0.21 | r = +0.28 |
| Affinity → JSD contribution | r = +0.14 | r = +0.22 |
| **Partial (Affinity → JSD | Coverage)** | **r = +0.12** | **r = +0.18** |
| Layers where partial p < 0.05 | 79% | 89% |

Binary coverage is the stronger predictor, but affinity has significant independent signal — especially in deep layers of large models where coverage drops near zero while affinity rises. Large models increasingly rely on frequency-specialized neurons.

### 3.4 Connecting to Known Neuron Types

Each axis maps onto an independently discovered functional type:

**Affinity ↔ Token frequency neurons (Stolfo et al., 2024).** Stolfo et al. found neurons that boost/suppress logits proportionally to log-frequency, shifting the output toward or away from the unigram distribution. Our affinity axis measures exactly the same thing from the input side — which neurons respond preferentially to high-freq vs low-freq tokens. **Prediction: high overlap between top-affinity neurons and Stolfo's identified token frequency neurons.**

**Mass ↔ Safety neurons (Chen et al., 2024).** We tested this directly on Llama 3.1 8B using activation contrasting between base vs instruct models. Results:
- 36% of safety neurons are in the top 5% by raw mass (7x enrichment)
- Safety neurons have dramatically higher mean activation (0.089 vs 0.015, d = +0.45)
- Safety neurons are NOT distinguished by breadth (coverage: 0.497 vs 0.500) or frequency preference (affinity: 0.495 vs 0.500)

**Mass ↔ Entropy neurons (Stolfo et al., 2024).** Entropy neurons have unusually high weight norms and regulate confidence via LayerNorm. High weight norm → high activation mass. Our mass axis likely captures both safety neurons and entropy neurons as "intense specialists."

**Coverage ↔ Universal neurons (Gurnee et al., 2024).** Universal neurons (1-5% of neurons consistent across random seeds) fire broadly across inputs. Our high-coverage neurons fire for 90%+ of inputs. **Prediction: significant overlap.**

**Low affinity ↔ Rare-token neurons (Liu et al., 2025).** Liu et al.'s ~1.7% "influential plateau" neurons disproportionately affect rare token prediction. Our low-affinity neurons preferentially respond to rare tokens. **Prediction: plateau neurons are low-affinity.**

### 3.5 What Predicts Specialization?

**Higher activation intensity → fewer semantic roles (less polysemantic).** This is monotonically true in OLMo models. In Pythia, near-zero-mass neurons create a misleading inverted-U because dead neurons are trivially monosemantic. Among active neurons, the relationship is monotonically negative or flat.

This connects to the safety neuron finding: safety neurons are intense specialists (high mass, few clusters) that fire strongly for their specific function.

---

## 4. Causal Validation: The Ablation Dissociation

Correlation is not causation. If the three axes identify genuinely different neuron populations, ablating neurons selected by each axis should produce different causal effects.

### 4.1 Method

Zero-ablation of post-activation MLP intermediate for selected neurons in the last third of each model. Eight selection strategies: affinity (both directions + combined), coverage (high + low), mass, jsd (upper bound), random (null control). Five models, five datasets, ablation percentages 1–20%.

### 4.2 The Core Result: Opposite Causal Signs

At 5% ablation across all 5 models and 5 datasets:

| Strategy | JSD change | Output KL | Interpretation |
| --- | --- | --- | --- |
| **affinity** | **-7.1%** | 0.14 | Frequency-selective neurons carry the frequency signal |
| **affinity_high_freq** | **-6.4%** | 0.056 | High-freq-preferring: most signal, least disruption |
| **affinity_low_freq** | **-5.0%** | 0.226 | Low-freq-preferring: less signal, more disruption |
| **coverage** | **+2.2%** | 0.60 | Removing broad neurons *exposes* frequency differences |
| **mass** | **+2.1%** | 0.88 | Removing intense neurons disrupts outputs most |
| **low_coverage** | -1.5% | 0.15 | Narrow specialists carry weak frequency signal |
| jsd | -10.0% | 0.12 | Upper bound |
| random | ~0% | 0.03 | Null |

**The sign split is the key result.** All frequency-selectivity strategies (affinity, directional affinity, low_coverage) *reduce* JSD when ablated. All breadth/intensity strategies (coverage, mass) *increase* JSD. Random is null. This is universal across all 5 models and 3 architectures (GPT-NeoX, OLMo, LLaMA).

This upgrades our correlational findings to causal ones: the three axes identify neuron populations with **opposite functional roles**.

### 4.3 Asymmetry: High-Freq Neurons Are More Separable

High-freq-preferring neurons account for more JSD reduction (-6.4%) than low-freq-preferring (-5.0%) at 5%, but cause much less output disruption (KL 0.056 vs 0.226). The model dedicates more routing capacity to "this is a common word" and this routing is more separable from the model's primary predictions. This connects to the finding that models optimize primarily on frequent tokens (2508.15390).

### 4.4 Why Coverage Ablation Increases JSD

Removing broad-firing neurons removes a shared activation floor common to both high-freq and low-freq inputs. This makes the remaining frequency-selective neurons more prominent in the JSD computation. The coverage neurons are not adding frequency information — they're providing the shared substrate that makes synonyms look similar despite their frequency differences.

**Important caveat:** JSD is computed on L1-normalized distributions. Part of the coverage/mass effect may be renormalization artifact. The probe-under-ablation experiment (next section) tests this with a normalization-free metric.

### 4.5 Universality

The sign split holds across every model tested:

| Model | affinity | coverage | mass | random |
| --- | --- | --- | --- | --- |
| Pythia-70M | - | + | + | ~0 |
| OLMo-1B | - | + | + | ~0 |
| Pythia-6.9B | - | + | + | ~0 |
| OLMo-7B | - | + | + | ~0 |
| Llama 3.1 8B | - | + | + | ~0 |

---

## 5. The Complete Frequency Pipeline

Assembling our results with prior work yields the first end-to-end account of frequency information flow through transformers:

### Stage 1: Embedding (known)
Token embeddings carry frequency information. Words with similar frequency cluster in embedding space. This is well-established in prior work.

### Stage 2: Early/Mid MLP Layers (our contribution)
MLP layers begin separating frequency from semantics. Affinity neurons activate differentially for high-freq vs low-freq tokens, routing frequency information through a specialist pathway. Coverage neurons build a shared computational substrate that treats synonyms similarly regardless of frequency. Frequency probe AUROC peaks in mid-layers (0.97+ in larger models) — the frequency signal is still present and strong, just increasingly concentrated in specialist neurons rather than distributed broadly.

### Stage 3: Late MLP Layers (our contribution)
The balance shifts: coverage/shared neurons dominate the residual stream while affinity/frequency specialists become a shrinking minority. A linear probe on the full residual stream shows declining frequency AUROC because the probe reads from a space now dominated by frequency-invariant neurons. But the frequency information hasn't been destroyed — it's concentrated in the minority of affinity specialists that the prediction head knows how to read.

Training dynamics confirm this is learned: Pythia-70M final-layer frequency AUROC drops from 0.83 (early training) to 0.57 (late training). The model learns to consolidate frequency routing into fewer specialists and expand shared-computation capacity.

### Stage 4: Prediction Head (Kobayashi et al., 2023)
The prediction head re-introduces frequency via its bias parameter `b_LN`, which correlates strongly with corpus word frequency (Spearman rho = 0.78 on GPT-2). The bias direction in the output embedding space IS the frequency direction. Hidden states arriving at the head are nearly orthogonal to this direction (cos ~ 0.08) — exactly as our probe results predict.

**Stolfo et al.'s token frequency neurons** operate at this stage too — they modulate logits proportionally to log-frequency, shifting the output distribution toward or away from unigram. These may be the output-side counterparts of our affinity neurons (the input-side frequency specialists).

### Theoretical Grounding

Zhang et al. (2025, "Saddle-to-Saddle Dynamics") show that SGD provably learns simple, low-rank features first due to simplicity bias. Frequency is a low-rank feature (Zipf's law makes it a simple function). Semantics requires higher-rank representations. This predicts:
- Frequency is encoded everywhere initially (low-rank, easy to learn)
- As training continues, the model discovers it only needs frequency at the output
- It learns to strip frequency from the residual stream, concentrating it in specialist neurons

Our training dynamics (Section 2.2) are consistent with this prediction.

---

## 6. Applications

### 6.1 Frequency Steering for Debiasing Generation

LLMs systematically over-generate common tokens. Kobayashi et al. showed that dampening `b_LN` increases generation diversity while maintaining quality (in larger models). Our decomposition enables the *internal* version of this:

**Method:** At inference, scale affinity neuron activations by factor alpha. If alpha > 1, amplify frequency routing (push outputs toward frequency-biased distribution). If alpha < 1, dampen it (push toward frequency-neutral).

**Expected result:** The ratio P(common synonym)/P(rare synonym) should shift monotonically with alpha. If this works, it provides a surgical intervention for improving rare-token generation in specialized domains (medical, legal, scientific) where rare terms carry critical information.

**Advantage over Kobayashi's bias manipulation:** Targeting affinity neurons modifies the *internal* representation, not just the output. This could affect downstream reasoning about rare tokens, not just their generation probability.

### 6.2 Mass as a Cheap Safety Neuron Proxy

Chen et al.'s method for identifying safety neurons requires expensive ablation studies across many prompts. Our finding that raw activation mass identifies the same neurons (7x enrichment at top-5%) provides a computationally cheap alternative:

1. Run one forward pass on any input
2. Compute mean ReLU activation per neuron
3. Top-5% by mass are 36% safety neurons

This enables monitoring safety-critical neurons during inference and rapid screening of fine-tuned models for safety neuron integrity.

### 6.3 Calibration Diagnostics

If our affinity neurons correspond to Stolfo et al.'s token frequency neurons (which regulate confidence by shifting the output toward/away from unigram), then monitoring affinity neuron activation patterns could serve as a real-time calibration diagnostic. Sudden shifts in affinity neuron behavior could signal distribution shift or adversarial inputs.

---

## 7. Limitations

1. **GELU approximation.** Polytope density is strictly defined for piecewise-linear activations. Pythia/OLMo use GELU. The Hamming code is a reasonable proxy but this is an approximation.

2. **Template confound in semantic probes.** Semantic probes likely achieve near-perfect AUROC partly because sentence templates carry domain-specific lexical cues. The frequency probe is more informative because both high/low variants share the same template. Follow-up with domain-neutral templates needed.

3. **L1 normalization confound in ablation JSD.** The coverage/mass ablation results (JSD increase) may be partly a renormalization artifact. The probe-under-ablation experiment addresses this with a normalization-free metric.

4. **Unification claims are predictions.** The mapping between our axes and Stolfo/Liu/Gurnee neuron types is argued on conceptual grounds. The empirical overlap tests (affinity ↔ token frequency neurons, low-affinity ↔ rare-token neurons) are designed but not yet run.

5. **ScaleJSD dataset scope.** Our results use 291 synonym pairs across 5 domains. While results are consistent across all domains and multiple model families, the synonym pairs are English-only and may not capture all aspects of frequency processing.

6. **Steering demonstration not yet run.** The frequency steering application (Section 6.1) is designed but experimental validation is pending.

---

## 8. Related Work

### Frequency Processing in Transformers

**Kobayashi et al. (2023)** showed that prediction head biases encode corpus word frequency and that hidden states are nearly orthogonal to the frequency direction. We explain *why* — MLP layers progressively strip frequency information — and identify *which neurons* implement the stripping.

**"Not a nuisance but useful: Outlier dimensions favor frequent tokens" (2503.21718)** found that outlier dimensions in the last layer favor frequent tokens, emerging early in training. Consistent with our finding that a minority of specialist neurons carry frequency signal in late layers.

**"Exploiting Vocabulary Frequency Imbalance" (2508.15390)** showed models reduce loss primarily by improving on frequent tokens, with larger output embedding norms for frequent tokens. This explains our finding that high-freq-preferring neurons carry more signal but less output importance — the model over-invests in common-token routing.

### Neuron Functional Types

**Stolfo et al. (2024)** discovered token frequency neurons (modulate logits proportional to log-frequency) and entropy neurons (regulate confidence via LayerNorm null space). We argue our affinity and mass axes recover these populations from a different measurement perspective, providing a unifying framework.

**Liu et al. (2025)** identified rare-token neurons forming a three-regime influence hierarchy in final MLP layers. We predict these correspond to our low-affinity (rare-preferring) neurons and provide the layerwise context they lack (their analysis is final-layer only).

**Chen et al. (2024)** identified safety neurons (~5% of MLP neurons) via activation contrasting. We showed raw activation mass is the differentiating axis (7x enrichment at top-5%).

**Gurnee et al. (2024)** found 1-5% of neurons are universal across random seeds. We predict these correspond to our high-coverage neurons.

### Layer-wise Information Processing

**Geva et al. (2021)** showed that lower FF layers capture superficial patterns while upper layers capture semantic ones. Our work adds the neuron-level decomposition: *which* neurons in each layer carry *which* type of information.

**"How Do LLMs Use Their Depth?" (2510.18871)** traced layerwise prediction dynamics showing non-uniform depth usage. Our probes provide a specific instance of this for frequency vs. semantic information.

**"Transfer Neurons" (2509.17030)** found that multilingual LLMs route language identity through specific neurons across layers — a structural analog to our frequency routing, but for a different type of surface feature.

### Theoretical Foundations

**Zhang et al. (2025)** proved that SGD exhibits saddle-to-saddle dynamics that learn simple (low-rank) features first. Frequency (governed by Zipf's law) is low-rank; semantics is high-rank. This provides the theoretical basis for why frequency is learned everywhere initially and then selectively removed.

**Rofin et al. (2026)** showed that gradient signal decomposes into direct, pre-cached, and circuit-sharing components, explaining why features not useful for immediate prediction emerge via shared gradients.

### Circuit-Level Interpretability

**Balogh (2026)** proposed "binary routing of continuous signals" as a framework for MLP computation — consensus neurons implement default processing while exception handlers engage for nonlinear cases. This is a different phenomenon (nonlinearity routing vs. frequency routing) but provides a useful conceptual parallel.

**Hadad et al. (2026)** introduced formal guarantees for circuit discovery via neural network verification. Their framework could be applied to formally verify our frequency routing circuit, though currently limited to vision models.

---

## Appendices (Planned)

- **A.** Full model-by-model probe results (all 15 Pythia-70M checkpoints)
- **B.** JSD and polytope density curves (all models, all domains)
- **C.** Ablation results at all percentages (1%, 2%, 5%, 10%, 20%)
- **D.** Polysemanticity (n_clusters) full results
- **E.** Output weight analysis / logit lens results
- **F.** Participation coverage vs binary coverage detailed comparison
- **G.** Safety neuron investigation full results
- **H.** Circuit decomposition attempt (negative result, methodology lessons)