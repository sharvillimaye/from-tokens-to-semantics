# Frequency Routing in Transformer LMs: Research Summary

**Status**: Active paper-in-progress. Previous iteration archived at `research-summary-v1-archived.md` (2026-04-15 rewrite).

**Authors**: Limaye\*, Ramesh\*, Rohweder, Zhou\*, Panda, Bhaskar, Sharma
**Prior workshop paper**: NeurIPS 2025 Mechanistic Interpretability Workshop (poster)
**Models**: Pythia-70M (15 checkpoints), Pythia-6.9B, OLMo-1B, OLMo-7B, Llama-3.1-8B
**Data**: ScaleJSD — 291 synonym pairs across 5 domains (emotion, medical, legal, scientific, verb), each pair a high/low-frequency token with identical meaning (e.g. "happy"/"elated"). Shared sentence templates isolate frequency from semantic content.

---

## 1. Abandoned Framings

Two previous framings are now **officially dropped**. Documenting them here so we don't accidentally resurrect them.

### 1.1 Dropped: Three-axis neuron decomposition (coverage / mass / affinity)

Prior versions of this document proposed decomposing MLP neurons along three "axes" — binary coverage (firing breadth), raw mass (activation intensity), frequency affinity (preference for high vs low freq) — and argued these axes identify neuron populations with distinct causal roles.

**Why we dropped it**:
- The original JSD-based "causal dissociation" between these populations was largely an L1 normalization artifact. With a normalization-free linear probe, ablating 5-10% of neurons selected by any axis produces essentially zero change in frequency decodability (see §3.3).
- Cross-dataset Jaccard overlap of top-5% "frequency writers" (by |diff_proj|) is ~0.06, near chance. There is no consistent population of "frequency neurons" across text domains.
- Top 1% of neurons accounts for only ~9% of |diff_proj|; frequency writing is distributed, not concentrated.
- The axes were descriptive (what neurons do) not functional (what role they play in a circuit). None of them map cleanly to established neuron types in the literature (Stolfo's token-frequency neurons, Gurnee's universal neurons, etc.).

**What remains**: The coverage-affinity experiment code (`coverage_affinity_experiment.py`) is still useful for the per-neuron statistics it computes, but we no longer interpret these as identifying distinct causal populations.

### 1.2 Dropped: Discrete-specialist "frequency circuit"

We initially expected that ablating carefully-selected neurons (by affinity, coverage, mass, or circuit-level |diff_proj|) would surgically disable the frequency circuit. This hypothesis is **falsified** by our experiments:

- Probe-under-ablation on Pythia-70M / Pythia-6.9B / OLMo-1B: affinity ablation produces AUROC drops of 0.000–0.007 (noise-level).
- Circuit-based ablation (selecting by |diff_proj| averaged across datasets, up to 50%): still no meaningful probe AUROC drop.

**Interpretation**: Frequency is not stored in a small set of specialist neurons. Many neurons write into the frequency channel additively and redundantly — we cannot kill the signal by removing any manageable fraction of them.

---

## 2. The Current Finding: Frequency Lives in a 1D Direction

### 2.1 Core claim

Token frequency in transformer residual streams is encoded as (effectively) a single direction. The direction can be surgically removed by rank-1 projection of the probe weight vector, which destroys downstream frequency decodability. Many MLP neurons write into this direction additively; no individual neurons are essential.

### 2.2 Evidence

**Linear probe trajectory** (per-layer frequency AUROC from logistic regression on residual stream):
| Model | Peak layer (AUROC) | Final layer (AUROC) |
| --- | --- | --- |
| Pythia-70M | L2 (0.876) | L5 (0.566) |
| Pythia-6.9B | L12-L13 (0.979-0.983) | L31 (0.914) |
| OLMo-1B | L9-L10 (0.973-0.978) | L15 (0.895) |
| OLMo-7B | L4 (0.984) | L31 (0.857) |

Frequency probe AUROC peaks mid-network then declines in late layers. Semantic probes (one-vs-rest per domain) saturate at 0.99+ by L2-5 and persist. *(Note: semantic probe validity is suspect in all-datasets mode due to a known bug in probe_under_ablation.py; in per-dataset mode category_labels are constant and return NaN. Needs fixing before semantic claims.)*

**Learned erasure** (Pythia-70M training dynamics):
| Layer | step3000 | step93000 | step143000 |
| --- | --- | --- | --- |
| L0 | 0.820 | 0.853 | 0.865 |
| L2 | 0.850 | 0.893 | 0.876 |
| L5 | 0.830 | 0.695 | **0.566** |

Final-layer AUROC degrades from 0.83 to 0.57 over training — the model **learns to remove** frequency from the residual stream. Consistent with saddle-to-saddle simplicity-bias dynamics (Zhang et al., 2512.20607): low-rank features are learned first then selectively strip when no longer needed.

**Rank-1 subspace ablation** (single-layer, probe direction, Pythia-6.9B ablated at L15):
| Layer | Clean AUROC | Ablated AUROC | Drop |
| --- | --- | --- | --- |
| L15 (ablated) | 0.970 | 0.970 | 0* |
| **L16** | 0.970 | **0.463** | **-0.507 (below chance)** |
| L17 | 0.968 | 0.546 | -0.422 |
| L20 | 0.961 | 0.706 | -0.255 |
| L31 | 0.919 | 0.750 | -0.169 |

*Zero drop at L15 is a known hook-ordering artifact — the residual capture runs before the ablation hook at the ablated layer. Fix pending.*

**Cross-model universality of the immediate-downstream drop**:
| Model | Ablate at | Next layer drop | Random baseline drop |
| --- | --- | --- | --- |
| Pythia-70M | L2 → L3 | 0.922 → 0.632 (-0.29) | 0.923 (null) |
| OLMo-1B | L7 → L8 | 0.974 → 0.604 (-0.37) | 0.974 (null) |
| Pythia-6.9B | L15 → L16 | 0.970 → 0.463 (-0.51) | 0.970 (null) |
| OLMo-7B | L15 → L16 | 0.975 → 0.578 (-0.40) | 0.975 (null) |
| Llama-3.1-8B | L15 → L16 | 0.963 → 0.679 (-0.28) | 0.963 (null) |

The effect grows with model size (Pythia family). Random-direction ablation is perfectly null across all models.

**Cumulative ablation** (ablate at every layer from start-L onward, measure AUROC at final layer):
| Model | Final clean | Cumulative ablated (worst starting point) | Output KL |
| --- | --- | --- | --- |
| Pythia-6.9B | 0.919 | **0.294** (from L30) | 0.012 |
| OLMo-1B | 0.949 | **0.366** (from L0) | 0.025 |
| OLMo-7B | 0.916 | **0.399** (from L30) | 0.003 |
| Llama-3.1-8B | 0.926 | **0.429** (from L30) | 0.001 |
| Pythia-70M | 0.830 | **0.479** (from L1) | 0.025 |

Cumulative ablation crashes final-layer probe AUROC to chance or below, regardless of where the ablation starts. Output KL stays small (0.001-0.025) — the residual-stream frequency pathway is largely **decoupled from the model's output behavior**.

### 2.3 Honest limitations of this finding

- The probe→direction→project methodology is essentially INLP (Ravfogel 2020) / amnesic probing (Elazar 2021) / LEACE (Belrose 2023). LEACE proves rank-1 sufficiency is **mathematically guaranteed** for binary/scalar concepts. The AUROC crash at the ablated layer is a tautology of linear algebra, not an empirical discovery.
- The **downstream non-recovery** (L16-L31 partial recovery to 0.75) IS empirically informative — it is not guaranteed by LEACE. Same for low output KL.
- "Features as directions" is established (Elhage 2022, Bricken 2023); our work is an instantiation for a specific feature.

### 2.4 What's genuinely novel (and limited)

Based on literature review (60+ papers across 6 areas):

1. **Neuron-vs-direction contrast**: First paper to run the clean head-to-head — "5-50% ablation of neurons selected by any criterion does nothing; rank-1 direction ablation is devastating." Every direction-intervention paper (Arditi, Marks, Engels) implicitly knows this but none publishes the comparison. (Most publishable element.)
2. **Downstream recovery dynamics**: Single-layer ablation followed by partial recovery over subsequent layers reveals active frequency-maintenance circuitry. Not guaranteed by LEACE.
3. **Decoupling**: Cumulative ablation kills frequency decodability while barely changing model output. Evidence for independent head-pathway (Kobayashi's b_LN) vs residual-stream pathway.
4. **Cross-architecture empirical grounding**: 5 models, 3 architectures, effect grows with scale.

None of these are main-track NeurIPS material on their own. Positioned for MI workshop, COLM, or ACL Findings.

---

## 3. Experimental Record

### 3.1 Completed experiments with in-repo artifacts

| Experiment | Script | Models completed | Key result | Artifacts |
| --- | --- | --- | --- | --- |
| Linear probes (per-layer freq + semantic) | (probes live in linear_probes repo) | Pythia-70M (15 ckpts), Pythia-6.9B (3 ckpts), OLMo-1B, OLMo-7B | Freq peaks mid-network, decays late; learned erasure | `results/linear_probes/` |
| Coverage / affinity / mass metrics | `coverage_affinity_experiment.py` | Pythia-70M, Pythia-6.9B, OLMo-1B, OLMo-7B, Llama-3.1-8B | Per-neuron statistics used to build (now abandoned) 3-axis framework | `results/coverage_affinity/` |
| n_clusters polysemanticity | `neuron_polysemanticity.py` | Pythia-70M, Pythia-6.9B, OLMo-1B, OLMo-7B | Polysemanticity decreases through layers but neurons stay polysemantic in 1B+ models | `results/n_clusters/` |
| JSD (legacy) | `coverage_affinity_experiment.py` | All | Group JSD decreases through layers (per workshop paper) | `results/coverage_affinity/*/group_jsd.csv` |
| Polytope density | `polytope/` | Pythia (70M–6.9B), OLMo (1B, 7B) | Polytope density decreases through layers | (per workshop paper) |
| Targeted ablation (JSD-based) | `targeted_ablation.py` | 5 models × 5 datasets × 8 strategies | Sign-consistent effect but modest in smaller models (70M and 6.9B ~0.4% at 5%); bigger in OLMo/Llama. L1 normalization confound | `results/ablation/` |
| **Probe-under-ablation** | `probe_under_ablation.py` | Pythia-70M, Pythia-6.9B, OLMo-1B (OLMo-7B and Llama runs cancelled due to HF token / download issues) | Neuron ablation barely moves probe AUROC — confirms neurons are not the right level of analysis | `results/probe_under_ablation/` |
| **Frequency circuit tracing v2** (per-layer direction from probe) | `frequency_circuit_tracing.py` | All 5 models × 5 datasets | MLP accounts for 80-99% of absolute projection along freq direction at each layer | `results/circuit_v2/` (per-neuron projections for Pythia-70M only; larger models kept on cluster) |
| **Concentration analysis** | `analyze_circuit_concentration.py` | Pythia-70M, OLMo-1B, Pythia-6.9B | Top 5% of neurons = 22-25% of |diff_proj|; cross-dataset Jaccard ~0.06. Frequency writing is distributed and domain-specific | `results/circuit_ablation/concentration/` |
| **Circuit-based ablation** (by |diff_proj|) | `circuit_ablation.py` | Pythia-70M, OLMo-1B, Pythia-6.9B | Even 50% ablation of top-|diff_proj| neurons fails to crash probe AUROC | `results/circuit_ablation/` |
| **Subspace ablation** (rank-1 probe direction + SVD rank-k + random baseline) | `subspace_ablation.py` | All 5 models | **THE core positive result.** Rank-1 probe ablation crashes immediate-downstream probe AUROC; cumulative ablation crashes final-layer AUROC to near chance; random is null. | `results/subspace_ablation/` |

### 3.2 Known bugs / caveats in current artifacts

1. **Hook ordering in subspace ablation** (`subspace_ablation.py`): `ResidualCapture` is registered before `SubspaceAblationHook`, so at the ablated layer itself the capture stores the pre-ablation residual. Fix: swap order so ablator runs first. All downstream-layer results are unaffected and correct.

2. **Semantic probe in `probe_under_ablation.py`**: In per-dataset mode, `category_labels` are constant (single value) so `train_semantic_probe` returns NaN. Any semantic-preservation claims from this script are invalid until datasets are merged before probing. (Subspace ablation script already fixes this — it merges datasets before computing freq directions.)

3. **Circuit_ablation averaging problem**: `load_circuit_neuron_projections` averages |diff_proj| across the 5 datasets before selecting neurons to ablate. Given the ~0.06 Jaccard finding, the averaged "top writers" are close to randomly selected. The null result this produced is thus not informative about whether per-dataset top writers would behave differently.

4. **MLP-vs-attention ratio uses absolute magnitudes**: `compute_component_freq_flow` reports `mlp_freq_fraction = |mlp_proj_diff| / (|mlp_proj_diff| + |attn_proj_diff|)`. This correctly supports "MLP contributes more absolute projection" but does not distinguish writers (positive projection differential) from erasers (negative). A signed decomposition is needed to claim "MLP writes, attention does X".

### 3.3 Key negative results (worth publishing as such)

- **Neuron-level selection doesn't localize frequency**: 10-50% ablation by any criterion (affinity, coverage, mass, |diff_proj| averaged, random) barely affects probe AUROC (0.000-0.007 delta averaged over ablated layers and datasets).
- **Averaged "top frequency writers" are not a consistent circuit**: Jaccard ~0.06 across 5 domains.
- **JSD dissociation is largely an L1 normalization artifact**: The workshop paper's "coverage drives JSD contribution, affinity has independent signal" story is sign-consistent with the new (weaker) probe-based measurement, but the magnitudes drop substantially once we remove normalization.

---

## 4. The Open Question: Mechanism

We have characterized the **representation** (1D direction, distributed writers, decoupled from output). We have not traced the **mechanism**. Open questions:

### 4.1 Who writes?
- Which specific attention heads at which layers project into the frequency direction? (Analogous to "name mover heads" in IOI.)
- Are there early-layer heads that attend to the target token position and move token-identity (which carries frequency via the embedding) into the residual stream for downstream MLPs to amplify?
- Which MLP layers are the signed "frequency writers" (positive differential projection for high-freq vs low-freq) vs "erasers" (negative)? Our current circuit tracing uses absolute values.

### 4.2 Who maintains?
- After single-layer ablation at L15 of Pythia-6.9B, probe AUROC recovers from 0.46 → 0.75 over layers 16-31. Which layers contribute most to this recovery? (Ablation at each single intermediate layer would localize the "maintenance writers".)
- Is recovery driven by MLP alone, attention alone, or both? Does it depend on scale (70M shows no recovery; 6.9B recovers most)?

### 4.3 Who erases?
- Late-layer probe AUROC declines even without intervention. Is this active erasure (specific components writing *against* the frequency direction) or passive dilution (other features growing, pushing the frequency-direction projection down relatively)?

### 4.4 What's the relationship with semantics?
- Are the frequency direction and the semantic (domain) directions orthogonal? If so, what allows them to be independent in the residual stream geometry?
- Does ablating the frequency direction affect semantic probe AUROC? If yes, frequency and semantics are entangled.
- Do any MLP neurons write to BOTH frequency AND semantic directions, or are the writers disjoint?
- Is the "frequency to semantics transition" actually "growth of semantic subspaces while frequency subspace shrinks" — i.e., a relative-norm phenomenon?

### 4.5 Does the residual-stream frequency direction align with the prediction-head b_LN direction?
- Kobayashi showed b_LN in the output embedding space correlates with frequency (cos ≈ 0.78 with log-freq on GPT-2).
- Our residual direction is in d_model space before unembedding.
- If we un-embed our direction (W_U @ v_L), does it align with b_LN?
- This would tell us whether the residual pathway and the head-bias pathway converge on the same output-space frequency axis or are genuinely independent pathways (the "decoupling" implied by low output KL).

---

## 5. Literature Synthesis

Synthesis of four parallel alphaxiv literature reviews (60+ papers across four areas, completed 2026-04-15). The detailed reviews are in `/tmp/agent_outputs/*.md` (not committed — reproducible via the review prompts in git history).

### 5.1 What's established (and what's NOT novel in our work)

**Methodology**: The pipeline of train-a-linear-probe → use-weight-vector-as-direction → project-out-from-residuals is known as **INLP** (Ravfogel 2020, arXiv:2004.07667), refined by **RLACE** (Ravfogel 2022, arXiv:2201.12091) and **LEACE** (Belrose 2023, arXiv:2306.03819). Amnesic probing (Elazar 2021) applies this per-layer. LEACE proves rank-1 sufficiency is mathematically guaranteed for any scalar/binary concept with different class-conditional means. Our probe-direction ablation at the ablated layer is therefore a mathematical tautology, not an empirical discovery.

**Features-as-directions**: Established by Elhage et al. 2022 "Toy Models of Superposition" (arXiv:2209.10652) and confirmed empirically by Arditi et al. 2024 "Refusal direction" (arXiv:2406.11717), Marks & Tegmark 2023 "Truth direction" (arXiv:2310.06824), Engels et al. 2024 "Not all features are linear" (arXiv:2405.14860), Park et al. 2023 "Linear Representation Hypothesis" (arXiv:2311.03658). Our finding that frequency = 1D direction is an instantiation of a well-established principle.

**Frequency in LMs (partial prior work, five separate findings)**:
- Kobayashi et al. 2023 (arXiv:2305.18294) — b_LN in prediction head encodes frequency direction in output space; Spearman ρ ≈ 0.78 with log-freq; hidden states orthogonal to b_LN (cos ≈ 0.08).
- Puccetti et al. 2022 (arXiv:2205.11380) — LayerNorm outlier dimensions correlate with token frequency; removing harms rare-token prediction.
- Macocco et al. 2025 (arXiv:2503.21718) — Last-layer outlier dimensions implement a "frequent-word heuristic"; counterbalanced when context demands rare tokens.
- Stolfo/Wu et al. 2024 (arXiv:2406.16254) — MLP "token frequency neurons" with output weights that push logits proportional to log-frequency; also "entropy neurons" that use the LayerNorm-mediated null space.
- Mu & Viswanath 2018 — Top PCs of static embeddings encode frequency; removing them improves isotropy.

**Training dynamics**: Chang & Bergen 2021 (arXiv:2110.02406) — models learn unigram distribution first, then bigrams, then context. Zhang, Saxe & Latham 2025 (arXiv:2512.20607) — saddle-to-saddle dynamics give a theoretical grounding: simplicity bias learns frequency-like features first.

**Entanglement and causal orthogonality**: Park et al. 2024 "Geometry of Categorical Concepts" (arXiv:2406.01506) proves causally separable concepts are orthogonal under the causal inner product. Wollschlager et al. 2025 "Geometry of Refusal" (arXiv:2502.17420) — refusal is mediated by multi-dimensional cones, not single direction. Introduces "representational independence" criterion, stricter than orthogonality.

**Circuit tracing methodology**: Wang et al. 2023 IOI (arXiv:2211.00593) is the gold standard — 26-head circuit with 7 functional head classes. Conmy et al. 2023 ACDC (arXiv:2304.14997) automates this. Zhang & Nanda 2023 (arXiv:2309.16042) provides best practices. Dunefsky et al. 2024 transcoders (arXiv:2406.11944) enable input-invariant MLP decomposition. Syed et al. 2023 attribution patching scales to large models.

### 5.2 What IS genuinely novel in our work (the defensible claims)

Based on cross-referencing 60+ papers, four aspects of our work do not appear in prior literature:

**1. Direct neuron-vs-direction contrast for the same concept.** Every direction-intervention paper (Arditi, Marks, Engels) implicitly assumes neuron ablation would fail and jumps straight to direction ablation. No paper has published the clean head-to-head: "we ablated 5-50% of neurons selected by every reasonable criterion (coverage, mass, affinity, |diff_proj|, random) and got 0.000-0.007 AUROC change; we ablated 1/d_model dimensions via rank-1 projection and got a 50-point crash." This is empirically novel even though theoretically expected.

**2. Downstream non-recovery dynamics.** LEACE proves the ablated layer itself must lose linear decodability; it says nothing about downstream layers. Our observation that L16 crashes to below-chance AUROC and then partially recovers to 0.75 by L31 after a single-layer ablation at L15 is NOT guaranteed by any theory. The recovery curve encodes information about frequency re-writers. Arditi applies ablation at all layers simultaneously and cannot observe recovery; LEACE concept scrubbing does the same. Our single-layer + downstream-probe protocol is a genuinely new observation.

**3. Decoupling between residual-stream frequency and output behavior.** Cumulative ablation across all late layers crashes probe AUROC (0.92 → 0.30) while output KL stays at 0.001-0.025. This is evidence for **two independent frequency pathways**: the residual-stream direction (which we ablate) and the prediction-head b_LN (which we don't). Agent C (frequency literature) notes that no prior paper has connected b_LN, outlier dimensions, and residual-stream frequency as potentially distinct pathways.

**4. Potential to unify five independent frequency findings.** Agent C identified that prior literature has found at least five separate frequency encodings — Kobayashi's b_LN, Puccetti's outlier dimensions, Macocco's last-layer outliers, Stolfo's token-frequency neurons, and now our residual-stream direction — and **no one has checked whether they are the same direction**. If they align via a single measurement pipeline, we unify five findings into one. If they don't align, we reveal that frequency has multiple representational pathways.

### 5.3 The most important gap in existing work

Across all four agent reviews, the most striking and consistent gap is: **nobody has traced the per-component attribution of a probe-identified direction**. Todd et al. 2024 "Function Vectors" come closest — they trace function vectors to specific attention heads — but for frequency (or any static feature direction), no paper has decomposed which heads and MLPs contribute to the direction at each layer. This is exactly the "who writes what" question we need to answer.

The residual stream at layer L is an additive sum: `x_L = x_0 + Σ attn_outs + Σ mlp_outs`. Projecting onto the frequency direction is linear, so we can exactly attribute the projection: `proj(x_L, v_L) = proj(x_0, v_L) + Σ proj(attn_out_i, v_L) + Σ proj(mlp_out_j, v_L)`. Per-head via z_h @ W_O[L,h].

No existing paper does this decomposition for a probe-identified direction. Doing it for frequency across 5 models is a novel contribution.

---

## 6. Prioritized Research Directions

Based on the literature synthesis, five directions ranked by novelty × feasibility:

### Direction 1 — **Unify the five frequency findings** (HIGHEST NOVELTY, HIGHEST FEASIBILITY)

**Question**: Are Kobayashi's b_LN, Puccetti's outlier dimensions, Macocco's last-layer outliers, Stolfo's token-frequency-neuron output weights, and our residual-stream 1D direction all manifestations of the same underlying direction, or are they distinct pathways?

**Why novel**: Agent C confirms this has not been done. Five papers describe frequency from five angles; their relationship is completely uncharacterized.

**Method**:
1. For a shared model (Pythia-6.9B or GPT-2), extract each of the five known frequency-related vectors:
   - b_LN bias parameter (Kobayashi)
   - Top outlier dimensions by hidden-state magnitude (Puccetti method)
   - Last-layer outlier dimensions (Macocco method)
   - Stolfo's v_freq vector derived from their token-frequency neurons (they publish the identification procedure)
   - Our probe weight vector v_L at each layer
2. Compute the full 5×5 cosine similarity matrix (or its L+4 variant including all layer-wise probe vectors)
3. Check: (a) Are Kobayashi/Puccetti/Macocco/Stolfo all ~colinear? (b) Does our residual direction align with them, and at which layer? (c) Do they rotate through training (Pythia checkpoints)?

**Predicted outcomes and interpretations**:
- If all align → we have unified the frequency literature; the paper contribution is now "there is ONE frequency direction in transformers that manifests five different ways depending on where you look."
- If our residual direction is orthogonal to b_LN → we have confirmed TWO pathways (residual stream and head bias) that independently encode frequency. This also explains the decoupling (low output KL under residual ablation).
- If they mostly align but one is off → the outlier is the interesting finding.

**Feasibility**: Extremely high. This is cosine similarities between known vectors. Could be done in 1-2 days of compute + analysis.

**Paper value**: Very high. Turns a potentially-tautological finding into a unifying story across literature.

### Direction 2 — **Per-component circuit decomposition + recovery attribution** (HIGH NOVELTY, HIGH FEASIBILITY)

**Question**: Which attention heads and MLP sublayers write the frequency direction, and which components drive the post-ablation recovery?

**Why novel**: Agent A and Agent B both flag this as the clearest literature gap. No paper has decomposed a probe-identified residual-stream direction into per-head + per-MLP contributions with signed differentials.

**Method**:
1. **Direct Logit Attribution (DLA) for the frequency direction** (Elhage's mathematical framework applied to a non-logit target):
   - For each (model, layer, component), compute the signed projection of that component's output onto the frequency direction at layer L.
   - Separate per-head and per-MLP decomposition; separate high-freq vs low-freq inputs.
   - Signed differential (high-low) × cosine with the direction = "frequency DLA score".
2. **Identify head types** analogously to IOI:
   - Early-layer heads that attend to the target token position = "frequency detector heads"
   - Mid-layer MLPs that broadly add = "frequency accumulator MLPs"
   - Late-layer MLPs that subtract = "frequency eraser MLPs" (predicted to exist based on the suppression we see)
   - Any late-layer heads that attend back to the accumulated signal = "frequency mover heads"
3. **Recovery attribution**: Ablate at L15 (single-layer, probe direction). Measure the per-layer contribution to recovery: for each downstream layer L', compute (projected_onto_freq(component_output) under ablation) - (same under clean). Layers where this delta is positive are "recovery writers".
4. **Hydra effect test**: Ablate the primary recovery writers identified in step 3. Do backup heads/MLPs activate? (McGrath et al. 2023.)

**Feasibility**: Medium. Needs TransformerLens + careful bookkeeping. Estimated 2-3 weeks on cluster + analysis. Pythia-6.9B is the primary target; extend to OLMo-1B and Llama-3.1-8B for cross-model universality.

**Paper value**: Very high. Turns the "representation" paper into a proper "circuit" paper, making it competitive with Wang et al. IOI in scope.

### Direction 3 — **Frequency-semantics geometric relationship** (MEDIUM-HIGH NOVELTY)

**Question**: Are the frequency direction and semantic (domain) directions orthogonal? Does ablating one affect the other's decodability?

**Why novel**: Agent D proposes this grounded in Park et al.'s causal orthogonality theorem. Park predicts causally separable concepts should be orthogonal under the causal inner product. Frequency and semantic domain ARE causally separable (each domain has its own Zipfian distribution), so the prediction is orthogonality — but only one prior paper (Marks & Tegmark on truth) has systematically tested such predictions for a specific concept.

**Method**:
1. Train one-vs-rest semantic probes per layer per domain (5 domains) on ScaleJSD: get 5 semantic direction vectors per layer.
2. Compute the Gram matrix (6 × 6 per layer, using freq direction + 5 semantic directions). Both under Euclidean inner product AND Park et al.'s causal inner product.
3. Layer-wise orthogonality trajectory: does cos(freq_dir, semantic_dir_i) decrease from early to late layers? (predicted by Voita 2019 surface-to-semantic progression)
4. Bidirectional ablation:
   - Rank-1 ablate freq direction → measure semantic probe AUROC → predict: unchanged if orthogonal
   - Rank-1 ablate each semantic direction → measure freq probe AUROC → same prediction
5. Per-neuron co-alignment: for each MLP neuron output weight w_out, compute cos(w_out, freq_dir) vs cos(w_out, max semantic_dir_i). Scatter plot separates disjoint writers from shared writers.

**Feasibility**: Medium. Probes need to be trained cleanly (fix the semantic probe bug in probe_under_ablation.py first). Estimated 1-2 weeks.

**Paper value**: Medium-high. If orthogonal → confirms Park et al. prediction for a new concept + shows natural disentanglement. If entangled → reveals model has a non-trivial causal model of language (freq and semantics ARE coupled for some domains).

### Direction 4 — **Residual-stream vs head-bias pathway test** (MEDIUM NOVELTY)

**Question**: Are the two frequency pathways (residual stream direction + b_LN) independent, or are they the same direction accessed at different points in computation?

**Why novel**: Directly tests our "decoupling" hypothesis from the cumulative-ablation-with-low-output-KL observation. No paper has connected residual-stream and head-bias frequency representations.

**Method**:
1. Un-embed the residual-stream frequency direction: `W_U @ v_L` for each layer L (result is in vocabulary space).
2. Compute cosine with b_LN direction (for models that have it) or with the unigram-log-probability direction (for all models).
3. Co-ablation test: perform rank-1 ablation of v_L AND zero out b_LN. Measure output KL and prediction behavior. If both pathways contribute to output frequency, co-ablation should produce much larger KL than either alone.
4. Check this for each layer L: does the un-embedded residual direction align with b_LN at any layer? At some layer boundary does it flip?

**Feasibility**: High. This is mostly linear algebra on existing matrices + a few forward passes with different ablations. Estimated <1 week.

**Paper value**: Medium. Supports the decoupling story. Mostly a confirmation/extension of existing findings.

### Direction 5 — **Training dynamics of the frequency direction** (MEDIUM NOVELTY, HIGH COMPUTE COST)

**Question**: When does the 1D frequency direction emerge during training? Does it consolidate from higher-dim to 1D over time? When does active erasure in late layers emerge?

**Why novel**: Zhang et al. 2025 predicts saddle-to-saddle dynamics should show frequency emerging first. Puccetti 2022 observed outlier dimensions emerge at ~80K training steps. But nobody has tracked the 1D residual-stream frequency direction across checkpoints.

**Method**:
1. For Pythia-70M (15 checkpoints), compute freq probe direction at each (checkpoint, layer).
2. Measure cos(direction at step t, direction at final step) to see stabilization.
3. Measure rank(cov of residuals that decode high-freq) — does it collapse from d_model to ~1 over training?
4. Measure when final-layer AUROC decay starts (active erasure emerges).

**Feasibility**: Medium. The compute is modest but the analysis is multi-dimensional. Estimated 2-3 weeks.

**Paper value**: Medium. More of a supporting experiment than a lead contribution. Useful as a section in a larger paper.

---

## 7. Revised Paper Strategy

### 7.1 Narrative arc (proposed)

The paper should be structured around a single unifying question: **"How does token frequency flow through a transformer language model?"**

**Act 1: The representation.** Token frequency is encoded in a 1D direction of the residual stream at every layer. Linear probes find it; rank-1 projection destroys it downstream; random projection is null. This is true across 5 models and 3 architectures. (Expected from LEACE theory but empirically grounded.)

**Act 2: The negative result.** The 1D direction is NOT carried by any identifiable set of specialist neurons. Distributed writing across many MLP neurons; no dataset-consistent writer population. This is the clean neuron-vs-direction contrast (novel).

**Act 3: The unification** (Direction 1 above). Show that our residual-stream 1D direction connects to (or doesn't connect to) Kobayashi's b_LN, Puccetti's outlier dimensions, Macocco's last-layer heuristic, and Stolfo's token-frequency neurons. Either outcome is publishable.

**Act 4: The circuit** (Direction 2 above). Per-head + per-MLP decomposition of the frequency direction. Identify the writers, the maintainers, and the erasers. Recovery dynamics after single-layer ablation reveal the maintenance circuit.

**Act 5: The geometry** (Direction 3 above). Frequency vs semantic direction orthogonality. Test Park et al.'s causal orthogonality prediction for a new concept.

Acts 1-2 use our existing data. Acts 3-4-5 require new experiments.

### 7.2 Venue calibration

Post-literature-review, the best venue options:

- **NeurIPS 2026 main track**: Possible if we land Direction 1 (unification) + Direction 2 (per-component circuit) cleanly. Acts 1-4 constitute a full paper with clear novelty despite LEACE/INLP dependence.
- **NeurIPS 2026 MI Workshop**: Very likely; this is the natural venue for the workshop-paper lineage anyway.
- **ICLR 2027**: If we also land Direction 3 (entanglement) to round out the story.
- **COLM 2025/2026**: Strong fit given LM focus.
- **ACL Main or Findings**: If we frame around the linguistic implications (frequency-sensitive generation, rare-token handling).

### 7.3 Two-sentence pitch (v2)

> "Token frequency in transformer language models is encoded as a single 1D direction in the residual stream, written into additively by many MLP neurons with no specialists — a clean empirical case where the same concept is localizable at the direction level but distributed at the neuron level. We trace this frequency circuit end-to-end (embedding → per-head attention → MLP writers/erasers → prediction-head bias), unify five separately-documented frequency-encoding phenomena (Kobayashi's b_LN, Puccetti's outlier dimensions, Macocco's last-layer heuristic, Stolfo's token-frequency neurons, and our direction) into a single framework, and show how the model actively maintains this direction against single-layer erasure while remaining functionally decoupled from the head-bias pathway."

---

## 8. Critical Bugs to Fix Before Direction 2 Runs

Logged here so they don't get lost:

1. **Hook ordering in `subspace_ablation.py`** — ablator must run before capture at the ablated layer. Current code swaps this, so L15-itself AUROC shows 0.97 instead of 0.5 (tautologically). Fix: in `ablated_forward()`, register ablator hooks first, then capture hooks.

2. **Semantic probe breakage in `probe_under_ablation.py`** — in per-dataset mode, `category_labels` is constant → NaN. Fix: pool datasets before training semantic probes.

3. **`circuit_ablation.py` averages |diff_proj| across datasets before selection** — given Jaccard ~0.06 cross-dataset overlap, this averages to nonsense. Fix: per-dataset selection, then union.

4. **`compute_component_freq_flow` uses absolute values for MLP/attn fraction** — claims should be "MLP dominates the ABSOLUTE projection onto the frequency direction", not "MLP writes more than attention". Fix: report signed and abs versions separately.

5. **Per-neuron freq_projections for 1B+ models are only on cluster** — ~1.4GB, not in repo. Need to either commit a sampled/sparsified version or an aggregate-by-layer version for reproducibility.

These should be fixed before the Direction 2 experiments so the data we collect is clean.

---

## 7. Paper Positioning (Current Understanding)

### 7.1 Likely venue
Workshop / Findings-level paper. Specifically:
- NeurIPS 2026 MI Workshop (natural successor to the NeurIPS 2025 workshop paper)
- ICML 2026 MI workshop
- COLM 2025 (LM-focused)
- ACL Findings (if framed linguistically)

### 7.2 Narrative spine
"Token frequency provides a clean test case for studying how features are routed through transformer residual streams. We show: (1) frequency is encoded as a 1D direction; (2) neuron-level interventions completely fail to localize this signal while rank-1 direction interventions are devastating — the first clean demonstration of the theoretically-predicted neuron-vs-direction dissociation for a specific feature; (3) after single-layer ablation, downstream layers partially reconstruct the frequency direction, revealing active maintenance circuitry; (4) cumulative ablation decouples residual-stream frequency from model output behavior, consistent with a separate head-bias pathway (Kobayashi et al. 2023). We then perform [to be filled in by experiments §6.1–6.5]."

### 7.3 Honest boundary: what this paper is NOT
- NOT claiming 1D subspace discovery is novel (LEACE proves this for scalar features).
- NOT claiming the probe→project methodology is new (INLP / amnesic probing).
- NOT claiming unique insight beyond what's possible for a single workshop-quality paper.

The contribution is empirical clarity on a specific, clean test case, with a crisp neuron-vs-direction comparison that should be generally instructive for the field.

---

## 8. Repo / Artifacts Map

Key scripts:
- `scripts/metrics/coverage_affinity_experiment.py` — per-neuron coverage/mass/affinity/JSD (legacy, still used for its stats)
- `scripts/metrics/neuron_polysemanticity.py` — HAC clustering on neuron embeddings
- `scripts/interventions/targeted_ablation.py` — JSD-based ablation (legacy)
- `scripts/interventions/probe_under_ablation.py` — neuron ablation + probe re-evaluation
- `scripts/interventions/frequency_circuit_tracing.py` — per-layer attn/MLP projection onto freq direction
- `scripts/interventions/circuit_ablation.py` — ablation by |diff_proj| selection
- `scripts/interventions/subspace_ablation.py` — **the core positive result**
- `scripts/plots/analyze_circuit_concentration.py` — concentration + Jaccard analysis

Key result directories (in-repo):
- `results/subspace_ablation/` — all 5 models, core positive result
- `results/circuit_v2/` — per-layer component flows, freq directions, summaries (per-neuron projections only for Pythia-70M)
- `results/circuit_ablation/` — 3 models + concentration analysis
- `results/probe_under_ablation/` — 3 models
- `results/linear_probes/` — probe trajectories and training dynamics
- `results/coverage_affinity/` — per-neuron stats
- `results/n_clusters/` — polysemanticity
- `results/ablation/` — JSD-based ablation (legacy)

Manuscript materials:
- `neurips-2026/notes/research-summary.md` — this document
- `neurips-2026/notes/research-summary-v1-archived.md` — previous version with 3-axes framework (for reference only, do not modify)
- `neurips-2026/notes/probe-under-ablation-design.md` — design doc (superseded by subspace_ablation)
- `neurips-2026/notes/coverage-principle-experiment-design.md` — design doc for abandoned framework
- `neurips-2026/notes/workshop-review.md` — reviewer feedback on the workshop paper
- `neurips-2026/manuscript/rough-draft.md` — **STALE** — tells the old "specialist neurons / three-axes" story. Needs rewrite once §4/§6 experiments complete.
