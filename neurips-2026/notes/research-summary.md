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

## 5. Literature Findings (in progress)

*Four subagents currently searching alphaxiv for:*
- *A. Circuit tracing methodology — IOI, docstring, path patching, SAE-based circuits*
- *B. Feature-as-direction literature — INLP, LEACE, amnesic probing, refusal/truth/day-month*
- *C. Frequency-in-LLMs prior work — Kobayashi, Puccetti, Macocco, Brinkman, training dynamics*
- *D. Feature entanglement / orthogonality — superposition, feature interactions, hierarchical reps*

*Results will be synthesized into §6 once agents return.*

---

## 6. Proposed Next Experiments (to be firmed up from literature findings)

Working candidates, ordered by current confidence:

### 6.1 Per-head signed frequency decomposition (highest priority)
Using TransformerLens or raw hooks on Pythia-6.9B:
- For each attention head h at layer L, compute its contribution to the residual stream via z_h @ W_O[L,h].
- Project each head's contribution onto the frequency direction v_L (trained probe weights at L).
- Separately for high-freq vs low-freq inputs.
- Signed differential per head = identifies "frequency writer heads" (positive differential) and "frequency eraser heads" (negative).
- Attention pattern analysis: do the frequency heads attend to the target token position?

This is the IOI-style decomposition applied to frequency. Expected outcome: a small number of early-layer heads that read token identity and route it into the frequency direction.

### 6.2 Signed MLP writer/eraser trajectory
Fix `compute_component_freq_flow` to report signed `mlp_proj_diff` not absolute. Re-run on all 5 models. Identify which layers are writers (positive differential) vs erasers (negative differential). Tentatively predict: early-mid layers are writers, late layers are erasers.

### 6.3 Frequency-semantics geometric relationship
- Train one-vs-rest semantic probes per layer per domain (5 domains). Get 5 semantic directions per layer.
- Compute cosine similarity matrix: freq direction vs each semantic direction, and semantic directions vs each other.
- Hypothesis: freq direction is near-orthogonal to semantic directions (otherwise synonyms with different freq couldn't be semantically consolidated).
- Test: does subspace ablation of freq direction affect semantic probe AUROC? Predict no — if orthogonal, ablation shouldn't interfere.

### 6.4 Writer localization via per-layer single-ablation
Extend subspace ablation: ablate at every single layer L individually (not just the few we tested), measure final-layer probe AUROC for each. Layers whose individual ablation causes the biggest final-layer drop are critical "maintenance writers". Generates a full "writer importance" profile.

### 6.5 Residual-direction vs b_LN alignment
For each model where we have b_LN extraction:
- Compute un-embedded residual freq direction: W_U^T @ v_L (for each layer L).
- Compute cosine with b_LN direction (for models that have it) or with unigram-log-probability direction (for all models).
- Shows whether the two "frequency pathways" (residual stream direction vs prediction-head bias) encode the same or different output-space directions.

### 6.6 Training-dynamics of the direction
For Pythia-70M checkpoints:
- Compute freq direction at each checkpoint × each layer.
- Measure cos between checkpoint-t direction and checkpoint-T direction (does the direction stabilize early?).
- Measure when the "active erasure" in late layers emerges during training.

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
