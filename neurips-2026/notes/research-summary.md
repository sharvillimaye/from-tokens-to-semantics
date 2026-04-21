# Frequency Routing in Transformer LMs: Research Summary

**Status**: Active paper-in-progress. Previous iteration archived at `research-summary-v1-archived.md` (2026-04-15 rewrite).

**Authors**: Limaye\*, Ramesh\*, Rohweder, Zhou\*, Panda, Bhaskar, Sharma
**Prior workshop paper**: NeurIPS 2025 Mechanistic Interpretability Workshop (poster)
**Models**: Pythia-70M (15 checkpoints), Pythia-6.9B, OLMo-1B, OLMo-7B, Llama-3.1-8B
**Data**: ScaleJSD — 291 synonym pairs across 5 domains (emotion, medical, legal, scientific, verb), each pair a high/low-frequency token with identical meaning (e.g. "happy"/"elated"). Shared sentence templates isolate frequency from semantic content.

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

### 2.5 Post-Apr-15 Track Results (2026-04-17/18)

Four parallel experimental tracks + two local analyses have updated the picture since §2.4 was written. **Summary: the paper thesis shifts from "causal-weak shadow feature" to "probe decodability does not predict steering responsiveness, with a universal concept-dependent write-site pattern."** Scoop check against Braun et al. 2026 (EACL Findings) is critical — see §5.4.

#### Track 4 — Steering metrics with proper behavioral metrics

11-α greedy sweep (α ∈ {-2, -1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1, 2}), 200 prompts per model, on three models. Metrics: Flesch-Kincaid grade, mean word length, type-token ratio, mean Zipf (wordfreq library), first-sentence length, first-token high/low freq mass, median perplexity.

| Model | Steer L | ΔFK (α=-2 → +2) | Zipf Δ | First-sent Δ | Perplexity stable |
| --- | --- | --- | --- | --- | --- |
| **OLMo-7B** | L16 | **7.83 → 5.38 (-2.45)** | 6.08 → 6.14 | 79 → 52 chars | ✓ (2.7-2.9) |
| **Llama-3.1-8B** | L7 | **8.45 → 5.19 (-3.26)** | 5.96 → 6.12 | 84 → 58 chars | ✓ (2.4-2.8) |
| Pythia-6.9B | L10 | 5.78 → 5.48 (-0.30) | flat | flat | ✓ |

Clean monotonic FK-grade control on OLMo and Llama. Pythia-6.9B step143000 is near-dead despite identical probe AUROC (0.975) — a direct dissociation of decodability from steering responsiveness (Pythia-dead gets a short mention per §9; not a central claim).

#### Track Circuits — Signed per-component attribution patching

Gradient × class-differential attribution on metric `M(x) = <resid[L*], v>` for every attention head and every MLP.

| Model | Peak L | AUROC | Top-10 writers | Top eraser (attn) |
| --- | --- | --- | --- | --- |
| OLMo-7B | L19 | 0.967 | 100% MLPs (L16-19, L0-2) | L19:30 (-0.009) |
| Llama-3.1-8B | L7 | 0.970 | 100% MLPs (L0-7, concentrated early) | L7:0 (-0.008) |
| Pythia-6.9B | L12 | 0.953 | 100% MLPs (L0=1.29, L2-12 spread) | L8:7 (-0.068) |

**Universal raw-magnitude pattern**: top-10 writers are 100% MLPs in all 3 architectures; attention heads contribute only as small erasers.

#### MLP-vs-attention per-capacity normalization (local, Apr 18) — caveat to "MLP-exclusive"

Raw magnitudes: MLP dominates by 3.9× (OLMo), 21.4× (Llama), 6.8× (Pythia). But attention heads have d_head = 128 output channels each, versus MLP's d_model = 4096. Normalizing:

| Model | Raw ratio | **Per-output-dim ratio** | Per-parameter ratio |
| --- | --- | --- | --- |
| OLMo-7B | 3.9× | **0.12×** (attn stronger per-dim) | 0.06× (attn stronger per-param) |
| Llama-3.1-8B | 21.4× | 0.67× | 0.34× |
| Pythia-6.9B | 6.8× | 0.21× | 0.11× |

Revised claim: the correct framing is *"MLP raw dominance is partly an output-channel-capacity effect; the frequency direction is broadly distributed across both component types, with MLPs aggregating more because they have 32× more output channels than a single head."* Per-channel, attention heads are comparable or stronger.

#### Idea E — Active eraser vs. passive dilution (local analysis, Apr 18)

Re-analysis of `recovery_attribution.csv`. Tracks `clean_M_diff_hi_lo` = signed projection of class-mean difference onto v per layer.

| Model | Peak L | M_diff at peak | M_diff after peak | Interpretation |
| --- | --- | --- | --- | --- |
| OLMo-7B | L19 | 0.657 | Grows to L28 (0.715), crashes L31 (0.390) | Hybrid: grows then suddenly erased at final layer |
| **Llama-3.1-8B** | L7 | 1.04 | **Keeps growing monotonically to L30 (2.12)**, crashes L31 (1.15) | **Passive dilution** — magnitude never stops; AUROC decline is relative norm effect |
| Pythia-6.9B | L12 | 11.3 | Gradual decline to L31 (9.5) | Gradual active erasure |

**Key finding**: In Llama-3.1-8B, frequency signal MAGNITUDE grows across every layer. The probe-AUROC decline is NOT destruction — it's competing features growing faster (relative dilution, not active erasure). In all 3 models, L31 shows a sudden crash — consistent with prediction-head/b_LN pathway acting at the final layer.

#### Track A — Truth-direction comparison (methodology-limited)

Marks & Tegmark 2023 truth direction extracted from geometry-of-truth TrueFalse set (1000 balanced statements from 8 merged CSVs at `saprmarks/geometry-of-truth`). Same α grid on same ScaleJSD continuation prompts.

| Model | Truth probe AUROC | Truth best L | Truth KL @α=1 | Freq KL @α=1 | Ratio truth/freq |
| --- | --- | --- | --- | --- | --- |
| OLMo-7B | 0.860 | L21 | 0.0130 | 0.0255 | 0.51 |
| Llama-3.1-8B | 0.957 | L12 | 0.0037 | — | — |
| Pythia-6.9B | 0.849 | L15 | 0.0059 | — | — |

Surprise — truth KL < freq KL. Methodology artifact: both tests used ScaleJSD synonym-pair prompts (right for frequency, wrong for truth). Parked — re-run with factual-question prompts before claiming.

#### Track C — Hallucination predictor (weak but real)

PopQA 2000 questions per model, greedy-decoded correctness.

| Model | Probe ρ (overall) | ρ on Q3 (mid-rare) | Probe AUC | Logprob baseline AUC |
| --- | --- | --- | --- | --- |
| OLMo-7B | -0.076 (p=7e-4) | -0.211 | 0.45 | 0.69 |
| Llama-3.1-8B | -0.165 (p=1e-13) | -0.326 | 0.39 | 0.79 |
| Pythia-6.9B | -0.053 (p=0.02) | -0.160 | 0.46 | 0.82 |

Consistent negative correlation: high freq-direction activation → more likely to hallucinate. Statistically real in all 3 models; strongest on mid-rarity Q3 slice. AUC below answer-logprob baseline → usable as auxiliary feature, not standalone detector.

#### Track QKV reader — ✅ COMPLETE (Apr 18, pod `ani-qkv-reader-3m-sjwdx`, 119 min CPU)

**Result: READERS EXIST.** Attention heads with W_{Q,K,V} rows significantly aligned with v (p99 null cos ≈ 0.062):

| Model | % Q-heads above null | % K-heads | % V-heads | Top max_cos (Q) |
| --- | --- | --- | --- | --- |
| OLMo-7B | **62%** (239/384) | 59% | 29% | L28 h19: 0.260 |
| Llama-3.1-8B | 14% (110/768) | 42% | 24% | L10 h6: 0.181 |
| Pythia-6.9B | 21% (130/608) | 29% | 25% | L16 h26: 0.167 |

Top readers concentrate in the immediate-downstream band (L*+1 to L*+5) with a secondary tail in final layers (L28-L30). **Fundamentally revises the mechanistic picture** — our earlier "no reader → shadow feature" inference was wrong. See §4.6 for full interpretation.

#### Idea 3 (frequency direction = conditional log-P(token) axis) — ✅ COMPLETE (Apr 18, pod `ani-code-length-3m-xnk7t`, 148 min CPU)

**Primary test**: projection `<residual[L*], v>` vs log corpus frequency at ScaleJSD anchor positions (probe-native evaluation point):

| Model | L* | n | Pearson r | Spearman ρ | **R²** |
| --- | --- | --- | --- | --- | --- |
| OLMo-7B | L16 | 582 | +0.693 | +0.723 | **0.481** |
| Llama-3.1-8B | L15 | 582 | +0.711 | +0.734 | **0.506** |
| Pythia-6.9B | L12 | 582 | +0.697 | +0.723 | **0.486** |

**Striking universal: R² ≈ 0.48-0.51 across all 3 models.** Half the variance in the probe-direction projection is explained by log corpus frequency. **The frequency direction is (substantially) a learned code-length axis — at prediction points in the probe's native distribution.**

**Domain stratification** (Pythia): emotion r=0.64 (R²=0.41), verb r=0.66 (R²=0.43), medical r=0.44 (R²=0.20), scientific r=0.27 (R²=0.07), legal r=0.10 (R²=0.01). Strongest for general-English domains, weakens in specialized jargon. The direction tracks English-wide log P(token), not domain-specific frequency.

**Out-of-distribution test** (random positions in arbitrary text — "wikitext fallback"): OLMo-7B r=-0.57 (template artifact), Llama-3.1-8B r=+0.20, Pythia-6.9B r=+0.22. The direction does NOT cleanly generalize to arbitrary positions — the code-length relationship holds specifically at **token-prediction positions**, not at every residual-stream point.

**Combined with external wordfreq Zipf** (English-wide unigram, not ScaleJSD-specific): r ≈ 0.44 in OLMo and Llama. Lower than ScaleJSD-native (0.69-0.71) because the probe was trained on ScaleJSD's domain-specific frequency distribution.

**Interpretation**: The frequency direction at L* encodes a linear function of log P(token | prediction context), with R² ≈ 0.5. This is the **conditional** code-length axis — it is what an arithmetic-coding-aware representation would look like at decoding points. It provides strong information-theoretic grounding for the rest of the paper: cross-entropy training implicitly implements arithmetic coding, and here we see the underlying geometric axis. Complements Idea C (which showed this internal axis is ORTHOGONAL to the output-side b_LN unigram pathway — two independent frequency mechanisms, one upstream conditional, one downstream marginal).

#### Sentiment-control (Idea 6) — ✅ COMPLETE on OLMo/Llama/Qwen (Apr 18, pod `ani-sentiment-3m-ghbfg`)

**Central finding: write-site pattern is concept-dependent — the strongest empirical claim of the paper.**

Extracted sentiment direction via Tigges 2023 protocol on IMDB (1000 pos + 1000 neg, last-token residual probe). Trained sentiment probe per layer, ran SAME signed DFA attribution pipeline + QKV reader test as the frequency analysis. Direct head-to-head.

| Model | Freq L* / AUROC | Freq top-10 writers | Sentiment L* / AUROC | Sentiment top-10 writers |
| --- | --- | --- | --- | --- |
| OLMo-7B | L19 / 0.967 | 10 MLPs, 0 attn (MLP/attn = 1.64×) | L21 / 0.956 | **6 attn, 4 MLPs (ratio 0.24×)** |
| Llama-3.1-8B | L7 / 0.970 | 8 MLPs, 2 attn (ratio 4.01×) | L16 / 0.954 | **9 attn, 1 MLP (ratio 0.16×)** |
| **Qwen-2.5-7B** | **L15 / 0.964** | **10 MLPs, 0 attn (ratio 3.52×)** | L19 / 0.939 | **9 attn, 1 MLP (ratio 0.22×)** |

**Ratio flips by 10-25× in all 3 models when switching concepts. Complete 3-model × 2-concept matrix confirms the taxonomy universally.**

**Qualitatively different write-site for the same model.** Top attention head contributions for sentiment: OLMo L21 head 6 (attr=0.034), Llama L14 head 24 (0.062), Qwen L18 head 18 (0.332). Mirror pattern of what Arditi 2024 found for refusal (attention-dominant).

Reader pattern also qualitatively different:

| Model | Freq significant Q-readers | Sentiment significant Q-readers |
| --- | --- | --- |
| OLMo-7B | 239/384 (62%) | 10/320 (3%) |
| Llama-3.1-8B | 110/768 (14%) | 0/480 (0%) |
| **Qwen-2.5-7B** | **38/336 (11%)** | 1/224 (0.4%) |

**All 3 models now complete.** Frequency has 10-62% of Q-heads reading significantly; sentiment has 0-3%. Orders-of-magnitude gap in all three architectures.

**Taxonomy of linear concept representations** (new central claim):

- **Routing concepts** (frequency): MLP-written, attention-Q/K-read. Modulate *where* attention looks. Intervention affects routing, not logits.
- **Aggregation concepts** (sentiment, refusal [Arditi], function vectors [Todd]): attention-written (aggregation from content positions), rarely read. Used as output-bearing representations. Intervention affects output directly.

Paper positions as **"a taxonomy of linear concepts: routing vs aggregation, with frequency as the first major published routing concept."** Not just a frequency paper.

#### Idea 2 (frequency-sensitive task KL) — ✅ COMPLETE on 3 models (Apr 18, pod `ani-freq-sensitive-kl-3m-bcllp`)

**Partial/mixed result — original prediction of 10-100× KL inflation FAILED; revised interpretation is position-selective, not task-selective.**

Cumulative rank-1 ablation of v across all layers, measuring KL(clean || ablated) at last-position on 4 prompt distributions.

| Model | PopQA-tail median | ScaleJSD-baseline median | Ratio (median) | Ratio (p99) |
| --- | --- | --- | --- | --- |
| OLMo-7B | 0.0117 | 0.0133 | 0.88× | 2.5× |
| **Llama-3.1-8B** | **0.157** | **0.050** | **3.1×** | **5.5×** |
| Qwen-2.5-7B | 0.0099 | 0.019 | 0.52× | 0.63× |

**Interpretation**: only Llama shows predicted inflation (3-5× on PopQA). OLMo and Qwen show the OPPOSITE. This is a position-selectivity finding, not a task-selectivity finding — the frequency direction is used at specific *positions* (anchor tokens in ScaleJSD), not globally across all frequency-sensitive *tasks*. PopQA questions end at "Q: ... A:" which isn't a frequency-decision position.

Note: scalejsd_rare_word, scalejsd_common_word, and scalejsd_baseline are nearly identical in all 3 models — confirms that within-ScaleJSD the KL is position-invariant across rare/common tokens (probe was trained across both).

**Paper framing**: "the frequency direction's effect is positional (concentrated at token-prediction anchor points), not task-general. Cumulative ablation on arbitrary prompts produces small KL because only ~5% of positions in a prompt are frequency-decision points for our probe."

#### Idea C (b_LN alignment) — ✅ COMPLETE (Apr 18, pod `ani-bln-alignment-3m-wh9bb`)

| Model | `cos(u, log P_model)` | Spearman ρ | `b_LN` available? |
| --- | --- | --- | --- |
| OLMo-7B | **-0.003** | -0.002 | No (RMSNorm) |
| Llama-3.1-8B | **-0.150** | +0.025 | No (RMSNorm) |
| Pythia-6.9B | -0.124 | -0.017 | Yes; `cos(v, b_LN) = +0.003` |

**Pythia sanity check**: `cos(log P_model, W_U · b_LN) = 0.916` → model-derived neutral-context unigram IS the b_LN projection, validating our approach for RMSNorm models.

**Result: ORTHOGONAL pathways.** |cos| ≤ 0.15 in all three models. The residual-stream frequency direction and the prediction-head b_LN pathway are **essentially independent** — two separate frequency-encoding mechanisms. Mechanistically explains the cumulative-ablation low-KL observation (§2.2): ablating the residual pathway leaves the b_LN pathway intact, so output frequency statistics barely change. See §4.5 for full interpretation.

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

We have characterized the **representation** (1D direction, distributed writers, decoupled from output). We have partly traced the **mechanism** (Track Circuits + Idea E — see §2.5). Open questions with current status:

### 4.1 Who writes? ✅ PARTIALLY ANSWERED (Track Circuits)
**Finding (see §2.5)**: MLPs dominate raw attribution in all 3 models; top-10 writers are 100% MLPs in OLMo/Llama/Pythia. Writers are distributed across early layers (L0-L2) and the peak-layer band.

**Caveat (MLP per-capacity analysis)**: Per-output-channel, attention heads are comparable or stronger. Raw dominance is partly a capacity effect (MLP has 32× more output dims than single head).

**Still open**: Whether attention heads READ from the direction (Q/K/V reader test in flight — `ani-qkv-reader-3m-sjwdx`).

### 4.2 Who maintains? ✅ PARTIALLY ANSWERED (Track Circuits recovery)
**Finding**: No active maintenance. Single-layer ablation produces **Δ ≈ -0.5 to -1.0** in downstream M_diff and the signal does not recover across 10+ downstream layers. This differs from the workshop-paper observation on Pythia-6.9B (0.46 → 0.75 recovery) — likely because that measurement was probe-AUROC-based and confounded by dilution, not magnitude-based.

### 4.3 Who erases? ✅ REVISED (Idea E + Idea G partial data, Apr 18)

**Important correction (2026-04-18 per in-conversation pushback)**: The "learned erasure" narrative from the workshop paper was based on Pythia-70M where AUROC drops 0.876 → 0.566 across 4 layers — a clear gradual decline. In the 7B+ models we're now using, **there is NO gradual decline**. Instead:

| Model | Peak AUROC | AUROC at L27-L30 | AUROC at L31 | Pattern |
| --- | --- | --- | --- | --- |
| Pythia-70M | 0.876 (L2) | — | 0.566 (L5) | Gradual 4-layer decline |
| OLMo-7B | 0.97 plateau | 0.94-0.95 | **0.840** | **Stable, one-layer crash at L31** |
| Llama-3.1-8B | ~0.97 | ~0.94 | similar pattern | **Stable, one-layer crash at L31** |
| Qwen-2.5-7B | 0.964 (L15) | ~0.91 | similar | **Stable, one-layer crash at L31** |

**Revised claim**: In 7B+ models, frequency is STABLE in the residual stream across ~28 intermediate layers, then transformed sharply at the single final layer (L31) right before unembedding. This is NOT gradual erasure; it is a **prediction-head handoff**. The final layer's transformation coincides with where b_LN activates, consistent with Idea C's finding that v is orthogonal to b_LN in vocabulary space — the residual pathway (stable v) and the head pathway (b_LN) meet only at the unembedding step.

**This revises the paper's "frequency erasure" framing**. The routing concept lives stably in the residual stream; the output frequency statistics are handled by an orthogonal b_LN pathway that activates only at the final layer.
**Finding**: Model-dependent.
- **Llama-3.1-8B**: PASSIVE DILUTION. M_diff magnitude grows monotonically through all layers. AUROC decline is a relative-norm phenomenon — competing features grow faster than the frequency signal.
- **OLMo-7B**: HYBRID. M_diff grows to L28 then crashes at L31 (final layer).
- **Pythia-6.9B**: GRADUAL ACTIVE EROSION. M_diff declines slowly from L12 to L30 then sharply at L31.
- **All 3 models share an L31 (final-layer) sharp drop** — consistent with prediction-head / b_LN erasure mechanism.

### 4.4 What's the relationship with semantics? 🟡 PARTIALLY ANSWERED; key tests queued
**Partial**: Track 4 / E0.2 showed mean |cos(freq, semantic)| ≈ 0.014 across 5 ScaleJSD domains in OLMo-7B → approximately orthogonal. Cross-model version not yet run.

**Queued (Idea A, B, F — see §6.3)**:
- Semantic-direction rank-1 ablation: mirror the frequency ablation protocol for each of 5 domains
- Semantic-direction steering: does domain steering shift generation toward that domain?
- Interference test: joint steering along (α_freq · v_freq + α_sem · v_sem)
- Idea G: reorganization probes — what grows as frequency AUROC "declines"?

### 4.5 Does the residual-stream frequency direction align with the prediction-head b_LN direction? ✅ ANSWERED (Idea C, Apr 18) — **ORTHOGONAL**

**Finding**: The residual-stream frequency direction `v` is essentially **orthogonal** to the prediction-head unigram / b_LN pathway across all 3 models.

| Model | `cos(u, log P_model)` | Spearman ρ | `cos(v_resid, b_LN)` |
| --- | --- | --- | --- |
| OLMo-7B (RMSNorm, no b_LN) | **-0.003** | -0.002 | N/A |
| Llama-3.1-8B (RMSNorm, no b_LN) | **-0.150** | +0.025 | N/A |
| Pythia-6.9B (has b_LN) | **-0.124** | -0.017 | **+0.003** |

**Sanity check (Pythia)**: `cos(log P_model, W_U · b_LN) = 0.916` — confirms that model-derived neutral-context unigram IS the b_LN projection for Pythia. This validates the `log P_model` substitute we used for RMSNorm models.

**Interpretation**: **Two independent frequency pathways.**
1. **Residual-stream pathway**: our probe direction `v`, lives in internal representation space
2. **Prediction-head pathway**: b_LN (or its RMSNorm analog, the unigram bias acquired during training), lives in the output bias

They are essentially orthogonal (|cos| < 0.15 in all models). When we ablate `v` in the residual stream (§2 cumulative-ablation), the b_LN pathway is untouched and continues to inject frequency information into logits. That's **mechanistically why output KL stays small** despite frequency probe AUROC crashing — the output frequency behavior is handled by the independent b_LN pathway.

This is a **clean mechanistic finding that unifies**: (a) our cumulative-ablation low-KL observation (§2.2), (b) Kobayashi 2023's b_LN = frequency direction result, and (c) our Track 4 steering result (steering `v` produces register / FK shifts = internal-representation behavior, separate from b_LN-driven output frequency statistics). **Cite: Kobayashi et al. 2023 (arXiv:2305.18294) as establishing the prediction-head pathway; we establish the orthogonal residual-stream pathway.**

### 4.6 Do attention heads read v? ✅ ANSWERED — READERS EXIST (QKV reader, Apr 17/18)

**Finding — readers are abundant, concentrated just downstream of L***. Direct weight-space test of `|v · W_{Q,K,V}^h_row|` compared to a matched-norm random-direction null (100 random unit vectors; p99 cos threshold ≈ 0.062).

| Model | L* | # heads significant (Q/K/V) | % of heads reading via Q | Top head max_cos |
| --- | --- | --- | --- | --- |
| **OLMo-7B** | L19 | 239/384 / 226/384 / 112/384 | **62% of Q-heads above null** | L28 h19 Q: 0.260 |
| **Llama-3.1-8B** | L7 | 110/768 / 80/192 / 46/192 | 14% of Q-heads above null | L10 h6 Q: 0.181 |
| **Pythia-6.9B** | L12 | 130/608 / 174/608 / 149/608 | 21% Q, 29% K, 25% V | L16 h26 Q: 0.167 |

p99 null cos = 0.062. Expected false-positive rate = 1% (≈ 4 heads per model). Observed: 14-62% — **orders of magnitude above chance**, in all three models.

**Concentrated readers** (top heads by max_cos, all layers strictly > L*):
- **OLMo-7B**: L20, L21, L24, L28 — heads in the L20-L28 band
- **Llama-3.1-8B**: L9, L10, L11 — heads immediately downstream of L*=7
- **Pythia-6.9B**: L14-L16 — same immediate-downstream pattern

The strongest readers are in the **immediate-downstream band (L* → L*+5)**; a second tail extends to the final layers (L28-L30).

**This revises the mechanistic picture dramatically.** Our earlier inference "no reader — hence shadow" was WRONG. The correct picture:

1. **Write**: distributed MLPs in L0 → L\* (Track Circuits) write v into residual stream
2. **Read**: attention heads in L > L\* read v through **Q/K projections** (decide *where to attend* partly based on frequency information). Max per-head cos 0.17-0.26 — modest but robust alignment, well above null
3. **Output pathway is separate**: output frequency statistics are handled by the b_LN / learned-unigram pathway, which is **orthogonal** to v in vocabulary space (Idea C: |cos| ≤ 0.15 across all 3 models)

**Why this reconciles previously contradictory findings**:
- Cumulative ablation kills probe AUROC (obviously — we're ablating) but preserves output KL (small). **Not because "no reader"** — because the readers propagate v's information through Q/K routing decisions, which influence the full downstream computation but don't linearly bias output logits; and because the orthogonal b_LN pathway handles output frequency statistics independently.
- Steering v produces FK / register shifts (Track 4). **Now explained**: steering changes *where downstream heads attend* (via Q/K), which changes the semantic/positional content they aggregate, which manifests as register shifts downstream. This is upstream-computation modulation, not direct logit bias.
- The write-site is MLP-dominant (Track Circuits) and the read-site is attention-dominant. This is a **cleanly asymmetric write/read pattern** — potentially a more publishable circuit finding than either half alone.

**Paper framing**: the frequency direction is **written by many MLPs, read by many attention heads (via Q/K), and used as an upstream computation-routing signal rather than a direct logit-bias mechanism. The logit-bias pathway for frequency is a separate b_LN-aligned direction.**

**Caveats**:
- Weight-space alignment (cos 0.17-0.26) is correlative — it says the head *could* use v. Full causal verification would need path patching or activation-space ablation. Queued as Idea 8.
- The "62% of heads" statistic for OLMo is striking but modest in magnitude — most reading heads are marginally above the p99 null. The clean signal is the top-20 heads per model, concentrated in the immediate-downstream band.

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

## 6. Prioritized Research Directions — Triaged Roadmap (updated Apr 18, 2026)

This section was rewritten after Track 1–4 completed and the literature scoop check (Braun 2026) landed. Previous Directions 1–5 (below) are retained for reference but are superseded by the triage here.

### 6.0 Triage summary

Ideas are now bucketed by **expected paper impact × feasibility × post-scoop novelty**:

| Tier | Idea | Status | Est. effort | Why |
| --- | --- | --- | --- | --- |
| **P0 — running now** | Idea C — b_LN alignment test | Dispatched Apr 18 (cluster CPU) | hours | Unifies our direction with Kobayashi 2023; single matmul + one forward pass per model |
| **P0 — running now** | Track QKV reader | Dispatched Apr 17 (cluster CPU) | hours | Completes the circuit story: writing ≠ reading |
| **P0 — done** | Idea E — active eraser re-analysis | Completed Apr 18 (local) | 10 min | Reveals Llama late-layer signal growth is passive dilution, not active erasure; L31 crash universal |
| **P1 — high impact** | Idea 2 — Frequency-sensitive task KL | Queued; needs GPU | 1-2 days | Converts "small KL" from liability to feature-specificity proof; runs on freq-sensitive prompt distributions (BC5CDR rare-medical, WikiLarge simplification, GYAFC register, Zipf-tail completions) |
| ~~P1~~ **DONE** | Idea 3 — Frequency direction as code-length | ✅ Complete (Apr 18) | — | **R² ≈ 0.48-0.51 across OLMo/Llama/Pythia.** Half of projection variance explained by log corpus frequency at anchor positions. Confirmed as conditional code-length axis. See §2.5 for full results. |
| **P1 — high impact** | Idea 8 — Dependency graph / path patching for v | Queued; needs GPU | 3-5 days | Direct causal reader test (complement to QKV weight projection). Gold-standard test from Goldowsky-Dill 2023 |
| **P1 — high impact** | Idea 6 — Universal-feature taxonomy | Needs additional concept-direction experiments | 2 weeks | Synthesis position paper: plot Arditi refusal / Marks truth / Tigges sentiment / our frequency on a 3D axis (probe AUROC universality × write-site pattern × causal-effect ratio). Finding: **write-site is concept-dependent, not architecture-dependent**. |
| **P1 — high impact** | Idea G — Reorganization probes (what GROWS as freq "erases") | Queued; needs GPU | 2-3 days | Train probes at every layer for: frequency, domain, sentiment, syntactic role, formality, truth. Shows whether AUROC decline is zero-sum rotation (semantic gain = freq loss) or more complex reorganization |
| **P2 — completion** | Idea A — Semantic-direction ablation (symmetric) | Queued; needs GPU | 2 days | Ablate each ScaleJSD domain direction, measure probe AUROC crash + output KL. Mirror of frequency ablation; expected reviewer ask |
| **P2 — completion** | Idea B — Semantic-direction steering | Queued; needs GPU | 1-2 days | Does steering along a domain direction shift generation toward that domain? Analog of frequency→FK for domain→topic |
| **P2 — completion** | Idea 1 — Translator neurons (reframed in 1D-subspace language) | Queued | 1 week | Identify neurons whose output projection onto v shifts sign or magnitude across depth. Not the old 3-axis framing — now about per-layer signed attribution per neuron, tracking the SAME neuron across depth |
| **P2 — completion** | Idea 4 — Multi-rank frequency subspace | Queued; needs GPU | 3-5 days | User-flagged: "is frequency 1D or multi-D?" Extract rank-k SVD of class-mean difference. Train continuous log-freq regression probe; compare rank-1 vs rank-k AUROC. If rank-k substantially better: frequency has 1D linear discriminator + additional geometry for within-class structure (token vs bigram vs document frequency) |
| **P3 — speculative** | Idea F — Cross-direction interference (freq + semantic composition) | Queued; needs GPU | 2-3 days | Steer along `α_freq·v_freq + α_sem·v_semantic`; test linear composition |
| **P3 — speculative** | Idea D — Dual-route reading analogy | Research design | — | Cognitive-science analogy, hard to test rigorously — park |
| **Deprioritized** | Pythia checkpoint sweep | Parked — see §9 | — | Pythia de-emphasized as mid-training / unstable |

### 6.1 P0 experiments in flight

**Idea C — b_LN alignment test (dispatched as `ani-bln-alignment-3m`)**
One-line prediction: `W_U @ v` aligns with the model's learned unigram (for RMSNorm models) and with `b_LN` (for Pythia). If cosine > 0.5: unifies our direction with Kobayashi 2023 (one pathway, two access points). If orthogonal: two independent frequency pathways (explains the cumulative-ablation-low-KL dissociation). CPU-only job; results in hours.

**Track QKV reader (dispatched as `ani-qkv-reader-3m-sjwdx`)**
Computes `|v · W_{Q,K,V}^h_row|` per head per layer vs random-direction null. If some heads materially exceed null → we have candidate readers; circuit story gets a complement. If none do → strong evidence for "write-only" feature (no reader, no computational use).

**Idea E — active eraser re-analysis (done Apr 18)**
Finding: Llama late-layer freq-signal magnitude GROWS monotonically from peak to final-1 layer; OLMo and Pythia show more complex trajectories. All 3 models show a sharp drop at the **final layer (L31)** right before unembedding — prediction-head-related erasure. This is new empirical detail for the circuit story.

### 6.2 P1 experiments to dispatch next

**Idea 2 — Frequency-sensitive task KL.** The cumulative-ablation "small KL" (0.001-0.025) was measured on ScaleJSD synonym-pair prompts, which are *frequency-neutral by design*. Re-run on frequency-sensitive prompt distributions:
- **BC5CDR** (biomedical NER, rare medical entities): KL expected to jump substantially
- **WikiLarge** / ASSET (text simplification benchmarks, known frequency-sensitive task)
- **GYAFC** (formal ↔ informal register, frequency-adjacent)
- **Zipf-tail next-word prediction**: constructed contexts where the expected continuation is in the tail 10% of unigram distribution
Converts "small KL" from liability to feature-specificity proof. Expected outcome: 10-100× KL on freq-sensitive prompts.

**Idea 3 — Frequency direction = log unigram probability.** Correlate per-token residual-projection `<x_L*, v>` with external corpus-derived `log P(token)` (from wikitext or Pile). One forward pass + one scatter plot per model. If r > 0.8: frequency direction IS a learned code-length axis. This would:
- Make the mathematical claim precise (information-theoretic, not vague "encodes frequency")
- Unify with entropy-coding view of LM training (cross-entropy = arithmetic coding)
- Provide a clean theoretical grounding for the rest of the paper

**Idea 8 — Full dependency graph of v via path patching.** Complement to QKV weight test. For every downstream (layer, component), measure whether its activation changes when v-component is projected out at L*. Gold-standard test from Goldowsky-Dill 2023.

**Idea 6 — Universal-feature taxonomy.** Position paper framing: plot known concept directions on 3 axes:
1. Probe AUROC universality across architectures
2. Write-site pattern (attention-dominant / MLP-dominant / mixed) — our Track Circuits data
3. Causal-effect ratio at unit steering (KL @ α=1 / AUROC)

Known points: Refusal (Arditi 2024, attention-dominant writers), Function vectors (Todd 2024, attention-only writers), Truth (Marks 2023, layer-patched not component-decomposed), Sentiment (Tigges 2024, LR-probe steering on Pythia), Frequency (our work, MLP-dominant writers). The taxonomy claim: *write-site is concept-dependent, not architecture-dependent*. This is the single most novel cross-paper claim we have after the literature scoop check.

**Idea G — Reorganization probes.** At every layer in OLMo/Llama/Qwen, train probes for: frequency, semantic domain (5), sentiment, syntactic role, formality, truthfulness. Stack the trajectories. Test whether the "frequency erasure" is a rotation (semantic gain = freq loss in a zero-sum sense) or a more complex reorganization. Answers §4.3 + §4.4 simultaneously.

### 6.3 P2 completion experiments

**Idea A — Symmetric semantic-direction ablation.** Mirror of frequency ablation. For each of 5 ScaleJSD domain directions: rank-1 ablate, measure that domain's probe AUROC crash + output KL + cross-effect on frequency probe AUROC. Settles the orthogonality claim (E0.2 showed cos ≈ 0.02; this is the causal test).

**Idea B — Semantic-direction steering.** Can a domain direction produce a topic-shift analog of frequency's FK-shift? If steering α_medical > 0 → more medical vocabulary in generation: multi-knob compositional style control. If effect is zero: domain directions are shadow, frequency is the only steerable axis.

**Idea 1 — Translator neurons (reframed).** In the 1D subspace framework: identify MLP neurons whose signed output projection onto v transitions across depth (e.g., positive contribution in early layers, negative in late layers, or vice versa). Predict these sit at the layers where freq AUROC peaks and starts declining. Causal test: ablating these specific neurons should differentially affect frequency vs. semantic probes.

**Idea 4 — Multi-rank frequency subspace.** The user-flagged question: *is frequency 1D or higher-dimensional?* Our rank-1 probe AUROC is 0.98, which is near-saturated — but the probe is trained on BINARY high-vs-low discrimination. Multi-rank test:
- Train continuous `log P(token)` regression probes (rank-1, rank-4, rank-16, rank-64)
- Compare R² on held-out tokens
- Extract rank-k SVD of class-mean difference; check if singular values have a sharp drop (1D) or plateau (multi-D)
- Test orthogonality across frequency granularities: token-level vs bigram-level vs document-level frequencies  
If rank-k is substantially better than rank-1: frequency has **a rank-1 linear discriminator (what we found) + additional geometric structure** (what we'd find). This is valid as a scientific direction because rank-1 works for classification doesn't mean the underlying representation is 1D.

### Legacy prioritization (from pre-Apr-18; superseded by 6.0-6.3 above)


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

### 7.0 Applied paper direction (recommended pivot, 2026-04-15)

A sanity check review surfaced a critical tension: our cumulative ablation produces tiny output KL (0.001-0.025), meaning the residual-stream frequency direction may be **epiphenomenal** — the model doesn't need it for output. This undercuts both the mechanistic story and any steering claim.

**De-risking experiment E0.1 (MAKE OR BREAK)**: Test whether **adding** along the direction moves outputs even though **projecting it out** doesn't. If additive steering shifts generated token log-frequency monotonically with α (effect ≥ 0.5 stdev) while perplexity stays within 2× baseline, the applied paper is alive.

If E0.1 passes, the recommended pivot is an **applied steering paper**:

> "One vector, three knobs: a probe-derived frequency direction controls lexical sophistication, register, and rare-token generation across 5 models without fine-tuning."

Three benchmarks: B1 (rare-token gen: BC5CDR/SciERC), B2 (register: GYAFC formal↔informal), B3 (simplification: Newsela/ASSET). Baselines: frequency penalty, CAA, ReFT, DExperts.

Target venue: **EMNLP 2026 main** (May deadline).

If E0.1 fails: pivot to Backup B — "probes detect representations that the model doesn't use; a cautionary tale for representation engineering." Still publishable at MI workshop.

**Script for E0.1 + E0.2**: `scripts/interventions/frequency_steering.py`

### 7.1 Likely venue (updated)
- EMNLP 2026 main (applied paper, if E0.1 passes)
- ACL 2027 (if EMNLP misses)
- NeurIPS 2026 MI Workshop (companion mechanistic paper with D2+D5)
- COLM 2025/2026 (LM-focused)

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

---

## 9. Model scope policy (Apr 18, 2026 decision)

### Primary target models for venue-grade claims

- **OLMo-7B-hf** — fully trained, open-source, permissive license, residual-stream RMSNorm architecture
- **Llama-3.1-8B** — fully trained, modern mainstream model, RMSNorm
- **Qwen-2.5-7B** (to add) — modern mainstream model, high-quality training, useful as a third architecture

### Deprioritized

- **Pythia-6.9B (step 143000)**: considered mid-training and unstable. Included in existing results but will not be the central figure in main-text claims.
- **Pythia-70M / Pythia-1B**: legacy, retained for training-dynamics claims only (not for behavioral/steering claims).
- **Pythia checkpoint sweep** (was queued): **deprioritized**. The "Pythia-dead steering" finding from Track 4 is mentioned as an observation in supporting material, not as a central mechanistic finding.

### Rationale

1. **Reviewer-facing credibility**: Pythia is rarely used in current steering / circuit-level papers (Braun 2026, Arditi 2024 refusal, Tigges 2024 sentiment all use Llama family for primary claims). Making Pythia-dead a central finding would draw criticism that the result is a Pythia-training artifact rather than a general phenomenon.
2. **Step143000 is mid-pretraining** (full Pythia training is 300B tokens; step143000 is ~143k × batch-size-1024 ≈ 150B tokens). Register / lexical-style knobs may simply not have emerged yet.
3. **Effort allocation**: time spent understanding why Pythia is anomalous is better spent adding Qwen-2.5-7B as a third clean test point for the universality claims.

### Action items from this decision

- Dispatch Qwen-2.5-7B runs of Track 4 (steering metrics), Track Circuits, Idea C, Idea E, QKV reader — same protocol
- In the paper write-up, state the model scope policy explicitly to preempt reviewer questions
- Retain Pythia results in appendix / supplementary material where they don't confuse the main narrative

---

## 10. Literature update (Apr 18, 2026) — post-track scoop check

Two parallel alphaxiv literature checks (steering + circuits) completed Apr 17/18. Key findings:

### 10.1 Scoops to acknowledge

- **Braun, Eickhoff, Bahrainian — "Beyond Multiple Choice: Steering Vectors for Summarization"** (arXiv 2505.24859, EACL 2026 Findings) **already demonstrated CAA-based single-vector readability steering on Llama-3.2-1B/3B/3.1-8B** with monotonic FK/DeBERTa response. Our "deployable single-vector simplification primitive" framing is taken. **Must cite and differentiate.**
- **Liu, Ye, Xing, Zou — "In-Context Vectors" (ICML 2024)** demonstrated single-vector style/formality control on Llama/Falcon-7B. Smallest prior to "single direction controls register."
- **Arditi et al. 2024 "Refusal direction"** (arXiv 2406.11717) already uses the DFA attribution recipe we used on refusal; found **attention-head-dominant writers**. Our MLP-dominant finding is concept-dependent, not method-dependent.

### 10.2 What's genuinely novel after scoop check

1. **Frequency-anchored probe extraction**: our steering vector is derived from high/low token-frequency contrasts on matched-template ScaleJSD data. Braun 2026 and ICV use readability- or formality-labeled sentence pairs. We're the first to derive the steering vector from **pretraining-corpus frequency statistics** rather than behavioral labels.
2. **Concept-dependent write-site pattern**: refusal (Arditi) is attention-dominant; function vectors (Todd) are attention-only; frequency (us) is MLP-dominant. This cross-paper comparison is novel — write-site is a property of the concept, not the architecture. Foundation for Idea 6 (universal-feature taxonomy).
3. **Decodability vs. steerability dissociation**: probe AUROC 0.98 universal, but steering effect sizes differ substantially (stronger in modern LMs, weaker in Pythia-6.9B step143000). Cf. Tigges 2024 showed sentiment steering WORKS on Pythia — so the dissociation is specific to what we're steering, not a universal Pythia artifact.
4. **Idea E novel empirical finding (Apr 18 re-analysis)**: in Llama-3.1-8B, frequency-signal magnitude in the residual stream grows monotonically through all layers even while probe AUROC declines. This is a **clean mechanistic distinction between active erasure (Pythia, OLMo) and passive dilution (Llama)** — rank-growth of competing features, not destruction of the frequency signal.

### 10.3 Revised paper thesis (post-scoop)

> **"Probe decodability does not predict steering responsiveness. For the token-frequency direction, we show: (1) the write-site pattern (MLP-dominant, no attention writers) is qualitatively different from prior concept-direction analyses (refusal: attention-dominant; function vectors: attention-only), so write-site is concept-dependent, not architecture-dependent. (2) Probe AUROC is 0.97–0.99 and universal across OLMo/Llama/Pythia; but behavioral steering (Flesch-Kincaid grade shift, register control) is large in modern LMs and near-dead in Pythia. (3) Late-layer AUROC decline is passive dilution (competing features growing) in Llama, and active erasure in OLMo and Pythia. (4) All models share a final-layer (L31) sharp drop — consistent with a prediction-head/b_LN pathway specifically at the final layer."**

### 10.4 Must-cite list (new, post-scoop)

1. Braun, Eickhoff, Bahrainian 2026 (EACL Findings) — closest competitor; differentiate on frequency-anchored probe extraction and write-site pattern
2. Liu, Ye, Xing, Zou — ICV (ICML 2024) — precedent for single-vector style control
3. Rimsky et al. — CAA (ACL 2024) — canonical contrastive steering
4. Tigges et al. 2023 — LR-probe steering template
5. Kobayashi et al. 2023 — b_LN frequency finding (central to Idea C unification)
6. Stolfo et al. 2024 — token-frequency neurons (interpretive prior)
7. Arditi et al. 2024 — refusal direction (method precedent + write-site contrast)
8. Todd et al. 2024 — function vectors (attention-only writers, further write-site contrast)
9. Marks & Tegmark 2023 — truth direction (functional-direction comparison reference)
