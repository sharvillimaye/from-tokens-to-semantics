# Workshop Review Audit: From Tokens to Semantics

Tracking every reviewer concern from the NeurIPS 2025 MechInterp Workshop reviews against current status, with prioritization for the conference paper.

---

## Reviewer 54oj

### Strengths Noted
- First systematic longitudinal study of polysemanticity during pretraining
- Three interdependent analyses that mutually reinforce findings
- Clever experimental design isolating frequency from semantics
- Successful integration of disparate interpretability approaches
- Reproducibility (score: 5/5)

### Concerns

| # | Concern | Status | Action Required |
|---|---------|--------|-----------------|
| 54oj-1 | Clustering threshold d_t chosen by "trial and error" (footnote 2) | Not addressed | Sensitivity analysis: sweep d_t over a range (e.g., 0.3–0.9 in steps of 0.1), rerun clustering, show n_clusters trends are qualitatively stable |
| 54oj-2 | Activation threshold 0.5 × max_activation lacks theoretical grounding | Not addressed | Same sweep approach. Also: try alternative thresholds (0.25×, 0.75×, absolute threshold > 0) and show robustness |
| 54oj-3 | Only analyzing every 20th neuron (70M) / every 60th neuron (160M) — 95-98% of neurons ignored | Partially addressed | Coverage-affinity experiment uses ALL neurons. Neuron embedding clustering (analysis 1) still subsamples. Either run on all neurons (compute permitting) or provide statistical justification (random sampling with CI) |
| 54oj-4 | Single semantic domain (country-capital pairs with rigid templates) | Addressed | ScaleJSD dataset: 5 domains (emotion, medical, legal, scientific, verb), 291 pairs, varied templates |
| 54oj-5 | Only MLP neurons — ignoring attention heads | Not addressed | Run JSD and polytope density on attention head outputs. At minimum for a subset of models. If same layerwise pattern holds, strengthens universality. If different, that's more interesting |
| 54oj-6 | No theoretical explanation for why coverage emerges as organizing principle | Not addressed (planned) | This is the normative argument — the centerpiece of the conference paper. See research-summary.md "Normative Argument" section |
| 54oj-7 | No connection to model capabilities / performance | Not addressed (planned) | Behavioral convergence experiment: track output distribution similarity for synonym pairs across checkpoints alongside internal metrics |
| 54oj-8 | Cross-lingual / multilingual analysis | Not addressed | Lower priority. Frame as future work unless a multilingual Pythia-equivalent exists with training checkpoints |

### Constructive Suggestions from 54oj
- Mathematical framework connecting coverage to capacity allocation (information-theoretic or lottery ticket)
- Track whether models that consolidate faster show better benchmark performance
- Investigate whether "explore-consolidate-stabilize" aligns with capability emergence
- Extend to temporal expressions, mathematical concepts, syntactic structures
- Cross-lingual analysis in multilingual models

---

## Reviewer Q3Mh

### Strengths Noted
- Truly novel longitudinal research
- Mostly clear method definitions
- Many complementary metrics and visualizations
- Clear Introduction, Conclusion, Limitations
- Web app + anonymized repo (nice touches)

### Concerns

| # | Concern | Status | Action Required |
|---|---------|--------|-----------------|
| Q3Mh-1 | Methods lack motivation before formal definitions — JSD analysis hard to understand | Writing quality | Rewrite methods section: motivate each analysis before defining it. "We want to measure X because Y, so we define Z" |
| Q3Mh-2 | No error bars (Figures 1, 2, 3) — want multiple training runs | Cannot fully address | Pythia/OLMo have single training runs. Partially address with: (a) bootstrap CIs over synonym pairs within a run, (b) cross-domain variance as a proxy for robustness, (c) acknowledge honestly |
| Q3Mh-3 | Figures 1, 3, 6 need color gradients to distinguish layers/models | Not addressed | Presentation fix. Use sequential colormaps (viridis/plasma) for layer depth, distinct palettes for model size |
| Q3Mh-4 | Alternative polysemanticity proxy — e.g., SAE features | Not addressed | Train SAEs at multiple checkpoints, measure feature-per-neuron counts, show trajectory tracks n_clusters. Would significantly strengthen claims with field-standard tooling |
| Q3Mh-5 | How noisy is clustering? Sensitivity to d_t, δ, k | Not addressed | Same as 54oj-1/54oj-2. Both reviewers flagged this — high priority |
| Q3Mh-6 | Alternative clustering approach for validation | Not addressed | Run k-means or DBSCAN alongside HAC. Show similar qualitative patterns |
| Q3Mh-7 | No normative explanation for polysemanticity evolution | Not addressed (planned) | Same as 54oj-6. The normative argument |
| Q3Mh-8 | Why only MLP neurons? Residual stream? Attention out? | Not addressed | Same as 54oj-5. Attention head analysis needed |
| Q3Mh-9 | Final sentence about "training-time tools" feels unsupported | Not addressed | Either remove the claim or support it with behavioral convergence results showing metrics predict capability emergence |

### Constructive Suggestions from Q3Mh
- Additional support for claims and extra experiments
- A normative explanation would upgrade the contribution significantly
- Cleaner figures throughout

---

## Area Chair (Jtpc)

| # | Concern | Status | Action Required |
|---|---------|--------|-----------------|
| AC-1 | "Curious to know how outlier activations would affect metrics/analyses" | Not addressed | Robustness check: rerun JSD and affinity metrics with top 1% activations clipped (winsorized). Check if a small fraction of extreme-activation neurons dominate the results. Report results either way |

---

## Consolidated Action Items by Priority

### Priority 1: Conference Paper Centerpiece (blocks submission)

| Item | Addresses | Effort | Description |
|------|-----------|--------|-------------|
| Normative argument | 54oj-6, Q3Mh-7 | 4-6 weeks | Gradient dynamics explanation for why layerwise decomposition emerges. Includes: gradient SNR tracking, linear probes, toy model. See research-summary.md for full plan |
| Behavioral convergence | 54oj-7, Q3Mh-9 | 1-2 weeks | Track output distribution similarity for synonym pairs across checkpoints. Show internal metrics lead behavioral convergence. Supports the "training-time tools" claim |

### Priority 2: Robustness (both reviewers flagged, will be flagged again)

| Item | Addresses | Effort | Description |
|------|-----------|--------|-------------|
| Clustering hyperparameter sweep | 54oj-1, 54oj-2, Q3Mh-5 | 3-5 days | Sweep d_t, activation threshold, show results stable. Table or figure showing n_clusters trends across parameter choices |
| Alternative clustering method | Q3Mh-6 | 2-3 days | Run DBSCAN or k-means alongside HAC, show qualitative agreement |
| Outlier activation analysis | AC-1 | 1-2 days | Winsorize top 1% activations, rerun JSD/affinity, report impact |
| Neuron subsampling justification | 54oj-3 | 2-3 days | Either run clustering on all neurons (if compute allows) or random-sample with bootstrap CIs |

### Priority 3: Scope Extension (both reviewers asked)

| Item | Addresses | Effort | Description |
|------|-----------|--------|-------------|
| Attention head analysis | 54oj-5, Q3Mh-8 | 1-2 weeks | Run JSD and polytope density on attention head outputs for subset of models. Report whether same layerwise pattern holds |
| SAE-based validation | Q3Mh-4 | 2-3 weeks | Train SAEs at multiple checkpoints, measure features-per-neuron, compare trajectory with n_clusters |

### Priority 4: Presentation (before submission)

| Item | Addresses | Effort | Description |
|------|-----------|--------|-------------|
| Methods rewrite | Q3Mh-1 | 2-3 days | Motivate each analysis before defining it |
| Figure improvements | Q3Mh-3 | 1-2 days | Sequential colormaps, better layer/model distinction |
| Error bars / confidence intervals | Q3Mh-2 | 2-3 days | Bootstrap CIs over synonym pairs, cross-domain variance |
| Remove or support "training-time tools" claim | Q3Mh-9 | Depends on behavioral convergence results |

### Lower Priority (future work is acceptable)

| Item | Addresses | Notes |
|------|-----------|-------|
| Cross-lingual analysis | 54oj-8 | Frame as future work |
| Temporal / mathematical / syntactic domains | 54oj constructive | 5 domains may be enough; mention in future work |

---

## Summary

The normative argument is the single biggest gap — both reviewers independently identified it and Q3Mh explicitly said it would have raised the score. But it is not the only gap. The robustness concerns (clustering sensitivity, outlier activations, subsampling) were flagged by both reviewers and will resurface at a main conference. The scope limitation (MLP-only) was also flagged by both.

For a conference submission, priorities 1-3 are likely all needed. Priority 4 is polish before submission.
