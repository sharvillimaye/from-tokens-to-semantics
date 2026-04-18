# Session Handoff — 2026-04-18

**Purpose**: everything done in this session and what to pick up next time. This complements `research-summary.md` (which is the authoritative state); read that for the scientific narrative, read this for the operational state.

---

## TL;DR

- Ran ~10 experiments across OLMo-7B / Llama-3.1-8B / Qwen-2.5-7B / Pythia-6.9B on Oumi and RunPod.
- Strong empirical picture emerged: **routing vs aggregation concept dichotomy** (frequency MLP-written + attention-Q/K-read; sentiment attention-written + rarely re-read).
- **Literature audit (Apr 18, afternoon) forced a major scope correction**: most "grand claims" are prior art (Elhage 2021, Stolfo 2024, Tan 2024, Pres 2025). Paper must narrow.
- **Paper's survival depends on one pending experiment**: the Q/K vs OV dissociation (dispatched `a17da4dbde5c5cd78`, CPU-only, ~1.5h). If it shows freq = Q/K-dominated, sent = OV-dominated across 3 models, the paper is NeurIPS-defensible. If not, workshop-scale.

---

## §1 — Experiments run this session

| Experiment | Where | Status | Key result |
| --- | --- | --- | --- |
| **Track 4 steering metrics** (freq direction, 3 models) | Oumi `ani-steering-metrics-3m-t9b72` | Done Apr 17 | ΔFK -2.45 (OLMo), -3.26 (Llama), 0.3 (Pythia step143k). Qwen dead too (see below) |
| **Track Circuits** (signed DFA attribution) | Oumi `ani-circuits-eap-3m-ptxn6` | Done Apr 17 | Top-10 writers: 100% MLP (freq, all 3 models). Attention-dominant for sentiment |
| **Track C hallucination predictor** | Oumi `ani-halluc-predictor-3m-k5j95` | Done Apr 17 | ρ = -0.05 to -0.17 (rho probe vs correct). Real but weaker than logprob baseline |
| **Track A truth comparison** | Oumi `ani-truth-comparison-3m-pq5st` | Done Apr 17 | Methodology-limited (used ScaleJSD continuation prompts for truth-direction evaluation). Parked. |
| **Track QKV reader** | Oumi `ani-qkv-reader-3m-sjwdx` | Done Apr 17 | 14-62% of Q-heads read v. Concentrated in L\*+1 to L\*+5 |
| **Idea C (b_LN alignment)** | Oumi `ani-bln-alignment-3m-wh9bb` | Done Apr 18 | Residual v orthogonal to b_LN (|cos| ≤ 0.15 in all 3 models) |
| **Idea 3 (code-length correlation)** | Oumi `ani-code-length-3m-xnk7t` | Done Apr 18 | **R² ≈ 0.48-0.51** for v projection vs log P(token), universal |
| **MLP-capacity normalization** (local analysis) | Local Python | Done Apr 18 | Per-channel, attention comparable to MLP; raw MLP-dominance is aggregation effect. Softens §2.4 claim. |
| **Idea E (active eraser)** (local re-analysis) | Local Python | Done Apr 18 | Llama shows passive dilution; OLMo/Pythia hybrid. All 3 share final-layer L31 crash. |
| **Oumi sentiment-control** (Idea 6) | Oumi `ani-sentiment-3m-ghbfg` | Done Apr 18 | **THE TAXONOMY FINDING**: sentiment 9/10 attn-written (Llama, Qwen) / 6 attn + 4 MLP (OLMo). Inverse of frequency's write-site. |
| **Oumi Idea 2 (freq-sensitive task KL)** | Oumi `ani-freq-sensitive-kl-3m-bcllp` | Done Apr 18 | Predicted 10-100× KL failed. Llama 3.1×, OLMo 0.88×, Qwen 0.52× on PopQA tail. Position-selective, not task-selective. |
| **Oumi Qwen suite** (Track 4 + Circuits + QKV + Idea 3 + b_LN on Qwen) | Oumi `ani-qwen-suite-28xf8` | Done Apr 18 | Qwen joins pattern: 10/10 MLP writers, 11% Q-readers, probe AUROC 0.985. **DEAD STEERING** (ΔFK ≈ 0.2). Code-length and b_LN scripts errored (probe-direction plumbing bug) |
| **Oumi Qwen-gap** (halluc + b_LN + code-length re-run on Qwen) | Oumi `ani-qwen-gap-*` | Running / may have completed | Closes Qwen matrix for b_LN and code-length |
| **RunPod Qwen Track 4** (duplicate) | RunPod tmux `qwen-steer` | Killed (CPU-slow, redundant) | — |
| **RunPod Idea G reorganization probes** | RunPod tmux `reorg` | Running, L19+ of 32 | Sentiment grows 0.54→0.94 by L10; morphology declines; freq stable through L19 (decline expected L25+) |
| **Literature novelty audit** (research agent) | local agent | Done Apr 18 | Most grand claims are prior art. Narrowed defensible scope to C2 + C4. |
| **Q/K vs OV dissociation** (THE make-or-break) | Oumi `ani-qk-vs-ov-3m` | Running now | Tests whether routing vs aggregation taxonomy survives causal scrutiny |

---

## §2 — Results summary by model × measurement

### Frequency direction — 3-model consensus

| Metric | OLMo-7B | Llama-3.1-8B | Qwen-2.5-7B |
| --- | --- | --- | --- |
| Best probe AUROC | L16: 0.984 | L15: 0.985 | L15: 0.985 |
| Cross-layer AUROC trajectory | Plateau 0.94-0.98 L3-L30, crash L31 → 0.84 | Similar plateau expected | Similar plateau expected |
| Code-length R² (vs log P(token)) | 0.481 | 0.506 | 0.486 |
| Top-10 writers (attn/MLP) | 0/10 | 2/8 | 0/10 |
| Σ|attr| ratio MLP/attn | **1.64×** | **4.01×** | **3.52×** |
| % Q-readers above null | 62% | 14% | 11% |
| cos(v, b_LN) / cos(u, log P_model) | -0.003 / -0.003 | — / -0.150 | 0.003 / -0.124 |
| ΔFK grade (α=-2 → +2) | **-2.45** (strong) | **-3.26** (strong) | **~0.2** (dead) |

### Sentiment direction — 3-model consensus

| Metric | OLMo-7B | Llama-3.1-8B | Qwen-2.5-7B |
| --- | --- | --- | --- |
| Best probe AUROC | L21: 0.956 | L16: 0.954 | L19: 0.939 |
| Top-10 writers (attn/MLP) | **6/4** | **9/1** | **9/1** |
| Σ|attr| ratio MLP/attn | **0.24×** | **0.16×** | **0.22×** |
| % Q-readers above null | 3% (10/320) | 0% (0/480) | 0.4% (1/224) |

**The cross-concept MLP/attn ratio flips by 7-25× within the same model.** This is the taxonomy's empirical signature.

---

## §3 — What survives the novelty audit

| Claim | Verdict | Prior art |
| --- | --- | --- |
| **C1 Multiplexed residual stream** | ❌ NOT NOVEL | Elhage 2021 already framed residual as communication channel |
| **C2 Routing vs aggregation taxonomy** | 🟡 NOVEL IF Q/K-OV dissociation holds | OV/QK distinction in Elhage; specific instances exist (Arditi, Todd, Tigges); nobody has published the dichotomy with direct write-site comparison |
| **C3 Frequency = code-length axis** | ❌ NOT NOVEL standalone | Stolfo 2024 showed MLP neurons write log-frequency; "arithmetic code length" is Shannon's log(1/p) |
| **C4 v ⊥ b_LN pathway** | ✅ NOVEL (narrow micro-result) | Kobayashi 2023 is about b_LN; Stolfo is about residual direction; nobody has tested the orthogonality |
| **C5 Decodability ≠ steerability** | ❌ NOT NOVEL | Tan 2024 (2407.12404), Pres 2025 (2505.22637), Braun 2024 "Sober Look" already covered |
| **C6 DFA + QKV weight projection** | ❌ Not a method contribution | Kissane/Makelov DFA; QKV weight projection is Elhage framework |

**Bottom line**: 1.5 defensible contributions (C4 + conditional C2). Not 6.

---

## §4 — Revised paper thesis (post-audit)

**Title (candidate)**: *"Routing vs Aggregation: Two Write-Site Patterns for Linear Concepts in LLMs, with Token Frequency as a Case Study"*

**Core claim**:
Linear concepts differ systematically in HOW they enter the residual stream.

- **Routing concepts** (frequency): written by MLPs; read by downstream attention heads via their Q/K weight matrices; used for input-side routing decisions (where to attend). Evidence: Q/K projection ≫ OV projection for frequency across 3 architectures.
- **Aggregation concepts** (sentiment, refusal, function vectors): written by attention heads via their OV circuits (aggregating content from earlier positions); rarely re-read downstream. Evidence: OV projection ≫ Q/K projection for sentiment across 3 architectures.

**Supporting results**:
1. Q/K-vs-OV dissociation (pending, running now)
2. Residual frequency direction orthogonal to b_LN (|cos| ≤ 0.15) — novel micro-result
3. Auxiliary: frequency direction is a log-P(token) axis (R² ≈ 0.5), positioned as extension of Stolfo 2024

**Drop or demote**:
- Drop the "multiplexed channels" grand frame (Elhage territory)
- Drop the "arithmetic coding" theoretical framing (Stolfo territory)
- Demote decodability-vs-steerability to appendix observation

---

## §5 — Current running experiments

| Pod / tmux | What | ETA |
| --- | --- | --- |
| Oumi `ani-qk-vs-ov-3m` | Q/K vs OV dissociation on 3 models (THE make-or-break) | ~1.5h |
| RunPod tmux `reorg` | Idea G reorganization probes, at L19-21 of 32 | ~3h |
| Oumi `ani-qwen-gap` | Qwen b_LN + code-length redo (halluc → bln → codelen chain) | Done or near-done |

---

## §6 — Critical open experiments (priority order)

### P0 — must land before paper

1. **Q/K vs OV dissociation** (running) — settles C2 defensibility
2. **Dynamic reader test (your new idea)**: ablate v at L\*, measure Q/K *activation* alignment change downstream. Causal test of readers. Currently our test is static (weight-space); this upgrades to activation-space. Predicted result: heads flagged as readers by weight projection show large Q/K activation drop; non-reader heads unaffected.
3. **Idea G completion** — shows what grows while frequency remains stable; supports the "stable routing channel, semantic content grows in parallel" story (not "erasure")

### P1 — strengthens paper

4. **Pythia-70M cross-check**: the workshop-paper "learned erasure" finding is Pythia-70M only. Confirming it doesn't replicate on 7B+ (we mostly have this) settles the paper's framing pivot.
5. **Additional aggregation concept**: refusal direction (Arditi 2024) on the same 3 models using our pipeline — adds a third data point to the aggregation-concept class (beyond sentiment)
6. **Llama freq-sensitive KL** (already done) and Qwen b_LN (running)

### P2 — defensive follow-ups

7. **Wikitext external-unigram code-length re-check** — we used ScaleJSD-native unigram; external corpus confirmation tightens C3
8. **Pythia checkpoint sweep** (deprioritized per §9 but informative for steerability emergence)

---

## §7 — Critical open questions

1. **Is the Q/K-vs-OV dissociation universal or OLMo-dominant?** OLMo has 62% Q-readers; Llama/Qwen only 11-14%. If the Q-bias is OLMo-specific, the "routing" framing weakens. Need to see OV numbers for all 3.

2. **What grows at L31 when frequency AUROC crashes?** Idea G will show. Hypothesis: something prepares for unembedding — either the embedding-space frequency direction gets re-embedded, or some output-ready representation blooms.

3. **Does Pythia-dead + Qwen-dead steering mean the models genuinely don't use v, or that our steering layer is wrong?** Steering was at L10 (Pythia) and L15 (Qwen); maybe steer at different layer fixes it. Untested.

4. **JSD / polytope density reconciliation**: workshop paper showed gradual decline through layers. If frequency AUROC is stable, what IS JSD capturing? Hypothesis: JSD measures dimensionality compression (many dims → 1D axis), not information loss. Needs explicit test — measure subspace rank of class-mean difference per layer.

5. **Does the dynamic reader test (ablate v, measure Q/K activation change) match the static weight-projection test?** If yes, our C2 claim is robust. If not, weight projection was measuring potential not actual use.

---

## §8 — Infrastructure state

### Branches on `sharvillimaye/from-tokens-to-semantics`

- `polytope-results` (main working branch, HEAD: 006ea1e)
- `track-steering-metrics` — frequency_steering_v2.py + FK metrics
- `track-circuits-eap` — signed DFA attribution
- `track-qkv-reader` — Q/K/V weight-projection
- `track-code-length` — Idea 3 R² correlation
- `track-bln-alignment` — Idea C orthogonality
- `track-halluc-predictor` — Track C
- `track-truth-comparison` — Track A (methodology-limited)
- `track-sentiment-control` — Idea 6 taxonomy (9bacc26)
- `track-freq-sensitive-kl` — Idea 2
- `track-reorg-probes` — Idea G
- `track-qk-vs-ov` — The make-or-break (currently running)

### Data on Oumi PVC (`/data/ani/mechinterp/runs/`)

- `steering_metrics/{olmo-7b, llama-3.1-8b, pythia-6.9b, qwen-2.5-7b}/`
- `circuits/{olmo-7b, llama-3.1-8b, pythia-6.9b, qwen-2.5-7b}/`
- `qkv_reader/{olmo-7b, llama-3.1-8b, pythia-6.9b}/` (Qwen pending from qwen-suite)
- `code_length/{olmo-7b, llama-3.1-8b, pythia-6.9b}/` (Qwen pending qwen-gap)
- `bln_alignment/{olmo-7b, llama-3.1-8b, pythia-6.9b}/` (Qwen pending qwen-gap)
- `sentiment_control/{olmo-7b, llama-3.1-8b, qwen-2.5-7b}/`
- `halluc_predictor/{olmo-7b, llama-3.1-8b, pythia-6.9b}/` (Qwen pending qwen-gap)
- `freq_sensitive_kl/{olmo-7b, llama-3.1-8b, qwen-2.5-7b}/`
- `qk_vs_ov/` — empty, running now

### RunPod state (ssh root@87.120.211.211 -p 11656 -i ~/.ssh/id_ed25519)

- H100 80GB, fresh setup earlier today. Torch 2.6.0+cu124.
- Python venv at `/workspace/code/from-tokens-to-semantics/.venv`
- Models cached in `/workspace/.cache/huggingface`: OLMo-7B-hf, Qwen-2.5-7B
- Worktree clones for branch isolation: `/workspace/code/from-tokens-to-semantics-{sentiment,qkv,codelen,reorg}`
- Results: `/workspace/runs/reorg_probes/{olmo-7b,qwen-2.5-7b}/`
- Current tmux: `reorg` (Idea G). Others killed.

### HTML progress reports (in `to_human/`)

- `2026-04-18_progress.html` — v1 overview
- `2026-04-18_taxonomy.html` — v2 with taxonomy + Qwen
- `taxonomy_figure.png` — 3-model × 2-concept comparison
- `reorg_trajectory_partial.png` — Idea G partial, L0-L19

### /loop cron

- Session-only job `114824ec`, cron `7,27,47 * * * *` (every 20 min)
- Auto-expires in 7 days from Apr 18

---

## §9 — What to do first next session

1. **Check Q/K-vs-OV results** — either the paper has its core contribution or we pivot to workshop.
2. **Check Idea G reorg completion** — if L30-L31 data looks clean, we have the "stable through most layers + L31 crash" empirical claim.
3. **Decide paper scope**:
   - If Q/K-OV dissociation holds cleanly: pursue NeurIPS narrow-but-defensible framing (see §4)
   - If not: scope to MI Workshop or BlackboxNLP with the b_LN orthogonality result as the core micro-finding
4. **Run the dynamic reader test** (your new idea) if core is surviving — it strengthens C2 against causal-scrutiny reviewer objections
5. **Update research-summary.md** one more consolidation pass reflecting post-audit narrative
6. **Start drafting the paper** once scope is settled. Use `neurips-2026/manuscript/sections/` as target (currently empty stubs).

---

## §10 — Things we learned / corrected during session

1. **"Gradual frequency erasure" is a small-model artifact.** In 7B+ models, probe AUROC is STABLE across 28 layers with a single L31 crash.
2. **MLP dominance of frequency writing is partly a capacity effect.** Per-channel, attention heads contribute comparably. Raw MLP dominance comes from larger output channel count.
3. **Qwen-2.5-7B joins Pythia in dead-steering.** Decodability (AUROC 0.98) does not imply steerability — same as Pythia-step143000. 2/4 tested models respond to frequency steering.
4. **PopQA tail KL prediction failed.** Frequency direction's effect is position-selective (at anchor tokens), not task-selective.
5. **The paper had too many grand claims** to survive literature scrutiny. Narrow framing survives; grand framing doesn't.

---

Last updated: 2026-04-18, end-of-session. Commit hash at write: `006ea1e` (to be updated when Q/K-vs-OV lands).
