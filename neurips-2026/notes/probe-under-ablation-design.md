# Probe Under Ablation: Design Document

## Goal

Test whether the ablation dissociation (affinity↓ JSD, coverage↑ JSD) survives when measured with a normalization-free metric. Specifically: train linear frequency probes on **residual stream** activations extracted while ablation hooks are active.

This closes the loop: correlation (Section 4) → probes (Step 3) → ablation (`scripts/interventions/targeted_ablation.py`) → **probes under ablation** (this experiment).

---

## Why This Experiment Matters

The current ablation measures JSD, which uses L1 normalization (`scripts/metrics/coverage_affinity_experiment.py`). When you ablate high-coverage neurons:
- Their 4H dimensions go to zero
- L1 normalization redistributes probability mass to remaining neurons
- Remaining specialized neurons get amplified → JSD increases

This **normalization confound** means the coverage/mass JSD increase might be partially mechanical. A linear probe on raw residual stream activations (no normalization) is the clean test.

**Prediction if the dissociation is real:**
- Affinity ablation → frequency probe AUROC drops (frequency info removed from residual stream)
- Coverage ablation → frequency probe AUROC stays flat or drops slightly (not increases)
- Random → no change

**Prediction if coverage effect was normalization artifact:**
- Coverage ablation → frequency probe AUROC also drops (removing any neurons hurts)
- Only the *magnitude* differs (affinity drops more than coverage)

---

## Architecture: What Lives Where

```
Input tokens
    ↓
[Embedding]
    ↓
Layer 0: Attention → MLP → Residual Add → layers_output:0  (H-dim)
    ↓                  ↑
                    post_act (4H-dim) ← THIS is where ablation hooks zero neurons
                       ↓
                    down_proj → (H-dim contribution to residual)
    ↓
Layer 1: ...
    ↓
...
    ↓
Layer N: ... → layers_output:N  (H-dim) ← THIS is what probes read
    ↓
[LM Head] → logits
```

Key distinction:
- **Ablation** operates on the **4H-dim post-activation intermediate** (input to down_proj)
- **Probes** operate on the **H-dim residual stream** (output of the full layer)
- These are different vector spaces. The down_proj maps 4H → H and adds to the residual.

When you zero a neuron in the 4H space, that neuron's column in down_proj contributes nothing to the H-dim residual stream. The residual stream changes, and all downstream layers see the changed input.

---

## Step-by-Step Implementation

### Step 1: Load model and data

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "EleutherAI/pythia-70m-deduped"
revision = "step143000"

tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, torch_dtype=torch.float16).to("cuda").eval()

# Load synonym pairs (reuse from scripts/metrics/coverage_affinity_experiment.py)
from scripts.metrics.coverage_affinity_experiment import load_synonym_pairs, pairs_to_samples
pairs = load_synonym_pairs("path/to/emotion_ngrams_dedup_filtered.jsonl")
samples = pairs_to_samples(pairs, tokenizer)
# Each sample has: token_ids, anchor (position of target word), frequency_category ("high_freq"/"low_freq"), pair_id
```

### Step 2: Build residual stream capture hooks

The ablation script captures the **4H post-activation** (MLP intermediate). For probes you need the **H-dim layer output** (residual stream after the full transformer block).

```python
# For Pythia (GPT-NeoX): model.gpt_neox.layers[L] is the full transformer block
# Its forward output IS the residual stream (H-dim)

class ResidualStreamCapture:
    """Capture H-dim residual stream at each layer's output."""

    def __init__(self, model, layers):
        self.layers = layers
        self.hooks = []
        self.outputs = {}  # layer -> tensor

    def register(self):
        model_layers = model.gpt_neox.layers  # or model.model.layers for LLaMA/OLMo
        for L in self.layers:
            def make_hook(layer_idx):
                def hook_fn(_mod, _inp, output):
                    # output is (hidden_states,) or hidden_states
                    h = output[0] if isinstance(output, tuple) else output
                    self.outputs[layer_idx] = h.detach().cpu()
                return hook_fn
            self.hooks.append(model_layers[L].register_forward_hook(make_hook(L)))

    def clear(self):
        self.outputs.clear()

    def remove(self):
        for h in self.hooks:
            h.remove()
```

**Important**: This captures the full layer output, which includes attention + MLP + residual connections. When you ablate MLP neurons, the residual stream changes because the MLP's contribution changes.

### Step 3: Build ablation hooks (reuse from scripts/interventions/targeted_ablation.py)

```python
from scripts.interventions.targeted_ablation import AblationHook, select_neurons
import pandas as pd

neuron_df = pd.read_csv("path/to/neuron_metrics.csv")
neuron_df = neuron_df[neuron_df["checkpoint"] == "step143000"]

# Select neurons for each condition
ablate_layers = [4, 5]  # last third of Pythia-70M
conditions = {}

conditions["clean"] = {}  # no ablation
conditions["affinity_5pct"] = {
    L: select_neurons(neuron_df, L, pct=5, strategy="affinity")
    for L in ablate_layers
}
conditions["coverage_5pct"] = {
    L: select_neurons(neuron_df, L, pct=5, strategy="coverage")
    for L in ablate_layers
}
conditions["random_5pct"] = {
    L: select_neurons(neuron_df, L, pct=5, strategy="random", seed=42)
    for L in ablate_layers
}
```

### Step 4: Extract residual stream under each condition

For each condition, run all samples through the model with ablation + capture hooks active:

```python
all_layers = list(range(6))  # all Pythia-70M layers

for cond_name, neurons_to_ablate in conditions.items():
    capture = ResidualStreamCapture(model, all_layers)
    capture.register()

    ablation = None
    if neurons_to_ablate:
        ablation = AblationHook(neurons_to_ablate)
        ablation.register(model)

    layer_activations = {L: [] for L in all_layers}  # L -> list of [H] vectors

    for sample in samples:
        capture.clear()
        with torch.no_grad():
            inputs = torch.tensor([sample["token_ids"]], device="cuda")
            model(input_ids=inputs)

        pos = sample["anchor"]
        for L in all_layers:
            h = capture.outputs[L]  # [1, seq_len, H]
            pos_clamped = min(pos, h.shape[1] - 1)
            layer_activations[L].append(h[0, pos_clamped, :].float().numpy())

    capture.remove()
    if ablation:
        ablation.remove()

    # Stack into [N_samples, H] arrays
    layer_activations = {L: np.stack(vecs) for L, vecs in layer_activations.items()}

    # Save for probing
    # ...
```

**Critical detail**: The capture hooks and ablation hooks coexist on the same model. The ablation pre-hook fires first (zeros neurons in 4H space before down_proj), then the layer completes its forward pass, then the capture hook fires (captures the resulting H-dim residual stream). Order is guaranteed because the ablation is on `down_proj` (inner module) and capture is on the full layer (outer module).

### Step 5: Train frequency probe per layer

For each condition × layer, train a simple logistic regression:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

labels = np.array([1 if s["frequency_category"] == "high_freq" else 0 for s in samples])
group_ids = np.array([s["pair_id"] for s in samples])

# Group-aware split: synonym pairs stay together
gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
train_idx, test_idx = next(gss.split(labels, labels, groups=group_ids))

for L in all_layers:
    X = layer_activations[L]  # [N, H] — raw, no normalization

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]

    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(X_train, y_train)

    y_prob = clf.predict_proba(X_test)[:, 1]
    auroc = roc_auc_score(y_test, y_prob)
    accuracy = clf.score(X_test, y_test)

    print(f"  {cond_name} L{L}: AUROC={auroc:.4f}, acc={accuracy:.4f}")
```

### Step 6: Compute raw distance metrics (normalization-free)

These directly address the L1 confound:

```python
from scipy.spatial.distance import cosine as cosine_dist

for L in all_layers:
    X = layer_activations[L]
    high_mask = labels == 1

    mean_high = X[high_mask].mean(axis=0)
    mean_low = X[~high_mask].mean(axis=0)

    cos_dist = cosine_dist(mean_high, mean_low)
    euc_dist = np.linalg.norm(mean_high - mean_low)

    print(f"  {cond_name} L{L}: cosine_dist={cos_dist:.6f}, euclidean={euc_dist:.4f}")
```

**Interpretation:**
- If affinity ablation reduces cosine distance → frequency signal genuinely removed from residual stream
- If coverage ablation increases cosine distance → the dissociation is real (not normalization artifact)
- If coverage ablation decreases or doesn't change cosine distance → the JSD increase was an artifact

---

## Expected Output

CSV with columns:
```
condition, layer, frequency_auroc, frequency_accuracy, cosine_dist_high_low, euclidean_dist_high_low
```

Example rows:
```
clean,         0, 0.865, 0.789, 0.00142, 1.234
clean,         5, 0.566, 0.556, 0.00031, 0.456
affinity_5pct, 0, 0.865, 0.789, 0.00142, 1.234  # L0 unaffected (ablation is in L4-5)
affinity_5pct, 5, 0.520, 0.510, 0.00018, 0.301  # ← key: does AUROC drop?
coverage_5pct, 5, 0.570, 0.558, 0.00033, 0.462  # ← key: does AUROC stay similar to clean?
random_5pct,   5, 0.563, 0.554, 0.00030, 0.451  # ← should be near clean
```

---

## What to Run

Models (same as targeted ablation):
- Pythia-70M (step143000) — ablate L4,L5, probe all 6 layers
- OLMo-1B — ablate L11-15, probe all 16 layers
- Pythia-6.9B (step143000) — ablate L22-31, probe all 32 layers
- OLMo-7B — ablate L22-31, probe all 32 layers
- Llama 3.1 8B — ablate L22-31, probe all 32 layers

Conditions per model:
- clean (no ablation)
- affinity_5pct
- coverage_5pct
- mass_5pct
- random_5pct (average over 5 seeds)

Datasets: all 5 (emotion, medical, legal, scientific, verb) — or just emotion + scientific for speed, since those showed the clearest patterns.

---

## Hook Interaction Diagram

```
Forward pass for Layer L (when L is an ablated layer):

  residual_stream_in (H-dim)
       ↓
  [Attention] → attn_output
       ↓
  residual + attn_output
       ↓
  [LayerNorm]
       ↓
  [MLP up_proj] → 4H-dim
       ↓
  [Activation fn (GELU/SiLU)]
       ↓
  post_activation (4H-dim)
       ↓
  ★ ABLATION PRE-HOOK: zero selected dimensions ★
       ↓
  [MLP down_proj] → H-dim (with zeroed neurons contributing nothing)
       ↓
  residual + mlp_output
       ↓
  ★ CAPTURE HOOK: save this H-dim vector ★
       ↓
  → next layer
```

For non-ablated layers, only the capture hook fires. The ablation in layer L affects all downstream layers because the residual stream carries the change forward.

---

## Key Predictions

| Metric | Affinity ablation | Coverage ablation | Random |
|--------|------------------|-------------------|--------|
| Frequency AUROC | **Drops** | Flat or slight drop | Flat |
| Cosine distance (high vs low) | **Decreases** | Flat or slight decrease | Flat |
| Euclidean distance | **Decreases** | Flat or slight decrease | Flat |

If coverage ablation **increases** AUROC or cosine distance, that would be surprising and would strengthen the dissociation claim beyond the JSD result.

If coverage ablation **decreases** AUROC (same direction as affinity, just weaker), then the JSD increase was likely a normalization artifact, and the real story is: "all ablation reduces frequency info, affinity just does it more efficiently."

Either outcome is informative for the paper.

---

## Relation to Existing Code

| Component | Existing code | Reuse? |
|-----------|--------------|--------|
| Model loading | `scripts/metrics/coverage_affinity_experiment.py` | Yes |
| Synonym pair loading | `scripts/metrics/coverage_affinity_experiment.py:load_synonym_pairs, pairs_to_samples` | Yes |
| Neuron selection | `scripts/interventions/targeted_ablation.py:select_neurons` | Yes |
| Ablation hooks | `scripts/interventions/targeted_ablation.py:AblationHook` | Yes |
| Residual stream capture | None (existing captures 4H, not H) | **New** |
| Linear probe training | None locally (linear_probes repo uses nnterp) | **New** (sklearn) |
| Distance metrics | None | **New** |

The new code is ~100 lines: ResidualStreamCapture class + sklearn probe + distance metrics + CLI glue.
