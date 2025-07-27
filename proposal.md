# Evolution of Polysemanticity in Large Language Models

Understanding how LLMs encode information is key to interpretability. In practice, individual neurons often respond to several unrelated concepts (the phenomenon of **polysemanticity**), making it hard to assign a single meaning to them. Elhage *et al.* (2022) show this naturally arises from **superposition**: a network must cram more features into fewer dimensions than available, causing neurons to “pack” multiple sparse features together. Black *et al.* (2022) argue that instead of neurons, the fundamental units may be the **piecewise-linear activation polytopes** themselves, which can correspond to monosemantic regions of activation space. Recent surveys likewise note that polysemantic neurons undermine the hypothesis that each neuron cleanly encodes one feature. However, past work has mostly provided *static snapshots* of these phenomena.

## Related Work

Several threads of interpretability research relate to our goal of characterizing and eventually untangling polysemantic structure. **Sparse Autoencoders (SAEs)** have become a standard tool: training a wider sparse layer to reconstruct a model’s activations yields “dictionary” features that are often much easier to interpret than individual neurons. Bricken *et al.* (2023) and Cunningham *et al.* (2024) demonstrate that SAEs on transformer layers produce monosemantic features in practice. However, these approaches only analyze a *fixed* model: they do not explain *when* and *how* those interpretable features emerge over training.

Other work targets disentangling polysemantic neurons. O’Mahony *et al.* (2023) introduce **concept vectors** to split a neuron’s behavior into distinct linear directions, showing that polysemantic neurons can indeed be decomposed into multiple human-understandable feature vectors. Dreyer *et al.* (2024) go further by identifying the sub-circuits (“relevant graph”) that implement each feature, effectively turning one polysemantic neuron into several “virtual” monosemantic neurons. Meanwhile, Black *et al.* (2022) propose the **polytope lens**: examining the set of linear regions (polytopes) induced by ReLU networks reveals that entire polytopes can correspond to coherent concepts even when individual directions do not. These geometric insights suggest that one can quantify polysemanticity by measuring the “size” or complexity of a neuron’s activation polytope.

Another related line considers data frequency. Merullo *et al.* (2025) find that the emergence of linear fact representations in LMs correlates with how often related terms co-occur in pretraining data. In particular, certain subject–object relations become linearly encoded only after each term appears thousands of times. This hints that **cumulative token exposure** might also govern how representations specialize or remain entangled. Indeed, frequency is known to affect feature formation: common tokens often have more robust, linear features.

## Identified Gap

Despite these advances, **no prior work explicitly tracks the *temporal dynamics*** of polysemantic neurons during training. We do not know, for example:

- *When* during pretraining neurons transition from generic to polysemantic or to a stable monosemantic feature.
- *How long* neurons remain in a mixed (polysemantic) state before specializing.
- *Whether these dynamics* are consistent across different models, layers, and scales (e.g., Pythia vs. OLMo-2 models).
- *How* the geometry of a neuron’s activation space (its polytope) evolves as its exposure grows.

Without this temporal perspective, efforts to interpret or control models are limited to static analysis of a fully trained network. Our proposal fills this gap by **dynamically tracking polysemanticity** from training start to finish, with an explicit focus on how cumulative n-gram frequency drives specialization.

## Contributions

Our project pursues four main goals:

- **Timing of Superposition:** Identify *when* and under what conditions neurons stop being “generic” and become reliably monosemantic or polysemantic.
- **Frequency–Geometry Link:** Establish a formal connection between a neuron’s cumulative n-gram exposure and the evolution of its activation geometry, revealing critical frequency thresholds where specialization occurs.
- **Generalization Across Models:** Verify that these patterns hold in multiple architectures and sizes, by comparing the Pythia family (70M–2.8B params) and OLMo-2 (1B & 7B) models.
- **Quantitative Polysemanticity Measure:** Introduce a geometric framework (polytope analysis) as a continuous metric of polysemanticity, complementing clustering-based approaches.

Collectively, these deliver a **temporal, geometric, and data-driven account of polysemanticity** in LLMs, suitable for mechanistic interpretability at a conference venue.

## Methodology

Our pipeline proceeds checkpoint-by-checkpoint in two model families: the Pythia suite (open checkpoints at 10% increments, trained on The Pile) and OLMo-2 (1B and 7B, trained on the 3T-token Dolma corpus). All pre-training data are open-source.

At each checkpoint **t**, we perform:

### Activation Snapshotting

- Run 1 million held-out tokens (never seen in training) through the model.
- Extract hidden states from four representative layers (25%, 50%, 75%, and final).
- Yields activation vectors per neuron.

### Cumulative n-gram Counting

- Use *Infini-gram* to compute n-gram frequency up to checkpoint *t*.
- Identify top-k triggers per neuron (activations > 1 stddev above mean).
- Exposure for neuron *j* is:
  

- Also sample n-grams across frequency bins (1–1k, 1k–100k, >100k) for polytope analysis.

### Neuron Embedding & Clustering

- Compute neuron embeddings via input-weight × activation.
- Cluster embeddings (e.g., HDBSCAN or k-means).
- Label neuron as **polysemantic** if multiple stable clusters; **monosemantic** otherwise.
- Use metrics like number of clusters, silhouette score, dispersion.

### Linking Frequency to Polysemanticity

Fit logistic regression or GAM:


- Validate via held-out predictive AUC.

### Geometric Analysis (Activation Polytopes)

- Use high-activation points as vertices of a convex polytope.
- Compute:
- Volume (in PCA subspace)
- Vertex count
- Expect volume/complexity to **decrease** as neurons specialize.

### Causal Intervention (Optional)

- Identify neurons near critical threshold.
- Retrain copy of model with frequent triggers downsampled or removed.
- Track whether neurons stay polysemantic longer than control.

### Staging

- **Stage 1:** Run pilot on Pythia-70M to validate pipeline.
- **Stage 2:** Scale to all Pythia + OLMo-2 checkpoints.

## Expected Outcomes

- Neurons consolidate into dominant features as exposure grows.
- Polysemanticity (via clustering and polytope volume) **declines** past a critical frequency.
- Logistic model identifies consistent thresholds (\~10³ counts).
- Patterns **replicate across models**, confirming generality.
- Causal test: downsampling n-grams **delays** specialization, supporting data-driven emergence.

## Limitations

- **Compute/Storage:** Expensive for n-gram counting and activation storage. Mitigated via Infini-gram and layer sampling.
- **Checkpoint Resolution:** 10% increments may miss transitions. Will add finer checkpoints if needed.
- **Clustering Noise:** Polysemanticity labels rely on clustering; unstable for noisy points. Compare clustering vs. geometric polytope metrics.
- **Causality Confounds:** Frequency might not be the *only* driver. Interpret perturbation results with care.

## References

Our approach builds directly on studies of superposition and polysemanticity, sparse autoencoders, concept vectors, and data frequency effects. All code, checkpoints, and data used will be public, following open science standards in mechanistic interpretability.

