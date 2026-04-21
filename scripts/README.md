# Scripts

The repository keeps runnable entry points under `scripts/`, grouped by purpose.

- `metrics/`: metric extraction and large-scale computation pipelines
- `interventions/`: ablations and causal intervention studies
- `plots/`: figure generation and downstream analysis helpers
- `cluster/`: lightweight cluster-side helpers

Recommended invocation from the repo root:

```bash
python -m scripts.metrics.coverage_affinity_experiment --help
python -m scripts.interventions.targeted_ablation --help
python -m scripts.plots.paper_figures --help
```
