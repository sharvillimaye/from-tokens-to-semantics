#!/usr/bin/env python3
"""
Probe under ablation: train linear probes on residual stream activations
extracted while ablation hooks are active.

Closes the loop: correlation (Section 4) -> probes (Step 3) -> ablation
(targeted_ablation.py) -> probes under ablation (this experiment).

The existing ablation script measures JSD on 4H post-activation vectors,
which are subject to L1 normalization confounds. This experiment instead
trains frequency and semantic probes on the H-dim residual stream (layer
output) — a normalization-free measure of what information survives ablation.

Conditions per model:
  - clean (no ablation)
  - affinity_5pct, affinity_10pct
  - coverage_5pct, coverage_10pct
  - mass_5pct, mass_10pct
  - random_5pct, random_10pct (averaged over multiple seeds)

Probes:
  - frequency: binary logistic regression (high_freq vs low_freq)
  - semantic:  multiclass logistic regression (per dataset category)

Metrics:
  - Probe AUROC and accuracy
  - Cosine distance between mean high-freq and low-freq residual vectors
  - Euclidean distance between mean high-freq and low-freq residual vectors

Usage:
    python -m scripts.interventions.probe_under_ablation \
        --model EleutherAI/pythia-70m-deduped \
        --revision step143000 \
        --all-datasets \
        --dataset-dir ~/scaleJSD/dataset/legacy/filtered \
        --metrics-dir results/coverage_affinity/pythia-70m \
        --output-dir results/probe_under_ablation/pythia-70m \
        --ablate-layers 4,5 \
        --ablate-pct 5,10
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine as cosine_dist
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
import torch

try:
    from scripts.interventions.targeted_ablation import (
        AblationHook,
        select_neurons,
    )
    from scripts.metrics.coverage_affinity_experiment import (
        _infer_layers,
        load_synonym_pairs,
        pairs_to_samples,
    )
except ImportError:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from scripts.interventions.targeted_ablation import (
        AblationHook,
        select_neurons,
    )
    from scripts.metrics.coverage_affinity_experiment import (
        _infer_layers,
        load_synonym_pairs,
        pairs_to_samples,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Residual stream capture (H-dim layer output, NOT the 4H post-activation)
# ─────────────────────────────────────────────────────────────────────────────


class ResidualStreamCapture:
    """Capture H-dim residual stream at each transformer layer's output."""

    def __init__(self, model: torch.nn.Module, layers: List[int]):
        self.model = model
        self.layers = layers
        self.hooks: list = []
        self.outputs: Dict[int, torch.Tensor] = {}

    def _find_model_layers(self):
        for attr in [
            "gpt_neox.layers",
            "model.layers",
            "transformer.h",
            "transformer.layers",
            "model.decoder.layers",
        ]:
            obj = self.model
            try:
                for part in attr.split("."):
                    obj = getattr(obj, part)
                return obj
            except AttributeError:
                continue
        raise RuntimeError("Cannot find transformer layers in model")

    def register(self):
        model_layers = self._find_model_layers()
        for L in self.layers:

            def make_hook(layer_idx):
                def hook_fn(_mod, _inp, output):
                    h = output[0] if isinstance(output, tuple) else output
                    self.outputs[layer_idx] = h.detach().cpu()

                return hook_fn

            self.hooks.append(model_layers[L].register_forward_hook(make_hook(L)))

    def clear(self):
        self.outputs.clear()

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


# ─────────────────────────────────────────────────────────────────────────────
# Extract residual stream under a given ablation condition
# ─────────────────────────────────────────────────────────────────────────────


def extract_residual_stream(
    model: torch.nn.Module,
    samples: List[Dict],
    layers: List[int],
    device: str,
    neurons_to_ablate: Optional[Dict[int, np.ndarray]] = None,
) -> Dict[int, np.ndarray]:
    """Run forward pass with optional ablation, return H-dim residual stream per layer.

    Returns: dict[layer] -> np.ndarray [N_samples, H]
    """
    capture = ResidualStreamCapture(model, layers)
    capture.register()

    ablation = None
    if neurons_to_ablate:
        ablation = AblationHook(neurons_to_ablate)
        ablation.register(model)

    activations: Dict[int, list] = {L: [] for L in layers}

    for idx, sample in enumerate(samples):
        if idx % 100 == 0:
            print(f"    [{idx + 1}/{len(samples)}]")

        capture.clear()
        with torch.no_grad():
            inputs = torch.tensor([sample["token_ids"]], device=device)
            model(input_ids=inputs)

        pos = sample["anchor"]
        for L in layers:
            h = capture.outputs.get(L)
            if h is None:
                raise RuntimeError(f"No residual stream captured for layer {L}")
            pos_clamped = min(pos, h.shape[1] - 1)
            activations[L].append(h[0, pos_clamped, :].float().numpy())

    capture.remove()
    if ablation:
        ablation.remove()

    return {L: np.stack(vecs) for L, vecs in activations.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Probe training
# ─────────────────────────────────────────────────────────────────────────────


def train_frequency_probe(
    X: np.ndarray,
    labels: np.ndarray,
    group_ids: np.ndarray,
    n_splits: int = 5,
) -> Dict[str, float]:
    """Train logistic regression frequency probe with group-aware CV.

    Returns mean AUROC and accuracy across splits.
    """
    gss = GroupShuffleSplit(n_splits=n_splits, test_size=0.3, random_state=42)
    aurocs, accs = [], []

    for train_idx, test_idx in gss.split(X, labels, groups=group_ids):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
        clf.fit(X_train, y_train)

        y_prob = clf.predict_proba(X_test)[:, 1]
        aurocs.append(roc_auc_score(y_test, y_prob))
        accs.append(clf.score(X_test, y_test))

    return {
        "frequency_auroc": float(np.mean(aurocs)),
        "frequency_auroc_std": float(np.std(aurocs)),
        "frequency_accuracy": float(np.mean(accs)),
    }


def train_semantic_probe(
    X: np.ndarray,
    category_labels: np.ndarray,
    group_ids: np.ndarray,
    n_splits: int = 5,
) -> Dict[str, float]:
    """Train multiclass logistic regression semantic probe.

    Returns mean AUROC (one-vs-rest) and accuracy across splits.
    """
    le = LabelEncoder()
    y = le.fit_transform(category_labels)
    n_classes = len(le.classes_)

    if n_classes < 2:
        return {
            "semantic_auroc": float("nan"),
            "semantic_auroc_std": float("nan"),
            "semantic_accuracy": float("nan"),
        }

    gss = GroupShuffleSplit(n_splits=n_splits, test_size=0.3, random_state=42)
    aurocs, accs = [], []

    for train_idx, test_idx in gss.split(X, y, groups=group_ids):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Skip splits where test set is missing a class
        if len(np.unique(y_test)) < n_classes:
            continue

        clf = LogisticRegression(
            max_iter=1000,
            C=1.0,
            solver="lbfgs",
            multi_class="multinomial",
        )
        clf.fit(X_train, y_train)

        y_prob = clf.predict_proba(X_test)
        auroc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")
        aurocs.append(auroc)
        accs.append(clf.score(X_test, y_test))

    if not aurocs:
        return {
            "semantic_auroc": float("nan"),
            "semantic_auroc_std": float("nan"),
            "semantic_accuracy": float("nan"),
        }

    return {
        "semantic_auroc": float(np.mean(aurocs)),
        "semantic_auroc_std": float(np.std(aurocs)),
        "semantic_accuracy": float(np.mean(accs)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Distance metrics (normalization-free)
# ─────────────────────────────────────────────────────────────────────────────


def compute_distance_metrics(
    X: np.ndarray,
    freq_labels: np.ndarray,
) -> Dict[str, float]:
    """Cosine and Euclidean distance between mean high-freq and low-freq vectors."""
    high_mask = freq_labels == 1
    if high_mask.sum() == 0 or (~high_mask).sum() == 0:
        return {"cosine_dist": float("nan"), "euclidean_dist": float("nan")}

    mean_high = X[high_mask].mean(axis=0)
    mean_low = X[~high_mask].mean(axis=0)

    return {
        "cosine_dist": float(cosine_dist(mean_high, mean_low)),
        "euclidean_dist": float(np.linalg.norm(mean_high - mean_low)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main experiment
# ─────────────────────────────────────────────────────────────────────────────

STRATEGIES = ["affinity", "coverage", "mass", "random"]
DATASET_NAMES = ["emotion", "medical", "legal", "scientific", "verb"]


def build_conditions(
    neuron_df: pd.DataFrame,
    ablate_layers: List[int],
    ablate_pcts: List[float],
    n_random_seeds: int = 5,
) -> List[Tuple[str, Optional[Dict[int, np.ndarray]]]]:
    """Build list of (condition_name, neurons_to_ablate_or_None)."""
    conditions: List[Tuple[str, Optional[Dict[int, np.ndarray]]]] = [("clean", None)]

    for pct in ablate_pcts:
        for strategy in STRATEGIES:
            if strategy == "random":
                for seed in range(n_random_seeds):
                    neurons = {
                        L: select_neurons(neuron_df, L, pct, strategy, seed=seed)
                        for L in ablate_layers
                    }
                    conditions.append((f"random_{int(pct)}pct_seed{seed}", neurons))
            else:
                neurons = {L: select_neurons(neuron_df, L, pct, strategy) for L in ablate_layers}
                conditions.append((f"{strategy}_{int(pct)}pct", neurons))

    return conditions


def run_probes_under_ablation(
    model: torch.nn.Module,
    samples: List[Dict],
    neuron_df: pd.DataFrame,
    all_layers: List[int],
    ablate_layers: List[int],
    ablate_pcts: List[float],
    device: str,
    dataset_name: str,
    n_random_seeds: int = 5,
    n_cv_splits: int = 5,
) -> List[Dict[str, Any]]:
    """Run probe-under-ablation experiment for one dataset."""

    freq_labels = np.array([1 if s["frequency_category"] == "high_freq" else 0 for s in samples])
    group_ids = np.array([s["pair_id"] for s in samples])

    # For semantic probe: use dataset categories from all loaded datasets
    # When running with --all-datasets, category comes from the dataset itself
    # When single dataset, all samples share the same category -> semantic probe is N/A
    category_labels = np.array([s.get("category", dataset_name) for s in samples])

    conditions = build_conditions(
        neuron_df,
        ablate_layers,
        ablate_pcts,
        n_random_seeds,
    )

    results = []

    for cond_name, neurons_to_ablate in conditions:
        n_ablated = sum(len(v) for v in neurons_to_ablate.values()) if neurons_to_ablate else 0
        print(f"\n  Condition: {cond_name} ({n_ablated} neurons ablated)")

        layer_acts = extract_residual_stream(
            model,
            samples,
            all_layers,
            device,
            neurons_to_ablate,
        )

        for L in all_layers:
            X = layer_acts[L]

            # Frequency probe
            freq_metrics = train_frequency_probe(X, freq_labels, group_ids, n_cv_splits)

            # Semantic probe (only meaningful with multi-category data)
            semantic_metrics = train_semantic_probe(
                X,
                category_labels,
                group_ids,
                n_cv_splits,
            )

            # Distance metrics
            dist_metrics = compute_distance_metrics(X, freq_labels)

            row = {
                "condition": cond_name,
                "layer": L,
                "n_ablated": n_ablated,
                "ablate_layers": ",".join(map(str, ablate_layers)),
                "dataset": dataset_name,
                **freq_metrics,
                **semantic_metrics,
                **dist_metrics,
            }
            results.append(row)

            print(
                f"    L{L}: freq_auroc={freq_metrics['frequency_auroc']:.4f} "
                f"sem_auroc={semantic_metrics['semantic_auroc']:.4f} "
                f"cos={dist_metrics['cosine_dist']:.6f}"
            )

    return results


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def _find_dataset(dataset_dir: Path, name: str) -> Optional[Path]:
    for pat in [
        "{name}_ngrams_dedup_filtered.jsonl",
        "{name}_ngrams_filtered_pairs.jsonl",
    ]:
        p = dataset_dir / pat.format(name=name)
        if p.exists():
            return p
    return None


def main():
    p = argparse.ArgumentParser(description="Probe under ablation experiment")

    p.add_argument("--model", default="EleutherAI/pythia-70m-deduped")
    p.add_argument("--revision", default="step143000")
    p.add_argument("--device", default=None)

    # Single dataset mode
    p.add_argument("--dataset", default=None)
    p.add_argument("--neuron-metrics", default=None)

    # All datasets mode
    p.add_argument("--all-datasets", action="store_true")
    p.add_argument("--dataset-dir", default=None)
    p.add_argument("--metrics-dir", default=None)

    p.add_argument("--output-dir", required=True)
    p.add_argument(
        "--ablate-layers",
        default=None,
        help="Comma-separated layers to ablate (default: last third)",
    )
    p.add_argument("--ablate-pct", default="5,10", help="Comma-separated ablation percentages")
    p.add_argument("--n-random-seeds", type=int, default=5)
    p.add_argument("--n-cv-splits", type=int, default=5)

    args = p.parse_args()

    device = args.device or (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    ablate_pcts = [float(x) for x in args.ablate_pct.split(",")]

    # Load model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    load_kw: Dict[str, Any] = {"torch_dtype": torch.float16}
    if args.revision != "main":
        load_kw["revision"] = args.revision

    print(f"Loading {args.model} @ {args.revision} on {device}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        **{k: v for k, v in load_kw.items() if k == "revision"},
    )
    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kw).to(device)
    model.eval()

    all_layers = _infer_layers(args.model)
    n_layers = len(all_layers)

    if args.ablate_layers:
        ablate_layers = [int(x) for x in args.ablate_layers.split(",")]
    else:
        start = n_layers * 2 // 3
        ablate_layers = list(range(start, n_layers))

    print(f"Layers: {n_layers}, ablating in: {ablate_layers}")

    # Resolve datasets
    if args.all_datasets:
        dataset_dir = Path(args.dataset_dir)
        metrics_dir = Path(args.metrics_dir)
        datasets = []
        for name in DATASET_NAMES:
            ds_path = _find_dataset(dataset_dir, name)
            met_path = metrics_dir / name / "neuron_metrics.csv"
            if ds_path and met_path.exists():
                datasets.append((name, str(ds_path), str(met_path)))
            else:
                print(
                    f"SKIP: {name} (dataset={'found' if ds_path else 'missing'}, "
                    f"metrics={'found' if met_path.exists() else 'missing'})"
                )
    elif args.dataset and args.neuron_metrics:
        name = Path(args.dataset).stem.split("_")[0]
        datasets = [(name, args.dataset, args.neuron_metrics)]
    else:
        p.error(
            "Provide --dataset + --neuron-metrics, or --all-datasets + --dataset-dir + --metrics-dir"
        )

    # Run per dataset
    all_results = []

    for name, dataset_path, metrics_path in datasets:
        print(f"\n{'=' * 60}")
        print(f"  Dataset: {name}")
        print(f"{'=' * 60}")

        pairs = load_synonym_pairs(dataset_path)
        samples = pairs_to_samples(pairs, tokenizer)

        neuron_df = pd.read_csv(metrics_path)
        if "checkpoint" in neuron_df.columns and args.revision in neuron_df["checkpoint"].values:
            neuron_df = neuron_df[neuron_df["checkpoint"] == args.revision]
        print(f"  Neuron metrics: {len(neuron_df)} rows")

        results = run_probes_under_ablation(
            model=model,
            samples=samples,
            neuron_df=neuron_df,
            all_layers=all_layers,
            ablate_layers=ablate_layers,
            ablate_pcts=ablate_pcts,
            device=device,
            dataset_name=name,
            n_random_seeds=args.n_random_seeds,
            n_cv_splits=args.n_cv_splits,
        )

        all_results.extend(results)

    # Save
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(all_results)
    df.to_csv(out / "probe_under_ablation.csv", index=False)
    print(f"\nSaved {len(df)} rows to {out / 'probe_under_ablation.csv'}")

    # Summary: compare conditions at ablated layers
    if len(df) > 0:
        # Normalize random seeds: average random conditions
        def normalize_condition(c):
            if c.startswith("random_") and "_seed" in c:
                return c.rsplit("_seed", 1)[0]
            return c

        df["condition_group"] = df["condition"].apply(normalize_condition)

        abl_df = df[df["layer"].isin(ablate_layers)]
        summary = (
            abl_df.groupby("condition_group")
            .agg(
                {
                    "frequency_auroc": "mean",
                    "semantic_auroc": "mean",
                    "cosine_dist": "mean",
                    "euclidean_dist": "mean",
                }
            )
            .reset_index()
        )

        print(f"\n{'=' * 60}")
        print(f"  SUMMARY (ablated layers only, all datasets)")
        print(f"{'=' * 60}")
        print(
            f"  {'Condition':<22} {'FreqAUROC':>10} {'SemAUROC':>10} "
            f"{'CosDist':>10} {'EucDist':>10}"
        )
        print(f"  {'-' * 66}")
        for _, row in summary.iterrows():
            print(
                f"  {row['condition_group']:<22} "
                f"{row['frequency_auroc']:>10.4f} "
                f"{row['semantic_auroc']:>10.4f} "
                f"{row['cosine_dist']:>10.6f} "
                f"{row['euclidean_dist']:>10.4f}"
            )

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
