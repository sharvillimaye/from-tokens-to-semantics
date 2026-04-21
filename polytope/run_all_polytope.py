#!/usr/bin/env python3
"""
Run polytope density pipeline across all models and datasets.

Usage:
    python -m polytope.run_all_polytope --code-dir /data/ani/mechinterp/code --output-root /data/ani/mechinterp/runs/polytope_density
    python -m polytope.run_all_polytope --model pythia-70m   # single model, all datasets
    python -m polytope.run_all_polytope --dataset emotion    # single dataset, all models
    python -m polytope.run_all_polytope --list               # show all combos
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

# ---------------------------------------------------------------------------
# Model configs
# ---------------------------------------------------------------------------

MODELS = {
    "pythia-70m": {
        "model_id": "EleutherAI/pythia-70m-deduped",
        "revision": "step143000",
        "num_layers": 6,
    },
    "pythia-1b": {
        "model_id": "EleutherAI/pythia-1b-deduped",
        "revision": "step143000",
        "num_layers": 16,
    },
    "pythia-6.9b": {
        "model_id": "EleutherAI/pythia-6.9b-deduped",
        "revision": "step143000",
        "num_layers": 32,
    },
    "olmo-1b": {
        "model_id": "allenai/OLMo-1B-hf",
        "revision": "main",
        "num_layers": 16,
    },
    "olmo-7b": {
        "model_id": "allenai/OLMo-7B-hf",
        "revision": "main",
        "num_layers": 32,
    },
    "llama-8b": {
        "model_id": "meta-llama/Llama-3.1-8B",
        "revision": "main",
        "num_layers": 32,
    },
}

DATASETS = ["emotion", "medical", "legal", "scientific", "verb"]

# Dataset file naming patterns (tried in order)
DATASET_PATTERNS = [
    "{name}_ngrams_dedup_filtered.jsonl",
    "{name}_ngrams_filtered_pairs.jsonl",
    "{name}_filtered_pairs.jsonl",
    "{name}_filtered.jsonl",
]

# Directories to search for datasets (relative to code_dir)
DATASET_DIRS = [
    "scaleJSD/dataset/filtered",
    "scaleJSD/dataset/consolidated",
    "scaleJSD/dataset",
]


def find_dataset(code_dir: Path, name: str) -> Optional[Path]:
    """Find the dataset JSONL file by searching known directories and patterns."""
    for dir_rel in DATASET_DIRS:
        dataset_dir = code_dir / dir_rel
        if not dataset_dir.exists():
            continue
        for pattern in DATASET_PATTERNS:
            path = dataset_dir / pattern.format(name=name)
            if path.exists():
                return path
    return None


def run_single(
    model_key: str,
    dataset_name: str,
    code_dir: Path,
    output_root: Path,
    save_records: bool = True,
    dry_run: bool = False,
) -> bool:
    """Run the polytope pipeline for one (model, dataset) combo."""
    config = MODELS[model_key]
    dataset_path = find_dataset(code_dir, dataset_name)

    if dataset_path is None:
        print(f"  SKIP: dataset '{dataset_name}' not found in any search path")
        return False

    output_dir = output_root / model_key / dataset_name
    layers = ",".join(str(i) for i in range(config["num_layers"]))

    cmd = [
        sys.executable, "-m", "polytope.polytope_pipeline",
        "--dataset", str(dataset_path),
        "--model", config["model_id"],
        "--revision", config["revision"],
        "--layers", layers,
        "--output-dir", str(output_dir),
    ]
    if save_records:
        cmd.append("--save-records")

    print(f"\n{'='*70}")
    print(f"MODEL: {model_key} ({config['model_id']})")
    print(f"DATASET: {dataset_name} ({dataset_path.name})")
    print(f"OUTPUT: {output_dir}")
    print(f"LAYERS: {config['num_layers']}")
    print(f"{'='*70}")

    if dry_run:
        print(f"  [DRY RUN] {' '.join(cmd)}")
        return True

    try:
        result = subprocess.run(cmd, check=True, cwd=str(code_dir))
        print(f"  OK: {model_key}/{dataset_name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  FAILED: {model_key}/{dataset_name} (exit code {e.returncode})")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run polytope density across all models/datasets")
    parser.add_argument("--code-dir", type=str, default=".",
                        help="Root of the code repo (default: cwd)")
    parser.add_argument("--output-root", type=str, default="runs/polytope_density",
                        help="Root output directory")
    parser.add_argument("--model", type=str, default=None,
                        help="Run only this model (e.g. pythia-70m)")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Run only this dataset (e.g. emotion)")
    parser.add_argument("--no-save-records", action="store_true",
                        help="Don't save intermediate activation records")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without running them")
    parser.add_argument("--list", action="store_true",
                        help="List all model/dataset combos and exit")
    args = parser.parse_args()

    code_dir = Path(args.code_dir).resolve()
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = code_dir / output_root

    # Filter models/datasets
    models = [args.model] if args.model else list(MODELS.keys())
    datasets = [args.dataset] if args.dataset else DATASETS

    # Validate
    for m in models:
        if m not in MODELS:
            print(f"Unknown model: {m}. Available: {list(MODELS.keys())}")
            sys.exit(1)
    for d in datasets:
        if d not in DATASETS:
            print(f"Unknown dataset: {d}. Available: {DATASETS}")
            sys.exit(1)

    total = len(models) * len(datasets)
    print(f"Polytope Density Pipeline: {len(models)} models x {len(datasets)} datasets = {total} runs")
    print(f"Code dir: {code_dir}")
    print(f"Output root: {output_root}")

    if args.list:
        for m in models:
            for d in datasets:
                cfg = MODELS[m]
                print(f"  {m:15s} x {d:12s}  ({cfg['model_id']}, {cfg['num_layers']} layers)")
        sys.exit(0)

    # Run all combos
    successes = 0
    failures = 0
    for m in models:
        for d in datasets:
            ok = run_single(
                m, d, code_dir, output_root,
                save_records=not args.no_save_records,
                dry_run=args.dry_run,
            )
            if ok:
                successes += 1
            else:
                failures += 1

    print(f"\n{'='*70}")
    print(f"DONE: {successes}/{total} succeeded, {failures} failed")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
