#!/usr/bin/env python3
"""
Direct Pair Polytope Density

Computes polytope boundary density per the methodology of Humayun et al. (2022):

    rho(x_i, x_j) = Hamming(spline_i, spline_j) / ||x_i - x_j||_2

where:
  - spline codes are binarized MLP pre-activations (from up-projection output)
  - Euclidean distance is on the MLP INPUT (residual stream), NOT the layer output
  - No normalization (raw Euclidean distance as per the paper)

The density is computed PAIRWISE between matched synonym pairs (high-freq word
paired with its low-freq synonym via pair_id), then averaged per layer.

Usage:
    python -m polytope.direct_pair_density <pkl_path> [--output-dir results/]
    python -m polytope.direct_pair_density --batch-dir <dir_of_pkls> [--output-dir results/]
"""

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_records(path: str) -> List[Dict[str, Any]]:
    """Load activation records from a pickle file."""
    import pickle
    with open(path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict) and "records" in data:
        return data["records"]
    if isinstance(data, list):
        return data
    raise ValueError(f"Unexpected pkl format in {path}: {type(data)}")


def load_batch_dir(batch_dir: str) -> List[Dict[str, Any]]:
    """Load and concatenate all pkl files in a directory."""
    records = []
    pkl_files = sorted(Path(batch_dir).glob("**/*.pkl"))
    if not pkl_files:
        raise FileNotFoundError(f"No .pkl files found in {batch_dir}")
    for pkl_path in pkl_files:
        print(f"  Loading {pkl_path.name} ...")
        records.extend(load_records(str(pkl_path)))
    print(f"  Total records: {len(records)}")
    return records


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _get_spline_code(record: Dict[str, Any]) -> np.ndarray:
    """Return spline_code if present, else fall back to binary_pattern."""
    sc = record.get("spline_code")
    if sc is not None:
        return np.asarray(sc, dtype=np.int8)
    return np.asarray(record["binary_pattern"], dtype=np.int8)


def _get_mlp_input(record: Dict[str, Any]) -> np.ndarray:
    """Return the MLP input vector (residual stream) for Euclidean distance.

    Falls back to activation_vector if mlp_input_vector is not available
    (e.g. records created before this field was added).
    """
    vec = record.get("mlp_input_vector")
    if vec is not None:
        return np.asarray(vec, dtype=np.float64)
    return np.asarray(record["activation_vector"], dtype=np.float64)


def _hamming(a: np.ndarray, b: np.ndarray) -> int:
    """Hamming distance between two binary vectors."""
    return int(np.sum(a != b))


# ---------------------------------------------------------------------------
# Pair matching
# ---------------------------------------------------------------------------

def _group_by_checkpoint_layer(
    records: List[Dict[str, Any]],
) -> Dict[Tuple[str, int], List[Dict[str, Any]]]:
    groups: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
    for r in records:
        key = (str(r["checkpoint_step"]), int(r["layer"]))
        groups.setdefault(key, []).append(r)
    return groups


def _pair_high_low(
    records: List[Dict[str, Any]],
) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """Match records into (high, low) pairs by pair_id.

    Only returns complete pairs where both high_freq and low_freq exist.
    """
    by_pair: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for r in records:
        pid = r.get("pair_id")
        if pid is None:
            continue
        cat = r.get("category", "")
        by_pair.setdefault(pid, {})[cat] = r

    pairs = []
    for pid, cats in sorted(by_pair.items()):
        if "high_freq" in cats and "low_freq" in cats:
            pairs.append((cats["high_freq"], cats["low_freq"]))
    return pairs


# ---------------------------------------------------------------------------
# Primary metric: Direct Pair Density (per Humayun et al.)
# ---------------------------------------------------------------------------

def compute_direct_pair_density(
    records: List[Dict[str, Any]],
    checkpoint_step: str,
    layer: int,
    eps: float = 1e-12,
) -> Dict[str, Any]:
    """
    For each synonym pair (high, low):
        rho = Hamming(spline_high, spline_low) / ||mlp_input_high - mlp_input_low||_2

    NO z-score normalization. Raw Euclidean distance on the MLP input (residual
    stream), consistent with measuring density in the input space of the
    piecewise-linear MLP.

    Returns per-pair rho values and aggregate stats.
    """
    subset = [
        r for r in records
        if str(r["checkpoint_step"]) == str(checkpoint_step) and int(r["layer"]) == layer
    ]
    pairs = _pair_high_low(subset)
    if not pairs:
        return {"n_pairs": 0, "rho_values": [], "pair_ids": []}

    rho_values = []
    hamming_values = []
    euclidean_values = []
    pair_ids = []

    for h_rec, l_rec in pairs:
        s_h = _get_spline_code(h_rec)
        s_l = _get_spline_code(l_rec)
        x_h = _get_mlp_input(h_rec)
        x_l = _get_mlp_input(l_rec)

        hamming_dist = _hamming(s_h, s_l)
        euclid_dist = float(np.linalg.norm(x_h - x_l))

        hamming_values.append(hamming_dist)
        euclidean_values.append(euclid_dist)

        if euclid_dist > eps:
            rho_values.append(float(hamming_dist / euclid_dist))
        else:
            rho_values.append(float("nan"))

        pair_ids.append(h_rec.get("pair_id", ""))

    arr = np.array(rho_values)
    valid = arr[np.isfinite(arr)]

    return {
        "n_pairs": len(pairs),
        "rho_values": rho_values,
        "pair_ids": pair_ids,
        "hamming_values": [int(h) for h in hamming_values],
        "euclidean_values": euclidean_values,
        "mean": float(np.mean(valid)) if len(valid) > 0 else float("nan"),
        "std": float(np.std(valid)) if len(valid) > 0 else float("nan"),
        "median": float(np.median(valid)) if len(valid) > 0 else float("nan"),
        "hamming_mean": float(np.mean(hamming_values)),
        "euclidean_mean": float(np.mean(euclidean_values)),
    }


# ---------------------------------------------------------------------------
# Normalized Hamming (layer-normalized, no Euclidean)
# ---------------------------------------------------------------------------

def compute_normalized_hamming(
    records: List[Dict[str, Any]],
    checkpoint_step: str,
    layer: int,
) -> Dict[str, Any]:
    """Hamming distance / spline code length for each pair. Useful as a sanity
    check that isn't confounded by activation magnitude."""
    subset = [
        r for r in records
        if str(r["checkpoint_step"]) == str(checkpoint_step) and int(r["layer"]) == layer
    ]
    pairs = _pair_high_low(subset)
    if not pairs:
        return {"n_pairs": 0}

    norm_hammings = []
    for h_rec, l_rec in pairs:
        s_h = _get_spline_code(h_rec)
        s_l = _get_spline_code(l_rec)
        code_len = max(len(s_h), 1)
        norm_hammings.append(_hamming(s_h, s_l) / code_len)

    arr = np.array(norm_hammings)
    return {
        "n_pairs": len(pairs),
        "normalized_hamming_mean": float(np.mean(arr)),
        "normalized_hamming_std": float(np.std(arr)),
        "normalized_hamming_median": float(np.median(arr)),
    }


# ---------------------------------------------------------------------------
# Full analysis across all checkpoints x layers
# ---------------------------------------------------------------------------

def analyze_all(
    records: List[Dict[str, Any]],
    output_dir: Optional[str] = None,
) -> pd.DataFrame:
    """Loop over all (checkpoint, layer) groups and compute pairwise density."""
    groups = _group_by_checkpoint_layer(records)

    rows = []
    for (ckpt, layer), grp in sorted(groups.items()):
        n_high = sum(1 for r in grp if r.get("category") == "high_freq")
        n_low = sum(1 for r in grp if r.get("category") == "low_freq")
        print(f"  checkpoint={ckpt}  layer={layer}  high={n_high} low={n_low}")

        primary = compute_direct_pair_density(records, ckpt, layer)
        norm_h = compute_normalized_hamming(records, ckpt, layer)

        rows.append({
            "checkpoint": ckpt,
            "layer": layer,
            "n_pairs": primary["n_pairs"],
            "density_mean": primary.get("mean", float("nan")),
            "density_std": primary.get("std", float("nan")),
            "density_median": primary.get("median", float("nan")),
            "hamming_mean": primary.get("hamming_mean", float("nan")),
            "euclidean_mean": primary.get("euclidean_mean", float("nan")),
            "normalized_hamming_mean": norm_h.get("normalized_hamming_mean", float("nan")),
            "normalized_hamming_std": norm_h.get("normalized_hamming_std", float("nan")),
        })

    df = pd.DataFrame(rows)

    if output_dir:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        csv_path = out / "direct_pair_density.csv"
        df.to_csv(csv_path, index=False)
        print(f"Saved: {csv_path}")
        json_path = out / "direct_pair_density.json"
        df.to_json(json_path, orient="records", indent=2)
        print(f"Saved: {json_path}")

    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Direct Pair Polytope Density (Humayun et al. methodology)"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pkl", help="Single pickle file with activation records")
    group.add_argument("--batch-dir", help="Directory containing multiple pkl files")
    parser.add_argument("--output-dir", default=None, help="Output directory (default: alongside input)")
    args = parser.parse_args()

    if args.pkl:
        print(f"Loading {args.pkl} ...")
        records = load_records(args.pkl)
        default_out = str(Path(args.pkl).parent)
    else:
        print(f"Loading batch dir {args.batch_dir} ...")
        records = load_batch_dir(args.batch_dir)
        default_out = args.batch_dir

    output_dir = args.output_dir or default_out

    df = analyze_all(records, output_dir)
    print(f"\nResults ({len(df)} rows):")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
