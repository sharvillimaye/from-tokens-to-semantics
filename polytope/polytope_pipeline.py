#!/usr/bin/env python3
"""
Polytope Density Pipeline — End-to-End

Complete pipeline from synonym pair dataset to pairwise polytope density results.

Steps:
  1. Load synonym pair dataset (JSONL with high_freq_ngram / low_freq_ngram)
  2. For each pair, extract activations at each layer:
     - mlp_input_vector: residual stream entering MLP (for Euclidean distance)
     - pre_activation_vector: up-projection output (for spline codes)
     - activation_vector: layer output (kept for compatibility)
  3. Compute pairwise polytope density per Humayun et al.:
       rho = Hamming(spline_high, spline_low) / ||mlp_input_high - mlp_input_low||_2
  4. Average per layer, save CSV/JSON results

Usage:
    python -m polytope.polytope_pipeline \\
        --dataset path/to/emotion_ngrams_filtered.jsonl \\
        --model EleutherAI/pythia-70m-deduped \\
        --revision step143000 \\
        --output-dir results/polytope_density/

    # Or skip extraction if you already have activation records:
    python -m polytope.polytope_pipeline \\
        --records path/to/activation_records.pkl \\
        --output-dir results/polytope_density/
"""

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from polytope.direct_pair_density import analyze_all


# ---------------------------------------------------------------------------
# Step 1: Load synonym pair dataset
# ---------------------------------------------------------------------------

def load_synonym_pairs(dataset_path: str) -> List[Dict[str, Any]]:
    """Load synonym pairs from JSONL file.

    Expected format per line:
        {"high_freq_ngram": "happy", "low_freq_ngram": "elated",
         "sentence_template": "She felt [TERM] after ...", "category": "emotion", ...}
    """
    pairs = []
    with open(dataset_path) as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    print(f"Loaded {len(pairs)} synonym pairs from {dataset_path}")
    return pairs


def pairs_to_samples(pairs: List[Dict[str, Any]], tokenizer) -> List[Dict[str, Any]]:
    """Convert synonym pairs into individual samples ready for activation extraction.

    Each pair produces TWO samples (high-freq and low-freq) sharing the same pair_id.
    """
    DEFAULT_TEMPLATE = "The word {ngram} means"
    samples = []

    for i, pair in enumerate(pairs):
        # Use synonym_pair_seed, id, or index as pair identifier
        pair_id = pair.get("synonym_pair_seed", pair.get("id", f"pair_{i}"))
        high_ngram = pair["high_freq_ngram"]
        low_ngram = pair["low_freq_ngram"]
        template = pair.get("sentence_template")

        for ngram, freq_cat in [(high_ngram, "high_freq"), (low_ngram, "low_freq")]:
            # Build text
            if template and "[TERM]" in template:
                text = template.replace("[TERM]", ngram)
            else:
                text = DEFAULT_TEMPLATE.format(ngram=ngram)

            token_ids = tokenizer.encode(text, add_special_tokens=False)

            # Find anchor: last token of the n-gram in the text
            if template and "[TERM]" in template:
                term_pos = template.find("[TERM]")
                suffix = template[term_pos + 6:]
                if suffix:
                    suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
                    anchor = len(token_ids) - len(suffix_ids) - 1
                else:
                    anchor = len(token_ids) - 1
            else:
                suffix_ids = tokenizer.encode(" means", add_special_tokens=False)
                anchor = len(token_ids) - len(suffix_ids) - 1

            anchor = max(0, min(anchor, len(token_ids) - 1))

            samples.append({
                "pair_id": pair_id,
                "phrase": ngram,
                "sentence": text,
                "frequency_category": freq_cat,
                "token_ids_sentence": token_ids,
                "activation_target_idx": anchor,
                "phrase_start_idx": 0,
                "phrase_end_idx": 0,
                "category": pair.get("category", "unknown"),
            })

    print(f"Created {len(samples)} samples ({len(pairs)} pairs x 2)")
    return samples


# ---------------------------------------------------------------------------
# Step 2: Extract activations
# ---------------------------------------------------------------------------

def extract_all_activations(
    samples: List[Dict[str, Any]],
    model_id: str,
    revision: str,
    target_layers: List[int],
    device: str = "cuda",
) -> List[Dict[str, Any]]:
    """Extract activations for all samples across all layers.

    Uses nnsight for hook-based extraction.  Returns flat list of activation records.
    """
    from nnsight import LanguageModel

    print(f"Loading model {model_id} @ {revision} on {device} ...")
    kwargs = {"device_map": device}
    # Some models (e.g. Llama) require a HuggingFace token
    import os
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if hf_token:
        kwargs["token"] = hf_token
    model = LanguageModel(model_id, revision=revision, **kwargs)
    model.eval()

    from polytope.checkpoint_analysis import (
        extract_layer_activations,
        create_activation_record,
        _detect_activation_function,
    )

    all_records = []
    for idx, sample in enumerate(samples):
        if idx % 20 == 0:
            print(f"  [{idx+1}/{len(samples)}] {sample['phrase']} ({sample['frequency_category']})")

        token_ids = sample["token_ids_sentence"]
        position = sample["activation_target_idx"]

        for layer in target_layers:
            try:
                activation_type = _detect_activation_function(model, layer)
                post_act, pre_act, mlp_input = extract_layer_activations(
                    model, token_ids, layer, position, strategy="single"
                )

                record = create_activation_record(
                    checkpoint_step=revision,
                    sample_idx=idx,
                    phrase=sample["phrase"],
                    sentence=sample["sentence"],
                    frequency_category=sample["frequency_category"],
                    layer=layer,
                    activation_target_idx=position,
                    phrase_start_idx=sample["phrase_start_idx"],
                    phrase_end_idx=sample["phrase_end_idx"],
                    activation_vector=post_act,
                    pre_activation_vector=pre_act,
                    mlp_input_vector=mlp_input,
                    activation_type=activation_type,
                    pair_id=sample["pair_id"],
                )
                all_records.append(record)

            except Exception as e:
                print(f"    WARN: layer {layer} failed for {sample['phrase']}: {e}")
                continue

    print(f"Extracted {len(all_records)} activation records")

    # Cleanup
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return all_records


# ---------------------------------------------------------------------------
# Step 3: Save intermediate records
# ---------------------------------------------------------------------------

def save_records(records: List[Dict[str, Any]], output_path: str):
    """Save activation records to pickle."""
    with open(output_path, "wb") as f:
        pickle.dump({"records": records}, f)
    print(f"Saved {len(records)} records to {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

# Default model configs
MODEL_LAYERS = {
    "pythia-70m": 6,
    "pythia-160m": 12,
    "pythia-410m": 24,
    "pythia-1b": 16,
    "pythia-1.4b": 24,
    "pythia-2.8b": 32,
    "pythia-6.9b": 32,
    "pythia-12b": 36,
    "olmo-1b": 16,
    "olmo-7b": 32,
}


def _infer_num_layers(model_id: str) -> int:
    """Guess number of layers from model name."""
    model_lower = model_id.lower()
    for key, n_layers in MODEL_LAYERS.items():
        if key in model_lower:
            return n_layers
    return 12  # conservative default


def main():
    parser = argparse.ArgumentParser(description="Polytope Density Pipeline")

    # Input: either a dataset + model, or pre-extracted records
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--dataset", help="Path to synonym pairs JSONL")
    input_group.add_argument("--records", help="Path to pre-extracted activation records .pkl")
    input_group.add_argument("--batch-dir", help="Directory of .pkl record files")

    # Model config (only needed with --dataset)
    parser.add_argument("--model", default="EleutherAI/pythia-70m-deduped")
    parser.add_argument("--revision", default="step143000")
    parser.add_argument("--layers", type=str, default=None,
                        help="Comma-separated layer indices (default: all layers)")
    parser.add_argument("--device", default=None,
                        help="Device (default: auto-detect cuda/mps/cpu)")

    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--save-records", action="store_true",
                        help="Save intermediate activation records to pkl")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load or extract records ---
    if args.dataset:
        # Full pipeline: dataset -> extraction -> density
        from transformers import AutoTokenizer

        device = args.device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

        tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)

        if args.layers:
            target_layers = [int(x) for x in args.layers.split(",")]
        else:
            n_layers = _infer_num_layers(args.model)
            target_layers = list(range(n_layers))

        print(f"Model: {args.model} @ {args.revision}")
        print(f"Layers: {target_layers}")
        print(f"Device: {device}")

        pairs = load_synonym_pairs(args.dataset)
        samples = pairs_to_samples(pairs, tokenizer)
        records = extract_all_activations(
            samples, args.model, args.revision, target_layers, device
        )

        if args.save_records:
            pkl_path = output_dir / "activation_records.pkl"
            save_records(records, str(pkl_path))

    elif args.records:
        print(f"Loading pre-extracted records from {args.records} ...")
        records_data = pickle.load(open(args.records, "rb"))
        if isinstance(records_data, dict) and "records" in records_data:
            records = records_data["records"]
        else:
            records = records_data
        print(f"Loaded {len(records)} records")

    else:
        # batch-dir
        from polytope.direct_pair_density import load_batch_dir
        records = load_batch_dir(args.batch_dir)

    # --- Compute pairwise polytope density ---
    print("\n=== Computing pairwise polytope density ===")
    df = analyze_all(records, str(output_dir))

    print(f"\n=== Results ({len(df)} rows) ===")
    print(df.to_string(index=False))

    # Summary by layer
    if len(df) > 0:
        print("\n=== Per-layer summary ===")
        layer_summary = df.groupby("layer").agg({
            "density_mean": "mean",
            "density_std": "mean",
            "hamming_mean": "mean",
            "euclidean_mean": "mean",
            "normalized_hamming_mean": "mean",
            "n_pairs": "first",
        })
        print(layer_summary.to_string())


if __name__ == "__main__":
    main()
