#!/usr/bin/env python3
"""
Logit Lens Analysis of MLP Neurons by Coverage and Affinity

For each MLP neuron, compute its "logit contribution" — what tokens it promotes
when it fires — by multiplying its W_down column through the unembedding matrix.

Then group neurons by coverage and affinity quartiles and characterize what
each group encodes.

Usage:
    python logit_lens_neurons.py \
        --model EleutherAI/pythia-70m-deduped \
        --revision step143000 \
        --metrics results/coverage_affinity/pythia-70m/emotion/neuron_metrics.csv \
        --output-dir results/logit_lens/pythia-70m/emotion/
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def get_model_components(model):
    """Extract W_down and unembedding matrix for each layer."""
    # Find layers
    for attr in ["gpt_neox.layers", "model.layers", "transformer.h"]:
        obj = model
        try:
            for part in attr.split("."):
                obj = getattr(obj, part)
            layers = obj
            break
        except AttributeError:
            continue
    else:
        raise RuntimeError("Cannot find transformer layers")

    # Find unembedding
    for attr in ["embed_out", "lm_head", "output"]:
        unembed = getattr(model, attr, None)
        if unembed is not None:
            break
    else:
        raise RuntimeError("Cannot find unembedding matrix")

    W_unembed = unembed.weight.detach().float()  # (vocab, hidden)

    layer_info = []
    for i, layer_mod in enumerate(layers):
        # Find down projection
        for mn in ["mlp", "feed_forward", "ff"]:
            mlp = getattr(layer_mod, mn, None)
            if mlp is None:
                continue
            for dn in ["dense_4h_to_h", "down_proj", "c_proj", "fc2", "w2"]:
                down = getattr(mlp, dn, None)
                if down is not None:
                    W_down = down.weight.detach().float()  # (hidden, 4H)
                    layer_info.append((i, W_down))
                    break
            if down is not None:
                break

    return layer_info, W_unembed


def analyze_neuron_logits(W_down, W_unembed, tokenizer, top_k=10):
    """For each neuron, compute top promoted and suppressed tokens.

    W_down: (hidden, 4H) — column h is neuron h's output vector
    W_unembed: (vocab, hidden)

    logit_contribution[h] = W_unembed @ W_down[:, h]  -> (vocab,)
    """
    # (vocab, hidden) @ (hidden, 4H) = (vocab, 4H)
    logit_matrix = W_unembed @ W_down  # each column is a neuron's vocab contribution
    n_neurons = logit_matrix.shape[1]

    results = []
    for h in range(n_neurons):
        logits = logit_matrix[:, h]
        top_vals, top_ids = torch.topk(logits, top_k)
        bot_vals, bot_ids = torch.topk(-logits, top_k)

        top_tokens = [tokenizer.decode([tid]) for tid in top_ids.tolist()]
        bot_tokens = [tokenizer.decode([tid]) for tid in bot_ids.tolist()]

        results.append({
            "neuron": h,
            "top_tokens": top_tokens,
            "top_logits": top_vals.tolist(),
            "bottom_tokens": bot_tokens,
            "bottom_logits": (-bot_vals).tolist(),
        })
    return results


def categorize_tokens(tokens, tokenizer):
    """Rough categorization of what kind of tokens a neuron promotes."""
    categories = defaultdict(int)
    for tok in tokens:
        tok_stripped = tok.strip()
        if not tok_stripped:
            categories["whitespace/empty"] += 1
        elif tok_stripped in ".,;:!?()[]{}\"'-/\\@#$%^&*":
            categories["punctuation"] += 1
        elif tok_stripped.lower() in {
            "the", "a", "an", "is", "was", "are", "were", "be", "been",
            "have", "has", "had", "do", "does", "did", "will", "would",
            "could", "should", "may", "might", "can", "shall", "must",
            "to", "of", "in", "for", "on", "with", "at", "by", "from",
            "that", "this", "it", "he", "she", "they", "we", "you", "i",
            "not", "but", "and", "or", "if", "so", "as", "than", "then",
        }:
            categories["function_word"] += 1
        elif tok_stripped[0].isupper() and len(tok_stripped) > 1:
            categories["capitalized/proper"] += 1
        elif any(c.isdigit() for c in tok_stripped):
            categories["numeric"] += 1
        elif len(tok_stripped) <= 2:
            categories["short_subword"] += 1
        else:
            categories["content_word"] += 1
    return dict(categories)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="EleutherAI/pythia-70m-deduped")
    parser.add_argument("--revision", default="step143000")
    parser.add_argument("--metrics", default="results/coverage_affinity/pythia-70m/emotion/neuron_metrics.csv")
    parser.add_argument("--output-dir", default="results/logit_lens/pythia-70m/emotion/")
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--datasets", nargs="*", help="Additional metric CSVs to merge")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {args.model} @ {args.revision}")
    hf_token = _hf_token()
    model_kw = {"revision": args.revision, "torch_dtype": torch.float32}
    tok_kw = {"revision": args.revision}
    if hf_token:
        model_kw["token"] = hf_token
        tok_kw["token"] = hf_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, **model_kw
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, **tok_kw)

    print("Extracting weight matrices...")
    layer_info, W_unembed = get_model_components(model)
    print(f"  Found {len(layer_info)} layers, vocab={W_unembed.shape[0]}, hidden={W_unembed.shape[1]}")

    # Load neuron metrics
    print(f"Loading metrics from: {args.metrics}")
    df = pd.read_csv(args.metrics)

    # If multiple datasets provided, merge them
    if args.datasets:
        dfs = [df]
        for path in args.datasets:
            dfs.append(pd.read_csv(path))
        df = pd.concat(dfs, ignore_index=True)

    # Filter to the target checkpoint
    df = df[df["checkpoint"] == args.revision].copy()
    print(f"  {len(df)} neuron records for {args.revision}")

    # Compute logit lens per layer
    all_results = []
    summary_by_group = {}

    for layer_idx, W_down in layer_info:
        print(f"\nLayer {layer_idx}: W_down shape = {W_down.shape}")
        n_neurons = W_down.shape[1]

        # Get logit contributions for all neurons
        neuron_logits = analyze_neuron_logits(W_down, W_unembed, tokenizer, top_k=args.top_k)

        # Get coverage/affinity for this layer
        layer_df = df[df["layer"] == layer_idx].copy()
        if len(layer_df) == 0:
            print(f"  No metrics for layer {layer_idx}, skipping")
            continue

        # Coverage quartiles
        layer_df["coverage_quartile"] = pd.qcut(
            layer_df["coverage_total"], 4, labels=["Q1_low", "Q2", "Q3", "Q4_high"],
            duplicates="drop"
        )

        # Affinity bins (relative to 0.5)
        layer_df["affinity_bin"] = pd.cut(
            layer_df["frequency_affinity"],
            bins=[0, 0.35, 0.45, 0.55, 0.65, 1.0],
            labels=["strong_low", "mild_low", "neutral", "mild_high", "strong_high"]
        )

        # Analyze each coverage quartile
        for quartile in ["Q1_low", "Q4_high"]:
            neurons_in_group = layer_df[layer_df["coverage_quartile"] == quartile]["neuron"].tolist()
            if not neurons_in_group:
                continue

            # Aggregate top tokens across neurons in this group
            all_top_tokens = []
            all_categories = defaultdict(int)
            example_neurons = []

            for n_idx in neurons_in_group[:200]:  # sample up to 200
                if n_idx < len(neuron_logits):
                    entry = neuron_logits[n_idx]
                    all_top_tokens.extend(entry["top_tokens"][:5])
                    cats = categorize_tokens(entry["top_tokens"][:10], tokenizer)
                    for k, v in cats.items():
                        all_categories[k] += v

                    if len(example_neurons) < 5:
                        example_neurons.append({
                            "neuron": n_idx,
                            "top_5_tokens": entry["top_tokens"][:5],
                            "top_5_logits": [round(x, 3) for x in entry["top_logits"][:5]],
                            "coverage": float(layer_df[layer_df["neuron"] == n_idx]["coverage_total"].iloc[0]) if len(layer_df[layer_df["neuron"] == n_idx]) > 0 else None,
                            "affinity": float(layer_df[layer_df["neuron"] == n_idx]["frequency_affinity"].iloc[0]) if len(layer_df[layer_df["neuron"] == n_idx]) > 0 else None,
                        })

            # Token frequency analysis
            token_counts = defaultdict(int)
            for t in all_top_tokens:
                token_counts[t.strip()] += 1
            most_common = sorted(token_counts.items(), key=lambda x: -x[1])[:20]

            total_cats = sum(all_categories.values())
            cat_pcts = {k: round(v / total_cats * 100, 1) for k, v in sorted(all_categories.items(), key=lambda x: -x[1])}

            key = f"layer_{layer_idx}_{quartile}"
            summary_by_group[key] = {
                "layer": layer_idx,
                "group": quartile,
                "n_neurons": len(neurons_in_group),
                "category_percentages": cat_pcts,
                "most_common_tokens": most_common[:15],
                "example_neurons": example_neurons,
                "mean_coverage": float(layer_df[layer_df["coverage_quartile"] == quartile]["coverage_total"].mean()),
                "mean_affinity": float(layer_df[layer_df["coverage_quartile"] == quartile]["frequency_affinity"].mean()),
                "mean_jsd_contrib": float(layer_df[layer_df["coverage_quartile"] == quartile]["jsd_contrib"].mean()),
            }

        # Also analyze by affinity in late layers
        if layer_idx >= len(layer_info) // 2:
            for aff_bin in ["strong_low", "strong_high"]:
                neurons_in_group = layer_df[layer_df["affinity_bin"] == aff_bin]["neuron"].tolist()
                if len(neurons_in_group) < 10:
                    continue

                all_top_tokens = []
                all_categories = defaultdict(int)
                example_neurons = []

                for n_idx in neurons_in_group[:200]:
                    if n_idx < len(neuron_logits):
                        entry = neuron_logits[n_idx]
                        all_top_tokens.extend(entry["top_tokens"][:5])
                        cats = categorize_tokens(entry["top_tokens"][:10], tokenizer)
                        for k, v in cats.items():
                            all_categories[k] += v

                        if len(example_neurons) < 5:
                            example_neurons.append({
                                "neuron": n_idx,
                                "top_5_tokens": entry["top_tokens"][:5],
                                "top_5_logits": [round(x, 3) for x in entry["top_logits"][:5]],
                                "coverage": float(layer_df[layer_df["neuron"] == n_idx]["coverage_total"].iloc[0]) if len(layer_df[layer_df["neuron"] == n_idx]) > 0 else None,
                                "affinity": float(layer_df[layer_df["neuron"] == n_idx]["frequency_affinity"].iloc[0]) if len(layer_df[layer_df["neuron"] == n_idx]) > 0 else None,
                            })

                token_counts = defaultdict(int)
                for t in all_top_tokens:
                    token_counts[t.strip()] += 1
                most_common = sorted(token_counts.items(), key=lambda x: -x[1])[:20]

                total_cats = sum(all_categories.values())
                cat_pcts = {k: round(v / total_cats * 100, 1) for k, v in sorted(all_categories.items(), key=lambda x: -x[1])} if total_cats > 0 else {}

                key = f"layer_{layer_idx}_{aff_bin}"
                summary_by_group[key] = {
                    "layer": layer_idx,
                    "group": aff_bin,
                    "n_neurons": len(neurons_in_group),
                    "category_percentages": cat_pcts,
                    "most_common_tokens": most_common[:15],
                    "example_neurons": example_neurons,
                    "mean_coverage": float(layer_df[layer_df["affinity_bin"] == aff_bin]["coverage_total"].mean()),
                    "mean_affinity": float(layer_df[layer_df["affinity_bin"] == aff_bin]["frequency_affinity"].mean()),
                    "mean_jsd_contrib": float(layer_df[layer_df["affinity_bin"] == aff_bin]["jsd_contrib"].mean()),
                }

    # Save results
    output_file = output_dir / "logit_lens_summary.json"
    with open(output_file, "w") as f:
        json.dump(summary_by_group, f, indent=2)
    print(f"\nSaved summary to {output_file}")

    # Print readable summary
    print("\n" + "=" * 80)
    print("LOGIT LENS ANALYSIS SUMMARY")
    print("=" * 80)

    for key in sorted(summary_by_group.keys()):
        info = summary_by_group[key]
        print(f"\n--- Layer {info['layer']} | {info['group']} ({info['n_neurons']} neurons) ---")
        print(f"    Mean coverage: {info['mean_coverage']:.3f}")
        print(f"    Mean affinity: {info['mean_affinity']:.3f}")
        print(f"    Mean JSD contrib: {info['mean_jsd_contrib']:.6f}")
        print(f"    Token categories: {info['category_percentages']}")
        print(f"    Most common promoted tokens: {[t for t, c in info['most_common_tokens'][:10]]}")
        if info['example_neurons']:
            print(f"    Example neuron: #{info['example_neurons'][0]['neuron']}")
            print(f"      promotes: {info['example_neurons'][0]['top_5_tokens']}")
            print(f"      coverage={info['example_neurons'][0]['coverage']:.3f}, affinity={info['example_neurons'][0]['affinity']:.3f}")


if __name__ == "__main__":
    main()
