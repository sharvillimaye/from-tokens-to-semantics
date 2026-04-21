#!/usr/bin/env python3
"""
Input Weight Analysis of MLP Neurons by Coverage and Affinity

For each MLP neuron h, compute what token embeddings are most aligned with
its input filter: W_up[h, :] @ W_embed.T -> similarity to each token.

This tells us what tokens each neuron is "listening for" — the input side,
complementing the logit lens (output side).

We also compute cosine similarity between each neuron's input weights and
the "frequency direction" (the direction that best separates high vs low
frequency token embeddings).
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


def get_model_weights(model):
    """Extract W_up, W_down, embedding, and unembedding for each layer."""
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

    # Find embedding
    for attr in ["gpt_neox.embed_in", "model.embed_tokens", "transformer.wte"]:
        obj = model
        try:
            for part in attr.split("."):
                obj = getattr(obj, part)
            W_embed = obj.weight.detach().float()  # (vocab, hidden)
            break
        except AttributeError:
            continue
    else:
        raise RuntimeError("Cannot find embedding matrix")

    # Find unembedding
    for attr in ["embed_out", "lm_head", "output"]:
        unembed = getattr(model, attr, None)
        if unembed is not None:
            W_unembed = unembed.weight.detach().float()
            break
    else:
        raise RuntimeError("Cannot find unembedding matrix")

    layer_weights = []
    for i, layer_mod in enumerate(layers):
        for mn in ["mlp", "feed_forward", "ff"]:
            mlp = getattr(layer_mod, mn, None)
            if mlp is None:
                continue
            # Up projection
            up = None
            for un in ["dense_h_to_4h", "up_proj", "c_fc", "fc1", "w1"]:
                up = getattr(mlp, un, None)
                if up is not None:
                    break
            # Down projection
            down = None
            for dn in ["dense_4h_to_h", "down_proj", "c_proj", "fc2", "w2"]:
                down = getattr(mlp, dn, None)
                if down is not None:
                    break
            if up is not None and down is not None:
                W_up = up.weight.detach().float()    # (4H, hidden)
                W_down = down.weight.detach().float()  # (hidden, 4H)
                layer_weights.append((i, W_up, W_down))
                break

    return layer_weights, W_embed, W_unembed


def compute_frequency_direction(W_embed, tokenizer, high_freq_tokens, low_freq_tokens):
    """Compute the direction in embedding space that separates high from low freq tokens.

    Returns a unit vector in R^hidden.
    """
    def get_mean_embedding(tokens):
        ids = []
        for tok in tokens:
            encoded = tokenizer.encode(tok, add_special_tokens=False)
            if encoded:
                ids.append(encoded[0])
        if not ids:
            return None
        embeddings = W_embed[ids]  # (n, hidden)
        return embeddings.mean(dim=0)

    high_mean = get_mean_embedding(high_freq_tokens)
    low_mean = get_mean_embedding(low_freq_tokens)

    if high_mean is None or low_mean is None:
        return None

    freq_dir = high_mean - low_mean
    freq_dir = freq_dir / freq_dir.norm()
    return freq_dir


def categorize_token(tok):
    """Categorize a single token."""
    tok_stripped = tok.strip()
    if not tok_stripped:
        return "whitespace"
    if tok_stripped in ".,;:!?()[]{}\"'-/\\@#$%^&*":
        return "punctuation"
    if tok_stripped.lower() in {
        "the", "a", "an", "is", "was", "are", "were", "be", "been",
        "have", "has", "had", "do", "does", "did", "will", "would",
        "could", "should", "may", "might", "can", "shall", "must",
        "to", "of", "in", "for", "on", "with", "at", "by", "from",
        "that", "this", "it", "he", "she", "they", "we", "you", "i",
        "not", "but", "and", "or", "if", "so", "as", "than", "then",
    }:
        return "function_word"
    if tok_stripped[0].isupper() and len(tok_stripped) > 1:
        return "capitalized"
    if any(c.isdigit() for c in tok_stripped):
        return "numeric"
    if len(tok_stripped) <= 2:
        return "short_subword"
    return "content_word"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="EleutherAI/pythia-70m-deduped")
    parser.add_argument("--revision", default="step143000")
    parser.add_argument("--metrics", default="results/coverage_affinity/pythia-70m/emotion/neuron_metrics.csv")
    parser.add_argument("--output-dir", default="results/input_weights/pythia-70m/")
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
    layer_weights, W_embed, W_unembed = get_model_weights(model)
    vocab_size, hidden_dim = W_embed.shape
    print(f"  {len(layer_weights)} layers, vocab={vocab_size}, hidden={hidden_dim}")

    # Load neuron metrics
    print(f"Loading metrics from: {args.metrics}")
    df = pd.read_csv(args.metrics)
    if args.datasets:
        for path in args.datasets:
            df = pd.concat([df, pd.read_csv(path)], ignore_index=True)
    df = df[df["checkpoint"] == args.revision].copy()
    print(f"  {len(df)} neuron records for {args.revision}")

    # Compute frequency direction using common high/low freq words
    high_freq_words = [
        "the", "of", "and", "to", "in", "is", "was", "for", "that", "it",
        "with", "as", "be", "on", "are", "by", "this", "had", "from", "or",
        "have", "an", "but", "not", "you", "all", "were", "her", "she", "there",
        "would", "their", "been", "has", "when", "who", "will", "more", "no", "if",
        "he", "do", "its", "my", "we", "they", "so", "can", "what", "which",
    ]
    low_freq_words = [
        "elated", "melancholy", "jubilant", "morose", "ecstatic",
        "despondent", "exuberant", "forlorn", "serene", "irate",
        "benevolent", "malevolent", "eloquent", "nefarious", "ephemeral",
        "ubiquitous", "sanguine", "perfunctory", "obsequious", "recalcitrant",
        "ameliorate", "exacerbate", "obfuscate", "promulgate", "adjudicate",
        "jurisprudence", "pathology", "etiology", "prognosis", "hemorrhage",
    ]
    freq_dir = compute_frequency_direction(W_embed, tokenizer, high_freq_words, low_freq_words)
    if freq_dir is not None:
        print(f"  Frequency direction computed (norm={freq_dir.norm():.3f})")

    # Normalize embeddings for cosine similarity
    W_embed_norm = W_embed / (W_embed.norm(dim=1, keepdim=True) + 1e-8)

    results = {}

    for layer_idx, W_up, W_down in layer_weights:
        print(f"\nLayer {layer_idx}: W_up shape={W_up.shape}")
        n_neurons = W_up.shape[0]

        # Input sensitivity: what tokens is each neuron listening for?
        # W_up[h, :] is the input weight vector for neuron h (row of W_up)
        # Cosine similarity with each token embedding
        W_up_norm = W_up / (W_up.norm(dim=1, keepdim=True) + 1e-8)  # (4H, hidden)

        # input_sim[h, v] = cosine(W_up[h], W_embed[v])
        input_sim = W_up_norm @ W_embed_norm.T  # (4H, vocab)

        # Output contribution (logit lens)
        output_logits = W_unembed @ W_down  # (vocab, 4H)

        # Frequency direction alignment per neuron
        if freq_dir is not None:
            # cosine(W_up[h], freq_dir)
            freq_alignment = (W_up_norm @ freq_dir).numpy()  # (4H,)
        else:
            freq_alignment = np.zeros(n_neurons)

        # Get metrics for this layer
        layer_df = df[df["layer"] == layer_idx].copy()
        if len(layer_df) == 0:
            continue

        layer_df["coverage_quartile"] = pd.qcut(
            layer_df["coverage_total"], 4, labels=["Q1_low", "Q2", "Q3", "Q4_high"],
            duplicates="drop"
        )
        layer_df["affinity_bin"] = pd.cut(
            layer_df["frequency_affinity"],
            bins=[0, 0.35, 0.45, 0.55, 0.65, 1.0],
            labels=["strong_low", "mild_low", "neutral", "mild_high", "strong_high"]
        )

        # Analyze each group
        groups_to_analyze = [
            ("Q1_low", layer_df[layer_df["coverage_quartile"] == "Q1_low"]),
            ("Q4_high", layer_df[layer_df["coverage_quartile"] == "Q4_high"]),
        ]
        # Add affinity groups for late layers
        if layer_idx >= len(layer_weights) // 2:
            for abin in ["strong_low", "strong_high"]:
                sub = layer_df[layer_df["affinity_bin"] == abin]
                if len(sub) >= 10:
                    groups_to_analyze.append((abin, sub))

        for group_name, group_df in groups_to_analyze:
            neurons = group_df["neuron"].tolist()
            if not neurons:
                continue

            # Aggregate input sensitivity
            all_input_tokens = []
            all_input_cats = defaultdict(int)
            all_output_tokens = []
            all_output_cats = defaultdict(int)
            freq_alignments = []
            example_neurons = []

            for n_idx in neurons[:300]:
                if n_idx >= n_neurons:
                    continue

                # Input: top tokens this neuron listens for
                in_sim = input_sim[n_idx]
                top_in_vals, top_in_ids = torch.topk(in_sim, args.top_k)
                in_tokens = [tokenizer.decode([tid]) for tid in top_in_ids.tolist()]

                # Output: top tokens this neuron promotes
                out_logits = output_logits[:, n_idx]
                top_out_vals, top_out_ids = torch.topk(out_logits, args.top_k)
                out_tokens = [tokenizer.decode([tid]) for tid in top_out_ids.tolist()]

                all_input_tokens.extend(in_tokens[:5])
                all_output_tokens.extend(out_tokens[:5])

                for t in in_tokens[:10]:
                    all_input_cats[categorize_token(t)] += 1
                for t in out_tokens[:10]:
                    all_output_cats[categorize_token(t)] += 1

                freq_alignments.append(freq_alignment[n_idx])

                if len(example_neurons) < 5:
                    cov_val = group_df[group_df["neuron"] == n_idx]["coverage_total"]
                    aff_val = group_df[group_df["neuron"] == n_idx]["frequency_affinity"]
                    example_neurons.append({
                        "neuron": int(n_idx),
                        "input_top5": in_tokens[:5],
                        "input_sims": [round(x, 4) for x in top_in_vals[:5].tolist()],
                        "output_top5": out_tokens[:5],
                        "output_logits": [round(x, 3) for x in top_out_vals[:5].tolist()],
                        "freq_alignment": round(float(freq_alignment[n_idx]), 4),
                        "coverage": round(float(cov_val.iloc[0]), 4) if len(cov_val) > 0 else None,
                        "affinity": round(float(aff_val.iloc[0]), 4) if len(aff_val) > 0 else None,
                    })

            # Summarize
            in_total = sum(all_input_cats.values())
            out_total = sum(all_output_cats.values())
            in_pcts = {k: round(v / in_total * 100, 1) for k, v in sorted(all_input_cats.items(), key=lambda x: -x[1])} if in_total > 0 else {}
            out_pcts = {k: round(v / out_total * 100, 1) for k, v in sorted(all_output_cats.items(), key=lambda x: -x[1])} if out_total > 0 else {}

            # Input token frequency
            in_counts = defaultdict(int)
            for t in all_input_tokens:
                t_clean = t.strip()
                if t_clean:
                    in_counts[t_clean] += 1
            top_input = sorted(in_counts.items(), key=lambda x: -x[1])[:15]

            out_counts = defaultdict(int)
            for t in all_output_tokens:
                t_clean = t.strip()
                if t_clean:
                    out_counts[t_clean] += 1
            top_output = sorted(out_counts.items(), key=lambda x: -x[1])[:15]

            key = f"layer_{layer_idx}_{group_name}"
            results[key] = {
                "layer": layer_idx,
                "group": group_name,
                "n_neurons": len(neurons),
                "input_categories": in_pcts,
                "output_categories": out_pcts,
                "top_input_tokens": top_input,
                "top_output_tokens": top_output,
                "mean_freq_alignment": round(float(np.mean(freq_alignments)), 5) if freq_alignments else None,
                "std_freq_alignment": round(float(np.std(freq_alignments)), 5) if freq_alignments else None,
                "mean_coverage": round(float(group_df["coverage_total"].mean()), 4),
                "mean_affinity": round(float(group_df["frequency_affinity"].mean()), 4),
                "mean_jsd": round(float(group_df["jsd_contrib"].mean()), 8),
                "example_neurons": example_neurons,
            }

    # Save
    with open(output_dir / "input_weight_analysis.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {output_dir / 'input_weight_analysis.json'}")

    # Print summary
    print("\n" + "=" * 90)
    print("INPUT vs OUTPUT WEIGHT ANALYSIS")
    print("=" * 90)

    for key in sorted(results.keys()):
        r = results[key]
        print(f"\n{'='*80}")
        print(f"Layer {r['layer']} | {r['group']} | n={r['n_neurons']}")
        print(f"  Coverage={r['mean_coverage']:.3f}  Affinity={r['mean_affinity']:.3f}  JSD={r['mean_jsd']:.8f}")
        print(f"  Freq direction alignment: mean={r['mean_freq_alignment']:.4f} std={r['std_freq_alignment']:.4f}")
        print(f"  INPUT categories (what it listens for):  {r['input_categories']}")
        print(f"  OUTPUT categories (what it promotes):    {r['output_categories']}")

        junk = {'', '\ufffd', '\ufffd\ufffd', '�', '��'}
        in_clean = [t for t, c in r['top_input_tokens'] if t not in junk][:8]
        out_clean = [t for t, c in r['top_output_tokens'] if t not in junk][:8]
        print(f"  INPUT top tokens:  {in_clean}")
        print(f"  OUTPUT top tokens: {out_clean}")

        if r['example_neurons']:
            ex = r['example_neurons'][0]
            print(f"  Example neuron #{ex['neuron']} (cov={ex['coverage']}, aff={ex['affinity']}, freq_align={ex['freq_alignment']}):")
            print(f"    listens for: {ex['input_top5']}")
            print(f"    promotes:    {ex['output_top5']}")

    # Summary table: frequency alignment by group
    print("\n" + "=" * 90)
    print("FREQUENCY DIRECTION ALIGNMENT SUMMARY")
    print("(positive = aligned with high-freq direction, negative = aligned with low-freq)")
    print("=" * 90)
    for layer_idx in range(len(layer_weights)):
        groups = [(k, v) for k, v in results.items() if v["layer"] == layer_idx]
        if groups:
            parts = []
            for k, v in sorted(groups, key=lambda x: x[0]):
                parts.append(f"{v['group']}: {v['mean_freq_alignment']:+.4f}")
            print(f"  Layer {layer_idx}: {' | '.join(parts)}")


if __name__ == "__main__":
    main()
