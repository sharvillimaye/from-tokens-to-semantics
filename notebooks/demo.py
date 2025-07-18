#!/usr/bin/env python
"""
1. Loads a TransformerLens model (default: GPT‑2‑small)
2. Streams WikiText‑2 test split through the model
3. Collects up to N high‑activation examples for a target neuron
4. Clusters the embeddings with HAC + cosine distance
5. Prints polysemanticity metrics
"""

import argparse
from pathlib import Path

import torch # type: ignore
from datasets import load_dataset # type: ignore
from transformer_lens import HookedTransformer # type: ignore

# ---- Toolkit imports (from the ne_tlk module) --------------------------------
from ne_tlk import (
    EmbeddingCollector,
    cluster_embeddings,
    polysemanticity_metrics,
)

# ------------------------------------------------------------------------------
def wikitext_loader(tokenizer, batch_size=4, split="test", max_examples=10_000):
    """Yields dicts that TransformerLens models accept (`input_ids`, `attention_mask`)."""
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split, streaming=False)
    for start in range(0, min(len(ds), max_examples), batch_size):
        texts = ds[start : start + batch_size]["text"]
        toks = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        yield toks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2-small", help="HF/TransformerLens model alias")
    parser.add_argument("--layer", required=True, help="Layer name containing the neuron (e.g. blocks.6.mlp)")
    parser.add_argument("--neuron", type=int, required=True, help="Row index of neuron in weight matrix")
    parser.add_argument("--max_examples", type=int, default=100, help="How many high‑act examples to keep")
    parser.add_argument("--threshold", type=float, default=0.75, help="Activation threshold (fraction of peak)")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--outfile", type=Path, help="Output file path (optional, will auto-generate if not provided)")
    args = parser.parse_args()

    # Generate descriptive filename if not provided
    if args.outfile is None:
        # Extract layer number from layer name (e.g., "blocks.6.mlp" -> "6")
        layer_num = args.layer.split('.')[1] if '.' in args.layer else "0"
        # Create results directory if it doesn't exist
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        args.outfile = results_dir / f"L{layer_num}N{args.neuron}_metrics.json"
    else:
        # If outfile is provided, ensure it's in the results directory
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        if not args.outfile.is_absolute():
            args.outfile = results_dir / args.outfile.name

    # 1. Load model
    model = HookedTransformer.from_pretrained(args.model)
    tokenizer = model.tokenizer

    # 2. Set up collector
    collector = EmbeddingCollector(
        model,
        layer_name=args.layer,
        neuron_idx=args.neuron,
        activation_threshold=args.threshold,
        max_examples=args.max_examples,
    )

    # 3. Stream data until collector is full
    embeds = collector.run(
        wikitext_loader(tokenizer, batch_size=args.batch_size)
    )  # shape (N, d_hidden)

    # 4. Cluster & metric summary
    labels = cluster_embeddings(embeds)
    metrics = polysemanticity_metrics(embeds, labels)

    # 5. Report
    import json

    args.outfile.write_text(json.dumps({"metrics": metrics, "labels": labels.tolist()}, indent=2))
    print("==== Polysemanticity metrics ====")
    for k, v in metrics.items():
        print(f"{k:>12}: {v:.4f}" if isinstance(v, float) else f"{k:>12}: {v}")
    print(f"\nSaved detailed output to {args.outfile.absolute()}")


if __name__ == "__main__":
    # CUDA makes this faster but is optional
    torch.set_grad_enabled(False)
    main()
