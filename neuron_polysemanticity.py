#!/usr/bin/env python3
"""
Efficient per-neuron polysemanticity (n_clusters) computation.

Unlike the original pipeline (neuron_embeddings/ne_tlk.py) which does a full
forward pass PER NEURON, this script does a SINGLE forward pass per layer to
capture all activations, then processes all neurons from cached data.

Algorithm:
  1. For each layer, register hooks to capture post-activation MLP intermediate
     (4H-dim, the down-projection input) and pre-MLP residual stream (H-dim).
  2. Stream a text dataset through the model in batches, maintaining per-neuron
     min-heaps of the top-k highest-activating positions.
  3. For each neuron, compute neuron embeddings as the Hadamard product of the
     up-projection weight vector and the pre-MLP vectors at top-k positions.
  4. Cluster embeddings via HAC with cosine distance; record n_clusters and
     intra/inter-cluster distances.

Output CSV columns: layer, neuron, n_clusters, n_examples, mean_intra, mean_inter, mean_dist
This is designed to feed directly into coverage_affinity_experiment.py --clusters.

Note on gated MLPs (LLaMA/Mistral): The intermediate activation depends on BOTH
gate_proj and up_proj. For neuron embeddings we use the up_proj weight vector
(what the neuron "listens for"), following Foote (2024). This is an approximation
since the gating modulates the signal.
"""
from __future__ import annotations

import argparse
import csv
import heapq
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances


# ---------------------------------------------------------------------------
# Architecture detection helpers
# ---------------------------------------------------------------------------

LAYER_PATH_CANDIDATES = [
    "gpt_neox.layers", "model.layers", "transformer.h",
    "transformer.layers", "model.decoder.layers",
]
MLP_NAMES = ["mlp", "feed_forward", "ff"]
DOWN_PROJ_NAMES = ["dense_4h_to_h", "down_proj", "c_proj", "fc2", "w2"]
UP_PROJ_NAMES = ["dense_h_to_4h", "up_proj", "c_fc", "fc1", "w1"]
GATE_PROJ_NAMES = ["gate_proj"]


def _resolve_attr(obj: Any, dotted: str) -> Any:
    """Resolve a dotted attribute path like 'model.layers'."""
    for part in dotted.split("."):
        obj = getattr(obj, part)
    return obj


def find_model_layers(model: nn.Module) -> nn.ModuleList:
    for path in LAYER_PATH_CANDIDATES:
        try:
            return _resolve_attr(model, path)
        except AttributeError:
            continue
    raise RuntimeError(
        "Cannot find transformer layers. Tried: " + ", ".join(LAYER_PATH_CANDIDATES)
    )


def find_mlp(layer_mod: nn.Module) -> nn.Module:
    for name in MLP_NAMES:
        mlp = getattr(layer_mod, name, None)
        if mlp is not None:
            return mlp
    raise RuntimeError(
        f"Cannot find MLP in layer. Children: {[n for n, _ in layer_mod.named_children()]}"
    )


def find_submodule(parent: nn.Module, candidates: List[str], label: str) -> nn.Module:
    for name in candidates:
        mod = getattr(parent, name, None)
        if mod is not None:
            return mod
    raise RuntimeError(
        f"Cannot find {label} in module. Children: {[n for n, _ in parent.named_children()]}"
    )


def find_down_proj(mlp: nn.Module) -> nn.Module:
    return find_submodule(mlp, DOWN_PROJ_NAMES, "down-projection")


def find_up_proj(mlp: nn.Module) -> nn.Module:
    return find_submodule(mlp, UP_PROJ_NAMES, "up-projection")


def has_gate_proj(mlp: nn.Module) -> bool:
    return any(getattr(mlp, name, None) is not None for name in GATE_PROJ_NAMES)


# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------

def auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def model_torch_dtype(device: str) -> torch.dtype:
    """Avoid half precision on CPU, where many HF models will fail to run."""
    return torch.float16 if device in {"cuda", "mps"} else torch.float32


def hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


# ---------------------------------------------------------------------------
# Top-k heap tracker for a single neuron
# ---------------------------------------------------------------------------

class NeuronTopK:
    """Min-heap tracking top-k (activation, pre_mlp_vector) pairs for one neuron."""

    __slots__ = ("k", "heap", "threshold")

    def __init__(self, k: int):
        self.k = k
        self.heap: List[Tuple[float, int, np.ndarray]] = []  # (act_val, tiebreak_id, pre_mlp)
        self.threshold = -float("inf")

    def maybe_add(self, act_val: float, pre_mlp: np.ndarray, uid: int) -> None:
        if act_val <= self.threshold and len(self.heap) >= self.k:
            return
        if len(self.heap) < self.k:
            heapq.heappush(self.heap, (act_val, uid, pre_mlp))
            if len(self.heap) == self.k:
                self.threshold = self.heap[0][0]
        else:
            heapq.heapreplace(self.heap, (act_val, uid, pre_mlp))
            self.threshold = self.heap[0][0]

    def get_pre_mlps(self) -> np.ndarray:
        """Return pre_mlp vectors sorted by activation (descending)."""
        items = sorted(self.heap, key=lambda x: -x[0])
        return np.stack([it[2] for it in items]) if items else np.empty((0,))


# ---------------------------------------------------------------------------
# Clustering and metrics (reuses logic from ne_tlk.py)
# ---------------------------------------------------------------------------

def cluster_and_metrics(
    embeddings: np.ndarray, distance_threshold: float
) -> Dict[str, float]:
    """Cluster neuron embeddings and return polysemanticity metrics."""
    n = len(embeddings)
    if n < 2:
        return {
            "n_clusters": 1 if n == 1 else 0,
            "n_examples": n,
            "mean_intra": float("nan"),
            "mean_inter": float("nan"),
            "mean_dist": float("nan"),
        }

    dists = pairwise_distances(embeddings, metric="cosine")
    # Clip NaN from zero-vector embeddings
    np.nan_to_num(dists, nan=1.0, copy=False)

    hac = AgglomerativeClustering(
        metric="precomputed",
        linkage="average",
        distance_threshold=distance_threshold,
        n_clusters=None,  # type: ignore[arg-type]
    )
    labels = hac.fit_predict(dists)

    intra, inter, all_d = [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            d = dists[i, j]
            all_d.append(d)
            if labels[i] == labels[j]:
                intra.append(d)
            else:
                inter.append(d)

    return {
        "n_clusters": int(len(set(labels))),
        "n_examples": n,
        "mean_intra": float(np.mean(intra)) if intra else float("nan"),
        "mean_inter": float(np.mean(inter)) if inter else float("nan"),
        "mean_dist": float(np.mean(all_d)) if all_d else float("nan"),
    }


# ---------------------------------------------------------------------------
# Dataset streaming
# ---------------------------------------------------------------------------

def stream_token_batches(
    dataset_name: str,
    split: str,
    tokenizer,
    batch_size: int,
    max_tokens: int,
    seq_len: int = 512,
):
    """Yield batches of input_ids from a HuggingFace dataset.

    Concatenates all text, chunks into seq_len windows, yields as batches.
    """
    from datasets import load_dataset

    # Handle dataset names with config like "wikitext/wikitext-2-raw-v1"
    parts = dataset_name.split("/")
    if len(parts) == 2 and not parts[0].startswith("http"):
        # Could be org/name or name/config — try as dataset with config first
        try:
            ds = load_dataset(parts[0], parts[1], split=split, )
        except Exception:
            ds = load_dataset(dataset_name, split=split, )
    else:
        ds = load_dataset(dataset_name, split=split, )

    # Find text column
    text_col = None
    for col in ["text", "content", "sentence"]:
        if col in ds.column_names:
            text_col = col
            break
    if text_col is None:
        text_col = ds.column_names[0]

    # Concatenate and tokenize
    all_ids: List[int] = []
    for row in ds:
        text = row[text_col]
        if not text or not text.strip():
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)
        all_ids.extend(ids)
        if len(all_ids) >= max_tokens:
            break

    all_ids = all_ids[:max_tokens]
    print(f"  Tokenized {len(all_ids):,} tokens from {dataset_name} ({split})")

    # Chunk into sequences
    n_seqs = len(all_ids) // seq_len
    if n_seqs == 0:
        raise RuntimeError(f"Not enough tokens ({len(all_ids)}) for seq_len={seq_len}")

    tensor = torch.tensor(all_ids[: n_seqs * seq_len]).reshape(n_seqs, seq_len)

    for start in range(0, n_seqs, batch_size):
        yield tensor[start : start + batch_size]


# ---------------------------------------------------------------------------
# Main processing: one layer at a time
# ---------------------------------------------------------------------------

def _precompute_token_batches(
    dataset_name: str,
    split: str,
    tokenizer,
    batch_size: int,
    max_tokens: int,
    seq_len: int = 512,
) -> List[torch.Tensor]:
    """Tokenize dataset once, return list of batch tensors for reuse across layers."""
    batches = list(stream_token_batches(dataset_name, split, tokenizer, batch_size, max_tokens, seq_len))
    print(f"  Pre-computed {len(batches)} batches ({sum(b.numel() for b in batches):,} tokens)")
    return batches


def process_layer(
    model: nn.Module,
    model_layers: nn.ModuleList,
    layer_idx: int,
    token_batches: List[torch.Tensor],
    device: str,
    top_k: int,
    distance_threshold: float,
    activation_threshold: float,
) -> List[Dict[str, Any]]:
    """Process a single layer: stream data, collect top-k, cluster, return metrics."""

    layer_mod = model_layers[layer_idx]
    mlp_mod = find_mlp(layer_mod)
    down_mod = find_down_proj(mlp_mod)
    up_mod = find_up_proj(mlp_mod)

    # Get up-projection weight: shape [4H, H] for standard, or similar
    up_weight = up_mod.weight.detach().float().cpu().numpy()  # [4H, H]
    n_neurons = up_weight.shape[0]
    h_dim = up_weight.shape[1]

    print(f"  Layer {layer_idx}: {n_neurons} neurons, H={h_dim}")

    # Storage for captured activations (set by hooks, read by main loop)
    captured: Dict[str, torch.Tensor] = {}

    def hook_mlp_input(_mod, inp, _out):
        x = inp[0] if isinstance(inp, tuple) else inp
        captured["pre_mlp"] = x.detach()

    def hook_down_input(_mod, inp, _out):
        x = inp[0] if isinstance(inp, tuple) else inp
        captured["post_act"] = x.detach()

    h1 = mlp_mod.register_forward_hook(hook_mlp_input)
    h2 = down_mod.register_forward_hook(hook_down_input)

    # Initialize top-k heaps for all neurons in this layer
    heaps = [NeuronTopK(top_k) for _ in range(n_neurons)]
    uid_counter = 0

    # Vectorized threshold array for fast batch filtering
    thresholds = np.full(n_neurons, -np.inf)

    try:
        n_batches = 0
        for batch_ids in token_batches:
            batch_ids = batch_ids.to(device)
            with torch.no_grad():
                model(batch_ids)

            # post_act: [batch, seq, 4H], pre_mlp: [batch, seq, H]
            post_act = captured["post_act"].float().cpu()  # [B, S, 4H]
            pre_mlp = captured["pre_mlp"].float().cpu()    # [B, S, H]

            _4H = post_act.shape[-1]

            # Flatten to [N_pos, 4H] and [N_pos, H]
            post_np = post_act.reshape(-1, _4H).numpy()
            pre_np = pre_mlp.reshape(-1, h_dim).numpy()
            n_pos = post_np.shape[0]

            # Vectorized: find max activation per neuron across all positions
            # Only process neurons where max > their current threshold
            max_per_neuron = post_np.max(axis=0)  # [4H]
            active_neurons = np.where(max_per_neuron > thresholds)[0]

            for neuron_idx in active_neurons:
                act_col = post_np[:, neuron_idx]  # [N_pos]
                heap = heaps[neuron_idx]

                # Filter positions above threshold
                if activation_threshold > 0.0:
                    candidate_indices = np.where(act_col > activation_threshold)[0]
                elif heap.threshold > -np.inf:
                    candidate_indices = np.where(act_col > heap.threshold)[0]
                else:
                    # Heap not full — take top positions to fill fast
                    candidate_indices = np.argpartition(act_col, -min(top_k * 2, n_pos))[-min(top_k * 2, n_pos):]

                for pos_idx in candidate_indices:
                    heap.maybe_add(
                        float(act_col[pos_idx]),
                        pre_np[pos_idx],
                        uid_counter,
                    )
                    uid_counter += 1

                # Update vectorized threshold
                thresholds[neuron_idx] = heap.threshold

            n_batches += 1
            if n_batches % 20 == 0:
                n_active = len(active_neurons)
                print(f"    Batch {n_batches}/{len(token_batches)} "
                      f"({n_active}/{n_neurons} neurons active)")

            captured.clear()

    finally:
        h1.remove()
        h2.remove()

    print(f"    Done streaming ({n_batches} batches). Clustering {n_neurons} neurons...")

    # Compute neuron embeddings and cluster
    results = []
    for neuron_idx in range(n_neurons):
        heap = heaps[neuron_idx]
        pre_mlps = heap.get_pre_mlps()  # [k, H]

        if len(pre_mlps) == 0 or pre_mlps.ndim < 2:
            results.append({
                "layer": layer_idx,
                "neuron": neuron_idx,
                "n_clusters": 0,
                "n_examples": 0,
                "mean_intra": float("nan"),
                "mean_inter": float("nan"),
                "mean_dist": float("nan"),
            })
            continue

        # Neuron embedding = pre_mlp * W_up[neuron_idx, :]  (Hadamard product)
        w_up = up_weight[neuron_idx]  # [H]
        embeddings = pre_mlps * w_up[np.newaxis, :]  # [k, H]

        metrics = cluster_and_metrics(embeddings, distance_threshold)
        results.append({
            "layer": layer_idx,
            "neuron": neuron_idx,
            **metrics,
        })

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute per-neuron polysemanticity (n_clusters) for any HF model."
    )
    p.add_argument("--model", default="EleutherAI/pythia-70m-deduped",
                    help="HuggingFace model ID")
    p.add_argument("--revision", default="main",
                    help="Model revision / checkpoint")
    p.add_argument("--dataset", default="wikitext/wikitext-2-raw-v1",
                    help="HuggingFace dataset (name or name/config)")
    p.add_argument("--split", default="test",
                    help="Dataset split")
    p.add_argument("--max-tokens", type=int, default=500_000,
                    help="Max tokens to process")
    p.add_argument("--top-k", type=int, default=100,
                    help="Top-k activating positions per neuron")
    p.add_argument("--distance-threshold", type=float, default=0.8,
                    help="HAC distance threshold for clustering")
    p.add_argument("--layers", default="all",
                    help="Comma-separated layer indices, or 'all'")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--output", required=True,
                    help="Output CSV path")
    p.add_argument("--device", default="auto",
                    choices=["auto", "cuda", "mps", "cpu"])
    p.add_argument("--activation-threshold", type=float, default=0.0,
                    help="Minimum activation to consider (0.0 = use top-k only)")
    return p.parse_args()


def main():
    args = parse_args()

    device = args.device if args.device != "auto" else auto_device()
    print(f"Device: {device}")

    # Load model and tokenizer
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading model: {args.model} (revision={args.revision})")
    token = hf_token()
    tok_kw = {"revision": args.revision}
    model_kw = {
        "revision": args.revision,
        "torch_dtype": model_torch_dtype(device),
    }
    if token:
        tok_kw["token"] = token
        model_kw["token"] = token

    tokenizer = AutoTokenizer.from_pretrained(args.model, **tok_kw)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        **model_kw,
    ).to(device).eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Find layers
    model_layers = find_model_layers(model)
    n_layers = len(model_layers)
    print(f"Model has {n_layers} layers")

    if args.layers == "all":
        layer_indices = list(range(n_layers))
    else:
        layer_indices = [int(x.strip()) for x in args.layers.split(",")]

    # Verify architecture detection on first layer
    test_layer = model_layers[layer_indices[0]]
    test_mlp = find_mlp(test_layer)
    test_down = find_down_proj(test_mlp)
    test_up = find_up_proj(test_mlp)
    is_gated = has_gate_proj(test_mlp)
    print(f"MLP type: {'gated' if is_gated else 'standard'}")
    print(f"  up_proj: {type(test_up).__name__} {tuple(test_up.weight.shape)}")
    print(f"  down_proj: {type(test_down).__name__} {tuple(test_down.weight.shape)}")

    # Pre-compute token batches once (reused for all layers)
    print(f"\nTokenizing dataset...")
    token_batches = _precompute_token_batches(
        args.dataset, args.split, tokenizer,
        args.batch_size, args.max_tokens,
    )

    # Process each layer
    all_results: List[Dict[str, Any]] = []

    for li, layer_idx in enumerate(layer_indices):
        print(f"\nProcessing layer {layer_idx} ({li+1}/{len(layer_indices)})")
        results = process_layer(
            model=model,
            model_layers=model_layers,
            layer_idx=layer_idx,
            token_batches=token_batches,
            device=device,
            top_k=args.top_k,
            distance_threshold=args.distance_threshold,
            activation_threshold=args.activation_threshold,
        )
        all_results.extend(results)

        # Print summary for this layer
        clusters = [r["n_clusters"] for r in results if r["n_clusters"] > 0]
        if clusters:
            print(f"    Clusters: mean={np.mean(clusters):.1f}, "
                  f"median={np.median(clusters):.0f}, "
                  f"max={max(clusters)}, "
                  f"monosemantic(1)={sum(1 for c in clusters if c == 1)}/{len(clusters)}")

    # Write CSV
    import os
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    fieldnames = ["layer", "neuron", "n_clusters", "n_examples",
                  "mean_intra", "mean_inter", "mean_dist"]
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_results:
            writer.writerow(row)

    print(f"\nWrote {len(all_results)} rows to {args.output}")


if __name__ == "__main__":
    main()
