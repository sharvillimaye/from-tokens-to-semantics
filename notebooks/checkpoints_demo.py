#!/usr/bin/env python
"""
A script that analyzes clustering and superposition over training checkpoints.
Mostly like demo.py, but supports model checkpoints, robust HF auth handling, and a --series mode. 

NEW:
 --series mode iterates over Pythia-70M training checkpoints and
   writes a JSONL (one row per checkpoint) and a CSV summary.

CHANGES:
- Robust HF auth handling: on 401 error, clear any
  bad tokens (env + local cache) and retry.
- Guard clustering against n < 2 excerpts to avoid HAC crash.
- Make metric keys consistent even on edge cases so CSV writing never fails.
"""

# TODO: Add advanced visualization 1) average inter- and intra-cluster distance at each checkpoint
#                                  2) average cluster size at each checkpoint
#                                  3) max cluster size at each checkpoint
# Run global tests over layers of Pythia-70M and Pythia-160M

import argparse
import os
from pathlib import Path
import re
from collections import Counter, defaultdict
import json
import csv
import time

# Fix tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch  # type: ignore
from datasets import load_dataset  # type: ignore
from transformer_lens import HookedTransformer  # type: ignore
from transformers import AutoTokenizer  # type: ignore
from transformer_lens.loading_from_pretrained import get_checkpoint_labels  # type: ignore

# Try to get HF hub helpers (optional, code works even if unavailable)
try:
    from huggingface_hub import logout as hf_logout, get_token as hf_get_token  # type: ignore
except Exception:  # pragma: no cover
    hf_logout = None
    hf_get_token = None

# ---- Toolkit imports (from the ne_tlk module) --------------------------------
from ne_tlk import (
    TransformerLensEmbeddingCollector,
    cluster_embeddings,
    polysemanticity_metrics,
)

# --------------------------- HF auth hardening --------------------------------
def _clear_hf_auth_env():
    """Delete any HF auth tokens to force requests."""
    for k in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        os.environ.pop(k, None)

def _logout_hf_if_configured():
    """If a token is configured in the hub cache, log it out."""
    try:
        if hf_get_token is not None and hf_get_token():
            if hf_logout is not None:
                try:
                    hf_logout()
                except Exception:
                    pass
    except Exception:
        # Older hub versions or unexpected errors — ignore silently
        pass

def _retry_anonymous_loader(load_fn, *args, **kwargs):
    """
    Call a HF loader function; if it fails with a 401, clear tokens (env + cache)
    and retry once anonymously.
    """
    try:
        return load_fn(*args, **kwargs)
    except OSError as e:
        msg = str(e)
        if "401" in msg or "Unauthorized" in msg:
            _clear_hf_auth_env()
            _logout_hf_if_configured()
            return load_fn(*args, **kwargs)
        raise

def load_step_model(model_name: str, step: int, device: str):
    """Load a model at a given training step, using HF token for authentication."""
    # Get the HF token from environment or cache
    import os
    from huggingface_hub import HfApi # type: ignore
    from pathlib import Path
    
    # Try to get token using env or cache
    token = os.getenv('HF_TOKEN') or os.getenv('HUGGING_FACE_HUB_TOKEN')
    
    if not token:
        # Try to get token using HF cache file
        token_file = Path.home() / ".cache" / "huggingface" / "token"
        if token_file.exists():
            token = token_file.read_text().strip()
        else:
            # Try to get using HF API
            try:
                api = HfApi()
                token = api.token
            except:
                token = None
    
    if token:
        print(f"Using HF token for authentication")
        os.environ['HF_TOKEN'] = token
        try:
            return HookedTransformer.from_pretrained(
                model_name,
                checkpoint_value=step,
                device=device,
            )
        finally:
            # Clean up env variable
            if 'HF_TOKEN' in os.environ:
                del os.environ['HF_TOKEN']
    else:
        print(f"No HF token found, trying without authentication")
        return _retry_anonymous_loader(
            HookedTransformer.from_pretrained,
            model_name,
            checkpoint_value=step,
            device=device,
        )

def load_tokenizer(repo_id: str):
    """Load a tokenizer, using HF token for authentication."""
    import os
    from huggingface_hub import HfApi # type: ignore
    from pathlib import Path
    
    # Try to get token using environment or cache
    token = os.getenv('HF_TOKEN') or os.getenv('HUGGING_FACE_HUB_TOKEN')
    
    if not token:
        # Try to get using HF cache file
        token_file = Path.home() / ".cache" / "huggingface" / "token"
        if token_file.exists():
            token = token_file.read_text().strip()
        else:
            # Try to get using HF API
            try:
                api = HfApi()
                token = api.token
            except:
                token = None
    
    if token:
        print(f"Using HF token for tokenizer authentication")
        # Set the token as env variable for internal use
        os.environ['HF_TOKEN'] = token
        try:
            return AutoTokenizer.from_pretrained(repo_id)
        finally:
            # Clean up env variable
            if 'HF_TOKEN' in os.environ:
                del os.environ['HF_TOKEN']
    else:
        print(f"No HF token found, trying tokenizer without authentication")
        return _retry_anonymous_loader(AutoTokenizer.from_pretrained, repo_id)

# --------------------------- Utility: keyword analysis -------------------------
def analyze_cluster_keywords(texts):
    """Extract the most frequent non-trivial words from a list of texts.
    
    This helps us understand what each cluster is about by looking at the most
    frequent meaningful words in the text excerpts.
    """
    stop_words = {
        'the','a','an','and','or','but','in','on','at','to','for','of','with','by',
        'is','are','was','were','be','been','being','have','has','had','do','does','did',
        'will','would','could','should','may','might','can','must','shall','this','that',
        'these','those','i','you','he','she','it','we','they','me','him','her','us','them',
        'my','your','his','its','our','their','mine','yours','hers','ours','theirs','am',
        'not','no','nor','neither','either','so','as','than','too','very','just','now',
        'then','here','there','when','where','why','how','all','any','both','each','few',
        'more','most','other','some','such','only','own','same','s','t','don','d','ll',
        'm','o','re','ve','y','ain','aren','couldn','didn','doesn','hadn','hasn','haven',
        'isn','ma','mightn','mustn','needn','shan','shouldn','wasn','weren','won','wouldn'
    }
    all_text = ' '.join(texts).lower()
    words = re.findall(r'\b[a-zA-Z]+\b', all_text)
    meaningful_words = [w for w in words if w not in stop_words and len(w) > 2]
    word_counts = Counter(meaningful_words)
    return [w for w, _ in word_counts.most_common(3)]

# --------------------------- Data loader ---------------------------
def wikitext_loader(tokenizer, batch_size=16, split="test", max_examples=30000):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split, streaming=True)
    batch_texts = []
    total_chars = 0
    for item in ds:
        txt = item["text"]
        if not txt:
            continue
        batch_texts.append(txt)
        total_chars += len(txt)
        if len(batch_texts) >= batch_size:
            toks = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            yield toks
            batch_texts = []
        if total_chars >= max_examples:  # character budget
            break
    if batch_texts:
        toks = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        yield toks

# --------------------------- Checkpoint helpers -------------------------------
def get_ckpt_list(model_name: str, mode: str = "log+milestones", max_ckpts: int | None = None, min_step: int = 0, max_step: int | None = None, start_step: int = 3000, skip_steps: int = 10000):
    """
    Return a sorted list of training steps for Pythia checkpoints.
    
    mode:
      - 'all' : all available checkpoints
      - 'log' : step0 + {1,2,4,...,512}
      - 'log+milestones' (default): log + {1000, 10000, 50000, 100000, 143000}
      - 'skip' : start at start_step and skip skip_steps each time
    min_step: minimum checkpoint step to include (default: 0)
    max_step: maximum checkpoint step to include (default: None, no limit)
    start_step: starting checkpoint step (for skip mode)
    skip_steps: number of steps to skip between checkpoints (for skip mode)
    """
    labels, label_type = get_checkpoint_labels(model_name)
    if label_type != "step":
        raise ValueError(f"Expected step-labeled checkpoints, got {label_type}")
    labels_set = set(labels)
    if mode == "all":
        sel = [s for s in labels if (min_step is None or s >= min_step) and (max_step is None or s <= max_step)]
    elif mode == "log":
        target = [0,1,2,4,8,16,32,64,128,256,512]
        sel = [s for s in target if s in labels_set and (min_step is None or s >= min_step) and (max_step is None or s <= max_step)]
    elif mode == "skip":
        # Generate checkpoints starting at start_step, incrementing by skip_steps
        sel = []
        current_step = start_step
        while current_step <= max(labels):
            if current_step in labels_set and (min_step is None or current_step >= min_step) and (max_step is None or current_step <= max_step):
                sel.append(current_step)
            current_step += skip_steps
    else:
        target = [0,1,2,4,8,16,32,64,128,256,512, 1000, 10_000, 50_000, 100_000, 143_000]
        sel = [s for s in target if s in labels_set and (min_step is None or s >= min_step) and (max_step is None or s <= max_step)]
    sel = sorted(sel)
    return sel[:max_ckpts] if max_ckpts else sel

def normalized_pythia_id(model_arg: str) -> str:
    """
    Normalize the --model argument to a proper HF id for Pythia models.
    
    Accepts 'pythia-70m', 'pythia-160m', 'pythia-160m-deduped', 'EleutherAI/pythia-160m-deduped' etc.
    """
    low = model_arg.lower()
    if "pythia-70m" in low and "dedup" in low:
        return "EleutherAI/pythia-70m-deduped-v0"
    if "pythia-70m" in low:
        return "EleutherAI/pythia-70m-v0"
    if "pythia-160m" in low and "dedup" in low:
        return "EleutherAI/pythia-160m-deduped-v0"
    if "pythia-160m" in low:
        return "EleutherAI/pythia-160m-v0"
    return model_arg  # non-Pythia models pass through unchanged

# --------------------------- Per-checkpoint runner ----------------------------
def run_over_checkpoints(args):
    model_name = normalized_pythia_id(args.model)
    print(f"Using model: {model_name}")

    # One tokenizer reused for all checkpoints → identical batches across steps.
    base_tokenizer = load_tokenizer(model_name)
    # Set padding token if not present
    if base_tokenizer.pad_token is None:
        base_tokenizer.pad_token = base_tokenizer.eos_token
    print("Tokenizer loaded successfully")

    # Materialize a deterministic list of raw texts so each checkpoint sees the same data.
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test", streaming=True)
    raw_texts = [ex["text"] for _, ex in zip(range(max(args.series_char_budget // 50, 1_000)), ds)]  # heuristic
    raw_texts = [t for t in raw_texts if t]  # drop empties
    print(f"Loaded {len(raw_texts)} text examples")

    def token_batch_iter():
        for i in range(0, len(raw_texts), args.batch_size):
            toks = base_tokenizer(
                raw_texts[i:i+args.batch_size],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            yield toks

    ckpts = get_ckpt_list(model_name, mode=args.ckpt_mode, max_ckpts=args.max_ckpts, min_step=args.min_step, max_step=args.max_step, start_step=args.start_step, skip_steps=args.skip_steps)
    print(f"Found checkpoints: {ckpts}")

    # Output paths
    layer_num = args.layer.split('.')[1] if '.' in args.layer else "0"
    outdir = Path("results"); outdir.mkdir(exist_ok=True)
    # Extract model size from model name for filename
    model_size = "70m" if "70m" in model_name else "160m" if "160m" in model_name else "unknown"
    series_jsonl = outdir / f"L{layer_num}N{args.neuron}_pythia{model_size}_ckpt_series.jsonl"
    summary_csv  = outdir / f"L{layer_num}N{args.neuron}_pythia{model_size}_ckpt_summary.csv"

    device = (
        "cuda" if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    print(f"Series run on device: {device}")

    with open(series_jsonl, "w") as jf, open(summary_csv, "w", newline="") as cf:
        csv_writer = csv.writer(cf)
        csv_writer.writerow([
            "checkpoint_step", "n_examples", "num_clusters",
            "mean_dist", "mean_intra", "cluster_sizes", "elapsed_sec"
        ])

        for step in ckpts:
            t0 = time.time()
            print(f"\n=== Checkpoint step {step} ===")

            # Load model at this training step
            print(f"Loading model at step {step}...")
            model = load_step_model(model_name, step, device)
            print(f"Model loaded successfully")

            print(f"Setting up collector for layer {args.layer}, neuron {args.neuron}")
            collector = TransformerLensEmbeddingCollector(
                model,
                layer_name=args.layer,
                neuron_idx=args.neuron,
                activation_threshold=args.threshold,
                peak_activation=args.peak_activation,
                max_examples=args.max_examples,
                device=device,
                decode_text=not args.no_text,
            )

            print(f"Running collector...")
            embeds = collector.run(token_batch_iter())  # (N, d_hidden) or None
            n_examples = 0 if embeds is None else int(embeds.shape[0])
            print(f"Collector finished. Embeddings shape: {embeds.shape if embeds is not None else 'None'}")
            print(f"Collected {n_examples} examples")

            # ---- Robust clustering & metrics ------------------
            if embeds is None or n_examples < 2:
                # Build consistent metrics dict
                metrics = {
                    "embeddings": n_examples,
                    "mean_dist": float("nan"),
                    "mean_intra": float("nan"),
                    "mean_inter": float("nan"),
                    "max_dist": float("nan"),
                    "num_clusters": 0 if n_examples == 0 else 1,
                    "single_element_clusters": n_examples,
                    "largest_cluster_size": n_examples if n_examples > 0 else 0,
                    "smallest_cluster_size": n_examples if n_examples > 0 else 0,
                    "cluster_sizes": ({0: n_examples} if n_examples > 0 else {}),
                }
                labels = torch.zeros(n_examples, dtype=torch.long)
                if n_examples < 2:
                    print("  Warning: fewer than 2 examples; skipping HAC and using degenerate metrics.")
            else:
                labels = cluster_embeddings(embeds, distance_threshold=args.distance_threshold)
                metrics = polysemanticity_metrics(embeds, labels)
                print(f"Clustering results: {metrics}")

            rec = {
                "checkpoint_step": step,
                "metrics": metrics,
                "cluster_labels": labels.tolist(),
                "text_examples": getattr(collector, "_text_cache", []),
                "n_examples": n_examples,
                "elapsed_sec": time.time() - t0,
            }
            jf.write(json.dumps(rec) + "\n")

            csv_writer.writerow([
                step,
                n_examples,
                metrics.get("num_clusters", 0),
                metrics.get("mean_dist", float("nan")),
                metrics.get("mean_intra", float("nan")),
                dict(metrics.get("cluster_sizes", {})),
                f"{rec['elapsed_sec']:.2f}",
            ])

            # Clear memory
            del model, collector
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"\nSaved JSONL to {series_jsonl}")
    print(f"Saved CSV   to {summary_csv}")

# --------------------------- Single-run main (original) -----------------------
def main(args):
    # Generate descriptive filename if not provided
    if args.outfile is None:
        layer_num = args.layer.split('.')[1] if '.' in args.layer else "0"
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        args.outfile = results_dir / f"L{layer_num}N{args.neuron}_metrics.json"
    else:
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        if not args.outfile.is_absolute():
            args.outfile = results_dir / args.outfile.name

    # 1. Load model
    model = HookedTransformer.from_pretrained(args.model)
    # IMPORTANT: for series we fix tokenizer separately; here keep original behavior.
    tokenizer = model.tokenizer

    # Set device
    if torch.backends.mps.is_available():
        device = "mps"
        print(f"Using MPS acceleration")
    elif torch.cuda.is_available():
        device = "cuda"
        print(f"Using CUDA acceleration")
    else:
        device = "cpu"
        print(f"Using CPU (no acceleration available)")

    # 2. Set up collector
    collector = TransformerLensEmbeddingCollector(
        model,
        layer_name=args.layer,
        neuron_idx=args.neuron,
        activation_threshold=args.threshold,
        peak_activation=args.peak_activation,
        max_examples=args.max_examples,
        device=device,
        decode_text=not args.no_text,
    )

    # 3. Stream data until collector is full
    embeds = collector.run(
        wikitext_loader(tokenizer, batch_size=args.batch_size)
    )  # shape (N, d_hidden) or None

    n_examples = 0 if embeds is None else int(embeds.shape[0])

    # 4. Cluster & metric summary (with small-N guard)
    if embeds is None or n_examples < 2:
        labels = torch.zeros(n_examples, dtype=torch.long)
        metrics = {
            "embeddings": n_examples,
            "mean_dist": float("nan"),
            "mean_intra": float("nan"),
            "mean_inter": float("nan"),
            "max_dist": float("nan"),
            "num_clusters": 0 if n_examples == 0 else 1,
            "single_element_clusters": n_examples,
            "largest_cluster_size": n_examples if n_examples > 0 else 0,
            "smallest_cluster_size": n_examples if n_examples > 0 else 0,
            "cluster_sizes": ({0: n_examples} if n_examples > 0 else {}),
        }
        if n_examples < 2:
            print("Warning: fewer than 2 examples; skipping HAC and using degenerate metrics.")
    else:
        labels = cluster_embeddings(embeds, distance_threshold=args.distance_threshold)
        metrics = polysemanticity_metrics(embeds, labels)

    # 5. Report
    output_data = {
        "metrics": metrics,
        "labels": labels.tolist(),
        "text_examples": collector._text_cache if hasattr(collector, '_text_cache') else []
    }
    args.outfile.write_text(json.dumps(output_data, indent=2))

    print("==== Polysemanticity metrics ====")
    for k, v in metrics.items():
        if k == 'cluster_sizes':
            print(f"{k:>12}: {dict(v)}")
        else:
            print(f"{k:>12}: {v:.4f}" if isinstance(v, float) else f"{k:>12}: {v}")

    # Print cluster analysis
    print(f"\n==== Cluster Analysis ====")
    text_examples = collector._text_cache if hasattr(collector, '_text_cache') else []

    cluster_texts = defaultdict(list)
    for i, (label, text) in enumerate(zip(labels, text_examples)):
        cluster_texts[int(label)].append(text)

    for cluster_id in sorted(cluster_texts.keys()):
        texts = cluster_texts[cluster_id]
        keywords = analyze_cluster_keywords(texts)
        print(f"\nCluster {cluster_id} ({len(texts)} examples) - Keywords: {', '.join(keywords)}")
        for i, text in enumerate(texts[:5]):
            print(f"  {i+1}. {text}")
        if len(texts) > 5:
            print(f"  ... and {len(texts) - 5} more examples")

    print(f"\nSaved detailed output to {args.outfile.absolute()}")

# --------------------------- CLI ---------------------------------------------
def build_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2-small", help="HF/TransformerLens model alias")
    parser.add_argument("--layer", required=True, help="Layer name containing the neuron (e.g. blocks.6.mlp)")
    parser.add_argument("--neuron", type=int, required=True, help="Row index of neuron in weight matrix")
    parser.add_argument("--max_examples", type=int, default=100, help="How many high-act examples to keep")
    parser.add_argument("--threshold", type=float, default=0.6, help="Threshold for inclusion (fraction of peak activation)")
    parser.add_argument("--peak_activation", type=float, default=2.5, help="Peak activation for this neuron (from Neuroscope)")
    parser.add_argument("--distance_threshold", type=float, default=0.8, help="Distance threshold for clustering (cosine distance)")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--no_text", action="store_true", help="Skip text decoding for maximum speed")
    parser.add_argument("--outfile", type=Path, help="Output file path (single-run metrics JSON)")

    # Series (checkpoint sweep) options
    parser.add_argument("--series", action="store_true", help="Run across Pythia checkpoints and write JSONL + CSV")
    parser.add_argument("--ckpt_mode", default="log+milestones", choices=["all", "log", "log+milestones", "skip"],
                        help="Which checkpoint subset to evaluate (series mode)")
    parser.add_argument("--start_step", type=int, default=3000, help="Starting checkpoint step (for skip mode)")
    parser.add_argument("--skip_steps", type=int, default=10000, help="Number of steps to skip between checkpoints (for skip mode)")
    parser.add_argument("--max_ckpts", type=int, default=None, help="Optional cap on number of checkpoints (series mode)")
    parser.add_argument("--min_step", type=int, default=None, help="Minimum checkpoint step to include (series mode)")
    parser.add_argument("--max_step", type=int, default=None, help="Maximum checkpoint step to include (series mode)")
    parser.add_argument("--series_char_budget", type=int, default=30_000,
                        help="Approximate total characters of text to stream per checkpoint in series mode")

    return parser

if __name__ == "__main__":
    torch.set_grad_enabled(False)
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.series:
        run_over_checkpoints(args)
    else:
        main(args)
