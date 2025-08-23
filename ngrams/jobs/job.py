#!/usr/bin/env python3
"""
Shard worker: process ONE parquet from
  pietrolesci/pile-deduped-pythia-preshuffled @ refs/convert/parquet
and emit per-file counts (corresponding to a 1k-step checkpoint).

Output CSV schema (one row per phrase):
  step, phrase, count_file, tokens_in_file, per_million_file

- step == the number in the filename (e.g., train-043000.parquet -> step 43000)
- tokens_in_file == 1000 * TOKENS_PER_STEP (Pythia batch size per step)
- per_million_file == count_file / tokens_in_file * 1e6

Run this as a SLURM array job: each task picks one parquet by index.
"""

from __future__ import annotations
import os, re, argparse, sys
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import list_repo_files, hf_hub_download
from transformers import AutoTokenizer

# -------------------------
# Hardcoded config
# -------------------------

DATASET_ID = "pietrolesci/pile-deduped-pythia-preshuffled"
REVISION   = "refs/convert/parquet"
PARQUET_RE = re.compile(r"^data/train-(\d{6})\.parquet$")
TOKENS_PER_STEP = 2_097_152  # tokens per step for Pythia (confirm for your exact run)
BATCH_ROWS_DEFAULT = 4096

# Hardcode your n-grams (TEXT) here; they’ll be tokenized with GPT-NeoX tokenizer.
# Keep each on a single line; leading/trailing spaces matter for tokenization.
PHRASES = [
    " in Washington",       # example (with leading space)
    " Washington,",         # example
    " Berlin,",             # example
    # --- add your real list below ---
]

TOKENIZER_NAME = "EleutherAI/gpt-neox-20b"   # Pythia uses GPT-NeoX tokenizer
TOKENIZER_REVISION = None  # optionally pin a specific tokenizer revision/commit

# -------------------------
# Data structures & helpers
# -------------------------

@dataclass(frozen=True)
class Query:
    phrase: str
    toks: np.ndarray  # int32

def list_parquet_index() -> List[Tuple[int, str]]:
    """
    Returns a sorted list of (step_end, filename_relative_to_repo).
    """
    files = list_repo_files(repo_id=DATASET_ID, repo_type="dataset", revision=REVISION)
    out: List[Tuple[int, str]] = []
    for f in files:
        m = PARQUET_RE.match(f)
        if m:
            out.append((int(m.group(1)), f))
    out.sort(key=lambda x: x[0])
    return out

def build_queries(phrases: Iterable[str], tokenizer_name: str, max_len: int = 3) -> Tuple[List[Query], List[str]]:
    tok = AutoTokenizer.from_pretrained(
        tokenizer_name,
        revision=TOKENIZER_REVISION,
        use_fast=True
    )
    queries: List[Query] = []
    filtered_out: List[str] = []
    for p in phrases:
        p = p.rstrip("\n")
        if not p:
            continue
        ids = tok.encode(p, add_special_tokens=False)
        if len(ids) == 0:
            filtered_out.append(f"(empty after tokenization) {repr(p)}")
            continue
        if max_len > 0 and len(ids) > max_len:
            # skip > n tokens (ngram cap)
            filtered_out.append(f"(>{max_len} tokens) {repr(p)}")
            continue
        queries.append(Query(phrase=p, toks=np.asarray(ids, dtype=np.int32)))
    if not queries:
        raise ValueError("No valid phrases after tokenization/length filtering.")
    return queries, filtered_out

def group_by_len_first(queries: List[Query]):
    """
    by_len[L][first_token] -> list of (qi, toks); also return lens, phrases.
    """
    by_len: Dict[int, Dict[int, List[Tuple[int, np.ndarray]]]] = {}
    phrases: List[str] = []
    for qi, q in enumerate(queries):
        L = int(q.toks.shape[0])
        first = int(q.toks[0])
        by_len.setdefault(L, {}).setdefault(first, []).append((qi, q.toks))
        phrases.append(q.phrase)
    lens = sorted(by_len.keys())
    return by_len, lens, phrases

def process_record_batch_sum(
    batch: pa.RecordBatch,
    token_col: str,
    by_len: Dict[int, Dict[int, List[Tuple[int, np.ndarray]]]],
    lens: List[int],
    n_queries: int,
) -> np.ndarray:
    """
    Returns a [n_queries] vector of counts summed across all rows in this batch.
    """
    toks_list: pa.ListArray = batch.column(token_col)
    offsets = toks_list.offsets.to_numpy(zero_copy_only=False)
    values  = toks_list.values.to_numpy(zero_copy_only=False)
    total = np.zeros((n_queries,), dtype=np.int64)

    n_rows = batch.num_rows
    for r in range(n_rows):
        start = int(offsets[r]); end = int(offsets[r+1])
        if end <= start:
            continue
        # Typically int64 view; cast once for dtype match vs int32 query tokens.
        seq = values[start:end].astype(np.int32, copy=False)
        seqlen = end - start

        for L in lens:
            if L > seqlen:
                continue
            upto = seqlen - L + 1
            # For this L, scan by first-token buckets.
            for first_tok, patterns in by_len[L].items():
                # early filter by positions of first token
                pos = np.flatnonzero(seq[:upto] == first_tok)
                if pos.size == 0:
                    continue
                if L == 1:
                    # all 1-gram queries sharing this token id get +1 per position
                    for qi, _ in patterns:
                        total[qi] += int(pos.size)
                else:
                    # check full match per candidate window
                    for p in pos:
                        window = seq[p:p+L]
                        # small list; brute-force is fine
                        for qi, ref in patterns:
                            if np.array_equal(window, ref):
                                total[qi] += 1
    return total

# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, required=True,
                    help="0-based index into the sorted parquet list (use SLURM_ARRAY_TASK_ID).")
    ap.add_argument("--outdir", required=True, help="Directory to write CSV (one file per parquet).")
    ap.add_argument("--batch-rows", type=int, default=BATCH_ROWS_DEFAULT,
                    help="Arrow record-batch rows to process at a time.")
    ap.add_argument("--max-ngram-len", type=int, default=3,
                    help="Skip phrases that tokenize longer than this (default: 3).")
    ap.add_argument("--progress", action="store_true",
                    help="Show a tqdm progress bar over record batches.")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")  # faster hub downloads when available

    # Resolve parquet by index
    files = list_parquet_index()
    if args.index < 0 or args.index >= len(files):
        raise SystemExit(f"--index {args.index} out of range (0..{len(files)-1})")
    step_end, rel_filename = files[args.index]

    # Tokenize + index queries
    queries, filtered = build_queries(PHRASES, TOKENIZER_NAME, max_len=args.max_ngram_len)
    if filtered:
        print(f"[info] {len(filtered)} phrases filtered out:", *filtered, sep="\n  ", file=sys.stderr)
    by_len, lens, phrases = group_by_len_first(queries)
    nQ = len(phrases)
    print(f"[info] step={step_end}  phrases_kept={nQ}  len_buckets={lens}", file=sys.stderr)

    # Download parquet locally, then open with Arrow
    local_path = hf_hub_download(
        repo_id=DATASET_ID,
        repo_type="dataset",
        revision=REVISION,
        filename=rel_filename,
    )
    pf = pq.ParquetFile(local_path)

    # Determine token column
    cols = set(pf.schema.names)
    token_col = "token_ids" if "token_ids" in cols else ("input_ids" if "input_ids" in cols else None)
    if token_col is None:
        raise ValueError(f"No token column found. Available columns: {sorted(cols)}")
    print(f"[info] using token column: {token_col}", file=sys.stderr)

    # Iterate batches and count
    total_counts = np.zeros((nQ,), dtype=np.int64)

    if args.progress:
        try:
            from tqdm import tqdm  # optional
            # ParquetFile doesn't expose total batches; show an indeterminate bar
            batch_iter = pf.iter_batches(batch_size=args.batch_rows, columns=[token_col], use_threads=True)
            for batch in tqdm(batch_iter, desc=f"Counting step {step_end}", unit="batch"):
                total_counts += process_record_batch_sum(batch, token_col, by_len, lens, nQ)
        except Exception as e:
            print(f"[warn] tqdm unavailable or failed ({e}); continuing without progress bar.", file=sys.stderr)
            for batch in pf.iter_batches(batch_size=args.batch_rows, columns=[token_col], use_threads=True):
                total_counts += process_record_batch_sum(batch, token_col, by_len, lens, nQ)
    else:
        for batch in pf.iter_batches(batch_size=args.batch_rows, columns=[token_col], use_threads=True):
            total_counts += process_record_batch_sum(batch, token_col, by_len, lens, nQ)

    # Per-file tokens and per-million
    tokens_in_file = 1000 * TOKENS_PER_STEP
    per_million = (total_counts / max(tokens_in_file, 1)) * 1e6

    # Write CSV named by step
    out_csv = os.path.join(args.outdir, f"freq_step_{step_end:06d}.csv")
    df = pd.DataFrame({
        "step": [step_end]*nQ,
        "phrase": phrases,
        "count_file": total_counts.astype(np.int64),
        "tokens_in_file": [tokens_in_file]*nQ,
        "per_million_file": per_million,
    })
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} ({len(df)} rows)")

if __name__ == "__main__":
    main()