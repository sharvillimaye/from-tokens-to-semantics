#!/usr/bin/env python3
"""
Merge per-shard n-gram CSVs into a cumulative time series.

Each shard CSV has columns:
  step, phrase, count_file, tokens_in_file, per_million_file

This script:
  - Discovers freq_step_*.csv in --indir
  - Verifies all expected steps (1000, 2000, ..., max) are present and contiguous
  - Loads all shards, strips optional outer quotes from phrases, checks consistency
  - Computes cumulative counts per phrase across steps and cumulative per-million
  - Writes a readable JSON and a long-format Parquet

JSON structure:
{
  "steps": [1000, 2000, ...],
  "phrases": [
    {"phrase": " in Washington", "count_cum": [...], "per_million_cum": [...]},
    ...
  ]
}

Parquet schema (long format):
  phrase:str, step:int64, count_cum:int64, per_million_cum:float64
"""

from __future__ import annotations
import os, sys, re, json, argparse
from typing import List, Tuple
from pathlib import Path

import numpy as np
import pandas as pd

FILENAME_RE = re.compile(r"^freq_step_(\d{6})\.csv$")

def strip_outer_quotes(s: str) -> str:
    """Remove surrounding single OR double quotes if the whole cell is quoted."""
    if not isinstance(s, str) or len(s) < 2:
        return s
    if (s[0] == s[-1]) and (s[0] in ("'", '"')):
        return s[1:-1]
    return s

def discover_files(indir: Path) -> List[Tuple[int, Path]]:
    files: List[Tuple[int, Path]] = []
    for name in os.listdir(indir):
        m = FILENAME_RE.match(name)
        if m:
            step = int(m.group(1))
            files.append((step, indir / name))
    if not files:
        raise SystemExit(f"ERROR: No files matching freq_step_*.csv in {indir}")
    files.sort(key=lambda x: x[0])
    return files

def verify_steps_contiguous(steps: List[int], start_step: int = 1000) -> None:
    if steps[0] != start_step:
        raise SystemExit(f"ERROR: first step is {steps[0]}, expected {start_step}.")
    missing = []
    for a, b in zip(steps, steps[1:]):
        if b - a != 1000:
            missing.extend(range(a + 1000, b, 1000))
    if missing:
        preview = ", ".join(map(str, missing[:20]))
        more = " ..." if len(missing) > 20 else ""
        raise SystemExit(f"ERROR: Missing shard CSV(s) for step(s): {preview}{more}")

def load_shard(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"phrase": str})
    required = {"step", "phrase", "count_file", "tokens_in_file"}
    if not required.issubset(df.columns):
        raise SystemExit(f"ERROR: {path.name} missing columns {sorted(required)}; has {sorted(df.columns)}")
    # normalize & strip quotes from phrases (original phrases have no quotes)
    df = df[["step", "phrase", "count_file", "tokens_in_file"]].copy()
    df["step"] = df["step"].astype("int64")
    df["phrase"] = df["phrase"].map(strip_outer_quotes)
    df["count_file"] = df["count_file"].astype("int64")
    df["tokens_in_file"] = df["tokens_in_file"].astype("int64")
    return df

def ensure_unique_phrases(phrases: List[str], context: str) -> None:
    dup = pd.Series(phrases).duplicated(keep=False)
    if dup.any():
        dups = sorted(set(pd.Series(phrases)[dup].tolist()))
        raise SystemExit(f"ERROR: Duplicate phrases after quote-stripping in {context}: {dups[:5]}{' ...' if len(dups)>5 else ''}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", required=True, help="Directory containing freq_step_*.csv")
    ap.add_argument("--out-json", required=True, help="Output JSON path (readable)")
    ap.add_argument("--out-parquet", required=True, help="Output Parquet path (long format)")
    ap.add_argument("--start-step", type=int, default=1000, help="Expected first step (default: 1000)")
    args = ap.parse_args()

    indir = Path(args.indir)
    files = discover_files(indir)
    steps = [s for s, _ in files]
    verify_steps_contiguous(steps, start_step=args.start_step)

    # Load first shard as canonical phrase order
    first_step, first_path = files[0]
    df0 = load_shard(first_path)
    phrases = df0["phrase"].tolist()
    ensure_unique_phrases(phrases, context=f"{first_path.name}")
    nP = len(phrases)

    # Prepare matrices
    nS = len(files)
    counts = np.zeros((nP, nS), dtype=np.int64)
    tokens_per_file = np.zeros((nS,), dtype=np.int64)

    # Fill matrices; align by phrase each time (order can differ across shards)
    base_index = pd.Index(phrases, name="phrase")
    for j, (step, path) in enumerate(files):
        df = load_shard(path)
        if df.shape[0] != nP:
            raise SystemExit(f"ERROR: Row count mismatch in {path.name}: {df.shape[0]} vs {nP} in {first_path.name}")

        # verify phrase set matches (after stripping quotes)
        ensure_unique_phrases(df["phrase"].tolist(), context=path.name)
        if set(df["phrase"]) != set(phrases):
            missing = set(phrases) - set(df["phrase"])
            extra   = set(df["phrase"]) - set(phrases)
            msg = []
            if missing: msg.append(f"missing={len(missing)}")
            if extra:   msg.append(f"extra={len(extra)}")
            raise SystemExit(f"ERROR: Phrase set mismatch in {path.name} ({', '.join(msg)})")

        # reindex to canonical phrase order
        tmp = df.set_index("phrase").reindex(base_index)
        # tokens_in_file should be constant across rows for this shard
        toks_unique = tmp["tokens_in_file"].unique()
        if len(toks_unique) != 1:
            raise SystemExit(f"ERROR: tokens_in_file not constant in {path.name}: {toks_unique[:5]}")
        tokens_per_file[j] = int(toks_unique[0])

        counts[:, j] = tmp["count_file"].to_numpy(dtype=np.int64)

    # Compute cumulative
    cum_counts = counts.cumsum(axis=1)  # shape (nP, nS)

    # Cumulative tokens are per-step (same for all phrases)
    cum_tokens = tokens_per_file.cumsum().astype(np.int64)  # shape (nS,)
    # Avoid division by zero just in case
    denom = cum_tokens.copy()
    denom[denom == 0] = 1
    per_million_cum = (cum_counts / denom[np.newaxis, :]) * 1e6  # float64

    # -------- Write JSON (readable) --------
    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    payload = {
        "steps": steps,
        "phrases": [
            {
                "phrase": phrases[i],
                "count_cum": [int(x) for x in cum_counts[i, :].tolist()],
                "per_million_cum": [float(x) for x in per_million_cum[i, :].tolist()],
            }
            for i in range(nP)
        ],
    }
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    print(f"[info] Wrote JSON -> {args.out_json}  (phrases={nP}, steps={nS})")

    # -------- Write Parquet (long format) --------
    os.makedirs(os.path.dirname(args.out_parquet) or ".", exist_ok=True)
    # Build long DF efficiently
    phrase_col = np.repeat(np.array(phrases, dtype=object), nS)
    step_col = np.tile(np.array(steps, dtype=np.int64), nP)
    count_cum_col = cum_counts.reshape(-1)
    per_million_cum_col = per_million_cum.reshape(-1)

    out_df = pd.DataFrame({
        "phrase": phrase_col,
        "step": step_col,
        "count_cum": count_cum_col,
        "per_million_cum": per_million_cum_col,
    })
    # Rely on pyarrow if available (you installed it), else pandas will fallback if engine present
    out_df.to_parquet(args.out_parquet, index=False)
    print(f"[info] Wrote Parquet -> {args.out_parquet}  (rows={out_df.shape[0]})")

if __name__ == "__main__":
    main()