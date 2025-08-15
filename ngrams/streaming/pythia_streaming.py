from __future__ import annotations
from typing import Dict, List, Sequence, Optional
import os
import glob
import numpy as np
from transformers import AutoTokenizer

PYTHIA_TOKENS_PER_STEP: int = 2_097_152  # 2^21

def canonical_pythia_steps(include_early: bool = True) -> np.ndarray:
    """
    {0,1,2,4,8,16,32,64,128,256,512,1000} plus every 1k up to 143k.
    """
    regular = np.arange(1000, 143_001, 1000, dtype=np.int64)
    if include_early:
        early = np.array([0,1,2,4,8,16,32,64,128,256,512,1000], dtype=np.int64)
        early = early[early != 1000]
        return np.concatenate([early, regular])
    return regular

def cutoffs_from_steps(steps: Sequence[int], *, tokens_per_step: int = PYTHIA_TOKENS_PER_STEP) -> np.ndarray:
    s = np.asarray(steps, dtype=np.int64).ravel()
    if s.ndim != 1:
        raise ValueError("steps must be 1-D")
    if tokens_per_step <= 0:
        raise ValueError("tokens_per_step must be > 0")
    return s * np.int64(tokens_per_step)

# ---------------------------
# Internal utilities
# ---------------------------

def _choose_dtype(shard_paths: List[str], vocab_size: int, override: Optional[np.dtype]) -> np.dtype:
    if override is not None:
        return override
    # Prefer uint16 for Pythia (vocab ~50k) if all shard sizes are even.
    if vocab_size <= (1 << 16):
        for p in shard_paths:
            if os.path.getsize(p) % 2 != 0:
                # some shard not aligned to 2 bytes -> likely uint32
                return np.uint32
        return np.uint16
    return np.uint32

def _pow_vec(base: np.uint64, L: int) -> np.ndarray:
    """Vector [base**(L-1), base**(L-2), ..., 1] in uint64 (wrapping)."""
    out = np.empty(L, dtype=np.uint64)
    cur = np.uint64(1)
    for i in range(L):
        out[L - 1 - i] = cur
        cur = (cur * base)  # wraps mod 2^64
    return out

def _tokenize_group_by_len(phrases: List[str], tok: AutoTokenizer) -> Dict[int, List[tuple]]:
    by_len: Dict[int, List[tuple]] = {}
    for p in phrases:
        ids = tok.encode(p, add_special_tokens=False)
        if not ids:
            raise ValueError(f"Phrase tokenized to empty sequence: {p!r}")
        arr = np.asarray(ids, dtype=np.uint32)
        by_len.setdefault(len(arr), []).append((p, arr))
    return by_len

# ---------------------------
# Core streaming counter
# ---------------------------

def count_ngrams_at_checkpoints(
    shards_glob: str,
    phrases: List[str],
    steps: Sequence[int],
    *,
    tokenizer_name: str = "EleutherAI/pythia-70m",
    chunk_tokens: int = 5_000_000,
    tokens_per_step: int = PYTHIA_TOKENS_PER_STEP,
    count_mode: str = "full",  # "full" (end-based) or "start" (start-based)
    keep_example_positions_per_pattern: int = 0,
    dtype_override: Optional[np.dtype] = None,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Stream across shard .bin files, count matches, and accumulate counts at given steps.

    Returns for each phrase:
        {
          "steps":   (M,) int64  -- the steps you requested (sorted)
          "cutoffs": (M,) int64  -- cumulative token cutoffs for those steps
          "count":   (M,) int64  -- occurrences <= each cutoff (by the chosen definition)
          "freq":    (M,) float64-- count / cutoff
          "examples":(<=K,) int64-- optional example START positions (global)
        }
    """
    if count_mode not in ("full", "start"):
        raise ValueError("count_mode must be 'full' or 'start'")

    # Tokenizer & phrases
    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    by_len = _tokenize_group_by_len(phrases, tok)
    if not by_len:
        raise ValueError("No valid phrases.")

    # Prepare cutoffs
    steps_arr = np.asarray(steps, dtype=np.int64).ravel()
    cutoffs = cutoffs_from_steps(steps_arr, tokens_per_step=tokens_per_step)
    order = np.argsort(cutoffs)
    cutoffs = cutoffs[order]
    steps_arr = steps_arr[order]
    M = cutoffs.size

    # Per-phrase accumulators via "include_from" trick -> prefix sum at the end
    include_from = {p: np.zeros(M, dtype=np.int64) for p in phrases}
    examples = {p: [] for p in phrases}

    # Build per-length hash tables: L -> (pow_vec, {hash: [(phrase, toks_np), ...]})
    BASE = np.uint64(1465329397)  # arbitrary odd base
    per_len_tables = {}
    Lmax = 0
    for L, plist in by_len.items():
        pv = _pow_vec(BASE, L)
        table: Dict[int, List[tuple]] = {}
        for p, t in plist:
            h = int((pv.astype(np.uint64) * t.astype(np.uint64)).sum(dtype=np.uint64))
            table.setdefault(h, []).append((p, t))
        per_len_tables[L] = (pv, table)
        Lmax = max(Lmax, L)

    # Discover shards & choose dtype
    shard_paths = sorted(glob.glob(shards_glob))
    if not shard_paths:
        raise FileNotFoundError(f"No shards matched glob {shards_glob!r}")

    dtype = _choose_dtype(shard_paths, tok.vocab_size, dtype_override)

    # Streaming state
    global_offset = 0  # start index of current shard in the global stream
    prev_tail_u64 = np.array([], dtype=np.uint64)  # last (Lmax-1) tokens of previous shard

    # Process shards sequentially
    for shard_idx, shard in enumerate(shard_paths):
        x = np.memmap(shard, dtype=dtype, mode="r")
        N = int(x.size)
        if N == 0:
            continue

        step = max(int(chunk_tokens), Lmax + 1)
        start = 0

        while start < N:
            end = min(N, start + step)
            # For cross-chunk coverage, take Lmax-1 overlap within shard:
            s_rel = start - (Lmax - 1) if start > 0 else start  # relative to shard
            chunk = np.asarray(x[max(0, s_rel):end])  # (end - s_rel) elements
            base_pos = global_offset + max(0, s_rel)  # global index of chunk[0]

            # For the *first* chunk in a shard, prepend prev_tail to catch cross-shard matches
            if start == 0 and prev_tail_u64.size > 0:
                needed = Lmax - 1
                tail = prev_tail_u64[-needed:] if prev_tail_u64.size >= needed else prev_tail_u64
                if tail.size > 0:
                    chunk = np.concatenate([tail.view(np.uint64), chunk.astype(np.uint64, copy=False)])
                    base_pos = global_offset - tail.size

            chunk_u64 = chunk.astype(np.uint64, copy=False)

            for L, (pv, table) in per_len_tables.items():
                if chunk_u64.size < L:
                    continue
                windows = np.lib.stride_tricks.sliding_window_view(chunk_u64, L)  # shape: (m-L+1, L)
                H = (windows * pv).sum(axis=1, dtype=np.uint64)  # (m-L+1,)

                for hkey, plist in table.items():
                    cand = np.nonzero(H == np.uint64(hkey))[0]
                    if cand.size == 0:
                        continue
                    Wc = windows[cand]  # candidate windows

                    for phrase, toks_np in plist:
                        eq = np.all(Wc == toks_np.astype(np.uint64), axis=1)
                        hits_local = cand[eq]  # local starts within 'chunk'
                        if hits_local.size == 0:
                            continue

                        starts = base_pos + hits_local.astype(np.int64)
                        ends = starts + L
                        if count_mode == "full":
                            # Count by end-position; drop windows entirely in the previous-shard tail
                            ref = ends
                            ref_mask = ends > global_offset
                            side = "left"   # ends <= cutoff
                        else:
                            # Count by start-position:
                            # - allow windows that cross into this shard (ends > global_offset)
                            # - still drop windows entirely inside the previous-shard tail (ends <= global_offset)
                            ref = starts
                            ref_mask = ends > global_offset
                            side = "right"  # start < cutoff (strict)

                        if np.any(ref_mask):
                            ref_valid = ref[ref_mask]
                            i_min = np.searchsorted(cutoffs, ref_valid, side=side)
                            in_range = i_min < M
                            if np.any(in_range):
                                np.add.at(include_from[phrase], i_min[in_range], 1)

                        if keep_example_positions_per_pattern > 0 and len(examples[phrase]) < keep_example_positions_per_pattern:
                            need = keep_example_positions_per_pattern - len(examples[phrase])
                            examples[phrase].extend(starts[:need].astype(int).tolist())

            start = end

        # Update cross-shard tail (last Lmax-1 tokens of this shard)
        if Lmax > 1:
            tail_len = min(Lmax - 1, N)
            prev_tail_u64 = np.asarray(x[N - tail_len : N], dtype=dtype).astype(np.uint64, copy=False)
        else:
            prev_tail_u64 = np.array([], dtype=np.uint64)

        global_offset += N

    # Finalize: prefix-sum accumulators -> counts; then frequencies
    out: Dict[str, Dict[str, np.ndarray]] = {}
    cutoffs_f = cutoffs.astype(np.float64)
    for p in phrases:
        counts = include_from[p].cumsum()
        freqs = counts.astype(np.float64) / np.maximum(cutoffs_f, 1.0)
        pack = {
            "steps": steps_arr.copy(),
            "cutoffs": cutoffs.copy(),
            "count": counts,
            "freq": freqs,
        }
        if keep_example_positions_per_pattern > 0:
            pack["examples"] = np.asarray(examples[p], dtype=np.int64)
        out[p] = pack
    return out

# ---------------------------
# Optional: cross-shard self-test
# ---------------------------

def _self_test_cross_shard() -> None:
    """
    Fabricate two tiny shards so that a phrase straddles the boundary.
    Verifies both 'full' and 'start' counting exactly.
    """
    import tempfile, shutil
    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
    phrase = " in Washington"
    ids = np.array(tok.encode(phrase, add_special_tokens=False), dtype=np.uint16)
    L = len(ids)
    assert L >= 2, "Pick a phrase that tokenizes to >=2 tokens for this test."

    filler = np.array([1], dtype=np.uint16)
    shardA = np.concatenate([filler.repeat(10), ids[:-1]], dtype=np.uint16)
    shardB = np.concatenate([ids[-1:], filler.repeat(10)], dtype=np.uint16)

    tmp = tempfile.mkdtemp(prefix="tkg_xshard_")
    try:
        pA = os.path.join(tmp, "document-00000-of-00002.bin")
        pB = os.path.join(tmp, "document-00001-of-00002.bin")
        shardA.tofile(pA); shardB.tofile(pB)

        steps = [len(shardA)-1, len(shardA), len(shardA)+1, len(shardA)+L]
        # FULL-SPAN
        res_full = count_ngrams_at_checkpoints(
            shards_glob=os.path.join(tmp, "document-*-of-*.bin"),
            phrases=[phrase],
            steps=steps,
            tokenizer_name=tok.name_or_path,
            chunk_tokens=64,
            tokens_per_step=1,
            count_mode="full",
            dtype_override=np.uint16,
        )
        stream_full = res_full[phrase]["count"]

        # START-BASED
        res_start = count_ngrams_at_checkpoints(
            shards_glob=os.path.join(tmp, "document-*-of-*.bin"),
            phrases=[phrase],
            steps=steps,
            tokenizer_name=tok.name_or_path,
            chunk_tokens=64,
            tokens_per_step=1,
            count_mode="start",
            dtype_override=np.uint16,
        )
        stream_start = res_start[phrase]["count"]

        # Ground truth on concatenation
        concat = np.concatenate([shardA, shardB]).astype(np.uint16)
        wins = np.lib.stride_tricks.sliding_window_view(concat, L)
        starts = np.where(np.all(wins == ids, axis=1))[0]
        ends = starts + L
        cutoffs = np.array(steps, dtype=np.int64)
        gt_full = np.array([(ends <= c).sum() for c in cutoffs], dtype=np.int64)
        gt_start = np.array([(starts <  c).sum() for c in cutoffs], dtype=np.int64)

        np.testing.assert_array_equal(stream_full,  gt_full)
        np.testing.assert_array_equal(stream_start, gt_start)
        print("✅ Cross-shard boundary test passed (full & start).")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

if __name__ == "__main__":
    _self_test_cross_shard()
