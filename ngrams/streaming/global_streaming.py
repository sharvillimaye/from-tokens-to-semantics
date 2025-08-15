# shardwise_streamer.py
from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Dict, List, Sequence, Optional, Tuple

import numpy as np
from huggingface_hub import hf_hub_download, list_repo_tree
from transformers import AutoTokenizer


# ---------------------------
# Checkpoint schedule helpers
# ---------------------------

def canonical_pythia_steps(include_early: bool = True) -> np.ndarray:
    """
    Canonical Pythia checkpoints:
      early: {0,1,2,4,8,16,32,64,128,256,512,1000}
      then every 1k up to 143k (inclusive)
    """
    regular = np.arange(1000, 143_001, 1000, dtype=np.int64)
    if include_early:
        early = np.array([0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000], dtype=np.int64)
        early = early[early != 1000]
        return np.concatenate([early, regular])
    return regular


# ---------------------------
# Internals
# ---------------------------

def _pow_vec(base: int, L: int) -> np.ndarray:
    """
    Return [base**(L-1), base**(L-2), ..., 1] modulo 2^64 as uint64.

    Uses pure-Python modular multiply to avoid NumPy overflow warnings and
    OverflowError on certain builds when converting large Python ints.
    """
    MOD = 1 << 64
    base64 = int(base) & (MOD - 1)

    out = np.empty(L, dtype=np.uint64)
    cur = 1  # Python int (arbitrary precision)
    for i in range(L):
        out[L - 1 - i] = np.uint64(cur & (MOD - 1))
        cur = (cur * base64) & (MOD - 1)  # mul mod 2^64
    return out


def _tokenize_group_by_len(phrases: List[str], tok: AutoTokenizer) -> Tuple[Dict[int, List[Tuple[str, np.ndarray]]], int]:
    by_len: Dict[int, List[Tuple[str, np.ndarray]]] = {}
    Lmax = 0
    for p in phrases:
        ids = tok.encode(p, add_special_tokens=False)
        arr = np.asarray(ids, dtype=np.uint32)
        if arr.size == 0:
            raise ValueError(f"Phrase tokenized to empty sequence: {p!r}")
        by_len.setdefault(arr.size, []).append((p, arr))
        Lmax = max(Lmax, arr.size)
    return by_len, Lmax


def _choose_dtype(bin_bytes: int, vocab_size: int) -> np.dtype:
    # Prefer u16 if possible; fall back to u32
    if vocab_size <= 65536 and (bin_bytes % 2 == 0):
        return np.uint16
    return np.uint32


# ---------------------------
# Main runner
# ---------------------------

class ShardwiseStreamer:
    """
    Stream or (optionally) index-per-shard over an HF dataset of token IDs.

    Workflow:
      - process_one_shard(download_dir): streaming
      - process_one_shard_indexed(download_dir, idx_dir): build index for this shard,
        answer queries, then delete idx+bin

    State is persisted after each shard under `state_dir`, so you can resume safely.
    """

    def __init__(
        self,
        repo_id: str,
        phrases: List[str],
        steps: Sequence[int],
        state_dir: Path,
        tokenizer_name: str,
        count_mode: str = "full",      # "full" (end-based) or "start" (start-based)
        chunk_tokens: int = 3_000_000, # streaming chunk size (tokens)
        tokens_per_step: int = 2_097_152,
    ) -> None:
        self.repo_id = repo_id
        self.tok = AutoTokenizer.from_pretrained(tokenizer_name)
        self.phrases = list(phrases)
        self.steps = np.asarray(steps, dtype=np.int64)
        self.cutoffs = self.steps * np.int64(tokens_per_step)
        self.count_mode = count_mode
        self.chunk_tokens = int(chunk_tokens)
        self.state_dir = state_dir

        self.by_len, self.Lmax = _tokenize_group_by_len(self.phrases, self.tok)
        self._build_hash_tables()
        self._load_or_init_state()

    # ---------- state ----------
    def _state_paths(self):
        return {
            "meta": self.state_dir / "meta.json",
            "counts": self.state_dir / "counts.npy",
            "tail": self.state_dir / "prev_tail.npy",
            "done": self.state_dir / "done_shards.json",
            "bins": self.state_dir / "remote_bins.json",
        }

    def _load_or_init_state(self):
        p = self._state_paths()
        self.state_dir.mkdir(parents=True, exist_ok=True)

        if p["meta"].exists():
            meta = json.loads(p["meta"].read_text())
            assert meta["count_mode"] == self.count_mode, "count_mode mismatch with saved state"
            assert meta["tokenizer"] == self.tok.name_or_path, "tokenizer mismatch with saved state"
            assert np.array_equal(np.array(meta["steps"], dtype=np.int64), self.steps), "steps mismatch with saved state"
            self.cutoffs = np.asarray(meta["cutoffs"], dtype=np.int64)
            self.global_offset = int(meta["global_offset"])
            self.next_idx = int(meta["next_idx"])
            self.total_shards = int(meta["total_shards"])
            # load bin list
            self.remote_bins = json.loads(p["bins"].read_text())
            # load arrays
            self.counts = np.load(p["counts"])
            self.prev_tail = np.load(p["tail"]) if p["tail"].exists() else np.array([], dtype=np.uint64)
            self.done = set(json.loads(p["done"].read_text()))
        else:
            # discover remote shard list once
            tree = list_repo_tree(self.repo_id, repo_type="dataset")
            bins = sorted([n.path for n in tree if n.path.endswith(".bin")])
            self.remote_bins = bins
            self.total_shards = len(bins)
            self.global_offset = 0
            self.next_idx = 0
            self.prev_tail = np.array([], dtype=np.uint64)
            self.done = set()

            meta = {
                "repo_id": self.repo_id,
                "tokenizer": self.tok.name_or_path,
                "count_mode": self.count_mode,
                "steps": self.steps.tolist(),
                "cutoffs": self.cutoffs.astype(np.int64).tolist(),
                "global_offset": self.global_offset,
                "next_idx": self.next_idx,
                "total_shards": self.total_shards,
                "phrases": self.phrases,
            }
            p["bins"].write_text(json.dumps(self.remote_bins, indent=2))
            p["meta"].write_text(json.dumps(meta, indent=2))
            self.counts = np.zeros((len(self.phrases), self.cutoffs.size), dtype=np.int64)
            np.save(p["counts"], self.counts)
            p["done"].write_text(json.dumps(sorted(self.done)))

    def _save_state(self):
        p = self._state_paths()
        meta = json.loads(p["meta"].read_text())
        meta.update({
            "global_offset": int(self.global_offset),
            "next_idx": int(self.next_idx),
        })
        p["meta"].write_text(json.dumps(meta, indent=2))
        np.save(p["counts"], self.counts)
        np.save(p["tail"], self.prev_tail.astype(np.uint64))
        p["done"].write_text(json.dumps(sorted(self.done)))

    # ---------- hashing ----------
    def _build_hash_tables(self):
        BASE = 1465329397  # arbitrary odd base
        self.per_len: Dict[int, Tuple[np.ndarray, Dict[int, List[Tuple[str, np.ndarray]]]]] = {}
        for L, plist in self.by_len.items():
            pv = _pow_vec(BASE, L)
            table: Dict[int, List[Tuple[str, np.ndarray]]] = {}
            # compute phrase hashes with overflow suppressed (we do math mod 2^64)
            with np.errstate(over="ignore"):
                for phrase, t in plist:
                    h = int((pv.astype(np.uint64) * t.astype(np.uint64)).sum(dtype=np.uint64))
                    table.setdefault(h, []).append((phrase, t))
            self.per_len[L] = (pv, table)

    # ---------- streaming of ONE local shard ----------
    def _process_local_bin(self, bin_path: Path):
        """
        Streaming path: scan the shard with rolling hash + exact check, update counts.
        """
        M = self.cutoffs.size
        include_from = {p: np.zeros(M, dtype=np.int64) for p in self.phrases}
        BASE_POS0 = self.global_offset

        n_bytes = bin_path.stat().st_size
        dtype = _choose_dtype(n_bytes, self.tok.vocab_size)
        x = np.memmap(bin_path, dtype=dtype, mode="r")
        N = int(x.size)

        step = max(self.chunk_tokens, self.Lmax + 1)
        start = 0

        while start < N:
            end = min(N, start + step)
            s_rel = start - (self.Lmax - 1) if start > 0 else start
            chunk = np.asarray(x[max(0, s_rel):end])
            base_pos = BASE_POS0 + max(0, s_rel)

            if start == 0 and self.prev_tail.size > 0:
                need = self.Lmax - 1
                tail = self.prev_tail[-need:] if self.prev_tail.size >= need else self.prev_tail
                if tail.size > 0:
                    chunk = np.concatenate([tail.view(np.uint64), chunk.astype(np.uint64, copy=False)])
                    base_pos = BASE_POS0 - tail.size

            cu64 = chunk.astype(np.uint64, copy=False)

            for L, (pv, table) in self.per_len.items():
                if cu64.size < L:
                    continue
                with np.errstate(over="ignore"):
                    wins = np.lib.stride_tricks.sliding_window_view(cu64, L)
                    H = (wins * pv).sum(axis=1, dtype=np.uint64)

                for hkey, plist in table.items():
                    cand = np.nonzero(H == np.uint64(hkey))[0]
                    if cand.size == 0:
                        continue
                    Wc = wins[cand]
                    for phrase, toks in plist:
                        eq = np.all(Wc == toks.astype(np.uint64), axis=1)
                        hits_local = cand[eq]
                        if hits_local.size == 0:
                            continue

                        starts = base_pos + hits_local.astype(np.int64)
                        ends = starts + L

                        if self.count_mode == "full":
                            ref = ends
                            mask = ends > BASE_POS0     # drop windows fully in prior tail
                            side = "left"               # end <= cutoff
                        else:
                            ref = starts
                            mask = ends > BASE_POS0     # allow cross-shard starts
                            side = "right"              # start < cutoff (strict)

                        if np.any(mask):
                            refv = ref[mask]
                            i_min = np.searchsorted(self.cutoffs, refv, side=side)
                            in_range = i_min < M
                            if np.any(in_range):
                                np.add.at(include_from[phrase], i_min[in_range], 1)

            start = end

        # prepare state for next shard
        if self.Lmax > 1:
            tail_len = min(self.Lmax - 1, N)
            self.prev_tail = np.asarray(x[N - tail_len : N], dtype=dtype).astype(np.uint64, copy=False)
        else:
            self.prev_tail = np.array([], dtype=np.uint64)

        self.global_offset += N

        # turn include_from -> delta counts and add to totals
        for idx, phrase in enumerate(self.phrases):
            self.counts[idx, :] += include_from[phrase].cumsum()

    # ---------- optional: indexed per-shard processing ----------
    def _process_local_bin_indexed(self, bin_path: Path, idx_dir: Path):
        """
        Build a per-shard tokengrams MemmapIndex, answer queries, add cross-boundary
        windows, update counts, then delete the shard index.
        """
        try:
            from tokengrams import MemmapIndex  # optional dependency
        except Exception as e:
            raise RuntimeError("tokengrams is not installed; install it to use the indexed path.") from e

        idx_dir.mkdir(parents=True, exist_ok=True)
        idx_path = idx_dir / (bin_path.name + ".tkg.idx")

        # Build (idempotent) and open the index
        MemmapIndex.build(str(bin_path), str(idx_path), vocab=int(self.tok.vocab_size), verbose=False)
        idx = MemmapIndex(str(bin_path), str(idx_path), vocab=int(self.tok.vocab_size))

        M = self.cutoffs.size
        include_from = {p: np.zeros(M, dtype=np.int64) for p in self.phrases}
        BASE_POS0 = self.global_offset

        n_bytes = bin_path.stat().st_size
        dtype = _choose_dtype(n_bytes, self.tok.vocab_size)
        x = np.memmap(bin_path, dtype=dtype, mode="r")
        N = int(x.size)

        # In-shard matches via index
        for L, plist in self.by_len.items():
            for phrase, toks_np in plist:
                pos_local = np.asarray(idx.positions(toks_np.tolist()), dtype=np.int64)
                if pos_local.size == 0:
                    continue
                starts = BASE_POS0 + pos_local
                ends = starts + L

                if self.count_mode == "full":
                    ref, side = ends, "left"     # end <= cutoff
                else:
                    ref, side = starts, "right"  # start < cutoff (strict)

                i_min = np.searchsorted(self.cutoffs, ref, side=side)
                in_range = i_min < M
                if np.any(in_range):
                    np.add.at(include_from[phrase], i_min[in_range], 1)

        # Cross-shard windows (prev tail + current head)
        if self.Lmax > 1:
            tail_need = min(self.Lmax - 1, self.prev_tail.size)
            head_need = min(self.Lmax - 1, N)
            if head_need > 0:
                head = np.asarray(x[:head_need], dtype=dtype).astype(np.uint64, copy=False)
                tail = self.prev_tail[-tail_need:] if tail_need > 0 else np.array([], dtype=np.uint64)
                buf = np.concatenate([tail.view(np.uint64), head]) if (tail.size or head.size) else np.array([], dtype=np.uint64)
                if buf.size >= 1:
                    base_buf = BASE_POS0 - tail_need
                    for L, plist in self.by_len.items():
                        if buf.size < L or tail_need == 0:
                            continue
                        wins = np.lib.stride_tricks.sliding_window_view(buf, L)
                        for phrase, toks_np in plist:
                            t64 = toks_np.astype(np.uint64, copy=False)
                            eq = np.all(wins == t64, axis=1)
                            if not np.any(eq):
                                continue
                            starts = base_buf + np.nonzero(eq)[0].astype(np.int64)
                            ends = starts + L
                            mask_x = (starts < BASE_POS0) & (ends > BASE_POS0)
                            if not np.any(mask_x):
                                continue
                            if self.count_mode == "full":
                                ref, side = ends[mask_x], "left"
                            else:
                                ref, side = starts[mask_x], "right"
                            i_min = np.searchsorted(self.cutoffs, ref, side=side)
                            in_range = i_min < M
                            if np.any(in_range):
                                np.add.at(include_from[phrase], i_min[in_range], 1)

        # finalize
        for i, phrase in enumerate(self.phrases):
            self.counts[i, :] += include_from[phrase].cumsum()

        if self.Lmax > 1:
            tlen = min(self.Lmax - 1, N)
            self.prev_tail = np.asarray(x[N - tlen : N], dtype=dtype).astype(np.uint64, copy=False)
        else:
            self.prev_tail = np.array([], dtype=np.uint64)

        self.global_offset += N

        # Delete the shard index to reclaim space
        try:
            os.remove(idx_path)
        except Exception:
            pass

    # ---------- public API ----------
    def shards_total(self) -> int:
        return self.total_shards

    def next_shard_index(self) -> int:
        return self.next_idx

    def next_remote_bin(self) -> Optional[str]:
        if self.next_idx >= self.total_shards:
            return None
        return self.remote_bins[self.next_idx]

    def process_one_shard(self, download_dir: Path) -> Optional[str]:
        """
        Streaming path: download next shard, stream-count, persist, delete shard.
        Returns the processed filename (or None if finished).
        """
        rel = self.next_remote_bin()
        if rel is None:
            return None
        local_path = hf_hub_download(
            repo_id=self.repo_id,
            repo_type="dataset",
            filename=rel,
            local_dir=str(download_dir),
            local_dir_use_symlinks=False,
        )
        local_bin = Path(local_path)
        print(f"[downloaded] shard {self.next_idx+1}/{self.total_shards}: {local_bin.name}")

        self._process_local_bin(local_bin)

        self.done.add(rel)
        self.next_idx += 1
        self._save_state()
        try:
            local_bin.unlink()
            print(f"[deleted ] {local_bin.name}")
        except Exception as e:
            print(f"[warn] could not delete {local_bin}: {e}")
        return rel

    def process_one_shard_indexed(self, download_dir: Path, idx_dir: Path) -> Optional[str]:
        """
        Indexed path: download next shard, build per-shard index, query, persist, delete idx+bin.
        Returns the processed filename (or None if finished).
        """
        rel = self.next_remote_bin()
        if rel is None:
            return None
        local_path = hf_hub_download(
            repo_id=self.repo_id,
            repo_type="dataset",
            filename=rel,
            local_dir=str(download_dir),
            local_dir_use_symlinks=False,
        )
        local_bin = Path(local_path)
        print(f"[downloaded] shard {self.next_idx+1}/{self.total_shards}: {local_bin.name}")

        self._process_local_bin_indexed(local_bin, idx_dir=idx_dir)

        self.done.add(rel)
        self.next_idx += 1
        self._save_state()
        try:
            local_bin.unlink()
            print(f"[deleted ] {local_bin.name}")
        except Exception as e:
            print(f"[warn] could not delete {local_bin}: {e}")
        return rel


# ---------------------------
# Example usage (commented)
# ---------------------------
#
# if __name__ == "__main__":
#     REPO_ID = "EleutherAI/pile-deduped-pythia-preshuffled"
#     TOKENIZER_NAME = "EleutherAI/pythia-70m"
#     LOCAL_SCRATCH = Path("./shard_scratch")
#     STATE_DIR = Path("./stream_state")
#
#     phrases = [" United States", " in Washington", " Paris", " capital of France"]
#     steps_all = canonical_pythia_steps(include_early=True)
#
#     runner = ShardwiseStreamer(
#         repo_id=REPO_ID,
#         phrases=phrases,
#         steps=steps_all,
#         state_dir=STATE_DIR,
#         tokenizer_name=TOKENIZER_NAME,
#         count_mode="full",
#         chunk_tokens=3_000_000,
#         tokens_per_step=2_097_152,
#     )
#
#     print("Total shards:", runner.shards_total())
#     print("Next shard idx:", runner.next_shard_index())
#
#     # Process one shard via streaming:
#     LOCAL_SCRATCH.mkdir(exist_ok=True)
#     done = runner.process_one_shard(LOCAL_SCRATCH)
#     print("Processed:", done)
#
#     # Or, process one shard via per-shard indexing (requires `pip install tokengrams`):
#     # IDX_TMP = Path("./idx_scratch"); IDX_TMP.mkdir(exist_ok=True)
#     # done = runner.process_one_shard_indexed(LOCAL_SCRATCH, IDX_TMP)
#     # print("Processed:", done)
#
#     # When finished, your cumulative counts are at:
#     # counts = np.load(STATE_DIR / "counts.npy")  # shape [len(phrases), 154]
