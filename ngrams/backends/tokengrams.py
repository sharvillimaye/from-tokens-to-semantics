from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple
import os
import tempfile
import numpy as np

from ngrams.interface import NGramIndex, NGram

@dataclass
class _MemmapConfig:
    corpus_path: str
    index_path: str
    vocab: int
    verbose: bool = False

@dataclass
class _ShardedConfig:
    shards: Sequence[Tuple[str, str]]  # (bin_path, idx_path) pairs
    vocab: int
    verbose: bool = False

class TokengramsIndex(NGramIndex):
    """
    Production backend using EleutherAI's `tokengrams` Python bindings.

    Supports building either a single-file `MemmapIndex` or a `ShardedMemmapIndex`.
    After build (or load), `positions(ngram)` returns a sorted NumPy array of start
    positions for the n-gram.

    Notes
    -----
    * Tokengrams builds indices from on-disk corpora of u16/u32 *token IDs*.
    * API surface used (from README):
        - MemmapIndex.build(corpus_bin, index_idx, vocab=..., verbose=True)
        - MemmapIndex(corpus_bin, index_idx, vocab=...)
        - ShardedMemmapIndex.build([(bin, idx), ...], vocab=..., verbose=True)
        - index.positions(list_of_token_ids)
    """

    def __init__(self) -> None:
        self._index = None  # MemmapIndex or ShardedMemmapIndex
        self._built = False
        self._cfg_memmap: Optional[_MemmapConfig] = None
        self._cfg_sharded: Optional[_ShardedConfig] = None

    # ------------------------------------------------------------------
    # Building / Loading
    # ------------------------------------------------------------------
    def build_index(
        self,
        token_stream: Optional[List[int]] = None,
        *,
        corpus_path: Optional[str] = None,
        index_path: Optional[str] = None,
        shards: Optional[Sequence[Tuple[str, str]]] = None,
        vocab: Optional[int] = None,
        verbose: bool = False,
        work_dir: Optional[str] = None,
        load_only: bool = False,
    ) -> None:
        """
        Build or load a tokengrams index.

        Options (choose one):
          1) Sharded build: provide `shards=[(bin, idx), ...]` and `vocab`.
          2) Memmap build: provide `corpus_path`, `index_path`, and `vocab`.
          3) From in-memory tokens: provide `token_stream`, `vocab`, and `work_dir`.
             We'll write `tokens.bin` + `tokens.idx` under work_dir and build.

        If `load_only=True`, we skip building and just load an existing index
        (memmap mode requires `corpus_path`, `index_path`, `vocab`).
        """
        try:
            from tokengrams import MemmapIndex, ShardedMemmapIndex  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "tokengrams is not installed."
            ) from e

        if shards is not None:
            if vocab is None:
                raise ValueError("`vocab` is required for sharded build")
            self._cfg_sharded = _ShardedConfig(shards=list(shards), vocab=int(vocab), verbose=verbose)
            if load_only:
                # ShardedMemmapIndex currently exposes only a build path in the README; rely on build (idempotent)
                self._index = ShardedMemmapIndex.build(self._cfg_sharded.shards, vocab=self._cfg_sharded.vocab, verbose=verbose)
            else:
                self._index = ShardedMemmapIndex.build(self._cfg_sharded.shards, vocab=self._cfg_sharded.vocab, verbose=verbose)
            self._built = True
            return

        # Memmap path (single file)
        if token_stream is not None and corpus_path is None:
            if vocab is None:
                raise ValueError("When passing token_stream, you must supply `vocab`.")
            # Write tokens to a binary file (u16 if vocab <= 2**16 else u32)
            work_dir = work_dir or tempfile.mkdtemp(prefix="tokengrams_")
            os.makedirs(work_dir, exist_ok=True)
            corpus_path = os.path.join(work_dir, "tokens.bin")
            index_path = os.path.join(work_dir, "tokens.idx")
            dtype = np.uint16 if int(vocab) <= 2**16 else np.uint32
            np.asarray(token_stream, dtype=dtype).tofile(corpus_path)

        if corpus_path is None or index_path is None or vocab is None:
            raise ValueError(
                "Provide either (shards,vocab) or (corpus_path,index_path,vocab); or token_stream+vocab(+work_dir)."
            )

        self._cfg_memmap = _MemmapConfig(corpus_path=corpus_path, index_path=index_path, vocab=int(vocab), verbose=verbose)

        if load_only:
            self._index = MemmapIndex(self._cfg_memmap.corpus_path, self._cfg_memmap.index_path, vocab=self._cfg_memmap.vocab)
        else:
            self._index = MemmapIndex.build(
                self._cfg_memmap.corpus_path,
                self._cfg_memmap.index_path,
                vocab=self._cfg_memmap.vocab,
                verbose=verbose,
            )
        self._built = True

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------
    def positions(self, ngram: NGram) -> np.ndarray:
        if not self._built or self._index is None:
            raise RuntimeError("TokengramsIndex not built; call build_index() first")
        # # tokengrams API expects a python list of ints
        # pos = self._index.positions(list(ngram))
        # # Ensure numpy array (sorted order guaranteed by backend)
        # return np.asarray(pos, dtype=np.int64)

        arr = np.asarray(self._index.positions(list(ngram)), dtype=np.int64)
        if arr.size > 1 and not np.all(arr[:-1] <= arr[1:]):
            arr = np.sort(arr, kind="stable")
        return arr