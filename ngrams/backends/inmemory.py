from __future__ import annotations
from typing import Dict, Iterable, List, Optional, Tuple
import numpy as np

from ngrams.interface import NGramIndex, NGram

class InMemoryIndex(NGramIndex):
    """
    A simple in-memory n-gram index for 1–3 token n-grams.

    Intended for unit tests, examples, and small corpora. Builds a dictionary
    mapping n-gram tuples -> sorted numpy array of start positions.
    """

    def __init__(self) -> None:
        self._pos_map: Dict[NGram, np.ndarray] = {}
        self._built: bool = False

    def build_index(
        self,
        token_stream: List[int],
        *,
        max_n: int = 3,
        include_only: Optional[Iterable[NGram]] = None,
        dtype: np.dtype = np.int64,
    ) -> None:
        """
        Build the index from a list of token IDs.

        Args:
            token_stream: sequence of token IDs in exact training order.
            max_n: largest n-gram length to index (1..3 supported).
            include_only: optional iterable of n-grams to index; if provided,
                only these n-grams are stored (useful to keep memory small).
            dtype: dtype for stored position arrays.
        """
        if max_n < 1 or max_n > 3:
            raise ValueError("max_n must be in {1,2,3}")

        # Normalize include_only to a set for O(1) membership checks
        restrict: Optional[set] = set(map(tuple, include_only)) if include_only is not None else None

        # Temporary lists for positions; convert to arrays at the end
        tmp: Dict[NGram, List[int]] = {}
        T = len(token_stream)

        # Unigrams
        for i in range(T):
            key = (token_stream[i],)
            if restrict is None or key in restrict:
                tmp.setdefault(key, []).append(i)

        if max_n >= 2 and T >= 2:
            for i in range(T - 1):
                key = (token_stream[i], token_stream[i + 1])
                if restrict is None or key in restrict:
                    tmp.setdefault(key, []).append(i)

        if max_n >= 3 and T >= 3:
            for i in range(T - 2):
                key = (token_stream[i], token_stream[i + 1], token_stream[i + 2])
                if restrict is None or key in restrict:
                    tmp.setdefault(key, []).append(i)

        # Freeze to numpy arrays (already sorted by construction)
        self._pos_map = {ng: np.asarray(pos, dtype=dtype) for ng, pos in tmp.items()}
        self._built = True

    def positions(self, ngram: NGram) -> np.ndarray:
        if not self._built:
            raise RuntimeError("InMemoryIndex not built; call build_index() first")
        return self._pos_map.get(tuple(ngram), np.asarray([], dtype=np.int64))
