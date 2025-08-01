from abc import ABC, abstractmethod
from typing import List, Tuple, Dict
import numpy as np

# Define a token n-gram as a tuple of token IDs (length 1 to 3)
NGram = Tuple[int, ...]

class NGramIndex(ABC):
    """
    Abstract interface for n-gram indexing backends.

    Backends must implement:
      - build_index(token_stream, **kwargs)
      - positions(ngram) -> np.ndarray of sorted start positions
    """

    @abstractmethod
    def build_index(self, token_stream: List[int], **kwargs) -> None:
        """
        Construct any on-disk or in-memory index from the full token stream.

        Args:
            token_stream: list of token IDs in training order.
            **kwargs: backend-specific options (shard size, index directory, etc.)
        """
        pass

    @abstractmethod
    def positions(self, ngram: NGram) -> np.ndarray:
        """
        Return a sorted numpy array of all start positions for the given token n-gram.

        Args:
            ngram: tuple of 1-3 token IDs.
        Returns:
            np.ndarray[int]: start indices where ngram occurs.
        """
        pass


def batch_counts(
    ngrams: List[NGram],
    cutoffs: np.ndarray,
    index: NGramIndex,
    batch_size: int = 5000
) -> Dict[NGram, np.ndarray]:
    """
    Compute cumulative counts for a list of n-grams across multiple checkpoint cutoffs.

    For each n-gram, fetch its occurrence positions once, then compute counts up to each cutoff
    by binary-searching the sorted positions array.

    Args:
        ngrams: list of token-ID tuples (1 to 3 IDs each).
        cutoffs: 1D array of token cutoff values (one per checkpoint).
        index: an NGramIndex backend instance (must be built already).
        batch_size: number of n-grams to process per micro-batch.

    Returns:
        dict mapping each n-gram to an array of counts (shape = cutoffs.shape).
    """
    results: Dict[NGram, np.ndarray] = {}
    num_checkpoints = cutoffs.shape[0]

    for start in range(0, len(ngrams), batch_size):
        batch = ngrams[start : start + batch_size]
        # Fetch positions for each n-gram in the batch
        pos_map: Dict[NGram, np.ndarray] = {ng: index.positions(ng) for ng in batch}

        # Compute counts for each checkpoint using vectorized search
        for ng, positions in pos_map.items():
            n = len(ng)
            # adjust cutoffs by (n-1) so end positions <= cutoff
            end_limits = cutoffs - (n - 1)
            # binary search: count of positions <= each end_limit
            counts = np.searchsorted(positions, end_limits, side="right")
            results[ng] = counts  # shape: (num_checkpoints,)

    return results