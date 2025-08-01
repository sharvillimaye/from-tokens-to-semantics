import pytest
import numpy as np
from ngrams.interface import NGramIndex, batch_counts

class DummyIndex(NGramIndex):
    """
    Dummy backend for testing. Returns pre-defined positions for given n-grams.
    """
    def __init__(self, pos_map):
        # pos_map: dict mapping ngram tuples to numpy arrays of positions
        self.pos_map = pos_map

    def build_index(self, token_stream, **kwargs):
        # no-op for dummy
        pass

    def positions(self, ngram):
        return self.pos_map.get(ngram, np.array([], dtype=int))


def test_batch_counts_single_ngram_single_checkpoint():
    # Single unigram appearing at positions [0, 2, 4]
    pos_map = { (1,): np.array([0, 2, 4]) }
    index = DummyIndex(pos_map)

    ngrams = [(1,)]
    cutoffs = np.array([3])  # cutoff at token index 3
    results = batch_counts(ngrams, cutoffs, index)

    # positions <= (3 - (1-1)) == 3: count should be 2
    assert results[(1,)].shape == (1,)
    assert results[(1,)][0] == 2


def test_batch_counts_multiple_ngrams_multiple_checkpoints():
    # Unigram (1) at [1,3,5], bigram (2,3) at [0,4]
    pos_map = {
        (1,): np.array([1, 3, 5]),
        (2,3): np.array([0, 4])
    }
    index = DummyIndex(pos_map)

    ngrams = [(1,), (2,3)]
    # Two checkpoints: cutoff 2 and 5
    cutoffs = np.array([2, 5])
    results = batch_counts(ngrams, cutoffs, index)

    # For (1,), positions <=2: count=1; <=5: count=3
    assert np.array_equal(results[(1,)], np.array([1, 3]))
    # For (2,3), bigram length=2 -> end_limit = cutoff -1
    # cutoffs [2,5] => end_limits [1,4]
    # positions [0,4] -> counts [1,1]
    assert np.array_equal(results[(2,3)], np.array([1, 2]))


def test_batch_counts_empty_positions():
    # N-gram not present
    pos_map = {}
    index = DummyIndex(pos_map)

    ngrams = [(7,), (8,9)]
    cutoffs = np.array([10, 20])
    results = batch_counts(ngrams, cutoffs, index)

    assert np.array_equal(results[(7,)], np.array([0, 0]))
    assert np.array_equal(results[(8,9)], np.array([0, 0]))