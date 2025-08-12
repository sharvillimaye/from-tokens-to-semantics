import numpy as np
import pytest

from ngrams.backends.inmemory import InMemoryIndex
from ngrams.interface import batch_counts


def test_inmemory_build_and_positions_basic():
    tokens = [1, 2, 1, 2, 3, 1]
    idx = InMemoryIndex()
    idx.build_index(tokens, max_n=3)

    # Unigram (1) at positions 0, 2, 5
    assert np.array_equal(idx.positions((1,)), np.array([0, 2, 5]))
    # Bigram (1,2) at positions 0, 2
    assert np.array_equal(idx.positions((1, 2)), np.array([0, 2]))
    # Trigram (1,2,3) at position 2
    assert np.array_equal(idx.positions((1, 2, 3)), np.array([2]))
    # Missing n-gram
    assert idx.positions((9,)).size == 0


def test_inmemory_include_only_restricts_index():
    tokens = [4, 4, 4, 5]
    idx = InMemoryIndex()
    idx.build_index(tokens, include_only=[(4,), (4, 4)])

    # Present (restricted)
    assert np.array_equal(idx.positions((4,)), np.array([0, 1, 2]))
    assert np.array_equal(idx.positions((4, 4)), np.array([0, 1]))
    # Absent due to restriction
    assert idx.positions((5,)).size == 0
    assert idx.positions((4, 5)).size == 0


def test_inmemory_with_batch_counts_multi_checkpoints():
    tokens = [1, 2, 1, 2, 3, 1]
    idx = InMemoryIndex()
    idx.build_index(tokens, max_n=3)

    ngrams = [(1,), (1, 2), (1, 2, 3)]
    # Suppose we want counts up to token cutoffs 2, 4, 6
    cutoffs = np.array([2, 4, 6])
    result = batch_counts(ngrams, cutoffs, idx)

    # Unigram (1): positions [0,2,5]; end_limits = [2,4,6]
    # counts: [2,2,3]
    assert np.array_equal(result[(1,)], np.array([2, 2, 3]))

    # Bigram (1,2): positions [0,2]; end_limits = [1,3,5]
    # counts: [1,2,2]
    assert np.array_equal(result[(1, 2)], np.array([1, 2, 2]))

    # Trigram (1,2,3): positions [2]; end_limits = [0,2,4]
    # counts: [0,1,1]
    assert np.array_equal(result[(1, 2, 3)], np.array([0, 1, 1]))


def test_positions_requires_build():
    idx = InMemoryIndex()
    with pytest.raises(RuntimeError):
        _ = idx.positions((1,))

def test_batch_counts_with_token_counts_flag():
    # tokens: [1,2,1,2,3,1] => length T=6
    # token-count cutoffs: [2,4,6]  (not indices)
    idx = InMemoryIndex(); idx.build_index([1,2,1,2,3,1])
    res = batch_counts([(1,), (1,2), (1,2,3)],
                       np.array([2,4,6]),
                       idx,
                       cutoffs_are_token_counts=True)
    assert np.array_equal(res[(1,)],     [1,2,3])
    assert np.array_equal(res[(1,2)],    [1,2,2])
    assert np.array_equal(res[(1,2,3)],  [0,0,1])