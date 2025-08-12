import os
import numpy as np
import pytest

from ngrams.backends.tokengrams import TokengramsIndex
from ngrams.interface import batch_counts

tokengrams = pytest.importorskip("tokengrams")

def _write_tokens_to_bin(tokens, path, vocab):
    dtype = np.uint16 if vocab <= 2**16 else np.uint32
    np.asarray(tokens, dtype=dtype).tofile(path)

def test_memmap_build_and_positions_roundtrip(tmp_path: pytest.TempPathFactory):
    # Tiny token stream
    tokens = [1, 2, 1, 2, 3, 1]
    vocab = 100

    bin_path = os.path.join(tmp_path, "tiny.bin")
    idx_path = os.path.join(tmp_path, "tiny.idx")
    _write_tokens_to_bin(tokens, bin_path, vocab)

    idx = TokengramsIndex()
    idx.build_index(corpus_path=bin_path, index_path=idx_path, vocab=vocab, verbose=False)

    # Check positions
    assert np.array_equal(idx.positions((1,)), np.array([0, 2, 5]))
    assert np.array_equal(idx.positions((1, 2)), np.array([0, 2]))
    assert np.array_equal(idx.positions((1, 2, 3)), np.array([2]))
    assert idx.positions((9,)).size == 0

def test_memmap_with_batch_counts(tmp_path: pytest.TempPathFactory):
    tokens = [1, 2, 1, 2, 3, 1]
    vocab = 100

    bin_path = os.path.join(tmp_path, "tiny.bin")
    idx_path = os.path.join(tmp_path, "tiny.idx")
    _write_tokens_to_bin(tokens, bin_path, vocab)

    idx = TokengramsIndex()
    idx.build_index(corpus_path=bin_path, index_path=idx_path, vocab=vocab, verbose=False)

    ngrams = [(1,), (1, 2), (1, 2, 3)]
    cutoffs = np.array([2, 4, 6])
    result = batch_counts(ngrams, cutoffs, idx)

    assert np.array_equal(result[(1,)], np.array([2, 2, 3]))
    assert np.array_equal(result[(1, 2)], np.array([1, 2, 2]))
    assert np.array_equal(result[(1, 2, 3)], np.array([0, 1, 1]))

def test_build_from_token_stream(tmp_path: pytest.TempPathFactory):
    tokens = [7, 7, 7, 8]
    vocab = 100

    idx = TokengramsIndex()
    idx.build_index(token_stream=tokens, vocab=vocab, work_dir=str(tmp_path))

    # Positions for (7) and (7,7)
    assert np.array_equal(idx.positions((7,)), np.array([0, 1, 2]))
    assert np.array_equal(idx.positions((7, 7)), np.array([0, 1]))