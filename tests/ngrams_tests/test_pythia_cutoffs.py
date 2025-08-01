import numpy as np
import pytest

from ngrams.parsers.pythia_cutoffs import (
    canonical_pythia_steps,
    canonical_pythia_cutoffs,
    steps_from_hf_revisions,
    cutoffs_from_steps,
    PYTHIA_TOKENS_PER_STEP,
    sanity_check_main_total,
)

def test_canonical_steps_includes_early_and_regular():
    steps = canonical_pythia_steps(include_early=True)
    # 11 early (excluding 1000) + 143 regular (1000..143000) = 154 total
    assert steps.dtype == np.int64
    assert steps.shape == (154,)
    assert steps[0] == 0
    assert 512 in steps
    assert 1000 in steps
    assert steps[-1] == 143_000


def test_canonical_steps_without_early():
    steps = canonical_pythia_steps(include_early=False)
    assert steps.shape == (143,)
    assert steps.min() == 1000
    assert steps.max() == 143_000
    # evenly spaced by 1000
    diffs = np.diff(steps)
    assert np.all(diffs == 1000)


def test_canonical_cutoffs_values():
    steps, cutoffs = canonical_pythia_cutoffs(include_early=True)
    # spot check a few known values
    i_1000 = np.where(steps == 1000)[0][0]
    assert cutoffs[i_1000] == 1000 * PYTHIA_TOKENS_PER_STEP
    i_143000 = np.where(steps == 143_000)[0][0]
    assert cutoffs[i_143000] == 143_000 * PYTHIA_TOKENS_PER_STEP
    # total tokens constant used in docs
    assert int(cutoffs[i_143000]) == 143_000 * 2_097_152  # 299,892,736,000


def test_sanity_check_main_total_passes():
    steps, cutoffs = canonical_pythia_cutoffs(include_early=True)
    # should not raise
    sanity_check_main_total(steps, cutoffs)


def test_steps_from_hf_revisions_parsing():
    names = ["step1", "step0001", "step512", "step1000", "foo", "step143000"]
    steps = steps_from_hf_revisions(names)
    # duplicates collapsed; sorted ascending
    assert np.array_equal(steps[:3], np.array([1, 512, 1000], dtype=np.int64))
    assert steps[-1] == 143_000


def test_cutoffs_from_steps_validation_and_values():
    steps = [0, 1, 1000, 143_000]
    cutoffs = cutoffs_from_steps(steps)
    assert np.array_equal(
        cutoffs,
        np.array([0, 1, 1000, 143_000], dtype=np.int64) * np.int64(PYTHIA_TOKENS_PER_STEP),
    )

    # invalid: non-1D input
    with pytest.raises(ValueError):
        cutoffs_from_steps(np.array([[0, 1]]))
    # invalid: tokens_per_step <= 0
    with pytest.raises(ValueError):
        cutoffs_from_steps(steps, tokens_per_step=0)