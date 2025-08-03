"""
Canonical Pythia checkpoint → token cutoff mapping.

Facts from the official Pythia docs:
- Models expose 154 checkpoints: steps {0,1,2,4,8,16,32,64,128,256,512,1000} plus every 1,000 up to 143,000.
- Effective tokens per step = 2,097,152 (2^21) for the unified step naming on Hugging Face.
- Thus token cutoff at step s is s * 2_097_152 tokens.

These conventions apply to the v1 suite and to v0 checkpoints as renamed on HF for
consistency (see Pythia README notes). If you parse step names from HF revisions (e.g.,
"step3000"), you can recover the same cutoffs via the helpers below.
"""
from __future__ import annotations
from typing import Iterable, List, Sequence, Tuple
import re
import numpy as np

# Constant per Pythia README / Hugging Face model cards
PYTHIA_TOKENS_PER_STEP: int = 2_097_152  # 2^21

# The 11 early checkpoints before the regular 1k interval, plus step 0
_EARLY_STEPS: Tuple[int, ...] = (0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000)


def canonical_pythia_steps(include_early: bool = True) -> np.ndarray:
    """
    Return the canonical list of Pythia checkpoint steps as a 1-D array (int64).

    Args:
        include_early: whether to include the early steps
            {0,1,2,4,8,16,32,64,128,256,512,1000}.

    Returns:
        np.ndarray[int64] of shape (154,) when include_early=True.
    """
    regular = np.arange(1000, 143_001, 1000, dtype=np.int64)  # 1000..143000 inclusive
    if include_early:
        early = np.array(_EARLY_STEPS, dtype=np.int64)
        # avoid double-counting 1000
        early_mask = early != 1000
        steps = np.concatenate([early[early_mask], regular])
    else:
        steps = regular
    return steps


def canonical_pythia_cutoffs(include_early: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (steps, token_cutoffs) using the canonical schedule and tokens/step.
    """
    steps = canonical_pythia_steps(include_early=include_early)
    cutoffs = steps * np.int64(PYTHIA_TOKENS_PER_STEP)
    return steps, cutoffs


_STEP_RE = re.compile(r"^step(?P<num>\d+)$")


def steps_from_hf_revisions(names: Iterable[str]) -> np.ndarray:
    """
    Parse a collection of Hugging Face checkpoint revision names like "step3000"
    into a sorted int64 array of steps.
    """
    steps: List[int] = []
    for name in names:
        m = _STEP_RE.match(name.strip())
        if m:
            steps.append(int(m.group("num")))
    steps_arr = np.array(sorted(set(steps)), dtype=np.int64)
    return steps_arr


def cutoffs_from_steps(steps: Sequence[int], *, tokens_per_step: int = PYTHIA_TOKENS_PER_STEP) -> np.ndarray:
    """
    Compute token cutoffs from given steps and a tokens-per-step factor (defaults to Pythia's 2,097,152).
    """
    s = np.asarray(steps, dtype=np.int64)
    if s.ndim != 1:
        raise ValueError("steps must be 1-D")
    if tokens_per_step <= 0:
        raise ValueError("tokens_per_step must be > 0")
    return s * np.int64(tokens_per_step)


def sanity_check_main_total(steps: np.ndarray, cutoffs: np.ndarray) -> None:
    """
    Assert that the final checkpoint equals 143000 steps and total tokens == 299,892,736,000.
    Raises AssertionError if the expectation does not hold.
    """
    assert steps.max() == 143_000, f"unexpected max step: {steps.max()}"
    expected_total = np.int64(143_000) * np.int64(PYTHIA_TOKENS_PER_STEP)
    assert cutoffs[steps.argmax()] == expected_total, (
        f"unexpected final tokens: {cutoffs[steps.argmax()]} vs {expected_total}"
    )
