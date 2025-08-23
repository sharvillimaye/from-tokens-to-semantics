"""N-gram dataset utilities for polytope analysis."""

from __future__ import annotations
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from tokengrams import MemmapIndex
from transformers import AutoTokenizer
from tqdm import tqdm

# Constants per Pythia README / Hugging Face model cards
PYTHIA_TOKENS_PER_STEP: int = 2_097_152  # 2^21

# The 11 early checkpoints before the regular 1k interval, plus step 0
_EARLY_STEPS: Tuple[int, ...] = (0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000)

# Basic English stopwords for filtering
ENGLISH_STOPWORDS = {
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he', 'in', 
    'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to', 'was', 'will', 'with'
}


def load_indexes_from_directory(directory: Path) -> List[MemmapIndex]:
    """Load MemmapIndex objects from .bin/.idx file pairs in a directory."""
    bin_paths = list(directory.glob('*.bin'))
    indexes = []
    
    for bin_path in tqdm(bin_paths, desc="Loading indexes"):
        idx_path = bin_path.with_suffix('.idx')
        if idx_path.exists():
            index = MemmapIndex(str(bin_path), str(idx_path))
            indexes.append(index)
    
    return indexes


def canonical_pythia_steps(include_early: bool = True) -> np.ndarray:
    """Return the canonical list of Pythia checkpoint steps as a 1-D array (int64)."""
    regular = np.arange(1000, 143_001, 1000, dtype=np.int64)  # 1000..143000 inclusive
    if include_early:
        early = np.array(_EARLY_STEPS, dtype=np.int64)
        early_mask = early != 1000  # avoid double-counting 1000
        steps = np.concatenate([early[early_mask], regular])
    else:
        steps = regular
    return steps


def canonical_pythia_cutoffs(include_early: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Return (steps, token_cutoffs) using the canonical schedule and tokens/step."""
    steps = canonical_pythia_steps(include_early=include_early)
    cutoffs = steps * np.int64(PYTHIA_TOKENS_PER_STEP)
    return steps, cutoffs


def steps_from_hf_revisions(names: Iterable[str]) -> np.ndarray:
    """Parse HuggingFace checkpoint revision names like 'step3000' into sorted int64 array."""
    step_re = re.compile(r"^step(?P<num>\d+)$")
    steps: List[int] = []
    
    for name in names:
        m = step_re.match(name.strip())
        if m:
            steps.append(int(m.group("num")))
    
    return np.array(sorted(set(steps)), dtype=np.int64)


def cutoffs_from_steps(steps: Sequence[int], *, tokens_per_step: int = PYTHIA_TOKENS_PER_STEP) -> np.ndarray:
    """Compute token cutoffs from given steps and a tokens-per-step factor."""
    s = np.asarray(steps, dtype=np.int64)
    if s.ndim != 1:
        raise ValueError("steps must be 1-D")
    if tokens_per_step <= 0:
        raise ValueError("tokens_per_step must be > 0")
    return s * np.int64(tokens_per_step)


def resolve_index_offsets(
    indexes: List[MemmapIndex],
    shard_token_counts: Optional[List[int]] = None,
) -> List[int]:
    """Resolve per-index global token offsets."""
    # 1) Use provided counts
    if shard_token_counts is not None:
        offsets: List[int] = []
        acc = 0
        for cnt in shard_token_counts:
            offsets.append(acc)
            acc += int(cnt)
        return offsets

    # 2) Try attribute 'offset'
    offsets_attr: List[int] = []
    has_all_offsets = True
    for idx in indexes:
        off = getattr(idx, 'offset', None)
        if off is None:
            has_all_offsets = False
            break
        offsets_attr.append(int(off))
    if has_all_offsets:
        return offsets_attr

    # 3) Try len(index) to build prefix-sum
    try:
        lengths: List[int] = [int(len(idx)) for idx in indexes]
        offsets: List[int] = []
        acc = 0
        for cnt in lengths:
            offsets.append(acc)
            acc += cnt
        return offsets
    except Exception:
        raise ValueError(
            "Cannot resolve index offsets. Provide shard_token_counts or use indexes with 'offset' or __len__."
        )


def get_all_positions_local(indexes: List[MemmapIndex], ngram: str, tokenizer: AutoTokenizer) -> Dict[int, np.ndarray]:
    """Return per-index local start positions for an n-gram."""
    token_ids = tokenizer.encode(ngram, add_special_tokens=False)
    out: Dict[int, np.ndarray] = {}
    
    if not token_ids:
        return {i: np.array([], dtype=np.int64) for i in range(len(indexes))}
    
    for i, index in enumerate(indexes):
        out[i] = np.asarray(index.positions(token_ids), dtype=np.int64)
    
    return out


def get_all_positions_global(
    indexes: List[MemmapIndex],
    ngram: str,
    tokenizer: AutoTokenizer,
    index_offsets: Optional[List[int]] = None,
    shard_token_counts: Optional[List[int]] = None,
) -> List[Tuple[int, int]]:
    """Return global (start, end) token positions for an n-gram across all indexes."""
    token_ids = tokenizer.encode(ngram, add_special_tokens=False)
    if not token_ids:
        return []
    
    n = len(token_ids)
    
    # Resolve offsets
    if index_offsets is None:
        index_offsets = resolve_index_offsets(indexes, shard_token_counts)
    
    out: List[Tuple[int, int]] = []
    for idx_i, index in enumerate(indexes):
        local_starts = np.asarray(index.positions(token_ids), dtype=np.int64)
        if local_starts.size == 0:
            continue
        global_starts = local_starts + int(index_offsets[idx_i])
        out.extend([(int(s), int(s + n)) for s in global_starts])
    
    return out


def get_context_sentences(
    indexes: List[MemmapIndex],
    ngram: str,
    tokenizer: AutoTokenizer,
    positions_by_index: Dict[int, np.ndarray],
    *,
    index_offsets: Optional[List[int]] = None,
    context_window: int = 100,
) -> List[Dict[str, Any]]:
    """Collect context sentences for an n-gram at the given positions."""
    token_ids = tokenizer.encode(ngram, add_special_tokens=False)
    if not token_ids:
        return []
    
    # Optional global offsets
    if index_offsets is None:
        try:
            index_offsets = resolve_index_offsets(indexes)
        except Exception:
            index_offsets = [0 for _ in indexes]

    contexts: List[Dict[str, Any]] = []
    for i, index in enumerate(indexes):
        local_starts = np.asarray(positions_by_index.get(i, np.array([], dtype=np.int64)), dtype=np.int64)
        if local_starts.size == 0:
            continue
            
        for local_start in local_starts:
            try:
                ctx_token_ids = index.context(int(local_start), token_ids, window=context_window)
                sentence_text = tokenizer.decode(ctx_token_ids, skip_special_tokens=True)
                
                # Find character-offset positions within the sentence
                sent_lower = sentence_text.lower()
                ngram_lower = tokenizer.decode(token_ids, skip_special_tokens=True).lower()
                pos_list: List[Tuple[int, int]] = []
                search_from = 0
                while True:
                    j = sent_lower.find(ngram_lower, search_from)
                    if j == -1:
                        break
                    pos_list.append((j, j + len(ngram_lower)))
                    search_from = j + 1
                
                contexts.append({
                    'sentence': sentence_text,
                    'local_start': int(local_start),
                    'global_position': (
                        int(local_start) + int(index_offsets[i]), 
                        int(local_start) + int(index_offsets[i]) + len(token_ids)
                    ),
                    'index_idx': i,
                    'ngram_positions_in_sentence': pos_list,
                })
            except Exception:
                continue
                
    return contexts


def is_meaningful_ngram(ngram: str, original_words: List[str]) -> bool:
    """Filter out nonsensical n-grams using heuristic rules."""
    words = ngram.split()
    
    # Basic length checks
    if len(words) < 2 or len(words) > 8:
        return False
    
    if len(ngram) < 4 or len(ngram) > 100:
        return False
    
    # Must contain at least one word from original (semantic relatedness)
    if not any(word in original_words for word in words):
        return False
    
    # Check for too many stopwords
    stopword_count = sum(1 for word in words if word in ENGLISH_STOPWORDS)
    if stopword_count > len(words) * 0.7:  # More than 70% stopwords
        return False
    
    # Avoid repetitive patterns
    if len(set(words)) == 1:  # All same word
        return False
    
    # Check for minimum meaningful content
    meaningful_words = [w for w in words if w not in ENGLISH_STOPWORDS and len(w) > 2]
    if len(meaningful_words) < 1:
        return False
    
    # Avoid purely numeric or single character patterns
    if all(w.isdigit() or len(w) == 1 for w in words):
        return False
    
    # Check for reasonable character distribution
    alpha_chars = sum(1 for c in ngram if c.isalpha())
    if alpha_chars < len(ngram) * 0.5:  # Less than 50% alphabetic
        return False
    
    # Avoid patterns with too many repeated characters
    for word in words:
        if len(word) > 2:
            char_counts = Counter(word)
            if max(char_counts.values()) > len(word) * 0.6:
                return False
    
    return True


def get_cumulative_frequency_counts(
    indexes: List[MemmapIndex], 
    ngrams: List[str], 
    tokenizer: AutoTokenizer,
    shard_token_counts: Optional[List[int]] = None
) -> Dict[str, Dict[str, Any]]:
    """Calculate cumulative frequency counts for n-grams across all shards."""
    results = {}
    
    for ngram in ngrams:
        # Get positions across all shards
        positions_by_index = get_all_positions_local(indexes, ngram, tokenizer)
        
        # Calculate frequencies per shard
        shard_frequencies = []
        total_frequency = 0
        
        for i in range(len(indexes)):
            shard_freq = len(positions_by_index.get(i, np.array([], dtype=np.int64)))
            shard_frequencies.append(shard_freq)
            total_frequency += shard_freq
        
        results[ngram] = {
            'cumulative_frequency': total_frequency,
            'shard_frequencies': shard_frequencies,
            'positions_by_shard': {
                i: positions_by_index.get(i, np.array([], dtype=np.int64)).tolist() 
                for i in range(len(indexes))
            }
        }
    
    return results


def calculate_checkpoint_frequencies(
    indexes: List[MemmapIndex],
    ngrams: List[str],
    tokenizer: AutoTokenizer,
    shard_token_counts: Optional[List[int]] = None,
    include_early: bool = True
) -> Dict[str, Dict[str, Any]]:
    """Calculate checkpoint-wise frequencies for n-grams using Pythia cutoffs."""
    # Get Pythia checkpoint information
    pythia_steps, token_cutoffs = canonical_pythia_cutoffs(include_early=include_early)
    
    # Resolve shard token offsets for global positions
    try:
        index_offsets = resolve_index_offsets(indexes, shard_token_counts)
    except Exception:
        # Fallback: assume sequential shards
        if shard_token_counts:
            index_offsets = []
            acc = 0
            for count in shard_token_counts:
                index_offsets.append(acc)
                acc += count
        else:
            index_offsets = [0] * len(indexes)
    
    results = {}
    
    for ngram in ngrams:
        # Get all global positions for this n-gram
        global_positions = get_all_positions_global(
            indexes, ngram, tokenizer, 
            index_offsets=index_offsets,
            shard_token_counts=shard_token_counts
        )
        
        # Extract start positions and sort them
        start_positions = sorted([pos[0] for pos in global_positions])
        total_frequency = len(start_positions)
        
        # Calculate frequency up to each checkpoint
        checkpoint_frequencies = {}
        
        for step, cutoff in zip(pythia_steps, token_cutoffs):
            # Count positions that occur before this cutoff
            freq = sum(1 for pos in start_positions if pos < cutoff)
            checkpoint_frequencies[int(step)] = freq
        
        results[ngram] = {
            'cumulative_frequency': total_frequency,
            'checkpoint_frequencies': checkpoint_frequencies,
            'pythia_steps': pythia_steps.tolist(),
            'token_cutoffs': token_cutoffs.tolist(),
            'global_positions': start_positions[:100]  # Limit to first 100 for size
        }
    
    return results


def generate_ngram_analysis_json(
    indexes: List[MemmapIndex],
    ngrams: List[str],
    tokenizer: AutoTokenizer,
    shard_token_counts: Optional[List[int]] = None,
    include_early: bool = True,
    output_path: Optional[str] = None
) -> str:
    """Generate a comprehensive JSON analysis of n-grams across shards with Pythia checkpoints."""
    # Calculate cumulative frequencies
    cumulative_data = get_cumulative_frequency_counts(
        indexes, ngrams, tokenizer, shard_token_counts
    )
    
    # Calculate checkpoint-wise frequencies
    checkpoint_data = calculate_checkpoint_frequencies(
        indexes, ngrams, tokenizer, shard_token_counts, include_early
    )
    
    # Combine the data into a comprehensive structure
    analysis_data = {
        'metadata': {
            'num_shards': len(indexes),
            'num_ngrams': len(ngrams),
            'include_early_checkpoints': include_early,
            'total_tokens': sum(shard_token_counts) if shard_token_counts else None,
            'pythia_tokens_per_step': PYTHIA_TOKENS_PER_STEP
        },
        'ngrams': {}
    }
    
    # Merge data for each n-gram
    for ngram in ngrams:
        cum_data = cumulative_data[ngram]
        checkpoint_info = checkpoint_data[ngram]
        
        analysis_data['ngrams'][ngram] = {
            'text': ngram,
            'cumulative_frequency': cum_data['cumulative_frequency'],
            'shard_frequencies': cum_data['shard_frequencies'],
            'checkpoint_frequencies': checkpoint_info['checkpoint_frequencies'],
            'pythia_steps': checkpoint_info['pythia_steps'],
            'token_cutoffs': checkpoint_info['token_cutoffs'],
            'sample_positions': checkpoint_info['global_positions']
        }
    
    # Generate clean JSON
    json_output = json.dumps(analysis_data, indent=2, ensure_ascii=False)
    
    # Save to file if requested
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(json_output)
    
    return json_output


def build_ngram_dataset(
    indexes: List[MemmapIndex],
    tokenizer: AutoTokenizer,
    ngrams: List[str],
    context_window: int = 100,
    max_examples_per_ngram: int = 20,
) -> Dict[str, Any]:
    """Create a simple dataset for checkpoint analysis."""
    dataset: Dict[str, Any] = {
        'ngrams': {}, 
        'summary': {'total_frequency': 0, 'ngrams_found': 0}
    }
    
    for ngram in ngrams:
        # Frequency: count all local positions across all indexes
        pos_by_index = get_all_positions_local(indexes, ngram, tokenizer)
        frequency = int(sum(int(len(v)) for v in pos_by_index.values()))
        
        if frequency == 0:
            dataset['ngrams'][ngram] = {'frequency': 0, 'examples': []}
            continue

        # Collect example sentences (cap per n-gram)
        contexts = get_context_sentences(
            indexes, ngram, tokenizer, pos_by_index, context_window=context_window
        )
        examples: List[Dict[str, Any]] = []
        
        for ctx in contexts[:max_examples_per_ngram]:
            examples.append({
                'sentence': ctx['sentence'],
                'positions': ctx['ngram_positions_in_sentence'],
            })

        dataset['ngrams'][ngram] = {'frequency': frequency, 'examples': examples}
        dataset['summary']['total_frequency'] += frequency
        dataset['summary']['ngrams_found'] += 1

    return dataset