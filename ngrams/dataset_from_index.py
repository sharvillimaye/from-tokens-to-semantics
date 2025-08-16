"""dataset from Memmap index"""

from collections import Counter
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import re
import nltk

import numpy as np
from tokengrams import MemmapIndex
from transformers import AutoTokenizer

ENGLISH_STOPWORDS = set(nltk.corpus.stopwords.words('english'))

def shards_to_index(shards: list[Path], index_paths: list[Path]) -> list:
    """create a list of MemmapIndex objects from a list of shards"""
    return [MemmapIndex(shard, index_path) for shard, index_path in zip(shards, index_paths)]

def get_mmap_data(shards: list[Path]) -> list[np.memmap]:
    """get the mmap data from the shards (uint16 tokens)."""
    return [np.memmap(shard, dtype=np.uint16, mode='r') for shard in shards]

def get_all_positions_local(indexes: list[MemmapIndex], ngram: str, tokenizer: AutoTokenizer) -> Dict[int, np.ndarray]:
    """Return per-index local start positions for an n-gram."""
    token_ids = tokenizer.encode(ngram, add_special_tokens=False)
    out: Dict[int, np.ndarray] = {}
    if not token_ids:
        return {i: np.array([], dtype=np.int64) for i in range(len(indexes))}
    for i, index in enumerate(indexes):
        out[i] = np.asarray(index.positions(token_ids), dtype=np.int64)
    return out

def resolve_index_offsets(
    indexes: list[MemmapIndex],
    shard_token_counts: Optional[list[int]] = None,
) -> list[int]:
    """Resolve per-index global token offsets.

    Priority:
      1) Use explicit shard_token_counts to build prefix-sum offsets
      2) Use index.offset attribute if present
      3) Use len(index) if supported to build prefix-sum offsets
    """
    # 1) Use provided counts
    if shard_token_counts is not None:
        offsets: list[int] = []
        acc = 0
        for cnt in shard_token_counts:
            offsets.append(acc)
            acc += int(cnt)
        return offsets

    # 2) Try attribute 'offset'
    offsets_attr: list[int] = []
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
        lengths: list[int] = [int(len(idx)) for idx in indexes]  # type: ignore[arg-type]
        offsets: list[int] = []
        acc = 0
        for cnt in lengths:
            offsets.append(acc)
            acc += cnt
        return offsets
    except Exception:
        raise ValueError(
            "Cannot resolve index offsets. Provide shard_token_counts or use indexes with 'offset' or __len__."
        )


def get_all_positions_global(
    indexes: list[MemmapIndex],
    ngram: str,
    tokenizer: AutoTokenizer,
    index_offsets: Optional[list[int]] = None,
    shard_token_counts: Optional[list[int]] = None,
) -> list[tuple]:
    """Return global (start, end) token positions for an n-gram across all indexes.

    You can provide either explicit index_offsets or shard_token_counts; if both are None
    the function will attempt to infer offsets from the indexes.
    """
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
    indexes: list[MemmapIndex],
    ngram: str,
    tokenizer: AutoTokenizer,
    positions_by_index: Dict[int, np.ndarray],
    *,
    index_offsets: Optional[list[int]] = None,
    context_window: int = 100,
) -> List[Dict[str, Any]]:
    """Collect context sentences for an n-gram at the given positions.

    positions_by_index maps index_idx -> local start positions.
    Returns a list of dicts with: sentence, global_position, index_idx, ngram_positions_in_sentence.
    """
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
                # Character-offset positions within the sentence
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
                    'global_position': (int(local_start) + int(index_offsets[i]), int(local_start) + int(index_offsets[i]) + len(token_ids)),
                    'index_idx': i,
                    'ngram_positions_in_sentence': pos_list,
                })
            except Exception:
                continue
    return contexts


def annotate_global_positions(
    contexts: List[Dict[str, Any]],
    index_offsets: list[int],
) -> None:
    """In-place annotate contexts with global_position using provided index offsets."""
    for ctx in contexts:
        if ctx.get('global_position') is None:
            idx_i = int(ctx['index_idx'])
            # The context knows only one local start implicitly; we cannot recover it here without changes.
            # For now, leave global_position as None (caller may compute separately if needed).
            # This function is provided as a placeholder if you later pass local starts through.
            pass

def extract_contextual_ngrams(original_ngram: str, context_sentence: str, 
                              min_ngram_len: int = 1, max_ngram_len: int = 3,
                              window_size: int = 10) -> list[str]:
    """extract meaningful n-grams from context sentences that are related to the original n-gram"""
    if not context_sentence or len(context_sentence.strip()) < 10:
        return set()
    
    # Clean and normalize
    sentence = context_sentence.lower().strip()
    sentence = re.sub(r'[^\w\s]', ' ', sentence)  # Remove punctuation
    sentence = re.sub(r'\s+', ' ', sentence)      # Normalize whitespace
    
    # Find position of original n-gram in sentence
    orig_words = original_ngram.lower().split()
    sent_words = sentence.split()
    
    # Find all occurrences of original n-gram
    original_positions = []
    for i in range(len(sent_words) - len(orig_words) + 1):
        if sent_words[i:i+len(orig_words)] == orig_words:
            original_positions.append((i, i + len(orig_words)))
    
    if not original_positions:
        # Fallback: look for partial matches
        for i, word in enumerate(sent_words):
            if word in orig_words:
                original_positions.append((max(0, i-1), min(len(sent_words), i+2)))
    
    if not original_positions:
        return set()
    
    extracted_ngrams: list[str] = []
    
    # Extract n-grams around each occurrence
    for start_pos, end_pos in original_positions:
        # Define extraction window
        window_start = max(0, start_pos - window_size)
        window_end = min(len(sent_words), end_pos + window_size)
        window_words = sent_words[window_start:window_end]
        
        # Generate all n-grams within the window
        for ngram_len in range(min_ngram_len, min(max_ngram_len + 1, len(window_words) + 1)):
            for i in range(len(window_words) - ngram_len + 1):
                candidate_words = window_words[i:i + ngram_len]
                candidate_ngram = ' '.join(candidate_words)
                
                # Skip if identical to original
                if candidate_ngram == original_ngram.lower():
                    continue
                
                # Basic filtering
                if is_meaningful_ngram(candidate_ngram, orig_words):
                    extracted_ngrams.append(candidate_ngram)
    
    return list(set(extracted_ngrams))

def is_meaningful_ngram(ngram: str, original_words: list[str]) -> bool:
    """filter out nonsensical n-grams using heuristic rules"""
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

def get_cutoff_frequencies(ngrams: list[str]) -> list[str]:
    """get the cutoff frequencies of the ngrams"""
    return []


def build_ngram_dataset(
    indexes: list[MemmapIndex],
    tokenizer: AutoTokenizer,
    ngrams: list[str],
    context_window: int = 100,
    max_examples_per_ngram: int = 20,
) -> dict[str, Any]:
    """Create a simple dataset for checkpoint analysis.

    Output schema:
      {
        'ngrams': {
           ngram: {
             'frequency': int,                     # cumulative occurrences across shards
             'examples': [                         # up to max_examples_per_ngram
                { 'sentence': str,
                  'positions': list[tuple[int,int]]  # char offsets within sentence
                }, ...
             ]
           }, ...
        },
        'summary': { 'total_frequency': int, 'ngrams_found': int }
      }
    """
    dataset: dict[str, Any] = { 'ngrams': {}, 'summary': { 'total_frequency': 0, 'ngrams_found': 0 } }
    for ngram in ngrams:
        # Frequency: count all local positions across all indexes
        pos_by_index = get_all_positions_local(indexes, ngram, tokenizer)
        frequency = int(sum(int(len(v)) for v in pos_by_index.values()))
        if frequency == 0:
            dataset['ngrams'][ngram] = { 'frequency': 0, 'examples': [] }
            continue

        # Collect example sentences (cap per n-gram)
        contexts = get_context_sentences(indexes, ngram, tokenizer, pos_by_index, context_window=context_window)
        examples: list[dict[str, Any]] = []
        for ctx in contexts[:max_examples_per_ngram]:
            examples.append({
                'sentence': ctx['sentence'],
                'positions': ctx['ngram_positions_in_sentence'],
            })

        dataset['ngrams'][ngram] = { 'frequency': frequency, 'examples': examples }
        dataset['summary']['total_frequency'] += frequency
        dataset['summary']['ngrams_found'] += 1

    return dataset
