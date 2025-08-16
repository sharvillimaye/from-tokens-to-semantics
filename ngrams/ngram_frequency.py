"""
High-throughput n-gram mining pipeline with GPU acceleration and robust error handling.
- Uses memmap/mmap to stream token data
- GPU-accelerated pattern matching using PyTorch
- Multiple parallelization strategies (multiprocessing, threading, GPU)
- Robust error handling and recovery
- Optimized memory usage and chunking
"""

import os
import math
import mmap
import numpy as np
import torch
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple, Sequence, Any, Optional, Set
import re
from collections import Counter, defaultdict
import multiprocessing as mp
from transformers import AutoTokenizer
import logging
from functools import partial
import time
import gc
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize NLTK resources
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    ENGLISH_STOPWORDS = set(stopwords.words('english'))
except:
    logger.warning("NLTK resources not available, using basic stopword list")
    ENGLISH_STOPWORDS = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being'}

# --------------------
# GPU Configuration
# --------------------
def get_device():
    """Get the best available device for computation"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

DEVICE = get_device()
logger.info(f"Using device: {DEVICE}")

# --------------------
# Helper Functions
# --------------------
VOID = lambda n: np.dtype((np.void, 2 * n))

_SENT_SPLIT_RE = re.compile(r'(?<=[\.!\?])\s+')

# Module-level globals to avoid pickling large banks for each task
_GLOBAL_BANKS = None
_GLOBAL_USE_GPU = False

def _init_worker(banks: Dict[int, dict], use_gpu: bool):
    """Initializer for worker processes to set global banks once."""
    global _GLOBAL_BANKS, _GLOBAL_USE_GPU
    _GLOBAL_BANKS = banks
    _GLOBAL_USE_GPU = use_gpu

def build_pattern_banks_gpu(tokenizer, phrases: Sequence[str]) -> Dict[int, dict]:
    """
    Build pattern banks optimized for GPU processing.
    Returns banks with both CPU and GPU tensors.
    """
    buckets = {}
    for p in phrases:
        ids = tokenizer.encode(p, add_special_tokens=False)
        if not ids:
            continue
        n = len(ids)
        buckets.setdefault(n, []).append(np.asarray(ids, dtype=np.uint16))

    banks = {}
    for n, arrs in buckets.items():
        arr = np.ascontiguousarray(np.stack(arrs, axis=0))  # (P_n, n)
        voids = arr.view(VOID(n)).ravel()
        
        # Unique voids (deduplicate identical token sequences)
        uniq_voids, inv = np.unique(voids, return_inverse=True)
        
        # Build mapping from void bytes -> local id (0..U-1)
        void_to_id = {uniq_voids[i].tobytes(): i for i in range(len(uniq_voids))}
        
        # Create GPU tensors for fast matching
        gpu_tokens = torch.from_numpy(arr).to(DEVICE, dtype=torch.int64)
        gpu_voids = torch.from_numpy(uniq_voids.view(np.uint16).reshape(-1, n)).to(DEVICE, dtype=torch.int64)
        
        # Compute rolling hash constants for GPU acceleration
        hash_base = 31
        hash_mod = 2**63 - 1  # Large prime for 64-bit hash
        
        # Precompute hash values for patterns
        pattern_hashes = []
        for i in range(len(uniq_voids)):
            void_bytes = uniq_voids[i].tobytes()
            tokens = np.frombuffer(void_bytes, dtype=np.uint16, count=n)
            hash_val = 0
            for j, token in enumerate(tokens):
                hash_val = (hash_val * hash_base + int(token)) % hash_mod
            pattern_hashes.append(hash_val)
        
        pattern_hashes = np.array(pattern_hashes, dtype=np.int64)
        sorted_indices = np.argsort(pattern_hashes)
        sorted_hashes = pattern_hashes[sorted_indices]
        
        banks[n] = {
            'tokens': arr,
            'voids': uniq_voids,
            'void_to_id': void_to_id,
            'first_ids': np.unique(arr[:, 0]).astype(np.uint16),
            'n': n,
            'size': len(uniq_voids),
            # GPU tensors
            'gpu_tokens': gpu_tokens,
            'gpu_voids': gpu_voids,
            'gpu_first_ids': torch.from_numpy(np.unique(arr[:, 0])).to(DEVICE, dtype=torch.int64),
            # Rolling hash acceleration
            'pattern_hashes': pattern_hashes,
            'sorted_hashes': sorted_hashes,
            'sorted_indices': sorted_indices,
            'hash_base': hash_base,
            'hash_mod': hash_mod
        }
    return banks

def gpu_pattern_matching_rolling_hash(buf: np.ndarray, bank: dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    GPU-accelerated pattern matching using rolling hash + binary search.
    Much faster than broadcasting for large pattern banks.
    """
    n = bank['n']
    if buf.shape[0] < n:
        return np.array([]), np.array([])
    
    # Convert buffer to GPU tensor
    buf_gpu = torch.from_numpy(buf).to(DEVICE, dtype=torch.int64)
    
    # Compute rolling hashes for all windows
    hash_base = bank['hash_base']
    hash_mod = bank['hash_mod']
    
    # First window hash
    num_windows = buf_gpu.shape[0] - n + 1
    if num_windows <= 0:
        return np.array([]), np.array([])
    
    # Compute all window hashes using rolling hash
    window_hashes = torch.zeros(num_windows, dtype=torch.int64, device=DEVICE)
    
    # Initial hash for first window
    hash_val = 0
    base_power = 1
    for i in range(n):
        hash_val = (hash_val * hash_base + buf_gpu[i]) % hash_mod
        if i < n - 1:
            base_power = (base_power * hash_base) % hash_mod
    
    window_hashes[0] = hash_val
    
    # Rolling hash for remaining windows
    for i in range(1, num_windows):
        # Remove leftmost character, add rightmost character
        old_char = buf_gpu[i - 1]
        new_char = buf_gpu[i + n - 1]
        hash_val = (hash_val - old_char * base_power) % hash_mod
        hash_val = (hash_val * hash_base + new_char) % hash_mod
        window_hashes[i] = hash_val
    
    # Binary search in sorted pattern hashes
    sorted_hashes = torch.from_numpy(bank['sorted_hashes']).to(DEVICE, dtype=torch.int64)
    
    # Find matches using searchsorted
    indices = torch.searchsorted(sorted_hashes, window_hashes)
    
    # Check for exact matches
    valid_mask = (indices < len(sorted_hashes)) & (sorted_hashes[indices] == window_hashes)
    
    if not valid_mask.any():
        return np.array([]), np.array([])
    
    # Get matched positions and convert pattern indices back to original order
    matched_positions = torch.nonzero(valid_mask).squeeze(-1)
    matched_sorted_indices = indices[valid_mask]
    
    # Convert back to original pattern indices
    sorted_indices = torch.from_numpy(bank['sorted_indices']).to(DEVICE, dtype=torch.int64)
    matched_pattern_ids = sorted_indices[matched_sorted_indices]
    
    return matched_positions.cpu().numpy(), matched_pattern_ids.cpu().numpy()

def gpu_pattern_matching(buf: np.ndarray, bank: dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    GPU-accelerated pattern matching - chooses best method based on pattern count.
    """
    n = bank['n']
    if buf.shape[0] < n:
        return np.array([]), np.array([])
    
    # Use rolling hash for large pattern banks, broadcasting for small ones
    if bank['size'] > 100:
        return gpu_pattern_matching_rolling_hash(buf, bank)
    else:
        return gpu_pattern_matching_broadcast(buf, bank)

def gpu_pattern_matching_broadcast(buf: np.ndarray, bank: dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Original broadcasting method for small pattern banks.
    """
    n = bank['n']
    if buf.shape[0] < n:
        return np.array([]), np.array([])
    
    # Convert buffer to GPU tensor
    buf_gpu = torch.from_numpy(buf).to(DEVICE, dtype=torch.int64)
    
    # Create sliding windows on GPU
    windows = buf_gpu.unfold(0, n, 1)  # (num_windows, n)
    
    # First-token prefiltering on GPU
    first_ids = bank['gpu_first_ids']
    if first_ids.numel() > 0:
        # Find windows that start with valid first tokens
        first_tokens = windows[:, 0]
        mask = torch.isin(first_tokens, first_ids)
        if not mask.any():
            return np.array([]), np.array([])
        
        candidate_windows = windows[mask]
        candidate_positions = torch.nonzero(mask).squeeze(-1)
    else:
        candidate_windows = windows
        candidate_positions = torch.arange(windows.shape[0], device=DEVICE)
    
    if candidate_windows.shape[0] == 0:
        return np.array([]), np.array([])
    
    # Compare with pattern bank using broadcasting
    pattern_bank = bank['gpu_voids']  # (num_patterns, n)
    
    # Reshape for broadcasting: (num_candidates, 1, n) vs (1, num_patterns, n)
    candidates_expanded = candidate_windows.unsqueeze(1)  # (num_candidates, 1, n)
    patterns_expanded = pattern_bank.unsqueeze(0)  # (1, num_patterns, n)
    
    # Find exact matches
    matches = (candidates_expanded == patterns_expanded).all(dim=2)  # (num_candidates, num_patterns)
    
    # Get matched positions and pattern IDs
    matched_candidates, matched_patterns = torch.nonzero(matches, as_tuple=True)
    
    if matched_candidates.numel() == 0:
        return np.array([]), np.array([])
    
    # Convert back to CPU
    matched_positions = candidate_positions[matched_candidates].cpu().numpy()
    matched_pattern_ids = matched_patterns.cpu().numpy()
    
    return matched_positions, matched_pattern_ids

def shard_chunk_worker_safe(args: Tuple[str, int, int, int]) -> Dict:
    """
    Safe worker function with comprehensive error handling.
    Uses global banks to avoid pickle overhead.
    """
    try:
        shard_path, offset, chunk_tokens, overlap = args
        shard_path = str(shard_path)
        
        # Get banks from global variable
        banks = _GLOBAL_BANKS
        if banks is None:
            return {'counts': {}, 'spans': [], 'error': 'no_global_banks'}
        
        # Validate file exists
        if not os.path.exists(shard_path):
            logger.warning(f"Shard file not found: {shard_path}")
            return {'counts': {}, 'spans': [], 'error': 'file_not_found'}
        
        # results
        counts_by_n = {}
        spans = []
        
        try:
            with open(shard_path, "rb") as f:
                with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                    file_tokens = len(mm) // 2
                    start = offset
                    end = min(offset + chunk_tokens + overlap, file_tokens)
                    
                    if start >= end:
                        return {'counts': {}, 'spans': []}
                    
                    # Read buffer
                    buf = np.frombuffer(mm[start*2:end*2], dtype=np.uint16)
                    
                    if buf.size == 0:
                        return {'counts': {}, 'spans': []}
                    
                    # Process each pattern length
                    for n, bank in banks.items():
                        if buf.shape[0] < n:
                            continue
                        
                        # Use GPU matching if available
                        if _GLOBAL_USE_GPU and DEVICE.type != 'cpu':
                            try:
                                matched_positions, matched_pattern_ids = gpu_pattern_matching(buf, bank)
                            except Exception as e:
                                logger.warning(f"GPU matching failed for n={n}, falling back to CPU: {e}")
                                matched_positions, matched_pattern_ids = cpu_pattern_matching(buf, bank)
                        else:
                            matched_positions, matched_pattern_ids = cpu_pattern_matching(buf, bank)
                        
                        if matched_positions.size > 0:
                            # Count occurrences
                            unique_ids, counts = np.unique(matched_pattern_ids, return_counts=True)
                            counts_by_n[n] = (unique_ids, counts)
                            
                            # Sample spans for decoding
                            max_sample = min(16, len(matched_positions))
                            chosen_positions = matched_positions[:max_sample]
                            
                            for pos_rel in chosen_positions:
                                s = int(start + int(pos_rel))
                                e = s + n
                                spans.append((s, e, n, bank['voids'][0].tobytes()))
        
        except Exception as e:
            logger.error(f"Error processing chunk in {shard_path}: {e}")
            return {'counts': {}, 'spans': [], 'error': str(e)}
        
        return {'counts': counts_by_n, 'spans': spans}
    
    except Exception as e:
        logger.error(f"Critical error in worker: {e}")
        return {'counts': {}, 'spans': [], 'error': str(e)}

def extract_contextual_ngrams(original_ngram: str, context_sentence: str, tokenizer, 
                              min_ngram_len: int = 2, max_ngram_len: int = 8,
                              window_size: int = 5) -> Set[str]:
    """
    Extract meaningful n-grams from context sentences that are related to the original n-gram.
    
    Args:
        original_ngram: The seed n-gram (e.g., "paris france")
        context_sentence: Sentence containing the original n-gram
        tokenizer: HuggingFace tokenizer for consistent tokenization
        min_ngram_len: Minimum n-gram length in words
        max_ngram_len: Maximum n-gram length in words
        window_size: Window around original n-gram to focus extraction
        
    Returns:
        Set of new n-gram candidates
    """
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
    
    extracted_ngrams = set()
    
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
                    extracted_ngrams.add(candidate_ngram)
    
    return extracted_ngrams

def is_meaningful_ngram(ngram: str, original_words: List[str]) -> bool:
    """
    Filter out nonsensical n-grams using heuristic rules.
    
    Args:
        ngram: Candidate n-gram to evaluate
        original_words: Words from the original seed n-gram
        
    Returns:
        True if n-gram appears meaningful, False otherwise
    """
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

def cpu_pattern_matching(buf: np.ndarray, bank: dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    CPU fallback for pattern matching.
    """
    n = bank['n']
    if buf.shape[0] < n:
        return np.array([]), np.array([])
    
    starts = buf.shape[0] - n + 1
    
    # First-token prefilter
    first_ids = bank['first_ids']
    if first_ids.size > 0:
        first_ok = np.isin(buf[:starts], first_ids, assume_unique=False)
        if not first_ok.any():
            return np.array([]), np.array([])
        
        candidate_idx = np.nonzero(first_ok)[0]
        windows = np.lib.stride_tricks.sliding_window_view(buf, n)
        cand_windows = windows[candidate_idx]
        cand_windows = np.ascontiguousarray(cand_windows)
        cand_void = cand_windows.view(VOID(n)).ravel()
        
        mask = np.isin(cand_void, bank['voids'], assume_unique=False)
        if not mask.any():
            return np.array([]), np.array([])
        
        matched_voids = cand_void[mask]
        matched_candidate_positions = candidate_idx[mask]
        
        ids = np.fromiter((bank['void_to_id'][v.tobytes()] for v in matched_voids),
                         count=matched_voids.shape[0], dtype=np.int64)
        
        return matched_candidate_positions, ids
    else:
        # Fallback to scanning all windows
        windows = np.lib.stride_tricks.sliding_window_view(buf, n)
        windows = np.ascontiguousarray(windows)
        win_void = windows.view(VOID(n)).ravel()
        mask = np.isin(win_void, bank['voids'], assume_unique=False)
        
        if not mask.any():
            return np.array([]), np.array([])
        
        matched_voids = win_void[mask]
        matched_positions = np.nonzero(mask)[0]
        
        ids = np.fromiter((bank['void_to_id'][v.tobytes()] for v in matched_voids),
                         count=matched_voids.shape[0], dtype=np.int64)
        
        return matched_positions, ids

# --------------------
# Optimized Miner Class
# --------------------
class FastNgramMiner:
    def __init__(self, 
                 tokenizer_name: str = "EleutherAI/gpt-neox-20b", 
                 chunk_tokens: int = 16_000_000,  # Reduced for better memory management
                 overlap_scale: int = 1, 
                 n_workers: int = None,
                 use_gpu: bool = True,
                 max_memory_gb: float = 8.0,
                 gpu_workers: int = None,
                 io_workers: int = None):
        """
        Optimized n-gram miner with GPU acceleration and robust error handling.
        
        Args:
            tokenizer_name: HuggingFace tokenizer name
            chunk_tokens: tokens per chunk (optimized for GPU memory)
            overlap_scale: overlap factor for pattern matching
            n_workers: total worker processes (None = auto-detect)
            use_gpu: whether to use GPU acceleration
            max_memory_gb: maximum memory usage in GB
            gpu_workers: dedicated GPU compute threads (None = auto-detect)
            io_workers: dedicated I/O threads (None = auto-detect)
        """
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.device = DEVICE
        self.chunk_tokens = int(chunk_tokens)
        self.overlap_scale = overlap_scale
        self.use_gpu = use_gpu and DEVICE.type != 'cpu'
        self.max_memory_gb = max_memory_gb
        
        # Advanced parallelism configuration
        cpu_count = mp.cpu_count()
        self.n_workers = n_workers or max(1, cpu_count - 2)  # Leave 2 cores for system
        
        if self.use_gpu:
            # GPU mode: separate I/O and compute workers for maximum utilization
            self.gpu_workers = gpu_workers or min(8, max(2, cpu_count // 4))  # GPU compute threads
            self.io_workers = io_workers or min(32, max(4, cpu_count // 2))   # I/O threads
            print(f"🚀 GPU Parallelism: {self.gpu_workers} GPU compute + {self.io_workers} I/O threads")
        else:
            # CPU mode: use all available cores for compute
            self.gpu_workers = 0
            self.io_workers = 0
            print(f"🚀 CPU Parallelism: {self.n_workers} worker processes")
        
        # Adjust chunk size based on available memory
        if self.use_gpu:
            if self.device.type == 'cuda':
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                self.chunk_tokens = min(self.chunk_tokens, int(gpu_memory * 0.3 * 1e6))  # 30% of GPU memory
            elif self.device.type == 'mps':
                # For MPS, use conservative memory estimate
                self.chunk_tokens = min(self.chunk_tokens, 8_000_000)  # 8M tokens for MPS
        
        logger.info(f"Initialized miner with {self.n_workers} workers, "
                   f"chunk_size={self.chunk_tokens}, GPU={self.use_gpu}")

    def create_ngram_dataset(self,
                           shard_paths: Sequence[Path],
                           initial_probe_phrases: Sequence[str],
                           max_sentences_per_shard: int = 1000,
                           batch_size: int = 4,
                           min_frequency_threshold: int = 5,
                           max_new_ngrams: int = 1000) -> Dict:
        """
        2-pass n-gram dataset creation pipeline.
        
        Pass 1: Find initial probe phrases and extract context sentences
        Pass 2: Extract new n-grams from contexts and create comprehensive dataset
        
        Args:
            shard_paths: Paths to data shard files
            initial_probe_phrases: Starting seed n-grams (e.g., country-capital pairs)
            max_sentences_per_shard: Max sentences to harvest per shard
            batch_size: Batch size for parallel processing
            min_frequency_threshold: Minimum frequency for n-grams in final dataset
            max_new_ngrams: Maximum new n-grams to extract in pass 2
            
        Returns:
            Structured dataset with n-grams, frequencies, sentences, and positions
        """
        print(f"\n🚀 Starting 2-pass n-gram dataset creation pipeline")
        print(f"📊 Configuration: 2 passes, {len(shard_paths)} shards, {len(initial_probe_phrases)} initial phrases")
        print(f"🖥️  Hardware: {'GPU' if self.use_gpu else 'CPU'} mode, {self.n_workers} workers")
        print(f"💾 Memory: {self.chunk_tokens:,} tokens per chunk")
        
        # Dataset structure for final output
        dataset = {
            'metadata': {
                'total_ngrams': 0,
                'total_sentences': 0,
                'processing_time': 0,
                'passes_completed': 2,
                'initial_phrases': initial_probe_phrases
            },
            'ngrams': {}  # Will contain: {ngram: {frequency, examples: [{sentence, positions}]}}
        }
        
        total_start_time = time.time()
        
        # PASS 1: Find initial phrases and extract context sentences
        print(f"\n{'='*60}")
        print(f"🔄 PASS 1: Mining initial phrases and extracting contexts")
        print(f"📝 Mining with {len(initial_probe_phrases)} seed patterns")
        print(f"{'='*60}")
        
        pass1_start = time.time()
        pass1_results = self._mine_and_collect_detailed(
            shard_paths=shard_paths,
            probe_phrases=initial_probe_phrases,
            max_sentences_per_shard=max_sentences_per_shard,
            batch_size=batch_size
        )
        pass1_time = time.time() - pass1_start
        
        print(f"\n📈 Pass 1 Results:")
        print(f"   🔢 Found {len(pass1_results['ngram_data'])} unique n-grams")
        print(f"   📄 Harvested {len(pass1_results['sentences'])} context sentences")
        print(f"   ⏱️  Processing time: {pass1_time:.2f}s")
        
        # PASS 2: Extract new n-grams and create comprehensive dataset
        print(f"\n{'='*60}")
        print(f"🔄 PASS 2: Extracting new n-grams and building dataset")
        print(f"📝 Analyzing {len(pass1_results['sentences'])} context sentences")
        print(f"{'='*60}")
        
        pass2_start = time.time()
        
        # Extract new n-grams from context sentences
        print(f"🔍 Extracting new n-grams from context sentences...")
        new_ngrams = self._extract_new_ngrams_from_sentences(
            sentences=pass1_results['sentences'],
            original_ngrams=initial_probe_phrases,
            max_new_ngrams=max_new_ngrams
        )
        
        # Combine original and new n-grams for final mining
        all_ngrams_for_pass2 = list(pass1_results['ngram_data'].keys()) + list(new_ngrams)
        print(f"   ✨ Total n-grams for Pass 2: {len(all_ngrams_for_pass2)} ({len(new_ngrams)} new)")
        
        # Mine all n-grams with detailed sentence information
        pass2_results = self._mine_and_collect_detailed(
            shard_paths=shard_paths,
            probe_phrases=all_ngrams_for_pass2,
            max_sentences_per_shard=max_sentences_per_shard,
            batch_size=batch_size,
            collect_positions=True  # Enable position tracking
        )
        pass2_time = time.time() - pass2_start
        
        print(f"\n📈 Pass 2 Results:")
        print(f"   🔢 Found {len(pass2_results['ngram_data'])} unique n-grams with positions")
        print(f"   📄 Collected {len(pass2_results['sentences'])} example sentences")
        print(f"   ⏱️  Processing time: {pass2_time:.2f}s")
        
        # Build final structured dataset
        print(f"\n🏗️  Building structured dataset...")
        
        for ngram, data in pass2_results['ngram_data'].items():
            if data['frequency'] >= min_frequency_threshold:
                dataset['ngrams'][ngram] = {
                    'frequency': data['frequency'],
                    'examples': []
                }
                
                # Add example sentences with positions
                for example in data['examples'][:10]:  # Limit to 10 examples per n-gram
                    dataset['ngrams'][ngram]['examples'].append({
                        'sentence': example['sentence'],
                        'positions': example['positions']  # List of (start, end) positions
                    })
        
        # Update metadata
        total_time = time.time() - total_start_time
        dataset['metadata'].update({
            'total_ngrams': len(dataset['ngrams']),
            'total_sentences': len(set(
                ex['sentence'] for ngram_data in dataset['ngrams'].values() 
                for ex in ngram_data['examples']
            )),
            'processing_time': total_time,
            'pass1_time': pass1_time,
            'pass2_time': pass2_time
        })
        
        print(f"\n{'='*60}")
        print(f"🎉 DATASET CREATION COMPLETED!")
        print(f"{'='*60}")
        print(f"📊 Final Dataset Statistics:")
        print(f"   🔢 Total n-grams: {dataset['metadata']['total_ngrams']:,}")
        print(f"   📄 Unique sentences: {dataset['metadata']['total_sentences']:,}")
        print(f"   ⏱️  Total processing time: {total_time:.2f}s")
        print(f"   🚀 Pass 1: {pass1_time:.2f}s, Pass 2: {pass2_time:.2f}s")
        
        # Show sample of the dataset structure
        print(f"\n📋 Sample Dataset Entries:")
        sample_ngrams = list(dataset['ngrams'].items())[:3]
        for ngram, data in sample_ngrams:
            print(f"   📝 '{ngram}': {data['frequency']} occurrences, {len(data['examples'])} examples")
            if data['examples']:
                ex = data['examples'][0]
                print(f"      📄 \"{ex['sentence'][:80]}...\"")
                print(f"      📍 Positions: {ex['positions']}")
        
        return dataset

    def _extract_new_ngrams_from_sentences(self, 
                                          sentences: List[str], 
                                          original_ngrams: List[str],
                                          max_new_ngrams: int = 1000) -> Set[str]:
        """
        Extract new n-grams from harvested sentences using contextual analysis.
        """
        print(f"   🔄 Processing {len(sentences)} sentences to extract new n-grams...")
        
        all_extracted = set()
        
        # Process sentences with progress bar
        with tqdm(sentences, desc="   Extracting n-grams", leave=False) as pbar:
            for sentence in pbar:
                for original_ngram in original_ngrams:
                    if original_ngram.lower() in sentence.lower():
                        extracted = extract_contextual_ngrams(
                            original_ngram=original_ngram,
                            context_sentence=sentence,
                            tokenizer=self.tokenizer,
                            min_ngram_len=2,
                            max_ngram_len=6,
                            window_size=5
                        )
                        all_extracted.update(extracted)
                        
                        if len(all_extracted) >= max_new_ngrams * 2:  # Extract more, filter later
                            break
                
                pbar.set_postfix({"Extracted": len(all_extracted)})
        
        # Rank and filter new n-grams
        if len(all_extracted) > max_new_ngrams:
            # Simple ranking by length and alphanumeric content
            scored_ngrams = []
            for ngram in all_extracted:
                words = ngram.split()
                # Score based on meaningful content
                score = len([w for w in words if w not in ENGLISH_STOPWORDS and len(w) > 2])
                score += len(words) * 0.1  # Slight preference for longer n-grams
                scored_ngrams.append((score, ngram))
            
            # Take top candidates
            scored_ngrams.sort(reverse=True)
            all_extracted = {ngram for _, ngram in scored_ngrams[:max_new_ngrams]}
        
        print(f"   ✅ Extracted and filtered {len(all_extracted)} new n-gram candidates")
        return all_extracted

    def mine_and_count(self,
                      shard_paths: Sequence[Path],
                      probe_phrases: Sequence[str],
                      max_sentences_per_shard: int = 1000,
                      batch_size: int = 4) -> Dict:
        """
        Optimized mining pipeline with multiple parallelization strategies.
        """
        logger.info(f"Starting mining with {len(shard_paths)} shards and {len(probe_phrases)} phrases")
        
        # 1) Build pattern banks
        print(f"🏗️  Building pattern banks for {len(probe_phrases)} phrases...")
        start_time = time.time()
        banks = build_pattern_banks_gpu(self.tokenizer, probe_phrases)
        if not banks:
            logger.warning("No valid patterns found")
            return {'counts': {}, 'sentences': []}
        
        build_time = time.time() - start_time
        print(f"✅ Built {len(banks)} pattern banks in {build_time:.2f}s")
        for n, bank in banks.items():
            print(f"   📏 N-gram length {n}: {bank['size']} unique patterns")
        
        # 2) Pre-allocate global counters
        global_counts = {n: np.zeros(bank['size'], dtype=np.int64) for n, bank in banks.items()}
        
        # 3) Build chunk tasks with better error handling
        print(f"\n📂 Analyzing {len(shard_paths)} data shards...")
        tasks = []
        total_size_gb = 0
        
        with tqdm(shard_paths, desc="🔍 Scanning shards", unit="files") as pbar:
            for shard in pbar:
                try:
                    with open(shard, "rb") as f:
                        size_bytes = os.fstat(f.fileno()).st_size
                    total_size_gb += size_bytes / 1e9
                    file_tokens = size_bytes // 2
                    n_max = max(banks.keys())
                    overlap = max(1, (n_max - 1) * self.overlap_scale)
                    
                    # Schedule chunk offsets
                    offset = 0
                    shard_tasks = 0
                    while offset < file_tokens:
                        tasks.append((str(shard), int(offset), int(self.chunk_tokens), int(overlap)))
                        offset += self.chunk_tokens
                        shard_tasks += 1
                    
                    pbar.set_postfix({
                        "Size": f"{size_bytes/1e6:.1f}MB", 
                        "Tasks": shard_tasks,
                        "Total": f"{total_size_gb:.1f}GB"
                    })
                except Exception as e:
                    logger.error(f"Error processing shard {shard}: {e}")
                    continue
        
        print(f"✅ Created {len(tasks)} chunk tasks from {total_size_gb:.2f}GB of data")
        print(f"📊 Average chunk size: {self.chunk_tokens/1e6:.1f}M tokens")
        
        # 4) Process tasks with multiple strategies
        print(f"\n🚀 Starting parallel processing with {self.n_workers} workers")
        sampled_spans_by_shard = defaultdict(list)
        completed_tasks = 0
        failed_tasks = 0
        
        # Advanced hybrid parallelism: GPU + CPU + I/O optimization
        if self.use_gpu:
            print(f"🖥️  Hybrid GPU mode: {self.io_workers} I/O threads + {self.gpu_workers} GPU compute threads")
            # Use hybrid approach: separate I/O and compute pools
            executor_ctx = self._create_hybrid_executor(banks)
        else:
            # CPU mode: use processes for parallel computation
            executor_class = ProcessPoolExecutor
            max_workers = self.n_workers
            print(f"🖥️  CPU mode: Using {max_workers} processes for parallel compute")
            # Use initializer for processes to avoid pickling banks
            executor_ctx = executor_class(max_workers=max_workers, 
                                        initializer=_init_worker, 
                                        initargs=(banks, self.use_gpu))
        
        with executor_ctx as ex:
            # Optimize batch size for maximum throughput
            if self.use_gpu:
                # GPU mode: larger batches to maximize GPU utilization
                optimal_batch = max(batch_size, min(len(tasks), self.gpu_workers * 4))
            else:
                # CPU mode: batch size matches worker count
                optimal_batch = max(batch_size, min(len(tasks), self.n_workers * 2))
            
            total_batches = (len(tasks) + optimal_batch - 1) // optimal_batch
            print(f"📦 Processing {len(tasks)} tasks in {total_batches} batches of {optimal_batch}")
            print(f"🎯 Target throughput: {optimal_batch * total_batches / (total_batches * 2):.0f} chunks/sec")
            
            # Submit all tasks at once for maximum parallelism
            all_futures = {}
            submitted = 0
            
            with tqdm(total=len(tasks), desc="⚡ Processing chunks", unit="chunks") as pbar:
                # Submit tasks in optimal batches
                for i in range(0, len(tasks), optimal_batch):
                    batch_tasks = tasks[i:i + optimal_batch]
                    
                    # Submit all tasks in batch simultaneously
                    batch_futures = {}
                    for task in batch_tasks:
                        future = ex.submit(shard_chunk_worker_safe, task)
                        batch_futures[future] = task
                        all_futures[future] = task
                        submitted += 1
                    
                    pbar.set_description(f"⚡ Submitted {submitted}/{len(tasks)} chunks")
                    
                    # Process completed tasks as they finish (streaming results)
                    completed_in_batch = 0
                    for fut in as_completed(batch_futures.keys()):
                        try:
                            res = fut.result(timeout=600)  # 10 minute timeout for large chunks
                            completed_tasks += 1
                            completed_in_batch += 1
                            
                            if 'error' in res:
                                failed_tasks += 1
                                logger.warning(f"Task failed: {res['error']}")
                            else:
                                # Aggregate results
                                counts_by_n = res['counts']
                                spans = res['spans']
                                
                                # Count patterns found in this chunk
                                patterns_found = sum(len(ids) for ids, cnts in counts_by_n.values())
                                
                                for n, (ids, cnts) in counts_by_n.items():
                                    global_counts[n][ids] += cnts
                                
                                if spans:
                                    shard_path = batch_futures[fut][0]
                                    sampled_spans_by_shard[shard_path].extend(spans)
                            
                            # Update progress with real-time stats
                            current_rate = completed_tasks / max(1, time.time() - start_time)
                            pbar.set_postfix({
                                "Rate": f"{current_rate:.1f}/s",
                                "Completed": completed_tasks,
                                "Failed": failed_tasks,
                                "GPU%": f"{torch.cuda.utilization() if torch.cuda.is_available() else 0}%"
                            })
                            pbar.update(1)
                        
                        except Exception as e:
                            failed_tasks += 1
                            logger.error(f"Task execution failed: {e}")
                            pbar.update(1)
                        
                        # Clean up completed future
                        del all_futures[fut]
                    
                    # Aggressive memory cleanup after each batch
                    if self.use_gpu and self.device.type == 'cuda':
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()  # Ensure GPU operations complete
                    gc.collect()
                    
                    # Show batch completion stats
                    batch_rate = completed_in_batch / max(1, time.time() - start_time) * len(batch_tasks) / completed_in_batch if completed_in_batch > 0 else 0
                    print(f"✅ Batch {i//optimal_batch + 1}/{total_batches} complete: {completed_in_batch}/{len(batch_tasks)} tasks, {batch_rate:.1f} chunks/sec")
        
        print(f"\n✅ Processing completed: {completed_tasks} successful, {failed_tasks} failed")
        
        # 5) Convert results to readable format
        print(f"\n📊 Converting results to readable format...")
        counts_out = {}
        total_patterns = 0
        
        with tqdm(banks.items(), desc="🔤 Converting patterns", unit="banks") as pbar:
            for n, bank in pbar:
                arr = global_counts[n]
                if arr.sum() == 0:
                    continue
                
                nz = np.nonzero(arr)[0]
                bank_patterns = len(nz)
                total_patterns += bank_patterns
                
                for idx in nz:
                    count = int(arr[idx])
                    void_bytes = bank['voids'][idx].tobytes()
                    toks = np.frombuffer(void_bytes, dtype=np.uint16, count=n)
                    text = self.tokenizer.decode(toks.tolist()).lower().strip()
                    counts_out[text] = counts_out.get(text, 0) + count
                
                pbar.set_postfix({"Length": n, "Patterns": bank_patterns, "Total": total_patterns})
        
        print(f"✅ Converted {total_patterns} pattern matches to {len(counts_out)} unique n-grams")
        
        # 6) Harvest sentences (simplified for speed)
        print(f"\n📝 Harvesting context sentences...")
        harvested_sentences = self._harvest_sentences_simple(sampled_spans_by_shard, max_sentences_per_shard)
        
        total_time = time.time() - start_time
        print(f"\n🎯 Mining completed in {total_time:.2f}s")
        print(f"📈 Final Results:")
        print(f"   🔢 Unique n-grams found: {len(counts_out):,}")
        print(f"   📄 Context sentences: {len(harvested_sentences):,}")
        print(f"   ⚡ Processing rate: {len(tasks)/total_time:.1f} chunks/sec")
        
        return {
            'counts': counts_out,
            'sentences': harvested_sentences,
            'stats': {
                'total_time': total_time,
                'completed_tasks': completed_tasks,
                'failed_tasks': failed_tasks,
                'total_patterns': len(counts_out)
            }
        }
    
    def _harvest_sentences_simple(self, sampled_spans_by_shard: Dict, max_sentences: int) -> List[str]:
        """
        Simplified sentence harvesting for speed.
        """
        harvested_sentences = []
        total_spans = sum(len(spans) for spans in sampled_spans_by_shard.values())
        
        print(f"🔍 Processing {len(sampled_spans_by_shard)} shards with {total_spans} total spans")
        
        with tqdm(sampled_spans_by_shard.items(), desc="📄 Harvesting sentences", unit="shards") as pbar:
            for shard_str, spans in pbar:
                if len(harvested_sentences) >= max_sentences:
                    break
                
                if not spans:
                    continue
                
                # Take first few spans from each shard
                spans_to_process = spans[:min(10, len(spans))]
                shard_sentences = 0
                
                try:
                    shard_path = Path(shard_str)
                    with open(shard_path, "rb") as f:
                        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                            file_tokens = len(mm) // 2
                            
                            for s, e, _, _ in spans_to_process:
                                if len(harvested_sentences) >= max_sentences:
                                    break
                                
                                start = max(0, s - 64)  # Reduced context
                                end = min(file_tokens, e + 64)
                                toks = np.frombuffer(mm[start*2:end*2], dtype=np.uint16)
                                text = self.tokenizer.decode(toks.tolist())
                                
                                # Simple sentence extraction
                                sentences = _SENT_SPLIT_RE.split(text)
                                for sent in sentences[:3]:  # Limit sentences per span
                                    sent = sent.strip()
                                    if sent and len(sent) > 10:
                                        harvested_sentences.append(sent.lower())
                                        shard_sentences += 1
                                        if len(harvested_sentences) >= max_sentences:
                                            break
                    
                    pbar.set_postfix({
                        "Shard sentences": shard_sentences,
                        "Total": len(harvested_sentences),
                        "Progress": f"{len(harvested_sentences)}/{max_sentences}"
                    })
                    
                except Exception as e:
                    logger.warning(f"Error harvesting sentences from {shard_str}: {e}")
                    continue
        
        return harvested_sentences[:max_sentences]
    
    def _mine_and_collect_detailed(self,
                                 shard_paths: Sequence[Path],
                                 probe_phrases: Sequence[str],
                                 max_sentences_per_shard: int = 1000,
                                 batch_size: int = 4,
                                 collect_positions: bool = False) -> Dict:
        """
        Enhanced mining that collects detailed information for dataset creation.
        
        Returns:
            {
                'ngram_data': {ngram: {'frequency': int, 'examples': [{'sentence': str, 'positions': [(start, end)]}]}},
                'sentences': [str]  # All harvested sentences
            }
        """
        print(f"🏗️  Building pattern banks for {len(probe_phrases)} phrases...")
        start_time = time.time()
        banks = build_pattern_banks_gpu(self.tokenizer, probe_phrases)
        if not banks:
            return {'ngram_data': {}, 'sentences': []}
        
        build_time = time.time() - start_time
        print(f"✅ Built {len(banks)} pattern banks in {build_time:.2f}s")
        
        # Enhanced data collection structures
        ngram_data = defaultdict(lambda: {'frequency': 0, 'examples': []})
        all_sentences = []
        
        # Process shards with detailed position tracking
        print(f"\n📂 Processing {len(shard_paths)} shards with position tracking...")
        
        # Build tasks
        tasks = []
        total_size_gb = 0
        
        with tqdm(shard_paths, desc="🔍 Scanning shards", unit="files") as pbar:
            for shard in pbar:
                try:
                    with open(shard, "rb") as f:
                        size_bytes = os.fstat(f.fileno()).st_size
                    total_size_gb += size_bytes / 1e9
                    file_tokens = size_bytes // 2
                    n_max = max(banks.keys())
                    overlap = max(1, (n_max - 1) * self.overlap_scale)
                    
                    offset = 0
                    shard_tasks = 0
                    while offset < file_tokens:
                        tasks.append((str(shard), int(offset), int(self.chunk_tokens), int(overlap)))
                        offset += self.chunk_tokens
                        shard_tasks += 1
                    
                    pbar.set_postfix({"Size": f"{size_bytes/1e6:.1f}MB", "Tasks": shard_tasks})
                except Exception as e:
                    logger.error(f"Error processing shard {shard}: {e}")
                    continue
        
        print(f"✅ Created {len(tasks)} chunk tasks from {total_size_gb:.2f}GB of data")
        
        # Process with detailed collection
        sampled_spans_by_shard = defaultdict(list)
        completed_tasks = 0
        failed_tasks = 0
        
        # Use same parallel processing as main pipeline
        if self.use_gpu:
            executor_ctx = self._create_hybrid_executor(banks)
        else:
            executor_ctx = ProcessPoolExecutor(max_workers=self.n_workers, 
                                             initializer=_init_worker, 
                                             initargs=(banks, self.use_gpu))
        
        with executor_ctx as ex:
            optimal_batch = max(batch_size, min(len(tasks), self.gpu_workers * 4 if self.use_gpu else self.n_workers * 2))
            
            with tqdm(total=len(tasks), desc="⚡ Processing chunks", unit="chunks") as pbar:
                for i in range(0, len(tasks), optimal_batch):
                    batch_tasks = tasks[i:i + optimal_batch]
                    batch_futures = {ex.submit(shard_chunk_worker_safe, task): task for task in batch_tasks}
                    
                    for fut in as_completed(batch_futures.keys()):
                        try:
                            res = fut.result(timeout=600)
                            completed_tasks += 1
                            
                            if 'error' not in res:
                                counts_by_n = res['counts']
                                spans = res['spans']
                                
                                # Collect frequency data
                                for n, (ids, cnts) in counts_by_n.items():
                                    bank = banks[n]
                                    for idx, count in zip(ids, cnts):
                                        void_bytes = bank['voids'][idx].tobytes()
                                        toks = np.frombuffer(void_bytes, dtype=np.uint16, count=n)
                                        ngram_text = self.tokenizer.decode(toks.tolist()).lower().strip()
                                        ngram_data[ngram_text]['frequency'] += count
                                
                                if spans:
                                    shard_path = batch_futures[fut][0]
                                    sampled_spans_by_shard[shard_path].extend(spans)
                            
                            pbar.update(1)
                        except Exception as e:
                            failed_tasks += 1
                            pbar.update(1)
                    
                    # Memory cleanup
                    if self.use_gpu and self.device.type == 'cuda':
                        torch.cuda.empty_cache()
                    gc.collect()
        
        # Collect detailed sentence examples with positions
        if collect_positions:
            print(f"\n📄 Collecting detailed sentence examples with positions...")
            self._collect_detailed_examples(sampled_spans_by_shard, ngram_data, max_sentences_per_shard, probe_phrases)
        
        # Harvest sentences for next pass
        all_sentences = self._harvest_sentences_simple(sampled_spans_by_shard, max_sentences_per_shard * len(shard_paths))
        
        return {
            'ngram_data': dict(ngram_data),
            'sentences': all_sentences
        }
    
    def _collect_detailed_examples(self, sampled_spans_by_shard, ngram_data, max_sentences_per_shard, probe_phrases):
        """Collect detailed examples with positions for each n-gram."""
        
        phrase_to_positions = defaultdict(list)  # Track where each phrase appears
        
        with tqdm(sampled_spans_by_shard.items(), desc="📍 Collecting positions", unit="shards") as pbar:
            for shard_str, spans in pbar:
                if not spans:
                    continue
                
                try:
                    shard_path = Path(shard_str)
                    with open(shard_path, "rb") as f:
                        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                            file_tokens = len(mm) // 2
                            
                            for s, e, n, void_bytes in spans[:max_sentences_per_shard]:
                                # Get sentence context
                                context_start = max(0, s - 100)
                                context_end = min(file_tokens, e + 100)
                                context_toks = np.frombuffer(mm[context_start*2:context_end*2], dtype=np.uint16)
                                full_sentence = self.tokenizer.decode(context_toks.tolist()).strip()
                                
                                # Get the specific n-gram
                                ngram_toks = np.frombuffer(void_bytes, dtype=np.uint16, count=n)
                                ngram_text = self.tokenizer.decode(ngram_toks.tolist()).lower().strip()
                                
                                # Find position of n-gram in sentence
                                sentence_lower = full_sentence.lower()
                                ngram_positions = []
                                
                                # Find all occurrences of the n-gram in the sentence
                                start_pos = 0
                                while True:
                                    pos = sentence_lower.find(ngram_text, start_pos)
                                    if pos == -1:
                                        break
                                    ngram_positions.append((pos, pos + len(ngram_text)))
                                    start_pos = pos + 1
                                
                                # Add to examples if positions found
                                if ngram_positions and ngram_text in ngram_data:
                                    example = {
                                        'sentence': full_sentence,
                                        'positions': ngram_positions
                                    }
                                    
                                    # Limit examples per n-gram
                                    if len(ngram_data[ngram_text]['examples']) < 20:
                                        ngram_data[ngram_text]['examples'].append(example)
                
                except Exception as e:
                    logger.warning(f"Error collecting examples from {shard_str}: {e}")
                    continue
    
    def _create_hybrid_executor(self, banks):
        """
        Create a hybrid executor that maximizes GPU+CPU+I/O parallelism.
        Uses separate thread pools for I/O and GPU compute to prevent blocking.
        """
        from concurrent.futures import ThreadPoolExecutor
        from contextlib import ExitStack
        
        class HybridExecutor:
            def __init__(self, io_workers, gpu_workers, banks, use_gpu):
                self.stack = ExitStack()
                # Separate pools for I/O and compute to prevent GPU blocking
                self.io_pool = self.stack.enter_context(ThreadPoolExecutor(max_workers=io_workers, thread_name_prefix="IO"))
                self.compute_pool = self.stack.enter_context(ThreadPoolExecutor(max_workers=gpu_workers, thread_name_prefix="GPU"))
                
                # Initialize GPU workers
                _init_worker(banks, use_gpu)
                
                # Pre-warm GPU threads by submitting dummy tasks
                print(f"🔥 Pre-warming {gpu_workers} GPU compute threads...")
                dummy_futures = []
                for i in range(gpu_workers):
                    dummy_task = ("", 0, 1000, 10)  # Small dummy task
                    future = self.compute_pool.submit(self._dummy_gpu_warmup, dummy_task)
                    dummy_futures.append(future)
                
                # Wait for warmup to complete
                for future in dummy_futures:
                    try:
                        future.result(timeout=5)
                    except:
                        pass  # Ignore dummy task failures
                
                print(f"✅ GPU threads pre-warmed and ready")
            
            def submit(self, fn, task):
                # Use compute pool for actual processing (GPU accelerated)
                return self.compute_pool.submit(fn, task)
            
            def __enter__(self):
                return self
            
            def __exit__(self, *args):
                self.stack.close()
            
            def _dummy_gpu_warmup(self, task):
                """Dummy task to pre-warm GPU threads"""
                try:
                    # Just allocate small GPU tensor to warm up CUDA context
                    if torch.cuda.is_available():
                        dummy = torch.ones(100, device=DEVICE)
                        dummy.sum()
                        del dummy
                        torch.cuda.empty_cache()
                except:
                    pass  # Ignore warmup errors
                return {'counts': {}, 'spans': []}
        
        return HybridExecutor(self.io_workers, self.gpu_workers, banks, self.use_gpu)

# --------------------
# Example usage
# --------------------
if __name__ == '__main__':
    # Set multiprocessing start method
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    # Example usage with enhanced multi-pass pipeline
    try:
        shard_paths = [
            Path(f"./data/datasets--EleutherAI--pile-deduped-pythia-preshuffled/snapshots/4647773ea142ab1ff5694602fa104bbf49088408/document-{i:05d}-of-00020.bin") 
            for i in range(3)
        ]
        
        # Filter existing files
        shard_paths = [p for p in shard_paths if p.exists()]
        
        if not shard_paths:
            print("❌ No shard files found. Please check the paths.")
            print("   Expected paths like: ./data/datasets--EleutherAI--pile-deduped-pythia-preshuffled/...")
            exit(1)
        
        # Country-capital seed n-grams
        initial_probes = [
            "paris france", "tokyo japan", "london england", "berlin germany",
            "moscow russia", "beijing china", "rome italy", "madrid spain",
            "washington america", "ottawa canada", "mexico city mexico"
        ]
        
        print(f"🌍 Starting country-capital n-gram analysis")
        print(f"🔍 Initial seed phrases: {len(initial_probes)}")
        
        # Initialize miner with RTX 5090 maximum performance settings
        miner = FastNgramMiner(
            chunk_tokens=100_000_000,   # Massive 100M token chunks for RTX 5090
            n_workers=28,               # Leave 4 cores for system (32-4)
            gpu_workers=16,             # 16 dedicated GPU compute threads
            io_workers=64,              # 64 I/O threads for maximum throughput
            use_gpu=True,               # Enable GPU acceleration
            max_memory_gb=130.0         # Use 130GB of 140GB VRAM
        )
        
        # Create structured n-gram dataset with 2 passes
        dataset = miner.create_ngram_dataset(
            shard_paths=shard_paths,
            initial_probe_phrases=initial_probes,
            max_sentences_per_shard=5000,      # More context extraction
            batch_size=64,                     # Large batches for GPU efficiency  
            min_frequency_threshold=5,         # Lower threshold for inclusion
            max_new_ngrams=1000                # New n-grams to extract in pass 2
        )
        
        print(f"\n🎯 DEMO: Dataset Structure and Contents")
        print(f"{'='*60}")
        
        # Show dataset metadata
        metadata = dataset['metadata']
        print(f"\n📊 Dataset Metadata:")
        print(f"   🔢 Total n-grams: {metadata['total_ngrams']:,}")
        print(f"   📄 Unique sentences: {metadata['total_sentences']:,}")
        print(f"   ⏱️  Processing time: {metadata['processing_time']:.2f}s")
        print(f"   🚀 Pass 1: {metadata['pass1_time']:.2f}s, Pass 2: {metadata['pass2_time']:.2f}s")
        print(f"   🌱 Original seed phrases: {len(metadata['initial_phrases'])}")
        
        # Show sample n-grams from dataset
        print(f"\n🌟 Sample N-grams with Examples:")
        sample_ngrams = sorted(dataset['ngrams'].items(), key=lambda x: -x[1]['frequency'])[:5]
        
        for i, (ngram, data) in enumerate(sample_ngrams, 1):
            print(f"\n   {i}. N-gram: '{ngram}'")
            print(f"      🔢 Frequency: {data['frequency']:,} occurrences")
            print(f"      📄 Examples: {len(data['examples'])} sentences")
            
            # Show first example with positions
            if data['examples']:
                example = data['examples'][0]
                sentence = example['sentence']
                positions = example['positions']
                
                print(f"      📝 Sample sentence: \"{sentence[:120]}{'...' if len(sentence) > 120 else ''}\"")
                print(f"      📍 Positions in sentence: {positions}")
                
                # Show the n-gram highlighted in context
                if positions:
                    start, end = positions[0]
                    before = sentence[:start]
                    ngram_part = sentence[start:end]
                    after = sentence[end:]
                    print(f"      🎯 Highlighted: \"{before}[{ngram_part}]{after[:50]}{'...' if len(after) > 50 else ''}\"")
        
        # Show frequency distribution
        frequencies = [data['frequency'] for data in dataset['ngrams'].values()]
        if frequencies:
            print(f"\n📈 Frequency Statistics:")
            print(f"   📊 Min frequency: {min(frequencies)}")
            print(f"   📊 Max frequency: {max(frequencies):,}")
            print(f"   📊 Average frequency: {sum(frequencies) / len(frequencies):.1f}")
            
            # Show frequency distribution buckets
            high_freq = sum(1 for f in frequencies if f >= 100)
            med_freq = sum(1 for f in frequencies if 10 <= f < 100)
            low_freq = sum(1 for f in frequencies if f < 10)
            
            print(f"   📊 High frequency (≥100): {high_freq} n-grams")
            print(f"   📊 Medium frequency (10-99): {med_freq} n-grams") 
            print(f"   📊 Low frequency (<10): {low_freq} n-grams")
        
        # Save dataset to JSON file
        import json
        output_path = Path("country_capital_ngram_dataset.json")
        print(f"\n💾 Saving dataset to: {output_path}")
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(dataset, f, indent=2, ensure_ascii=False)
            print(f"✅ Dataset saved successfully!")
            print(f"   📁 File size: {output_path.stat().st_size / 1e6:.1f} MB")
        except Exception as e:
            print(f"❌ Error saving dataset: {e}")
        
        print(f"\n🎉 Dataset creation completed!")
        print(f"   Use: dataset['ngrams'][ngram]['frequency'] to get frequency")
        print(f"   Use: dataset['ngrams'][ngram]['examples'] to get sentences and positions")
        
    except Exception as e:
        logger.error(f"Example execution failed: {e}")
        import traceback
        traceback.print_exc()
