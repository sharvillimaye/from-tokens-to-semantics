from __future__ import annotations

import json
import math
import os
import queue
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from datasets import load_dataset
from loguru import logger
from transformers import AutoTokenizer


try:
    import cupy as cp
except Exception as exc:  # pragma: no cover - explicit error for missing GPU deps
    raise RuntimeError(
        "CuPy is required for GPU n-gram processing. Install a CUDA-enabled CuPy wheel, e.g. 'cupy-cuda12x'."
    ) from exc


@dataclass
class PipelineConfig:
    dataset_name: str = "pietrolesci/pile-deduped-pythia-preshuffled"
    ngram_size: int = 2
    max_queue_size: int = 100
    num_workers: int = 16
    batch_size: int = 6  # number of chunks per GPU batch
    context_window: int = 100
    max_contexts_per_ngram: int = 4
    max_chunks: Optional[int] = None  # for debugging; None = stream all
    output_path: str = "pythia_ngrams.jsonl"  # newline-delimited JSON for scalability


class HFDatasetStreamer:
    """Streams pre-tokenized samples from HuggingFace with backpressure via a queue.

    Assumes each record contains 'input_ids' (List[int]). If a checkpoint id/metadata is
    present (e.g., 'checkpoint_id'), it is forwarded; otherwise a monotonic index is used.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.dataset = load_dataset(self.config.dataset_name, streaming=True)
        self.chunk_queue: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=self.config.max_queue_size)
        self._producer_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    def _producer(self) -> None:
        sent = 0
        # Prefer 'train' split if available; otherwise iterate over all splits
        splits = ["train"] if "train" in self.dataset else list(self.dataset.keys())
        for split in splits:
            for idx, record in enumerate(self.dataset[split]):
                if self._stop_event.is_set():
                    break
                input_ids = record.get("input_ids")
                if input_ids is None:
                    continue
                checkpoint_id = record.get("checkpoint_id", record.get("epoch", None))
                metadata = {
                    "checkpoint_id": int(checkpoint_id) if checkpoint_id is not None else idx,
                    "num_tokens": len(input_ids),
                    "split": split,
                    "row_index": idx,
                }
                sample = {"input_ids": input_ids, "metadata": metadata}
                self.chunk_queue.put(sample)
                sent += 1
                if self.config.max_chunks is not None and sent >= self.config.max_chunks:
                    break
            if self.config.max_chunks is not None and sent >= self.config.max_chunks:
                break
        # Signal completion
        self.chunk_queue.put(None)  # type: ignore[arg-type]

    def start(self) -> None:
        if self._producer_thread is None:
            self._producer_thread = threading.Thread(target=self._producer, daemon=True)
            self._producer_thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._producer_thread is not None:
            self._producer_thread.join(timeout=5)


class CPUWorker:
    """Prepares batches for GPU: padding, dtype normalization, and metadata staging."""

    def __init__(self, ngram_size: int) -> None:
        self.ngram_size = ngram_size

    def process_chunks(self, chunk_batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Filter out too-short sequences early to avoid GPU work
        filtered = [c for c in chunk_batch if len(c["input_ids"]) >= self.ngram_size]
        if not filtered:
            return {
                "tokens": None,
                "lengths": [],
                "metas": [],
            }

        lengths = [len(c["input_ids"]) for c in filtered]
        max_len = max(lengths)
        batch_size = len(filtered)
        tokens_np = np.full((batch_size, max_len), fill_value=-1, dtype=np.int32)
        metas: List[Dict[str, Any]] = []
        for i, sample in enumerate(filtered):
            ids = np.asarray(sample["input_ids"], dtype=np.int32)
            tokens_np[i, : ids.shape[0]] = ids
            metas.append(sample["metadata"])
        return {
            "tokens": tokens_np,
            "lengths": lengths,
            "metas": metas,
        }


class ThreadManager:
    """Coordinates CPU workers to form GPU-ready batches from the stream."""

    def __init__(self, num_workers: int, ngram_size: int) -> None:
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
        self.worker = CPUWorker(ngram_size=ngram_size)

    def to_gpu_ready(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Simple single worker for the given batch; can shard if needed
        return self.worker.process_chunks(chunks)


class GPUMemoryManager:
    """Configures CuPy memory pool and optionally pre-allocates reusable buffers."""

    def __init__(self, vram_limit_gb: int = 64) -> None:
        self.memory_pool = cp.get_default_memory_pool()
        self.pinned_memory_pool = cp.get_default_pinned_memory_pool()
        try:
            self.memory_pool.set_limit(size=vram_limit_gb * 1024**3)
        except Exception:
            # Some CuPy versions may not support dynamic limits; proceed best-effort
            pass


class VectorizedNGramProcessor:
    """Core GPU processor to compute n-gram hashes, counts, and positions in batch.

    Uses cp.lib.stride_tricks.sliding_window_view for vectorized window extraction.
    """

    def __init__(self, ngram_size: int) -> None:
        self.ngram_size = ngram_size

    def sliding_window_ngrams(self, tokens_gpu: cp.ndarray, ngram_size: Optional[int] = None) -> cp.ndarray:
        # tokens_gpu shape: (batch, seq_len)
        size = int(self.ngram_size if ngram_size is None else ngram_size)
        windows = cp.lib.stride_tricks.sliding_window_view(
            tokens_gpu, window_shape=size, axis=1
        )
        return windows  # shape: (batch, num_windows, ngram_size)

    def _hash_ngrams(self, ngram_windows: cp.ndarray) -> cp.ndarray:
        """FNV-1a 64-bit hash per n-gram window for stable CPU/GPU consistency.

        Returns shape: (batch, num_windows) of dtype uint64.
        """
        tokens = ngram_windows.astype(cp.uint64)
        h = cp.uint64(1469598103934665603)
        prime = cp.uint64(1099511628211)
        # Initialize with offset for each window
        hashed = cp.full(tokens.shape[:-1], h, dtype=cp.uint64)
        for j in range(tokens.shape[-1]):
            hashed = (hashed ^ tokens[..., j]) * prime
        return hashed

    def extract_ngrams_gpu(
        self,
        token_arrays_batch: np.ndarray,
        valid_lengths: List[int],
    ) -> Dict[str, Any]:
        if token_arrays_batch is None or token_arrays_batch.size == 0:
            return {
                "global_counts": {},
                "per_sample": [],
            }

        tokens_gpu = cp.asarray(token_arrays_batch)
        windows = self.sliding_window_ngrams(tokens_gpu, ngram_size=self.ngram_size)
        hashed = self._hash_ngrams(windows)

        per_sample_results: List[Dict[str, Any]] = []
        global_counts: Dict[int, int] = defaultdict(int)

        batch_size = hashed.shape[0]
        for i in range(batch_size):
            seq_len = valid_lengths[i]
            num_windows = max(0, seq_len - self.ngram_size + 1)
            if num_windows <= 0:
                per_sample_results.append({"counts": {}, "positions": {}})
                continue
            hashed_i = hashed[i, :num_windows]
            unique_vals, inverse, counts = cp.unique(
                hashed_i, return_inverse=True, return_counts=True
            )
            # Build positions map: value -> list of positions (on GPU)
            positions_map: Dict[int, List[int]] = {}
            # Grouping by label on GPU; iterate unique values for clarity
            for label, val in enumerate(unique_vals.tolist()):
                mask = (inverse == label)
                pos = cp.nonzero(mask)[0]
                positions_map[int(val)] = pos.get().tolist()
                global_counts[int(val)] += int(counts[label].get())
            counts_map = {int(val): int(cnt.get()) for val, cnt in zip(unique_vals.tolist(), counts)}
            per_sample_results.append({"counts": counts_map, "positions": positions_map})

        return {
            "global_counts": dict(global_counts),
            "per_sample": per_sample_results,
        }

    def extract_selected_ngrams_gpu(
        self,
        token_arrays_batch: np.ndarray,
        valid_lengths: List[int],
        size_to_target_hashes: Dict[int, List[int]],
    ) -> Dict[str, Any]:
        if token_arrays_batch is None or token_arrays_batch.size == 0:
            return {
                "size_to_global_counts": {},
                "size_to_per_sample": {},
            }

        tokens_gpu = cp.asarray(token_arrays_batch)
        size_to_global_counts: Dict[int, Dict[int, int]] = {}
        size_to_per_sample: Dict[int, List[Dict[str, Any]]] = {}

        for size, target_hashes in size_to_target_hashes.items():
            if not target_hashes:
                continue
            windows = self.sliding_window_ngrams(tokens_gpu, ngram_size=size)
            hashed = self._hash_ngrams(windows)
            global_counts: Dict[int, int] = defaultdict(int)
            per_sample_results: List[Dict[str, Any]] = []
            batch_size = hashed.shape[0]
            target_hashes_gpu = cp.asarray(target_hashes, dtype=cp.uint64)
            for i in range(batch_size):
                seq_len = valid_lengths[i]
                num_windows = max(0, seq_len - size + 1)
                if num_windows <= 0:
                    per_sample_results.append({"counts": {}, "positions": {}})
                    continue
                hashed_i = hashed[i, :num_windows]
                counts_map: Dict[int, int] = {}
                positions_map: Dict[int, List[int]] = {}
                # For each target hash, find positions where it appears
                for h in target_hashes:
                    h64 = cp.uint64(h)
                    mask = (hashed_i == h64)
                    cnt = int(mask.sum().get())
                    if cnt > 0:
                        pos = cp.nonzero(mask)[0]
                        positions_map[int(h)] = pos.get().tolist()
                        counts_map[int(h)] = cnt
                        global_counts[int(h)] += cnt
                per_sample_results.append({"counts": counts_map, "positions": positions_map})

            size_to_global_counts[size] = dict(global_counts)
            size_to_per_sample[size] = per_sample_results

        return {
            "size_to_global_counts": size_to_global_counts,
            "size_to_per_sample": size_to_per_sample,
        }


class ContextExtractor:
    """Converts token positions back into readable context windows using GPT-NeoX tokenizer."""

    def __init__(self, context_window: int) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.context_window = context_window

    def extract_context_sentences(
        self,
        ngram_hashes: Iterable[int],
        per_sample_positions: List[Dict[str, List[int]]],
        token_arrays: np.ndarray,
        ngram_size: int,
        max_contexts_per_ngram: int,
        metas: List[Dict[str, Any]],
    ) -> Dict[int, List[Dict[str, Any]]]:
        results: Dict[int, List[Dict[str, Any]]] = defaultdict(list)

        for sample_idx, sample_positions in enumerate(per_sample_positions):
            tokens = token_arrays[sample_idx]
            valid_len = int(np.count_nonzero(tokens != -1))
            if valid_len <= 0:
                continue
            for ngram_hash in ngram_hashes:
                positions = sample_positions.get(ngram_hash, [])
                if not positions:
                    continue
                # Limit contexts per n-gram per sample for memory control
                for pos in positions[:max_contexts_per_ngram]:
                    start = max(0, pos - self.context_window)
                    end = min(valid_len, pos + ngram_size + self.context_window)
                    context_token_ids = tokens[start:end]
                    # Extract sentence-like string via simple decode
                    sentence = self.tokenizer.decode(
                        context_token_ids[context_token_ids != -1].tolist(),
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True,
                    )
                    # Human-readable n-gram text
                    ngram_text_ids = tokens[pos : pos + ngram_size]
                    ngram_text = self.tokenizer.decode(
                        ngram_text_ids[ngram_text_ids != -1].tolist(),
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True,
                    )
                    results[ngram_hash].append(
                        {
                            "checkpoint_id": metas[sample_idx]["checkpoint_id"],
                            "position": int(pos),
                            "sentence": sentence,
                            "ngram_text": ngram_text,
                        }
                    )

        return results


class JSONResultBuilder:
    """Aggregates and writes results in a scalable JSONL format."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.meta = {
            "metadata": {
                "dataset": self.config.dataset_name,
                "total_checkpoints": 142,  # informational; not strictly enforced here
                "ngram_size": self.config.ngram_size,
                "processing_date": datetime.now().isoformat(),
            }
        }
        self.global_frequency: Dict[int, int] = defaultdict(int)

        # Prepare output file
        if os.path.exists(self.config.output_path):
            os.remove(self.config.output_path)

    def write_batch(
        self,
        gpu_counts: Dict[int, int],
        batch_per_sample: List[Dict[str, Any]],
        contexts: Dict[int, List[Dict[str, Any]]],
        metas: List[Dict[str, Any]],
    ) -> None:
        # Update global frequency
        for k, v in gpu_counts.items():
            self.global_frequency[k] += v

        # Build batch JSONL entries per n-gram hash
        by_checkpoint: List[int] = [m["checkpoint_id"] for m in metas]
        batch_frequencies: Dict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for sample_idx, sample in enumerate(batch_per_sample):
            ckpt = str(by_checkpoint[sample_idx])
            for ngram_hash, count in sample.get("counts", {}).items():
                batch_frequencies[ngram_hash][f"checkpoint_{ckpt}"] += int(count)

        with open(self.config.output_path, "a", encoding="utf-8") as f:
            for ngram_hash, per_ckpt_counts in batch_frequencies.items():
                contexts_list = contexts.get(ngram_hash, [])
                ngram_text = contexts_list[0]["ngram_text"] if contexts_list else ""
                record = {
                    "ngram_hash": int(ngram_hash),
                    "ngram_text": ngram_text,
                    "global_frequency": int(self.global_frequency[ngram_hash]),
                    "batch_frequencies": dict(per_ckpt_counts),
                    "contexts": contexts_list,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def write_selected_batch(
        self,
        size_to_counts: Dict[int, Dict[int, int]],
        size_to_per_sample: Dict[int, List[Dict[str, Any]]],
        size_to_contexts: Dict[int, Dict[int, List[Dict[str, Any]]]],
        metas: List[Dict[str, Any]],
    ) -> None:
        # Update global frequencies
        for _size, counts in size_to_counts.items():
            for k, v in counts.items():
                self.global_frequency[k] += v

        by_checkpoint: List[int] = [m["checkpoint_id"] for m in metas]

        with open(self.config.output_path, "a", encoding="utf-8") as f:
            for size, per_sample in size_to_per_sample.items():
                # Build per-ckpt frequencies for this size
                batch_frequencies: Dict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
                for sample_idx, sample in enumerate(per_sample):
                    ckpt = str(by_checkpoint[sample_idx])
                    for ngram_hash, count in sample.get("counts", {}).items():
                        batch_frequencies[ngram_hash][f"checkpoint_{ckpt}"] += int(count)

                contexts_by_hash = size_to_contexts.get(size, {})
                for ngram_hash, per_ckpt_counts in batch_frequencies.items():
                    contexts_list = contexts_by_hash.get(ngram_hash, [])
                    ngram_text = contexts_list[0]["ngram_text"] if contexts_list else ""
                    record = {
                        "ngram_size": size,
                        "ngram_hash": int(ngram_hash),
                        "ngram_text": ngram_text,
                        "global_frequency": int(self.global_frequency[ngram_hash]),
                        "batch_frequencies": dict(per_ckpt_counts),
                        "contexts": contexts_list,
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")


class PythiaNgramPipeline:
    """End-to-end orchestrator for streaming, GPU n-gram mining, context extraction, and JSON output."""

    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        self.config = config or PipelineConfig()
        self.streamer = HFDatasetStreamer(self.config)
        self.thread_manager = ThreadManager(num_workers=self.config.num_workers, ngram_size=self.config.ngram_size)
        self.gpu_memory = GPUMemoryManager()
        self.gpu_processor = VectorizedNGramProcessor(ngram_size=self.config.ngram_size)
        self.context_extractor = ContextExtractor(context_window=self.config.context_window)
        self.result_builder = JSONResultBuilder(self.config)

    def _consume_stream_to_batches(self) -> Iterable[List[Dict[str, Any]]]:
        current_batch: List[Dict[str, Any]] = []
        while True:
            item = self.streamer.chunk_queue.get()
            if item is None:
                if current_batch:
                    yield current_batch
                break
            current_batch.append(item)
            if len(current_batch) >= self.config.batch_size:
                yield current_batch
                current_batch = []

    def run(self) -> None:
        logger.info("Starting dataset streaming...")
        self.streamer.start()

        try:
            for chunk_batch in self._consume_stream_to_batches():
                # CPU preprocessing
                gpu_ready = self.thread_manager.to_gpu_ready(chunk_batch)
                tokens_np = gpu_ready["tokens"]
                lengths: List[int] = gpu_ready["lengths"]
                metas: List[Dict[str, Any]] = gpu_ready["metas"]
                if tokens_np is None or len(lengths) == 0:
                    continue

                # GPU processing for configured ngram size
                gpu_results = self.gpu_processor.extract_ngrams_gpu(tokens_np, lengths)
                global_counts: Dict[int, int] = gpu_results["global_counts"]
                per_sample: List[Dict[str, Any]] = gpu_results["per_sample"]

                # Select which n-grams to extract contexts for: use top by frequency in batch
                if not global_counts:
                    continue
                sorted_ngrams = sorted(global_counts.items(), key=lambda kv: kv[1], reverse=True)
                top_hashes = [h for h, _ in sorted_ngrams[: 256]]  # cap contexts per batch

                per_sample_positions = [s.get("positions", {}) for s in per_sample]
                contexts = self.context_extractor.extract_context_sentences(
                    ngram_hashes=top_hashes,
                    per_sample_positions=per_sample_positions,
                    token_arrays=tokens_np,
                    ngram_size=self.config.ngram_size,
                    max_contexts_per_ngram=self.config.max_contexts_per_ngram,
                    metas=metas,
                )

                # Write batch to JSONL
                self.result_builder.write_batch(
                    gpu_counts=global_counts,
                    batch_per_sample=per_sample,
                    contexts=contexts,
                    metas=metas,
                )

        finally:
            self.streamer.stop()
            logger.info("Pipeline finished. Output written to {}", self.config.output_path)

    def run_for_selected_ngrams(self, ngrams: List[List[int]]) -> None:
        """Accepts a set of specific n-grams (as token id lists), computes positions and batchwise
        frequencies for uni/bi/tri-grams, and writes JSONL with contexts.

        Example input: [[42], [10, 20], [3, 5, 7]]
        """
        # Precompute target hashes per size on CPU to match GPU hashing (FNV-1a 64-bit)
        size_to_target_hashes: Dict[int, List[int]] = defaultdict(list)
        for seq in ngrams:
            if not (1 <= len(seq) <= 3):
                continue
            size = len(seq)
            h = np.uint64(1469598103934665603)
            prime = np.uint64(1099511628211)
            for tok in seq:
                h = (h ^ np.uint64(np.int64(tok))) * prime
            h = int(h)
            size_to_target_hashes[size].append(h)

        logger.info("Starting dataset streaming (selected n-grams)...")
        self.streamer.start()

        try:
            for chunk_batch in self._consume_stream_to_batches():
                gpu_ready = self.thread_manager.to_gpu_ready(chunk_batch)
                tokens_np = gpu_ready["tokens"]
                lengths: List[int] = gpu_ready["lengths"]
                metas: List[Dict[str, Any]] = gpu_ready["metas"]
                if tokens_np is None or len(lengths) == 0:
                    continue

                # GPU processing filtering to selected hashes for sizes 1..3
                selected_gpu_results = self.gpu_processor.extract_selected_ngrams_gpu(
                    token_arrays_batch=tokens_np,
                    valid_lengths=lengths,
                    size_to_target_hashes=size_to_target_hashes,
                )

                size_to_counts: Dict[int, Dict[int, int]] = selected_gpu_results["size_to_global_counts"]
                size_to_per_sample: Dict[int, List[Dict[str, Any]]] = selected_gpu_results["size_to_per_sample"]

                # Build contexts per size using positions
                size_to_contexts: Dict[int, Dict[int, List[Dict[str, Any]]]] = {}
                for size, per_sample in size_to_per_sample.items():
                    per_sample_positions = [s.get("positions", {}) for s in per_sample]
                    # Only for the selected hashes of this size
                    hashes = size_to_target_hashes.get(size, [])
                    contexts = self.context_extractor.extract_context_sentences(
                        ngram_hashes=hashes,
                        per_sample_positions=per_sample_positions,
                        token_arrays=tokens_np,
                        ngram_size=size,
                        max_contexts_per_ngram=self.config.max_contexts_per_ngram,
                        metas=metas,
                    )
                    size_to_contexts[size] = contexts

                # Write JSONL entries for the selected n-grams (all sizes)
                self.result_builder.write_selected_batch(
                    size_to_counts=size_to_counts,
                    size_to_per_sample=size_to_per_sample,
                    size_to_contexts=size_to_contexts,
                    metas=metas,
                )

        finally:
            self.streamer.stop()
            logger.info("Selected n-gram pipeline finished. Output written to {}", self.config.output_path)

    def mine_selected_ngrams(self, ngrams: List[List[int]]) -> Dict[str, Any]:
        """Mine positions and per-checkpoint frequencies for provided uni/bi/tri-grams.

        Args:
            ngrams: List of token id sequences, each of length 1..3

        Returns:
            Dict with metadata and list of n-gram records containing tokens, text, global_frequency,
            batch_frequencies per checkpoint, and sampled contexts.
        """
        # Build target hash sets and info
        size_to_target_hashes: Dict[int, List[int]] = defaultdict(list)
        hash_to_info: Dict[int, Dict[str, Any]] = {}
        tokenizer = self.context_extractor.tokenizer
        # For final JSON fields

        for seq in ngrams:
            if not (1 <= len(seq) <= 3):
                continue
            size = len(seq)
            h = np.uint64(1469598103934665603)
            prime = np.uint64(1099511628211)
            for tok in seq:
                h = (h ^ np.uint64(np.int64(tok))) * prime
            h = int(h)
            if h not in hash_to_info:
                text = tokenizer.decode(seq, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                hash_to_info[h] = {"tokens": list(map(int, seq)), "size": size, "text": text}
            size_to_target_hashes[size].append(h)

        # Aggregation structures
        global_counts: Dict[int, int] = defaultdict(int)
        per_checkpoint_counts: Dict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        contexts_by_hash: Dict[int, List[Dict[str, Any]]] = defaultdict(list)

        logger.info("Streaming dataset for targeted n-grams ({} total)...", sum(len(v) for v in size_to_target_hashes.values()))
        self.streamer.start()

        try:
            for chunk_batch in self._consume_stream_to_batches():
                gpu_ready = self.thread_manager.to_gpu_ready(chunk_batch)
                tokens_np = gpu_ready["tokens"]
                lengths: List[int] = gpu_ready["lengths"]
                metas: List[Dict[str, Any]] = gpu_ready["metas"]
                if tokens_np is None or len(lengths) == 0:
                    continue

                selected_gpu_results = self.gpu_processor.extract_selected_ngrams_gpu(
                    token_arrays_batch=tokens_np,
                    valid_lengths=lengths,
                    size_to_target_hashes=size_to_target_hashes,
                )

                size_to_counts: Dict[int, Dict[int, int]] = selected_gpu_results["size_to_global_counts"]
                size_to_per_sample: Dict[int, List[Dict[str, Any]]] = selected_gpu_results["size_to_per_sample"]

                # Update global counts and per-checkpoint counts
                by_checkpoint: List[int] = [m["checkpoint_id"] for m in metas]
                for size, per_sample in size_to_per_sample.items():
                    for sample_idx, sample in enumerate(per_sample):
                        ckpt_key = f"checkpoint_{by_checkpoint[sample_idx]}"
                        for h, cnt in sample.get("counts", {}).items():
                            per_checkpoint_counts[h][ckpt_key] += int(cnt)
                for _size, counts in size_to_counts.items():
                    for h, cnt in counts.items():
                        global_counts[h] += int(cnt)

                # Context extraction, honoring global cap per n-gram
                for size, per_sample in size_to_per_sample.items():
                    per_sample_positions = [s.get("positions", {}) for s in per_sample]
                    hashes = size_to_target_hashes.get(size, [])
                    batch_contexts = self.context_extractor.extract_context_sentences(
                        ngram_hashes=hashes,
                        per_sample_positions=per_sample_positions,
                        token_arrays=tokens_np,
                        ngram_size=size,
                        max_contexts_per_ngram=self.config.max_contexts_per_ngram,
                        metas=metas,
                    )
                    for h, ctxs in batch_contexts.items():
                        if not ctxs:
                            continue
                        existing = contexts_by_hash[h]
                        remaining = max(0, self.config.max_contexts_per_ngram - len(existing))
                        if remaining > 0:
                            # Attach n-gram text from hash mapping for consistency
                            for item in ctxs[:remaining]:
                                item["ngram_text"] = hash_to_info.get(h, {}).get("text", item.get("ngram_text", ""))
                            existing.extend(ctxs[:remaining])

        finally:
            self.streamer.stop()

        # Build return JSON structure
        result = {
            "metadata": {
                "dataset": self.config.dataset_name,
                "total_checkpoints": 142,
                "ngram_sizes": sorted(list({info["size"] for info in hash_to_info.values()})),
                "processing_date": datetime.now().isoformat(),
            },
            "ngrams": [],
        }

        for h, info in hash_to_info.items():
            record = {
                "ngram_tokens": info["tokens"],
                "ngram_text": info["text"],
                "global_frequency": int(global_counts.get(h, 0)),
                "batch_frequencies": dict(per_checkpoint_counts.get(h, {})),
                "contexts": contexts_by_hash.get(h, []),
            }
            result["ngrams"].append(record)

        return result


if __name__ == "__main__":
    # Basic CLI via env vars or default config
    target_ngrams = [
        [42],         # uni-gram
        [10, 20],     # bi-gram
        [3, 5, 7],    # tri-gram
    ]

    cfg = PipelineConfig(
        ngram_size=3,            # not used for selected mining; supports sizes 1–3 automatically
        max_chunks=200,          # for testing; set None to process full stream
        max_contexts_per_ngram=4 # per n-gram cap across the run
    )
    pipeline = PythiaNgramPipeline(cfg)
    result_json = pipeline.mine_selected_ngrams(target_ngrams)


