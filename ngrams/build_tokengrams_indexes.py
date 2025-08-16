import os
import time
import psutil
import shutil
import tempfile
import traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple, Optional

from tqdm import tqdm


# Simplified directories for Runpod:
# - Work only on local disk at /scratch
# - Store final .idx files on network storage at /workspace
SCRATCH_DIR = os.environ.get("SCRATCH_DIR", "/scratch")
INDEX_DIR = os.environ.get("INDEX_DIR", "/workspace")


def ensure_dir(path: str) -> None:
    """Create directory if missing."""
    if not path:
        return
    Path(path).mkdir(parents=True, exist_ok=True)


def find_shard_pairs(data_dir: str, index_dir: Optional[str] = INDEX_DIR) -> Tuple[List[Tuple[str, str]], float, float]:
    """Return list of (bin_path, idx_path) with average and total sizes in GB.

    If index_dir is provided, write .idx files there using the same filenames.
    """
    bin_files = sorted(Path(data_dir).glob("document-*-of-*.bin"))
    if index_dir:
        idx_root = Path(index_dir)
        shard_pairs = [(str(f), str(idx_root / f.with_suffix(".idx").name)) for f in bin_files]
    else:
        shard_pairs = [(str(f), str(f.with_suffix(".idx"))) for f in bin_files]
    total_size = sum(f.stat().st_size for f in bin_files) / (1024**3) if bin_files else 0.0
    avg_size = total_size / len(shard_pairs) if shard_pairs else 0.0
    return shard_pairs, avg_size, total_size


def build_single_index(args: Tuple[str, str, int, int, bool]) -> Tuple[int, bool, str, float]:
    """Build index using scratch space: copy → build → move back → cleanup."""
    bin_path, idx_path, shard_id, vocab_size, force = args

    # Late import inside subprocess for compatibility
    try:
        from tokengrams import MemmapIndex  # type: ignore
    except Exception as e:
        return shard_id, False, f"❌ Shard {shard_id}: tokengrams import failed: {e}", 0.0

    scratch_bin = None
    scratch_idx_tmp = None
    final_idx_tmp = None

    try:
        idx_dir = os.path.dirname(idx_path) or "."
        os.makedirs(idx_dir, exist_ok=True)

        if os.path.exists(idx_path) and not force:
            return shard_id, True, f"⏭️ Shard {shard_id}: exists", 0.0

        start = time.time()
        mem_before = psutil.Process().memory_info().rss / (1024**3)

        # Always work on local disk under SCRATCH_DIR
        ensure_dir(SCRATCH_DIR)
        bin_size = os.path.getsize(bin_path)
        pid = os.getpid()
        scratch_dir = SCRATCH_DIR
        scratch_bin = os.path.join(scratch_dir, f"shard-{shard_id}-{pid}.bin")
        scratch_idx_tmp = os.path.join(scratch_dir, f"shard-{shard_id}-{pid}.idx.tmp")

        # Copy corpus to scratch only if scratch is not the same file
        if os.path.abspath(bin_path) != os.path.abspath(scratch_bin):
            shutil.copy2(bin_path, scratch_bin)
        else:
            scratch_bin = bin_path  # already in place

        final_idx_tmp = f"{idx_path}.tmp"

        # Build index to a temp path (either on scratch or directly on target dir)
        MemmapIndex.build(scratch_bin, scratch_idx_tmp, vocab=int(vocab_size), verbose=False)

        # Ensure destination directory still exists (in case of race)
        ensure_dir(idx_dir)

        # Move idx from scratch to a temp next to final, then atomically replace if same filesystem
        try:
            if os.path.abspath(os.path.dirname(scratch_idx_tmp)) != os.path.abspath(idx_dir):
                shutil.move(scratch_idx_tmp, final_idx_tmp)
                os.replace(final_idx_tmp, idx_path)
            else:
                os.replace(scratch_idx_tmp, idx_path)
        finally:
            if final_idx_tmp and os.path.exists(final_idx_tmp):
                try:
                    os.remove(final_idx_tmp)
                except Exception:
                    pass

        # Metrics
        mem_after = psutil.Process().memory_info().rss / (1024**3)
        duration = time.time() - start
        idx_size_mb = os.path.getsize(idx_path) / (1024**2)
        return shard_id, True, f"✅ Shard {shard_id}: {duration:.1f}s, {idx_size_mb:.1f}MB, ΔRAM {mem_after - mem_before:.2f}GB", duration

    except Exception as e:
        tb = traceback.format_exc(limit=5)
        return shard_id, False, f"❌ Shard {shard_id}: {e}\n{tb}", 0.0

    finally:
        # Cleanup scratch files
        for f in [scratch_bin, scratch_idx_tmp]:
            try:
                if f and os.path.exists(f) and os.path.abspath(f) != os.path.abspath(bin_path):
                    os.remove(f)
            except Exception:
                pass


def build_all_indexes(
    shard_pairs: List[Tuple[str, str]],
    vocab_size: int,
    max_workers: int = 8,
    force: bool = False,
):
    """Parallel build of indexes using scratch space."""
    print(f"🚀 Building {len(shard_pairs)} indexes with {max_workers} workers")
    print(f"💾 Scratch dir: {SCRATCH_DIR}")

    results = []
    max_workers = max(1, min(int(max_workers), os.cpu_count() or 1))
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(build_single_index, (bin_p, idx_p, i, int(vocab_size), force)): i
            for i, (bin_p, idx_p) in enumerate(shard_pairs)
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Building", smoothing=0.1):
            results.append(fut.result())

    # Summary
    success = sum(1 for _, ok, _, _ in results if ok)
    failed_msgs = [msg for _, ok, msg, _ in results if not ok]
    total_time = sum(dur for _, ok, _, dur in results if ok)

    print(f"\n📊 Summary: {success}/{len(results)} succeeded")
    print(f"⏱️  Total build time (sum of successful shards): {total_time:.1f}s")

    if failed_msgs:
        print("❌ Failures:")
        for msg in failed_msgs:
            print("   ", msg)

    return results


if __name__ == "__main__":
    # Work on /scratch (local disk) and store indices under /workspace (network)
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        vocab_size = int(getattr(tok, "vocab_size", 0) or getattr(tok, "vocab", None) or 0)
        if not vocab_size:
            # Fallback: GPT-NeoX default vocab
            vocab_size = 50432
    except Exception:
        # Avoid large downloads in offline pods; default to common GPT-NeoX vocab size
        vocab_size = 50432

    data_dir = os.environ.get("SHARD_DIR", ".")
    ensure_dir(SCRATCH_DIR)
    ensure_dir(INDEX_DIR)
    shard_pairs, avg, total = find_shard_pairs(data_dir, INDEX_DIR)
    idx_target = INDEX_DIR or "(same as shards)"
    print(f"📂 Found {len(shard_pairs)} shards, avg {avg:.2f} GB, total {total:.2f} GB under {data_dir}")
    print(f"📦 Index target dir: {idx_target}")

    if shard_pairs:
        workers_env = os.environ.get("WORKERS")
        workers = int(workers_env) if workers_env and workers_env.isdigit() else 4
        build_all_indexes(shard_pairs, vocab_size, max_workers=workers, force=False)
    else:
        print("❌ No shards found.")


