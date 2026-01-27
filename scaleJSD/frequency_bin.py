import json
from pathlib import Path
import time

import requests

INFINIGRAM_API_URL = "https://api.infini-gram.io/"
DEFAULT_INDEX = "v4_dolma-v1_7_llama"


def get_ngram_count(ngram: str, index: str = DEFAULT_INDEX, max_retries: int = 100) -> dict:
    """Query infini-gram API to get the count of an n-gram with retry logic."""
    payload = {
        "index": index,
        "query_type": "count",
        "query": ngram,
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(INFINIGRAM_API_URL, json=payload, timeout=30)
            if response.status_code == 429 or response.status_code == 400:
                # Rate limited - exponential backoff starting at 2s
                wait_time = 2 * (2**attempt)
                print(f"  Rate limited ({response.status_code}), waiting {wait_time}s...")
                time.sleep(wait_time)
                continue
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            if attempt < max_retries - 1:
                wait_time = 2 * (2**attempt)
                print(f"  Error: {e}, retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                return {"error": str(e), "count": -1}

    return {"error": "Max retries exceeded", "count": -1}


def extract_frequency(input_dir: Path, index: str = DEFAULT_INDEX):
    """Extract frequency counts for high and low frequency n-grams from JSONL files."""
    files: list[Path] = list(input_dir.glob("*.jsonl"))
    # Skip already processed files
    files = [f for f in files if not f.stem.endswith("_freq")]

    for file in files:
        print(f"Processing {file.name}...")
        results = []

        with file.open("r") as f:
            lines = [line.strip() for line in f if line.strip()]

        for i, line in enumerate(lines):
            item = json.loads(line)
            enriched = infinigram_index(item, index)
            results.append(enriched)

            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(lines)} items")

        output_path = file.parent / f"{file.stem}_freq.jsonl"
        with output_path.open("w") as f:
            for result in results:
                f.write(json.dumps(result) + "\n")

        print(f"  Written {len(results)} items to {output_path.name}")


def infinigram_index(item: dict, index: str = DEFAULT_INDEX) -> dict:
    """Get frequency counts for high and low frequency n-grams in an item."""
    high_ngram = item["high_freq_ngram"]
    low_ngram = item["low_freq_ngram"]

    high_result = get_ngram_count(high_ngram, index)
    # Delay between requests to avoid rate limiting
    time.sleep(2.0)
    low_result = get_ngram_count(low_ngram, index)
    time.sleep(2.0)

    return {
        **item,
        "high_freq_count": high_result.get("count", -1),
        "high_freq_approx": high_result.get("approx", False),
        "low_freq_count": low_result.get("count", -1),
        "low_freq_approx": low_result.get("approx", False),
    }


def deduplicate_dataset_file(input_file: Path) -> Path:
    """Remove duplicate items from a single dataset file based on high/low freq ngram pairs."""
    seen = set()
    output_path = input_file.parent / f"{input_file.stem}_dedup.jsonl"
    count = 0

    with output_path.open("w") as f_out:
        with input_file.open("r") as f_in:
            for line in f_in:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                # Use the ngram pair as a unique key
                key = (item["high_freq_ngram"], item["low_freq_ngram"])
                if key not in seen:
                    f_out.write(json.dumps(item) + "\n")
                    seen.add(key)
                    count += 1

    print(f"  Written {count} unique items to {output_path.name}")
    return output_path


def deduplicate_all_datasets(dataset_dir: Path) -> list[Path]:
    """Deduplicate each dataset file separately."""
    # Get original dataset files (not dedup or freq files)
    files = [
        f
        for f in dataset_dir.glob("*.jsonl")
        if not f.stem.endswith("_freq") and not f.stem.endswith("_dedup")
    ]

    dedup_files = []
    for file in files:
        print(f"Deduplicating {file.name}...")
        dedup_file = deduplicate_dataset_file(file)
        dedup_files.append(dedup_file)

    return dedup_files


def extract_frequency_single_file(input_file: Path, index: str = DEFAULT_INDEX):
    """Extract frequency counts for a single JSONL file."""
    print(f"Processing {input_file.name}...")
    results = []

    with input_file.open("r") as f:
        lines = [line.strip() for line in f if line.strip()]

    for i, line in enumerate(lines):
        item = json.loads(line)
        enriched = infinigram_index(item, index)
        results.append(enriched)

        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(lines)} items")

    output_path = input_file.parent / f"{input_file.stem}_freq.jsonl"
    with output_path.open("w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    print(f"Written {len(results)} items to {output_path.name}")
    return output_path


def filter_by_frequency(
    input_file: Path,
    min_high_freq: int = 1000,
    min_low_freq: int = 100,
    min_ratio: float = 10.0,
) -> Path:
    """
    Filter n-gram pairs by frequency criteria.

    Args:
        input_file: Path to the _freq.jsonl file
        min_high_freq: Minimum count for the high-frequency term
        min_low_freq: Minimum count for the low-frequency term
        min_ratio: Minimum ratio of high_freq_count / low_freq_count

    Returns:
        Path to the filtered output file
    """
    print(f"Filtering {input_file.name}...")

    results = []
    stats = {
        "total": 0,
        "invalid_counts": 0,
        "below_min_high": 0,
        "below_min_low": 0,
        "below_min_ratio": 0,
        "wrong_direction": 0,
        "passed": 0,
    }

    with input_file.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            item = json.loads(line)
            stats["total"] += 1

            high_count = item.get("high_freq_count", -1)
            low_count = item.get("low_freq_count", -1)

            # Check for invalid counts (API errors)
            if high_count <= 0 or low_count <= 0:
                stats["invalid_counts"] += 1
                continue

            # Check if direction is correct (high should be > low)
            if high_count <= low_count:
                stats["wrong_direction"] += 1
                continue

            # Check minimum thresholds
            if high_count < min_high_freq:
                stats["below_min_high"] += 1
                continue

            if low_count < min_low_freq:
                stats["below_min_low"] += 1
                continue

            # Check ratio
            ratio = high_count / low_count
            if ratio < min_ratio:
                stats["below_min_ratio"] += 1
                continue

            # Add ratio to the item for reference
            item["frequency_ratio"] = round(ratio, 2)
            results.append(item)
            stats["passed"] += 1

    # Write filtered results
    output_path = input_file.parent / f"{input_file.stem.replace('_freq', '')}_filtered.jsonl"
    with output_path.open("w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    # Print statistics
    print(f"  Total: {stats['total']}")
    print(f"  Invalid counts (API errors): {stats['invalid_counts']}")
    print(f"  Wrong direction (high <= low): {stats['wrong_direction']}")
    print(f"  Below min high freq ({min_high_freq}): {stats['below_min_high']}")
    print(f"  Below min low freq ({min_low_freq}): {stats['below_min_low']}")
    print(f"  Below min ratio ({min_ratio}x): {stats['below_min_ratio']}")
    print(f"  Passed: {stats['passed']}")
    print(f"  Written to {output_path.name}")

    return output_path


def filter_all_freq_files(
    dataset_dir: Path,
    min_high_freq: int = 1000,
    min_low_freq: int = 100,
    min_ratio: float = 10.0,
) -> list[Path]:
    """Filter all frequency files in the dataset directory."""
    freq_files = list(dataset_dir.glob("*_dedup_freq.jsonl"))

    filtered_files = []
    for freq_file in freq_files:
        filtered_file = filter_by_frequency(
            freq_file,
            min_high_freq=min_high_freq,
            min_low_freq=min_low_freq,
            min_ratio=min_ratio,
        )
        filtered_files.append(filtered_file)

    return filtered_files


if __name__ == "__main__":
    dataset_dir = Path(__file__).parent / "dataset"

    # Step 1: Deduplicate each dataset separately
    print("Step 1: Deduplicating datasets...")
    dedup_files = deduplicate_all_datasets(dataset_dir)

    # Step 2: Extract frequencies from each deduplicated file
    print("\nStep 2: Extracting frequencies...")
    for dedup_file in dedup_files:
        extract_frequency_single_file(dedup_file)
