"""
Analyze and filter n-gram pairs for JSD-based mechanistic interpretability experiments.

This script:
1. Filters n-gram pairs based on frequency criteria
2. Validates the criteria for mechanistic interpretability
3. Outputs the top candidates for the experiment
"""

from dataclasses import dataclass
import json
import math
from pathlib import Path


@dataclass
class FilterCriteria:
    """Criteria for filtering n-gram pairs."""

    min_high_freq: int = 1000
    min_low_freq: int = 100
    min_ratio: float = 10.0


@dataclass
class NGramPair:
    """A pair of high/low frequency n-grams."""

    high_ngram: str
    low_ngram: str
    high_count: int
    low_count: int
    category: str
    shared_meaning: str
    ratio: float
    log_ratio: float

    @classmethod
    def from_dict(cls, d: dict) -> "NGramPair":
        high_count = d.get("high_freq_count", 0)
        low_count = d.get("low_freq_count", 0)
        ratio = high_count / low_count if low_count > 0 else 0
        log_ratio = math.log10(ratio) if ratio > 0 else 0

        return cls(
            high_ngram=d["high_freq_ngram"],
            low_ngram=d["low_freq_ngram"],
            high_count=high_count,
            low_count=low_count,
            category=d.get("category", "unknown"),
            shared_meaning=d.get("shared_meaning", ""),
            ratio=ratio,
            log_ratio=log_ratio,
        )


def load_freq_file(path: Path) -> list[dict]:
    """Load a frequency JSONL file."""
    items = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def filter_ngrams(
    items: list[dict],
    criteria: FilterCriteria,
) -> tuple[list[NGramPair], dict]:
    """
    Filter n-gram pairs based on criteria.

    Returns:
        tuple of (filtered pairs, statistics dict)
    """
    stats = {
        "total": 0,
        "invalid_counts": 0,
        "wrong_direction": 0,
        "below_min_high": 0,
        "below_min_low": 0,
        "below_min_ratio": 0,
        "passed": 0,
    }

    filtered = []

    for item in items:
        stats["total"] += 1

        high_count = item.get("high_freq_count", -1)
        low_count = item.get("low_freq_count", -1)

        # Check for invalid counts
        if high_count <= 0 or low_count <= 0:
            stats["invalid_counts"] += 1
            continue

        # Check direction
        if high_count <= low_count:
            stats["wrong_direction"] += 1
            continue

        # Check minimum thresholds
        if high_count < criteria.min_high_freq:
            stats["below_min_high"] += 1
            continue

        if low_count < criteria.min_low_freq:
            stats["below_min_low"] += 1
            continue

        # Check ratio
        ratio = high_count / low_count
        if ratio < criteria.min_ratio:
            stats["below_min_ratio"] += 1
            continue

        filtered.append(NGramPair.from_dict(item))
        stats["passed"] += 1

    return filtered, stats


def analyze_distribution(pairs: list[NGramPair]) -> dict:
    """Analyze the distribution of frequency ratios."""
    if not pairs:
        return {}

    ratios = [p.ratio for p in pairs]
    log_ratios = [p.log_ratio for p in pairs]

    ratios_sorted = sorted(ratios)
    n = len(ratios_sorted)

    return {
        "count": n,
        "min_ratio": min(ratios),
        "max_ratio": max(ratios),
        "mean_ratio": sum(ratios) / n,
        "median_ratio": ratios_sorted[n // 2],
        "p25_ratio": ratios_sorted[n // 4],
        "p75_ratio": ratios_sorted[3 * n // 4],
        "min_log_ratio": min(log_ratios),
        "max_log_ratio": max(log_ratios),
        "mean_log_ratio": sum(log_ratios) / n,
    }


def validate_for_mechinterp(pairs: list[NGramPair]) -> dict:
    """
    Validate if the filtered pairs are suitable for mechanistic interpretability.

    Key considerations:
    1. Sufficient frequency difference for statistical significance
    2. Both terms appear often enough for reliable embedding analysis
    3. Semantic similarity (same meaning, different frequency)
    """
    if not pairs:
        return {"valid": False, "reason": "No pairs passed filtering"}

    issues = []
    recommendations = []

    # Check sample size
    if len(pairs) < 10:
        issues.append(f"Small sample size ({len(pairs)} pairs)")
        recommendations.append("Consider relaxing filter criteria")

    # Check ratio distribution
    ratios = [p.ratio for p in pairs]
    mean_ratio = sum(ratios) / len(ratios)

    if mean_ratio < 5:
        issues.append(f"Low mean frequency ratio ({mean_ratio:.1f}x)")
        recommendations.append("Frequency differences may be too small for clear JSD signal")

    if mean_ratio > 1000:
        recommendations.append(
            "Very high ratios - low freq terms may be rare enough to have noisy embeddings"
        )

    # Check low frequency counts
    low_counts = [p.low_count for p in pairs]
    min_low = min(low_counts)

    if min_low < 500:
        issues.append(f"Some low-freq terms have very few occurrences ({min_low})")
        recommendations.append(
            "Consider increasing min_low_freq threshold for more reliable analysis"
        )

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "recommendations": recommendations,
        "summary": {
            "num_pairs": len(pairs),
            "mean_ratio": round(mean_ratio, 2),
            "min_low_count": min_low,
        },
    }


def get_top_pairs(
    pairs: list[NGramPair],
    n: int = 20,
    sort_by: str = "ratio",
) -> list[NGramPair]:
    """Get top N pairs sorted by the specified criterion."""
    if sort_by == "ratio":
        return sorted(pairs, key=lambda p: p.ratio, reverse=True)[:n]
    elif sort_by == "log_ratio":
        return sorted(pairs, key=lambda p: p.log_ratio, reverse=True)[:n]
    elif sort_by == "low_count":
        # Higher low_count = more reliable
        return sorted(pairs, key=lambda p: p.low_count, reverse=True)[:n]
    else:
        return pairs[:n]


def print_pairs(pairs: list[NGramPair], title: str = "N-gram Pairs"):
    """Pretty print a list of pairs."""
    print(f"\n{'=' * 80}")
    print(f"{title}")
    print(f"{'=' * 80}")
    print(
        f"{'High Freq Term':<25} {'Count':>12} {'Low Freq Term':<25} {'Count':>12} {'Ratio':>10}"
    )
    print("-" * 80)
    for p in pairs:
        print(
            f"{p.high_ngram:<25} {p.high_count:>12,} {p.low_ngram:<25} {p.low_count:>12,} {p.ratio:>10.1f}x"
        )


def main():
    dataset_dir = Path(__file__).parent / "dataset"
    freq_dir = dataset_dir / "freq"
    filtered_dir = dataset_dir / "filtered"

    # Find all frequency files
    freq_files = list(freq_dir.glob("*_dedup_freq.jsonl"))

    if not freq_files:
        print("No frequency files found. Run frequency_bin.py first.")
        return

    print("=" * 80)
    print("N-GRAM PAIR ANALYSIS FOR JSD MECHANISTIC INTERPRETABILITY")
    print("=" * 80)

    # Define filtering criteria
    criteria = FilterCriteria(
        min_high_freq=1000,
        min_low_freq=100,
        min_ratio=10.0,
    )

    print(f"\nFilter Criteria:")
    print(f"  - Minimum high-freq count: {criteria.min_high_freq:,}")
    print(f"  - Minimum low-freq count: {criteria.min_low_freq:,}")
    print(f"  - Minimum frequency ratio: {criteria.min_ratio}x")

    all_pairs: dict[str, list[NGramPair]] = {}

    # Process each dataset
    for freq_file in sorted(freq_files):
        dataset_name = freq_file.stem.replace("_dedup_freq", "")
        print(f"\n{'=' * 60}")
        print(f"Dataset: {dataset_name}")
        print("=" * 60)

        items = load_freq_file(freq_file)
        pairs, stats = filter_ngrams(items, criteria)

        print(f"\nFiltering Statistics:")
        print(f"  Total items: {stats['total']}")
        print(f"  Invalid counts: {stats['invalid_counts']}")
        print(f"  Wrong direction: {stats['wrong_direction']}")
        print(f"  Below min high freq: {stats['below_min_high']}")
        print(f"  Below min low freq: {stats['below_min_low']}")
        print(f"  Below min ratio: {stats['below_min_ratio']}")
        print(f"  PASSED: {stats['passed']}")

        if pairs:
            dist = analyze_distribution(pairs)
            print(f"\nRatio Distribution:")
            print(f"  Min: {dist['min_ratio']:.1f}x, Max: {dist['max_ratio']:.1f}x")
            print(f"  Mean: {dist['mean_ratio']:.1f}x, Median: {dist['median_ratio']:.1f}x")
            print(f"  Log10 range: {dist['min_log_ratio']:.2f} to {dist['max_log_ratio']:.2f}")

            # Validation for mechinterp
            validation = validate_for_mechinterp(pairs)
            print(f"\nMechInterp Validation:")
            print(f"  Valid: {validation['valid']}")
            if validation["issues"]:
                print(f"  Issues: {', '.join(validation['issues'])}")
            if validation["recommendations"]:
                print(f"  Recommendations:")
                for rec in validation["recommendations"]:
                    print(f"    - {rec}")

            # Show top pairs by ratio
            top_by_ratio = get_top_pairs(pairs, n=5, sort_by="ratio")
            print_pairs(top_by_ratio, f"Top 5 by Frequency Ratio ({dataset_name})")

            all_pairs[dataset_name] = pairs

    # Overall summary
    if all_pairs:
        # Ensure filtered directory exists
        filtered_dir.mkdir(exist_ok=True)

        # Save filtered pairs to file
        for dataset in all_pairs:
            output_path = filtered_dir / f"{dataset}_filtered_pairs.jsonl"
            with output_path.open("w") as f:
                for p in all_pairs[dataset]:
                    f.write(
                        json.dumps(
                            {
                                "high_freq_ngram": p.high_ngram,
                                "low_freq_ngram": p.low_ngram,
                                "high_freq_count": p.high_count,
                                "low_freq_count": p.low_count,
                                "category": p.category,
                                "shared_meaning": p.shared_meaning,
                                "frequency_ratio": round(p.ratio, 2),
                                "log_ratio": round(p.log_ratio, 3),
                            }
                        )
                        + "\n"
                    )
            print(f"created dataset for: {dataset}")


if __name__ == "__main__":
    main()
