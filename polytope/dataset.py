import json
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any, Optional
from collections import Counter


def load_json(file_path: Path) -> dict:
    with open(file_path, 'r') as f:
        return json.load(f)
    
def save_json(data: dict, file_path: Path) -> None:
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

def load_dataset(file_path: Path) -> dict:
    return load_json(file_path)

def parquet_to_df(file_path: Path) -> pd.DataFrame:
    return pd.read_parquet(file_path)


def simple_frequency_binning(frequencies: pd.Series) -> pd.DataFrame:
    """
    Simple binary frequency split using median threshold.
    
    This is the most research-appropriate method for polytope analysis:
    - Maximizes statistical power for detecting superposition effects
    - Aligns with established neuroscience practice
    - Minimizes arbitrary boundaries that could create artifacts
    - Provides clear interpretability for academic publications
    
    Args:
        frequencies: Series of frequency values
        
    Returns:
        DataFrame with phrases, frequencies, and binary categories
    """
    median_freq = frequencies.median()
    
    categories = pd.Series(
        ['high_frequency' if freq >= median_freq else 'low_frequency' 
         for freq in frequencies],
        index=frequencies.index
    )
    
    return pd.DataFrame({
        'phrase': frequencies.index,
        'frequency': frequencies.values,
        'category': categories,
        'median_threshold': median_freq
    })


def zipf_frequency_binning(frequencies: pd.Series, num_bins: int = 3) -> pd.DataFrame:
    """
    True Zipf-based frequency binning using rank-frequency relationship.

    - Sort by frequency descending to obtain Zipf ranks
    - Break ties deterministically by phrase (index) ascending
    - Split ranks into equal-sized bins by rank position (not percentiles)

    Args:
        frequencies: Series of frequency values (index are phrases)
        num_bins: Number of bins (default 3)

    Returns:
        DataFrame with phrases, frequencies, categories, and zipf ranks
    """
    if frequencies is None or len(frequencies) == 0:
        return pd.DataFrame(columns=['phrase', 'frequency', 'category', 'zipf_rank'])

    # Build a deterministic ordering: freq desc, then index asc
    order_df = pd.DataFrame({
        'phrase': frequencies.index.astype(str),
        'frequency': frequencies.values
    })
    order_df = order_df.sort_values(by=['frequency', 'phrase'], ascending=[False, True], kind='mergesort')

    # Map phrase -> 1-based rank position
    phrase_to_rank: Dict[str, int] = {phrase: rank for rank, phrase in enumerate(order_df['phrase'].tolist(), start=1)}

    total_items = len(order_df)
    # Rank boundaries (positions), e.g., for 5 bins: [ceil(N/5), ceil(2N/5), ..., ceil(4N/5)]
    rank_boundaries: List[int] = [int(np.ceil((i / num_bins) * total_items)) for i in range(1, num_bins)]

    # Labels
    if num_bins == 3:
        bin_labels: List[str] = ['high_frequency', 'medium_frequency', 'low_frequency']
    elif num_bins == 4:
        bin_labels = ['very_high', 'high', 'medium', 'low']
    elif num_bins == 5:
        bin_labels = ['ultra_high', 'very_high', 'high', 'medium', 'low']
    else:
        bin_labels = [f'zipf_rank_{i}' for i in range(num_bins)]

    def rank_to_bin_label(rank: int) -> str:
        # Determine bin index by rank position relative to boundaries
        bin_index = 0
        for j, boundary in enumerate(rank_boundaries):
            if rank <= boundary:
                bin_index = j
                break
        else:
            bin_index = len(rank_boundaries)
        return bin_labels[bin_index]

    # Build result rows
    categories: List[str] = []
    ranks: List[int] = []
    for phrase in frequencies.index.astype(str):
        rank = phrase_to_rank[phrase]
        ranks.append(rank)
        categories.append(rank_to_bin_label(rank))

    return pd.DataFrame({
        'phrase': frequencies.index,
        'frequency': frequencies.values,
        'category': categories,
        'zipf_rank': ranks
    })


def analyze_frequency_distribution(df: pd.DataFrame, 
                                 step: int = None,
                                 method: str = "zipf") -> Dict[str, Any]:
    """
    Analyze frequency distribution of phrases using research-appropriate methods.
    
    Args:
        df: DataFrame with phrase, step, count_cum, per_million_cum columns
        step: Specific step to analyze (if None, uses latest step for each phrase)
        method: Binning method ("zipf" or "binary")
        
    Returns:
        Dictionary with frequency analysis results
    """
    if step is not None:
        # Filter to specific step
        step_df = df[df['step'] == step].copy()
    else:
        # Use latest step for each phrase
        step_df = df.loc[df.groupby('phrase')['step'].idxmax()].copy()
    
    # Get unique phrases and their frequencies
    phrase_freqs = step_df.groupby('phrase').agg({
        'count_cum': 'max',
        'per_million_cum': 'max'
    }).reset_index()
    
    # Apply appropriate binning method
    if method == "zipf":
        binned_data = zipf_frequency_binning(phrase_freqs.set_index('phrase')['count_cum'], num_bins=5)
        median_threshold = None
    elif method == "binary":
        binned_data = simple_frequency_binning(phrase_freqs.set_index('phrase')['count_cum'])
        median_threshold = binned_data['median_threshold'].iloc[0]
    else:
        raise ValueError(f"Unknown method: {method}. Use 'zipf' or 'binary'")
    
    # Merge back with original data and clean up columns
    # Ensure both sides merge on a column with homogeneous dtype
    binned_data = binned_data.astype({'phrase': str}).reset_index(drop=True)
    phrase_freqs = phrase_freqs.astype({'phrase': str}).reset_index(drop=True)
    result_df = phrase_freqs.merge(binned_data, on='phrase', how='left')
    
    # Remove duplicate columns and clean up
    if 'phrase_x' in result_df.columns and 'phrase_y' in result_df.columns:
        result_df = result_df.drop(['phrase_x', 'phrase_y'], axis=1)
    elif 'phrase_x' in result_df.columns:
        result_df = result_df.drop(['phrase_x'], axis=1)
    elif 'phrase_y' in result_df.columns:
        result_df = result_df.drop(['phrase_y'], axis=1)
    
    # Calculate statistics
    analysis = {
        'method': method,
        'step_analyzed': step,
        'total_unique_phrases': len(result_df),
        'frequency_distribution': result_df['category'].value_counts().to_dict(),
        'phrase_frequencies': result_df,
        'frequency_stats': {
            'min': result_df['count_cum'].min(),
            'max': result_df['count_cum'].max(),
            'mean': result_df['count_cum'].mean(),
            'median': result_df['count_cum'].median(),
            'std': result_df['count_cum'].std()
        }
    }
    
    if median_threshold is not None:
        analysis['median_threshold'] = median_threshold
    
    # Add per-million frequency analysis
    if 'per_million_cum' in result_df.columns:
        analysis['per_million_stats'] = {
            'min': result_df['per_million_cum'].min(),
            'max': result_df['per_million_cum'].max(),
            'mean': result_df['per_million_cum'].mean(),
            'median': result_df['per_million_cum'].median()
        }
    
    return analysis


def get_frequency_categories(df: pd.DataFrame, 
                           method: str = "zipf",
                           step: int = None) -> pd.DataFrame:
    """
    Get phrases categorized by frequency with their categories.
    
    Args:
        df: DataFrame with phrase, step, count_cum, per_million_cum columns
        method: Binning method ("zipf" or "binary")
        step: Specific step to analyze (if None, uses latest step for each phrase)
        
    Returns:
        DataFrame with phrases and their frequency categories
    """
    analysis = analyze_frequency_distribution(df, step=step, method=method)
    return analysis['phrase_frequencies']


def df_to_dataset(binned_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Convert binned DataFrame to dataset format. Need to embed phrases into semantically neutral template sentences.
    
    Args:
        binned_df: DataFrame with phrases and their frequency categories
        
    Returns:
        Dictionary with dataset format
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

    template_sentence: str = "The {phrase} is a geographical place"

    def find_subsequence_end(haystack: List[int], needle: List[int]) -> Optional[int]:
        """Return the end index of the last occurrence of `needle` in `haystack`, or None.

        This searches for the contiguous subsequence `needle` inside `haystack` and returns
        the index of the final token of the match. If multiple matches exist, the last one
        is returned.
        """
        if not needle or not haystack or len(needle) > len(haystack):
            return None
        last_end: Optional[int] = None
        first_token = needle[0]
        max_start = len(haystack) - len(needle)
        for start in range(0, max_start + 1):
            if haystack[start] != first_token:
                continue
            if haystack[start:start + len(needle)] == needle:
                last_end = start + len(needle) - 1
        return last_end

    dataset: List[Dict[str, Any]] = []
    for _, row in binned_df.iterrows():
        phrase = str(row['phrase'])
        frequency_category = row['category']
        sentence = template_sentence.format(phrase=phrase)

        # Token ids (not token strings) for reliable matching
        sentence_ids: List[int] = tokenizer.encode(sentence, add_special_tokens=False)
        phrase_ids_plain: List[int] = tokenizer.encode(phrase, add_special_tokens=False)
        phrase_ids_space: List[int] = tokenizer.encode(" " + phrase, add_special_tokens=False)

        # Prefer match with leading space (more likely inside sentence), then fallback
        end_pos: Optional[int] = None
        for candidate in (phrase_ids_space, phrase_ids_plain):
            if candidate:
                end_pos = find_subsequence_end(sentence_ids, candidate)
            if end_pos is not None:
                break

        # Token strings for readability/debugging
        tokens_sentence: List[str] = tokenizer.convert_ids_to_tokens(sentence_ids)
        tokens_phrase: List[str] = tokenizer.convert_ids_to_tokens(
            phrase_ids_space if (end_pos is not None and phrase_ids_space) else phrase_ids_plain
        )

        dataset.append({
            "phrase": phrase,
            "frequency_category": frequency_category,
            "sentence": sentence,
            "token_ids_sentence": sentence_ids,
            "token_ids_phrase": phrase_ids_plain,
            "tokens_sentence": tokens_sentence,
            "tokens_phrase": tokens_phrase,
            "phrase_last_token_index": end_pos,
        })

    return dataset

# Example usage and testing
if __name__ == "__main__":
    file_path = Path("/Applications/team-aasa/polytope/cumulative.parquet")
    df = parquet_to_df(file_path)
    
    print("Dataset shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("\nSample data:")
    print(df.head())
    
    # Analyze frequency distribution
    print("\n" + "="*50)
    print("FREQUENCY ANALYSIS")
    print("="*50)
    
    # Get latest step for each phrase
    latest_df = df.loc[df.groupby('phrase')['step'].idxmax()]
    print(f"Total unique phrases: {len(latest_df)}")
    
    # Test Zipf binning (primary recommendation)
    print(f"\n--- ZIPF BINNING (RECOMMENDED) ---")
    zipf_analysis = analyze_frequency_distribution(df, method="zipf")
    
    print(f"Frequency distribution:")
    for category, count in zipf_analysis['frequency_distribution'].items():
        print(f"  {category}: {count} phrases")
    
    # Show some example phrases from each category
    phrase_freqs = zipf_analysis['phrase_frequencies']
    print(f"\nExample phrases by category:")
    for category in phrase_freqs['category'].unique():
        examples = phrase_freqs[phrase_freqs['category'] == category].head(3)
        print(f"  {category}: {examples['phrase'].tolist()}")
    
    # Test binary binning (secondary option)
    print(f"\n--- BINARY BINNING (SECONDARY) ---")
    binary_analysis = analyze_frequency_distribution(df, method="binary")
    
    print(f"Frequency distribution:")
    for category, count in binary_analysis['frequency_distribution'].items():
        print(f"  {category}: {count} phrases")
    
    print(f"Median threshold: {binary_analysis['median_threshold']:,.0f}")
    
    # Save categorized data
    categorized_df = get_frequency_categories(df, method="zipf")
    print(categorized_df.head())

    dataset = df_to_dataset(categorized_df)
    print(dataset)
    
