import json
from pathlib import Path
import pandas as pd
import numpy as np
from typing import List, Dict, Any

from transformers import AutoTokenizer


def load_json(file_path: Path) -> dict:
    with open(file_path, 'r') as f:
        return json.load(f)
    
def save_json(data: dict, file_path: Path) -> None:
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

def load_dataset(file_path: Path) -> dict:
    """Load ngram dataset with metadata and ngrams structure."""
    return load_json(file_path)

def extract_ngram_frequencies(dataset: dict) -> pd.DataFrame:
    """
    Extract ngram frequencies from the new JSON structure.
    
    Args:
        dataset: Dictionary with 'metadata' and 'ngrams' keys
        
    Returns:
        DataFrame with phrase, cumulative_frequency columns
    """
    if 'ngrams' not in dataset:
        raise ValueError("Dataset must contain 'ngrams' key")
    
    data = []
    for phrase, ngram_data in dataset['ngrams'].items():
        data.append({
            'phrase': phrase,
            'text': ngram_data.get('text', phrase),
            'cumulative_frequency': ngram_data.get('cumulative_frequency', 0),
            'shard_frequencies': ngram_data.get('shard_frequencies', []),
            'checkpoint_frequencies': ngram_data.get('checkpoint_frequencies', {})
        })
    
    return pd.DataFrame(data)

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


def analyze_frequency_distribution_from_json(dataset: dict, method: str = "zipf") -> Dict[str, Any]:
    """
    Analyze frequency distribution directly from JSON ngram dataset.
    
    Args:
        dataset: Dictionary with 'metadata' and 'ngrams' keys
        method: Binning method ("zipf" or "binary")
        
    Returns:
        Dictionary with frequency analysis results
    """
    if 'ngrams' not in dataset:
        raise ValueError("Dataset must contain 'ngrams' key")
    
    # Extract frequencies from JSON structure
    phrase_freqs_data = []
    for phrase, ngram_data in dataset['ngrams'].items():
        phrase_freqs_data.append({
            'phrase': phrase,
            'cumulative_frequency': ngram_data.get('cumulative_frequency', 0)
        })
    
    phrase_freqs_df = pd.DataFrame(phrase_freqs_data)
    
    # Apply appropriate binning method
    if method == "zipf":
        binned_data = zipf_frequency_binning(
            phrase_freqs_df.set_index('phrase')['cumulative_frequency'], 
            num_bins=3
        )
        median_threshold = None
    elif method == "binary":
        binned_data = simple_frequency_binning(
            phrase_freqs_df.set_index('phrase')['cumulative_frequency']
        )
        median_threshold = binned_data['median_threshold'].iloc[0]
    else:
        raise ValueError(f"Unknown method: {method}. Use 'zipf' or 'binary'")
    
    # Merge back with original data
    binned_data = binned_data.astype({'phrase': str}).reset_index(drop=True)
    phrase_freqs_df = phrase_freqs_df.astype({'phrase': str}).reset_index(drop=True)
    result_df = phrase_freqs_df.merge(binned_data, on='phrase', how='left')
    
    # Remove duplicate columns if they exist
    if 'frequency' in result_df.columns and 'cumulative_frequency' in result_df.columns:
        # Keep cumulative_frequency, drop frequency (from binning function)
        result_df = result_df.drop(['frequency'], axis=1)
        result_df = result_df.rename(columns={'cumulative_frequency': 'frequency'})
    
    # Calculate statistics
    analysis = {
        'method': method,
        'total_unique_phrases': len(result_df),
        'frequency_distribution': result_df['category'].value_counts().to_dict(),
        'phrase_frequencies': result_df,
        'frequency_stats': {
            'min': result_df['frequency'].min(),
            'max': result_df['frequency'].max(),
            'mean': result_df['frequency'].mean(),
            'median': result_df['frequency'].median(),
            'std': result_df['frequency'].std()
        }
    }
    
    if median_threshold is not None:
        analysis['median_threshold'] = median_threshold
    
    return analysis


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
        binned_data = zipf_frequency_binning(phrase_freqs.set_index('phrase')['count_cum'], num_bins=3)
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


def get_frequency_categories_from_json(dataset: dict, method: str = "zipf") -> pd.DataFrame:
    """
    Get phrases categorized by frequency from JSON dataset.
    
    Args:
        dataset: Dictionary with 'metadata' and 'ngrams' keys
        method: Binning method ("zipf" or "binary")
        
    Returns:
        DataFrame with phrases and their frequency categories
    """
    analysis = analyze_frequency_distribution_from_json(dataset, method=method)
    return analysis['phrase_frequencies']


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
    Convert binned DataFrame to dataset format optimized for superposition research.
    Creates multiple samples per phrase using different templates.
    """
    
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    
    # Research-optimized neutral templates for countries
    templates = [
        "The capital of {country} is",
        "People from {country} speak", 
        "The economy of {country} relies on",
        "The government of {country} is located in",
    ]
    
    def find_phrase_positions(sentence_ids: List[int], phrase_ids: List[int], phrase_text: str):
        """Enhanced phrase position finding."""
        if not phrase_ids or not sentence_ids:
            return {"start": None, "end": None}
        
        # Try with space prefix first (common in middle of sentences), then plain
        candidates = [
            tokenizer.encode(" " + phrase_text, add_special_tokens=False),
            phrase_ids,
        ]
        
        for candidate in candidates:
            if not candidate:
                continue
            for i in range(len(sentence_ids) - len(candidate) + 1):
                if sentence_ids[i:i+len(candidate)] == candidate:
                    return {
                        "start": i, 
                        "end": i + len(candidate) - 1
                    }
        
        return {"start": None, "end": None}
    
    dataset = []
    for _, row in binned_df.iterrows():
        phrase = str(row['phrase']).strip()
        frequency_category = row['category']
        
        # Create one sample for each template
        for template_idx, template in enumerate(templates):
            sentence = template.format(country=phrase)
            
            sentence_ids = tokenizer.encode(sentence, add_special_tokens=False)
            phrase_ids = tokenizer.encode(phrase, add_special_tokens=False)
            positions = find_phrase_positions(sentence_ids, phrase_ids, phrase)
            
            dataset.append({
                "phrase": phrase,
                "frequency_category": frequency_category,
                "sentence": sentence,
                "token_ids_sentence": sentence_ids,
                "token_ids_phrase": phrase_ids,
                "phrase_start_idx": positions["start"],
                "phrase_end_idx": positions["end"],
                "activation_target_idx": positions["end"],  # For extraction
                "template_used": template,
                "template_idx": template_idx
            })
    
    return {
        "data": dataset,
        "templates": templates,
        "total_samples": len(dataset),
        "samples_per_phrase": len(templates),
        "unique_phrases": len(binned_df),
        "categories": binned_df['category'].value_counts().to_dict()
    }


# Example usage and testing
if __name__ == "__main__":
    # Test with new JSON structure
    json_file_path = Path("/Applications/team-aasa/polytope/ngram_dataset_countries.json")
    
    if json_file_path.exists():
        print("="*60)
        print("TESTING WITH NEW JSON STRUCTURE")
        print("="*60)
        
        # Load JSON dataset
        dataset = load_dataset(json_file_path)
        print(f"Loaded dataset with {len(dataset['ngrams'])} ngrams")
        print(f"Metadata: {dataset['metadata']}")
        
        # Test Zipf binning (3 bins: high, medium, low)
        print(f"\n--- ZIPF BINNING (3 BINS: HIGH/MEDIUM/LOW) ---")
        zipf_analysis = analyze_frequency_distribution_from_json(dataset, method="zipf")
        
        print(f"Frequency distribution:")
        for category, count in zipf_analysis['frequency_distribution'].items():
            print(f"  {category}: {count} phrases")
        
        # Show example phrases from each category
        phrase_freqs = zipf_analysis['phrase_frequencies']
        print(f"\nExample phrases by category:")
        for category in phrase_freqs['category'].unique():
            examples = phrase_freqs[phrase_freqs['category'] == category].head(3)
            freqs = examples['frequency'].tolist()
            phrases = examples['phrase'].tolist()
            for phrase, freq in zip(phrases, freqs):
                print(f"  {category}: {phrase} (freq: {freq:,})")
        
        # Test binary binning
        print(f"\n--- BINARY BINNING ---")
        binary_analysis = analyze_frequency_distribution_from_json(dataset, method="binary")
        
        print(f"Frequency distribution:")
        for category, count in binary_analysis['frequency_distribution'].items():
            print(f"  {category}: {count} phrases")
        
        print(f"Median threshold: {binary_analysis['median_threshold']:,.0f}")
        
        # Create dataset with multiple templates
        print(f"\n--- CREATING POLYTOPE DATASET ---")
        categorized_df = get_frequency_categories_from_json(dataset, method="zipf")
        polytope_dataset = df_to_dataset(categorized_df)
        
        print(f"Generated dataset:")
        print(f"  Total samples: {polytope_dataset['total_samples']}")
        print(f"  Samples per phrase: {polytope_dataset['samples_per_phrase']}")
        print(f"  Unique phrases: {polytope_dataset['unique_phrases']}")
        print(f"  Templates: {len(polytope_dataset['templates'])}")
        print(f"  Categories: {polytope_dataset['categories']}")
        
        # Show a few examples
        print(f"\nExample generated samples:")
        for i in range(min(6, len(polytope_dataset['data']))):
            sample = polytope_dataset['data'][i]
            print(f"  {sample['phrase']} ({sample['frequency_category']}):")
            print(f"    Template {sample['template_idx']}: {sample['sentence']}")
            print(f"    Phrase position: {sample['phrase_start_idx']}-{sample['phrase_end_idx']}")
        
        # Save the polytope dataset
        save_json(polytope_dataset, Path("/Applications/team-aasa/polytope/country_capital_polytope_dataset.json"))
        print(f"\nSaved polytope dataset to: country_capital_polytope_dataset.json")
        
    else:
        print(f"JSON file not found: {json_file_path}")
        print("Falling back to parquet testing...")
        
        # Fallback to old parquet testing
        file_path = Path("/Applications/team-aasa/polytope/cumulative.parquet")
        if file_path.exists():
            df = parquet_to_df(file_path)
            print("Dataset shape:", df.shape)
            print("Columns:", df.columns.tolist())
            
            # Test with old structure
            categorized_df = get_frequency_categories(df, method="zipf")
            dataset = df_to_dataset(categorized_df)
            save_json(dataset, Path("/Applications/team-aasa/polytope/dataset.json"))
        else:
            print("No data files found for testing.")
    
