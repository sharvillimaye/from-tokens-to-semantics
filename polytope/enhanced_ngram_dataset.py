#!/usr/bin/env python3
"""
Robust and Flexible Enhanced N-gram Dataset Builder for Research
Supports multiple datasets, configurable semantic grouping, and flexible data processing
"""

import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any, Optional, Union
import re
import json
import time
import random
from tqdm import tqdm
from datasets import load_dataset
from pathlib import Path
import logging
import warnings
from dataclasses import dataclass, asdict
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DatasetSource(Enum):
    """Available dataset sources"""
    DOLMA = "dolma"
    PILE = "pile"
    OPENWEBTEXT = "openwebtext"
    DUMMY = "dummy"


@dataclass
class DatasetConfig:
    """Configuration for dataset building"""
    n_gram_size: int = 2
    num_samples: int = 1000
    source: DatasetSource = DatasetSource.DOLMA
    include_semantics: bool = True
    min_text_length: int = 50
    max_text_length: int = 2000
    min_word_length: int = 2
    filter_digits: bool = True
    max_examples_per_ngram: int = 3
    max_total_ngrams: int = 1000
    cache_dir: str = "./cache"
    random_seed: Optional[int] = None


@dataclass
class FrequencyBins:
    """Configurable frequency bins"""
    high: Tuple[int, float] = (1000, float('inf'))
    medium: Tuple[int, int] = (100, 1000)
    low: Tuple[int, int] = (10, 100)
    very_low: Tuple[int, int] = (1, 10)
    
    def to_dict(self) -> Dict[str, Tuple[int, float]]:
        return asdict(self)


# Default semantic categories - can be extended/modified
DEFAULT_SEMANTIC_CATEGORIES = {
    'geography': ['country', 'city', 'state', 'nation', 'continent', 'region', 'capital', 'territory'],
    'technology': ['computer', 'software', 'algorithm', 'database', 'network', 'system', 'technology'],
    'science': ['research', 'experiment', 'theory', 'hypothesis', 'analysis', 'study', 'discovery'],
    'business': ['company', 'business', 'market', 'industry', 'profit', 'revenue', 'economy'],
    'politics': ['government', 'policy', 'election', 'democracy', 'legislation', 'political'],
    'health': ['medical', 'health', 'disease', 'treatment', 'patient', 'doctor', 'hospital'],
    'education': ['education', 'learning', 'teaching', 'student', 'school', 'university', 'academic'],
    'sports': ['sport', 'game', 'team', 'player', 'coach', 'championship', 'athletic'],
    'entertainment': ['movie', 'film', 'music', 'artist', 'entertainment', 'performance', 'media'],
    'general': ['general', 'common', 'basic', 'fundamental', 'essential', 'important']
}


def load_dataset_robust(config: DatasetConfig) -> List[str]:
    """Robustly load dataset from various sources"""
    logger.info(f"Loading {config.num_samples} samples from {config.source.value}")
    
    if config.random_seed is not None:
        random.seed(config.random_seed)
        np.random.seed(config.random_seed)
    
    try:
        if config.source == DatasetSource.DOLMA:
            return _load_dolma(config)
        elif config.source == DatasetSource.PILE:
            return _load_pile(config)
        elif config.source == DatasetSource.OPENWEBTEXT:
            return _load_openwebtext(config)
        elif config.source == DatasetSource.DUMMY:
            return generate_dummy_texts(config.num_samples)
        else:
            logger.warning(f"Unknown source {config.source}, using dummy data")
            return generate_dummy_texts(config.num_samples)
            
    except Exception as e:
        logger.error(f"Failed to load {config.source.value}: {e}")
        logger.info("Falling back to dummy data")
        return generate_dummy_texts(config.num_samples)


def _load_dolma(config: DatasetConfig) -> List[str]:
    """Load from Dolma dataset with multiple fallback options"""
    dataset_configs = [
        "allenai/dolma",
        "allenai/dolma-v1_5", 
        "allenai/dolma-v1_6"
    ]
    
    dataset = None
    for dataset_name in dataset_configs:
        try:
            logger.info(f"Trying dataset: {dataset_name}")
            dataset = load_dataset(
                dataset_name, 
                split="train", 
                streaming=True, 
                trust_remote_code=True
            )
            logger.info(f"✓ Successfully loaded: {dataset_name}")
            break
        except Exception as e:
            logger.warning(f"✗ Failed {dataset_name}: {e}")
            continue
    
    if dataset is None:
        raise RuntimeError("Could not load any Dolma configuration")

    return _extract_texts_from_dataset(dataset, config)


def _load_pile(config: DatasetConfig) -> List[str]:
    """Load from The Pile dataset"""
    try:
        dataset = load_dataset(
            "EleutherAI/pile",
            split="train",
            streaming=True
        )
        logger.info("✓ Successfully loaded The Pile")
        return _extract_texts_from_dataset(dataset, config)
    except Exception as e:
        logger.error(f"Failed to load The Pile: {e}")
        raise


def _load_openwebtext(config: DatasetConfig) -> List[str]:
    """Load from OpenWebText dataset"""
    try:
        dataset = load_dataset(
            "openwebtext",
            split="train",
            streaming=True
        )
        logger.info("✓ Successfully loaded OpenWebText")
        return _extract_texts_from_dataset(dataset, config)
    except Exception as e:
        logger.error(f"Failed to load OpenWebText: {e}")
        raise


def _extract_texts_from_dataset(dataset, config: DatasetConfig) -> List[str]:
    """Extract and filter texts from any HuggingFace dataset"""
    texts = []
    valid_count = 0
    
    with tqdm(total=config.num_samples, desc="Loading texts") as pbar:
        for i, sample in enumerate(dataset):
            if valid_count >= config.num_samples:
                break

            # Handle different possible field names
            text = None
            for field in ['text', 'content', 'passage', 'document']:
                if field in sample:
                    text = sample[field]
                    break
            
            if text is None:
                continue
                
            if isinstance(text, list):
                text = ' '.join(text)
            
            # Apply length filters
            text_len = len(text.strip())
            if config.min_text_length <= text_len <= config.max_text_length:
                texts.append(text.strip())
                valid_count += 1
                pbar.update(1)

            # Log progress
            if i % 500 == 0 and i > 0:
                logger.info(f"Processed {i} samples, collected {valid_count} valid texts")

    logger.info(f"Final: {len(texts)} valid texts from dataset")
    return texts


def generate_dummy_texts(num_samples: int = 100) -> List[str]:
    """Generate dummy texts for testing"""
    templates = [
        "The {adj} {noun} {verb} quickly in the {place}.",
        "Machine learning {verb} {adj} patterns in {noun} data.",
        "The {adj} algorithm {verb} {noun} with {adj2} accuracy.",
        "Deep learning {verb} {adj} features from {noun} datasets.",
        "Scientists {verb} {adj} {noun} in their research."
    ]

    words = {
        'adj': ["advanced", "complex", "efficient", "robust", "scalable", "optimal"],
        'adj2': ["better", "faster", "stronger", "smarter", "cleaner", "simpler"],
        'noun': ["model", "system", "network", "algorithm", "data", "method"],
        'verb': ["processes", "analyzes", "computes", "learns", "predicts", "optimizes"],
        'place': ["cloud", "server", "database", "memory", "cache", "pipeline"]
    }

    texts = []
    for _ in range(num_samples):
        template = random.choice(templates)
        text = template.format(**{k: random.choice(v) for k, v in words.items()})
        texts.append(text)

    return texts


def classify_semantic_category(ngram: str, 
                              text_context: str = "", 
                              semantic_categories: Optional[Dict[str, List[str]]] = None,
                              confidence_threshold: float = 0.1) -> Tuple[str, float]:
    """
    Classify n-gram into semantic categories with confidence scoring
    
    Args:
        ngram: The n-gram to classify
        text_context: Surrounding context for better classification
        semantic_categories: Custom semantic categories (uses default if None)
        confidence_threshold: Minimum confidence for non-general classification
        
    Returns:
        Tuple of (category, confidence_score)
    """
    if semantic_categories is None:
        semantic_categories = DEFAULT_SEMANTIC_CATEGORIES
    
    analysis_text = f"{ngram} {text_context}".lower()
    category_scores = defaultdict(float)
    
    # Score each category based on keyword matches
    total_possible_score = 0
    for category, keywords in semantic_categories.items():
        score = 0.0
        for keyword in keywords:
            if keyword.lower() in analysis_text:
                # Weight exact ngram matches higher than context matches
                if keyword.lower() in ngram.lower():
                    score += 2.0
                else:
                    score += 1.0
        
        category_scores[category] = score
        total_possible_score += len(keywords) * 2  # Max possible score
    
    # Calculate confidence and select best category
    if category_scores and total_possible_score > 0:
        best_category = max(category_scores, key=category_scores.get)
        max_score = category_scores[best_category]
        
        if max_score > 0:
            confidence = min(max_score / (len(semantic_categories[best_category]) * 2), 1.0)
            
            if confidence >= confidence_threshold:
                return best_category, confidence
    
    return 'general', 0.0


def extract_ngrams_with_semantics(text: str, 
                                  config: DatasetConfig,
                                  semantic_categories: Dict[str, List[str]]) -> List[Dict[str, Any]]:
    """Extract n-grams with semantic classification using configuration"""
    # Clean text
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    text = re.sub(r'\s+', ' ', text.strip())
    words = text.split()

    if len(words) < config.n_gram_size:
        return []

    ngrams_with_semantics = []
    for i in range(len(words) - config.n_gram_size + 1):
        ngram = ' '.join(words[i:i + config.n_gram_size])
        
        # Apply word-level filtering
        ngram_words = ngram.split()
        if not _passes_quality_filter(ngram_words, config):
            continue
        
        # Get context (words around the n-gram)
        context_start = max(0, i - 3)
        context_end = min(len(words), i + config.n_gram_size + 3)
        context = ' '.join(words[context_start:context_end])
        
        # Classify semantic category
        semantic_category, confidence = classify_semantic_category(
            ngram, context, semantic_categories
        )
        
        ngrams_with_semantics.append({
            'ngram': ngram,
            'position': i,
            'context': context,
            'semantic_category': semantic_category,
            'semantic_confidence': confidence
        })

    return ngrams_with_semantics


def count_ngrams_with_semantics(texts: List[str], 
                               config: DatasetConfig,
                               semantic_categories: Dict[str, List[str]]) -> Tuple[Counter, Dict[str, List[Dict[str, Any]]]]:
    """Count n-grams with frequency and semantic information"""
    ngram_counts = Counter()
    ngram_details = defaultdict(list)

    logger.info(f"Extracting {config.n_gram_size}-grams with semantic analysis from {len(texts)} texts...")

    for text in tqdm(texts, desc="Processing texts"):
        ngrams_with_semantics = extract_ngrams_with_semantics(text, config, semantic_categories)

        for ngram_info in ngrams_with_semantics:
            ngram = ngram_info['ngram']
            ngram_counts[ngram] += 1
            
            # Store semantic information (limit examples)
            if len(ngram_details[ngram]) < config.max_examples_per_ngram:
                ngram_details[ngram].append({
                    'text': text[:200],  # Truncate text
                    'semantic_category': ngram_info['semantic_category'],
                    'semantic_confidence': ngram_info['semantic_confidence'],
                    'context': ngram_info['context']
                })

    logger.info(f"Found {len(ngram_counts)} unique n-grams with semantic classification")
    return ngram_counts, dict(ngram_details)


def _passes_quality_filter(words: List[str], config: DatasetConfig) -> bool:
    """Check if n-gram passes quality filters"""
    if len(words) != config.n_gram_size:
        return False
    
    for word in words:
        # Check minimum word length
        if len(word) < config.min_word_length:
            return False
            
        # Check for digits if filtering enabled
        if config.filter_digits and word.isdigit():
            return False
    
    return True


def categorize_by_frequency(frequency: int, frequency_bins: FrequencyBins) -> str:
    """Categorize n-gram by frequency bin"""
    bins_dict = frequency_bins.to_dict()
    for category, (min_freq, max_freq) in bins_dict.items():
        if min_freq <= frequency < max_freq:
            return category
    return 'very_low'


def estimate_global_frequency(local_freq: int, estimation_factor: Tuple[float, float] = (50, 200)) -> int:
    """Estimate global frequency from local frequency with configurable factor"""
    min_factor, max_factor = estimation_factor
    return max(1, int(local_freq * random.uniform(min_factor, max_factor)))


def build_enhanced_dataset(config: Optional[DatasetConfig] = None,
                          frequency_bins: Optional[FrequencyBins] = None,
                          semantic_categories: Optional[Dict[str, List[str]]] = None) -> Dict[str, Any]:
    """
    Build enhanced dataset with flexible configuration
    
    Args:
        config: Dataset configuration (uses defaults if None)
        frequency_bins: Custom frequency bins (uses defaults if None)
        semantic_categories: Custom semantic categories (uses defaults if None)
        
    Returns:
        Dictionary containing the processed dataset
    """
    if config is None:
        config = DatasetConfig()
    
    if frequency_bins is None:
        frequency_bins = FrequencyBins()
        
    if semantic_categories is None:
        semantic_categories = DEFAULT_SEMANTIC_CATEGORIES
    
    logger.info(f"🚀 Building enhanced {config.n_gram_size}-gram dataset")
    logger.info(f"Configuration: {config}")
    start_time = time.time()

    # Setup cache directory
    cache_dir = Path(config.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Step 1: Load texts from source
        texts = load_dataset_robust(config)
        
        if not texts:
            raise ValueError("No texts loaded from dataset")

        # Step 2: Extract and count n-grams
        if config.include_semantics:
            ngram_counts, ngram_details = count_ngrams_with_semantics(
                texts, config, semantic_categories
            )
        else:
            ngram_counts, ngram_details = count_ngrams_simple(texts, config)

        # Step 3: Create final dataset structure
        final_dataset = create_final_dataset(
            ngram_counts, ngram_details, config, frequency_bins, semantic_categories
        )

        elapsed_time = time.time() - start_time
        logger.info(f"✅ Dataset built successfully in {elapsed_time:.1f} seconds!")

        return final_dataset
        
    except Exception as e:
        logger.error(f"Failed to build dataset: {e}")
        raise


def count_ngrams_simple(texts: List[str], config: DatasetConfig) -> Tuple[Counter, Dict[str, List[Dict[str, Any]]]]:
    """Simple n-gram counting without semantic analysis"""
    ngram_counts = Counter()
    ngram_details = defaultdict(list)

    logger.info(f"Extracting {config.n_gram_size}-grams from {len(texts)} texts...")

    for text in tqdm(texts, desc="Processing texts"):
        # Clean text
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        text = re.sub(r'\s+', ' ', text.strip())
        words = text.split()

        if len(words) < config.n_gram_size:
            continue

        for i in range(len(words) - config.n_gram_size + 1):
            ngram = ' '.join(words[i:i + config.n_gram_size])
            
            # Apply quality filtering
            words_ngram = ngram.split()
            if _passes_quality_filter(words_ngram, config):
                ngram_counts[ngram] += 1
                if len(ngram_details[ngram]) < config.max_examples_per_ngram:
                    ngram_details[ngram].append({
                        'text': text[:200],
                        'semantic_category': 'general',
                        'semantic_confidence': 0.0
                    })

    logger.info(f"Found {len(ngram_counts)} unique n-grams")
    return ngram_counts, dict(ngram_details)


def create_final_dataset(ngram_counts: Counter, 
                        ngram_details: Dict[str, List[Dict[str, Any]]], 
                        config: DatasetConfig,
                        frequency_bins: FrequencyBins,
                        semantic_categories: Dict[str, List[str]]) -> Dict[str, Any]:
    """Create final dataset structure with flexible configuration"""
    
    final_dataset = {
        'texts': [], 'ngrams': [], 'frequency_categories': [],
        'local_frequencies': [], 'global_frequencies': []
    }
    
    # Add semantic fields if enabled
    if config.include_semantics:
        final_dataset['semantic_categories'] = []
        final_dataset['semantic_confidence'] = []

    # Process each n-gram (limit to max_total_ngrams)
    for ngram, local_freq in ngram_counts.most_common(config.max_total_ngrams):
        global_freq = estimate_global_frequency(local_freq)
        freq_category = categorize_by_frequency(global_freq, frequency_bins)
        
        # Get text examples and semantic info
        details = ngram_details.get(ngram, [{'text': '', 'semantic_category': 'general', 'semantic_confidence': 0.0}])
        
        for detail in details[:1]:  # Just 1 example per n-gram for final dataset
            final_dataset['texts'].append(detail['text'])
            final_dataset['ngrams'].append(ngram)
            final_dataset['frequency_categories'].append(freq_category)
            final_dataset['local_frequencies'].append(local_freq)
            final_dataset['global_frequencies'].append(global_freq)
            
            if config.include_semantics:
                final_dataset['semantic_categories'].append(detail['semantic_category'])
                final_dataset['semantic_confidence'].append(detail.get('semantic_confidence', 0.0))

    # Add comprehensive metadata
    final_dataset['metadata'] = {
        'config': asdict(config),
        'frequency_bins': frequency_bins.to_dict(),
        'semantic_categories': semantic_categories,
        'total_samples': len(final_dataset['texts']),
        'creation_timestamp': time.time(),
        'unique_ngrams': len(set(final_dataset['ngrams'])),
        'source_dataset': config.source.value
    }

    logger.info(f"Created final dataset with {len(final_dataset['texts'])} samples")
    return final_dataset


def save_dataset(dataset: Dict[str, Any], output_path: str):
    """Save dataset to JSON"""
    output_path = Path(output_path).with_suffix('.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=2, default=str)
    print(f"💾 Saved dataset to {output_path}")


def analyze_semantic_distribution(dataset: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze semantic distribution of the dataset"""
    if 'semantic_categories' not in dataset:
        return {'error': 'No semantic categories found in dataset'}
    
    semantic_counts = Counter(dataset['semantic_categories'])
    frequency_counts = Counter(dataset['frequency_categories'])
    
    analysis = {
        'total_samples': len(dataset['semantic_categories']),
        'semantic_distribution': dict(semantic_counts),
        'frequency_distribution': dict(frequency_counts),
    }
    
    return analysis


def filter_by_semantic_category(dataset: Dict[str, Any], category: str) -> Dict[str, Any]:
    """Filter dataset by semantic category"""
    if 'semantic_categories' not in dataset:
        return dataset
    
    indices = [i for i, cat in enumerate(dataset['semantic_categories']) if cat == category]
    
    filtered_dataset = {}
    for key, values in dataset.items():
        if key == 'metadata':
            filtered_dataset[key] = values
        elif isinstance(values, list) and len(values) == len(dataset['texts']):
            filtered_dataset[key] = [values[i] for i in indices]
        else:
            filtered_dataset[key] = values
    
    return filtered_dataset


def filter_by_frequency_category(dataset: Dict[str, Any], frequency_category: str) -> Dict[str, Any]:
    """Filter dataset by frequency category"""
    indices = [i for i, cat in enumerate(dataset['frequency_categories']) if cat == frequency_category]
    
    filtered_dataset = {}
    for key, values in dataset.items():
        if key == 'metadata':
            filtered_dataset[key] = values
        elif isinstance(values, list) and len(values) == len(dataset['texts']):
            filtered_dataset[key] = [values[i] for i in indices]
        else:
            filtered_dataset[key] = values
    
    return filtered_dataset


def get_semantic_frequency_breakdown(dataset: Dict[str, Any]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Get low and high frequency n-grams for each semantic category
    
    Returns:
        Dict with structure: {semantic_category: {'low': {...}, 'high': {...}}}
    """
    if 'semantic_categories' not in dataset:
        logger.warning("No semantic categories found in dataset")
        return {}
    
    breakdown = {}
    
    # Get unique semantic categories
    unique_categories = set(dataset['semantic_categories'])
    
    for category in unique_categories:
        breakdown[category] = {'low': {}, 'high': {}}
        
        # Filter by semantic category first
        semantic_dataset = filter_by_semantic_category(dataset, category)
        
        if len(semantic_dataset['texts']) == 0:
            continue
            
        # Then filter by frequency within that semantic category
        low_freq_data = filter_by_frequency_category(semantic_dataset, 'low')
        high_freq_data = filter_by_frequency_category(semantic_dataset, 'high')
        
        # Add medium frequency as "high" if no high frequency exists
        if len(high_freq_data['texts']) == 0:
            high_freq_data = filter_by_frequency_category(semantic_dataset, 'medium')
            
        # Add very_low as "low" if no low frequency exists  
        if len(low_freq_data['texts']) == 0:
            low_freq_data = filter_by_frequency_category(semantic_dataset, 'very_low')
        
        # Store results with statistics
        breakdown[category]['low'] = {
            'dataset': low_freq_data,
            'count': len(low_freq_data['texts']),
            'unique_ngrams': len(set(low_freq_data['ngrams'])) if low_freq_data['ngrams'] else 0,
            'avg_frequency': np.mean(low_freq_data['local_frequencies']) if low_freq_data['local_frequencies'] else 0
        }
        
        breakdown[category]['high'] = {
            'dataset': high_freq_data,
            'count': len(high_freq_data['texts']),
            'unique_ngrams': len(set(high_freq_data['ngrams'])) if high_freq_data['ngrams'] else 0,
            'avg_frequency': np.mean(high_freq_data['local_frequencies']) if high_freq_data['local_frequencies'] else 0
        }
    
    return breakdown


def save_semantic_frequency_datasets(breakdown: Dict[str, Dict[str, Dict[str, Any]]], 
                                   output_dir: str = "./datasets/semantic_frequency"):
    """Save low and high frequency datasets for each semantic category"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for category, freq_data in breakdown.items():
        for freq_type in ['low', 'high']:
            if breakdown[category][freq_type]['count'] > 0:
                dataset = breakdown[category][freq_type]['dataset']
                filename = f"{category}_{freq_type}_freq_dataset"
                save_dataset(dataset, output_path / filename)
                
    logger.info(f"Saved semantic frequency datasets to {output_path}")


def print_semantic_frequency_summary(breakdown: Dict[str, Dict[str, Dict[str, Any]]]):
    """Print a summary of the semantic frequency breakdown"""
    print("\n🎯 Semantic Frequency Breakdown:")
    print("=" * 60)
    
    for category, freq_data in breakdown.items():
        low_count = freq_data['low']['count']
        high_count = freq_data['high']['count']
        low_avg = freq_data['low']['avg_frequency']
        high_avg = freq_data['high']['avg_frequency']
        
        print(f"\n📂 {category.upper()}:")
        print(f"  Low frequency:  {low_count:3d} samples (avg freq: {low_avg:.1f})")
        print(f"  High frequency: {high_count:3d} samples (avg freq: {high_avg:.1f})")
        
        # Show some examples
        if low_count > 0:
            low_ngrams = freq_data['low']['dataset']['ngrams'][:3]
            low_examples = ', '.join([f"'{ng}'" for ng in low_ngrams])
            print(f"  Low examples:   {low_examples}")
            
        if high_count > 0:
            high_ngrams = freq_data['high']['dataset']['ngrams'][:3]
            high_examples = ', '.join([f"'{ng}'" for ng in high_ngrams])
            print(f"  High examples:  {high_examples}")


# Convenience functions for common use cases
def quick_build_dataset(n_gram_size: int = 2, 
                       samples: int = 500, 
                       source: str = "dolma",
                       include_semantics: bool = True,
                       random_seed: Optional[int] = None) -> Dict[str, Any]:
    """Quick way to build a dataset for research"""
    config = DatasetConfig(
        n_gram_size=n_gram_size,
        num_samples=samples,
        source=DatasetSource(source),
        include_semantics=include_semantics,
        random_seed=random_seed
    )
    return build_enhanced_dataset(config)


def quick_semantic_frequency_analysis(n_gram_size: int = 2, 
                                     samples: int = 500,
                                     source: str = "dummy",
                                     save_datasets: bool = True) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Quick function to build dataset and get semantic frequency breakdown
    
    Returns:
        Dictionary with low/high frequency data for each semantic category
    """
    print("🔬 Quick Semantic Frequency Analysis")
    print("=" * 50)
    
    # Build dataset
    print("1️⃣ Building dataset...")
    dataset = quick_build_dataset(n_gram_size=n_gram_size, samples=samples, source=source)
    
    # Get breakdown
    print("2️⃣ Analyzing semantic frequencies...")
    breakdown = get_semantic_frequency_breakdown(dataset)
    
    # Print summary
    print_semantic_frequency_summary(breakdown)
    
    # Save datasets if requested
    if save_datasets:
        print("\n3️⃣ Saving datasets...")
        save_semantic_frequency_datasets(breakdown)
    
    return breakdown


def build_custom_semantic_dataset(semantic_categories: Dict[str, List[str]],
                                 n_gram_size: int = 2,
                                 samples: int = 500) -> Dict[str, Any]:
    """Build dataset with custom semantic categories"""
    config = DatasetConfig(n_gram_size=n_gram_size, num_samples=samples)
    return build_enhanced_dataset(config, semantic_categories=semantic_categories)


def build_frequency_focused_dataset(frequency_bins: FrequencyBins,
                                   n_gram_size: int = 2,
                                   samples: int = 500) -> Dict[str, Any]:
    """Build dataset with custom frequency bins"""
    config = DatasetConfig(n_gram_size=n_gram_size, num_samples=samples)
    return build_enhanced_dataset(config, frequency_bins=frequency_bins)


def validate_dataset(dataset: Dict[str, Any]) -> Dict[str, Any]:
    """Validate dataset structure and return diagnostic information"""
    diagnostics = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'stats': {}
    }
    
    # Check required fields
    required_fields = ['texts', 'ngrams', 'frequency_categories', 'local_frequencies', 'global_frequencies']
    for field in required_fields:
        if field not in dataset:
            diagnostics['errors'].append(f"Missing required field: {field}")
            diagnostics['valid'] = False
    
    if not diagnostics['valid']:
        return diagnostics
    
    # Check field consistency
    text_count = len(dataset['texts'])
    for field in required_fields:
        if len(dataset[field]) != text_count:
            diagnostics['errors'].append(f"Field {field} length mismatch: {len(dataset[field])} vs {text_count}")
            diagnostics['valid'] = False
    
    # Generate statistics
    if diagnostics['valid']:
        diagnostics['stats'] = {
            'total_samples': text_count,
            'unique_ngrams': len(set(dataset['ngrams'])),
            'frequency_distribution': dict(Counter(dataset['frequency_categories'])),
        }
        
        if 'semantic_categories' in dataset:
            diagnostics['stats']['semantic_distribution'] = dict(Counter(dataset['semantic_categories']))
    
    return diagnostics


if __name__ == "__main__":
    print("🔥 Robust Enhanced N-gram Dataset Builder for Research")
    print("=" * 60)

    try:
        # Quick semantic frequency analysis
        print("\n1️⃣ Running semantic frequency analysis...")
        breakdown = quick_semantic_frequency_analysis(
            n_gram_size=2, 
            samples=200, 
            source="dummy",  # Use dummy for reliable testing
            save_datasets=True
        )
        
        print(f"\n2️⃣ Analysis complete! Found data for {len(breakdown)} semantic categories")
        
        # Show detailed breakdown for polytope research
        print("\n3️⃣ Detailed breakdown for polytope research:")
        for category, freq_data in breakdown.items():
            low_count = freq_data['low']['count']  
            high_count = freq_data['high']['count']
            
            if low_count > 0 or high_count > 0:
                print(f"\n🔬 {category.upper()} category:")
                print(f"   • Low frequency dataset:  {low_count} samples")
                print(f"   • High frequency dataset: {high_count} samples")
                
                # For polytope analysis, you can now use:
                if low_count > 0:
                    low_dataset = freq_data['low']['dataset']
                    print(f"   • Low freq n-grams: {low_dataset['ngrams'][:2]}")
                    
                if high_count > 0:
                    high_dataset = freq_data['high']['dataset']  
                    print(f"   • High freq n-grams: {high_dataset['ngrams'][:2]}")
        
        print("\n✅ Semantic frequency analysis completed!")
        print("📁 Datasets saved to './datasets/semantic_frequency/'")
        print("🎯 You can now compare polytope evolution between low/high freq for each semantic category")
        
    except Exception as e:
        logger.error(f"Example failed: {e}")
        print(f"❌ Example failed: {e}")
        print("This might happen if datasets are not accessible. Try with source='dummy' for testing.")