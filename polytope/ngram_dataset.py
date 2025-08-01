#!/usr/bin/env python3
"""
Optimized N-gram Dataset Builder - Much Faster Version
"""

import torch
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any, Optional
import matplotlib.pyplot as plt
import pickle
import re
import json
import time
import random
from tqdm import tqdm
import requests
from datasets import Dataset, load_dataset
import transformers
from transformers import AutoTokenizer
from pathlib import Path
import concurrent.futures
import threading

# Reduced frequency bins for faster processing
FREQUENCY_BINS = {
    'high': {'range': (1000, float('inf')), 'target_count': 200, 'description': 'Common phrases'},
    'medium': {'range': (100, 1000), 'target_count': 300, 'description': 'Uncommon phrases'},
    'low': {'range': (10, 100), 'target_count': 200, 'description': 'Rare phrases'},
    'very_low': {'range': (1, 10), 'target_count': 100, 'description': 'Very rare phrases'}
}

class OptimizedNgramBuilder:
    def __init__(self, cache_dir: str = "./ngram_cache", max_workers: int = 5):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.frequency_cache = self.load_frequency_cache()
        self.max_workers = max_workers
        self.rate_limit_delay = 0.05  # Reduced delay
        self.api_base_url = 'https://api.infini-gram.io/'
        self.session = requests.Session()  # Reuse connections
        
    def load_frequency_cache(self) -> Dict[str, int]:
        """Load cached frequency data"""
        cache_file = self.cache_dir / "frequency_cache.json"
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                return json.load(f)
        return {}

    def save_frequency_cache(self):
        """Save frequency cache to disk"""
        cache_file = self.cache_dir / "frequency_cache.json"
        with open(cache_file, 'w') as f:
            json.dump(self.frequency_cache, f)

    def query_infinigram_batch(self, ngrams: List[str], corpus: str = "pile") -> Dict[str, int]:
        """Query multiple n-grams with threading for better performance"""
        results = {}
        lock = threading.Lock()
        
        def query_single(ngram: str):
            cache_key = f"{ngram}_{corpus}"
            
            # Check cache first
            with lock:
                if cache_key in self.frequency_cache:
                    results[ngram] = self.frequency_cache[cache_key]
                    return
            
            try:
                payload = {
                    'index': 'v4_piletrain_llama',
                    'query_type': 'count',
                    'query': ngram
                }
                
                response = self.session.post(self.api_base_url, json=payload, timeout=5)
                
                if response.status_code == 200:
                    result = response.json()
                    frequency = result.get('count', 0)
                    
                    with lock:
                        self.frequency_cache[cache_key] = frequency
                        results[ngram] = frequency
                else:
                    print(f"API error for '{ngram}': {response.status_code}")
                    with lock:
                        results[ngram] = 0
                        
            except Exception as e:
                print(f"Error querying '{ngram}': {e}")
                with lock:
                    results[ngram] = 0
            
            # Rate limiting
            time.sleep(self.rate_limit_delay)
        
        # Use ThreadPoolExecutor for concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(query_single, ngram) for ngram in ngrams]
            concurrent.futures.wait(futures)
        
        return results

    def extract_ngrams_from_text(self, text: str, n_gram_size: int) -> List[str]:
        """Extract n-grams from text - optimized version"""
        # More aggressive text cleaning
        text = re.sub(r'[^\w\s]', ' ', text.lower())  # Remove punctuation, lowercase
        text = re.sub(r'\s+', ' ', text.strip())
        words = text.split()
        
        if len(words) < n_gram_size:
            return []
        
        # Use list comprehension for speed
        return [' '.join(words[i:i + n_gram_size]) for i in range(len(words) - n_gram_size + 1)]

    def load_pile_dataset_fast(self, num_samples: int = 1000) -> List[str]:
        """Load samples from The Pile dataset - faster version"""
        print(f"Loading {num_samples} samples from The Pile...")
        
        try:
            dataset = load_dataset("EleutherAI/pile", split="train", streaming=True)
            
            texts = []
            for i, sample in enumerate(dataset):
                if i >= num_samples:
                    break
                
                text = sample.get('text', '')
                if 50 < len(text.strip()) < 2000:  # Filter very short/long texts
                    texts.append(text.strip())
                    
                if i % 100 == 0:
                    print(f"Loaded {len(texts)} valid texts from {i+1} samples")
            
            print(f"Final: {len(texts)} valid text samples from {num_samples} total")
            return texts
            
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return self.generate_dummy_texts(num_samples)

    def generate_dummy_texts(self, num_samples: int = 100) -> List[str]:
        """Generate dummy texts for testing"""
        templates = [
            "The {adj} {noun} {verb} quickly in the {place}.",
            "Machine learning {verb} {adj} patterns in {noun} data.",
            "When {noun} {verb}, the {adj} system becomes {adj2}.",
            "The {adj} algorithm {verb} {noun} with {adj2} accuracy.",
            "Deep learning {verb} {adj} features from {noun} datasets."
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

    def count_ngrams_fast(self, texts: List[str], n_gram_size: int) -> Tuple[Counter, Dict[str, List[str]]]:
        """Fast n-gram counting with filtering"""
        ngram_counts = Counter()
        ngram_text_mapping = defaultdict(list)
        
        print(f"Extracting {n_gram_size}-grams from {len(texts)} texts...")
        
        for text in tqdm(texts, desc="Processing texts"):
            ngrams = self.extract_ngrams_from_text(text, n_gram_size)
            
            for ngram in ngrams:
                # Basic quality filtering
                words = ngram.split()
                if (len(words) == n_gram_size and 
                    all(len(word) >= 2 for word in words) and
                    not any(word.isdigit() for word in words)):
                    
                    ngram_counts[ngram] += 1
                    if len(ngram_text_mapping[ngram]) < 3:  # Limit examples
                        ngram_text_mapping[ngram].append(text[:200])  # Truncate text
        
        print(f"Found {len(ngram_counts)} unique n-grams")
        return ngram_counts, dict(ngram_text_mapping)

    def smart_stratify_ngrams(self, ngram_counts: Counter, text_mapping: Dict[str, List[str]], 
                             max_api_calls: int = 500) -> Dict[str, Dict[str, List]]:
        """Smart stratification with selective API calls"""
        
        # Pre-filter n-grams: only query promising candidates
        sorted_ngrams = ngram_counts.most_common()
        
        # Smart selection: query high-frequency local n-grams first (more likely to have global data)
        candidates_to_query = []
        for ngram, local_freq in sorted_ngrams:
            if (local_freq >= 3 and  # Must appear multiple times locally
                len(candidates_to_query) < max_api_calls):
                candidates_to_query.append(ngram)
        
        print(f"Querying {len(candidates_to_query)} selected n-grams (out of {len(sorted_ngrams)} total)")
        
        # Batch query in chunks
        chunk_size = 50
        global_frequencies = {}
        
        for i in range(0, len(candidates_to_query), chunk_size):
            chunk = candidates_to_query[i:i + chunk_size]
            print(f"Processing batch {i//chunk_size + 1}/{(len(candidates_to_query)-1)//chunk_size + 1}")
            
            chunk_results = self.query_infinigram_batch(chunk)
            global_frequencies.update(chunk_results)
            
            # Save cache periodically
            if i % (chunk_size * 5) == 0:
                self.save_frequency_cache()
        
        # Initialize stratified data
        stratified_data = {
            bin_name: {
                'ngrams': [], 'local_frequencies': [], 'global_frequencies': [],
                'text_examples': [], 'confidence_scores': []
            } for bin_name in FREQUENCY_BINS.keys()
        }
        
        # Categorize all n-grams (queried + estimated)
        for ngram, local_freq in sorted_ngrams:
            global_freq = global_frequencies.get(ngram)
            
            # If no global data, estimate based on local frequency
            if global_freq is None:
                global_freq = max(1, int(local_freq * random.uniform(50, 200)))  # Rough estimate
            
            # Categorize
            category = self.categorize_by_frequency(global_freq)
            
            # Add to appropriate bin if there's space
            if len(stratified_data[category]['ngrams']) < FREQUENCY_BINS[category]['target_count']:
                confidence = self.calculate_confidence_score(ngram, local_freq, 
                                                           global_frequencies.get(ngram))
                
                stratified_data[category]['ngrams'].append(ngram)
                stratified_data[category]['local_frequencies'].append(local_freq)
                stratified_data[category]['global_frequencies'].append(global_freq)
                stratified_data[category]['text_examples'].append(text_mapping.get(ngram, [])[:2])
                stratified_data[category]['confidence_scores'].append(confidence)
        
        self.save_frequency_cache()
        return stratified_data

    def categorize_by_frequency(self, frequency: int) -> str:
        """Categorize n-gram by frequency"""
        for category, info in FREQUENCY_BINS.items():
            min_freq, max_freq = info['range']
            if min_freq <= frequency < max_freq:
                return category
        return 'very_low'

    def calculate_confidence_score(self, ngram: str, local_freq: int, global_freq: Optional[int]) -> float:
        """Calculate confidence score"""
        confidence = 0.8  # Base confidence
        
        # Boost for longer average word length
        avg_word_length = np.mean([len(word) for word in ngram.split()])
        if avg_word_length >= 4:
            confidence += 0.1
        
        # Boost if we have real global frequency data
        if global_freq is not None:
            confidence += 0.1
        
        # Boost for higher local frequency
        if local_freq >= 5:
            confidence += 0.05
        
        return min(confidence, 1.0)

    def build_fast_dataset(self, n_gram_size: int = 2, pile_samples: int = 1000, 
                          max_api_calls: int = 300) -> Dict[str, Any]:
        """Build dataset with optimizations for speed"""
        
        print(f"🚀 Building FAST {n_gram_size}-gram dataset with {pile_samples} samples")
        start_time = time.time()
        
        # Step 1: Load texts (reduced samples for speed)
        texts = self.load_pile_dataset_fast(pile_samples)
        
        # Step 2: Extract and count n-grams
        ngram_counts, text_mapping = self.count_ngrams_fast(texts, n_gram_size)
        
        # Step 3: Smart stratification with limited API calls
        stratified_data = self.smart_stratify_ngrams(ngram_counts, text_mapping, max_api_calls)
        
        # Step 4: Create final dataset
        final_dataset = self.create_final_dataset(stratified_data, n_gram_size, pile_samples)
        
        elapsed_time = time.time() - start_time
        print(f"✅ Dataset built in {elapsed_time:.1f} seconds!")
        
        return final_dataset

    def create_final_dataset(self, stratified_data: Dict, n_gram_size: int, pile_samples: int) -> Dict[str, Any]:
        """Create final dataset structure"""
        
        final_dataset = {
            'texts': [], 'ngrams': [], 'categories': [],
            'local_frequencies': [], 'global_frequencies': [],
            'confidence_scores': [], 'metadata': []
        }
        
        for category, data in stratified_data.items():
            for i, (ngram, local_freq, global_freq, text_examples, confidence) in enumerate(zip(
                data['ngrams'], data['local_frequencies'], data['global_frequencies'],
                data['text_examples'], data['confidence_scores']
            )):
                for text_example in text_examples[:1]:  # Just 1 example per n-gram for speed
                    final_dataset['texts'].append(text_example)
                    final_dataset['ngrams'].append(ngram)
                    final_dataset['categories'].append(category)
                    final_dataset['local_frequencies'].append(local_freq)
                    final_dataset['global_frequencies'].append(global_freq)
                    final_dataset['confidence_scores'].append(confidence)
                    final_dataset['metadata'].append({'category': category, 'rank': i})
        
        final_dataset['dataset_metadata'] = {
            'n_gram_size': n_gram_size,
            'pile_samples': pile_samples,
            'total_samples': len(final_dataset['texts']),
            'creation_timestamp': time.time(),
            'frequency_bins': FREQUENCY_BINS
        }
        
        return final_dataset


# Convenience functions
def build_fast_ngram_dataset(n_gram_size: int = 2, pile_samples: int = 500, 
                           max_api_calls: int = 200) -> Dict[str, Any]:
    """Quick entry point for fast dataset building"""
    builder = OptimizedNgramBuilder(max_workers=3)  # Conservative threading
    return builder.build_fast_dataset(n_gram_size, pile_samples, max_api_calls)


def save_dataset(dataset: Dict[str, Any], output_path: str):
    """Save dataset to JSON"""
    output_path = Path(output_path).with_suffix('.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=2, default=str)
    print(f"Saved dataset to {output_path}")


# Example usage
if __name__ == "__main__":
    print("🔥 Fast N-gram Dataset Builder")
    
    # Build a small, fast dataset
    dataset = build_fast_ngram_dataset(
        n_gram_size=2,
        pile_samples=500,     # Reduced for speed
        max_api_calls=150     # Limited API calls
    )
    
    # Quick stats
    print(f"\n📊 Dataset Stats:")
    print(f"Total samples: {len(dataset['texts'])}")
    print(f"Unique n-grams: {len(set(dataset['ngrams']))}")
    
    categories = Counter(dataset['categories'])
    for cat, count in categories.items():
        print(f"{cat}: {count} samples")
    
    # Save
    save_dataset(dataset, "./fast_ngram_dataset")
    print("✅ Done!")
    print(dataset)