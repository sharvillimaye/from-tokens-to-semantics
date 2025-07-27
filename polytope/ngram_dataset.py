#!/usr/bin/env python3
"""
Improved N-gram Dataset Builder for Polytope Analysis
Robust implementation for analyzing n-gram frequency patterns and activation evolution
"""

import torch
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any, Optional, Generator
import matplotlib.pyplot as plt
import seaborn as sns
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


class ImprovedNGramDatasetBuilder:
    """Enhanced N-gram dataset builder with robust frequency binning and text processing"""
    
    def __init__(self, 
                 api_base_url: str = 'https://api.infini-gram.io/',
                 cache_dir: str = "./ngram_cache",
                 max_samples: int = 10000):
        self.api_base_url = api_base_url
        self.cache_dir = Path(cache_dir)
        self.max_samples = max_samples
        self.rate_limit_delay = 0.1
        
        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Frequency bins for stratification
        self.frequency_bins = {
            'very_high': {'range': (100000, float('inf')), 'target_count': 500, 'description': 'Extremely common phrases'},
            'high': {'range': (10000, 100000), 'target_count': 1000, 'description': 'Very common phrases'},
            'medium': {'range': (1000, 10000), 'target_count': 1500, 'description': 'Common phrases'},
            'low': {'range': (100, 1000), 'target_count': 1000, 'description': 'Uncommon phrases'},
            'rare': {'range': (10, 100), 'target_count': 500, 'description': 'Rare phrases'},
            'very_rare': {'range': (1, 10), 'target_count': 300, 'description': 'Very rare phrases'}
        }
        
        # Initialize frequency cache
        self.frequency_cache = self._load_frequency_cache()
    
    def _load_frequency_cache(self) -> Dict[str, int]:
        """Load cached frequency data"""
        cache_file = self.cache_dir / "frequency_cache.json"
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_frequency_cache(self):
        """Save frequency cache to disk"""
        cache_file = self.cache_dir / "frequency_cache.json"
        with open(cache_file, 'w') as f:
            json.dump(self.frequency_cache, f)
    
    def query_infinigram_frequency(self, ngram: str, corpus: str = "pile") -> int:
        """Query Infinigram API with caching and error handling"""
        
        # Check cache first
        cache_key = f"{ngram}_{corpus}"
        if cache_key in self.frequency_cache:
            return self.frequency_cache[cache_key]
        
        try:
            # Infinigram API call
            payload = {
                'index': 'v4_piletrain_llama',
                'query_type': 'count',
                'query': ngram
            }
            
            response = requests.post(self.api_base_url, json=payload, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                frequency = result.get('count', 0)
                
                # Cache the result
                self.frequency_cache[cache_key] = frequency
                
                # Rate limiting
                time.sleep(self.rate_limit_delay)
                return frequency
            else:
                print(f"API error for '{ngram}': {response.status_code}")
                return 0
                
        except Exception as e:
            print(f"Error querying frequency for '{ngram}': {e}")
            return 0
    
    def extract_ngrams_robust(self, text: str, n: int = 2, min_length: int = 2) -> List[Dict[str, Any]]:
        """Extract n-grams with position information and cleaning"""
        
        # Clean text while preserving positions
        cleaned_text = re.sub(r'\s+', ' ', text.strip())
        
        # Tokenize preserving positions
        words_with_positions = []
        for match in re.finditer(r'\b\w+\b', cleaned_text):
            words_with_positions.append({
                'word': match.group().lower(),
                'start': match.start(),
                'end': match.end()
            })
        
        ngrams_info = []
        
        # Extract n-grams with position tracking
        for i in range(len(words_with_positions) - n + 1):
            ngram_words = words_with_positions[i:i+n]
            ngram_text = ' '.join([w['word'] for w in ngram_words])
            
            # Skip if any word is too short
            if any(len(w['word']) < min_length for w in ngram_words):
                continue
            
            # Calculate character positions
            char_start = ngram_words[0]['start']
            char_end = ngram_words[-1]['end']
            
            ngrams_info.append({
                'ngram': ngram_text,
                'char_start': char_start,
                'char_end': char_end,
                'words': [w['word'] for w in ngram_words],
                'word_positions': [(w['start'], w['end']) for w in ngram_words]
            })
        
        return ngrams_info
    
    def load_pile_dataset_robust(self, max_samples: int = None) -> Generator[str, None, None]:
        """Robust dataset loading with multiple fallback strategies"""
        
        if max_samples is None:
            max_samples = self.max_samples
        
        dataset_strategies = [
            # Strategy 1: Direct Pile loading
            {
                'name': 'EleutherAI/pile',
                'config': 'all',
                'split': 'train'
            },
            # Strategy 2: Alternative Pile dataset
            {
                'name': 'monology/pile-uncopyrighted',
                'config': None,
                'split': 'train'
            },
            # Strategy 3: C4 dataset as fallback
            {
                'name': 'c4',
                'config': 'en',
                'split': 'train'
            }
        ]
        
        for strategy in dataset_strategies:
            try:
                print(f"Attempting to load {strategy['name']}...")
                
                if strategy['config']:
                    dataset = load_dataset(
                        strategy['name'], 
                        strategy['config'], 
                        split=strategy['split'], 
                        streaming=True
                    )
                else:
                    dataset = load_dataset(
                        strategy['name'], 
                        split=strategy['split'], 
                        streaming=True
                    )
                
                count = 0
                for item in dataset:
                    if count >= max_samples:
                        break
                    
                    text = item.get('text', '')
                    if len(text) > 200:  # Filter short texts
                        yield text
                        count += 1
                
                print(f"Successfully loaded {count} texts from {strategy['name']}")
                return
                
            except Exception as e:
                print(f"Failed to load {strategy['name']}: {e}")
                continue
        
        # Ultimate fallback
        print("All dataset loading failed, using synthetic data...")
        yield from self._generate_synthetic_texts(max_samples)
    
    def _generate_synthetic_texts(self, max_samples: int) -> Generator[str, None, None]:
        """Generate synthetic texts for testing when datasets fail"""
        
        templates = [
            "Machine learning algorithms are transforming the way we analyze data in various industries.",
            "Climate change represents one of the most significant challenges facing humanity today.",
            "The rapid development of artificial intelligence has profound implications for society.",
            "Data science combines statistical analysis with computational methods to extract insights.",
            "Neural networks can learn complex patterns from large datasets through training processes.",
            "Natural language processing enables computers to understand and generate human language.",
            "Deep learning models have achieved remarkable success in computer vision tasks.",
            "The internet of things connects everyday devices to create smart environments.",
            "Quantum computing promises to solve certain problems exponentially faster than classical computers.",
            "Biotechnology advances are revolutionizing medicine and pharmaceutical research."
        ]
        
        for i in range(max_samples):
            # Create variations by combining templates
            base_text = random.choice(templates)
            variation = base_text.replace("the", "a").replace("are", "have been")
            yield variation
    
    def build_stratified_ngram_dataset(self, 
                                     n_gram_size: int = 2,
                                     pile_samples: int = 5000,
                                     min_word_length: int = 2) -> Dict[str, Any]:
        """Build stratified n-gram dataset with improved frequency analysis"""
        
        print(f"Building stratified {n_gram_size}-gram dataset...")
        print(f"Target samples: {pile_samples}")
        
        # Step 1: Collect n-grams from corpus
        ngram_frequency_map = defaultdict(int)
        ngram_text_mapping = defaultdict(list)  # Track source texts
        
        print("Extracting n-grams from corpus...")
        text_count = 0
        
        for text in tqdm(self.load_pile_dataset_robust(pile_samples), desc="Processing texts"):
            ngrams_info = self.extract_ngrams_robust(text, n_gram_size, min_word_length)
            
            # Sample n-grams to manage memory
            sampled_ngrams = random.sample(ngrams_info, min(30, len(ngrams_info)))
            
            for ngram_info in sampled_ngrams:
                ngram = ngram_info['ngram']
                ngram_frequency_map[ngram] += 1
                ngram_text_mapping[ngram].append({
                    'text_id': text_count,
                    'text': text,
                    'char_start': ngram_info['char_start'],
                    'char_end': ngram_info['char_end'],
                    'words': ngram_info['words']
                })
            
            text_count += 1
            
            # Progress reporting
            if text_count % 1000 == 0:
                print(f"Processed {text_count} texts, found {len(ngram_frequency_map)} unique n-grams")
        
        print(f"Extracted {len(ngram_frequency_map)} unique {n_gram_size}-grams from {text_count} texts")
        
        # Step 2: Stratify by frequency and get global frequencies
        stratified_dataset = self._stratify_and_validate_ngrams(
            ngram_frequency_map, 
            ngram_text_mapping,
            max_api_calls=1000  # Limit API calls
        )
        
        # Step 3: Create final dataset structure
        final_dataset = self._create_final_dataset_structure(stratified_dataset)
        
        # Save frequency cache
        self._save_frequency_cache()
        
        return final_dataset
    
    def _stratify_and_validate_ngrams(self, 
                                    local_freq_map: Dict[str, int],
                                    text_mapping: Dict[str, List[Dict]],
                                    max_api_calls: int = 1000) -> Dict[str, Any]:
        """Stratify n-grams by frequency with API validation"""
        
        # Sort by local frequency
        sorted_ngrams = sorted(local_freq_map.items(), key=lambda x: x[1], reverse=True)
        
        stratified_data = {
            bin_name: {
                'ngrams': [],
                'local_frequencies': [],
                'global_frequencies': [],
                'text_examples': [],
                'confidence_scores': []
            } for bin_name in self.frequency_bins.keys()
        }
        
        api_call_count = 0
        
        print("Stratifying n-grams and querying global frequencies...")
        
        for ngram, local_freq in tqdm(sorted_ngrams[:5000], desc="Processing n-grams"):
            
            # Get global frequency (with API limiting)
            if api_call_count < max_api_calls:
                global_freq = self.query_infinigram_frequency(ngram)
                api_call_count += 1
            else:
                # Estimate based on local frequency for remaining items
                global_freq = self._estimate_global_frequency(local_freq)
            
            # Categorize based on global frequency (or estimate)
            category = self._categorize_by_frequency(global_freq or local_freq)
            
            # Check if we need more samples in this category
            if len(stratified_data[category]['ngrams']) < self.frequency_bins[category]['target_count']:
                
                # Calculate confidence score
                confidence = self._calculate_confidence_score(ngram, local_freq, global_freq)
                
                stratified_data[category]['ngrams'].append(ngram)
                stratified_data[category]['local_frequencies'].append(local_freq)
                stratified_data[category]['global_frequencies'].append(global_freq or -1)
                stratified_data[category]['text_examples'].append(text_mapping[ngram][:3])  # First 3 examples
                stratified_data[category]['confidence_scores'].append(confidence)
        
        print(f"Made {api_call_count} API calls")
        return stratified_data
    
    def _estimate_global_frequency(self, local_freq: int) -> int:
        """Estimate global frequency based on local frequency"""
        # Simple heuristic: assume local corpus is ~0.1% of global corpus
        return int(local_freq * 1000)
    
    def _categorize_by_frequency(self, frequency: int) -> str:
        """Categorize n-gram by frequency"""
        for category, info in self.frequency_bins.items():
            min_freq, max_freq = info['range']
            if min_freq <= frequency < max_freq:
                return category
        return 'very_rare'  # Default
    
    def _calculate_confidence_score(self, ngram: str, local_freq: int, global_freq: Optional[int]) -> float:
        """Calculate confidence score for n-gram quality"""
        
        confidence = 1.0
        
        # Penalize very short n-grams
        avg_word_length = np.mean([len(word) for word in ngram.split()])
        if avg_word_length < 3:
            confidence *= 0.8
        
        # Penalize single character words
        if any(len(word) == 1 for word in ngram.split()):
            confidence *= 0.6
        
        # Boost if we have global frequency data
        if global_freq is not None and global_freq > 0:
            confidence *= 1.2
        
        # Boost based on local frequency (indicates robustness)
        if local_freq >= 5:
            confidence *= 1.1
        
        return min(confidence, 1.0)
    
    def _create_final_dataset_structure(self, stratified_data: Dict) -> Dict[str, Any]:
        """Create final dataset with organized structure"""
        
        final_dataset = {
            'texts': [],
            'ngrams': [],
            'categories': [],
            'local_frequencies': [],
            'global_frequencies': [],
            'positions': [],
            'char_start': [],
            'char_end': [],
            'confidence_scores': [],
            'metadata': []
        }
        
        total_samples = 0
        
        for category, data in stratified_data.items():
            print(f"Processing category '{category}': {len(data['ngrams'])} n-grams")
            
            for i, (ngram, local_freq, global_freq, text_examples, confidence) in enumerate(zip(
                data['ngrams'], 
                data['local_frequencies'], 
                data['global_frequencies'],
                data['text_examples'],
                data['confidence_scores']
            )):
                
                # Use the first example text
                if text_examples:
                    example = text_examples[0]
                    text = example['text']
                    
                    final_dataset['texts'].append(text)
                    final_dataset['ngrams'].append(ngram)
                    final_dataset['categories'].append(category)
                    final_dataset['local_frequencies'].append(local_freq)
                    final_dataset['global_frequencies'].append(global_freq)
                    final_dataset['positions'].append(example['char_start'])
                    final_dataset['char_start'].append(example['char_start'])
                    final_dataset['char_end'].append(example['char_end'])
                    final_dataset['confidence_scores'].append(confidence)
                    final_dataset['metadata'].append({
                        'text_length': len(text),
                        'ngram_word_count': len(ngram.split()),
                        'words': example['words'],
                        'category_description': self.frequency_bins[category]['description'],
                        'frequency_range': self.frequency_bins[category]['range']
                    })
                    
                    total_samples += 1
        
        print(f"Created final dataset with {total_samples} samples")
        return final_dataset
    
    def save_dataset(self, dataset: Dict[str, Any], 
                    save_path: str = "./ngram_dataset",
                    format: str = "both") -> None:
        """Save dataset in multiple formats"""
        
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        if format in ["json", "both"]:
            # Save as JSON
            json_path = save_dir / "dataset.json"
            with open(json_path, 'w') as f:
                json.dump(dataset, f, indent=2, default=str)
            print(f"Saved JSON dataset to {json_path}")
        
        if format in ["hf", "both"]:
            # Save as HuggingFace dataset
            hf_dataset = Dataset.from_dict(dataset)
            hf_path = save_dir / "hf_dataset"
            hf_dataset.save_to_disk(str(hf_path))
            print(f"Saved HuggingFace dataset to {hf_path}")
        
        # Save analysis report
        report = self.generate_analysis_report(dataset)
        report_path = save_dir / "analysis_report.md"
        with open(report_path, 'w') as f:
            f.write(report)
        print(f"Saved analysis report to {report_path}")
    
    def generate_analysis_report(self, dataset: Dict[str, Any]) -> str:
        """Generate comprehensive analysis report"""
        
        df = pd.DataFrame(dataset)
        
        report = f"""# N-Gram Dataset Analysis Report

## Dataset Overview
- **Total samples**: {len(df)}
- **Unique n-grams**: {df['ngrams'].nunique()}
- **Categories**: {df['categories'].nunique()}
- **Mean confidence score**: {df['confidence_scores'].mean():.3f}

## Category Distribution
```
{df['categories'].value_counts().to_string()}
```

## Frequency Analysis
### Local Frequencies
- **Range**: {df['local_frequencies'].min()} - {df['local_frequencies'].max()}
- **Mean**: {df['local_frequencies'].mean():.2f}
- **Median**: {df['local_frequencies'].median():.2f}

### Global Frequencies (where available)
"""
        
        global_freq_available = df[df['global_frequencies'] > 0]
        if len(global_freq_available) > 0:
            report += f"""- **Available**: {len(global_freq_available)} / {len(df)} samples
- **Range**: {global_freq_available['global_frequencies'].min()} - {global_freq_available['global_frequencies'].max()}
- **Mean**: {global_freq_available['global_frequencies'].mean():.2f}
"""
        else:
            report += "- No global frequency data available\n"
        
        report += f"""
## Text Statistics
- **Mean text length**: {df['texts'].str.len().mean():.0f} characters
- **Text length range**: {df['texts'].str.len().min()} - {df['texts'].str.len().max()}

## Sample N-grams by Category
"""
        
        for category in df['categories'].unique():
            cat_df = df[df['categories'] == category]
            sample_ngrams = cat_df.nlargest(5, 'confidence_scores')['ngrams'].tolist()
            report += f"\n### {category.replace('_', ' ').title()}\n"
            report += f"**Count**: {len(cat_df)}\n"
            report += f"**Sample n-grams**: {', '.join(sample_ngrams)}\n"
        
        return report


def main():
    """Main function to build and save dataset"""
    
    builder = ImprovedNGramDatasetBuilder(max_samples=3000)
    
    print("Building improved n-gram dataset...")
    dataset = builder.build_stratified_ngram_dataset(
        n_gram_size=2,
        pile_samples=2000,
        min_word_length=2
    )
    
    print("\nSaving dataset...")
    builder.save_dataset(dataset, "./improved_ngram_dataset", format="both")
    
    print("\nDataset creation complete!")
    return dataset


if __name__ == "__main__":
    dataset = main()
