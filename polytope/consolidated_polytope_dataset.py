#!/usr/bin/env python3
"""
Consolidated Polytope Dataset Builder
Combines multiple n-gram datasets and categorizes by frequency for polytope analysis
"""

import json
import numpy as np
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Any, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_all_datasets(data_dir: str = ".") -> Dict[str, Dict[str, Any]]:
    """Load all available n-gram datasets"""
    data_dir = Path(data_dir)
    datasets = {}
    
    # Define dataset files to load
    dataset_files = [
        "consolidated_ngram_dataset.json",
        "country_capital_ngram_dataset.json", 
        "stratified_ngram_dataset.json"
    ]
    
    for filename in dataset_files:
        filepath = data_dir / filename
        if filepath.exists():
            logger.info(f"Loading {filename}...")
            with open(filepath, 'r') as f:
                datasets[filename.replace('.json', '')] = json.load(f)
            logger.info(f"Loaded {len(datasets[filename.replace('.json', '')]['texts'])} records from {filename}")
    
    return datasets


def categorize_by_frequency_heuristics(ngram: str, text: str, source_dataset: str) -> str:
    """
    Categorize n-grams as high or low frequency based on heuristics
    """
    
    # For country-capital dataset, we already have frequency categories
    if source_dataset == "country_capital_ngram_dataset":
        return None  # Will use existing frequency_categories
    
    # High frequency patterns (common words and phrases)
    high_freq_patterns = [
        'the', 'and', 'in', 'to', 'of', 'for', 'with', 'is', 'are', 'was', 'were',
        'system', 'data', 'network', 'process', 'method', 'algorithm', 'learning',
        'technology', 'computer', 'software', 'research', 'analysis', 'study'
    ]
    
    # Low frequency patterns (technical terms, specific concepts)
    low_freq_patterns = [
        'computational', 'pharmaceutical', 'empirical', 'theoretical', 'pedagogical',
        'entrepreneurial', 'technological', 'systematic', 'automated', 'scalable',
        'robust', 'efficient', 'complex', 'advanced', 'intelligent'
    ]
    
    ngram_lower = ngram.lower()
    
    # Check for high frequency patterns
    high_score = sum(1 for pattern in high_freq_patterns if pattern in ngram_lower)
    low_score = sum(1 for pattern in low_freq_patterns if pattern in ngram_lower)
    
    # Additional heuristics
    words = ngram_lower.split()
    
    # Very common function words = high frequency
    if any(word in ['the', 'and', 'in', 'to', 'of', 'for', 'with'] for word in words):
        high_score += 2
    
    # Technical terms = low frequency  
    if any(len(word) > 8 for word in words):  # Long technical words
        low_score += 1
        
    # Repeated common patterns = high frequency
    if any(word in ['system', 'data', 'network', 'process'] for word in words):
        high_score += 1
    
    # Decide based on scores
    if high_score > low_score:
        return 'high'
    elif low_score > high_score:
        return 'low'
    else:
        # Default for tie - use text length as heuristic
        return 'high' if len(ngram) < 15 else 'low'


def consolidate_datasets(datasets: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Consolidate all datasets into a single unified dataset with proper frequency categorization
    """
    
    logger.info("Consolidating datasets...")
    
    consolidated = {
        'texts': [],
        'ngrams': [],
        'ngram_sizes': [],
        'semantic_categories': [],
        'semantic_confidence': [],
        'frequency_categories': [],
        'local_frequencies': [],
        'estimated_global_frequencies': [],
        'source_datasets': [],
        'metadata': {
            'consolidation_info': {},
            'frequency_categorization': 'heuristic_based'
        }
    }
    
    total_records = 0
    
    for dataset_name, dataset in datasets.items():
        logger.info(f"Processing {dataset_name}...")
        
        n_records = len(dataset['texts'])
        
        for i in range(n_records):
            # Extract record data
            text = dataset['texts'][i]
            ngram = dataset['ngrams'][i]
            ngram_size = dataset.get('ngram_sizes', [2] * n_records)[i]
            
            # Handle semantic categories
            if 'semantic_categories' in dataset:
                semantic_cat = dataset['semantic_categories'][i]
                semantic_conf = dataset.get('semantic_confidence', [0.5] * n_records)[i]
            else:
                # Assign default semantic category based on dataset
                if 'country_capital' in dataset_name:
                    semantic_cat = 'geography'
                elif 'technology' in text.lower() or any(word in text.lower() for word in ['algorithm', 'learning', 'network', 'system']):
                    semantic_cat = 'technology'
                else:
                    semantic_cat = 'general'
                semantic_conf = 0.5
            
            # Handle frequency categories
            if 'frequency_categories' in dataset:
                freq_cat = dataset['frequency_categories'][i]
            else:
                freq_cat = categorize_by_frequency_heuristics(ngram, text, dataset_name)
            
            # Handle frequencies
            local_freq = dataset.get('local_frequencies', [1] * n_records)[i]
            global_freq = dataset.get('estimated_global_frequencies', [1] * n_records)[i]
            
            # Add to consolidated dataset
            consolidated['texts'].append(text)
            consolidated['ngrams'].append(ngram)
            consolidated['ngram_sizes'].append(ngram_size)
            consolidated['semantic_categories'].append(semantic_cat)
            consolidated['semantic_confidence'].append(semantic_conf)
            consolidated['frequency_categories'].append(freq_cat)
            consolidated['local_frequencies'].append(local_freq)
            consolidated['estimated_global_frequencies'].append(global_freq)
            consolidated['source_datasets'].append(dataset_name)
            
            total_records += 1
        
        # Update consolidation info
        consolidated['metadata']['consolidation_info'][dataset_name] = {
            'records_added': n_records,
            'original_keys': list(dataset.keys())
        }
    
    logger.info(f"Consolidated {total_records} records from {len(datasets)} datasets")
    
    # Add final metadata
    consolidated['metadata'].update({
        'total_records': total_records,
        'unique_ngrams': len(set(consolidated['ngrams'])),
        'source_datasets': list(datasets.keys()),
        'frequency_distribution': dict(Counter(consolidated['frequency_categories'])),
        'semantic_distribution': dict(Counter(consolidated['semantic_categories']))
    })
    
    return consolidated


def separate_by_frequency(dataset: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Separate dataset into high-frequency and low-frequency subsets
    """
    
    logger.info("Separating by frequency categories...")
    
    high_freq_dataset = {
        'texts': [],
        'ngrams': [],
        'ngram_sizes': [],
        'semantic_categories': [],
        'semantic_confidence': [],
        'frequency_categories': [],
        'local_frequencies': [],
        'estimated_global_frequencies': [],
        'source_datasets': []
    }
    
    low_freq_dataset = {
        'texts': [],
        'ngrams': [],
        'ngram_sizes': [],
        'semantic_categories': [],
        'semantic_confidence': [],
        'frequency_categories': [],
        'local_frequencies': [],
        'estimated_global_frequencies': [],
        'source_datasets': []
    }
    
    for i, freq_cat in enumerate(dataset['frequency_categories']):
        target_dataset = high_freq_dataset if freq_cat == 'high' else low_freq_dataset
        
        for key in target_dataset.keys():
            target_dataset[key].append(dataset[key][i])
    
    # Add metadata
    high_freq_dataset['metadata'] = {
        'frequency_category': 'high',
        'total_records': len(high_freq_dataset['texts']),
        'unique_ngrams': len(set(high_freq_dataset['ngrams'])),
        'semantic_distribution': dict(Counter(high_freq_dataset['semantic_categories']))
    }
    
    low_freq_dataset['metadata'] = {
        'frequency_category': 'low',
        'total_records': len(low_freq_dataset['texts']),
        'unique_ngrams': len(set(low_freq_dataset['ngrams'])),
        'semantic_distribution': dict(Counter(low_freq_dataset['semantic_categories']))
    }
    
    logger.info(f"High frequency dataset: {len(high_freq_dataset['texts'])} records")
    logger.info(f"Low frequency dataset: {len(low_freq_dataset['texts'])} records")
    
    return high_freq_dataset, low_freq_dataset


def prepare_for_polytope_analysis(high_freq_data: Dict[str, Any], 
                                 low_freq_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prepare separated datasets for polytope analysis with convex hull computation
    """
    
    logger.info("Preparing datasets for polytope analysis...")
    
    polytope_data = {
        'high_frequency_data': {
            'texts': high_freq_data['texts'],
            'ngrams': high_freq_data['ngrams'],
            'categories': ['high'] * len(high_freq_data['texts']),
            'semantic_categories': high_freq_data['semantic_categories'],
            'frequency_categories': high_freq_data['frequency_categories'],
            'metadata': high_freq_data['metadata']
        },
        'low_frequency_data': {
            'texts': low_freq_data['texts'],
            'ngrams': low_freq_data['ngrams'],
            'categories': ['low'] * len(low_freq_data['texts']),
            'semantic_categories': low_freq_data['semantic_categories'],
            'frequency_categories': low_freq_data['frequency_categories'],
            'metadata': low_freq_data['metadata']
        },
        'analysis_config': {
            'convex_hull_layers': 'all',  # Compute convex hull for all layers
            'comparison_method': 'frequency_based',
            'polytope_metrics': ['volume', 'complexity_score', 'effective_dimension', 'stability_score']
        }
    }
    
    return polytope_data


def save_consolidated_datasets(consolidated_data: Dict[str, Any],
                              high_freq_data: Dict[str, Any],
                              low_freq_data: Dict[str, Any],
                              polytope_data: Dict[str, Any],
                              output_dir: str = ".") -> Dict[str, str]:
    """
    Save all generated datasets to files
    """
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    saved_files = {}
    
    # Save consolidated dataset
    consolidated_path = output_dir / "consolidated_complete_dataset.json"
    with open(consolidated_path, 'w') as f:
        json.dump(consolidated_data, f, indent=2, default=str)
    saved_files['consolidated'] = str(consolidated_path)
    
    # Save frequency-separated datasets
    high_freq_path = output_dir / "high_frequency_dataset.json" 
    with open(high_freq_path, 'w') as f:
        json.dump(high_freq_data, f, indent=2, default=str)
    saved_files['high_frequency'] = str(high_freq_path)
    
    low_freq_path = output_dir / "low_frequency_dataset.json"
    with open(low_freq_path, 'w') as f:
        json.dump(low_freq_data, f, indent=2, default=str)
    saved_files['low_frequency'] = str(low_freq_path)
    
    # Save polytope analysis ready data
    polytope_path = output_dir / "polytope_analysis_data.json"
    with open(polytope_path, 'w') as f:
        json.dump(polytope_data, f, indent=2, default=str)
    saved_files['polytope_analysis'] = str(polytope_path)
    
    logger.info(f"Saved all datasets to {output_dir}")
    return saved_files


def print_consolidation_summary(consolidated_data: Dict[str, Any],
                               high_freq_data: Dict[str, Any],
                               low_freq_data: Dict[str, Any]):
    """
    Print comprehensive summary of the consolidation process
    """
    
    print("\n" + "="*60)
    print("DATASET CONSOLIDATION SUMMARY")
    print("="*60)
    
    print(f"\n📊 CONSOLIDATED DATASET:")
    print(f"   Total records: {len(consolidated_data['texts'])}")
    print(f"   Unique n-grams: {len(set(consolidated_data['ngrams']))}")
    print(f"   Source datasets: {len(consolidated_data['metadata']['source_datasets'])}")
    
    print(f"\n📈 FREQUENCY DISTRIBUTION:")
    freq_dist = consolidated_data['metadata']['frequency_distribution']
    for freq_type, count in freq_dist.items():
        print(f"   {freq_type}: {count} records ({count/len(consolidated_data['texts'])*100:.1f}%)")
    
    print(f"\n🏷️  SEMANTIC DISTRIBUTION:")
    sem_dist = consolidated_data['metadata']['semantic_distribution']
    for sem_type, count in sem_dist.items():
        print(f"   {sem_type}: {count} records")
    
    print(f"\n🔥 HIGH FREQUENCY SUBSET:")
    print(f"   Records: {len(high_freq_data['texts'])}")
    print(f"   Unique n-grams: {len(set(high_freq_data['ngrams']))}")
    print(f"   Semantic categories: {len(set(high_freq_data['semantic_categories']))}")
    
    print(f"\n❄️  LOW FREQUENCY SUBSET:")
    print(f"   Records: {len(low_freq_data['texts'])}")
    print(f"   Unique n-grams: {len(set(low_freq_data['ngrams']))}")
    print(f"   Semantic categories: {len(set(low_freq_data['semantic_categories']))}")
    
    print(f"\n✅ READY FOR POLYTOPE ANALYSIS")
    print(f"   High-freq data prepared for convex hull analysis")
    print(f"   Low-freq data prepared for comparison")
    print(f"   Frequency-based superposition analysis ready")
    
    print("="*60)


def main():
    """
    Main consolidation pipeline
    """
    
    print("🔥 Consolidating N-gram Datasets for Polytope Analysis")
    
    # Step 1: Load all available datasets
    datasets = load_all_datasets()
    
    if not datasets:
        raise ValueError("No datasets found. Ensure the dataset JSON files are in the current directory.")
    
    # Step 2: Consolidate all datasets
    consolidated_data = consolidate_datasets(datasets)
    
    # Step 3: Separate by frequency
    high_freq_data, low_freq_data = separate_by_frequency(consolidated_data)
    
    # Step 4: Prepare for polytope analysis
    polytope_data = prepare_for_polytope_analysis(high_freq_data, low_freq_data)
    
    # Step 5: Save all datasets
    saved_files = save_consolidated_datasets(
        consolidated_data, high_freq_data, low_freq_data, polytope_data
    )
    
    # Step 6: Print summary
    print_consolidation_summary(consolidated_data, high_freq_data, low_freq_data)
    
    print(f"\n💾 SAVED FILES:")
    for file_type, filepath in saved_files.items():
        print(f"   {file_type}: {filepath}")
    
    return {
        'consolidated_data': consolidated_data,
        'high_freq_data': high_freq_data,
        'low_freq_data': low_freq_data,
        'polytope_data': polytope_data,
        'saved_files': saved_files
    }


if __name__ == "__main__":
    result = main()