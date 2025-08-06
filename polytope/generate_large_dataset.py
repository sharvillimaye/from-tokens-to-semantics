#!/usr/bin/env python3
"""
Generate large datasets for reliable polytope analysis
Addresses the small sample size issue (50 samples) by creating datasets with 500+ samples per group
"""

import json
from pathlib import Path
from polytope.ngram_dataset import (
    build_stratified_ngram_dataset, 
    build_country_capital_dataset,
    DatasetSource,
    NGramConfig
)
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_large_semantic_dataset():
    """Generate a large semantic dataset with 500+ samples per group"""
    
    logger.info("Generating large semantic dataset...")
    
    # Use larger sample sizes for reliable polytope analysis
    config = NGramConfig(
        n_gram_size=2,
        num_samples=10000,  # Much larger base dataset
        source=DatasetSource.PILE,
        random_seed=42
    )
    
    dataset = build_stratified_ngram_dataset(
        n_gram_size=2,
        source=DatasetSource.PILE,
        num_samples=10000,
        samples_per_group=500  # 500 samples per frequency group
    )
    
    # Save the dataset
    output_path = "datasets/semantic_frequency/large_semantic_dataset.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    logger.info(f"✅ Large semantic dataset saved to {output_path}")
    logger.info(f"Dataset contains {len(dataset['texts'])} total texts")
    
    # Print summary
    high_freq_count = len([r for r in dataset['records'] if r['frequency_category'] == 'high'])
    low_freq_count = len([r for r in dataset['records'] if r['frequency_category'] == 'low'])
    
    logger.info(f"High frequency samples: {high_freq_count}")
    logger.info(f"Low frequency samples: {low_freq_count}")
    
    return dataset

def generate_large_country_capital_dataset():
    """Generate a large country-capital dataset with 500+ samples per group"""
    
    logger.info("Generating large country-capital dataset...")
    
    dataset = build_country_capital_dataset(
        num_samples=2000,  # Larger base dataset
        samples_per_group=500  # 500 samples per frequency group
    )
    
    # Save the dataset
    output_path = "datasets/semantic_frequency/large_country_capital_dataset.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    logger.info(f"✅ Large country-capital dataset saved to {output_path}")
    logger.info(f"Dataset contains {len(dataset['texts'])} total texts")
    
    # Print summary
    high_freq_count = len([r for r in dataset['records'] if r['frequency_category'] == 'high'])
    low_freq_count = len([r for r in dataset['records'] if r['frequency_category'] == 'low'])
    
    logger.info(f"High frequency samples: {high_freq_count}")
    logger.info(f"Low frequency samples: {low_freq_count}")
    
    return dataset

def generate_combined_large_dataset():
    """Generate a combined large dataset with multiple sources"""
    
    logger.info("Generating combined large dataset...")
    
    # Generate both datasets
    semantic_dataset = generate_large_semantic_dataset()
    country_dataset = generate_large_country_capital_dataset()
    
    # Combine datasets
    combined_dataset = {
        'texts': semantic_dataset['texts'] + country_dataset['texts'],
        'records': semantic_dataset['records'] + country_dataset['records'],
        'metadata': {
            'semantic_samples': len(semantic_dataset['texts']),
            'country_capital_samples': len(country_dataset['texts']),
            'total_samples': len(semantic_dataset['texts']) + len(country_dataset['texts']),
            'description': 'Combined large dataset for reliable polytope analysis'
        }
    }
    
    # Save combined dataset
    output_path = "datasets/semantic_frequency/combined_large_dataset.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(combined_dataset, f, indent=2)
    
    logger.info(f"✅ Combined large dataset saved to {output_path}")
    
    # Print final summary
    high_freq_count = len([r for r in combined_dataset['records'] if r['frequency_category'] == 'high'])
    low_freq_count = len([r for r in combined_dataset['records'] if r['frequency_category'] == 'low'])
    
    logger.info(f"Combined dataset summary:")
    logger.info(f"  Total texts: {len(combined_dataset['texts'])}")
    logger.info(f"  High frequency samples: {high_freq_count}")
    logger.info(f"  Low frequency samples: {low_freq_count}")
    logger.info(f"  Semantic samples: {len(semantic_dataset['texts'])}")
    logger.info(f"  Country-capital samples: {len(country_dataset['texts'])}")
    
    return combined_dataset

def main():
    """Generate all large datasets"""
    
    logger.info("🚀 Generating large datasets for reliable polytope analysis...")
    logger.info("This addresses the small sample size issue (50 samples → 500+ samples)")
    
    try:
        # Generate individual datasets
        semantic_dataset = generate_large_semantic_dataset()
        country_dataset = generate_large_country_capital_dataset()
        
        # Generate combined dataset
        combined_dataset = generate_combined_large_dataset()
        
        logger.info("🎉 All large datasets generated successfully!")
        logger.info("")
        logger.info("Next steps:")
        logger.info("1. Use these larger datasets for checkpoint analysis")
        logger.info("2. Expect much more stable and reliable polytope metrics")
        logger.info("3. Look for clearer separation between high and low frequency patterns")
        
    except Exception as e:
        logger.error(f"❌ Error generating datasets: {e}")
        raise

if __name__ == "__main__":
    main() 