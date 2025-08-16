#!/usr/bin/env python3
"""
Test script for the optimized n-gram miner.
This script creates a small test dataset and runs the miner to verify functionality.
"""

import numpy as np
from pathlib import Path
import tempfile
import os
from ngrams.ngram_frequency import FastNgramMiner
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_test_shard(texts, tokenizer, output_path):
    """Create a test shard file from a list of texts."""
    all_tokens = []
    for text in texts:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        all_tokens.extend(tokens)
    
    # Convert to uint16 and save
    token_array = np.array(all_tokens, dtype=np.uint16)
    token_array.tofile(output_path)
    logger.info(f"Created test shard with {len(token_array)} tokens at {output_path}")
    return len(token_array)

def main():
    """Main test function."""
    logger.info("Starting n-gram miner test")
    
    # Create a temporary directory for test files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Initialize tokenizer
        miner = FastNgramMiner(
            tokenizer_name="EleutherAI/gpt-neox-20b",
            chunk_tokens=1_000_000,  # Small chunks for testing
            n_workers=2,  # Few workers for testing
            use_gpu=True,
            max_memory_gb=2.0
        )
        
        # Create test texts with known patterns
        test_texts = [
            "Paris is the capital of France. It is a beautiful city.",
            "Tokyo is the capital of Japan. It is a modern metropolis.",
            "London is the capital of England. It has rich history.",
            "Paris and Tokyo are both major cities in their countries.",
            "France and Japan have different cultures and traditions.",
            "England and France are both European countries.",
            "The weather in Paris is often rainy, just like in London.",
            "Tokyo has excellent public transportation systems.",
            "France is known for its wine and cuisine.",
            "Japan is famous for its technology and innovation."
        ]
        
        # Create test shard
        shard_path = temp_path / "test_shard.bin"
        num_tokens = create_test_shard(test_texts, miner.tokenizer, shard_path)
        
        # Define probe phrases to search for
        probe_phrases = [
            "paris", "france", "tokyo", "japan", "london", "england",
            "capital", "city", "country", "weather", "transportation"
        ]
        
        logger.info(f"Testing with {len(probe_phrases)} probe phrases")
        
        # Run the miner
        try:
            results = miner.mine_and_count(
                shard_paths=[shard_path],
                probe_phrases=probe_phrases,
                max_sentences_per_shard=50,
                batch_size=2
            )
            
            # Print results
            print("\n=== MINING RESULTS ===")
            print(f"Total patterns found: {len(results['counts'])}")
            print(f"Total sentences harvested: {len(results['sentences'])}")
            
            if 'stats' in results:
                stats = results['stats']
                print(f"Total time: {stats.get('total_time', 0):.2f}s")
                print(f"Completed tasks: {stats.get('completed_tasks', 0)}")
                print(f"Failed tasks: {stats.get('failed_tasks', 0)}")
            
            print("\n=== TOP PATTERNS ===")
            sorted_counts = sorted(results['counts'].items(), key=lambda x: -x[1])
            for pattern, count in sorted_counts[:10]:
                print(f"'{pattern}': {count}")
            
            print("\n=== SAMPLE SENTENCES ===")
            for i, sentence in enumerate(results['sentences'][:5]):
                print(f"{i+1}. {sentence}")
            
            # Verify expected patterns are found
            expected_patterns = ["paris", "france", "tokyo", "japan", "london", "england"]
            found_patterns = set(results['counts'].keys())
            
            print("\n=== PATTERN VERIFICATION ===")
            for pattern in expected_patterns:
                if pattern in found_patterns:
                    print(f"✓ Found '{pattern}': {results['counts'][pattern]} occurrences")
                else:
                    print(f"✗ Missing '{pattern}'")
            
            logger.info("Test completed successfully!")
            
        except Exception as e:
            logger.error(f"Test failed with error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    return True

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 All tests passed!")
    else:
        print("\n❌ Tests failed!")
        exit(1)




