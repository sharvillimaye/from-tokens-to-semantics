#!/usr/bin/env python3
"""
Test script for the optimized n-gram miner with rolling hash GPU acceleration.
"""

import numpy as np
from pathlib import Path
import tempfile
import multiprocessing as mp
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
    # Set multiprocessing start method
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    logger.info("Starting optimized n-gram miner test")
    
    # Create a temporary directory for test files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Initialize miner with optimized settings
        miner = FastNgramMiner(
            tokenizer_name="EleutherAI/gpt-neox-20b",
            chunk_tokens=2_000_000,  # Small chunks for testing
            n_workers=2,  # Few workers for testing
            use_gpu=True,
            max_memory_gb=4.0
        )
        
        # Create test texts with known patterns
        test_texts = [
            "Paris is the capital of France. It is a beautiful city with many attractions.",
            "Tokyo is the capital of Japan. It is a modern metropolis with advanced technology.",
            "London is the capital of England. It has rich history and cultural heritage.",
            "Paris and Tokyo are both major cities in their respective countries.",
            "France and Japan have different cultures and culinary traditions.",
            "England and France are both European countries with long histories.",
            "The weather in Paris is often rainy, just like in London during winter.",
            "Tokyo has excellent public transportation systems and efficient railways.",
            "France is known for its wine, cheese, and exquisite cuisine.",
            "Japan is famous for its technology, anime, and traditional culture.",
            # Add more repetitions to test pattern frequency
            "Paris, the city of lights, attracts millions of tourists every year.",
            "Tokyo Tower is one of the most famous landmarks in Japan.",
            "London Bridge is falling down, falling down, according to the nursery rhyme.",
            "France produces some of the world's finest wines in regions like Bordeaux.",
            "Japan's cherry blossoms bloom beautifully in spring across the country."
        ]
        
        # Create test shard
        shard_path = temp_path / "test_shard.bin"
        num_tokens = create_test_shard(test_texts, miner.tokenizer, shard_path)
        
        # Define probe phrases to search for
        probe_phrases = [
            "paris", "france", "tokyo", "japan", "london", "england",
            "capital", "city", "country", "weather", "transportation",
            "technology", "culture", "history", "famous", "beautiful"
        ]
        
        logger.info(f"Testing with {len(probe_phrases)} probe phrases")
        
        # Run the optimized miner
        try:
            results = miner.mine_and_count(
                shard_paths=[shard_path],
                probe_phrases=probe_phrases,
                max_sentences_per_shard=50,
                batch_size=2
            )
            
            # Print results
            print("\n=== OPTIMIZED MINING RESULTS ===")
            print(f"Total patterns found: {len(results['counts'])}")
            print(f"Total sentences harvested: {len(results['sentences'])}")
            
            if 'stats' in results:
                stats = results['stats']
                print(f"Total time: {stats.get('total_time', 0):.2f}s")
                print(f"Completed tasks: {stats.get('completed_tasks', 0)}")
                print(f"Failed tasks: {stats.get('failed_tasks', 0)}")
                
                if stats.get('completed_tasks', 0) > 0:
                    tokens_per_sec = num_tokens / stats.get('total_time', 1)
                    print(f"Processing speed: {tokens_per_sec:,.0f} tokens/sec")
            
            print("\n=== TOP PATTERNS ===")
            sorted_counts = sorted(results['counts'].items(), key=lambda x: -x[1])
            for pattern, count in sorted_counts[:15]:
                print(f"'{pattern}': {count}")
            
            print("\n=== SAMPLE SENTENCES ===")
            for i, sentence in enumerate(results['sentences'][:5]):
                print(f"{i+1}. {sentence[:100]}{'...' if len(sentence) > 100 else ''}")
            
            # Verify expected patterns are found
            expected_patterns = ["paris", "france", "tokyo", "japan", "london", "england"]
            found_patterns = set(results['counts'].keys())
            
            print("\n=== PATTERN VERIFICATION ===")
            success_count = 0
            for pattern in expected_patterns:
                if pattern in found_patterns:
                    print(f"✓ Found '{pattern}': {results['counts'][pattern]} occurrences")
                    success_count += 1
                else:
                    print(f"✗ Missing '{pattern}'")
            
            # Performance check
            total_patterns = len(results['counts'])
            if total_patterns > 5:
                print(f"✓ Found {total_patterns} total patterns (good coverage)")
            else:
                print(f"⚠ Only found {total_patterns} patterns (may need tuning)")
            
            logger.info("Optimized test completed successfully!")
            return success_count >= 4  # At least 4/6 expected patterns
            
        except Exception as e:
            logger.error(f"Test failed with error: {e}")
            import traceback
            traceback.print_exc()
            return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Optimized n-gram miner test passed!")
    else:
        print("\n❌ Test failed!")
        exit(1)




