#!/usr/bin/env python
"""
Script to run checkpoint analysis for every twentieth neuron in Layers 0-23 of Pythia-410M
"""

import subprocess
import sys
import time
from pathlib import Path

def run_checkpoint_analysis(layer, neuron):
    print(f"\n{'='*60}")
    print(f"Analyzing L{layer}N{neuron}")
    print(f"{'='*60}")
    
    cmd = [
        sys.executable, "checkpoints_demo.py",
        "--series",
        "--model", "pythia-410m",
        "--layer", f"blocks.{layer}.mlp",
        "--neuron", str(neuron),
        "--ckpt_mode", "skip",
        "--start_step", "3000",
        "--skip_steps", "10000",
        "--distance_threshold", "0.8",
        "--peak_activation", "2.8",
        "--max_examples", "100",
        "--batch_size", "4"
    ]
    
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        elapsed = time.time() - start_time
        print(f"✓ L1N{neuron} done in {elapsed:.1f}s")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ L1N{neuron} failed: {e}")
        return False

def main():
    
    # Define layers to analyze
    layers = list(range(24)) # Layers 0, 1,..., 23
    
    # Generate list of neurons
    neurons = list(range(0, 4096, 20)) # 0, 20, 40, ..., 4080 (every 20th)
    
    total_analyses = len(layers) * len(neurons)
    
    print(f"Starting analysis for {total_analyses} neurons over {len(layers)} layers (every 20th neuron)")
    print(f"Layers: {layers}")
    print(f"Neurons per layer: {neurons}")

    # Create results directory if it doesn't exist
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    successful = 0
    failed = 0
    
    for layer in layers:

        for i, neuron in enumerate(neurons, 1):

            if run_checkpoint_analysis(layer, neuron):
                successful += 1
            else:
                failed += 1
            
            # Small delay between runs to avoid overwhelming the system
            time.sleep(1)
    
    print(f"\n{'='*60}")
    print(f"ANALYSIS DONE")
    print(f"{'='*60}")
    print(f"Successful: {successful}/{len(neurons)}")
    print(f"Failed: {failed}/{len(neurons)}")

if __name__ == "__main__":
    main() 