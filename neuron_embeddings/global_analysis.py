#!/usr/bin/env python
"""
Script to run checkpoint analysis for every twentieth neuron in Layers 0-5 of Pythia-70M
(or Layers 0-n of another model - you'll have to change lines 47 and 50 to reflect this)
Now uses individual max activations scraped from Neuroscope for each neuron.
"""

import subprocess
import sys
import time
import json
from pathlib import Path

def load_max_activations(model_name="pythia-70m"):
    """Load the scraped peak activations from the JSON file."""
    activations_file = Path("results") / f"{model_name}_max_activations.json"
    
    if not activations_file.exists():
        print("First, run the Neuroscope scraping script: python scrape_neuroscope_activations.py")
        return None
    
    try:
        with open(activations_file, 'r') as f:
            data = json.load(f)
        
        print(f"Loaded peak activations for {data['total_neurons']} neurons")
        return data['activations']
    
    except Exception as e:
        print(f"Error loading peak activations: {e}")
        return None

def run_checkpoint_analysis(layer, neuron, max_activations):
    print(f"\n{'='*60}")
    print(f"Analyzing L{layer}N{neuron}")
    print(f"{'='*60}")

    # Get the max activation for this specific neuron
    neuron_key = f"L{layer}N{neuron}"
    if neuron_key in max_activations:
        peak_activation = max_activations[neuron_key]
        print(f"Using max activation: {peak_activation}")
    else:
        print(f"No max activation found for {neuron_key}, using default 2.8")
        peak_activation = 2.8

    cmd = [
        sys.executable, "checkpoints_demo.py",
        "--series",
        "--model", "pythia-70m",
        "--layer", f"blocks.{layer}.mlp",
        "--neuron", str(neuron),
        "--ckpt_mode", "skip",
        "--start_step", "3000",
        "--skip_steps", "10000",
        "--distance_threshold", "0.7",
        "--peak_activation", str(peak_activation),
        "--max_examples", "100",
        "--batch_size", "4"
    ]
    
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        elapsed = time.time() - start_time
        print(f"✓ L{layer}N{neuron} done in {elapsed:.1f}s")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ L{layer}N{neuron} failed: {e}")
        return False

def main():
    
    # Load max activations
    max_activations = load_max_activations()
    if max_activations is None:
        print("Can't proceed w/o peak activation data")
        return
    
    # Define layers to analyze
    layers = list(range(6)) # Layers 0, 1,..., 5
    
    # Generate list of neurons
    neurons = list(range(0, 2048, 20)) # 0, 20, 40, ..., 2040 (every 20th)
    
    total_analyses = len(layers) * len(neurons)
    
    print(f"Starting analysis for {total_analyses} neurons over {len(layers)} layers (every 20th neuron)")
    print(f"Layers: {layers}")
    print(f"Neurons per layer: {neurons}")
    print(f"Using individual max activations from Neuroscope")

    # Create results directory if it doesn't exist
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    successful = 0
    failed = 0
    
    for layer in layers:

        for i, neuron in enumerate(neurons, 1):

            if run_checkpoint_analysis(layer, neuron, max_activations):
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