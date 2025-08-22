#!/usr/bin/env python
"""
Script to scrape max activations from Neuroscope for Pythia models.
Scrapes URLs like https://neuroscope.io/pythia-70m/0/0.html
"""

import requests # type: ignore
import re
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple
import argparse

def get_max_activation(url: str) -> float:
    """
    Scrape the max activation value from a Neuroscope URL.
    
    Args:
        url: The Neuroscope URL to scrape
        
    Returns:
        The max activation value as a float, or None if not found
    """
    try:
        # Add a small delay to be respectful to the server
        time.sleep(0.1)
        
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        content = response.text
        
        # Look for the pattern "Max Act: " followed by a number in bold tags
        # The pattern appears multiple times, we want the first occurrence
        match = re.search(r'Max Act:\s*<b>([0-9.-]+)</b>', content)
        
        if match:
            return float(match.group(1))
        else:
            print(f"⚠️  No max activation found in {url}")
            return None
            
    except requests.RequestException as e:
        print(f"❌ Error fetching {url}: {e}")
        return None
    except ValueError as e:
        print(f"❌ Error parsing max activation from {url}: {e}")
        return None

def scrape_model_activations(model_name: str, num_layers: int, neurons_per_layer: int) -> Dict[str, float]:
    """
    Scrape max activations for all neurons in a model.
    
    Args:
        model_name: The model name (e.g., 'pythia-70m', 'pythia-160m')
        num_layers: Number of layers in the model
        neurons_per_layer: Number of neurons per layer
        
    Returns:
        Dictionary mapping neuron identifiers to max activations
    """
    activations = {}
    base_url = f"https://neuroscope.io/{model_name}"
    
    print(f"🔍 Scraping max activations for {model_name}")
    print(f"   Layers: 0-{num_layers-1}, Neurons per layer: 0-{neurons_per_layer-1}")
    
    total_neurons = num_layers * neurons_per_layer
    processed = 0
    
    for layer in range(num_layers):
        for neuron in range(neurons_per_layer):
            url = f"{base_url}/{layer}/{neuron}.html"
            neuron_id = f"L{layer}N{neuron}"
            
            print(f"   Processing {neuron_id} ({processed + 1}/{total_neurons})...")
            
            max_act = get_max_activation(url)
            if max_act is not None:
                activations[neuron_id] = max_act
                print(f"     ✅ Max activation: {max_act}")
            else:
                print(f"     ❌ Failed to get max activation")
            
            processed += 1
    
    print(f"\n✅ Completed scraping {model_name}")
    print(f"   Successfully scraped: {len(activations)}/{total_neurons} neurons")
    
    return activations

def save_activations(activations: Dict[str, float], model_name: str, output_dir: str = "results"):
    """
    Save the scraped activations to a JSON file.
    
    Args:
        activations: Dictionary of neuron activations
        model_name: The model name
        output_dir: Directory to save the results
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    filename = f"{model_name}_max_activations.json"
    filepath = output_path / filename
    
    # Convert to a more structured format
    structured_data = {
        "model": model_name,
        "total_neurons": len(activations),
        "activations": activations,
        "statistics": {
            "mean": sum(activations.values()) / len(activations) if activations else 0,
            "max": max(activations.values()) if activations else 0,
            "min": min(activations.values()) if activations else 0,
        }
    }
    
    with open(filepath, 'w') as f:
        json.dump(structured_data, f, indent=2)
    
    print(f"💾 Saved activations to {filepath}")
    
    # Also save as a simple list
    list_filename = f"{model_name}_max_activations_list.json"
    list_filepath = output_path / list_filename
    
    # Extract just the activation values in order
    activation_list = []
    for neuron_id in sorted(activations.keys()):
        activation_list.append(activations[neuron_id])
    
    with open(list_filepath, 'w') as f:
        json.dump(activation_list, f, indent=2)
    
    print(f"💾 Saved activation list to {list_filepath}")

def main():
    parser = argparse.ArgumentParser(description="Scrape max activations from Neuroscope")
    parser.add_argument("--model", type=str, default="pythia-70m", 
                       help="Model name (e.g., pythia-70m, pythia-160m)")
    parser.add_argument("--layers", type=int, default=6, 
                       help="Number of layers in the model")
    parser.add_argument("--neurons-per-layer", type=int, default=1024, 
                       help="Number of neurons per layer")
    parser.add_argument("--output-dir", type=str, default="results", 
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    print("🚀 Starting Neuroscope activation scraper")
    print(f"   Model: {args.model}")
    print(f"   Layers: {args.layers}")
    print(f"   Neurons per layer: {args.neurons_per_layer}")
    print(f"   Total neurons to scrape: {args.layers * args.neurons_per_layer}")
    print()
    
    # Scrape the activations
    activations = scrape_model_activations(
        args.model, 
        args.layers, 
        args.neurons_per_layer
    )
    
    # Save the results
    save_activations(activations, args.model, args.output_dir)
    
    print(f"\n🎉 Scraping completed!")
    print(f"   Model: {args.model}")
    print(f"   Neurons scraped: {len(activations)}")
    if activations:
        print(f"   Average max activation: {sum(activations.values()) / len(activations):.4f}")
        print(f"   Max activation: {max(activations.values()):.4f}")
        print(f"   Min activation: {min(activations.values()):.4f}")

if __name__ == "__main__":
    main()
