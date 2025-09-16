#!/usr/bin/env python3
"""
Complete polytope analysis pipeline with optimized convex hull computation.
"""

import sys
from pathlib import Path
from polytope.polytope_analysis import PolytopeAnalyzer

def main():
    """Run the complete polytope analysis pipeline."""
    
    # Initialize analyzer with optimized settings
    analyzer = PolytopeAnalyzer(
        random_seed=42,
        cache_dir="cache/polytope_analysis"
    )
    
    # Example checkpoint paths - replace with your actual paths
    checkpoint_paths = [
        "path/to/checkpoint_1000.pkl",
        "path/to/checkpoint_2000.pkl", 
        "path/to/checkpoint_3000.pkl",
        "path/to/checkpoint_4000.pkl"
    ]
    
    print("🚀 Starting polytope analysis pipeline...")
    print(f"📊 Analyzing {len(checkpoint_paths)} checkpoints")
    
    # Step 1: Run multi-checkpoint analysis with optimizations
    print("\n1️⃣ Running multi-checkpoint analysis...")
    results = analyzer.run_multiple_checkpoints(
        checkpoint_paths=checkpoint_paths,
        output_dir="cache/polytope_analysis/results",
        parallel_checkpoints=True,
        parallel_layers=True,
        max_workers=4  # Adjust based on your system
    )
    
    print(f"✅ Completed analysis for {len(results)} checkpoints")
    
    # Step 2: Analyze evolution across checkpoints
    print("\n2️⃣ Analyzing checkpoint evolution...")
    evolution_summary = analyzer.get_checkpoint_evolution_summary(results)
    flattened_results = analyzer.flatten_multi_checkpoint_results(results)
    
    print(f"📈 Evolution analysis complete for {len(evolution_summary['layers'])} layers")
    
    # Step 3: Print summary statistics
    print("\n3️⃣ Summary Statistics:")
    print(f"   • Checkpoints analyzed: {len(evolution_summary['checkpoints'])}")
    print(f"   • Layers analyzed: {len(evolution_summary['layers'])}")
    
    for layer in evolution_summary['layers']:
        layer_data = evolution_summary['evolution_by_layer'][layer]
        print(f"   • Layer {layer}: {layer_data['checkpoints_analyzed']} checkpoints")
        
        # Show participation ratio trends
        if 'high_freq_participation_ratio' in layer_data['metrics_over_time']:
            trend = layer_data['metrics_over_time']['high_freq_participation_ratio']['trend']
            mean_val = layer_data['metrics_over_time']['high_freq_participation_ratio']['mean']
            print(f"     - High freq participation ratio: {mean_val:.3f} ({trend})")
    
    # Step 4: Check for any validation issues
    print("\n4️⃣ Checking subspace validation results...")
    validation_issues = 0
    for checkpoint_step, checkpoint_result in results.items():
        if 'layer_results' in checkpoint_result:
            for layer, layer_result in checkpoint_result['layer_results'].items():
                if 'subspace_analysis' in layer_result:
                    validation = layer_result['subspace_analysis'].get('subspace_validation', {})
                    if not validation.get('hull_computation_recommended', True):
                        validation_issues += 1
                        print(f"   ⚠️  Validation issue: {checkpoint_step} layer {layer}")
    
    if validation_issues == 0:
        print("   ✅ All subspace validations passed")
    else:
        print(f"   ⚠️  {validation_issues} validation issues found")
    
    print("\n🎉 Pipeline complete! Results saved to cache/polytope_analysis/results/")
    
    return results, evolution_summary

if __name__ == "__main__":
    try:
        results, evolution = main()
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        sys.exit(1)
