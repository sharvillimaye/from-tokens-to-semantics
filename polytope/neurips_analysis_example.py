#!/usr/bin/env python3
"""
NeurIPS Analysis Example: Multi-Dimensional Superposition Timeline Analysis
Demonstrates the complete three-pronged visualization strategy
"""

import sys
from pathlib import Path
import numpy as np

# Add polytope modules to path
sys.path.append(str(Path(__file__).parent))

from polytope.polytope_metrics import (
    enhanced_run_analysis,
    temporal_evolution_analysis, 
    layer_progression_analysis,
    enhanced_correlation_analysis,
    create_neurips_figures
)
from polytope.checkpoint_analysis import (
    load_semantic_frequency_datasets,
    run_checkpoint_analysis_pipeline
)


def create_demo_analysis():
    """
    Demo analysis using synthetic data that mimics the real analysis workflow
    """
    print("=== NeurIPS Multi-Dimensional Analysis Demo ===")
    
    # Load available datasets to understand structure
    try:
        datasets = load_semantic_frequency_datasets()
        print(f"Available datasets: {list(datasets.keys())}")
        
        if datasets:
            sample_dataset = list(datasets.values())[0]
            print(f"Sample dataset structure: {list(sample_dataset.keys())}")
    except Exception as e:
        print(f"Could not load datasets: {e}")
        datasets = {}
    
    # Create realistic synthetic data for demonstration
    print("\nCreating synthetic demonstration data...")
    np.random.seed(42)
    
    # Simulate realistic training dynamics
    checkpoints = ['500', '1000', '2000', '4000', '6000', '8000', '10000', '15000']
    layers = [2, 4, 6, 8, 10, 12]  # Typical transformer layers
    
    demo_records = []
    
    for checkpoint in checkpoints:
        checkpoint_num = int(checkpoint)
        
        for layer in layers:
            # Realistic training dynamics
            # Early training: high variance, later training: more structured
            training_progress = checkpoint_num / 15000  # 0 to 1
            
            # Layer-dependent patterns
            layer_depth_factor = layer / 12.0  # 0 to 1
            
            # Base activation scale
            base_scale = 0.8 + training_progress * 0.4
            layer_scale = base_scale * (1.0 + layer_depth_factor * 0.5)
            
            n_samples = 40  # Records per layer per checkpoint
            
            for i in range(n_samples):
                # Create realistic activation patterns
                activation_dim = 768  # Typical transformer hidden size
                activation = np.random.randn(activation_dim) * layer_scale
                
                # Add structure that evolves with training
                if training_progress > 0.3:  # After some training
                    # Add correlated structure
                    n_structured = int(activation_dim * 0.2 * training_progress)
                    structured_dims = np.random.choice(activation_dim, n_structured, replace=False)
                    base_pattern = np.random.randn() * 2.0
                    activation[structured_dims] += base_pattern
                
                # Later layers become more specialized
                if layer > 6:
                    specialization_factor = (layer - 6) / 6.0
                    n_specialized = int(activation_dim * 0.1 * specialization_factor)
                    specialized_dims = np.random.choice(activation_dim, n_specialized, replace=False)
                    activation[specialized_dims] *= (1.5 + specialization_factor)
                
                # Frequency patterns (realistic n-gram frequency distribution)
                # High frequency: simple patterns, low frequency: complex patterns
                freq_category = np.random.choice(['low', 'medium', 'high'], p=[0.3, 0.4, 0.3])
                if freq_category == 'high':
                    ngram_freq = np.random.uniform(0.01, 0.1)  
                elif freq_category == 'medium':
                    ngram_freq = np.random.uniform(0.001, 0.01)
                else:  # low
                    ngram_freq = np.random.uniform(0.0001, 0.001)
                
                # Higher frequency n-grams have more consistent activations
                if freq_category == 'high':
                    activation *= 0.8  # More consistent
                elif freq_category == 'low':
                    activation *= 1.3  # More variable
                
                record = {
                    'checkpoint_step': checkpoint,
                    'layer': layer,
                    'activation_vector': activation,
                    'activation_norm': np.linalg.norm(activation),
                    'ngram_frequency': ngram_freq,
                    'frequency_category': freq_category,
                    'sparsity': 1.0 - (np.count_nonzero(activation) / len(activation)),
                    'n_active_neurons': int(np.count_nonzero(activation)),
                    'dataset_name': 'synthetic_demo'
                }
                demo_records.append(record)
    
    print(f"Created {len(demo_records):,} synthetic records")
    print(f"Checkpoints: {checkpoints}")
    print(f"Layers: {layers}")
    
    return demo_records


def run_complete_neurips_analysis(records):
    """
    Run the complete NeurIPS analysis pipeline
    """
    print("\n" + "="*60)
    print("RUNNING COMPLETE NEURIPS ANALYSIS PIPELINE")
    print("="*60)
    
    # Enhanced analysis with all three strategies
    results = enhanced_run_analysis(
        records=records,
        target_layers=[4, 6, 8, 10],  # Focus on middle layers
        checkpoint_selection="adaptive",  # Use intelligent checkpoint selection
        max_checkpoints=12
    )
    
    return results


def demonstrate_individual_analyses(records):
    """
    Demonstrate each analysis component individually
    """
    print("\n" + "="*60)
    print("DEMONSTRATING INDIVIDUAL ANALYSIS COMPONENTS")  
    print("="*60)
    
    # 1. Temporal Evolution Analysis
    print("\n1. TEMPORAL EVOLUTION ANALYSIS")
    print("-" * 40)
    temporal_results = temporal_evolution_analysis(
        records=records,
        target_layer=6,  # Middle layer
        frequency_bins=3
    )
    
    if 'error' not in temporal_results:
        stats = temporal_results.get('statistics', {})
        print(f"Volume trend correlation: {stats.get('volume_trend', 0):.3f}")
        print(f"Vertex trend correlation: {stats.get('vertex_trend', 0):.3f}")
        print(f"Final volume: {stats.get('final_volume', 0):.6f}")
    
    # 2. Layer Progression Analysis  
    print("\n2. LAYER PROGRESSION ANALYSIS")
    print("-" * 40)
    progression_results = layer_progression_analysis(
        records=records,
        target_layers=[4, 6, 8, 10]
    )
    
    if not progression_results['progression_data'].empty:
        df = progression_results['progression_data']
        print(f"Analyzed {len(df)} layer-checkpoint combinations")
        print("Volume by layer (mean):")
        volume_by_layer = df.groupby('layer')['volume'].mean()
        for layer, volume in volume_by_layer.items():
            print(f"  Layer {layer}: {volume:.6f}")
    
    # 3. Enhanced Correlation Analysis
    print("\n3. ENHANCED CORRELATION ANALYSIS")
    print("-" * 40)
    correlation_results = enhanced_correlation_analysis(
        records=records,
        target_layers=[6, 8, 10]
    )
    
    if 'error' not in correlation_results:
        summary = correlation_results['summary']
        print(f"Data points analyzed: {summary['n_data_points']}")
        print(f"Frequency range: {summary['frequency_range'][0]:.4f} - {summary['frequency_range'][1]:.4f}")
        print(f"Training stages: {summary['stages_analyzed']}")
    
    return {
        'temporal': temporal_results,
        'progression': progression_results, 
        'correlation': correlation_results
    }


def analyze_results(results):
    """
    Analyze and summarize the key findings
    """
    print("\n" + "="*60)
    print("KEY FINDINGS SUMMARY")
    print("="*60)
    
    # Overall summary
    if 'analysis_summary' in results:
        summary = results['analysis_summary']
        print(f"\nAnalysis Scale:")
        print(f"  Records: {summary['n_records']:,}")
        print(f"  Checkpoints: {summary['n_checkpoints']}")
        print(f"  Layers: {summary['n_layers']}")
        print(f"  Primary layer: {summary['primary_layer']}")
    
    # Volume evolution findings
    if 'summary_statistics' in results and 'volume_evolution' in results['summary_statistics']:
        vol_stats = results['summary_statistics']['volume_evolution']
        print(f"\nVolume Evolution Findings:")
        print(f"  Trend correlation: {vol_stats['trend_correlation']:.3f}")
        print(f"  Mean volume: {vol_stats['mean']:.6f}")
        print(f"  95% CI: [{vol_stats['ci_lower']:.6f}, {vol_stats['ci_upper']:.6f}]")
        
        if vol_stats['trend_correlation'] > 0.5:
            print("  → FINDING: Strong positive volume growth during training")
        elif vol_stats['trend_correlation'] < -0.5:
            print("  → FINDING: Volume compression during training")
        else:
            print("  → FINDING: Stable volume throughout training")
    
    # Layer complexity findings
    if 'summary_statistics' in results and 'layer_complexity' in results['summary_statistics']:
        complexity_stats = results['summary_statistics']['layer_complexity']
        print(f"\nLayer Complexity Findings:")
        print(f"  Overall mean complexity: {complexity_stats['overall_mean']:.2f}")
        print(f"  Complexity variance: {complexity_stats['complexity_variance']:.2f}")
        
        layer_means = complexity_stats['mean_by_layer']
        if layer_means:
            min_layer = min(layer_means, key=layer_means.get)
            max_layer = max(layer_means, key=layer_means.get)
            print(f"  Lowest complexity: Layer {min_layer} ({layer_means[min_layer]:.2f})")
            print(f"  Highest complexity: Layer {max_layer} ({layer_means[max_layer]:.2f})")
    
    # Figure generation
    if 'figures' in results:
        figures = results['figures']
        print(f"\nGenerated Figures:")
        for fig_name, fig_path in figures.items():
            print(f"  {fig_name}: {fig_path}")
    
    print(f"\nAnalysis complete! Ready for NeurIPS submission.")


def main():
    """
    Main demonstration function
    """
    print("NeurIPS Multi-Dimensional Superposition Analysis")
    print("Polytope Evolution Across Training Checkpoints")
    print("=" * 60)
    
    # Create demonstration data
    demo_records = create_demo_analysis()
    
    # Run individual analyses for detailed examination
    individual_results = demonstrate_individual_analyses(demo_records)
    
    # Run complete pipeline
    complete_results = run_complete_neurips_analysis(demo_records)
    
    # Analyze and summarize findings
    analyze_results(complete_results)
    
    return complete_results


if __name__ == "__main__":
    results = main()