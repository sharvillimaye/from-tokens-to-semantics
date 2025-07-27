#!/usr/bin/env python3
"""
Example Usage of Refactored Polytope Analysis Pipeline
Demonstrates complete workflow from n-gram dataset creation to polytope analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Import our refactored modules
from unified_polytope_analysis import UnifiedPolytopeAnalyzer
from ngram_dataset import ImprovedNGramDatasetBuilder
from checkpoint_analysis import ImprovedCheckpointAnalyzer
from polytope_metrics import AdvancedPolytopeAnalyzer, get_polytope_metrics


def example_complete_pipeline():
    """
    Example 1: Complete pipeline analysis
    This demonstrates the full pipeline from n-gram generation to polytope analysis
    """
    
    print("🚀 Example 1: Complete Pipeline Analysis")
    print("="*60)
    
    # Initialize the unified analyzer
    analyzer = UnifiedPolytopeAnalyzer(
        working_dir="./example_results",
        cache_activations=True,
        cache_ngrams=True
    )
    
    # Run the complete analysis
    try:
        results = analyzer.run_complete_analysis(
            model_name="EleutherAI/pythia-70m",  # Small model for testing
            checkpoints=["1000", "2000"],  # Just 2 checkpoints for example
            n_gram_size=2,
            pile_samples=500,  # Small sample for testing
            target_layers=[0, 2],  # Specific layers
            frequency_analysis=True
        )
        
        # Generate visualizations
        print("\n📊 Generating visualizations...")
        figures = analyzer.visualize_results(save_plots=True)
        
        # Export results
        print("\n💾 Exporting results...")
        export_paths = analyzer.export_results_for_analysis()
        
        print("\n✅ Complete pipeline finished successfully!")
        print("Results saved in ./example_results/")
        
        return analyzer, results
        
    except Exception as e:
        print(f"❌ Error in complete pipeline: {e}")
        print("This might be due to model availability or computational resources.")
        return None, None


def example_ngram_dataset_only():
    """
    Example 2: N-gram dataset creation only
    This shows how to use just the n-gram dataset builder
    """
    
    print("\n🔍 Example 2: N-gram Dataset Creation")
    print("="*60)
    
    # Initialize the n-gram dataset builder
    builder = ImprovedNGramDatasetBuilder(
        cache_dir="./example_ngram_cache",
        max_samples=1000
    )
    
    try:
        # Build a stratified n-gram dataset
        dataset = builder.build_stratified_ngram_dataset(
            n_gram_size=2,
            pile_samples=800,  # Smaller for example
            min_word_length=2
        )
        
        # Save the dataset
        builder.save_dataset(dataset, "./example_ngram_dataset", format="both")
        
        # Generate analysis report
        report = builder.generate_analysis_report(dataset)
        print("\n📋 Dataset Analysis:")
        print(report[:500] + "..." if len(report) > 500 else report)
        
        print(f"\n✅ N-gram dataset created with {len(dataset['texts'])} samples")
        print("Dataset saved in ./example_ngram_dataset/")
        
        return dataset
        
    except Exception as e:
        print(f"❌ Error in n-gram dataset creation: {e}")
        print("Using fallback synthetic data...")
        
        # Create synthetic dataset for demonstration
        synthetic_dataset = {
            'texts': [
                "Machine learning algorithms are transforming data analysis.",
                "Neural networks can learn complex patterns from data.",
                "Deep learning models achieve remarkable performance.",
                "Artificial intelligence has many practical applications."
            ],
            'ngrams': ["machine learning", "neural networks", "deep learning", "artificial intelligence"],
            'categories': ["medium", "medium", "high", "medium"],
            'local_frequencies': [45, 32, 67, 28],
            'global_frequencies': [12000, 8500, 25000, 9200],
            'positions': [0, 0, 0, 0],
            'char_start': [0, 0, 0, 0],
            'char_end': [16, 15, 13, 20],
            'confidence_scores': [0.95, 0.92, 0.98, 0.90],
            'metadata': [
                {'text_length': 52, 'ngram_word_count': 2},
                {'text_length': 48, 'ngram_word_count': 2},
                {'text_length': 45, 'ngram_word_count': 2},
                {'text_length': 50, 'ngram_word_count': 2}
            ]
        }
        
        print(f"✅ Synthetic dataset created with {len(synthetic_dataset['texts'])} samples")
        return synthetic_dataset


def example_polytope_analysis_only():
    """
    Example 3: Polytope analysis with synthetic activation data
    This shows how to use just the polytope analyzer
    """
    
    print("\n📐 Example 3: Polytope Analysis with Synthetic Data")
    print("="*60)
    
    # Generate synthetic activation data
    np.random.seed(42)
    
    # Simulate different types of activations
    print("🎲 Generating synthetic activation data...")
    
    # High-frequency n-gram: more structured activations
    high_freq_activations = np.random.normal(0, 1, (50, 512))
    high_freq_activations[:, :20] *= 3  # Emphasize first 20 dimensions
    
    # Medium-frequency n-gram: moderately structured
    medium_freq_activations = np.random.normal(0, 1, (50, 512))
    medium_freq_activations[:, :50] *= 2  # Emphasize first 50 dimensions
    
    # Low-frequency n-gram: more distributed activations
    low_freq_activations = np.random.normal(0, 1, (50, 512))
    
    # Initialize polytope analyzer
    analyzer = AdvancedPolytopeAnalyzer(
        pca_components=10,
        min_points_for_hull=4,
        stability_samples=50
    )
    
    try:
        # Analyze each activation type
        print("📊 Analyzing polytopes...")
        
        high_freq_metrics = analyzer.analyze_activation_polytope(high_freq_activations)
        medium_freq_metrics = analyzer.analyze_activation_polytope(medium_freq_activations)
        low_freq_metrics = analyzer.analyze_activation_polytope(low_freq_activations)
        
        # Display results
        print("\n📈 Polytope Analysis Results:")
        print("-" * 40)
        
        metrics_data = [
            ("High Frequency", high_freq_metrics),
            ("Medium Frequency", medium_freq_metrics),
            ("Low Frequency", low_freq_metrics)
        ]
        
        for name, metrics in metrics_data:
            print(f"\n{name} N-gram:")
            print(f"  Volume: {metrics.volume:.6f}")
            print(f"  Complexity: {metrics.complexity_score:.4f}")
            print(f"  Effective Dimension: {metrics.effective_dimension}")
            print(f"  Stability: {metrics.stability_score:.4f}")
            print(f"  Vertices: {metrics.n_vertices}")
        
        # Compare polytopes
        print("\n🔄 Comparing polytopes...")
        comparison = analyzer.compare_polytopes(
            [high_freq_metrics, medium_freq_metrics, low_freq_metrics],
            ['High-freq', 'Medium-freq', 'Low-freq']
        )
        
        print(f"\nVolume ratios (relative to high-freq):")
        for i, label in enumerate(comparison['labels']):
            print(f"  {label}: {comparison['volume_ratios'][i]:.4f}")
        
        # Visualize evolution across synthetic checkpoints
        print("\n📊 Simulating polytope evolution...")
        checkpoint_data = {}
        
        for i, checkpoint in enumerate(["1000", "2000", "3000"]):
            # Simulate evolution: complexity decreases over time
            noise_factor = 1.0 - (i * 0.2)  # Decreasing noise
            evolved_activations = high_freq_activations + np.random.normal(0, noise_factor, high_freq_activations.shape)
            checkpoint_data[checkpoint] = evolved_activations
        
        evolution_df = analyzer.analyze_polytope_evolution(checkpoint_data)
        print(f"Evolution analysis complete: {len(evolution_df)} checkpoints")
        
        # Create visualization
        if len(evolution_df) > 1:
            fig = analyzer.visualize_polytope_evolution(evolution_df)
            plt.savefig('./example_polytope_evolution.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("Evolution plot saved as './example_polytope_evolution.png'")
        
        print("\n✅ Polytope analysis completed successfully!")
        
        return {
            'high_freq': high_freq_metrics,
            'medium_freq': medium_freq_metrics,
            'low_freq': low_freq_metrics,
            'comparison': comparison,
            'evolution': evolution_df
        }
        
    except Exception as e:
        print(f"❌ Error in polytope analysis: {e}")
        return None


def example_custom_analysis():
    """
    Example 4: Custom analysis combining components
    This shows how to combine individual components for custom analysis
    """
    
    print("\n🔧 Example 4: Custom Component Integration")
    print("="*60)
    
    try:
        # Step 1: Create custom n-gram dataset
        print("📝 Creating custom n-gram dataset...")
        
        custom_dataset = {
            'texts': [
                "The cat sat on the mat",
                "Machine learning is transforming AI",
                "Deep neural networks are powerful",
                "Natural language processing advances",
                "Computer vision detects objects"
            ],
            'ngrams': ["the cat", "machine learning", "neural networks", "language processing", "computer vision"],
            'categories': ["high", "medium", "medium", "low", "low"],
            'local_frequencies': [150, 45, 32, 18, 22],
            'char_start': [0, 0, 5, 8, 0],
            'char_end': [7, 16, 20, 28, 15]
        }
        
        print(f"✅ Custom dataset created with {len(custom_dataset['texts'])} samples")
        
        # Step 2: Simulate activation extraction
        print("⚡ Simulating activation extraction...")
        
        np.random.seed(123)
        simulated_activations = {}
        
        for i, (ngram, category) in enumerate(zip(custom_dataset['ngrams'], custom_dataset['categories'])):
            # Different activation patterns based on category
            if category == "high":
                # High frequency: more concentrated activations
                activations = np.random.normal(0, 0.5, (20, 128))
                activations[:, :10] *= 4
            elif category == "medium":
                # Medium frequency: moderately concentrated
                activations = np.random.normal(0, 0.8, (15, 128))
                activations[:, :25] *= 2
            else:
                # Low frequency: distributed activations
                activations = np.random.normal(0, 1, (10, 128))
            
            simulated_activations[ngram] = activations
        
        print(f"✅ Simulated activations for {len(simulated_activations)} n-grams")
        
        # Step 3: Analyze polytopes for each n-gram
        print("📐 Analyzing polytopes for each n-gram...")
        
        analyzer = AdvancedPolytopeAnalyzer(pca_components=8)
        ngram_polytope_results = {}
        
        for ngram, activations in simulated_activations.items():
            metrics = analyzer.analyze_activation_polytope(activations)
            ngram_polytope_results[ngram] = metrics
            
            category = custom_dataset['categories'][custom_dataset['ngrams'].index(ngram)]
            print(f"  {ngram} ({category}): volume={metrics.volume:.6f}, complexity={metrics.complexity_score:.4f}")
        
        # Step 4: Create summary analysis
        print("\n📊 Summary Analysis:")
        print("-" * 30)
        
        # Group by category
        category_metrics = {}
        for ngram, metrics in ngram_polytope_results.items():
            category = custom_dataset['categories'][custom_dataset['ngrams'].index(ngram)]
            if category not in category_metrics:
                category_metrics[category] = []
            category_metrics[category].append(metrics.volume)
        
        for category, volumes in category_metrics.items():
            mean_volume = np.mean(volumes)
            print(f"  {category} frequency: mean volume = {mean_volume:.6f}")
        
        # Step 5: Export results
        print("\n💾 Exporting custom analysis results...")
        
        results_df = pd.DataFrame([
            {
                'ngram': ngram,
                'category': custom_dataset['categories'][i],
                'local_frequency': custom_dataset['local_frequencies'][i],
                'volume': metrics.volume,
                'complexity': metrics.complexity_score,
                'effective_dimension': metrics.effective_dimension,
                'stability': metrics.stability_score
            }
            for i, (ngram, metrics) in enumerate(ngram_polytope_results.items())
        ])
        
        results_df.to_csv('./custom_analysis_results.csv', index=False)
        print("Results saved to './custom_analysis_results.csv'")
        
        print("\n✅ Custom analysis completed successfully!")
        
        return {
            'dataset': custom_dataset,
            'activations': simulated_activations,
            'polytope_results': ngram_polytope_results,
            'summary_df': results_df
        }
        
    except Exception as e:
        print(f"❌ Error in custom analysis: {e}")
        return None


def main():
    """
    Run all examples to demonstrate the refactored polytope analysis pipeline
    """
    
    print("🎯 Polytope Analysis Pipeline Examples")
    print("="*70)
    print("This script demonstrates the refactored polytope analysis components.")
    print("Each example shows different ways to use the pipeline.\n")
    
    # Example 1: Complete pipeline (may require actual models and data)
    print("Note: Example 1 requires model download and may take time...")
    
    # Example 2: N-gram dataset creation
    dataset = example_ngram_dataset_only()
    
    # Example 3: Polytope analysis only
    polytope_results = example_polytope_analysis_only()
    
    # Example 4: Custom component integration
    custom_results = example_custom_analysis()
    
    # Summary
    print("\n🎉 All Examples Completed!")
    print("="*70)
    print("📁 Generated files:")
    print("  - ./example_ngram_dataset/ (n-gram dataset)")
    print("  - ./example_polytope_evolution.png (evolution plot)")
    print("  - ./custom_analysis_results.csv (custom analysis)")
    print("\n📖 Check the code for detailed explanations of each component.")
    print("💡 Modify the parameters to experiment with different configurations.")
    
    return {
        'ngram_dataset': dataset,
        'polytope_results': polytope_results,
        'custom_results': custom_results
    }


if __name__ == "__main__":
    results = main() 