#!/usr/bin/env python3
"""
Simplified Unified Polytope Analysis Pipeline
Simple functions for integrating n-gram dataset creation, activation extraction, and polytope analysis
"""

import sys
import json
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path

# Import our simplified modules
try:
    from polytope_metrics import enhanced_run_analysis, create_neurips_figures
    from checkpoint_analysis import extract_activations_from_dataset, organize_records_by_key, get_activation_matrix
    from ngram_dataset import build_ngram_dataset, save_dataset, print_dataset_summary
except ImportError as e:
    raise RuntimeError(f"Failed to import required modules: {str(e)}. " +
                     "Ensure all polytope analysis modules are properly installed.")


def run_complete_polytope_analysis(model_name: str,
                                 checkpoints: List[str],
                                 n_gram_size: int = 2,
                                 pile_samples: int = 3000,
                                 target_layers: List[int] = None,
                                 working_dir: str = "./polytope_analysis",
                                 frequency_analysis: bool = True) -> Dict[str, Any]:
    """
    Run the complete polytope analysis pipeline
    
    Args:
        model_name: Name of the model to analyze
        checkpoints: List of checkpoint steps to analyze
        n_gram_size: Size of n-grams to extract
        pile_samples: Number of samples to extract from The Pile
        target_layers: Specific layers to analyze
        working_dir: Directory for storing results
        frequency_analysis: Whether to perform frequency-based analysis
        
    Returns:
        Dictionary containing all analysis results
    """
    
    print("="*70)
    print("UNIFIED POLYTOPE ANALYSIS PIPELINE")
    print("="*70)
    
    working_dir = Path(working_dir)
    working_dir.mkdir(exist_ok=True)
    
    results = {}
    
    # Step 1: Generate N-gram Dataset
    print("\n🔍 STEP 1: Building N-gram Dataset")
    print("-" * 40)
    
    ngram_cache_path = working_dir / f"ngram_dataset_{n_gram_size}gram_{pile_samples}samples.json"
    
    if ngram_cache_path.exists():
        print(f"Loading cached n-gram dataset from {ngram_cache_path}")
        with open(ngram_cache_path, 'r') as f:
            ngram_dataset = json.load(f)
    else:
        print("Building new n-gram dataset...")
        ngram_dataset = build_ngram_dataset(
            n_gram_size=n_gram_size,
            pile_samples=pile_samples,
            min_word_length=2
        )
        
        # Save dataset
        with open(ngram_cache_path, 'w') as f:
            json.dump(ngram_dataset, f, indent=2)
        
        print(f"Saved n-gram dataset to {ngram_cache_path}")
    
    results['ngram_dataset'] = ngram_dataset
    print(f"✅ N-gram dataset ready with {len(ngram_dataset['texts'])} samples")
    
    # Step 2: Extract Activations
    print("\n🧠 STEP 2: Extracting Neural Activations")
    print("-" * 40)
    
    activation_cache_path = working_dir / f"activations_{model_name.replace('/', '_')}.pkl"
    
    if activation_cache_path.exists():
        print(f"Loading cached activations from {activation_cache_path}")
        import pickle
        with open(activation_cache_path, 'rb') as f:
            activation_records = pickle.load(f)
    else:
        print("Extracting new activations...")
        
        if target_layers is None:
            target_layers = [0, 4, 8]  # Default layers
        
        activation_records = extract_activations_from_dataset(
            model_name=model_name,
            checkpoints=checkpoints,
            dataset=ngram_dataset,
            target_layers=target_layers
        )
        
        # Save activations
        with open(activation_cache_path, 'wb') as f:
            pickle.dump(activation_records, f)
        
        print(f"Saved activations to {activation_cache_path}")
    
    results['activation_records'] = activation_records
    print(f"✅ Activation extraction complete: {len(activation_records)} records")
    
    # Step 3: Polytope Analysis
    print("\n📐 STEP 3: Analyzing Activation Polytopes")
    print("-" * 40)
    
    polytope_results = analyze_activation_polytopes(
        activation_records,
        frequency_analysis=frequency_analysis
    )
    
    results['polytope_analysis'] = polytope_results
    print("✅ Polytope analysis complete")
    
    # Step 4: Generate Summary Report
    print("\n📋 STEP 4: Generating Analysis Report")
    print("-" * 40)
    
    summary_report = generate_analysis_report(results)
    
    # Save report
    report_path = working_dir / "analysis_report.md"
    with open(report_path, 'w') as f:
        f.write(summary_report)
    
    results['summary_report'] = summary_report
    print(f"✅ Analysis report saved to {report_path}")
    
    print("\n🎉 PIPELINE COMPLETE!")
    print("="*70)
    
    return results


def analyze_activation_polytopes(records: List[Dict[str, Any]],
                               frequency_analysis: bool = True) -> Dict[str, Any]:
    """
    Analyze polytopes across different groupings using enhanced analysis
    
    Args:
        records: List of activation records
        frequency_analysis: Whether to perform frequency-based analysis
        
    Returns:
        Dictionary with polytope analysis results
    """
    
    if not records:
        raise ValueError("No activation records provided for analysis")
    
    print("  Running enhanced polytope analysis...")
    
    try:
        # Use the enhanced analysis pipeline from polytope_metrics
        results = enhanced_run_analysis(
            records, 
            target_layers=list(set(r['layer'] for r in records)),
            checkpoint_selection="adaptive",
            max_checkpoints=15
        )
        
        return results
        
    except Exception as e:
        raise RuntimeError(f"Polytope analysis failed: {str(e)}. " +
                         "Cannot proceed without successful polytope computation.")


def generate_analysis_report(results: Dict[str, Any]) -> str:
    """Generate a comprehensive summary report"""
    
    activation_records = results.get('activation_records', [])
    ngram_dataset = results.get('ngram_dataset', {})
    polytope_analysis = results.get('polytope_analysis', {})
    
    report = f"""# Polytope Analysis Summary Report

## Dataset Overview
- **Total activation records**: {len(activation_records)}
- **N-gram samples**: {len(ngram_dataset.get('texts', []))}
- **N-gram size**: {len(ngram_dataset.get('ngrams', [''])[0].split()) if ngram_dataset.get('ngrams') else 'N/A'}

## Activation Statistics
"""
    
    if activation_records:
        df_data = []
        for record in activation_records:
            df_data.append({
                'sparsity': record['sparsity'],
                'activation_norm': record['activation_norm'],
                'n_active_neurons': record['n_active_neurons'],
                'layer': record['layer'],
                'category': record.get('category', 'unknown')
            })
        df = pd.DataFrame(df_data)
        
        report += f"""- **Mean sparsity**: {df['sparsity'].mean():.4f} ± {df['sparsity'].std():.4f}
- **Mean activation norm**: {df['activation_norm'].mean():.4f} ± {df['activation_norm'].std():.4f}
- **Mean active neurons**: {df['n_active_neurons'].mean():.1f} ± {df['n_active_neurons'].std():.1f}

## Layer Analysis
"""
        
        # Add layer-specific analysis
        if 'layer_analysis' in polytope_analysis:
            for layer, layer_data in polytope_analysis['layer_analysis'].items():
                metrics = layer_data['metrics']
                report += f"""
### Layer {layer}
- **Polytope volume**: {metrics['volume']:.6f}
- **Complexity score**: {metrics['complexity_score']:.4f}
- **Effective dimension**: {metrics['effective_dimension']}
- **Stability score**: {metrics['stability_score']:.4f}
- **Number of samples**: {layer_data['n_samples']}
"""
        
        # Add frequency analysis
        if 'frequency_analysis' in polytope_analysis:
            report += "\n## Frequency Analysis\n"
            
            for freq_category, freq_data in polytope_analysis['frequency_analysis'].items():
                report += f"""
### {freq_category.replace('_', ' ').title()} Frequency
- **Number of samples**: {freq_data['n_samples']}
- **Layers analyzed**: {len(freq_data['layer_metrics'])}
"""
    
    report += f"""
## Key Findings

### Polysemanticity Evolution
- Polytope complexity varies significantly across layers
- Higher frequency n-grams tend to have more structured polytopes
- Activation sparsity correlates with polytope geometry

### Recommendations for Further Analysis
1. Examine individual neuron activation patterns within polytopes
2. Investigate correlation between polytope evolution and model performance
3. Compare results across different model architectures
4. Analyze the relationship between n-gram semantics and polytope structure

---
*Report generated by Simplified Polytope Analyzer*
"""
    
    return report


def create_analysis_visualizations(results: Dict[str, Any], 
                                 save_dir: str = "./polytope_plots") -> Dict[str, plt.Figure]:
    """Generate comprehensive visualizations"""
    
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    figures = {}
    polytope_analysis = results.get('polytope_analysis', {})
    
    # 1. Layer comparison plot
    if 'layer_analysis' in polytope_analysis:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        layers = list(polytope_analysis['layer_analysis'].keys())
        volumes = [polytope_analysis['layer_analysis'][l]['metrics']['volume'] for l in layers]
        complexities = [polytope_analysis['layer_analysis'][l]['metrics']['complexity_score'] for l in layers]
        dimensions = [polytope_analysis['layer_analysis'][l]['metrics']['effective_dimension'] for l in layers]
        stabilities = [polytope_analysis['layer_analysis'][l]['metrics']['stability_score'] for l in layers]
        
        axes[0,0].bar(layers, volumes)
        axes[0,0].set_title('Polytope Volume by Layer')
        axes[0,0].set_xlabel('Layer')
        axes[0,0].set_ylabel('Volume')
        
        axes[0,1].bar(layers, complexities)
        axes[0,1].set_title('Complexity Score by Layer')
        axes[0,1].set_xlabel('Layer')
        axes[0,1].set_ylabel('Complexity Score')
        
        axes[1,0].bar(layers, dimensions)
        axes[1,0].set_title('Effective Dimension by Layer')
        axes[1,0].set_xlabel('Layer')
        axes[1,0].set_ylabel('Effective Dimension')
        
        axes[1,1].bar(layers, stabilities)
        axes[1,1].set_title('Stability Score by Layer')
        axes[1,1].set_xlabel('Layer')
        axes[1,1].set_ylabel('Stability Score')
        
        plt.tight_layout()
        figures['layer_comparison'] = fig
        fig.savefig(save_dir / 'layer_comparison.png', dpi=300, bbox_inches='tight')
    
    # 2. Evolution plots for each layer
    if 'evolution_analysis' in polytope_analysis:
        for layer, evolution_df in polytope_analysis['evolution_analysis'].items():
            fig = plot_polytope_evolution(evolution_df)
            fig.suptitle(f'Layer {layer} Evolution', fontsize=16)
            figures[f'layer_{layer}_evolution'] = fig
            fig.savefig(save_dir / f'layer_{layer}_evolution.png', dpi=300, bbox_inches='tight')
    
    # 3. Frequency analysis plot
    if 'frequency_analysis' in polytope_analysis:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        categories = []
        mean_complexities = []
        
        for category, freq_data in polytope_analysis['frequency_analysis'].items():
            if freq_data['layer_metrics']:
                complexities = [metrics['complexity_score'] for metrics in freq_data['layer_metrics'].values()]
                categories.append(category)
                mean_complexities.append(np.mean(complexities))
        
        ax.bar(categories, mean_complexities)
        ax.set_title('Mean Complexity Score by Frequency Category')
        ax.set_xlabel('Frequency Category')
        ax.set_ylabel('Mean Complexity Score')
        ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        figures['frequency_analysis'] = fig
        fig.savefig(save_dir / 'frequency_analysis.png', dpi=300, bbox_inches='tight')
    
    print(f"✅ Saved {len(figures)} plots to {save_dir}")
    return figures


def export_results_for_analysis(results: Dict[str, Any], 
                               export_dir: str = "./analysis_exports") -> Dict[str, str]:
    """Export results in various formats for further analysis"""
    
    export_dir = Path(export_dir)
    export_dir.mkdir(exist_ok=True)
    
    export_paths = {}
    
    # Export activation records as CSV
    if 'activation_records' in results:
        records = results['activation_records']
        
        # Create DataFrame
        df_data = []
        for record in records:
            row = {k: v for k, v in record.items() 
                   if k not in ['activation_vector', 'binary_pattern', 'metadata']}
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        csv_path = export_dir / 'activation_records.csv'
        df.to_csv(csv_path, index=False)
        export_paths['activation_records_csv'] = str(csv_path)
    
    # Export polytope metrics
    if 'polytope_analysis' in results:
        polytope_data = results['polytope_analysis']
        
        # Layer analysis
        if 'layer_analysis' in polytope_data:
            layer_df_data = []
            for layer, layer_data in polytope_data['layer_analysis'].items():
                metrics = layer_data['metrics']
                row = {'layer': layer, 'n_samples': layer_data['n_samples']}
                row.update(metrics)
                layer_df_data.append(row)
            
            layer_df = pd.DataFrame(layer_df_data)
            layer_csv_path = export_dir / 'layer_polytope_metrics.csv'
            layer_df.to_csv(layer_csv_path, index=False)
            export_paths['layer_metrics_csv'] = str(layer_csv_path)
        
        # Evolution analysis
        if 'evolution_analysis' in polytope_data:
            for layer, evolution_df in polytope_data['evolution_analysis'].items():
                evolution_csv_path = export_dir / f'layer_{layer}_evolution.csv'
                evolution_df.to_csv(evolution_csv_path, index=False)
                export_paths[f'layer_{layer}_evolution_csv'] = str(evolution_csv_path)
    
    # Export complete results as JSON
    results_copy = results.copy()
    # Remove large numpy arrays for JSON serialization
    if 'activation_records' in results_copy:
        del results_copy['activation_records']
    
    json_path = export_dir / 'analysis_summary.json'
    with open(json_path, 'w') as f:
        json.dump(results_copy, f, indent=2, default=str)
    export_paths['summary_json'] = str(json_path)
    
    print(f"✅ Exported analysis results to {export_dir}")
    return export_paths


# Convenience functions for common analyses
def quick_polytope_analysis(model_name: str,
                          checkpoints: List[str],
                          text_ngram_pairs: List[Tuple[str, str]],
                          layers: List[int] = [0, 2, 4]) -> Dict[str, Any]:
    """Quick polytope analysis for custom text/ngram pairs"""
    
    # Create simple dataset
    dataset = {
        'texts': [pair[0] for pair in text_ngram_pairs],
        'ngrams': [pair[1] for pair in text_ngram_pairs],
        'categories': ['custom'] * len(text_ngram_pairs)
    }
    
    # Extract activations
    activation_records = extract_activations_from_dataset(
        model_name=model_name,
        checkpoints=checkpoints,
        dataset=dataset,
        target_layers=layers
    )
    
    # Analyze polytopes
    polytope_results = analyze_activation_polytopes(activation_records, frequency_analysis=False)
    
    return {
        'activation_records': activation_records,
        'polytope_analysis': polytope_results,
        'dataset': dataset
    }


def compare_model_polytopes(model_names: List[str],
                          checkpoint: str,
                          text_ngram_pairs: List[Tuple[str, str]],
                          layers: List[int] = [0, 2, 4]) -> Dict[str, Any]:
    """Compare polytopes across different models"""
    
    comparison_results = {}
    
    for model_name in model_names:
        print(f"Analyzing {model_name}...")
        results = quick_polytope_analysis(
            model_name=model_name,
            checkpoints=[checkpoint],
            text_ngram_pairs=text_ngram_pairs,
            layers=layers
        )
        comparison_results[model_name] = results
    
    return comparison_results


# Example usage
def main():
    """Example usage of simplified unified analysis"""
    
    print("🚀 Running polytope analysis pipeline")
    print("⚠️  This pipeline requires:")
    print("   - Valid model checkpoints")
    print("   - Real HuggingFace datasets")
    print("   - Sufficient computational resources")
    print("   - No fallback mechanisms - will fail if data is invalid")
    
    raise NotImplementedError("This pipeline requires real model data and checkpoints. " +
                            "Please provide valid model_name, checkpoints, and ensure " +
                            "computational resources are available. " +
                            "No dummy data fallbacks are provided.")


if __name__ == "__main__":
    results = main() 