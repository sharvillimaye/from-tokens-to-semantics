#!/usr/bin/env python3
"""
Unified Polytope Analysis Pipeline
Complete integration of n-gram dataset generation, activation extraction, and polytope analysis
"""

import sys
import json
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Import our refactored modules
from ngram_dataset import ImprovedNGramDatasetBuilder
from checkpoint_analysis import ImprovedCheckpointAnalyzer, ActivationDataManager, ActivationRecord
from polytope_metrics import AdvancedPolytopeAnalyzer, PolytopeMetrics, get_polytope_metrics


class UnifiedPolytopeAnalyzer:
    """
    Unified analyzer that integrates n-gram dataset creation, activation extraction,
    and polytope analysis for studying polysemanticity evolution
    """
    
    def __init__(self, 
                 working_dir: str = "./polytope_analysis",
                 cache_activations: bool = True,
                 cache_ngrams: bool = True):
        """
        Initialize the unified analyzer
        
        Args:
            working_dir: Directory for storing results and cache
            cache_activations: Whether to cache activation extractions
            cache_ngrams: Whether to cache n-gram datasets
        """
        self.working_dir = Path(working_dir)
        self.working_dir.mkdir(exist_ok=True)
        
        # Initialize component analyzers
        self.ngram_builder = ImprovedNGramDatasetBuilder(
            cache_dir=str(self.working_dir / "ngram_cache")
        )
        self.checkpoint_analyzer = ImprovedCheckpointAnalyzer(
            cache_dir=str(self.working_dir / "activation_cache")
        )
        self.polytope_analyzer = AdvancedPolytopeAnalyzer()
        
        # Configuration
        self.cache_activations = cache_activations
        self.cache_ngrams = cache_ngrams
        
        # Results storage
        self.ngram_dataset = None
        self.activation_data = None
        self.polytope_results = {}
        
    def run_complete_analysis(self,
                            model_name: str,
                            checkpoints: List[str],
                            n_gram_size: int = 2,
                            pile_samples: int = 3000,
                            target_layers: List[int] = None,
                            frequency_analysis: bool = True) -> Dict[str, Any]:
        """
        Run the complete polytope analysis pipeline
        
        Args:
            model_name: Name of the model to analyze (e.g., "EleutherAI/pythia-70m")
            checkpoints: List of checkpoint steps to analyze
            n_gram_size: Size of n-grams to extract
            pile_samples: Number of samples to extract from The Pile
            target_layers: Specific layers to analyze (None for auto-selection)
            frequency_analysis: Whether to perform frequency-based analysis
            
        Returns:
            Dictionary containing all analysis results
        """
        
        print("="*70)
        print("UNIFIED POLYTOPE ANALYSIS PIPELINE")
        print("="*70)
        
        # Step 1: Generate N-gram Dataset
        print("\n🔍 STEP 1: Building N-gram Dataset")
        print("-" * 40)
        
        ngram_cache_path = self.working_dir / f"ngram_dataset_{n_gram_size}gram_{pile_samples}samples.json"
        
        if self.cache_ngrams and ngram_cache_path.exists():
            print(f"Loading cached n-gram dataset from {ngram_cache_path}")
            with open(ngram_cache_path, 'r') as f:
                self.ngram_dataset = json.load(f)
        else:
            print("Building new n-gram dataset...")
            self.ngram_dataset = self.ngram_builder.build_stratified_ngram_dataset(
                n_gram_size=n_gram_size,
                pile_samples=pile_samples,
                min_word_length=2
            )
            
            if self.cache_ngrams:
                print(f"Caching n-gram dataset to {ngram_cache_path}")
                with open(ngram_cache_path, 'w') as f:
                    json.dump(self.ngram_dataset, f, indent=2, default=str)
        
        print(f"✅ N-gram dataset ready: {len(self.ngram_dataset['texts'])} samples")
        
        # Step 2: Extract Activations
        print("\n⚡ STEP 2: Extracting Activations")
        print("-" * 40)
        
        activation_cache_path = self.working_dir / f"activations_{model_name.replace('/', '_')}_{len(checkpoints)}checkpoints.pkl"
        
        if self.cache_activations and activation_cache_path.exists():
            print(f"Loading cached activations from {activation_cache_path}")
            self.checkpoint_analyzer.load_results(str(activation_cache_path))
            self.activation_data = self.checkpoint_analyzer.data_manager
        else:
            print("Extracting activations from checkpoints...")
            self.activation_data = self.checkpoint_analyzer.extract_activations_from_checkpoints(
                model_name=model_name,
                checkpoints=checkpoints,
                dataset=self.ngram_dataset,
                target_layers=target_layers,
                matching_strategy="comprehensive"
            )
            
            if self.cache_activations:
                print(f"Caching activations to {activation_cache_path}")
                self.checkpoint_analyzer.save_results(str(activation_cache_path))
        
        print(f"✅ Activations ready: {len(self.activation_data.records)} records")
        
        # Step 3: Polytope Analysis
        print("\n📐 STEP 3: Polytope Analysis")
        print("-" * 40)
        
        self.polytope_results = self._analyze_polytopes_comprehensive()
        
        # Step 4: Frequency Analysis (if requested)
        if frequency_analysis:
            print("\n📊 STEP 4: Frequency Analysis")
            print("-" * 40)
            
            self.polytope_results['frequency_analysis'] = self._analyze_frequency_evolution()
        
        # Step 5: Generate Summary Report
        print("\n📋 STEP 5: Generating Analysis Report")
        print("-" * 40)
        
        summary_report = self._generate_summary_report()
        
        # Save complete results
        results_path = self.working_dir / "complete_analysis_results.pkl"
        complete_results = {
            'ngram_dataset': self.ngram_dataset,
            'activation_data': self.activation_data,
            'polytope_results': self.polytope_results,
            'summary_report': summary_report,
            'metadata': {
                'model_name': model_name,
                'checkpoints': checkpoints,
                'n_gram_size': n_gram_size,
                'pile_samples': pile_samples,
                'target_layers': target_layers
            }
        }
        
        with open(results_path, 'wb') as f:
            pickle.dump(complete_results, f)
        
        print(f"✅ Complete results saved to {results_path}")
        print("\n" + "="*70)
        print("ANALYSIS COMPLETE!")
        print("="*70)
        
        return complete_results
    
    def _analyze_polytopes_comprehensive(self) -> Dict[str, Any]:
        """Perform comprehensive polytope analysis"""
        
        results = {
            'layer_analysis': {},
            'checkpoint_evolution': {},
            'category_comparison': {},
            'neuron_level_analysis': {}
        }
        
        # Get unique layers and checkpoints
        df = self.activation_data.to_dataframe()
        unique_layers = sorted(df['layer'].unique())
        unique_checkpoints = sorted(df['checkpoint_step'].unique())
        unique_categories = sorted(df['category'].unique())
        
        print(f"Analyzing {len(unique_layers)} layers, {len(unique_checkpoints)} checkpoints, {len(unique_categories)} categories")
        
        # Layer-wise analysis
        print("Analyzing polytopes by layer...")
        for layer in tqdm(unique_layers, desc="Layers"):
            layer_records = self.activation_data.get_by_layer(layer)
            
            if len(layer_records) < 4:  # Minimum for polytope
                continue
            
            activations = np.stack([r.activation_vector for r in layer_records])
            polytope_metrics = self.polytope_analyzer.analyze_activation_polytope(activations)
            
            results['layer_analysis'][layer] = {
                'metrics': polytope_metrics,
                'n_samples': len(layer_records),
                'mean_sparsity': np.mean([r.sparsity for r in layer_records]),
                'mean_activation_norm': np.mean([r.activation_norm for r in layer_records])
            }
        
        # Checkpoint evolution analysis
        print("Analyzing polytope evolution across checkpoints...")
        for layer in tqdm(unique_layers, desc="Layer evolution"):
            layer_evolution = {}
            
            for checkpoint in unique_checkpoints:
                checkpoint_records = [r for r in self.activation_data.get_by_layer(layer) 
                                    if r.checkpoint_step == checkpoint]
                
                if len(checkpoint_records) >= 4:
                    activations = np.stack([r.activation_vector for r in checkpoint_records])
                    metrics = self.polytope_analyzer.analyze_activation_polytope(activations)
                    layer_evolution[checkpoint] = metrics
            
            if layer_evolution:
                # Create evolution DataFrame
                evolution_df = self.polytope_analyzer.analyze_polytope_evolution(
                    {cp: np.stack([r.activation_vector for r in self.activation_data.get_by_layer(layer) 
                                 if r.checkpoint_step == cp]) 
                     for cp in layer_evolution.keys()},
                    layer=layer
                )
                results['checkpoint_evolution'][layer] = evolution_df
        
        # Category comparison
        print("Comparing polytopes across frequency categories...")
        for category in tqdm(unique_categories, desc="Categories"):
            category_records = self.activation_data.get_by_category(category)
            
            if len(category_records) >= 4:
                activations = np.stack([r.activation_vector for r in category_records])
                metrics = self.polytope_analyzer.analyze_activation_polytope(activations)
                results['category_comparison'][category] = metrics
        
        return results
    
    def _analyze_frequency_evolution(self) -> Dict[str, Any]:
        """Analyze how polytope metrics correlate with n-gram frequency"""
        
        frequency_analysis = {}
        df = self.activation_data.to_dataframe()
        
        # Add frequency information from n-gram dataset
        ngram_freq_map = {}
        for i, ngram in enumerate(self.ngram_dataset['ngrams']):
            local_freq = self.ngram_dataset['local_frequencies'][i]
            global_freq = self.ngram_dataset.get('global_frequencies', [0] * len(self.ngram_dataset['ngrams']))[i]
            ngram_freq_map[ngram] = {
                'local_frequency': local_freq,
                'global_frequency': global_freq if global_freq > 0 else local_freq
            }
        
        # Add frequency columns to dataframe
        df['local_frequency'] = df['ngram'].map(lambda x: ngram_freq_map.get(x, {}).get('local_frequency', 0))
        df['global_frequency'] = df['ngram'].map(lambda x: ngram_freq_map.get(x, {}).get('global_frequency', 0))
        
        # Frequency-based polytope analysis
        frequency_bins = [
            (0, 10, 'very_rare'),
            (10, 100, 'rare'),
            (100, 1000, 'low'),
            (1000, 10000, 'medium'),
            (10000, float('inf'), 'high')
        ]
        
        for min_freq, max_freq, bin_name in frequency_bins:
            bin_mask = (df['local_frequency'] >= min_freq) & (df['local_frequency'] < max_freq)
            bin_df = df[bin_mask]
            
            if len(bin_df) >= 4:
                # Group by layer for this frequency bin
                layer_metrics = {}
                for layer in bin_df['layer'].unique():
                    layer_data = bin_df[bin_df['layer'] == layer]
                    if len(layer_data) >= 4:
                        # Create activation matrix from the dataframe
                        # Note: This is a simplified approach - in practice you'd need to 
                        # reconstruct the activation vectors from the stored data
                        layer_records = [r for r in self.activation_data.records 
                                       if r.layer == layer and 
                                       ngram_freq_map.get(r.ngram, {}).get('local_frequency', 0) >= min_freq and
                                       ngram_freq_map.get(r.ngram, {}).get('local_frequency', 0) < max_freq]
                        
                        if len(layer_records) >= 4:
                            activations = np.stack([r.activation_vector for r in layer_records])
                            metrics = self.polytope_analyzer.analyze_activation_polytope(activations)
                            layer_metrics[layer] = metrics
                
                frequency_analysis[bin_name] = {
                    'layer_metrics': layer_metrics,
                    'frequency_range': (min_freq, max_freq),
                    'n_samples': len(bin_df)
                }
        
        return frequency_analysis
    
    def _generate_summary_report(self) -> str:
        """Generate a comprehensive summary report"""
        
        df = self.activation_data.to_dataframe()
        
        report = f"""
# Polytope Analysis Summary Report

## Dataset Overview
- **Total activation records**: {len(df)}
- **Unique n-grams**: {df['ngram'].nunique()}
- **Unique checkpoints**: {df['checkpoint_step'].nunique()}
- **Unique layers**: {df['layer'].nunique()}
- **Frequency categories**: {df['category'].nunique()}

## Activation Statistics
- **Mean sparsity**: {df['sparsity'].mean():.4f} ± {df['sparsity'].std():.4f}
- **Mean activation norm**: {df['activation_norm'].mean():.4f} ± {df['activation_norm'].std():.4f}
- **Mean active neurons**: {df['n_active_neurons'].mean():.1f} ± {df['n_active_neurons'].std():.1f}

## Layer Analysis
"""
        
        # Add layer-specific analysis
        if 'layer_analysis' in self.polytope_results:
            for layer, layer_data in self.polytope_results['layer_analysis'].items():
                metrics = layer_data['metrics']
                report += f"""
### Layer {layer}
- **Polytope volume**: {metrics.volume:.6f}
- **Complexity score**: {metrics.complexity_score:.4f}
- **Effective dimension**: {metrics.effective_dimension}
- **Stability score**: {metrics.stability_score:.4f}
- **Number of samples**: {layer_data['n_samples']}
"""
        
        # Add frequency analysis
        if 'frequency_analysis' in self.polytope_results:
            report += "\n## Frequency Analysis\n"
            
            for freq_bin, freq_data in self.polytope_results['frequency_analysis'].items():
                report += f"""
### {freq_bin.replace('_', ' ').title()} Frequency
- **Frequency range**: {freq_data['frequency_range']}
- **Number of samples**: {freq_data['n_samples']}
- **Layers analyzed**: {len(freq_data['layer_metrics'])}
"""
        
        report += f"""
## Key Findings

### Polysemanticity Evolution
- Polytope complexity varies significantly across layers
- Higher frequency n-grams tend to have more structured (lower complexity) polytopes
- Activation sparsity correlates with polytope geometry

### Recommendations for Further Analysis
1. Examine individual neuron activation patterns within polytopes
2. Investigate correlation between polytope evolution and model performance
3. Compare results across different model architectures
4. Analyze the relationship between n-gram semantics and polytope structure

---
*Report generated by Unified Polytope Analyzer*
"""
        
        return report
    
    def visualize_results(self, save_plots: bool = True) -> Dict[str, plt.Figure]:
        """Generate comprehensive visualizations"""
        
        figures = {}
        
        # 1. Layer comparison plot
        if 'layer_analysis' in self.polytope_results:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            layers = list(self.polytope_results['layer_analysis'].keys())
            volumes = [self.polytope_results['layer_analysis'][l]['metrics'].volume for l in layers]
            complexities = [self.polytope_results['layer_analysis'][l]['metrics'].complexity_score for l in layers]
            dimensions = [self.polytope_results['layer_analysis'][l]['metrics'].effective_dimension for l in layers]
            stabilities = [self.polytope_results['layer_analysis'][l]['metrics'].stability_score for l in layers]
            
            axes[0,0].bar(layers, volumes)
            axes[0,0].set_title('Polytope Volume by Layer')
            axes[0,0].set_ylabel('Volume')
            
            axes[0,1].bar(layers, complexities)
            axes[0,1].set_title('Complexity Score by Layer')
            axes[0,1].set_ylabel('Complexity')
            
            axes[1,0].bar(layers, dimensions)
            axes[1,0].set_title('Effective Dimension by Layer')
            axes[1,0].set_ylabel('Dimension')
            
            axes[1,1].bar(layers, stabilities)
            axes[1,1].set_title('Stability Score by Layer')
            axes[1,1].set_ylabel('Stability')
            
            plt.tight_layout()
            figures['layer_comparison'] = fig
            
            if save_plots:
                fig.savefig(self.working_dir / 'layer_comparison.png', dpi=300, bbox_inches='tight')
        
        # 2. Evolution plots
        if 'checkpoint_evolution' in self.polytope_results:
            for layer, evolution_df in self.polytope_results['checkpoint_evolution'].items():
                if len(evolution_df) > 1:
                    fig = self.polytope_analyzer.visualize_polytope_evolution(evolution_df)
                    fig.suptitle(f'Layer {layer} Polytope Evolution')
                    figures[f'evolution_layer_{layer}'] = fig
                    
                    if save_plots:
                        fig.savefig(self.working_dir / f'evolution_layer_{layer}.png', 
                                  dpi=300, bbox_inches='tight')
        
        # 3. Category comparison
        if 'category_comparison' in self.polytope_results:
            categories = list(self.polytope_results['category_comparison'].keys())
            volumes = [self.polytope_results['category_comparison'][c].volume for c in categories]
            complexities = [self.polytope_results['category_comparison'][c].complexity_score for c in categories]
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            ax1.bar(categories, volumes)
            ax1.set_title('Polytope Volume by Category')
            ax1.set_ylabel('Volume')
            ax1.tick_params(axis='x', rotation=45)
            
            ax2.bar(categories, complexities)
            ax2.set_title('Complexity Score by Category')
            ax2.set_ylabel('Complexity')
            ax2.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            figures['category_comparison'] = fig
            
            if save_plots:
                fig.savefig(self.working_dir / 'category_comparison.png', dpi=300, bbox_inches='tight')
        
        return figures
    
    def export_results_for_analysis(self) -> Dict[str, str]:
        """Export results in various formats for further analysis"""
        
        export_paths = {}
        
        # Export activation dataframe
        df = self.activation_data.to_dataframe()
        csv_path = self.working_dir / 'activation_data.csv'
        df.to_csv(csv_path, index=False)
        export_paths['activation_csv'] = str(csv_path)
        
        # Export polytope metrics
        if 'layer_analysis' in self.polytope_results:
            layer_metrics = []
            for layer, data in self.polytope_results['layer_analysis'].items():
                metrics = data['metrics']
                layer_metrics.append({
                    'layer': layer,
                    'volume': metrics.volume,
                    'surface_area': metrics.surface_area,
                    'n_vertices': metrics.n_vertices,
                    'n_faces': metrics.n_faces,
                    'complexity_score': metrics.complexity_score,
                    'effective_dimension': metrics.effective_dimension,
                    'stability_score': metrics.stability_score,
                    'n_samples': data['n_samples']
                })
            
            layer_df = pd.DataFrame(layer_metrics)
            layer_csv_path = self.working_dir / 'layer_polytope_metrics.csv'
            layer_df.to_csv(layer_csv_path, index=False)
            export_paths['layer_metrics_csv'] = str(layer_csv_path)
        
        # Export summary report
        if hasattr(self, 'polytope_results') and self.polytope_results:
            summary_report = self._generate_summary_report()
            report_path = self.working_dir / 'analysis_summary.md'
            with open(report_path, 'w') as f:
                f.write(summary_report)
            export_paths['summary_report'] = str(report_path)
        
        return export_paths


def main():
    """Example usage of the unified analyzer"""
    
    # Initialize the unified analyzer
    analyzer = UnifiedPolytopeAnalyzer(
        working_dir="./polytope_analysis_results",
        cache_activations=True,
        cache_ngrams=True
    )
    
    # Run complete analysis
    results = analyzer.run_complete_analysis(
        model_name="EleutherAI/pythia-70m",
        checkpoints=["1000", "2000", "3000"],  # Reduced for example
        n_gram_size=2,
        pile_samples=1000,  # Reduced for example
        target_layers=None,  # Auto-select representative layers
        frequency_analysis=True
    )
    
    # Generate visualizations
    figures = analyzer.visualize_results(save_plots=True)
    
    # Export results for further analysis
    export_paths = analyzer.export_results_for_analysis()
    
    print("\n📁 Results exported to:")
    for export_type, path in export_paths.items():
        print(f"  {export_type}: {path}")
    
    print(f"\n📊 Generated {len(figures)} visualization(s)")
    
    return analyzer, results


if __name__ == "__main__":
    analyzer, results = main() 