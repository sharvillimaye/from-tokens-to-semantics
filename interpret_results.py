#!/usr/bin/env python3
"""
Script to help interpret polytope superposition analysis results.
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_results():
    """Load the analysis results."""
    with open('cache/multi_checkpoint_analysis/multi_checkpoint_results.pkl', 'rb') as f:
        return pickle.load(f)

def analyze_dataset_characteristics(data):
    """Analyze the characteristics of the dataset that might affect results."""
    print("=" * 80)
    print("DATASET CHARACTERISTICS ANALYSIS")
    print("=" * 80)
    
    # Sample size analysis
    sample_ckpt = list(data['individual_analyses'].keys())[0]
    sample_data = data['individual_analyses'][sample_ckpt]['4']
    
    print(f"Sample sizes:")
    print(f"  High frequency: {sample_data['high_freq']['binary_patterns'].shape[0]} samples")
    print(f"  Low frequency: {sample_data['low_freq']['binary_patterns'].shape[0]} samples")
    print(f"  Activation dimension: {sample_data['high_freq']['binary_patterns'].shape[1]}")
    
    # Sparsity analysis
    hf_sparsity = sample_data['high_freq']['sparsity']
    lf_sparsity = sample_data['low_freq']['sparsity']
    
    print(f"\nSparsity analysis:")
    print(f"  High frequency sparsity: {np.mean(hf_sparsity):.3f} ± {np.std(hf_sparsity):.3f}")
    print(f"  Low frequency sparsity: {np.mean(lf_sparsity):.3f} ± {np.std(lf_sparsity):.3f}")
    print(f"  Sparsity difference: {np.mean(hf_sparsity) - np.mean(lf_sparsity):.3f}")
    
    # Unique patterns analysis
    hf_unique = sample_data['high_freq']['n_unique_patterns']
    lf_unique = sample_data['low_freq']['n_unique_patterns']
    
    print(f"\nUnique pattern analysis:")
    print(f"  High frequency unique patterns: {hf_unique}")
    print(f"  Low frequency unique patterns: {lf_unique}")
    print(f"  Compression ratio (high): {sample_data['high_freq']['compression_ratio']:.3f}")
    print(f"  Compression ratio (low): {sample_data['low_freq']['compression_ratio']:.3f}")
    
    # Participation ratio analysis
    hf_pr = sample_data['high_freq']['participation_ratio']
    lf_pr = sample_data['low_freq']['participation_ratio']
    
    print(f"\nParticipation ratio analysis:")
    print(f"  High frequency participation ratio: {hf_pr:.2f}")
    print(f"  Low frequency participation ratio: {lf_pr:.2f}")
    print(f"  Ratio (high/low): {hf_pr/lf_pr:.2f}")
    
    # Superposition analysis
    hf_sup = sample_data['high_freq']['interference_per_dimension']
    lf_sup = sample_data['low_freq']['interference_per_dimension']
    
    print(f"\nSuperposition analysis:")
    print(f"  High frequency interference: {hf_sup:.4f}")
    print(f"  Low frequency interference: {lf_sup:.4f}")
    print(f"  Superposition difference (high - low): {hf_sup - lf_sup:.4f}")
    print(f"  Expected strong interference threshold: {1.0/512:.4f}")
    
    return sample_data

def analyze_evolution_trends(data):
    """Analyze how metrics evolve across checkpoints."""
    print("\n" + "=" * 80)
    print("EVOLUTION TREND ANALYSIS")
    print("=" * 80)
    
    # Extract evolution data
    evolution_data = []
    for checkpoint, checkpoint_data in data['individual_analyses'].items():
        if '4' in checkpoint_data and 'comparison' in checkpoint_data['4']:
            comp = checkpoint_data['4']['comparison']
            evolution_data.append({
                'checkpoint': int(checkpoint),
                'superposition_diff': comp.get('superposition_strength_difference', 0),
                'density_ratio': comp.get('mean_density_ratio', 1.0),
                'interference_ratio': comp.get('interference_per_dimension_ratio', 1.0),
                'participation_ratio_diff': comp.get('participation_ratio_difference', 0),
                'unique_patterns_ratio': comp.get('n_unique_patterns_ratio', 1.0)
            })
    
    evolution_data.sort(key=lambda x: x['checkpoint'])
    
    # Calculate trends
    superposition_diffs = [d['superposition_diff'] for d in evolution_data]
    density_ratios = [d['density_ratio'] for d in evolution_data]
    
    print(f"Superposition strength evolution:")
    print(f"  Range: [{min(superposition_diffs):.4f}, {max(superposition_diffs):.4f}]")
    print(f"  Mean: {np.mean(superposition_diffs):.4f} ± {np.std(superposition_diffs):.4f}")
    print(f"  Trend: {'Increasing' if np.mean(superposition_diffs[-10:]) > np.mean(superposition_diffs[:10]) else 'Decreasing'}")
    
    print(f"\nDensity ratio evolution:")
    print(f"  Range: [{min(density_ratios):.3f}, {max(density_ratios):.3f}]")
    print(f"  Mean: {np.mean(density_ratios):.3f} ± {np.std(density_ratios):.3f}")
    print(f"  Trend: {'Increasing' if np.mean(density_ratios[-10:]) > np.mean(density_ratios[:10]) else 'Decreasing'}")
    
    return evolution_data

def explain_why_results_might_seem_counterintuitive(sample_data):
    """Explain potential reasons why results might seem counterintuitive."""
    print("\n" + "=" * 80)
    print("WHY RESULTS MIGHT SEEM COUNTERINTUITIVE")
    print("=" * 80)
    
    print("1. SMALL SAMPLE SIZE EFFECTS:")
    print("   - Only 21 high-frequency and 19 low-frequency samples")
    print("   - Small samples can lead to high variance in measurements")
    print("   - Statistical significance may be limited")
    
    print("\n2. HIGH SPARSITY ENVIRONMENT:")
    print("   - Both groups have ~94% sparsity (very sparse activations)")
    print("   - High sparsity can mask superposition effects")
    print("   - Most neurons are inactive, limiting interference patterns")
    
    print("\n3. SIMILAR COMPRESSION RATIOS:")
    print("   - High freq: 20 unique patterns / 512 dimensions = 0.039")
    print("   - Low freq: 18 unique patterns / 512 dimensions = 0.035")
    print("   - Very similar compression suggests similar representational efficiency")
    
    print("\n4. STRONG SUPERPOSITION IN BOTH GROUPS:")
    print("   - Both classified as 'strong' superposition")
    print("   - Interference > 1/512 threshold in both cases")
    print("   - Suggests both groups use superposition, just to different degrees")
    
    print("\n5. LAYER-SPECIFIC EFFECTS:")
    print("   - Only analyzing layer 4 (middle layer)")
    print("   - Superposition patterns may vary by layer")
    print("   - Early layers might show different patterns than later layers")

def create_interpretation_plots(evolution_data, sample_data):
    """Create plots to help interpret the results."""
    print("\n" + "=" * 80)
    print("CREATING INTERPRETATION PLOTS")
    print("=" * 80)
    
    # Create figure with multiple subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Superposition evolution
    checkpoints = [d['checkpoint'] for d in evolution_data]
    superposition_diffs = [d['superposition_diff'] for d in evolution_data]
    
    ax1.plot(checkpoints, superposition_diffs, 'b-', linewidth=2, alpha=0.7)
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='No difference')
    ax1.set_xlabel('Checkpoint Step')
    ax1.set_ylabel('Superposition Difference\n(High - Low Frequency)')
    ax1.set_title('Superposition Strength Evolution')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Density ratio evolution
    density_ratios = [d['density_ratio'] for d in evolution_data]
    
    ax2.plot(checkpoints, density_ratios, 'g-', linewidth=2, alpha=0.7)
    ax2.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Equal density')
    ax2.set_xlabel('Checkpoint Step')
    ax2.set_ylabel('Density Ratio\n(High / Low Frequency)')
    ax2.set_title('Polytope Density Evolution')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: Sparsity comparison
    hf_sparsity = sample_data['high_freq']['sparsity']
    lf_sparsity = sample_data['low_freq']['sparsity']
    
    ax3.hist(hf_sparsity, bins=10, alpha=0.6, label='High Frequency', color='blue')
    ax3.hist(lf_sparsity, bins=10, alpha=0.6, label='Low Frequency', color='red')
    ax3.set_xlabel('Sparsity (fraction of active neurons)')
    ax3.set_ylabel('Number of samples')
    ax3.set_title('Sparsity Distribution Comparison')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Participation ratio comparison
    hf_pr = sample_data['high_freq']['participation_ratio']
    lf_pr = sample_data['low_freq']['participation_ratio']
    
    categories = ['High Frequency', 'Low Frequency']
    participation_ratios = [hf_pr, lf_pr]
    colors = ['blue', 'red']
    
    bars = ax4.bar(categories, participation_ratios, color=colors, alpha=0.7)
    ax4.set_ylabel('Participation Ratio')
    ax4.set_title('Participation Ratio Comparison')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, value in zip(bars, participation_ratios):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{value:.1f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('cache/multi_checkpoint_analysis/interpretation_plots.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Interpretation plots saved to: cache/multi_checkpoint_analysis/interpretation_plots.png")

def main():
    """Main analysis function."""
    print("Loading analysis results...")
    data = load_results()
    
    # Analyze dataset characteristics
    sample_data = analyze_dataset_characteristics(data)
    
    # Analyze evolution trends
    evolution_data = analyze_evolution_trends(data)
    
    # Explain why results might seem counterintuitive
    explain_why_results_might_seem_counterintuitive(sample_data)
    
    # Create interpretation plots
    create_interpretation_plots(evolution_data, sample_data)
    
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS FOR FURTHER ANALYSIS")
    print("=" * 80)
    print("1. Increase sample size for more robust statistics")
    print("2. Analyze multiple layers to see layer-specific patterns")
    print("3. Use more diverse n-gram categories (not just country-capital pairs)")
    print("4. Consider analyzing earlier training stages for stronger effects")
    print("5. Examine individual neuron activation patterns")
    print("6. Compare with theoretical superposition predictions")

if __name__ == "__main__":
    main() 