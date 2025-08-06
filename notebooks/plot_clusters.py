#!/usr/bin/env python
"""
Script to analyze the evolution of polysemanticity over training checkpoints
"""

import pandas as pd # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
import argparse 
from pathlib import Path
from scipy import stats # type: ignore

def load_and_analyze_data(layer, neuron, model_size="70m"):
    """Load and analyze checkpoint data"""
    
    # Construct file paths
    results_dir = Path("results")
    csv_file = results_dir / f"L{layer}N{neuron}_pythia{model_size}_ckpt_summary.csv"
    
    # Check if file exists
    if not csv_file.exists():
        print(f"Error: File {csv_file} not found!")
        print("Run checkpoints_demo.py first to generate the checkpoint data.")
        return None
    
    # Load the data
    df = pd.read_csv(csv_file)
    
    # Extract layer and neuron info for global use
    global layer_neuron_info
    layer_neuron_info = f"L{layer}N{neuron}"
    
    print("=== POLYSEMANTICITY EVOLUTION ANALYSIS ===")
    print(f"Dataset: {len(df)} checkpoints from {df['checkpoint_step'].min()} to {df['checkpoint_step'].max()} steps")
    print(f"Neuron: {neuron} in Layer {layer} (blocks.{layer}.mlp) of Pythia-{model_size.upper()}")
    print()
    
    # Basic statistics
    print("=== BASIC STATISTICS ===")
    print(f"Mean clusters: {df['num_clusters'].mean():.2f} ± {df['num_clusters'].std():.2f}")
    print(f"Median clusters: {df['num_clusters'].median():.1f}")
    print(f"Min clusters: {df['num_clusters'].min()} (step {df.loc[df['num_clusters'].idxmin(), 'checkpoint_step']})")
    print(f"Max clusters: {df['num_clusters'].max()} (step {df.loc[df['num_clusters'].idxmax(), 'checkpoint_step']})")
    print()
    
    # Trend analysis
    print("=== TREND ANALYSIS ===")
    slope, intercept, r_value, p_value, std_err = stats.linregress(df['checkpoint_step'], df['num_clusters'])
    print(f"Linear trend slope: {slope:.6f}")
    print(f"R-squared: {r_value**2:.4f}")
    print(f"P-value: {p_value:.6f}")
    print(f"Standard error: {std_err:.6f}")
    print()
    
    # Phase analysis
    print("=== PHASE ANALYSIS ===")
    early = df[df['checkpoint_step'] <= 60000]
    middle = df[(df['checkpoint_step'] > 60000) & (df['checkpoint_step'] <= 80000)]
    late = df[df['checkpoint_step'] > 80000]
    
    print(f"Early phase (≤60k steps, n={len(early)}): {early['num_clusters'].mean():.1f} ± {early['num_clusters'].std():.1f} clusters")
    print(f"Middle phase (60k-80k steps, n={len(middle)}): {middle['num_clusters'].mean():.1f} ± {middle['num_clusters'].std():.1f} clusters")
    print(f"Late phase (>80k steps, n={len(late)}): {late['num_clusters'].mean():.1f} ± {late['num_clusters'].std():.1f} clusters")
    print()
    
    # Statistical tests
    print("=== STATISTICAL TESTS ===")
    # Early vs Late
    t_stat, p_val = stats.ttest_ind(early['num_clusters'], late['num_clusters'])
    print(f"Early vs Late t-test: t={t_stat:.3f}, p={p_val:.6f}")
    
    # Correlation with training step
    corr, p_corr = stats.pearsonr(df['checkpoint_step'], df['num_clusters'])
    print(f"Correlation with training step: r={corr:.3f}, p={p_corr:.6f}")
    print()
    
    # Cluster size analysis
    print("=== CLUSTER SIZE ANALYSIS ===")
    # Parse cluster sizes
    cluster_sizes = []
    for sizes_str in df['cluster_sizes']:
        try:
            # Convert string like "{15: 1, 2: 25, ...}" to list of sizes
            sizes = eval(sizes_str)
            cluster_sizes.extend([size for size in sizes.values()])
        except:
            continue
    
    if cluster_sizes:
        print(f"Total clusters analyzed: {len(cluster_sizes)}")
        print(f"Mean cluster size: {np.mean(cluster_sizes):.2f}")
        print(f"Median cluster size: {np.median(cluster_sizes):.1f}")
        print(f"Largest cluster: {max(cluster_sizes)}")
        print(f"Smallest cluster: {min(cluster_sizes)}")
        print(f"Single-element clusters: {sum(1 for s in cluster_sizes if s == 1)} ({sum(1 for s in cluster_sizes if s == 1)/len(cluster_sizes)*100:.1f}%)")
    print()
    
    print("=== STATISTICAL SIGNIFICANCE ===")
    if p_val < 0.05:
        print("   • Early vs Late difference is statistically significant")
        print("   • Polysemanticity decrease is real, not random")
    else:
        print("   • Early vs Late difference is not statistically significant")
        print("   • May need more data or different analysis")
    print()
    
    return df

def create_advanced_plots(df):
    """Create advanced visualization plots"""
    
    # Create a multi-panel figure
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    # Add main title using the global layer_neuron_info
    fig.suptitle(f'Checkpoint Analysis: {layer_neuron_info}', fontsize=14, fontweight='bold')
    
    # Plot 1: Evolution with trend line and moving average overlay
    axes[0,0].plot(df['checkpoint_step'], df['num_clusters'], 'o-', color='#2E86AB', alpha=0.7, markersize=3, label='Raw data')
    z = np.polyfit(df['checkpoint_step'], df['num_clusters'], 1)
    p = np.poly1d(z)
    axes[0,0].plot(df['checkpoint_step'], p(df['checkpoint_step']), '--', color='#A23B72', linewidth=1.5, label='Linear trend')
    
    # Add 5-point moving average overlay
    window = 5
    moving_avg = df['num_clusters'].rolling(window=window, center=True).mean()
    axes[0,0].plot(df['checkpoint_step'], moving_avg, '-', linewidth=2, color='#F18F01', label=f'{window}-point moving avg')
    
    axes[0,0].set_title('Evolution of Polysemanticity', fontweight='bold', fontsize=10)
    axes[0,0].set_xlabel('Training Step', fontsize=9)
    axes[0,0].set_ylabel('Number of Clusters', fontsize=9)
    axes[0,0].legend(fontsize=8)
    axes[0,0].grid(True, alpha=0.3)
    
    # Plot 2: Distribution and statistics
    max_val = df['num_clusters'].max()
    min_val = df['num_clusters'].min()
    max_step = df.loc[df['num_clusters'].idxmax(), 'checkpoint_step']
    min_step = df.loc[df['num_clusters'].idxmin(), 'checkpoint_step']
    
    axes[0,1].hist(df['num_clusters'], bins=12, alpha=0.7, color='#2E86AB', edgecolor='black')
    axes[0,1].axvline(df['num_clusters'].mean(), color='red', linestyle='--', 
                     label=f'Mean: {df["num_clusters"].mean():.1f}')
    axes[0,1].axvline(df['num_clusters'].median(), color='green', linestyle='--', 
                     label=f'Median: {df["num_clusters"].median():.1f}')
    
    # Add max and min lines to histogram
    axes[0,1].axvline(max_val, color='red', linestyle='-', linewidth=2, alpha=0.8,
                     label=f'Max: {max_val}')
    axes[0,1].axvline(min_val, color='green', linestyle='-', linewidth=2, alpha=0.8,
                     label=f'Min: {min_val}')
    
    axes[0,1].set_title('Distribution of Cluster Counts', fontweight='bold', fontsize=10)
    axes[0,1].set_xlabel('Number of Clusters', fontsize=9)
    axes[0,1].set_ylabel('Frequency', fontsize=9)
    axes[0,1].legend(fontsize=8)
    axes[0,1].grid(True, alpha=0.3)
    
    # Plot 3: Number of examples over time
    axes[1,0].plot(df['checkpoint_step'], df['n_examples'], 'o-', color='#2E86AB', alpha=0.7, markersize=3)
    axes[1,0].set_title('Number of High-Activation Examples', fontweight='bold', fontsize=10)
    axes[1,0].set_xlabel('Training Step', fontsize=9)
    axes[1,0].set_ylabel('Number of Examples', fontsize=9)
    axes[1,0].grid(True, alpha=0.3)
    
    # Plot 4: Scatter plot with trend
    axes[1,1].scatter(df['checkpoint_step'], df['num_clusters'], alpha=0.7, color='#2E86AB', s=30)
    
    # Add trend line
    z = np.polyfit(df['checkpoint_step'], df['num_clusters'], 1)
    p = np.poly1d(z)
    axes[1,1].plot(df['checkpoint_step'], p(df['checkpoint_step']), 
                   "--", alpha=0.8, color='#A23B72', linewidth=2, 
                   label=f'Trend (slope: {z[0]:.6f})')
    
    axes[1,1].set_title('Cluster Evolution Scatter Plot', fontweight='bold', fontsize=10)
    axes[1,1].set_xlabel('Training Step', fontsize=9)
    axes[1,1].set_ylabel('Number of Clusters', fontsize=9)
    axes[1,1].legend(fontsize=8)
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save with dynamic filename
    results_dir = Path("results")
    plot_file = results_dir / f"{layer_neuron_info}_cluster_statistics.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Advanced analysis plot saved to {plot_file}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze polysemanticity evolution across training checkpoints.")
    parser.add_argument("layer", type=int, help="Layer number (e.g., 5 for blocks.5.mlp)")
    parser.add_argument("neuron", type=int, help="Neuron number (e.g., 100 for L5N100)")
    parser.add_argument("--model_size", type=str, default="70m", help="Model size (e.g., 70m, 1.3b, 2.7b)")
    
    args = parser.parse_args()
    
    df = load_and_analyze_data(args.layer, args.neuron, args.model_size)
    if df is not None:
        create_advanced_plots(df) 