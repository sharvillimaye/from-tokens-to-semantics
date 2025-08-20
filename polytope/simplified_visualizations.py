#!/usr/bin/env python3
"""
Simplified Polytope Visualization Dashboard

A streamlined visualization module that consolidates layer-wise polytope analysis
results into intuitive dashboard-style plots for easier interpretation.

IMPROVEMENTS OVER ORIGINAL:
===========================
- Dashboard-style layouts with multiple layers in single view
- Heatmap visualizations for cross-layer/checkpoint comparisons  
- Simplified metric focus on most important indicators
- Interactive-style layouts suitable for research presentations
- Reduced file clutter with consolidated outputs

KEY VISUALIZATIONS:
==================
1. Layer Evolution Dashboard - All layers in grid layout
2. Metric Heatmaps - Checkpoint vs Layer comparisons
3. Summary Dashboard - Key findings at-a-glance
4. Evolution Trends - Simplified time series view
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger

# Set style for clean visualizations
plt.style.use('default')
sns.set_palette("husl")


class SimplifiedPolytopeVisualizer:
    """
    Simplified visualization dashboard for polytope analysis results.
    
    Creates consolidated, easy-to-interpret visualizations from refined analyzer output.
    """
    
    def __init__(self, figsize_base: tuple = (12, 8), dpi: int = 300):
        """
        Initialize visualizer with display preferences.
        
        Args:
            figsize_base: Base figure size for plots
            dpi: Resolution for saved figures
        """
        self.figsize_base = figsize_base
        self.dpi = dpi
        self.colors = {
            'high_freq': '#1f77b4',  # Blue
            'low_freq': '#ff7f0e',   # Orange
            'difference': '#2ca02c'   # Green
        }
        
    def create_dashboard_from_results(self, results: List[Dict[str, Any]], output_dir: str) -> None:
        """
        Create complete simplified dashboard from analysis results.
        
        Args:
            results: List of analysis results from refined_polytope_analyzer
            output_dir: Directory to save visualizations
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Convert results to DataFrame
        df = self._results_to_dataframe(results)
        
        if df.empty:
            logger.warning("No valid data for simplified visualization dashboard")
            return
            
        # Create dashboard visualizations
        self.create_layer_evolution_dashboard(df, output_path)
        self.create_metric_heatmaps(df, output_path) 
        self.create_summary_dashboard(df, output_path)
        self.create_evolution_trends(df, output_path)
        
        logger.info(f"Simplified visualization dashboard created in {output_path}")
        
    def create_layer_evolution_dashboard(self, df: pd.DataFrame, output_path: Path) -> None:
        """
        Create dashboard showing all layers' evolution in one view.
        
        Shows key metrics across layers in a grid layout for easy comparison.
        """
        layers = sorted(df['layer'].unique())
        n_layers = len(layers)
        
        # Create grid layout - aim for square-ish arrangement
        cols = int(np.ceil(np.sqrt(n_layers)))
        rows = int(np.ceil(n_layers / cols))
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        fig.suptitle('Polytope Evolution Dashboard - All Layers', fontsize=16, fontweight='bold')
        
        # Flatten axes array for easier indexing
        if n_layers == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            pass  # axes is already 1D
        else:
            axes = axes.flatten()
        
        for i, layer in enumerate(layers):
            ax = axes[i] if n_layers > 1 else axes
            layer_df = df[df['layer'] == layer]
            
            # Plot key metric: density difference (shows superposition strength)
            high_data = layer_df[layer_df['group'] == 'high_freq'].set_index('checkpoint')['density_mean']
            low_data = layer_df[layer_df['group'] == 'low_freq'].set_index('checkpoint')['density_mean']
            
            if not high_data.empty and not low_data.empty:
                # Plot both groups
                ax.plot(high_data.index, high_data.values, 
                       marker='o', color=self.colors['high_freq'], 
                       label='High Freq', linewidth=2, markersize=6)
                ax.plot(low_data.index, low_data.values,
                       marker='s', color=self.colors['low_freq'], 
                       label='Low Freq', linewidth=2, markersize=6)
                
                # Highlight difference with fill
                ax.fill_between(high_data.index, high_data.values, low_data.values,
                               alpha=0.2, color=self.colors['difference'])
            
            ax.set_title(f'Layer {layer}', fontweight='bold')
            ax.set_xlabel('Checkpoint')
            ax.set_ylabel('Polytope Density')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
            
            # Add summary stats as text
            if not high_data.empty and not low_data.empty:
                diff_mean = (high_data - low_data).mean()
                ax.text(0.05, 0.95, f'Δ={diff_mean:.3f}', transform=ax.transAxes,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor=self.colors['difference'], alpha=0.7),
                       fontsize=8, verticalalignment='top')
        
        # Remove empty subplots
        for i in range(n_layers, len(axes)):
            fig.delaxes(axes[i])
            
        plt.tight_layout()
        plt.savefig(output_path / 'layer_evolution_dashboard.png', dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
    def create_metric_heatmaps(self, df: pd.DataFrame, output_path: Path) -> None:
        """
        Create heatmap visualizations showing metric patterns across layers and checkpoints.
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Polytope Metrics Heatmap Analysis', fontsize=16, fontweight='bold')
        
        metrics = [
            ('density_mean', 'Polytope Density'),
            ('participation_ratio', 'Participation Ratio'),
            ('pattern_reuse_rate', 'Pattern Reuse Rate'),
            ('activation_norm', 'Activation Norm')
        ]
        
        for idx, (metric, title) in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]
            
            # Create pivot table for heatmap - show difference between high/low freq
            high_df = df[df['group'] == 'high_freq'].pivot(index='layer', columns='checkpoint', values=metric)
            low_df = df[df['group'] == 'low_freq'].pivot(index='layer', columns='checkpoint', values=metric)
            diff_df = high_df - low_df
            
            # Create heatmap
            sns.heatmap(diff_df, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
                       ax=ax, cbar_kws={'label': 'High Freq - Low Freq'})
            
            ax.set_title(f'{title}\n(High Freq - Low Freq Difference)', fontweight='bold')
            ax.set_xlabel('Training Checkpoint')
            ax.set_ylabel('Layer')
            
        plt.tight_layout()
        plt.savefig(output_path / 'metric_heatmaps.png', dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
    def create_summary_dashboard(self, df: pd.DataFrame, output_path: Path) -> None:
        """
        Create high-level summary dashboard with key findings.
        """
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Main title
        fig.suptitle('Polytope Superposition Summary Dashboard', fontsize=18, fontweight='bold')
        
        # 1. Overall density evolution (large plot)
        ax1 = fig.add_subplot(gs[0, :2])
        for group in ['high_freq', 'low_freq']:
            group_data = df[df['group'] == group].groupby('checkpoint')['density_mean'].mean()
            ax1.plot(group_data.index, group_data.values, 
                    marker='o', label=group.replace('_', ' ').title(),
                    color=self.colors[group], linewidth=3, markersize=8)
        ax1.set_title('Overall Polytope Density Evolution', fontweight='bold', fontsize=14)
        ax1.set_xlabel('Training Checkpoint')
        ax1.set_ylabel('Mean Density (All Layers)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Layer-wise difference summary
        ax2 = fig.add_subplot(gs[0, 2])
        high_means = df[df['group'] == 'high_freq'].groupby('layer')['density_mean'].mean()
        low_means = df[df['group'] == 'low_freq'].groupby('layer')['density_mean'].mean()
        diff_means = high_means - low_means
        
        bars = ax2.bar(range(len(diff_means)), diff_means.values, 
                      color=[self.colors['high_freq'] if x > 0 else self.colors['low_freq'] 
                            for x in diff_means.values])
        ax2.set_title('Density Difference\nby Layer', fontweight='bold')
        ax2.set_xlabel('Layer')
        ax2.set_ylabel('High - Low Freq')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.set_xticks(range(len(diff_means)))
        ax2.set_xticklabels([f'L{i}' for i in diff_means.index])
        
        # 3. Participation ratio trends
        ax3 = fig.add_subplot(gs[1, :])
        layers = sorted(df['layer'].unique())
        checkpoints = sorted(df['checkpoint'].unique())
        
        for layer in layers:
            layer_data = df[df['layer'] == layer]
            high_pr = layer_data[layer_data['group'] == 'high_freq'].set_index('checkpoint')['participation_ratio']
            low_pr = layer_data[layer_data['group'] == 'low_freq'].set_index('checkpoint')['participation_ratio']
            
            if not high_pr.empty and not low_pr.empty:
                pr_diff = high_pr - low_pr
                ax3.plot(pr_diff.index, pr_diff.values, 
                        marker='o', label=f'Layer {layer}', alpha=0.7)
        
        ax3.set_title('Participation Ratio Difference Evolution (High - Low Freq)', fontweight='bold')
        ax3.set_xlabel('Training Checkpoint')
        ax3.set_ylabel('Participation Ratio Difference')
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax3.grid(True, alpha=0.3)
        
        # 4. Summary statistics text
        ax4 = fig.add_subplot(gs[2, :])
        ax4.axis('off')
        
        # Calculate key statistics
        overall_density_diff = (df[df['group'] == 'high_freq']['density_mean'].mean() - 
                               df[df['group'] == 'low_freq']['density_mean'].mean())
        max_layer_diff = diff_means.max()
        max_layer = diff_means.idxmax()
        
        summary_text = f"""
KEY FINDINGS:
• Overall Density Difference (High - Low Freq): {overall_density_diff:.4f}
• Strongest Superposition in Layer {max_layer}: Δ = {max_layer_diff:.4f}  
• Total Layers Analyzed: {len(layers)}
• Training Checkpoints: {len(checkpoints)} ({min(checkpoints)} - {max(checkpoints)})
• {"High frequency n-grams show higher polytope density" if overall_density_diff > 0 else "Low frequency n-grams show higher polytope density"}
        """
        
        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", 
                facecolor='lightgray', alpha=0.8))
        
        plt.savefig(output_path / 'summary_dashboard.png', dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
    def create_evolution_trends(self, df: pd.DataFrame, output_path: Path) -> None:
        """
        Create simplified evolution trend visualizations focusing on key patterns.
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Polytope Evolution Trends - Simplified View', fontsize=16, fontweight='bold')
        
        # 1. Density trend with confidence bands
        ax1 = axes[0]
        for group in ['high_freq', 'low_freq']:
            group_data = df[df['group'] == group]
            
            # Calculate mean and std for each checkpoint
            stats = group_data.groupby('checkpoint')['density_mean'].agg(['mean', 'std']).fillna(0)
            
            ax1.plot(stats.index, stats['mean'], 
                    marker='o', label=group.replace('_', ' ').title(),
                    color=self.colors[group], linewidth=2, markersize=6)
            
            # Add confidence band
            ax1.fill_between(stats.index, 
                           stats['mean'] - stats['std'], 
                           stats['mean'] + stats['std'],
                           alpha=0.2, color=self.colors[group])
        
        ax1.set_title('Density Evolution\n(with std bands)', fontweight='bold')
        ax1.set_xlabel('Checkpoint')
        ax1.set_ylabel('Polytope Density')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Pattern reuse rate comparison
        ax2 = axes[1]
        for group in ['high_freq', 'low_freq']:
            group_data = df[df['group'] == group]
            stats = group_data.groupby('checkpoint')['pattern_reuse_rate'].mean()
            
            ax2.plot(stats.index, stats.values,
                    marker='s', label=group.replace('_', ' ').title(),
                    color=self.colors[group], linewidth=2, markersize=6)
        
        ax2.set_title('Pattern Reuse Rate\n(Superposition Indicator)', fontweight='bold')
        ax2.set_xlabel('Checkpoint') 
        ax2.set_ylabel('Reuse Rate')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Cross-layer variability
        ax3 = axes[2]
        checkpoints = sorted(df['checkpoint'].unique())
        
        high_variability = []
        low_variability = []
        
        for checkpoint in checkpoints:
            cp_data = df[df['checkpoint'] == checkpoint]
            
            high_layers = cp_data[cp_data['group'] == 'high_freq']['density_mean']
            low_layers = cp_data[cp_data['group'] == 'low_freq']['density_mean']
            
            high_variability.append(high_layers.std() if len(high_layers) > 1 else 0)
            low_variability.append(low_layers.std() if len(low_layers) > 1 else 0)
        
        ax3.plot(checkpoints, high_variability, marker='o', color=self.colors['high_freq'], 
                label='High Freq', linewidth=2)
        ax3.plot(checkpoints, low_variability, marker='s', color=self.colors['low_freq'],
                label='Low Freq', linewidth=2)
        
        ax3.set_title('Cross-Layer Variability\n(Density Std Dev)', fontweight='bold')
        ax3.set_xlabel('Checkpoint')
        ax3.set_ylabel('Std Dev Across Layers')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'evolution_trends_simplified.png', dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
    def _results_to_dataframe(self, results: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Convert analysis results to DataFrame format for visualization.
        """
        df_data = []
        
        for result in results:
            if 'error' in result:
                continue
                
            checkpoint = result['checkpoint_step']
            
            # Handle layer-wise results
            if result.get('analysis_type') == 'layer_wise':
                for layer, layer_data in result.get('layer_results', {}).items():
                    if isinstance(layer_data, dict) and 'high_freq' in layer_data and 'low_freq' in layer_data:
                        for group in ['high_freq', 'low_freq']:
                            group_data = layer_data[group]
                            
                            df_data.append({
                                'checkpoint': int(checkpoint),
                                'layer': int(layer),
                                'group': group,
                                'density_mean': group_data.get('density_mean', 0),
                                'participation_ratio': group_data.get('participation_ratio', 0),
                                'pattern_reuse_rate': group_data.get('pattern_reuse_rate', 0),
                                'activation_norm': group_data.get('activation_norm', 0),
                                'sparsity': group_data.get('sparsity', 0)
                            })
            else:
                # Handle legacy format (averaged across layers)
                for group in ['high_freq', 'low_freq']:
                    density_data = result.get(f'{group}_density', {})
                    
                    df_data.append({
                        'checkpoint': int(checkpoint),
                        'layer': 0,  # Use layer 0 for averaged results
                        'group': group,
                        'density_mean': density_data.get('density_mean', 0),
                        'participation_ratio': result.get(f'{group}_participation_ratio', 0),
                        'pattern_reuse_rate': density_data.get('pattern_reuse_rate', 0),
                        'activation_norm': density_data.get('activation_norm', 0),
                        'sparsity': density_data.get('sparsity', 0)
                    })
        
        return pd.DataFrame(df_data)


def create_simplified_visualizations_from_json(results_json_path: str, output_dir: str) -> None:
    """
    Convenience function to create simplified visualizations from saved JSON results.
    
    Args:
        results_json_path: Path to saved analysis results JSON
        output_dir: Directory to save simplified visualizations
    """
    import json
    
    with open(results_json_path, 'r') as f:
        results = json.load(f)
    
    visualizer = SimplifiedPolytopeVisualizer()
    visualizer.create_dashboard_from_results(results, output_dir)
    
    logger.info(f"Simplified visualizations created from {results_json_path}")


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python simplified_visualizations.py <results_json> <output_dir>")
        sys.exit(1)
        
    create_simplified_visualizations_from_json(sys.argv[1], sys.argv[2])