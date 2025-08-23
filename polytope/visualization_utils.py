"""
Clean visualization utilities for polytope density and participation ratio analysis.
Replicates the clear layerwise plotting style from polytope_analyzer.
Designed for NeurIPS paper quality figures with PDF export support.

Key Features:
- PDF export by default for publication-quality figures
- PNG export option for web/presentation use
- Optimized font embedding and vector graphics for PDFs
- Layer-wise and checkpoint evolution visualizations
- Publication-ready styling and formatting
"""
import json
from pathlib import Path
from typing import Dict, List, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

# Set publication-quality defaults optimized for Overleaf
plt.rcParams.update({
    'font.size': 14,          # Larger for paper PDFs
    'axes.titlesize': 14,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 14,
    'font.family': 'serif',
    'font.serif': ['Times', 'Times New Roman', 'DejaVu Serif'],
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 1.2,    # Thicker axes
    'grid.linewidth': 0.7,
    'lines.linewidth': 2.0,
    'lines.markersize': 7,
    'xtick.major.size': 5.0,
    'xtick.major.width': 1.0,
    'ytick.major.size': 5.0,
    'ytick.major.width': 1.0,
    # PDF-specific optimizations for Overleaf
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'text.usetex': False,
    'mathtext.fontset': 'stix',
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.format': 'pdf',
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
    # Color-blind friendly defaults
    'axes.prop_cycle': plt.cycler('color', [
        '#1f77b4',  # Blue
        '#ff7f0e',  # Orange  
        '#2ca02c',  # Green
        '#d62728',  # Red
        '#9467bd',  # Purple
        '#8c564b',  # Brown
        '#e377c2',  # Pink
        '#7f7f7f'   # Gray
    ])
})


class PolytopeVisualizationUtils:
    """Utilities for creating clear polytope analysis visualizations matching polytope_analyzer style."""
    
    def __init__(self, dpi: int = 300, figsize_scale: float = 1.0, output_format: str = 'pdf'):
        """Initialize visualization utilities.
        
        Args:
            dpi: Resolution for saved figures
            figsize_scale: Scale factor for figure sizes
            output_format: Output format for figures ('pdf' or 'png')
        """
        self.dpi = dpi
        self.figsize_scale = figsize_scale
        self.output_format = output_format.lower()
        
        # Use the same colors as the original polytope analyzer
        self.colors = {
            'high_freq': '#1f77b4',  # Blue for high frequency
            'low_freq': '#ff7f0e'    # Orange for low frequency  
        }
        
        # Same markers as original
        self.markers = {
            'high_freq': 'o',  # Circle for high
            'low_freq': 's'    # Square for low
        }
    
    def _save_figure(self, output_path: Path, filename: str) -> None:
        """Save figure with the specified format and handle PDF optimization for Overleaf."""
        file_extension = self.output_format
        full_path = output_path / f"{filename}.{file_extension}"
        
        if self.output_format == 'pdf':
            # PDF-specific optimizations for Overleaf compatibility
            plt.savefig(full_path, format='pdf', bbox_inches='tight', 
                       pad_inches=0.05,  # Minimal padding for Overleaf
                       dpi=self.dpi, 
                       backend='pdf',    # Force PDF backend
                       metadata={'Creator': 'Polytope Analysis Pipeline',
                                'Title': filename,
                                'Subject': 'Neural Superposition Research'},
                       transparent=False,  # Solid white background for Overleaf
                       facecolor='white',  # Ensure white background
                       edgecolor='none')   # No edge color
        else:
            # PNG fallback with high DPI
            plt.savefig(full_path, format='png', dpi=self.dpi, bbox_inches='tight',
                       facecolor='white', transparent=False)
        
        plt.close()
        logger.info(f"Saved figure: {full_path} (Overleaf-ready)")
    
    def generate_overleaf_latex_code(self, output_dir: str) -> str:
        """Generate LaTeX code for including figures in Overleaf."""
        latex_code = []
        latex_code.append("% LaTeX code for including polytope analysis figures in Overleaf")
        latex_code.append("% Place these figures in your Overleaf project and use the code below\n")
        
        figure_mappings = {
            'density_mean_heatmap_layers_by_checkpoint_high_freq': 'Polytope Density Heatmap (High Freq): Layers × Checkpoints',
            'density_mean_heatmap_layers_by_checkpoint_low_freq': 'Polytope Density Heatmap (Low Freq): Layers × Checkpoints',
            'participation_ratio_heatmap_layers_by_checkpoint_high_freq': 'Participation Ratio Heatmap (High Freq): Layers × Checkpoints',
            'participation_ratio_heatmap_layers_by_checkpoint_low_freq': 'Participation Ratio Heatmap (Low Freq): Layers × Checkpoints',
            'density_mean_heatmap_diff_high_minus_low': 'Polytope Density Δ (High − Low)',
            'participation_ratio_heatmap_diff_high_minus_low': 'Participation Ratio Δ (High − Low)',
            'small_multiples_layers_over_time_density_mean': 'Per-Layer Polytope Density Over Checkpoints',
            'small_multiples_layers_over_time_participation_ratio': 'Per-Layer Participation Ratio Over Checkpoints'
        }
        
        for filename, caption in figure_mappings.items():
            latex_code.append(f"\\begin{{figure}}[htbp]")
            latex_code.append(f"    \\centering")
            latex_code.append(f"    \\includegraphics[width=0.9\\textwidth]{{{filename}.{self.output_format}}}")
            latex_code.append(f"    \\caption{{{caption}}}")
            latex_code.append(f"    \\label{{fig:{filename}}}")
            latex_code.append(f"\\end{{figure}}\n")
        
        # Note: per-layer figures removed to reduce clutter; use small-multiples instead.
        
        return "\n".join(latex_code)
    
    def save_overleaf_readme(self, output_dir: str) -> None:
        """Save README with Overleaf usage instructions."""
        readme_content = f"""# Polytope Analysis Figures for Overleaf

This directory contains publication-ready figures in {self.output_format.upper()} format for use in Overleaf.

## Files Generated:
- `density_mean_heatmap_layers_by_checkpoint_high_freq.{self.output_format}`: Density heatmap (High Freq)
- `density_mean_heatmap_layers_by_checkpoint_low_freq.{self.output_format}`: Density heatmap (Low Freq)
- `participation_ratio_heatmap_layers_by_checkpoint_high_freq.{self.output_format}`: PR heatmap (High Freq)
- `participation_ratio_heatmap_layers_by_checkpoint_low_freq.{self.output_format}`: PR heatmap (Low Freq)
- `density_mean_heatmap_diff_high_minus_low.{self.output_format}`: Density difference heatmap (High − Low)
- `participation_ratio_heatmap_diff_high_minus_low.{self.output_format}`: PR difference heatmap (High − Low)
- `small_multiples_layers_over_time_density_mean.{self.output_format}`: Per-layer density time series
- `small_multiples_layers_over_time_participation_ratio.{self.output_format}`: Per-layer PR time series
- `polytope_evolution_legacy.{self.output_format}`: Legacy averaged evolution (if applicable)

## Overleaf Usage:

1. **Upload to Overleaf**: 
   - Zip all .{self.output_format} files
   - Upload to your Overleaf project
   - Extract in the main directory or a `figures/` subdirectory

2. **LaTeX Inclusion**:
   ```latex
   \\usepackage{{graphicx}}
   
   \\begin{{figure}}[htbp]
       \\centering
       \\includegraphics[width=0.95\\textwidth]{{small_multiples_layers_over_time_density_mean.{self.output_format}}}
       \\caption{{Per-layer polytope density over checkpoints}}
       \\label{{fig:small_multiples_density}}
   \\end{{figure}}
   ```

3. **For Multi-panel Figures**: Use `width=\\textwidth` for better visibility
4. **For Individual Layers**: Include specific layer numbers in filenames

## Figure Quality:
- Resolution: {self.dpi} DPI
- Format: {self.output_format.upper()}
- Optimized for: Academic publications
- Compatible with: Overleaf, LaTeX, academic journals

## Citing Figures:
Reference figures using \\ref{{fig:label_name}} in your text.
"""
        
        readme_path = Path(output_dir) / "OVERLEAF_README.md"
        with open(readme_path, 'w') as f:
            f.write(readme_content)
            
        # Also save LaTeX code
        latex_code = self.generate_overleaf_latex_code(output_dir)
        latex_path = Path(output_dir) / "figures_latex_code.tex"
        with open(latex_path, 'w') as f:
            f.write(latex_code)
            
        logger.info(f"Overleaf documentation saved to {output_dir}")
    
    def visualize_from_json(self,
                           json_path: Union[str, Path],
                           output_dir: str = "visualizations") -> None:
        """Create layerwise visualizations from polytope analyzer JSON results.
        
        Args:
            json_path: Path to the polytope_analysis_results.json file
            output_dir: Directory to save visualizations
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Load and parse JSON results
        df = self._parse_json_results(json_path)
        
        if df.empty:
            logger.error("No data found in JSON results")
            return
        
        # Check if we have layer-wise data
        layer_data = df[df['layer'] != 'averaged']
        
        if not layer_data.empty:
            logger.info("Creating layer-wise visualizations")
            # Only create heatmaps and small multiples plots
            self._plot_layer_checkpoint_heatmaps(df, output_path)
            self._plot_group_difference_heatmaps(df, output_path)
            self._plot_small_multiples_layers_over_time(df, output_path)
        else:
            logger.info("Creating legacy (averaged) visualizations")  
            self._plot_legacy_evolution(df, output_path)
        
        # Generate Overleaf-specific documentation and LaTeX code
        self.save_overleaf_readme(str(output_path))
        
        logger.info(f"Visualizations saved to {output_path}")
        logger.info(f"Overleaf-ready {self.output_format.upper()} files with documentation generated")
    
    def _parse_json_results(self, json_path: Union[str, Path]) -> pd.DataFrame:
        """Parse polytope analyzer JSON results into DataFrame for visualization."""
        with open(json_path, 'r') as f:
            results = json.load(f)
        
        metrics_data = []
        
        for result in results:
            if 'error' in result:
                continue
                
            checkpoint = result.get('checkpoint_step', 'unknown')
            analysis_type = result.get('analysis_type', 'legacy')
            
            if analysis_type == 'layer_wise':
                # Process layer-wise results
                layer_results = result.get('layer_results', {})
                for layer_idx, layer_data in layer_results.items():
                    if 'error' in layer_data:
                        continue
                        
                    # Extract metrics for both frequency groups
                    for freq_type in ['high_freq', 'low_freq']:
                        density_key = f'{freq_type}_density'
                        pr_key = f'{freq_type}_participation_ratio'
                        sparsity_key = f'{freq_type}_sparsity'
                        norm_key = f'{freq_type}_activation_norm'
                        reuse_key = f'{freq_type}_pattern_reuse_rate'
                        
                        density_data = layer_data.get(density_key, {})
                        
                        metrics_data.append({
                            'checkpoint': int(checkpoint) if str(checkpoint).isdigit() else checkpoint,
                            'layer': int(layer_idx),
                            'group': freq_type,
                            'density_mean': density_data.get('density_mean', 0),
                            'density_std': density_data.get('density_std', 0),
                            'boundary_crossings': density_data.get('boundary_crossings', 0),
                            'boundary_crossing_rate': density_data.get('boundary_crossing_rate', 0),
                            'participation_ratio': layer_data.get(pr_key, 0),
                            'sparsity': layer_data.get(sparsity_key, 0),
                            'activation_norm': layer_data.get(norm_key, 0),
                            'pattern_reuse_rate': layer_data.get(reuse_key, 0),
                        })
            else:
                # Process legacy results (averaged across layers)
                for freq_type in ['high_freq', 'low_freq']:
                    density_key = f'{freq_type}_density'
                    pr_key = f'{freq_type}_participation_ratio'
                    sparsity_key = f'{freq_type}_sparsity'
                    norm_key = f'{freq_type}_activation_norm'
                    
                    density_data = result.get(density_key, {})
                    
                    metrics_data.append({
                        'checkpoint': int(checkpoint) if str(checkpoint).isdigit() else checkpoint,
                        'layer': 'averaged',
                        'group': freq_type,
                        'density_mean': density_data.get('density_mean', 0),
                        'density_std': density_data.get('density_std', 0),
                        'boundary_crossings': density_data.get('boundary_crossings', 0),
                        'boundary_crossing_rate': density_data.get('boundary_crossing_rate', 0),
                        'participation_ratio': result.get(pr_key, 0),
                        'sparsity': result.get(sparsity_key, 0),
                        'activation_norm': result.get(norm_key, 0),
                        'pattern_reuse_rate': 0,  # Not available in legacy
                    })
        
        return pd.DataFrame(metrics_data)
    
    def _plot_layer_wise_evolution(self, df: pd.DataFrame, output_path: Path) -> None:
        """Plot layer-wise evolution with separate plots per layer (replicating polytope_analyzer style)."""
        layer_data = df[df['layer'] != 'averaged']
        if layer_data.empty:
            logger.warning("No layer-wise data found")
            return
            
        layers = sorted(layer_data['layer'].unique())
        
        # Core metrics to plot (matching original polytope analyzer)
        metrics = [
            ('density_mean', 'Polytope Density'),
            ('participation_ratio', 'Participation Ratio'),
            ('sparsity', 'Sparsity'),
            ('activation_norm', 'Activation Norm'),
            ('pattern_reuse_rate', 'Pattern Reuse Rate')
        ]
        
        # Create separate plots for each layer (like original)
        for layer in layers:
            layer_df = layer_data[layer_data['layer'] == layer]
            if layer_df.empty:
                continue
                
            fig, axes = plt.subplots(2, 3, figsize=(18 * self.figsize_scale, 12 * self.figsize_scale))
            # Removed overall figure title for cleaner PDF placement
            
            for idx, (metric, title) in enumerate(metrics):
                if idx < 6:  # We have 6 subplots (2x3)
                    ax = axes[idx // 3, idx % 3]
                    
                    for group in ['high_freq', 'low_freq']:
                        group_data = layer_df[layer_df['group'] == group].sort_values('checkpoint')
                        if not group_data.empty and metric in group_data.columns:
                            # Use same styling as original
                            color = self.colors[group]
                            marker = self.markers[group]
                            ax.plot(group_data['checkpoint'], group_data[metric],
                                   marker=marker, color=color, linewidth=2, markersize=8,
                                   label=group.replace('_', ' ').title() + ' N-grams')
                    
                    ax.set_xlabel('Training Checkpoint', fontsize=12)
                    ax.set_ylabel(title, fontsize=12)
                    # No per-axis title for cleaner PDF
                    ax.legend(fontsize=10)
                    ax.grid(True, alpha=0.3)
                    
                    # Add interpretive annotations (like original)
                    if metric == 'density_mean':
                        ax.text(0.02, 0.98, 'Higher = More Polytope Structure',
                               transform=ax.transAxes, fontsize=8, alpha=0.7, verticalalignment='top')
                    elif metric == 'pattern_reuse_rate':
                        ax.text(0.02, 0.98, 'Higher = More Superposition',
                               transform=ax.transAxes, fontsize=8, alpha=0.7, verticalalignment='top')
                    elif metric == 'participation_ratio':
                        ax.text(0.02, 0.98, 'Higher = More Effective Dimensions',
                               transform=ax.transAxes, fontsize=8, alpha=0.7, verticalalignment='top')
            
            # Remove empty subplot if we have fewer than 6 metrics
            if len(metrics) < 6:
                fig.delaxes(axes[1, 2])
            
            plt.tight_layout()
            self._save_figure(output_path, f'polytope_evolution_layer_{layer}')
            
            logger.info(f"Created layer {layer} visualization")
    
    def _plot_cross_layer_comparison(self, df: pd.DataFrame, output_path: Path) -> None:
        """Create plots comparing metrics across layers (replicating polytope_analyzer style)."""
        layer_data = df[df['layer'] != 'averaged']
        if layer_data.empty:
            return
            
        checkpoints = sorted(layer_data['checkpoint'].unique())
        layers = sorted(layer_data['layer'].unique())
        
        # Plot density evolution across layers for each checkpoint
        fig, axes = plt.subplots(1, 2, figsize=(15 * self.figsize_scale, 6 * self.figsize_scale))
        # Removed overall figure title for cleaner PDF
        
        for group_idx, group in enumerate(['high_freq', 'low_freq']):
            ax = axes[group_idx]
            
            for checkpoint in checkpoints:
                checkpoint_data = layer_data[
                    (layer_data['checkpoint'] == checkpoint) & (layer_data['group'] == group)
                ].sort_values('layer')
                
                if not checkpoint_data.empty:
                    ax.plot(checkpoint_data['layer'], checkpoint_data['density_mean'],
                           marker='o', linewidth=2, markersize=6, label=f'Checkpoint {checkpoint}')
            
            ax.set_xlabel('Layer')
            ax.set_ylabel('Polytope Density')
            # No per-axis title for cleaner PDF
            # Prevent tick label overlap by using strategic tick placement
            if layers:
                try:
                    min_layer, max_layer = int(min(layers)), int(max(layers))
                except Exception:
                    min_layer, max_layer = layers[0], layers[-1]
                
                # Generous padding to prevent edge label crowding
                padding = max(3, (max_layer - min_layer) * 0.08)
                ax.set_xlim(min_layer - padding, max_layer + padding)
                
                # Strategic tick selection - avoid both endpoints if too crowded
                if len(layers) > 6:
                    # Use middle values and avoid endpoints entirely if problematic
                    step = max(8, len(layers) // 4)  # Even larger steps
                    xticks = list(range(min_layer + step, max_layer, step))
                    # Only include endpoints if there's enough separation
                    if not xticks or xticks[0] - min_layer > step // 2:
                        xticks.insert(0, min_layer)
                    if not xticks or max_layer - xticks[-1] > step // 2:
                        xticks.append(max_layer)
                    ax.set_xticks(xticks)
                    ax.set_xticklabels([str(x) for x in xticks])
                else:
                    # For fewer layers, use all but add padding
                    ax.set_xticks(layers)
                ax.margins(x=0.08)  # Even more margin
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self._save_figure(output_path, 'cross_layer_density_comparison')
        
        # Additional cross-layer participation ratio comparison  
        fig, axes = plt.subplots(1, 2, figsize=(15 * self.figsize_scale, 6 * self.figsize_scale))
        # Removed overall figure title for cleaner PDF
        
        for group_idx, group in enumerate(['high_freq', 'low_freq']):
            ax = axes[group_idx]
            
            for checkpoint in checkpoints:
                checkpoint_data = layer_data[
                    (layer_data['checkpoint'] == checkpoint) & (layer_data['group'] == group)
                ].sort_values('layer')
                
                if not checkpoint_data.empty:
                    ax.plot(checkpoint_data['layer'], checkpoint_data['participation_ratio'],
                           marker='s', linewidth=2, markersize=6, label=f'Checkpoint {checkpoint}')
            
            ax.set_xlabel('Layer')
            ax.set_ylabel('Participation Ratio')
            # No per-axis title for cleaner PDF
            # Prevent tick label overlap by using strategic tick placement
            if layers:
                try:
                    min_layer, max_layer = int(min(layers)), int(max(layers))
                except Exception:
                    min_layer, max_layer = layers[0], layers[-1]
                
                # Generous padding to prevent edge label crowding
                padding = max(3, (max_layer - min_layer) * 0.08)
                ax.set_xlim(min_layer - padding, max_layer + padding)
                
                # Strategic tick selection - avoid both endpoints if too crowded
                if len(layers) > 6:
                    # Use middle values and avoid endpoints entirely if problematic
                    step = max(8, len(layers) // 4)  # Even larger steps
                    xticks = list(range(min_layer + step, max_layer, step))
                    # Only include endpoints if there's enough separation
                    if not xticks or xticks[0] - min_layer > step // 2:
                        xticks.insert(0, min_layer)
                    if not xticks or max_layer - xticks[-1] > step // 2:
                        xticks.append(max_layer)
                    ax.set_xticks(xticks)
                    ax.set_xticklabels([str(x) for x in xticks])
                else:
                    # For fewer layers, use all but add padding
                    ax.set_xticks(layers)
                ax.margins(x=0.08)  # Even more margin
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self._save_figure(output_path, 'cross_layer_participation_comparison')
        
        logger.info("Created cross-layer comparison visualizations")
    
    def _plot_legacy_evolution(self, df: pd.DataFrame, output_path: Path) -> None:
        """Plot legacy evolution (averaged across layers)."""
        fig, axes = plt.subplots(2, 2, figsize=(15 * self.figsize_scale, 10 * self.figsize_scale))
        # Removed overall figure title for cleaner PDF
        
        metrics = [
            ('density_mean', 'Polytope Density'),
            ('participation_ratio', 'Participation Ratio'),
            ('sparsity', 'Sparsity'),
            ('activation_norm', 'Activation Norm')
        ]
        
        for idx, (metric, title) in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]
            
            for group in ['high_freq', 'low_freq']:
                group_data = df[df['group'] == group].sort_values('checkpoint')
                if not group_data.empty:
                    color = self.colors[group]
                    marker = self.markers[group]
                    ax.plot(group_data['checkpoint'], group_data[metric],
                           marker=marker, color=color, linewidth=2, markersize=8,
                           label=group.replace('_', ' ').title() + ' N-grams')
            
            ax.set_xlabel('Training Checkpoint')
            ax.set_ylabel(title)
            # No per-axis title for cleaner PDF
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self._save_figure(output_path, 'polytope_evolution_legacy')

    # Removed unused individual plot helpers to reduce code surface area
    
    # Removed combined 3-panel figure in favor of simpler, clearer figures

    # ===== New: NeurIPS-quality checkpoint evolution visuals =====
    def _prepare_grid(self, df: pd.DataFrame, metric: str, group: str):
        """Return sorted layers, checkpoints, and a (layers x checkpoints) grid for a metric/group."""
        layer_data = df[df['layer'] != 'averaged']
        sub = layer_data[layer_data['group'] == group]
        if sub.empty:
            return [], [], np.zeros((0, 0))

        # Ensure numeric checkpoints
        def _to_int(x):
            try:
                return int(x)
            except Exception:
                return x
        sub = sub.copy()
        sub['checkpoint_num'] = sub['checkpoint'].apply(_to_int)

        layers = sorted(sub['layer'].unique())
        checkpoints = sorted(sub['checkpoint_num'].unique())

        # Build grid
        pivot = sub.pivot_table(index='layer', columns='checkpoint_num', values=metric, aggfunc='mean')
        # Reindex to ensure full grid order
        pivot = pivot.reindex(index=layers, columns=checkpoints)
        grid = pivot.values
        return layers, checkpoints, grid

    def _plot_layer_checkpoint_heatmaps(self, df: pd.DataFrame, output_path: Path) -> None:
        """Heatmaps of metric across (layer x checkpoint) for both groups."""
        metrics = [
            ('density_mean', 'Polytope Density', 'viridis'),
            ('participation_ratio', 'Participation Ratio', 'magma')
        ]

        for metric, title, cmap in metrics:
            for group in ['high_freq', 'low_freq']:
                layers, checkpoints, grid = self._prepare_grid(df, metric, group)
                if grid.size == 0:
                    continue

                fig, ax = plt.subplots(1, 1, figsize=(12 * self.figsize_scale, 6 * self.figsize_scale))
                im = ax.imshow(grid, aspect='auto', origin='lower', cmap=cmap)
                # No heatmap title
                ax.set_xlabel('Checkpoint')
                ax.set_ylabel('Layer')

                # Tick subsampling to avoid clutter
                if len(checkpoints) > 10:
                    step = max(1, len(checkpoints) // 10)
                    xticks_idx = list(range(0, len(checkpoints), step))
                else:
                    xticks_idx = list(range(len(checkpoints)))
                ax.set_xticks(xticks_idx)
                ax.set_xticklabels([str(checkpoints[i]) for i in xticks_idx], rotation=45, ha='right')

                if len(layers) > 20:
                    step = max(1, len(layers) // 20)
                    yticks_idx = list(range(0, len(layers), step))
                else:
                    yticks_idx = list(range(len(layers)))
                ax.set_yticks(yticks_idx)
                ax.set_yticklabels([str(layers[i]) for i in yticks_idx])

                cbar = plt.colorbar(im, ax=ax)
                cbar.set_label(title)

                plt.tight_layout()
                fname = f'{metric}_heatmap_layers_by_checkpoint_{group}'
                self._save_figure(output_path, fname)
                logger.info(f"Saved {fname}.{self.output_format}")

    def _plot_group_difference_heatmaps(self, df: pd.DataFrame, output_path: Path) -> None:
        """Heatmaps of (high_freq - low_freq) across (layer x checkpoint)."""
        metrics = [
            ('density_mean', 'Polytope Density', 'coolwarm'),
            ('participation_ratio', 'Participation Ratio', 'coolwarm')
        ]

        for metric, title, cmap in metrics:
            layers_h, checkpoints_h, grid_h = self._prepare_grid(df, metric, 'high_freq')
            layers_l, checkpoints_l, grid_l = self._prepare_grid(df, metric, 'low_freq')

            if grid_h.size == 0 or grid_l.size == 0:
                continue

            # Align shapes if needed (intersection of layers/checkpoints)
            layers = sorted(list(set(layers_h).intersection(layers_l)))
            checkpoints = sorted(list(set(checkpoints_h).intersection(checkpoints_l)))
            if not layers or not checkpoints:
                continue

            # Reconstruct aligned grids
            def grid_for(group):
                sub = df[(df['layer'] != 'averaged') & (df['group'] == group)].copy()
                def _to_int(x):
                    try:
                        return int(x)
                    except Exception:
                        return x
                sub['checkpoint_num'] = sub['checkpoint'].apply(_to_int)
                pivot = sub.pivot_table(index='layer', columns='checkpoint_num', values=metric, aggfunc='mean')
                pivot = pivot.reindex(index=layers, columns=checkpoints)
                return pivot.values

            Gh = grid_for('high_freq')
            Gl = grid_for('low_freq')
            diff = Gh - Gl

            v = np.nanmax(np.abs(diff)) if np.isfinite(diff).any() else 1.0
            v = v if v > 0 else 1.0

            fig, ax = plt.subplots(1, 1, figsize=(12 * self.figsize_scale, 6 * self.figsize_scale))
            im = ax.imshow(diff, aspect='auto', origin='lower', cmap=cmap, vmin=-v, vmax=v)
            # No heatmap title
            ax.set_xlabel('Checkpoint')
            ax.set_ylabel('Layer')

            # Ticks
            if len(checkpoints) > 10:
                step = max(1, len(checkpoints) // 10)
                xticks_idx = list(range(0, len(checkpoints), step))
            else:
                xticks_idx = list(range(len(checkpoints)))
            ax.set_xticks(xticks_idx)
            ax.set_xticklabels([str(checkpoints[i]) for i in xticks_idx], rotation=45, ha='right')

            if len(layers) > 20:
                step = max(1, len(layers) // 20)
                yticks_idx = list(range(0, len(layers), step))
            else:
                yticks_idx = list(range(len(layers)))
            ax.set_yticks(yticks_idx)
            ax.set_yticklabels([str(layers[i]) for i in yticks_idx])

            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label(f'{title} (High − Low)')

            plt.tight_layout()
            fname = f'{metric}_heatmap_diff_high_minus_low'
            self._save_figure(output_path, fname)
            logger.info(f"Saved {fname}.{self.output_format}")

    def _plot_checkpoint_trend_summaries(self, df: pd.DataFrame, output_path: Path) -> None:
        """Layer-aggregated checkpoint evolution with uncertainty bands for both groups."""
        layer_data = df[df['layer'] != 'averaged'].copy()
        if layer_data.empty:
            return

        def _to_int(x):
            try:
                return int(x)
            except Exception:
                return x
        layer_data['checkpoint_num'] = layer_data['checkpoint'].apply(_to_int)

        metrics = [
            ('density_mean', 'Polytope Density'),
            ('participation_ratio', 'Participation Ratio')
        ]

        for metric, title in metrics:
            fig, ax = plt.subplots(1, 1, figsize=(10 * self.figsize_scale, 6 * self.figsize_scale))
            for group in ['high_freq', 'low_freq']:
                sub = layer_data[layer_data['group'] == group]
                if sub.empty:
                    continue
                grouped = sub.groupby('checkpoint_num')[metric]
                mean = grouped.mean()
                q25 = grouped.quantile(0.25)
                q75 = grouped.quantile(0.75)

                xs = mean.index.values
                color = self.colors[group]
                label = group.replace('_', ' ').title()
                ax.plot(xs, mean.values, color=color, linewidth=2, label=label)
                ax.fill_between(xs, q25.values, q75.values, color=color, alpha=0.2)

            ax.set_xlabel('Checkpoint')
            ax.set_ylabel(title)
            # No per-axis title for cleaner PDF
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            fname = f'checkpoint_trends_{metric}'
            self._save_figure(output_path, fname)
            logger.info(f"Saved {fname}.{self.output_format}")

    def _plot_small_multiples_layers_over_time(self, df: pd.DataFrame, output_path: Path, max_layers: int = 12) -> None:
        """Small-multiples: per-layer time series for both groups with improved x-axis readability."""
        layer_data = df[df['layer'] != 'averaged']
        if layer_data.empty:
            return

        layers = sorted(layer_data['layer'].unique())
        if len(layers) > max_layers:
            # Evenly sample layers for display
            idxs = np.linspace(0, len(layers) - 1, max_layers).astype(int)
            layers = [layers[i] for i in idxs]

        def _to_int(x):
            try:
                return int(x)
            except Exception:
                return x
        layer_data = layer_data.copy()
        layer_data['checkpoint_num'] = layer_data['checkpoint'].apply(_to_int)

        # Get all checkpoints for consistent x-axis formatting
        all_checkpoints = sorted(layer_data['checkpoint_num'].unique())

        metrics = [
            ('density_mean', 'Polytope Density'),
            ('participation_ratio', 'Participation Ratio')
        ]

        n = len(layers)
        # Reduce columns for better spacing and readability
        ncols = 3 if n >= 9 else 2 if n >= 4 else 1
        nrows = int(np.ceil(n / ncols))

        for metric, title in metrics:
            # Increase figure size for better readability
            fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols * self.figsize_scale, 4 * nrows * self.figsize_scale))
            if nrows == 1 and ncols == 1:
                axes = np.array([axes])
            elif nrows == 1 or ncols == 1:
                axes = axes.reshape(nrows, ncols)
            # No overall suptitle for small multiples

            for idx, layer in enumerate(layers):
                r, c = divmod(idx, ncols)
                ax = axes[r, c]
                sub = layer_data[layer_data['layer'] == layer]
                for group in ['high_freq', 'low_freq']:
                    gsub = sub[sub['group'] == group].sort_values('checkpoint_num')
                    if not gsub.empty:
                        color = self.colors[group]
                        marker = self.markers[group]
                        ax.plot(gsub['checkpoint_num'], gsub[metric], marker=marker, color=color, 
                               linewidth=2, markersize=5, label=group if idx == 0 else None)
                
                # No per-axis title to maximize data-ink ratio
                ax.grid(True, alpha=0.3)
                
                # Improved x-axis formatting
                if len(all_checkpoints) > 6:
                    # Show fewer ticks but make them readable
                    step = max(1, len(all_checkpoints) // 4)
                    tick_positions = all_checkpoints[::step]
                    ax.set_xticks(tick_positions)
                    ax.set_xticklabels([str(int(x)) for x in tick_positions], rotation=45, ha='right')
                else:
                    ax.set_xticks(all_checkpoints)
                    ax.set_xticklabels([str(int(x)) for x in all_checkpoints], rotation=45, ha='right')
                
                # Only show x-label on bottom row
                if r == nrows - 1:
                    ax.set_xlabel('Checkpoint', fontsize=10)
                # Only show y-label on leftmost column
                if c == 0:
                    ax.set_ylabel(title, fontsize=10)

            # Remove unused axes
            for j in range(n, nrows * ncols):
                r, c = divmod(j, ncols)
                if r < nrows and c < ncols:
                    fig.delaxes(axes[r, c])

            # Add legend
            handles, labels = None, None
            for r in range(nrows):
                for c in range(ncols):
                    if r * ncols + c < len(layers):
                        h, l = axes[r, c].get_legend_handles_labels()
                        if h:
                            handles, labels = h, l
                            break
                if handles:
                    break
            
            if handles:
                fig.legend(handles, [lbl.replace('_', ' ').title() for lbl in labels], 
                          loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=2, fontsize=11)

            plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # Leave space for legend
            fname = f'small_multiples_layers_over_time_{metric}'
            self._save_figure(output_path, fname)
            logger.info(f"Saved {fname}.{self.output_format}")


# Convenience function for direct usage
def create_visualizations_from_json(json_path: Union[str, Path],
                                   output_dir: str = "visualizations", 
                                   dpi: int = 300,
                                   output_format: str = 'pdf') -> None:
    """Create all polytope visualizations from JSON results file.
    
    Args:
        json_path: Path to polytope_analysis_results.json file
        output_dir: Output directory for visualizations
        dpi: Resolution for saved figures
        output_format: Output format for figures ('pdf' or 'png')
        
    Example:
        create_visualizations_from_json("analysis_results/polytope_analysis_results.json", "neurips_figures")
    """
    visualizer = PolytopeVisualizationUtils(dpi=dpi, output_format=output_format)
    visualizer.visualize_from_json(json_path, output_dir)
# NOTE: Avoid running visualization generation on import in module context.