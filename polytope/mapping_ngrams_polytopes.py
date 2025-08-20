#!/usr/bin/env python3
"""
N-gram to Polytope Correlation Pipeline

This module builds upon the refined polytope analyzer to create direct correlations
between n-grams and polytope structures, enabling analysis of how specific
linguistic patterns relate to geometric activation structures.

Key Features:
- Load activation records from pickle files in refined polytope analyzer format
- Group activations by n-gram patterns and frequency categories
- Compute polytope metrics for each n-gram group
- Generate correlation visualizations and statistical analyses
- Track evolution of n-gram-polytope relationships across checkpoints
"""

import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from scipy.stats import pearsonr, spearmanr, mannwhitneyu
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

warnings.filterwarnings('ignore', category=FutureWarning)


class NGramPolytopeMapper:
    """
    Maps n-grams to polytope structures and analyzes their correlations.
    
    This class provides a complete pipeline for:
    1. Loading activation records from refined polytope analyzer format
    2. Grouping by n-grams and computing polytope metrics
    3. Analyzing correlations between linguistic patterns and geometric structures
    4. Generating visualizations and statistical summaries
    """
    
    def __init__(self, cache_dir: str = "cache", random_seed: int = 42):
        """Initialize the n-gram polytope mapper."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.random_seed = random_seed
        
        # Configure logging
        logger.add(
            self.cache_dir / "ngram_polytope_mapping.log",
            rotation="10 MB",
            level="INFO",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
        )
        
        # Set random seed
        np.random.seed(random_seed)
        self.rng = np.random.default_rng(random_seed)
        
        # Data storage
        self.activation_records: List[Dict[str, Any]] = []
        self.ngram_groups: Dict[str, List[Dict[str, Any]]] = {}
        self.polytope_metrics: Dict[str, Dict[str, float]] = {}
        
        logger.info("Initialized NGramPolytopeMapper")
    
    def load_activation_records(self, pickle_path: str) -> None:
        """Load activation records from pickle file."""
        pickle_path = Path(pickle_path)
        if not pickle_path.exists():
            raise FileNotFoundError(f"Pickle file not found: {pickle_path}")
        
        logger.info(f"Loading activation records from {pickle_path}")
        
        with open(pickle_path, 'rb') as f:
            self.activation_records = pickle.load(f)
        
        logger.info(f"Loaded {len(self.activation_records)} activation records")
        
        # Log data structure info
        if self.activation_records:
            sample = self.activation_records[0]
            logger.info(f"Record structure: {list(sample.keys())}")
            logger.info(f"Sample n-gram: '{sample.get('ngram', 'N/A')}'")
            logger.info(f"Sample activation shape: {sample.get('activation_vector', np.array([])).shape}")
    
    def group_by_ngrams(self, min_samples: int = 10) -> None:
        """Group activation records by n-gram patterns."""
        logger.info("Grouping activation records by n-grams")
        
        ngram_counts = {}
        for record in self.activation_records:
            ngram = record.get('ngram', '')
            if ngram:
                ngram_counts[ngram] = ngram_counts.get(ngram, 0) + 1
        
        # Filter n-grams with sufficient samples
        valid_ngrams = {ngram for ngram, count in ngram_counts.items() 
                       if count >= min_samples}
        
        logger.info(f"Found {len(valid_ngrams)} n-grams with >= {min_samples} samples")
        
        # Group records by n-gram
        self.ngram_groups = {}
        for record in self.activation_records:
            ngram = record.get('ngram', '')
            if ngram in valid_ngrams:
                if ngram not in self.ngram_groups:
                    self.ngram_groups[ngram] = []
                self.ngram_groups[ngram].append(record)
        
        # Log grouping results
        for ngram, records in list(self.ngram_groups.items())[:5]:
            logger.info(f"N-gram '{ngram}': {len(records)} records")
    
    def compute_polytope_metrics_for_ngram(self, ngram: str, records: List[Dict]) -> Dict[str, float]:
        """Compute polytope metrics for a specific n-gram group."""
        if not records:
            return {}
        
        # Extract activation vectors
        activations = []
        for record in records:
            activation = record.get('activation_vector')
            if activation is not None and len(activation) > 0:
                activations.append(activation)
        
        if len(activations) < 2:
            logger.warning(f"Insufficient activations for n-gram '{ngram}': {len(activations)}")
            return {}
        
        activations = np.array(activations)
        
        # Compute metrics
        metrics = {}
        
        # Basic statistics
        metrics['n_samples'] = len(activations)
        metrics['mean_activation_norm'] = float(np.mean([record.get('activation_norm', 0) 
                                                        for record in records]))
        metrics['mean_sparsity'] = float(np.mean([record.get('sparsity', 0) 
                                                 for record in records]))
        
        # Polytope density metrics
        try:
            # Standardize activations
            scaler = StandardScaler()
            activations_std = scaler.fit_transform(activations)
            
            # Compute pairwise distances
            from scipy.spatial.distance import pdist
            euclidean_dists = pdist(activations_std, metric='euclidean')
            hamming_dists = pdist(activations > 0, metric='hamming')
            
            if len(euclidean_dists) > 0 and len(hamming_dists) > 0:
                metrics['mean_euclidean_distance'] = float(np.mean(euclidean_dists))
                metrics['mean_hamming_distance'] = float(np.mean(hamming_dists))
                metrics['polytope_density'] = float(np.mean(hamming_dists) / 
                                                  (np.mean(euclidean_dists) + 1e-8))
        except Exception as e:
            logger.warning(f"Error computing distances for '{ngram}': {e}")
        
        # Participation ratio (effective dimensionality)
        try:
            pca = PCA()
            pca.fit(activations_std)
            explained_var = pca.explained_variance_
            participation_ratio = (np.sum(explained_var) ** 2) / np.sum(explained_var ** 2)
            metrics['participation_ratio'] = float(participation_ratio)
        except Exception as e:
            logger.warning(f"Error computing PCA for '{ngram}': {e}")
        
        # Cluster quality metrics
        try:
            if len(activations) >= 10:  # Minimum samples for clustering
                clustering = DBSCAN(eps=0.5, min_samples=3)
                cluster_labels = clustering.fit_predict(activations_std)
                
                if len(set(cluster_labels)) > 1:  # At least 2 clusters
                    sil_score = silhouette_score(activations_std, cluster_labels)
                    metrics['silhouette_score'] = float(sil_score)
                    metrics['n_clusters'] = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        except Exception as e:
            logger.warning(f"Error computing clustering for '{ngram}': {e}")
        
        return metrics
    
    def compute_all_polytope_metrics(self) -> None:
        """Compute polytope metrics for all n-gram groups."""
        logger.info("Computing polytope metrics for all n-gram groups")
        
        self.polytope_metrics = {}
        for ngram, records in self.ngram_groups.items():
            metrics = self.compute_polytope_metrics_for_ngram(ngram, records)
            if metrics:
                self.polytope_metrics[ngram] = metrics
        
        logger.info(f"Computed metrics for {len(self.polytope_metrics)} n-grams")
    
    def analyze_ngram_frequency_correlations(self) -> pd.DataFrame:
        """Analyze correlations between n-gram frequency and polytope metrics."""
        logger.info("Analyzing n-gram frequency correlations")
        
        if not self.polytope_metrics:
            logger.error("No polytope metrics available. Run compute_all_polytope_metrics first.")
            return pd.DataFrame()
        
        # Create correlation dataframe
        correlation_data = []
        for ngram, metrics in self.polytope_metrics.items():
            row = {'ngram': ngram}
            row.update(metrics)
            correlation_data.append(row)
        
        df = pd.DataFrame(correlation_data)
        
        # Add frequency information from original records
        ngram_frequencies = {}
        for record in self.activation_records:
            ngram = record.get('ngram', '')
            if ngram in ngram_frequencies:
                ngram_frequencies[ngram] += 1
            else:
                ngram_frequencies[ngram] = 1
        
        df['frequency'] = df['ngram'].map(ngram_frequencies)
        
        # Compute correlations with frequency
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        correlations = {}
        
        for col in numeric_cols:
            if col != 'frequency':
                try:
                    pearson_r, pearson_p = pearsonr(df['frequency'], df[col])
                    spearman_r, spearman_p = spearmanr(df['frequency'], df[col])
                    
                    correlations[col] = {
                        'pearson_r': pearson_r,
                        'pearson_p': pearson_p,
                        'spearman_r': spearman_r,
                        'spearman_p': spearman_p
                    }
                except Exception as e:
                    logger.warning(f"Error computing correlation for {col}: {e}")
        
        # Save correlations
        self.frequency_correlations = correlations
        
        logger.info("Frequency correlation analysis completed")
        return df
    
    def create_correlation_visualizations(self, output_dir: str = "visualizations") -> None:
        """Create comprehensive visualizations of n-gram to polytope correlations."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Creating visualizations in {output_dir}")
        
        if not self.polytope_metrics:
            logger.error("No polytope metrics available for visualization")
            return
        
        # Create dataframe for visualization
        viz_data = []
        for ngram, metrics in self.polytope_metrics.items():
            row = {'ngram': ngram}
            row.update(metrics)
            viz_data.append(row)
        
        df = pd.DataFrame(viz_data)
        
        # Add frequency information
        ngram_frequencies = {}
        for record in self.activation_records:
            ngram = record.get('ngram', '')
            ngram_frequencies[ngram] = ngram_frequencies.get(ngram, 0) + 1
        
        df['frequency'] = df['ngram'].map(ngram_frequencies)
        
        # 1. Frequency vs Polytope Density Scatter Plot
        plt.figure(figsize=(10, 6))
        plt.scatter(df['frequency'], df.get('polytope_density', []), alpha=0.6)
        plt.xlabel('N-gram Frequency')
        plt.ylabel('Polytope Density')
        plt.title('N-gram Frequency vs Polytope Density')
        plt.tight_layout()
        plt.savefig(output_dir / 'frequency_vs_polytope_density.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Participation Ratio vs Frequency
        if 'participation_ratio' in df.columns:
            plt.figure(figsize=(10, 6))
            plt.scatter(df['frequency'], df['participation_ratio'], alpha=0.6, color='orange')
            plt.xlabel('N-gram Frequency')
            plt.ylabel('Participation Ratio')
            plt.title('N-gram Frequency vs Participation Ratio')
            plt.tight_layout()
            plt.savefig(output_dir / 'frequency_vs_participation_ratio.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. Correlation Heatmap
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            plt.figure(figsize=(12, 8))
            correlation_matrix = df[numeric_cols].corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                       square=True, fmt='.3f')
            plt.title('Correlation Matrix of N-gram and Polytope Metrics')
            plt.tight_layout()
            plt.savefig(output_dir / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 4. Top N-grams by Polytope Density
        if 'polytope_density' in df.columns:
            top_ngrams = df.nlargest(10, 'polytope_density')
            plt.figure(figsize=(12, 6))
            plt.bar(range(len(top_ngrams)), top_ngrams['polytope_density'])
            plt.xlabel('N-gram Rank')
            plt.ylabel('Polytope Density')
            plt.title('Top 10 N-grams by Polytope Density')
            plt.xticks(range(len(top_ngrams)), top_ngrams['ngram'], rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(output_dir / 'top_ngrams_polytope_density.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 5. Distribution plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Frequency distribution
        axes[0, 0].hist(df['frequency'], bins=30, alpha=0.7, edgecolor='black')
        axes[0, 0].set_xlabel('Frequency')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_title('N-gram Frequency Distribution')
        
        # Polytope density distribution
        if 'polytope_density' in df.columns:
            axes[0, 1].hist(df['polytope_density'], bins=30, alpha=0.7, 
                           edgecolor='black', color='orange')
            axes[0, 1].set_xlabel('Polytope Density')
            axes[0, 1].set_ylabel('Count')
            axes[0, 1].set_title('Polytope Density Distribution')
        
        # Participation ratio distribution
        if 'participation_ratio' in df.columns:
            axes[1, 0].hist(df['participation_ratio'], bins=30, alpha=0.7, 
                           edgecolor='black', color='green')
            axes[1, 0].set_xlabel('Participation Ratio')
            axes[1, 0].set_ylabel('Count')
            axes[1, 0].set_title('Participation Ratio Distribution')
        
        # Sparsity distribution
        if 'mean_sparsity' in df.columns:
            axes[1, 1].hist(df['mean_sparsity'], bins=30, alpha=0.7, 
                           edgecolor='black', color='red')
            axes[1, 1].set_xlabel('Mean Sparsity')
            axes[1, 1].set_ylabel('Count')
            axes[1, 1].set_title('Mean Sparsity Distribution')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'metric_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualizations saved to {output_dir}")
    
    def generate_correlation_report(self, output_path: str = "ngram_polytope_report.json") -> Dict:
        """Generate comprehensive correlation analysis report."""
        logger.info("Generating correlation report")
        
        # Analyze correlations
        df = self.analyze_ngram_frequency_correlations()
        
        report = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'total_records': len(self.activation_records),
            'unique_ngrams': len(self.ngram_groups),
            'ngrams_with_metrics': len(self.polytope_metrics),
            'frequency_correlations': getattr(self, 'frequency_correlations', {}),
            'summary_statistics': {}
        }
        
        # Add summary statistics
        if not df.empty:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                report['summary_statistics'][col] = {
                    'mean': float(df[col].mean()),
                    'std': float(df[col].std()),
                    'min': float(df[col].min()),
                    'max': float(df[col].max()),
                    'median': float(df[col].median())
                }
        
        # Top correlations with frequency
        if hasattr(self, 'frequency_correlations'):
            sorted_correlations = sorted(
                self.frequency_correlations.items(),
                key=lambda x: abs(x[1]['pearson_r']),
                reverse=True
            )
            report['top_frequency_correlations'] = sorted_correlations[:5]
        
        # Save report
        output_path = Path(output_path)
        with open(output_path, 'w') as f:
            import json
            json.dump(report, f, indent=2)
        
        logger.info(f"Report saved to {output_path}")
        return report
    
    def run_full_pipeline(self, pickle_path: str, output_dir: str = "ngram_polytope_analysis",
                         min_samples: int = 10) -> Dict:
        """Run the complete n-gram to polytope correlation pipeline."""
        logger.info("Starting full n-gram polytope correlation pipeline")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Load data
            self.load_activation_records(pickle_path)
            
            # Group by n-grams
            self.group_by_ngrams(min_samples=min_samples)
            
            # Compute polytope metrics
            self.compute_all_polytope_metrics()
            
            # Create visualizations
            self.create_correlation_visualizations(output_dir / "visualizations")
            
            # Generate report
            report = self.generate_correlation_report(output_dir / "report.json")
            
            logger.info("Full pipeline completed successfully")
            return report
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise


def main():
    """Example usage of the NGramPolytopeMapper."""
    mapper = NGramPolytopeMapper()
    
    # Run full pipeline
    pickle_path = "/Applications/team-aasa/polytope/activation_records.pkl"
    report = mapper.run_full_pipeline(pickle_path)
    
    print("Pipeline completed!")
    print(f"Analyzed {report['total_records']} records")
    print(f"Found {report['unique_ngrams']} unique n-grams")
    print(f"Computed metrics for {report['ngrams_with_metrics']} n-grams")


if __name__ == "__main__":
    main()