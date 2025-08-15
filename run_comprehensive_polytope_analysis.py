#!/usr/bin/env python3
"""
Comprehensive Multi-Checkpoint Polytope Analysis Script

This script integrates checkpoint_analysis.py output with the SuperpositionAnalyzer
to conduct polytope analysis across multiple checkpoints and different layers for 
high frequency and low frequency patterns.

Usage:
    python run_comprehensive_polytope_analysis.py --checkpoint_file path/to/checkpoint_results.pkl
    
    # Or with custom parameters:
    python run_comprehensive_polytope_analysis.py \
        --checkpoint_file path/to/checkpoint_results.pkl \
        --output_dir cache/custom_analysis \
        --min_samples 10 \
        --visualize
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any, List
import pickle
import pandas as pd

# Add the polytope module to path
sys.path.append(str(Path(__file__).parent))

from polytope.polytope_analyzer import run_multi_checkpoint_analysis, MultiCheckpointPolytopeAnalyzer
from polytope.checkpoint_analysis import (
    load_activation_records, 
    analyze_activation_patterns,
    organize_records_by_key
)


def validate_checkpoint_data(checkpoint_file: str) -> Dict[str, Any]:
    """Validate and analyze checkpoint data structure"""
    
    print(f"Loading and validating checkpoint data from: {checkpoint_file}")
    
    try:
        with open(checkpoint_file, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        raise ValueError(f"Failed to load checkpoint file {checkpoint_file}: {e}")
    
    # Handle both formats: wrapped dict or plain list
    if isinstance(data, list):
        # Plain list of records - create wrapper structure
        records = data
        
        # Infer metadata from records
        checkpoints = set()
        layers = set()
        for record in records:
            checkpoints.add(record.get('checkpoint_step'))
            layers.add(record.get('layer'))
        
        metadata = {
            'model_name': 'Unknown',
            'checkpoints': sorted(list(checkpoints)),
            'target_layers': sorted(list(layers)),
            'n_total_records': len(records),
            'format': 'plain_list'
        }
        
        data = {'records': records, 'metadata': metadata}
        print("✓ Adapted plain list format to expected structure")
        
    elif isinstance(data, dict) and 'records' in data and 'metadata' in data:
        # Already in expected format
        records = data['records']
        metadata = data['metadata']
        
    else:
        raise ValueError("Checkpoint file must contain either a list of records or a dict with 'records' and 'metadata' keys")
    
    records = data['records']
    metadata = data['metadata']
    
    print(f"✓ Loaded {len(records):,} activation records")
    print(f"✓ Model: {metadata.get('model_name', 'Unknown')}")
    print(f"✓ Checkpoints: {metadata.get('checkpoints', [])}")
    print(f"✓ Target layers: {metadata.get('target_layers', [])}")
    
    # Analyze data distribution
    if records:
        # Check frequency categories
        categories = [r.get('category', 'unknown') for r in records]
        category_counts = pd.Series(categories).value_counts()
        print(f"✓ Frequency categories: {dict(category_counts)}")
        
        # Check checkpoint distribution
        checkpoints = [r.get('checkpoint_step', 'unknown') for r in records]
        checkpoint_counts = pd.Series(checkpoints).value_counts()
        print(f"✓ Records per checkpoint: {dict(checkpoint_counts)}")
        
        # Check layer distribution
        layers = [r.get('layer', 'unknown') for r in records]
        layer_counts = pd.Series(layers).value_counts()
        print(f"✓ Records per layer: {dict(layer_counts)}")
    
    return data


def run_data_quality_checks(data: Dict[str, Any]) -> Dict[str, Any]:
    """Run data quality checks and report potential issues"""
    
    print("\n" + "="*60)
    print("DATA QUALITY ASSESSMENT")
    print("="*60)
    
    records = data['records']
    metadata = data['metadata']
    
    quality_report = {
        'total_records': len(records),
        'checkpoints': set(),
        'layers': set(),
        'categories': set(),
        'issues': [],
        'recommendations': []
    }
    
    # Collect basic statistics
    for record in records:
        quality_report['checkpoints'].add(record.get('checkpoint_step'))
        quality_report['layers'].add(record.get('layer'))
        quality_report['categories'].add(record.get('category'))
    
    # Check for missing frequency categories
    unknown_categories = sum(1 for r in records if r.get('category') == 'unknown')
    if unknown_categories > 0:
        pct_unknown = (unknown_categories / len(records)) * 100
        quality_report['issues'].append(f"{unknown_categories} records ({pct_unknown:.1f}%) have unknown frequency category")
        if pct_unknown > 20:
            quality_report['recommendations'].append("Consider re-running checkpoint analysis with proper frequency categorization")
    
    # Check for balanced frequency distribution
    category_counts = pd.Series([r.get('category', 'unknown') for r in records]).value_counts()
    freq_categories = ['high_freq', 'low_freq', 'high', 'low']  # Support both naming conventions
    found_categories = [cat for cat in freq_categories if cat in category_counts]
    
    if len(found_categories) >= 2:
        counts = [category_counts[cat] for cat in found_categories[:2]]
        ratio = max(counts) / min(counts) if min(counts) > 0 else float('inf')
        if ratio > 3.0:
            quality_report['issues'].append(f"Imbalanced frequency categories (ratio: {ratio:.1f}:1)")
            quality_report['recommendations'].append("Consider balancing high/low frequency samples for more robust analysis")
    
    # Check activation vector dimensions
    activation_dims = set()
    for record in records[:100]:  # Sample first 100 records
        if 'activation_vector' in record and hasattr(record['activation_vector'], 'shape'):
            activation_dims.add(record['activation_vector'].shape[0])
    
    if len(activation_dims) > 1:
        quality_report['issues'].append(f"Inconsistent activation vector dimensions: {activation_dims}")
        quality_report['recommendations'].append("Verify that all records use the same model architecture")
    
    # Print quality report
    print(f"Total records: {quality_report['total_records']:,}")
    print(f"Unique checkpoints: {len(quality_report['checkpoints'])}")
    print(f"Unique layers: {len(quality_report['layers'])}")  
    print(f"Unique categories: {len(quality_report['categories'])}")
    
    if quality_report['issues']:
        print(f"\n⚠️  ISSUES DETECTED ({len(quality_report['issues'])}):")
        for i, issue in enumerate(quality_report['issues'], 1):
            print(f"  {i}. {issue}")
    
    if quality_report['recommendations']:
        print(f"\n💡 RECOMMENDATIONS ({len(quality_report['recommendations'])}):")
        for i, rec in enumerate(quality_report['recommendations'], 1):
            print(f"  {i}. {rec}")
    
    if not quality_report['issues']:
        print("✅ No major data quality issues detected")
    
    return quality_report


def filter_records_for_analysis(records: List[Dict[str, Any]], 
                                min_samples: int = 5) -> List[Dict[str, Any]]:
    """Filter records to ensure sufficient samples for robust analysis"""
    
    print(f"\nFiltering records for analysis (min {min_samples} samples per group)...")
    
    # Organize by checkpoint, layer, category
    organized = {}
    for record in records:
        checkpoint = record.get('checkpoint_step', 'unknown')
        layer = record.get('layer', 'unknown')
        category = record.get('category', 'unknown')
        
        key = (checkpoint, layer, category)
        if key not in organized:
            organized[key] = []
        organized[key].append(record)
    
    # Filter groups with sufficient samples
    filtered_records = []
    group_stats = {'kept': 0, 'filtered': 0}
    
    # Group by checkpoint-layer pairs
    checkpoint_layer_groups = {}
    for (checkpoint, layer, category), group_records in organized.items():
        cl_key = (checkpoint, layer)
        if cl_key not in checkpoint_layer_groups:
            checkpoint_layer_groups[cl_key] = {}
        checkpoint_layer_groups[cl_key][category] = group_records
    
    for (checkpoint, layer), categories in checkpoint_layer_groups.items():
        # Check if we have both high and low freq with sufficient samples
        # Support both naming conventions
        high_freq = categories.get('high_freq', categories.get('high', []))
        low_freq = categories.get('low_freq', categories.get('low', []))
        
        if len(high_freq) >= min_samples and len(low_freq) >= min_samples:
            filtered_records.extend(high_freq)
            filtered_records.extend(low_freq)
            group_stats['kept'] += 1
        else:
            group_stats['filtered'] += 1
    
    print(f"✓ Kept {group_stats['kept']} checkpoint-layer groups")
    print(f"✓ Filtered {group_stats['filtered']} groups with insufficient samples")
    print(f"✓ Final dataset: {len(filtered_records):,} records")
    
    return filtered_records


def main():
    parser = argparse.ArgumentParser(
        description="Run comprehensive multi-checkpoint polytope analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic analysis
    python run_comprehensive_polytope_analysis.py --checkpoint_file cache/checkpoint_analysis_20250101_120000.pkl
    
    # With custom settings
    python run_comprehensive_polytope_analysis.py \
        --checkpoint_file cache/checkpoint_analysis_20250101_120000.pkl \
        --output_dir cache/my_analysis \
        --min_samples 10 \
        --no_visualize
        """
    )
    
    parser.add_argument(
        '--checkpoint_file',
        type=str,
        required=True,
        help='Path to checkpoint analysis results (.pkl file)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='cache/comprehensive_polytope_analysis',
        help='Output directory for analysis results (default: cache/comprehensive_polytope_analysis)'
    )
    
    parser.add_argument(
        '--min_samples',
        type=int,
        default=5,
        help='Minimum samples required per frequency category (default: 5)'
    )
    
    parser.add_argument(
        '--visualize',
        action='store_true',
        default=True,
        help='Generate visualization plots (default: True)'
    )
    
    parser.add_argument(
        '--no_visualize',
        action='store_true',
        help='Skip visualization generation'
    )
    
    parser.add_argument(
        '--quality_check_only',
        action='store_true',
        help='Only run data quality checks, skip analysis'
    )
    
    args = parser.parse_args()
    
    # Handle visualization flag
    if args.no_visualize:
        args.visualize = False
    
    print("=" * 80)
    print("COMPREHENSIVE MULTI-CHECKPOINT POLYTOPE ANALYSIS")
    print("=" * 80)
    print(f"Checkpoint file: {args.checkpoint_file}")
    print(f"Output directory: {args.output_dir}")
    print(f"Minimum samples: {args.min_samples}")
    print(f"Generate visualizations: {args.visualize}")
    print()
    
    # Validate input file
    if not Path(args.checkpoint_file).exists():
        print(f"❌ Error: Checkpoint file not found: {args.checkpoint_file}")
        return 1
    
    try:
        # Step 1: Load and validate data
        data = validate_checkpoint_data(args.checkpoint_file)
        
        # Step 2: Run quality checks
        quality_report = run_data_quality_checks(data)
        
        # Stop here if only quality check requested
        if args.quality_check_only:
            print("\n✅ Quality check complete. Use --no-quality-check-only to run full analysis.")
            return 0
        
        # Step 3: Filter data for robust analysis
        filtered_records = filter_records_for_analysis(data['records'], args.min_samples)
        
        if len(filtered_records) == 0:
            print("❌ Error: No valid data remaining after filtering")
            print("   Try reducing --min_samples or checking data quality")
            return 1
        
        # Update data with filtered records
        data['records'] = filtered_records
        
        # Save filtered data temporarily
        temp_file = Path(args.output_dir) / "filtered_checkpoint_data.pkl"
        temp_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(temp_file, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"\n✓ Saved filtered data to: {temp_file}")
        
        # Step 4: Run comprehensive polytope analysis
        print("\n" + "="*60)
        print("RUNNING MULTI-CHECKPOINT POLYTOPE ANALYSIS")
        print("="*60)
        
        results = run_multi_checkpoint_analysis(
            checkpoint_file=str(temp_file),
            output_dir=args.output_dir
        )
        
        # Step 5: Generate additional summary
        print("\n" + "="*60)
        print("ANALYSIS SUMMARY")
        print("="*60)
        
        individual_analyses = results.get('individual_analyses', {})
        successful_analyses = 0
        failed_analyses = 0
        
        for checkpoint_data in individual_analyses.values():
            if isinstance(checkpoint_data, dict):
                for layer_data in checkpoint_data.values():
                    if isinstance(layer_data, dict) and 'comparison' in layer_data:
                        successful_analyses += 1
                    else:
                        failed_analyses += 1
        
        print(f"✅ Successful analyses: {successful_analyses}")
        if failed_analyses > 0:
            print(f"⚠️  Failed analyses: {failed_analyses}")
        
        # Print key findings
        evolution_patterns = results.get('evolution_patterns', {})
        if evolution_patterns and 'training_dynamics' in evolution_patterns:
            training_dynamics = evolution_patterns['training_dynamics']
            if training_dynamics:
                print("\n🔍 KEY FINDINGS:")
                
                # Find checkpoint with strongest frequency effects
                max_effects = {}
                for checkpoint, layers in training_dynamics.items():
                    for layer, metrics in layers.items():
                        effect_size = abs(metrics.get('superposition_diff', 0))
                        if checkpoint not in max_effects or effect_size > max_effects[checkpoint]['effect']:
                            max_effects[checkpoint] = {
                                'effect': effect_size,
                                'layer': layer,
                                'metrics': metrics
                            }
                
                # Show top 3 strongest effects
                sorted_effects = sorted(max_effects.items(), 
                                       key=lambda x: x[1]['effect'], 
                                       reverse=True)[:3]
                
                for i, (checkpoint, info) in enumerate(sorted_effects, 1):
                    print(f"  {i}. Checkpoint {checkpoint}, Layer {info['layer']}: "
                          f"superposition diff = {info['metrics']['superposition_diff']:.3f}")
        
        print(f"\n✅ Analysis complete! Results saved to: {args.output_dir}")
        
        # Clean up temporary file
        if temp_file.exists():
            temp_file.unlink()
        
        return 0
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)