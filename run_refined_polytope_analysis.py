#!/usr/bin/env python3
"""
End-to-End Refined Polytope Analysis Pipeline

This script provides a complete pipeline for mechanistic interpretability research
on polytope superposition in LLMs, integrating:

1. Checkpoint analysis with spline code extraction
2. Refined polytope analysis with visualizations  
3. Evolution tracking across training checkpoints
4. Research-grade statistical analysis and reporting

USAGE:
======
python run_refined_polytope_analysis.py --model EleutherAI/pythia-6.9b --checkpoints 1000,2000,4000 --dataset polytope/country_capital_polytope_dataset.json

Or run interactively:
python run_refined_polytope_analysis.py --interactive
"""

import argparse
from datetime import datetime
import json
from pathlib import Path
from typing import List

from loguru import logger

# Local imports
from polytope.checkpoint_analysis import (
    _get_optimal_layers_for_model,
    extract_activations_from_dataset,
    validate_model_for_analysis,
)
from polytope.refined_polytope_analyzer import RefinedPolytopeAnalyzer


def setup_logging(output_dir: str) -> None:
    """Setup comprehensive logging for the pipeline."""
    log_dir = Path(output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"polytope_pipeline_{timestamp}.log"
    
    logger.add(
        log_file,
        rotation="50 MB",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name} | {message}",
        backtrace=True,
        diagnose=True
    )
    
    # Also log to console
    logger.add(
        lambda msg: print(msg, end=""),
        level="INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | {message}",
        colorize=True
    )
    
    logger.info(f"Logging initialized. Log file: {log_file}")


def run_checkpoint_analysis(model_name: str, 
                          checkpoints: List[str],
                          dataset_path: str,
                          target_layers: List[int] = None,
                          output_dir: str = "cache/refined_analysis") -> str:
    """
    Run checkpoint analysis to extract activations and spline codes.
    
    Returns:
        Path to checkpoint analysis results file
    """
    logger.info(f"Starting checkpoint analysis for {model_name}")
    logger.info(f"Checkpoints: {checkpoints}")
    logger.info(f"Dataset: {dataset_path}")
    
    # Load dataset
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    
    # Get optimal layers if not specified
    if target_layers is None:
        target_layers = _get_optimal_layers_for_model(model_name)
        logger.info(f"Using optimal layers: {target_layers}")
    
    # Validate model
    validation = validate_model_for_analysis(model_name)
    logger.info(f"Model validation: {validation}")
    
    # Extract activations with spline codes
    logger.info("Extracting activations and spline codes...")
    records = extract_activations_from_dataset(
        model_name=model_name,
        checkpoints=checkpoints,
        dataset=dataset,
        target_layers=target_layers,
        activation_strategy="single",
        batch_size=16  # Conservative batch size for stability
    )
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_path / f"checkpoint_analysis_{timestamp}.pkl"
    
    analysis_metadata = {
        'model_name': model_name,
        'checkpoints': checkpoints,
        'dataset_path': dataset_path,
        'target_layers': target_layers,
        'n_total_records': len(records),
        'timestamp': timestamp,
        'spline_codes_extracted': True
    }
    
    # Save with metadata
    import pickle
    save_data = {
        'records': records,
        'metadata': analysis_metadata
    }
    
    with open(results_file, 'wb') as f:
        pickle.dump(save_data, f)
    
    logger.info(f"Checkpoint analysis complete. {len(records)} records saved to {results_file}")
    return str(results_file)


def run_polytope_analysis(checkpoint_file: str, output_dir: str = "cache/polytope_results") -> str:
    """
    Run refined polytope analysis on checkpoint results.
    
    Returns:
        Path to polytope analysis results directory
    """
    logger.info(f"Starting polytope analysis on {checkpoint_file}")
    
    analyzer = RefinedPolytopeAnalyzer(cache_dir=output_dir)
    results_dir = analyzer.run_full_analysis(checkpoint_file, output_dir)
    
    logger.info(f"Polytope analysis complete. Results in {results_dir}")
    return results_dir


def interactive_mode():
    """Interactive mode for exploring and configuring analysis."""
    print("\n" + "="*60)
    print("   REFINED POLYTOPE ANALYSIS - INTERACTIVE MODE")
    print("="*60)
    
    # Model selection
    print("\nAvailable models:")
    models = [
        "EleutherAI/pythia-70m",
        "EleutherAI/pythia-160m", 
        "EleutherAI/pythia-410m",
        "EleutherAI/pythia-1b",
        "EleutherAI/pythia-1.4b",
        "EleutherAI/pythia-2.8b",
        "EleutherAI/pythia-6.9b"
    ]
    
    for i, model in enumerate(models, 1):
        print(f"  {i}. {model}")
    
    while True:
        try:
            choice = int(input(f"\nSelect model (1-{len(models)}): ")) - 1
            if 0 <= choice < len(models):
                model_name = models[choice]
                break
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Please enter a number.")
    
    # Checkpoint selection
    print(f"\nSelected model: {model_name}")
    print("Checkpoint suggestions:")
    print("  - Early training: 1, 2, 4, 8, 16, 32")  
    print("  - Mid training: 1000, 2000, 4000, 8000")
    print("  - Late training: 16000, 32000, 64000, 128000")
    
    checkpoints_input = input("\nEnter checkpoints (comma-separated): ").strip()
    checkpoints = [s.strip() for s in checkpoints_input.split(',') if s.strip()]
    
    # Dataset selection
    print("\nAvailable datasets:")
    datasets = list(Path("polytope").glob("*dataset*.json"))
    for i, dataset in enumerate(datasets, 1):
        print(f"  {i}. {dataset}")
    
    while True:
        try:
            choice = int(input(f"\nSelect dataset (1-{len(datasets)}): ")) - 1
            if 0 <= choice < len(datasets):
                dataset_path = str(datasets[choice])
                break
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Please enter a number.")
    
    # Output directory
    output_dir = input(f"\nOutput directory [cache/refined_analysis_{datetime.now().strftime('%Y%m%d')}]: ").strip()
    if not output_dir:
        output_dir = f"cache/refined_analysis_{datetime.now().strftime('%Y%m%d')}"
    
    # Confirmation
    print("\n" + "="*60)
    print("ANALYSIS CONFIGURATION:")
    print(f"Model: {model_name}")
    print(f"Checkpoints: {checkpoints}")
    print(f"Dataset: {dataset_path}")
    print(f"Output: {output_dir}")
    print("="*60)
    
    confirm = input("\nProceed with analysis? (y/N): ").strip().lower()
    if confirm != 'y':
        print("Analysis cancelled.")
        return
    
    return run_full_pipeline(model_name, checkpoints, dataset_path, output_dir)


def run_full_pipeline(model_name: str, 
                     checkpoints: List[str], 
                     dataset_path: str, 
                     output_dir: str) -> str:
    """Run the complete analysis pipeline."""
    
    # Setup logging
    setup_logging(output_dir)
    
    logger.info("Starting refined polytope analysis pipeline")
    logger.info(f"Model: {model_name}")
    logger.info(f"Checkpoints: {checkpoints}")
    logger.info(f"Dataset: {dataset_path}")
    logger.info(f"Output: {output_dir}")
    
    try:
        # Stage 1: Checkpoint analysis
        logger.info("STAGE 1: Checkpoint Analysis")
        checkpoint_file = run_checkpoint_analysis(
            model_name=model_name,
            checkpoints=checkpoints,
            dataset_path=dataset_path,
            output_dir=output_dir
        )
        
        # Stage 2: Polytope analysis  
        logger.info("STAGE 2: Polytope Analysis")
        results_dir = run_polytope_analysis(
            checkpoint_file=checkpoint_file,
            output_dir=output_dir
        )
        
        # Stage 3: Generate final report
        logger.info("STAGE 3: Final Report Generation")
        
        final_report = f"""
REFINED POLYTOPE ANALYSIS COMPLETE
=====================================

Analysis Configuration:
- Model: {model_name}
- Checkpoints analyzed: {len(checkpoints)}
- Dataset: {dataset_path}
- Output directory: {output_dir}

Results Available:
- Checkpoint analysis: {checkpoint_file}
- Polytope analysis: {results_dir}
- Visualizations: {results_dir}/polytope_evolution.png
- Summary report: {results_dir}/analysis_summary.txt
- Detailed logs: {output_dir}/logs/

Research Insights:
- Spline code analysis for polytope structure
- Evolution tracking across training checkpoints  
- High vs low frequency n-gram comparisons
- Statistical significance testing
- Publication-ready visualizations

Next Steps:
1. Review visualizations for evolution patterns
2. Examine statistical tests in summary report  
3. Analyze spline code phrase index for interpretability
4. Consider extending analysis to additional layers/checkpoints

Analysis completed successfully at {datetime.now().isoformat()}
"""
        
        report_file = Path(output_dir) / "ANALYSIS_COMPLETE.txt"
        with open(report_file, 'w') as f:
            f.write(final_report)
        
        logger.info("Pipeline complete!")
        print(final_report)
        
        return str(results_dir)
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description="Refined Polytope Analysis Pipeline")
    parser.add_argument("--model", type=str, help="Model name (e.g., EleutherAI/pythia-6.9b)")
    parser.add_argument("--checkpoints", type=str, help="Comma-separated checkpoint steps")
    parser.add_argument("--dataset", type=str, help="Path to dataset JSON file")
    parser.add_argument("--output", type=str, help="Output directory")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_mode()
    elif args.model and args.checkpoints and args.dataset:
        checkpoints = [s.strip() for s in args.checkpoints.split(',')]
        output_dir = args.output or f"cache/refined_analysis_{datetime.now().strftime('%Y%m%d')}"
        
        run_full_pipeline(args.model, checkpoints, args.dataset, output_dir)
    else:
        print("Please provide --model, --checkpoints, and --dataset, or use --interactive")
        parser.print_help()


if __name__ == "__main__":
    main()