#!/usr/bin/env python3
"""
Network Storage Checkpoint Analysis - Crash-Resistant Runner

This script saves activation records incrementally to network storage,
allowing you to resume from where you left off if the process crashes.
"""

import json
import pickle
import sys
from pathlib import Path
from datetime import datetime

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from checkpoint_analysis import (
    extract_activations_from_dataset_with_network_storage,
    load_completed_checkpoints
)

def run_network_storage_analysis(network_storage_path: str = "/workspace/activation_cache",
                                start_checkpoint: int = 1000,
                                end_checkpoint: int = 144000,
                                step_size: int = 1000,
                                resume: bool = True):
    """
    Run checkpoint analysis with network storage backup
    
    Args:
        network_storage_path: Path to network storage (e.g., /workspace/activation_cache)
        start_checkpoint: Starting checkpoint (default: 1000)
        end_checkpoint: Ending checkpoint (default: 144000)  
        step_size: Step size between checkpoints (default: 1000)
        resume: Whether to resume from existing progress
    """
    
    model_name = "EleutherAI/pythia-6.9b"
    dataset_path = "polytope/country_capital_polytope_dataset.json"
    
    # Generate checkpoint list
    checkpoints = [str(i) for i in range(start_checkpoint, end_checkpoint, step_size)]
    
    print(f"🚀 Starting Network Storage Analysis")
    print(f"Model: {model_name}")
    print(f"Dataset: {dataset_path}")
    print(f"Network storage: {network_storage_path}")
    print(f"Checkpoints: {len(checkpoints)} total ({start_checkpoint} to {end_checkpoint-step_size})")
    print(f"Resume mode: {'ON' if resume else 'OFF'}")
    
    # Check if resuming
    if resume:
        completed_checkpoints, existing_records = load_completed_checkpoints(
            network_storage_path, model_name
        )
        if completed_checkpoints:
            print(f"🔄 Found existing progress:")
            print(f"   • {len(completed_checkpoints)} checkpoints completed")
            print(f"   • {len(existing_records)} records already extracted")
            print(f"   • Last checkpoint: {max(completed_checkpoints) if completed_checkpoints else 'None'}")
    
    # Load dataset
    try:
        with open(dataset_path, 'r') as f:
            dataset = json.load(f)
        print(f"📊 Dataset loaded: {len(dataset['data'])} samples")
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return
    
    try:
        # Run optimized extraction with network storage
        all_records = extract_activations_from_dataset_with_network_storage(
            model_name=model_name,
            checkpoints=checkpoints,
            dataset=dataset,
            network_storage_path=network_storage_path,
            target_layers=None,  # Use optimal layers for Pythia 6.9B
            activation_strategy="single",
            batch_size=32,  # Larger batches for speed
            resume=resume
        )
        
        print(f"\n🎉 SUCCESS! Extracted {len(all_records)} total activation records")
        print(f"💾 All data saved to network storage: {network_storage_path}")
        
        return all_records
        
    except KeyboardInterrupt:
        print(f"\n⚠️ Analysis interrupted by user")
        print(f"💾 Progress automatically saved to network storage")
        print(f"🔄 Run again with resume=True to continue from where you left off")
        
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        print(f"💾 Progress saved to network storage up to last successful checkpoint")
        print(f"🔄 Run again with resume=True to continue from where you left off")

def check_progress(network_storage_path: str = "/workspace/activation_cache",
                  model_name: str = "EleutherAI/pythia-6.9b"):
    """Check current progress in network storage"""
    
    completed_checkpoints, existing_records = load_completed_checkpoints(
        network_storage_path, model_name
    )
    
    if not completed_checkpoints:
        print("❌ No progress found in network storage")
        return
    
    print(f"📊 Progress Report:")
    print(f"   • Model: {model_name}")
    print(f"   • Completed checkpoints: {len(completed_checkpoints)}")
    print(f"   • Total records: {len(existing_records)}")
    print(f"   • Last checkpoint: {max(completed_checkpoints)}")
    print(f"   • Network storage: {network_storage_path}")
    
    # Show recent checkpoints
    recent = sorted(completed_checkpoints, key=int)[-10:]
    print(f"   • Recent checkpoints: {recent}")
    
    return completed_checkpoints, existing_records

def combine_network_storage_results(network_storage_path: str = "/workspace/activation_cache",
                                   model_name: str = "EleutherAI/pythia-6.9b",
                                   output_path: str = "cache/final_activation_records.pkl"):
    """Combine all network storage results into a single file"""
    
    completed_checkpoints, all_records = load_completed_checkpoints(
        network_storage_path, model_name
    )
    
    if not all_records:
        print("❌ No records found in network storage")
        return
    
    # Save combined results
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    final_data = {
        'records': all_records,
        'metadata': {
            'model_name': model_name,
            'completed_checkpoints': completed_checkpoints,
            'total_records': len(all_records),
            'network_storage_path': network_storage_path,
            'export_timestamp': datetime.now().isoformat()
        }
    }
    
    with open(output_file, 'wb') as f:
        pickle.dump(final_data, f)
    
    print(f"✅ Combined {len(all_records)} records from {len(completed_checkpoints)} checkpoints")
    print(f"💾 Saved to: {output_path}")
    
    return output_path

if __name__ == "__main__":
    import sys
    
    # Default network storage path (modify this to your network storage location)
    NETWORK_STORAGE = "/workspace/activation_cache"  # Change this to your network storage path
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "check":
            # Check current progress
            check_progress(NETWORK_STORAGE)
            
        elif command == "resume":
            # Resume from where we left off
            print("🔄 Resuming analysis from network storage...")
            run_network_storage_analysis(NETWORK_STORAGE, resume=True)
            
        elif command == "restart":
            # Start fresh (ignoring existing progress)
            print("🆕 Starting fresh analysis...")
            run_network_storage_analysis(NETWORK_STORAGE, resume=False)
            
        elif command == "combine":
            # Combine all results into final file
            print("🔗 Combining all network storage results...")
            combine_network_storage_results(NETWORK_STORAGE)
            
        else:
            print(f"❌ Unknown command: {command}")
            print("Available commands: check, resume, restart, combine")
    else:
        # Default: run with resume
        print("🚀 Running analysis with auto-resume...")
        run_network_storage_analysis(NETWORK_STORAGE, resume=True)
