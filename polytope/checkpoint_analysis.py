#!/usr/bin/env python3
"""
Optimized Checkpoint Analysis for Polytope Evolution
Enhanced functions for extracting position-based activations from model checkpoints
using pre-computed token positions and tokenized text from dataset.json
"""

import torch
import numpy as np
import pandas as pd
from nnsight import LanguageModel
from typing import List, Dict, Any, Tuple, Optional, Union
import json
from tqdm import tqdm
from collections import defaultdict
from pathlib import Path
import pickle
from datetime import datetime

def compute_cett_threshold(activation_vector: np.ndarray, 
                          target_cett: float = 0.01) -> float:
    """
    Compute CETT-based threshold for activation vector
    
    Args:
        activation_vector: Neural activation vector
        target_cett: Target CETT value (0.01 = 1% error tolerance)
    
    Returns:
        Optimal threshold value
    """
    # Sort activations by magnitude
    magnitudes = np.abs(activation_vector)
    sorted_indices = np.argsort(magnitudes)
    sorted_magnitudes = magnitudes[sorted_indices]
    
    # Compute FFN total output norm (denominator)
    total_norm = np.linalg.norm(activation_vector)
    
    if total_norm == 0:
        return 0.0
    
    # Binary search for optimal threshold
    left, right = 0, len(sorted_magnitudes) - 1
    best_threshold = 0.0
    
    while left <= right:
        mid = (left + right) // 2
        threshold = sorted_magnitudes[mid]
        
        # Compute CETT for this threshold
        below_threshold_mask = magnitudes < threshold
        tail_activations = activation_vector * below_threshold_mask
        tail_norm = np.linalg.norm(tail_activations)
        
        current_cett = tail_norm / total_norm
        
        if current_cett <= target_cett:
            best_threshold = threshold
            left = mid + 1
        else:
            right = mid - 1
    
    return best_threshold

def create_activation_record(checkpoint_step: str,
                           sample_idx: int,
                           phrase: str,
                           sentence: str,
                           frequency_category: str,
                           layer: int,
                           activation_target_idx: int,
                           phrase_start_idx: int,
                           phrase_end_idx: int,
                           activation_vector: np.ndarray,
                           neuron_idx: Optional[int] = None,
                           spline_code: Optional[np.ndarray] = None,
                           **kwargs) -> Dict[str, Any]:
    """Create a structured activation record as dictionary"""
    
    # binary pattern based on CETT-PPL-1% threshold
    cett_threshold = compute_cett_threshold(activation_vector, 0.01)
    binary_pattern = (np.abs(activation_vector) > cett_threshold).astype(int)
    sparsity = np.sum(binary_pattern) / len(binary_pattern)
    activation_norm = np.linalg.norm(activation_vector)
    n_active_neurons = np.sum(binary_pattern)
    
    record = {
        'checkpoint_step': checkpoint_step,
        'sample_idx': sample_idx,
        'phrase': phrase,
        'sentence': sentence,
        'frequency_category': frequency_category,
        'layer': layer,
        'neuron_idx': neuron_idx,
        'activation_target_idx': activation_target_idx,
        'phrase_start_idx': phrase_start_idx,
        'phrase_end_idx': phrase_end_idx,
        'activation_vector': activation_vector,
        'binary_pattern': binary_pattern,
        'sparsity': sparsity,
        'activation_norm': activation_norm,
        'n_active_neurons': n_active_neurons,
        'metadata': kwargs
    }
    if spline_code is not None:
        record['spline_code'] = spline_code
    return record

def organize_records_by_key(records: List[Dict[str, Any]], key: str) -> Dict[Any, List[Dict[str, Any]]]:
    """Organize records by a specific key"""
    organized = defaultdict(list)
    for record in records:
        organized[record[key]].append(record)
    return dict(organized)

def get_activation_matrix(records: List[Dict[str, Any]]) -> np.ndarray:
    """Extract activation matrix from records"""
    if not records:
        return np.array([])
    return np.stack([r['activation_vector'] for r in records])

def convert_records_to_dataframe(records: List[Dict[str, Any]]) -> pd.DataFrame:
    """Convert records to pandas DataFrame"""
    if not records:
        return pd.DataFrame()
    
    # Extract non-array fields for DataFrame
    df_data = []
    for record in records:
        row = {k: v for k, v in record.items() 
               if k not in ['activation_vector', 'binary_pattern', 'metadata']}
        # Add select metadata fields
        if 'metadata' in record:
            for meta_key, meta_val in record['metadata'].items():
                row[f'meta_{meta_key}'] = meta_val
        df_data.append(row)
    
    return pd.DataFrame(df_data)

def validate_dataset_structure(dataset: Dict[str, Any]) -> bool:
    """
    Validate that dataset has the expected structure for checkpoint analysis
    
    Args:
        dataset: Dataset dictionary to validate
        
    Returns:
        True if valid, raises ValueError if invalid
    """
    if not isinstance(dataset, dict):
        raise ValueError("Dataset must be a dictionary")
    
    if 'data' not in dataset:
        raise ValueError("Dataset must contain 'data' key")
    
    if not isinstance(dataset['data'], list):
        raise ValueError("Dataset 'data' must be a list")
    
    if len(dataset['data']) == 0:
        raise ValueError("Dataset 'data' cannot be empty")
    
    # Check first sample for required fields
    sample = dataset['data'][0]
    required_fields = [
        'phrase', 'sentence', 'frequency_category', 
        'token_ids_sentence', 'activation_target_idx',
        'phrase_start_idx', 'phrase_end_idx'
    ]
    
    missing_fields = [field for field in required_fields if field not in sample]
    if missing_fields:
        raise ValueError(f"Sample missing required fields: {missing_fields}")
    
    # Validate data types
    if not isinstance(sample['token_ids_sentence'], list):
        raise ValueError("token_ids_sentence must be a list")
    
    if not isinstance(sample['activation_target_idx'], int):
        raise ValueError("activation_target_idx must be an integer")
    
    if not isinstance(sample['phrase_start_idx'], int):
        raise ValueError("phrase_start_idx must be an integer")
    
    if not isinstance(sample['phrase_end_idx'], int):
        raise ValueError("phrase_end_idx must be an integer")
    
    return True

def _get_model_layers(model: LanguageModel) -> Any:
    """Get the layers attribute for different model architectures"""
    # Try different model architectures
    for attr_path in ['gpt_neox.layers', 'transformer.h', 'model.layers', 'layers']:
        try:
            layers = model
            for attr in attr_path.split('.'):
                layers = getattr(layers, attr)
            return layers
        except AttributeError:
            continue
    
    raise RuntimeError(f"Could not find layers in model architecture. "
                      f"Model type: {type(model).__name__}")

def extract_layer_activations(model: LanguageModel,
                            token_ids: List[int],
                            layer: int,
                            position: int = -1,
                            strategy: str = "single",
                            capture_preactivation: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Extract activations from a specific layer and position using pre-tokenized input
    
    Args:
        model: nnsight LanguageModel
        token_ids: Pre-tokenized input token IDs
        layer: Layer index
        position: Token position (-1 for last token)
        strategy: Extraction strategy ('single', 'last', 'mean')
                 - 'single': Use specific position
                 - 'last': Always use last token
                 - 'mean': Average across all tokens
        capture_preactivation: Whether to capture pre-activation sign patterns
        
    Returns:
        If capture_preactivation is False: activation vector as numpy array
        If True: tuple (post_activation, pre_activation_sign_code)
    """
    try:
        # Set pad token if not set
        if model.tokenizer.pad_token is None:
            model.tokenizer.pad_token = model.tokenizer.eos_token
            
        # Convert token IDs to tensor
        inputs = {"input_ids": torch.tensor([token_ids], device=model.device)}
        
        with model.trace(inputs):
            # Forward pass
            _ = model(**inputs)
            
            # Get layer activations - architecture agnostic
            model_layers = _get_model_layers(model)
            layer_activations = model_layers[layer].output[0]
            
            # Try to obtain pre-activation for common activation modules if requested
            pre_code = None
            if capture_preactivation:
                try:
                    # Common patterns: model.layers[L].mlp.act or .mlp.activation
                    pre_mod = None
                    for act_attr in ['mlp.act', 'mlp.activation', 'feed_forward.act', 'ff.act', 'ff.activation']:
                        cur = model_layers[layer]
                        ok = True
                        for part in act_attr.split('.'):
                            if hasattr(cur, part):
                                cur = getattr(cur, part)
                            else:
                                ok = False
                                break
                        if ok:
                            pre_mod = cur
                            break
                    if pre_mod is not None and hasattr(pre_mod, 'input'):
                        # pre_mod.input is a tuple; take first tensor [batch, pos, dim]
                        pre_act_tensor = pre_mod.input[0][0]  # shape: [seq, dim]
                        if strategy == "mean":
                            pre_vec = pre_act_tensor.mean(dim=0).save()
                        elif strategy == "last" or position == -1:
                            pre_vec = pre_act_tensor[-1, :].save()
                        else:
                            pre_vec = pre_act_tensor[position, :].save()
                        pre_code = (pre_vec.detach().cpu().numpy() > 0).astype(int)
                except Exception:
                    pre_code = None
            
            # Extract activation based on strategy
            if strategy == "mean":
                # Average across all token positions
                activation_vector = layer_activations[0, :, :].mean(dim=0).save()
            elif strategy == "last":
                # Always use last token
                activation_vector = layer_activations[0, -1, :].save()
            else:  # "single" strategy
                # Use specific position
                if position == -1:
                    position = layer_activations.shape[1] - 1
                activation_vector = layer_activations[0, position, :].save()
        
        # Access the saved activation vector after trace execution
        act_np = activation_vector.detach().cpu().numpy()
        if capture_preactivation:
            return act_np, (pre_code if pre_code is not None else None)
        return act_np
        
    except Exception as e:
        raise RuntimeError(f"Failed to extract activations from layer {layer}: {str(e)}. " +
                         "Cannot proceed without valid activation data.")

def extract_activations_for_sample(model: LanguageModel,
                                 model_name: str,
                                 checkpoint: str,
                                 sample: Dict[str, Any],
                                 target_layers: List[int],
                                 sample_idx: int = 0,
                                 activation_strategy: str = "single",
                                 capture_preactivation: bool = False) -> List[Dict[str, Any]]:
    """
    Extract activations for a specific sample using pre-computed token positions
    
    Args:
        model: Already loaded LanguageModel
        model_name: Model identifier
        checkpoint: Checkpoint step (for record keeping)
        sample: Sample from dataset with pre-computed token positions
        target_layers: List of layer indices
        sample_idx: Sample index for tracking
        activation_strategy: Strategy for activation extraction
        capture_preactivation: Whether to capture pre-activation patterns
        
    Returns:
        List of activation records
    """
    
    records = []
    
    try:
        # Extract pre-computed information from sample
        phrase = sample['phrase']
        sentence = sample['sentence']
        frequency_category = sample['frequency_category']
        token_ids_sentence = sample['token_ids_sentence']
        activation_target_idx = sample['activation_target_idx']
        phrase_start_idx = sample['phrase_start_idx']
        phrase_end_idx = sample['phrase_end_idx']
        
        # Extract activations from each target layer
        for layer in target_layers:
            try:
                if capture_preactivation:
                    out = extract_layer_activations(
                        model, token_ids_sentence, layer, 
                        activation_target_idx, activation_strategy, 
                        capture_preactivation=True
                    )
                    if isinstance(out, tuple):
                        activation_vector, pre_code = out
                    else:
                        activation_vector, pre_code = out, None
                else:
                    activation_vector = extract_layer_activations(
                        model, token_ids_sentence, layer, 
                        activation_target_idx, activation_strategy
                    )
                    pre_code = None
                
                if len(activation_vector) > 0:
                    record = create_activation_record(
                        checkpoint_step=checkpoint,
                        sample_idx=sample_idx,
                        phrase=phrase,
                        sentence=sentence,
                        frequency_category=frequency_category,
                        layer=layer,
                        activation_target_idx=activation_target_idx,
                        phrase_start_idx=phrase_start_idx,
                        phrase_end_idx=phrase_end_idx,
                        activation_vector=activation_vector,
                        spline_code=pre_code,
                        model_name=model_name
                    )
                    records.append(record)
                    
            except Exception as e:
                print(f"Warning: Failed to extract activations from layer {layer} for sample {sample_idx}: {e}")
                continue
        
    except Exception as e:
        print(f"Warning: Failed to process sample {sample_idx}: {e}")
    
    return records

def extract_activations_from_dataset(model_name: str,
                                   checkpoints: List[str],
                                   dataset: Dict[str, Any],
                                   target_layers: List[int] = None,
                                   activation_strategy: str = "single",
                                   capture_preactivation: bool = False) -> List[Dict[str, Any]]:
    """
    Extract activations from a complete dataset across checkpoints
    Optimized version using pre-computed token positions from dataset.json
    
    Args:
        model_name: Model identifier
        checkpoints: List of checkpoint steps
        dataset: Dataset with 'data' key containing samples with pre-computed token positions
        target_layers: Layer indices to analyze
        activation_strategy: Strategy for activation extraction
        capture_preactivation: Whether to capture pre-activation patterns
        
    Returns:
        List of all activation records
    """
    
    if target_layers is None:
        target_layers = [0, 2, 4, 6, 8, 10, 12]  # Default layers
    
    # Validate dataset structure
    validate_dataset_structure(dataset)
    samples = dataset['data']
    all_records = []
    total_samples = len(samples)
    
    print(f"Extracting activations for {len(checkpoints)} checkpoints")
    print(f"Dataset size: {total_samples} samples")
    print(f"Target layers: {target_layers}")
    
    # Check frequency categories
    if samples and 'frequency_category' in samples[0]:
        category_counts = {}
        for sample in samples:
            cat = sample['frequency_category']
            category_counts[cat] = category_counts.get(cat, 0) + 1
        print(f"Frequency categories: {category_counts}")
    
    for checkpoint_idx, checkpoint in enumerate(checkpoints):
        print(f"\nProcessing checkpoint {checkpoint_idx+1}/{len(checkpoints)}: step{checkpoint}")
        
        checkpoint_records = []
        
        # Load model once per checkpoint (more efficient)
        try:
            print(f"Loading model {model_name} at checkpoint {checkpoint}...")
            model = LanguageModel(model_name, revision=f"step{checkpoint}", device_map='auto')
            
            # Set pad token to avoid warnings
            if model.tokenizer.pad_token is None:
                model.tokenizer.pad_token = model.tokenizer.eos_token
            
            print(f"Model loaded successfully for checkpoint {checkpoint}")
            
            # Process all samples with this model
            for sample_idx in tqdm(range(total_samples), desc=f"Checkpoint {checkpoint}"):
                sample = samples[sample_idx]
                
                # Extract activations for this sample using pre-computed token positions
                sample_records = extract_activations_for_sample(
                    model=model,
                    model_name=model_name,
                    checkpoint=checkpoint,
                    sample=sample,
                    target_layers=target_layers,
                    sample_idx=sample_idx,
                    activation_strategy=activation_strategy,
                    capture_preactivation=capture_preactivation
                )
                
                checkpoint_records.extend(sample_records)
            
            # Clean up model
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error processing checkpoint {checkpoint}: {e}")
            continue
        
        print(f"Extracted {len(checkpoint_records)} activation records for checkpoint {checkpoint}")
        all_records.extend(checkpoint_records)
    
    print(f"\nTotal activation records extracted: {len(all_records)}")
    
    # Print category distribution in results
    if all_records:
        categories = [r['frequency_category'] for r in all_records]
        category_counts = {cat: categories.count(cat) for cat in set(categories)}
        print(f"Final category distribution: {category_counts}")
    
    return all_records

def save_activation_records(records: List[Dict[str, Any]], 
                          output_path: str,
                          format: str = "pickle") -> None:
    """
    Save activation records to file
    
    Args:
        records: List of activation records
        output_path: Output file path
        format: Save format ('pickle', 'csv', 'json')
    """
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == "pickle":
        with open(output_path.with_suffix('.pkl'), 'wb') as f:
            pickle.dump(records, f)
            
    elif format == "csv":
        df = convert_records_to_dataframe(records)
        df.to_csv(output_path.with_suffix('.csv'), index=False)
        
    elif format == "json":
        # Convert numpy arrays to lists for JSON serialization
        json_records = []
        for record in records:
            json_record = record.copy()
            json_record['activation_vector'] = record['activation_vector'].tolist()
            json_record['binary_pattern'] = record['binary_pattern'].tolist()
            if 'spline_code' in record and record['spline_code'] is not None:
                json_record['spline_code'] = np.asarray(record['spline_code']).tolist()
            json_records.append(json_record)
        
        with open(output_path.with_suffix('.json'), 'w') as f:
            json.dump(json_records, f, indent=2)
    
    print(f"Saved {len(records)} records to {output_path}")

def load_activation_records(input_path: str) -> List[Dict[str, Any]]:
    """Load activation records from file"""
    
    input_path = Path(input_path)
    
    if input_path.suffix == '.pkl':
        with open(input_path, 'rb') as f:
            return pickle.load(f)
            
    elif input_path.suffix == '.json':
        with open(input_path, 'r') as f:
            json_records = json.load(f)
        
        # Convert lists back to numpy arrays
        for record in json_records:
            record['activation_vector'] = np.array(record['activation_vector'])
            record['binary_pattern'] = np.array(record['binary_pattern'])
            if 'spline_code' in record and record['spline_code'] is not None:
                record['spline_code'] = np.array(record['spline_code']).astype(int)
        
        return json_records
    
    else:
        raise ValueError(f"Unsupported file format: {input_path.suffix}")

def analyze_activation_patterns(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze patterns in activation records
    
    Args:
        records: List of activation records
        
    Returns:
        Analysis results dictionary
    """
    
    if not records:
        return {'error': 'No records provided'}
    
    df = convert_records_to_dataframe(records)
    
    # Basic statistics
    analysis = {
        'total_records': len(records),
        'unique_checkpoints': df['checkpoint_step'].nunique(),
        'unique_phrases': df['phrase'].nunique(),
        'unique_layers': df['layer'].nunique(),
        'mean_sparsity': df['sparsity'].mean(),
        'std_sparsity': df['sparsity'].std(),
        'mean_activation_norm': df['activation_norm'].mean(),
        'std_activation_norm': df['activation_norm'].std(),
        'mean_active_neurons': df['n_active_neurons'].mean(),
        'std_active_neurons': df['n_active_neurons'].std()
    }
    
    # Layer-wise analysis
    layer_analysis = {}
    for layer in df['layer'].unique():
        layer_df = df[df['layer'] == layer]
        layer_analysis[layer] = {
            'n_records': len(layer_df),
            'mean_sparsity': layer_df['sparsity'].mean(),
            'mean_norm': layer_df['activation_norm'].mean(),
            'mean_active': layer_df['n_active_neurons'].mean()
        }
    
    analysis['layer_analysis'] = layer_analysis
    
    # Category analysis
    if 'frequency_category' in df.columns:
        category_analysis = {}
        for category in df['frequency_category'].unique():
            cat_df = df[df['frequency_category'] == category]
            category_analysis[category] = {
                'n_records': len(cat_df),
                'mean_sparsity': cat_df['sparsity'].mean(),
                'mean_norm': cat_df['activation_norm'].mean()
            }
        analysis['category_analysis'] = category_analysis
    
    return analysis

def run_checkpoint_analysis_pipeline(model_name: str,
                                    checkpoints: List[str],
                                    dataset_path: str = "polytope/dataset.json",
                                    target_layers: List[int] = None,
                                    output_dir: str = "cache/checkpoint_analysis",
                                    activation_strategy: str = "single",
                                    capture_preactivation: bool = False) -> str:
    """
    Complete pipeline for extracting and analyzing checkpoints using dataset.json
    
    Args:
        model_name: Model identifier for nnsight
        checkpoints: List of checkpoint steps to analyze
        dataset_path: Path to dataset.json file
        target_layers: Layers to analyze
        output_dir: Output directory for results
        activation_strategy: Strategy for activation extraction
        capture_preactivation: Whether to capture pre-activation patterns
        
    Returns:
        Path to saved analysis results
    """
    
    # Load dataset
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    
    # Default target layers (typical transformer layers)
    if target_layers is None:
        target_layers = [2, 4, 6, 8, 10, 12]
    
    print(f"=== Checkpoint Analysis Pipeline ===")
    print(f"Model: {model_name}")
    print(f"Checkpoints: {checkpoints}")
    print(f"Dataset: {dataset_path}")
    print(f"Total samples: {dataset.get('total_samples', len(dataset.get('data', [])))}")
    print(f"Categories: {dataset.get('categories', {})}")
    print(f"Target layers: {target_layers}")
    
    # Extract activations
    all_records = extract_activations_from_dataset(
        model_name=model_name,
        checkpoints=checkpoints,
        dataset=dataset,
        target_layers=target_layers,
        activation_strategy=activation_strategy,
        capture_preactivation=capture_preactivation
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
        'n_total_records': len(all_records),
        'timestamp': timestamp,
        'activation_strategy': activation_strategy,
        'capture_preactivation': capture_preactivation
    }
    
    # Save with metadata
    save_data = {
        'records': all_records,
        'metadata': analysis_metadata
    }
    
    with open(results_file, 'wb') as f:
        pickle.dump(save_data, f)
    
    print(f"\n=== Analysis Complete ===")
    print(f"Total records extracted: {len(all_records):,}")
    print(f"Results saved to: {results_file}")
    
    return str(results_file)

def main():
    """Example usage of optimized checkpoint analysis"""
    model_name = "EleutherAI/pythia-70m"
    # all checkpoints 1000 to 143000
    checkpoints = [str(2**i) for i in range(0, 10)] # 1 to 512
    checkpoints.extend([str(i) for i in range(1000, 143000, 1000)])
    print(checkpoints)
    dataset_path = "polytope/dataset.json"
    
    # Load dataset
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    
    print("Dataset structure:")
    print(f"Total samples: {dataset.get('total_samples', len(dataset.get('data', [])))}")
    print(f"Categories: {dataset.get('categories', {})}")
    print(f"Template: {dataset.get('template', 'N/A')}")
    
    # Show sample data structure
    if dataset.get('data'):
        sample = dataset['data'][0]
        print(f"\nSample data keys: {list(sample.keys())}")
        print(f"Sample phrase: {sample['phrase']}")
        print(f"Sample frequency category: {sample['frequency_category']}")
        print(f"Sample activation target idx: {sample['activation_target_idx']}")

    target_layers = [1, 2, 3, 4, 5]
    records = extract_activations_from_dataset(
        model_name=model_name,
        checkpoints=checkpoints,
        dataset=dataset,
        target_layers=target_layers,
        activation_strategy="single"
    )

    print(f"\nExtracted {len(records)} activation records")
    if records:
        print("Sample record categories:", [r['frequency_category'] for r in records[:10]])

    # Analyze patterns
    analysis = analyze_activation_patterns(records)
    print("\nActivation Analysis:")
    print(analysis)

    return records

if __name__ == "__main__":
    records = main()
    save_activation_records(records, 'cache/activation_records.pkl')