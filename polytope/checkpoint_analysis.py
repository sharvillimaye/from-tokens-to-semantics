#!/usr/bin/env python3
"""
Optimized Checkpoint Analysis for Polytope Evolution
Enhanced functions for extracting position-based activations from model checkpoints
using pre-computed token positions and tokenized text from dataset.json
"""
# Dependencies: pip install nnsight pandas torch
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
import os
import shutil

def clear_hf_cache_for_revision(model_id: str, revision: str) -> None:
    """Delete Hugging Face cached snapshot for a specific model revision.

    Removes the snapshot directory referenced by refs/<revision> and the ref file
    itself, allowing space to be reclaimed by the cache GC. Safe to call even if
    the cache paths do not exist.
    """
    try:
        hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
        hub_dir = os.path.join(hf_home, "hub")
        repo_dir = os.path.join(hub_dir, f"models--{model_id.replace('/', '--')}")
        ref_path = os.path.join(repo_dir, "refs", revision)

        if os.path.isfile(ref_path):
            with open(ref_path, "r") as f:
                commit_hash = f.read().strip()
            snap_dir = os.path.join(repo_dir, "snapshots", commit_hash)

            shutil.rmtree(snap_dir, ignore_errors=True)
            try:
                os.remove(ref_path)
            except FileNotFoundError:
                pass
            print(f"Cleared HF cache for {model_id}@{revision} ({commit_hash})")
    except Exception as e:
        print(f"Cache cleanup skipped for {model_id}@{revision}: {e}")

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
                           pre_activation_vector: np.ndarray,
                           activation_type: str = 'unknown',  # Architecture type from extraction
                           neuron_idx: Optional[int] = None,
                           **kwargs) -> Dict[str, Any]:
    """Create a structured activation record with correct spline codes for polytope analysis"""
    
    # Analyze POST-activation vector (main activation vector)
    cett_threshold = compute_cett_threshold(activation_vector, 0.01)
    binary_pattern = (np.abs(activation_vector) > cett_threshold).astype(int)
    sparsity = np.sum(binary_pattern) / len(binary_pattern)
    activation_norm = np.linalg.norm(activation_vector)
    n_active_neurons = np.sum(binary_pattern)
    
    # CORRECTED: Generate proper spline code based on activation function type
    spline_code = None
    pre_activation_analysis = {}
    
    if pre_activation_vector is not None:
        pre_activation_norm = np.linalg.norm(pre_activation_vector)
        
        # Generate spline code based on activation function type
        if activation_type == 'pythia':
            # Pythia uses GELU activation: spline code is sign-based
            spline_code = (pre_activation_vector > 0).astype(int)
            
        elif activation_type in ['olmo', 'llama_like']:
            # OLMo/Llama use SiLU/Swish: spline code is also sign-based  
            spline_code = (pre_activation_vector > 0).astype(int)
            
        else:
            # Default: assume ReLU-like behavior (sign-based)
            spline_code = (pre_activation_vector > 0).astype(int)
        
        pre_n_active_neurons = np.sum(spline_code)
        
        pre_activation_analysis = {
            'pre_activation_vector': pre_activation_vector,
            'pre_activation_norm': pre_activation_norm,
            'pre_n_active_neurons': pre_n_active_neurons,
            'spline_code': spline_code,
            'activation_function_type': activation_type
        }
    
    record = {
        'checkpoint_step': checkpoint_step,
        'sample_idx': sample_idx,
        'phrase': phrase,
        'sentence': sentence,
        'frequency_category': frequency_category,
        # Canonical category label used by downstream analyzers
        'category': (
            'high_freq' if str(frequency_category).strip().lower() in {
                'high_frequency', 'highfreq', 'high-freq', 'high'
            } else 'low_freq' if str(frequency_category).strip().lower() in {
                'low_frequency', 'lowfreq', 'low-freq', 'low'
            } else str(frequency_category)
        ),
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
        'cett_threshold': cett_threshold,

        'spline_code': spline_code,
        
        'metadata': kwargs
    }
    
    # Add pre-activation analysis if available
    record.update(pre_activation_analysis)
    
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


def _get_optimal_layers_for_model(model_name: str, num_layers: Optional[int] = None) -> List[int]:
    """
    Get optimal layer selection for different model sizes, focusing on early, middle, and later layers.
    
    Args:
        model_name: Model identifier (e.g., "EleutherAI/pythia-6.9b")
        num_layers: Number of layers in model (auto-detected if None)
        
    Returns:
        List of layer indices optimized for the model size
    """
    
    # Auto-detect number of layers from model name if not provided
    if num_layers is None:
        model_size_mapping = {
            '70m': 6,      # Pythia 70M has 6 layers
            '160m': 12,    # Pythia 160M has 12 layers  
            '410m': 24,    # Pythia 410M has 24 layers
            '1b': 16,      # Pythia 1B has 16 layers
            '1.4b': 24,    # Pythia 1.4B has 24 layers
            '2.8b': 32,    # Pythia 2.8B has 32 layers
            '6.9b': 32,    # Pythia 6.9B has 32 layers
            '12b': 36,     # Pythia 12B has 36 layers
        }
        
        for size_key, layers in model_size_mapping.items():
            if size_key in model_name.lower():
                num_layers = layers
                break
        
        if num_layers is None:
            # Default fallback for unknown models
            num_layers = 12
            print(f"Warning: Could not auto-detect layers for {model_name}, using default {num_layers}")
    
    # Select layers based on model depth
    if num_layers <= 6:
        # Very small models: analyze most layers
        target_layers = list(range(1, min(num_layers, 6)))
    elif num_layers <= 12:
        # Small models (like 70M, 160M): early, middle, late
        target_layers = [1, 2, num_layers // 3, num_layers // 2, 
                        (2 * num_layers) // 3, num_layers - 2, num_layers - 1]
    elif num_layers <= 24:
        # Medium models (like 410M, 1.4B): strategic sampling
        target_layers = [2, 4, num_layers // 4, num_layers // 3, num_layers // 2,
                        (2 * num_layers) // 3, (3 * num_layers) // 4, num_layers - 3, num_layers - 1]
    else:
        # Large models (like 6.9B with 32 layers): focus on key transition points
        early_layers = [1, 2, 4, 6]                                    # Very early processing
        early_mid = [num_layers // 8, num_layers // 4]                 # Early-middle (layers ~4, 8)
        middle = [num_layers // 2]                                     # True middle (layer 16)
        late_mid = [(3 * num_layers) // 4, (7 * num_layers) // 8]     # Late-middle (layers ~24, 28)
        late_layers = [num_layers - 3, num_layers - 1]                # Final processing
        
        target_layers = early_layers + early_mid + middle + late_mid + late_layers
    
    # Remove duplicates and sort
    target_layers = sorted(list(set([l for l in target_layers if 0 <= l < num_layers])))
    
    print(f"Selected layers for {model_name} ({num_layers} total layers): {target_layers}")
    print(f"  • Early layers: {[l for l in target_layers if l < num_layers // 4]}")
    print(f"  • Middle layers: {[l for l in target_layers if num_layers // 4 <= l < 3 * num_layers // 4]}")  
    print(f"  • Later layers: {[l for l in target_layers if l >= 3 * num_layers // 4]}")
    
    return target_layers


def extract_activations_from_dataset(model_name: str,
                                   checkpoints: List[str],
                                   dataset: Dict[str, Any],
                                   target_layers: List[int] = None,
                                   batch_size: int = 4) -> List[Dict[str, Any]]:
    """
    Extract both pre and post activations from a complete dataset across checkpoints
    
    Args:
        model_name: Model identifier
        checkpoints: List of checkpoint steps
        dataset: Dataset with 'data' key containing samples with pre-computed token positions
        target_layers: Layer indices to analyze
        
    Returns:
        List of all activation records with both pre and post activation data
    """
    
    if target_layers is None:
        target_layers = _get_optimal_layers_for_model(model_name)
    
    # Validate dataset structure
    validate_dataset_structure(dataset)
    samples = dataset['data']
    all_records = []
    total_samples = len(samples)

    def _get_model_architecture_info(model: LanguageModel) -> Dict[str, Any]:
        """Detect model architecture and return access patterns for correct polytope analysis"""
        
        # Debug: Print available attributes
        print(f"Model attributes: {dir(model)}")
        if hasattr(model, 'config'):
            print(f"Model config: {model.config}")
        
        # Pythia (GPT-NeoX based) - GELU activation
        # Try multiple detection methods for Pythia
        try:
            # Method 1: Direct gpt_neox attribute
            if hasattr(model, 'gpt_neox'):
                print("Detected Pythia via gpt_neox attribute")
                return {
                    'type': 'pythia',
                    'layer_access': lambda idx: model.gpt_neox.layers[idx],
                    'mlp_pre_activation': lambda layer: layer.mlp.dense_h_to_4h.output,
                    'mlp_post_activation': lambda layer: layer.mlp.act.output,  
                    'layer_post_activation': lambda layer: layer.output,
                    'activation_function': 'gelu'
                }
        except Exception as e:
            print(f"Method 1 failed: {e}")
        
        # Method 2: Check config for model type
        try:
            if hasattr(model, 'config') and hasattr(model.config, 'model_type'):
                if model.config.model_type == 'gpt_neox':
                    print("Detected Pythia via config.model_type")
                    return {
                        'type': 'pythia',
                        'layer_access': lambda idx: model.gpt_neox.layers[idx],
                        'mlp_pre_activation': lambda layer: layer.mlp.dense_h_to_4h.output,
                        'mlp_post_activation': lambda layer: layer.mlp.act.output,
                        'layer_post_activation': lambda layer: layer.output,
                        'activation_function': 'gelu'
                    }
        except Exception as e:
            print(f"Method 2 failed: {e}")
        
        # Method 3: Check for transformer attribute (alternative structure)
        try:
            if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
                print("Detected GPT-like via transformer.h")
                return {
                    'type': 'gpt_like',
                    'layer_access': lambda idx: model.transformer.h[idx],
                    'mlp_pre_activation': lambda layer: layer.mlp.c_fc.output,
                    'mlp_post_activation': lambda layer: layer.mlp.c_proj.input,  # After activation
                    'layer_post_activation': lambda layer: layer.output,
                    'activation_function': 'gelu'
                }
        except Exception as e:
            print(f"Method 3 failed: {e}")
        
        # OLMo - SwiGLU activation (more complex)
        try:
            if hasattr(model, 'transformer') and hasattr(model.transformer, 'blocks'):
                return {
                    'type': 'olmo',
                    'layer_access': lambda idx: model.transformer.blocks[idx],
                    'mlp_pre_activation': lambda layer: layer.feed_forward.w1.output,
                    'mlp_post_activation': lambda layer: layer.feed_forward.output,
                    'layer_post_activation': lambda layer: layer.output,
                    'activation_function': 'swiglu'
                }
        except Exception as e:
            print(f"OLMo detection failed: {e}")
        
        # Llama-like (SwiGLU architecture)  
        try:
            if hasattr(model, 'layers'):
                return {
                    'type': 'llama_like',
                    'layer_access': lambda idx: model.layers[idx],
                    'mlp_pre_activation': lambda layer: layer.mlp.up_proj.output,
                    'mlp_post_activation': lambda layer: layer.mlp.output,
                    'layer_post_activation': lambda layer: layer.output,
                    'activation_function': 'swiglu'
                }
        except Exception as e:
            print(f"Llama detection failed: {e}")
        
        raise ValueError(f"Unsupported model architecture - could not detect any compatible layer structure")


    def _prepare_batch(batch_samples: List[Dict[str, Any]], tokenizer) -> Tuple[Dict[str, torch.Tensor], List[int]]:
        token_id_batches: List[List[int]] = []
        positions: List[int] = []
        for s in batch_samples:
            token_id_batches.append(s['token_ids_sentence'])
            positions.append(int(s['activation_target_idx']))
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        max_len = max(len(ids) for ids in token_id_batches)
        padded = [ids + [pad_id] * (max_len - len(ids)) for ids in token_id_batches]
        attention_mask = [[1] * len(ids) + [0] * (max_len - len(ids)) for ids in token_id_batches]
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        inputs = {
            'input_ids': torch.tensor(padded, device=device),
            'attention_mask': torch.tensor(attention_mask, device=device)
        }
        return inputs, positions

    for checkpoint in checkpoints:
        checkpoint_records = []
        
        try:
            # Load model for this checkpoint
            model = LanguageModel(
                model_name,
                revision=f"step{checkpoint}",
                device_map='cuda:0',
                torch_dtype=torch.float16
            )
            model.eval()
            
            # Set pad token to avoid warnings
            if model.tokenizer.pad_token is None:
                model.tokenizer.pad_token = model.tokenizer.eos_token
            
            # Get architecture-specific access patterns
            arch_info = _get_model_architecture_info(model)
            print(f"Detected {arch_info['type']} architecture for checkpoint {checkpoint}")
            
            for batch_start in tqdm(range(0, total_samples, max(1, int(batch_size))), 
                                   desc=f"Checkpoint {checkpoint}"):
                batch_end = min(batch_start + max(1, int(batch_size)), total_samples)
                batch_samples = samples[batch_start:batch_end]
                
                inputs, batch_positions = _prepare_batch(batch_samples, model.tokenizer)
                
                # Store handles for activations
                pre_handles: Dict[int, Any] = {}
                post_handles: Dict[int, Any] = {}
                
                with torch.no_grad():
                    with model.trace(inputs):
                        for layer_idx in target_layers:
                            try:
                                # Get the layer using architecture-specific access
                                layer_proxy = arch_info['layer_access'](layer_idx)
                                
                                # Extract pre-activation (before activation function)
                                pre_handles[layer_idx] = arch_info['mlp_pre_activation'](layer_proxy).save()
                                
                                # Extract post-activation (after activation function)
                                post_handles[layer_idx] = arch_info['mlp_post_activation'](layer_proxy).save()
                                
                                
                            except Exception as e:
                                print(f"Warning: Failed to set up extraction for layer {layer_idx}: {e}")
                                continue
                        
                        # Execute the model
                        _ = model(**inputs)
                
                # Process the saved activations
                for i, sample in enumerate(batch_samples):
                    pos = batch_positions[i]
                    sample_idx = batch_start + i
                    
                    # Extract sample metadata
                    phrase = sample['phrase']
                    sentence = sample['sentence']
                    frequency_category = sample['frequency_category']
                    activation_target_idx = sample['activation_target_idx']
                    phrase_start_idx = sample['phrase_start_idx']
                    phrase_end_idx = sample['phrase_end_idx']
                    
                    for layer_idx in target_layers:
                        if layer_idx not in pre_handles or layer_idx not in post_handles:
                            continue
                        
                        try:
                            # Get pre-activation
                            pre_tensor = pre_handles[layer_idx]
                            if pre_tensor.dim() == 2:
                                pre_tensor = pre_tensor.unsqueeze(0)
                            
                            # Get post-activation  
                            post_tensor = post_handles[layer_idx]
                            if post_tensor.dim() == 2:
                                post_tensor = post_tensor.unsqueeze(0)
                            
                            # Ensure position is valid
                            if pos >= pre_tensor.shape[1]:
                                pos = pre_tensor.shape[1] - 1
                            
                            # Extract activations at specific position
                            pre_act = pre_tensor[i, pos, :].detach().cpu().numpy()
                            post_act = post_tensor[i, pos, :].detach().cpu().numpy()
                            
                            # Create activation record
                            record = create_activation_record(
                                checkpoint_step=checkpoint,
                                sample_idx=sample_idx,
                                phrase=phrase,
                                sentence=sentence,
                                frequency_category=frequency_category,
                                layer=layer_idx,
                                activation_target_idx=activation_target_idx,
                                phrase_start_idx=phrase_start_idx,
                                phrase_end_idx=phrase_end_idx,
                                activation_vector=post_act,
                                pre_activation_vector=pre_act,
                                activation_type=arch_info['type'],
                                model_name=model_name
                            )
                            checkpoint_records.append(record)
                            
                        except Exception as e:
                            print(f"Warning: Failed to create record for sample {sample_idx}, layer {layer_idx}: {e}")
                
                # Clear GPU memory after each batch
                del pre_handles, post_handles
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Clean up model
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                import gc
                gc.collect()
            # Remove HF cache for this specific checkpoint revision
            clear_hf_cache_for_revision(model_name, f"step{checkpoint}")
                
        except Exception as e:
            print(f"Error processing checkpoint {checkpoint}: {e}")
            # Clean up even on error
            try:
                if 'model' in locals():
                    del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                import gc
                gc.collect()
                # Remove HF cache for this specific checkpoint revision even on error
                clear_hf_cache_for_revision(model_name, f"step{checkpoint}")
            except:
                pass
            continue
        
        all_records.extend(checkpoint_records)
    
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
            # Handle pre-activation vectors
            if 'pre_activation_vector' in record and record['pre_activation_vector'] is not None:
                json_record['pre_activation_vector'] = record['pre_activation_vector'].tolist()
            if 'pre_binary_pattern' in record and record['pre_binary_pattern'] is not None:
                json_record['pre_binary_pattern'] = record['pre_binary_pattern'].tolist()
            json_records.append(json_record)
        
        with open(output_path.with_suffix('.json'), 'w') as f:
            json.dump(json_records, f, indent=2)
    

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
            # Handle pre-activation vectors
            if 'pre_activation_vector' in record and record['pre_activation_vector'] is not None:
                record['pre_activation_vector'] = np.array(record['pre_activation_vector'])
            if 'pre_binary_pattern' in record and record['pre_binary_pattern'] is not None:
                record['pre_binary_pattern'] = np.array(record['pre_binary_pattern']).astype(int)
        
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

def validate_model_for_analysis(model_name: str) -> Dict[str, Any]:
    """
    Validate model compatibility and provide recommendations for analysis.
    
    Args:
        model_name: Model identifier
        
    Returns:
        Validation results and recommendations
    """
    validation = {
        'is_supported': False,
        'model_family': 'unknown',
        'estimated_layers': 12,
        'activation_function': 'unknown',
        'architecture_path': 'unknown',
        'recommendations': []
    }
    
    model_lower = model_name.lower()
    
    # Check for Pythia models (primary target)
    if 'pythia' in model_lower:
        validation['is_supported'] = True
        validation['model_family'] = 'pythia'
        validation['activation_function'] = 'gelu'
        validation['architecture_path'] = 'gpt_neox.layers'
        
        # Get layer count for Pythia
        size_mapping = {
            '70m': 6, '160m': 12, '410m': 24, '1b': 16, 
            '1.4b': 24, '2.8b': 32, '6.9b': 32, '12b': 36
        }
        
        for size, layers in size_mapping.items():
            if size in model_lower:
                validation['estimated_layers'] = layers
                break
        
        if '6.9b' in model_lower:
            validation['recommendations'].extend([
                "Pythia 6.9B detected - using optimized 32-layer analysis",
                "Will capture early (1,2,4,6), middle (16), and late (24,28,29,31) layers",
                "GELU activation function - spline codes will use zero-crossing boundaries",
                "Pre-activation extraction from mlp.dense_h_to_4h layer"
            ])
    
    # Check for other supported models
    elif any(family in model_lower for family in ['gpt2', 'gpt-2']):
        validation['is_supported'] = True
        validation['model_family'] = 'gpt2'
        validation['activation_function'] = 'gelu'
        validation['architecture_path'] = 'transformer.h'
        validation['recommendations'].append("GPT-2 model detected - using transformer.h layer path")
    
    elif any(family in model_lower for family in ['llama', 'mistral']):
        validation['is_supported'] = True
        validation['model_family'] = 'llama'  
        validation['activation_function'] = 'swish'
        validation['architecture_path'] = 'model.layers'
        validation['recommendations'].append("Llama/Mistral model detected - using model.layers path")
    
    # General warnings and recommendations
    if not validation['is_supported']:
        validation['recommendations'].extend([
            "Unknown model architecture - using fallback detection",
            "Consider adding explicit support for this model family",
            "Pre-activation extraction may need manual verification"
        ])
    
    return validation

def validate_spline_codes(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Validate spline code generation for polytope analysis"""
    
    validation = {
        'total_records': len(records),
        'records_with_spline_codes': 0,
        'unique_spline_codes_per_layer': {},
        'spline_code_statistics': {}
    }
    
    for record in records:
        if 'spline_code' in record and record['spline_code'] is not None:
            validation['records_with_spline_codes'] += 1
            
            layer = record['layer']
            if layer not in validation['unique_spline_codes_per_layer']:
                validation['unique_spline_codes_per_layer'][layer] = set()
            
            # Convert spline code to hashable tuple
            spline_tuple = tuple(record['spline_code'])
            validation['unique_spline_codes_per_layer'][layer].add(spline_tuple)
    
    # Convert sets to counts
    for layer in validation['unique_spline_codes_per_layer']:
        validation['unique_spline_codes_per_layer'][layer] = len(validation['unique_spline_codes_per_layer'][layer])
    
    return validation

def run_checkpoint_analysis_pipeline(model_name: str,
                                    checkpoints: List[str],
                                    dataset_path: str = "polytope/dataset.json",
                                    target_layers: List[int] = None,
                                    output_dir: str = "cache/checkpoint_analysis",
                                    batch_size: int = 8) -> str:
    """
    Complete pipeline for extracting both pre and post activations using dataset.json
    
    Args:
        model_name: Model identifier for nnsight
        checkpoints: List of checkpoint steps to analyze
        dataset_path: Path to dataset.json file
        target_layers: Layers to analyze
        output_dir: Output directory for results
        
    Returns:
        Path to saved analysis results
    """
    
    # Load dataset
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    
    # Default target layers (optimized for model size)
    if target_layers is None:
        target_layers = _get_optimal_layers_for_model(model_name)

    
    # Validate model compatibility
    validation = validate_model_for_analysis(model_name)
    
    if validation['recommendations']:
        print("Model-specific recommendations:")
        for rec in validation['recommendations']:
            print(f"  • {rec}")
    
    
    # Extract activations
    all_records = extract_activations_from_dataset(
        model_name=model_name,
        checkpoints=checkpoints,
        dataset=dataset,
        target_layers=target_layers,
        batch_size=batch_size
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
        'capture_preactivation': True  # Always capturing pre-activation vectors now
    }
    
    # Save with metadata
    save_data = {
        'records': all_records,
        'metadata': analysis_metadata
    }
    
    with open(results_file, 'wb') as f:
        pickle.dump(save_data, f)
    return str(results_file)

def main():
    """Example usage of optimized checkpoint analysis"""
    model_names = ["EleutherAI/pythia-70m","EleutherAI/pythia-410m","EleutherAI/pythia-1b","EleutherAI/pythia-6.9b"]

    # all checkpoints 1000 to 143000
    checkpoints = ['1', '2', '4', '8', '16', '32', '64', '128', '256', '512', '1000', '13000', '23000', 
    '33000', '43000', '53000', '63000', '73000', '83000', '93000', '103000', '113000', '123000', '133000', '143000']
    print(checkpoints)
   
    datasets = ['/workspace/activation_datasets/body_polytope_dataset.json',
               '/workspace/activation_datasets/capital_polytope_dataset.json',
               '/workspace/activation_datasets/emotion_polytope_dataset.json',
               '/workspace/activation_datasets/jumbled_polytope_dataset.json',
               '/workspace/activation_datasets/material_polytope_dataset.json']
    
    all_records = []  # Store records from all combinations
    
    for model_name in model_names:
        for dataset_path in datasets:
            try:
                # Load dataset
                with open(dataset_path, 'r') as f:
                    dataset = json.load(f)
                
                print(f"\nProcessing {model_name} with {dataset_path}")
                print("Dataset structure:")
                
                # Show sample data structure
                if dataset.get('data'):
                    sample = dataset['data'][0]
                    print(f"Sample data keys: {list(sample.keys())}")
                    print(f"Sample phrase: {sample['phrase']}")
                    print(f"Sample frequency category: {sample['frequency_category']}")
                    print(f"Sample activation target idx: {sample['activation_target_idx']}")
            
                # Use automatic layer selection for each model
                target_layers = None  # Will use _get_optimal_layers_for_model
                records = extract_activations_from_dataset(
                    model_name=model_name,
                    checkpoints=checkpoints,
                    dataset=dataset,
                    target_layers=target_layers,
                    batch_size=4
                )
            
                print(f"Extracted {len(records)} activation records")
                if records:
                    print("Sample record categories:", [r['frequency_category'] for r in records[:10]])
                
                # Analyze patterns for this combination
                analysis = analyze_activation_patterns(records)
                print(f"Activation Analysis for {model_name}:")
                print(analysis)
                
                # Save records for this specific model-dataset combination
                dataset_name = Path(dataset_path).stem  # Extract filename without extension
                model_short = model_name.split('/')[-1]  # Extract model name without org
                output_path = f'/workspace/pythia_activations/{dataset_name}/activation_records_{model_short}.pkl'
                save_activation_records(records, output_path)
                print(f"Saved records to {output_path}")
                
                # Add to overall collection
                all_records.extend(records)
                
            except Exception as e:
                print(f"Error processing {model_name} with {dataset_path}: {e}")
                continue
    
    print(f"\nTotal records extracted across all combinations: {len(all_records)}")
    return all_records

if __name__ == "__main__": 
    main()