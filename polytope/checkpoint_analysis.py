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
                           pre_activation_vector: Optional[np.ndarray] = None,
                           mlp_input_vector: Optional[np.ndarray] = None,
                           neuron_idx: Optional[int] = None,
                           activation_type: str = 'relu',
                           pair_id: Optional[str] = None,
                           **kwargs) -> Dict[str, Any]:
    """Create a structured activation record with pre/post activation and MLP input vectors.

    Args:
        activation_vector: Layer output (post-MLP + residual).
        pre_activation_vector: MLP up-projection output (for spline codes).
        mlp_input_vector: Residual stream entering MLP (H-dim). Used for
            Euclidean distance in polytope density computation.
        pair_id: Identifier linking this record to its synonym partner.
    """

    # Analyze POST-activation vector (main activation vector)
    cett_threshold = compute_cett_threshold(activation_vector, 0.01)
    binary_pattern = (np.abs(activation_vector) > cett_threshold).astype(int)
    sparsity = np.sum(binary_pattern) / len(binary_pattern)
    activation_norm = np.linalg.norm(activation_vector)
    n_active_neurons = np.sum(binary_pattern)

    # Analyze PRE-activation vector and compute spline code from it
    pre_activation_analysis = {}
    spline_code = None
    if pre_activation_vector is not None:
        pre_cett_threshold = compute_cett_threshold(pre_activation_vector, 0.01)
        pre_binary_pattern = (np.abs(pre_activation_vector) > pre_cett_threshold).astype(int)

        spline_code = _compute_spline_code_for_activation(pre_activation_vector, activation_type)

        pre_activation_analysis = {
            'pre_activation_vector': pre_activation_vector,
            'pre_binary_pattern': pre_binary_pattern,
            'pre_sparsity': np.sum(pre_binary_pattern) / len(pre_binary_pattern),
            'pre_activation_norm': np.linalg.norm(pre_activation_vector),
            'pre_n_active_neurons': np.sum(pre_binary_pattern),
            'pre_cett_threshold': pre_cett_threshold,
            'spline_code': spline_code
        }

    record = {
        'checkpoint_step': checkpoint_step,
        'sample_idx': sample_idx,
        'phrase': phrase,
        'sentence': sentence,
        'frequency_category': frequency_category,
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
        'pair_id': pair_id,

        # POST-activation data (layer output)
        'activation_vector': activation_vector,
        'binary_pattern': binary_pattern,
        'sparsity': sparsity,
        'activation_norm': activation_norm,
        'n_active_neurons': n_active_neurons,
        'cett_threshold': cett_threshold,

        # MLP input: residual stream entering MLP (H-dim).
        # This is the correct vector for Euclidean distance in polytope density.
        'mlp_input_vector': mlp_input_vector if mlp_input_vector is not None else activation_vector,

        # Spline code from pre-activation (main binary pattern for analysis)
        'spline_code': spline_code,

        'metadata': kwargs
    }

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
    # Try different model architectures (Pythia uses gpt_neox.layers)
    for attr_path in ['gpt_neox.layers', 'transformer.h', 'model.layers', 'layers']:
        try:
            layers = model
            for attr in attr_path.split('.'):
                layers = getattr(layers, attr)
            return layers
        except AttributeError:
            continue
    
    raise RuntimeError(f"Could not find layers in model architecture. "
                      f"Model type: {type(model).__name__}. "
                      f"For Pythia models, ensure the model uses 'gpt_neox.layers' structure.")

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

def _detect_activation_function(model: LanguageModel, layer: int) -> str:
    """Detect the activation function used in the MLP of a specific layer"""
    # 1) Prefer config when available (most reliable, avoids Envoy wrappers)
    try:
        if hasattr(model, 'config') and model.config is not None:
            for key in ['hidden_act', 'activation_function', 'hidden_activation']:
                act = getattr(model.config, key, None)
                if isinstance(act, str) and len(act) > 0:
                    act_l = act.lower()
                    if 'relu' in act_l:
                        return 'relu'
                    if 'gelu' in act_l:
                        return 'gelu'
                    if 'silu' in act_l or 'swish' in act_l:
                        return 'swish'
                    return act_l
    except Exception:
        pass

    # Helper to unwrap nnsight Envoy wrappers
    def _unwrap_envoy(obj: Any) -> Any:
        for attr in ('module', 'obj', 'value', '_obj', '_module'):
            try:
                if hasattr(obj, attr):
                    unwrapped = getattr(obj, attr)
                    if unwrapped is not None:
                        return unwrapped
            except Exception:
                continue
        return obj

    # 2) Fallback: inspect layer modules (handle Envoy wrappers)
    try:
        model_layers = _get_model_layers(model)

        mlp_paths = ['mlp', 'feed_forward', 'ff', 'mlp_1']
        act_attrs = ['act_fn', 'activation_fn', 'act', 'activation', 'gelu', 'relu', 'silu', 'swish']

        for mlp_path in mlp_paths:
            try:
                mlp_module = getattr(model_layers[layer], mlp_path)
                mlp_module = _unwrap_envoy(mlp_module)
                for attr in act_attrs:
                    if hasattr(mlp_module, attr):
                        act_fn = getattr(mlp_module, attr)
                        act_fn = _unwrap_envoy(act_fn)
                        act_type = type(act_fn).__name__.lower()
                        if 'relu' in act_type:
                            return 'relu'
                        if 'gelu' in act_type:
                            return 'gelu'
                        if 'silu' in act_type or 'swish' in act_type:
                            return 'swish'
                        return act_type
            except AttributeError:
                continue
    except Exception:
        pass

    # 3) Model-family heuristic: Pythia/NeoX -> GELU
    try:
        model_name = getattr(model, 'name', '') or ''
        model_repo = getattr(model, 'repo', '') or ''
        joined = f"{model_name} {model_repo}".lower()
        if any(key in joined for key in ['pythia', 'gpt-neox', 'neox']):
            return 'gelu'
    except Exception:
        pass

    return 'unknown'

def _compute_spline_code_for_activation(pre_activation_vector: np.ndarray, 
                                       activation_type: str) -> np.ndarray:
    """
    Compute spline code based on activation function type.
    
    Spline codes represent the polytope regions defined by activation function boundaries.
    For research-grade analysis, these should capture the true linear regions.
    """
    if activation_type == 'relu':
        # ReLU: simple threshold at 0 (exact spline boundary)
        return (pre_activation_vector > 0).astype(int)
    elif activation_type == 'gelu':
        # GELU: Gaussian Error Linear Unit used in Pythia models
        # GELU(x) ≈ x * Φ(x) where Φ is the standard normal CDF
        # The inflection point is approximately at x ≈ -0.67 for research purposes
        # For polytope analysis, we use the zero-crossing as the primary boundary
        # Additional boundary at inflection point for finer-grained analysis
        primary_boundary = (pre_activation_vector > 0).astype(int)
        
        # For compatibility with existing analysis pipeline, return binary version
        # Research note: Full multi-bit spline codes could be implemented here
        return primary_boundary
    elif activation_type == 'swish' or activation_type == 'silu':
        # Swish/SiLU: x * sigmoid(x), inflection around x ≈ 0
        return (pre_activation_vector > 0).astype(int)
    else:
        # Default to ReLU-style for unknown activations
        # Log warning for research tracking
        print(f"Warning: Unknown activation type '{activation_type}', using ReLU-style spline code")
        return (pre_activation_vector > 0).astype(int)

def extract_layer_activations(model: LanguageModel,
                            token_ids: List[int],
                            layer: int,
                            position: int = -1,
                            strategy: str = "single") -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Extract pre-activation, post-activation, and MLP input vectors from a layer.

    Args:
        model: nnsight LanguageModel
        token_ids: Pre-tokenized input token IDs
        layer: Layer index
        position: Token position (-1 for last token)
        strategy: Extraction strategy ('single', 'last', 'mean')

    Returns:
        Tuple (post_activation_vector, pre_activation_vector, mlp_input_vector):
            - post_activation_vector: Layer output (after activation function + residual)
            - pre_activation_vector: MLP intermediate output (before activation function)
            - mlp_input_vector: Residual stream entering the MLP (H-dim input space).
              This is the correct vector for Euclidean distance in polytope density.
    """
    try:
        if model.tokenizer.pad_token is None:
            model.tokenizer.pad_token = model.tokenizer.eos_token

        inputs = {"input_ids": torch.tensor([token_ids], device="cuda")}

        with model.trace(inputs):
            _ = model(**inputs)

            model_layers = _get_model_layers(model)

            # Save full hidden state for this layer (batch, seq, hidden)
            hidden_handle = model_layers[layer].output.save()

            # --- MLP input: the residual stream entering the MLP ---
            # This is the H-dim vector the MLP sees as input, i.e. the correct
            # space for Euclidean distance in polytope density (per Humayun et al.)
            mlp_input_handle = None
            mlp_paths = [
                'mlp',
                'feed_forward',
                'ff',
                'mlp_1'
            ]
            for mlp_path in mlp_paths:
                mlp_module = getattr(model_layers[layer], mlp_path, None)
                if mlp_module is not None:
                    try:
                        mlp_input_handle = mlp_module.input.save()
                        break
                    except Exception:
                        continue

            if mlp_input_handle is None:
                print(f"WARNING: Could not capture MLP input for layer {layer}. "
                      "Falling back to layer output for Euclidean distance.")

            # --- MLP pre-activation: output of up-projection (for spline codes) ---
            pre_handle = None

            intermediate_names = [
                'dense_h_to_4h',
                'up_proj',
                'c_fc',
                'fc1',
                'intermediate'
            ]

            found_preactivation = False
            for mlp_path in mlp_paths:
                if found_preactivation:
                    break
                try:
                    mlp_module = getattr(model_layers[layer], mlp_path)
                    for intermediate_name in intermediate_names:
                        if hasattr(mlp_module, intermediate_name):
                            intermediate_layer = getattr(mlp_module, intermediate_name)
                            pre_handle = intermediate_layer.output.save()
                            found_preactivation = True
                            break
                except AttributeError:
                    continue

            if not found_preactivation:
                try:
                    for mlp_path in mlp_paths:
                        mlp_module = getattr(model_layers[layer], mlp_path, None)
                        if mlp_module is None:
                            continue
                        act_names = ['act_fn', 'activation_fn', 'act', 'gelu', 'relu', 'swish']
                        for act_name in act_names:
                            if hasattr(mlp_module, act_name):
                                act_fn = getattr(mlp_module, act_name)
                                if hasattr(act_fn, 'input') and act_fn.input is not None:
                                    pre_handle = act_fn.input[0].save()
                                    found_preactivation = True
                                    break
                        if found_preactivation:
                            break
                except Exception:
                    pass

            if not found_preactivation:
                print(f"WARNING: Could not find MLP preactivation for layer {layer}. Spline codes will not be available.")
                pre_handle = None

        # --- Materialize saved tensors ---
        def _materialize(handle):
            if handle is None:
                return None
            val = getattr(handle, 'value', handle)
            if isinstance(val, (tuple, list)):
                return val[0]
            return val

        hidden_tensor = _materialize(hidden_handle)
        pre_tensor = _materialize(pre_handle)
        mlp_input_tensor = _materialize(mlp_input_handle)

        # Handle MLP input which may come as tuple (input,) from module.input
        if mlp_input_tensor is not None and isinstance(mlp_input_tensor, (tuple, list)):
            mlp_input_tensor = mlp_input_tensor[0]

        # Ensure tensors are 3D: (batch, seq, hidden)
        for t_name in ['hidden_tensor', 'pre_tensor', 'mlp_input_tensor']:
            t = locals()[t_name]
            if t is not None and t.dim() == 2:
                locals()[t_name] = t.unsqueeze(0)
        # Re-read after potential unsqueeze
        if hidden_tensor.dim() == 2:
            hidden_tensor = hidden_tensor.unsqueeze(0)
        if pre_tensor is not None and pre_tensor.dim() == 2:
            pre_tensor = pre_tensor.unsqueeze(0)
        if mlp_input_tensor is not None and mlp_input_tensor.dim() == 2:
            mlp_input_tensor = mlp_input_tensor.unsqueeze(0)

        # Select vectors based on strategy
        def _select_position(tensor, pos, strat):
            if tensor is None:
                return None
            if strat == "mean":
                return tensor[0, :, :].mean(dim=0)
            elif strat == "last" or pos == -1:
                return tensor[0, -1, :]
            else:
                p = min(pos, tensor.shape[1] - 1)
                return tensor[0, p, :]

        post_act = _select_position(hidden_tensor, position, strategy)
        pre_act = _select_position(pre_tensor, position, strategy)
        mlp_input = _select_position(mlp_input_tensor, position, strategy)

        # Convert to numpy
        post_act_np = post_act.detach().cpu().numpy()
        pre_act_np = pre_act.detach().cpu().numpy() if pre_act is not None else None
        mlp_input_np = mlp_input.detach().cpu().numpy() if mlp_input is not None else None

        return post_act_np, pre_act_np, mlp_input_np

    except Exception as e:
        raise RuntimeError(f"Failed to extract activations from layer {layer}: {str(e)}. " +
                         "Cannot proceed without valid activation data.")

def extract_activations_for_sample(model: LanguageModel,
                                 model_name: str,
                                 checkpoint: str,
                                 sample: Dict[str, Any],
                                 target_layers: List[int],
                                 sample_idx: int = 0,
                                 activation_strategy: str = "single") -> List[Dict[str, Any]]:
    """
    Extract both pre and post activation vectors for a specific sample
    
    Args:
        model: Already loaded LanguageModel
        model_name: Model identifier
        checkpoint: Checkpoint step (for record keeping)
        sample: Sample from dataset with pre-computed token positions
        target_layers: List of layer indices
        sample_idx: Sample index for tracking
        activation_strategy: Strategy for activation extraction
        
    Returns:
        List of activation records with both pre and post activation data
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
                # Detect activation function type for this layer
                activation_type = _detect_activation_function(model, layer)
                
                # Extract post-activation, pre-activation, and MLP input vectors
                post_activation_vector, pre_activation_vector, mlp_input_vector = extract_layer_activations(
                    model, token_ids_sentence, layer,
                    activation_target_idx, activation_strategy
                )

                if len(post_activation_vector) > 0:
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
                        activation_vector=post_activation_vector,
                        pre_activation_vector=pre_activation_vector,
                        mlp_input_vector=mlp_input_vector,
                        activation_type=activation_type,
                        pair_id=sample.get('pair_id'),
                        model_name=model_name
                    )
                    records.append(record)
                    
            except Exception as e:
                print(f"Warning: Failed to extract activations from layer {layer} for sample {sample_idx}: {e}")
                continue
        
    except Exception as e:
        print(f"Warning: Failed to process sample {sample_idx}: {e}")
    
    return records

def save_checkpoint_progress(checkpoint: str, 
                           records: List[Dict[str, Any]], 
                           network_storage_path: str,
                           model_name: str) -> None:
    """
    Save checkpoint progress to network storage with resume capability
    
    Args:
        checkpoint: Current checkpoint being processed
        records: Records from this checkpoint
        network_storage_path: Network storage directory path
        model_name: Model name for organizing files
    """
    storage_path = Path(network_storage_path)
    storage_path.mkdir(parents=True, exist_ok=True)
    
    # Create model-specific subdirectory
    model_dir = storage_path / model_name.replace('/', '_')
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Save individual checkpoint file
    checkpoint_file = model_dir / f"checkpoint_{checkpoint}.pkl"
    with open(checkpoint_file, 'wb') as f:
        pickle.dump({
            'checkpoint': checkpoint,
            'records': records,
            'timestamp': datetime.now().isoformat(),
            'model_name': model_name,
            'n_records': len(records)
        }, f)
    
    # Update progress tracker
    progress_file = model_dir / "progress.json"
    progress_data = {
        'model_name': model_name,
        'last_completed_checkpoint': checkpoint,
        'last_update': datetime.now().isoformat(),
        'total_records_so_far': len(records),
        'completed_checkpoints': []
    }
    
    # Load existing progress if available
    if progress_file.exists():
        try:
            with open(progress_file, 'r') as f:
                existing_progress = json.load(f)
                progress_data['completed_checkpoints'] = existing_progress.get('completed_checkpoints', [])
                # Count total records from all completed checkpoints
                total_records = 0
                for ckpt in progress_data['completed_checkpoints']:
                    ckpt_file = model_dir / f"checkpoint_{ckpt}.pkl"
                    if ckpt_file.exists():
                        with open(ckpt_file, 'rb') as cf:
                            ckpt_data = pickle.load(cf)
                            total_records += ckpt_data['n_records']
                progress_data['total_records_so_far'] = total_records + len(records)
        except Exception as e:
            print(f"Warning: Could not load existing progress: {e}")
    
    # Add current checkpoint to completed list
    if checkpoint not in progress_data['completed_checkpoints']:
        progress_data['completed_checkpoints'].append(checkpoint)
    
    with open(progress_file, 'w') as f:
        json.dump(progress_data, f, indent=2)
    
    print(f"💾 Saved checkpoint {checkpoint} to network storage: {checkpoint_file}")
    print(f"📊 Progress: {len(progress_data['completed_checkpoints'])} checkpoints completed")

def load_completed_checkpoints(network_storage_path: str, model_name: str) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    Load all completed checkpoints from network storage
    
    Args:
        network_storage_path: Network storage directory path
        model_name: Model name
        
    Returns:
        Tuple of (completed_checkpoint_list, all_records)
    """
    storage_path = Path(network_storage_path)
    model_dir = storage_path / model_name.replace('/', '_')
    
    if not model_dir.exists():
        return [], []
    
    progress_file = model_dir / "progress.json"
    if not progress_file.exists():
        return [], []
    
    # Load progress
    with open(progress_file, 'r') as f:
        progress_data = json.load(f)
    
    completed_checkpoints = progress_data.get('completed_checkpoints', [])
    all_records = []
    
    print(f"📁 Found {len(completed_checkpoints)} completed checkpoints in network storage")
    
    # Load all completed checkpoint files
    for checkpoint in completed_checkpoints:
        checkpoint_file = model_dir / f"checkpoint_{checkpoint}.pkl"
        if checkpoint_file.exists():
            try:
                with open(checkpoint_file, 'rb') as f:
                    data = pickle.load(f)
                    all_records.extend(data['records'])
                    print(f"✅ Loaded {data['n_records']} records from checkpoint {checkpoint}")
            except Exception as e:
                print(f"⚠️ Warning: Could not load checkpoint {checkpoint}: {e}")
    
    return completed_checkpoints, all_records

def extract_activations_from_dataset_with_network_storage(model_name: str,
                                                        checkpoints: List[str],
                                                        dataset: Dict[str, Any],
                                                        network_storage_path: str,
                                                        target_layers: List[int] = None,
                                                        activation_strategy: str = "single",
                                                        batch_size: int = 32,
                                                        resume: bool = True) -> List[Dict[str, Any]]:
    """
    Extract activations with network storage backup and resume capability
    
    Args:
        model_name: Model identifier
        checkpoints: List of checkpoint steps
        dataset: Dataset with pre-computed token positions
        network_storage_path: Path to network storage for incremental saves
        target_layers: Layer indices to analyze
        activation_strategy: Strategy for activation extraction
        batch_size: Batch size for processing
        resume: Whether to resume from existing progress
        
    Returns:
        List of all activation records
    """
    
    if target_layers is None:
        target_layers = _get_optimal_layers_for_model(model_name)
    
    validate_dataset_structure(dataset)
    samples = dataset['data']
    total_samples = len(samples)
    
    # Load existing progress if resuming
    completed_checkpoints = []
    all_records = []
    
    if resume:
        completed_checkpoints, all_records = load_completed_checkpoints(
            network_storage_path, model_name
        )
        print(f"🔄 Resuming: Already have {len(all_records)} records from {len(completed_checkpoints)} checkpoints")
    
    # Filter out already completed checkpoints
    remaining_checkpoints = [ckpt for ckpt in checkpoints if ckpt not in completed_checkpoints]
    
    if not remaining_checkpoints:
        print("✅ All checkpoints already completed!")
        return all_records
    
    print(f"📋 Processing {len(remaining_checkpoints)} remaining checkpoints")
    print(f"🎯 Target layers: {target_layers}")
    print(f"📦 Batch size: {batch_size}")
    print(f"💾 Network storage: {network_storage_path}")
    
    # Process remaining checkpoints with network storage
    for checkpoint_idx, checkpoint in enumerate(remaining_checkpoints):
        print(f"\n[{checkpoint_idx+1}/{len(remaining_checkpoints)}] Processing checkpoint step{checkpoint}")
        
        try:
            # Load model with optimized settings
            print(f"Loading model...")
            model = LanguageModel(
                model_name,
                revision=f"step{checkpoint}",
                device_map='cuda:0',
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
            model.eval()
            
            if model.tokenizer.pad_token is None:
                model.tokenizer.pad_token = model.tokenizer.eos_token
            
            # Pre-compute for efficiency
            model_layers = _get_model_layers(model)
            act_map = {l: _detect_activation_function(model, l) for l in target_layers}
            
            checkpoint_records = []
            
            # Process in larger batches
            for batch_start in tqdm(range(0, total_samples, batch_size), 
                                  desc=f"Checkpoint {checkpoint}", 
                                  unit="batch"):
                batch_end = min(batch_start + batch_size, total_samples)
                batch_samples = samples[batch_start:batch_end]
                
                # Prepare batch inputs
                token_id_batches = [s['token_ids_sentence'] for s in batch_samples]
                positions = [int(s['activation_target_idx']) for s in batch_samples]
                
                pad_id = model.tokenizer.pad_token_id or model.tokenizer.eos_token_id
                max_len = max(len(ids) for ids in token_id_batches)
                
                padded = [ids + [pad_id] * (max_len - len(ids)) for ids in token_id_batches]
                attention_mask = [[1] * len(ids) + [0] * (max_len - len(ids)) for ids in token_id_batches]
                
                inputs = {
                    'input_ids': torch.tensor(padded, device='cuda'),
                    'attention_mask': torch.tensor(attention_mask, device='cuda')
                }
                
                # Single forward pass for entire batch
                with torch.inference_mode(), torch.cuda.amp.autocast():
                    with model.trace(inputs):
                        # Register all layer outputs and pre-activations
                        hidden_handles = {}
                        pre_handles = {}
                        
                        for layer in target_layers:
                            try:
                                hidden_handles[layer] = model_layers[layer].output.save()
                            except:
                                hidden_handles[layer] = None
                            
                            # Get pre-activation handle
                            pre_handle = None
                            try:
                                for mlp_path in ['mlp', 'feed_forward', 'ff']:
                                    if hasattr(model_layers[layer], mlp_path):
                                        mlp = getattr(model_layers[layer], mlp_path)
                                        for inter_name in ['dense_h_to_4h', 'up_proj', 'c_fc']:
                                            if hasattr(mlp, inter_name):
                                                pre_handle = getattr(mlp, inter_name).output.save()
                                                break
                                        if pre_handle:
                                            break
                            except:
                                pass
                            pre_handles[layer] = pre_handle
                        
                        # Execute model
                        _ = model(**inputs)
                
                # Process batch results immediately
                for i, sample in enumerate(batch_samples):
                    sample_idx = batch_start + i
                    pos = positions[i]
                    
                    for layer in target_layers:
                        if hidden_handles[layer] is None:
                            continue
                            
                        try:
                            # Get tensors
                            hidden_val = getattr(hidden_handles[layer], 'value', hidden_handles[layer])
                            hidden_tensor = hidden_val[0] if isinstance(hidden_val, (tuple, list)) else hidden_val
                            
                            if hidden_tensor.dim() == 2:
                                hidden_tensor = hidden_tensor.unsqueeze(0)
                            
                            pre_tensor = None
                            if pre_handles[layer]:
                                pre_val = getattr(pre_handles[layer], 'value', pre_handles[layer])
                                pre_tensor = pre_val[0] if isinstance(pre_val, (tuple, list)) else pre_val
                                if pre_tensor is not None and pre_tensor.dim() == 2:
                                    pre_tensor = pre_tensor.unsqueeze(0)
                            
                            # Extract position-specific activations
                            if pos >= hidden_tensor.shape[1]:
                                pos = hidden_tensor.shape[1] - 1
                            
                            post_act = hidden_tensor[i, pos, :].detach().cpu().numpy()
                            pre_act = (pre_tensor[i, pos, :].detach().cpu().numpy() 
                                     if pre_tensor is not None else None)
                            
                            # Create record
                            record = create_activation_record(
                                checkpoint_step=checkpoint,
                                sample_idx=sample_idx,
                                phrase=sample['phrase'],
                                sentence=sample['sentence'],
                                frequency_category=sample['frequency_category'],
                                layer=layer,
                                activation_target_idx=sample['activation_target_idx'],
                                phrase_start_idx=sample['phrase_start_idx'],
                                phrase_end_idx=sample['phrase_end_idx'],
                                activation_vector=post_act,
                                pre_activation_vector=pre_act,
                                activation_type=act_map.get(layer, 'unknown'),
                                model_name=model_name
                            )
                            checkpoint_records.append(record)
                            
                        except Exception as e:
                            print(f"Warning: Failed to process sample {sample_idx}, layer {layer}: {e}")
                
                # Cleanup batch tensors immediately
                del hidden_handles, pre_handles, inputs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # SAVE TO NETWORK STORAGE after each checkpoint
            save_checkpoint_progress(checkpoint, checkpoint_records, network_storage_path, model_name)
            all_records.extend(checkpoint_records)
            
            print(f"✅ Checkpoint {checkpoint} complete: {len(checkpoint_records)} records")
            print(f"📈 Total records so far: {len(all_records)}")
            
        except Exception as e:
            print(f"❌ Error processing checkpoint {checkpoint}: {e}")
            print(f"💾 Progress saved up to previous checkpoint in network storage")
            
        finally:
            # Always cleanup model
            if 'model' in locals():
                try:
                    del model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    clear_hf_cache_for_revision(model_name, f"step{checkpoint}")
                    import gc
                    gc.collect()
                except:
                    pass
    
    print(f"\n🎉 Analysis complete! Total records: {len(all_records)}")
    return all_records

def extract_activations_from_dataset(model_name: str,
                                   checkpoints: List[str],
                                   dataset: Dict[str, Any],
                                   target_layers: List[int] = None,
                                   activation_strategy: str = "single",
                                   batch_size: int = 4) -> List[Dict[str, Any]]:
    """
    Extract both pre and post activations from a complete dataset across checkpoints
    
    Args:
        model_name: Model identifier
        checkpoints: List of checkpoint steps
        dataset: Dataset with 'data' key containing samples with pre-computed token positions
        target_layers: Layer indices to analyze
        activation_strategy: Strategy for activation extraction
        
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
            # Force explicit CUDA device management to prevent memory fragmentation
            model = LanguageModel(
                model_name,
                revision=f"step{checkpoint}",
                device_map='cuda:0',  # Explicit single GPU to avoid fragmentation
                torch_dtype=torch.float16  # Use half precision to reduce memory usage
            )
            model.eval()
            # Set pad token to avoid warnings
            if model.tokenizer.pad_token is None:
                model.tokenizer.pad_token = model.tokenizer.eos_token
            
            print(f"Model loaded successfully for checkpoint {checkpoint}")
            
            # Batched processing to reduce forward passes and leverage GPU parallelism
            def _prepare_batch(batch_samples: List[Dict[str, Any]]):
                token_id_batches: List[List[int]] = []
                positions: List[int] = []
                for s in batch_samples:
                    token_id_batches.append(s['token_ids_sentence'])
                    positions.append(int(s['activation_target_idx']))
                pad_id = model.tokenizer.pad_token_id if model.tokenizer.pad_token_id is not None else model.tokenizer.eos_token_id
                max_len = max(len(ids) for ids in token_id_batches)
                padded = [ids + [pad_id] * (max_len - len(ids)) for ids in token_id_batches]
                attention_mask = [[1] * len(ids) + [0] * (max_len - len(ids)) for ids in token_id_batches]
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                inputs = {
                    'input_ids': torch.tensor(padded, device=device),
                    'attention_mask': torch.tensor(attention_mask, device=device)
                }
                return inputs, positions

            def _detect_activation_map(model: LanguageModel, layers: List[int]) -> Dict[int, str]:
                mapping: Dict[int, str] = {}
                for l in layers:
                    try:
                        mapping[l] = _detect_activation_function(model, l)
                    except Exception:
                        mapping[l] = 'unknown'
                return mapping

            model_layers = _get_model_layers(model)
            act_map = _detect_activation_map(model, target_layers)

            for batch_start in tqdm(range(0, total_samples, max(1, int(batch_size))), desc=f"Checkpoint {checkpoint}"):
                batch_end = min(batch_start + max(1, int(batch_size)), total_samples)
                batch_samples = samples[batch_start:batch_end]

                inputs, batch_positions = _prepare_batch(batch_samples)

                with torch.inference_mode():
                    with model.trace(inputs):
                        hidden_handles: Dict[int, Any] = {}
                        pre_handles: Dict[int, Optional[Any]] = {}

                        mlp_paths = ['mlp', 'feed_forward', 'ff', 'mlp_1']
                        intermediate_names = ['dense_h_to_4h', 'up_proj', 'c_fc', 'fc1', 'intermediate']

                        for layer in target_layers:
                            try:
                                hidden_handles[layer] = model_layers[layer].output.save()
                            except Exception:
                                hidden_handles[layer] = None

                            pre_handle = None
                            try:
                                mlp_module = None
                                for mlp_path in mlp_paths:
                                    if hasattr(model_layers[layer], mlp_path):
                                        mlp_module = getattr(model_layers[layer], mlp_path)
                                        break
                                if mlp_module is not None:
                                    for name in intermediate_names:
                                        if hasattr(mlp_module, name):
                                            inter = getattr(mlp_module, name)
                                            try:
                                                pre_handle = inter.output.save()
                                                break
                                            except Exception:
                                                continue
                                if pre_handle is None and mlp_module is not None:
                                    for attr in ['act_fn', 'activation_fn', 'act', 'gelu', 'relu', 'swish']:
                                        if hasattr(mlp_module, attr):
                                            act_fn = getattr(mlp_module, attr)
                                            if hasattr(act_fn, 'input') and act_fn.input is not None:
                                                try:
                                                    pre_handle = act_fn.input[0].save()
                                                    break
                                                except Exception:
                                                    pass
                            except Exception:
                                pre_handle = None
                            pre_handles[layer] = pre_handle

                        _ = model(**inputs)

                # Materialize saved tensors and create records per sample and layer
                for i, sample in enumerate(batch_samples):
                    pos = batch_positions[i]
                    sample_idx = batch_start + i

                    phrase = sample['phrase']
                    sentence = sample['sentence']
                    frequency_category = sample['frequency_category']
                    activation_target_idx = sample['activation_target_idx']
                    phrase_start_idx = sample['phrase_start_idx']
                    phrase_end_idx = sample['phrase_end_idx']

                    for layer in target_layers:
                        hidden_handle = hidden_handles.get(layer)
                        if hidden_handle is None:
                            continue
                        hidden_val = getattr(hidden_handle, 'value', hidden_handle)
                        hidden_tensor = hidden_val[0] if isinstance(hidden_val, (tuple, list)) else hidden_val
                        if hidden_tensor.dim() == 2:
                            hidden_tensor = hidden_tensor.unsqueeze(0)

                        pre_tensor = None
                        pre_handle = pre_handles.get(layer)
                        if pre_handle is not None:
                            pre_val = getattr(pre_handle, 'value', pre_handle)
                            pre_tensor = pre_val[0] if isinstance(pre_val, (tuple, list)) else pre_val
                            if pre_tensor is not None and pre_tensor.dim() == 2:
                                pre_tensor = pre_tensor.unsqueeze(0)

                        if activation_strategy == "mean":
                            post_act = hidden_tensor[i, :, :].mean(dim=0)
                            pre_act = pre_tensor[i, :, :].mean(dim=0) if pre_tensor is not None else None
                        elif activation_strategy == "last" or pos == -1:
                            post_act = hidden_tensor[i, -1, :]
                            pre_act = pre_tensor[i, -1, :] if pre_tensor is not None else None
                        else:
                            if pos >= hidden_tensor.shape[1]:
                                pos = hidden_tensor.shape[1] - 1
                            post_act = hidden_tensor[i, pos, :]
                            pre_act = pre_tensor[i, pos, :] if pre_tensor is not None else None

                        # Immediately move to CPU and convert to numpy to free GPU memory
                        post_np = post_act.detach().cpu().numpy()
                        pre_np = pre_act.detach().cpu().numpy() if pre_act is not None else None
                        
                        # Clear references to GPU tensors immediately
                        del post_act
                        if pre_act is not None:
                            del pre_act

                        try:
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
                                activation_vector=post_np,
                                pre_activation_vector=pre_np,
                                activation_type=act_map.get(layer, 'unknown'),
                                model_name=model_name
                            )
                            checkpoint_records.append(record)
                        except Exception as e:
                            print(f"Warning: Failed to create record for sample {sample_idx}, layer {layer}: {e}")
                            
                # Clear batch handles after processing to free GPU memory
                del hidden_handles
                del pre_handles
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Aggressive memory cleanup
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                # Force garbage collection
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
            # Handle pre-activation vectors
            if 'pre_activation_vector' in record and record['pre_activation_vector'] is not None:
                json_record['pre_activation_vector'] = record['pre_activation_vector'].tolist()
            if 'pre_binary_pattern' in record and record['pre_binary_pattern'] is not None:
                json_record['pre_binary_pattern'] = record['pre_binary_pattern'].tolist()
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

def run_checkpoint_analysis_pipeline(model_name: str,
                                    checkpoints: List[str],
                                    dataset_path: str = "polytope/dataset.json",
                                    target_layers: List[int] = None,
                                    output_dir: str = "cache/checkpoint_analysis",
                                    activation_strategy: str = "single",
                                    batch_size: int = 8) -> str:
    """
    Complete pipeline for extracting both pre and post activations using dataset.json
    
    Args:
        model_name: Model identifier for nnsight
        checkpoints: List of checkpoint steps to analyze
        dataset_path: Path to dataset.json file
        target_layers: Layers to analyze
        output_dir: Output directory for results
        activation_strategy: Strategy for activation extraction
        
    Returns:
        Path to saved analysis results
    """
    
    # Load dataset
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    
    # Default target layers (optimized for model size)
    if target_layers is None:
        target_layers = _get_optimal_layers_for_model(model_name)
    
    print(f"=== Checkpoint Analysis Pipeline ===")
    print(f"Model: {model_name}")
    
    # Validate model compatibility
    validation = validate_model_for_analysis(model_name)
    print(f"Model family: {validation['model_family']}")
    print(f"Estimated layers: {validation['estimated_layers']}")
    print(f"Activation function: {validation['activation_function']}")
    
    if validation['recommendations']:
        print("Model-specific recommendations:")
        for rec in validation['recommendations']:
            print(f"  • {rec}")
    
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
        'activation_strategy': activation_strategy,
        'capture_preactivation': True  # Always capturing pre-activation vectors now
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
    model_name = "EleutherAI/pythia-6.9b"
    # all checkpoints 1000 to 143000
    checkpoints = [str(2**i) for i in range(0, 10)] # 1 to 512
    checkpoints.extend([str(i) for i in range(1000, 144000, 1000)])
    print(checkpoints)
    dataset_path = "/workspace/country_capital_polytope_dataset.json"
    
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

    # Use automatic layer selection for Pythia 6.9B (will select early, middle, late layers)
    target_layers = None  # Will use _get_optimal_layers_for_model
    records = extract_activations_from_dataset(
        model_name=model_name,
        checkpoints=checkpoints,
        dataset=dataset,
        target_layers=target_layers,
        activation_strategy="single",
        batch_size=256
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