#!/usr/bin/env python3
"""
Simplified Checkpoint Analysis for Polytope Evolution
Simple functions for extracting position-based activations from model checkpoints
"""

import torch
import numpy as np
import pandas as pd
from nnsight import LanguageModel
from typing import List, Dict, Any, Tuple, Optional, Union
import re
from difflib import SequenceMatcher
import json
from tqdm import tqdm
from collections import defaultdict
from pathlib import Path

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
                           text_idx: int,
                           text: str,
                           ngram: str,
                           matched_text: str,
                           match_confidence: float,
                           matching_strategy: str,
                           category: str,
                           layer: int,
                           token_position: int,
                           char_start: int,
                           char_end: int,
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
        'text_idx': text_idx,
        'text': text,
        'ngram': ngram,
        'matched_text': matched_text,
        'match_confidence': match_confidence,
        'matching_strategy': matching_strategy,
        'category': category,
        'layer': layer,
        'neuron_idx': neuron_idx,
        'token_position': token_position,
        'char_start': char_start,
        'char_end': char_end,
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


def find_ngram_matches(text: str, ngram: str, strategy: str = "comprehensive") -> List[Tuple[int, int, float, str]]:
    """
    Find all occurrences of an n-gram in text with confidence scores
    
    Args:
        text: Input text to search
        ngram: N-gram to find
        strategy: Matching strategy ('exact', 'fuzzy', 'comprehensive')
        
    Returns:
        List of (start_pos, end_pos, confidence, matched_text) tuples
    """
    
    if strategy == "exact":
        return _find_exact_matches(text, ngram)
    elif strategy == "fuzzy":
        return _find_fuzzy_matches(text, ngram)
    elif strategy == "comprehensive":
        return _find_comprehensive_matches(text, ngram)
    else:
        return _find_exact_matches(text, ngram)


def _find_exact_matches(text: str, ngram: str) -> List[Tuple[int, int, float, str]]:
    """Find exact string matches"""
    matches = []
    start = 0
    while True:
        pos = text.find(ngram, start)
        if pos == -1:
            break
        matches.append((pos, pos + len(ngram), 1.0, ngram))
        start = pos + 1
    return matches


def _find_fuzzy_matches(text: str, ngram: str, threshold: float = 0.8) -> List[Tuple[int, int, float, str]]:
    """Find fuzzy matches using sequence similarity"""
    matches = []
    ngram_len = len(ngram)
    
    for i in range(len(text) - ngram_len + 1):
        window = text[i:i + ngram_len]
        similarity = SequenceMatcher(None, ngram.lower(), window.lower()).ratio()
        
        if similarity >= threshold:
            confidence = similarity * 0.8  # Lower confidence for fuzzy matches
            matches.append((i, i + ngram_len, confidence, window))
    
    return matches


def _find_comprehensive_matches(text: str, ngram: str) -> List[Tuple[int, int, float, str]]:
    """Find matches using multiple strategies"""
    all_matches = []
    
    # Strategy 1: Exact matches
    exact_matches = _find_exact_matches(text, ngram)
    all_matches.extend([(start, end, conf, matched) for start, end, conf, matched in exact_matches])
    
    # Strategy 2: Case insensitive
    for match in re.finditer(re.escape(ngram), text, re.IGNORECASE):
        all_matches.append((match.start(), match.end(), 0.9, match.group()))
    
    # Strategy 3: Word boundaries
    escaped_ngram = re.escape(ngram)
    for pattern, confidence in [(rf'\b{escaped_ngram}\b', 0.95), (rf'\b{escaped_ngram}', 0.7)]:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            all_matches.append((match.start(), match.end(), confidence, match.group()))
    
    # Remove duplicates and sort by confidence
    unique_matches = []
    seen_positions = set()
    
    for start, end, conf, matched in sorted(all_matches, key=lambda x: x[2], reverse=True):
        pos_key = (start, end)
        if pos_key not in seen_positions:
            unique_matches.append((start, end, conf, matched))
            seen_positions.add(pos_key)
    
    return unique_matches[:5]  # Return top 5 matches


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


def _get_precise_token_position(tokenizer, text: str, char_position: int) -> int:
    """Get precise token position for character position using offset mapping"""
    try:
        # Tokenize with offset mapping
        encoding = tokenizer(text, return_offsets_mapping=True, return_tensors="pt")
        offset_mapping = encoding['offset_mapping'][0]  # Remove batch dimension
        
        # Find token that contains the character position
        for token_idx, (start_char, end_char) in enumerate(offset_mapping):
            if start_char <= char_position < end_char:
                return token_idx
            elif start_char > char_position:
                # Character position is before this token, return previous
                return max(0, token_idx - 1)
        
        # If we didn't find it, return last token position
        return len(offset_mapping) - 1
        
    except Exception:
        # Fallback to approximate method if offset mapping fails
        tokens = tokenizer(text[:char_position], return_tensors="pt")
        return max(0, tokens['input_ids'].shape[1] - 1)


def extract_layer_activations(model: LanguageModel,
                            text: str,
                            layer: int,
                            position: int = -1,
                            strategy: str = "single",
                            capture_preactivation: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Extract activations from a specific layer and position
    
    Args:
        model: nnsight LanguageModel
        text: Input text
        layer: Layer index
        position: Token position (-1 for last token)
        strategy: Extraction strategy ('single', 'last', 'mean')
                 - 'single': Use specific position
                 - 'last': Always use last token
                 - 'mean': Average across all tokens
        
    Returns:
        If capture_preactivation is False: activation vector as numpy array
        If True: tuple (post_activation, pre_activation_sign_code)
    """
    try:
        # Set pad token if not set
        if model.tokenizer.pad_token is None:
            model.tokenizer.pad_token = model.tokenizer.eos_token
            
        # Tokenize input first
        inputs = model.tokenizer(text, return_tensors="pt")
        
        with model.trace(inputs):
            # Forward pass
            _ = model(**inputs)
            
            # Get layer activations - architecture agnostic
            model_layers = _get_model_layers(model)
            layer_activations = model_layers[layer].output[0]
            # Try to obtain pre-activation for common activation modules if requested
            if capture_preactivation:
                pre_code = None
                # Attempt to derive pre-activation sign pattern from residual MLP input and linear output
                # nnsight exposes Module.input and Module.output; many transformer MLPs use activation in between.
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


def extract_activations_for_ngram(model_name: str,
                                checkpoint: str,
                                text: str,
                                ngram: str,
                                target_layers: List[int],
                                category: str = "unknown",
                                text_idx: int = 0,
                                 activation_strategy: str = "single",
                                 capture_preactivation: bool = False) -> List[Dict[str, Any]]:
    """
    Extract activations for a specific n-gram across multiple layers
    
    Args:
        model_name: Model identifier
        checkpoint: Checkpoint step
        text: Input text containing the n-gram
        ngram: Target n-gram
        target_layers: List of layer indices
        category: Frequency category
        text_idx: Text index for tracking
        
    Returns:
        List of activation records
    """
    
    records = []
    
    try:
        # Load model at checkpoint with auto device mapping
        model = LanguageModel(model_name, revision=f"step{checkpoint}", device_map='auto')
        
        # Debug: verify model checkpoint loading (simplified)
        try:
            # Simple verification that model loaded correctly
            model_info = f"Model {model_name} checkpoint {checkpoint} loaded"
            print(f"Debug: {model_info}")
        except Exception as e:
            print(f"Debug: Model verification failed for checkpoint {checkpoint}: {e}")
        
        # Set pad token to avoid warnings
        if model.tokenizer.pad_token is None:
            model.tokenizer.pad_token = model.tokenizer.eos_token
        
        # Find n-gram matches in text
        matches = find_ngram_matches(text, ngram, strategy="comprehensive")
        
        if not matches:
            return records
        
        # Use best match
        char_start, char_end, confidence, matched_text = matches[0]
        
        # Get precise token position using offset mapping
        token_position = _get_precise_token_position(model.tokenizer, text, char_start)
        
        # Extract activations from each target layer
        for layer in target_layers:
            try:
                if capture_preactivation:
                    out = extract_layer_activations(model, text, layer, token_position, activation_strategy, capture_preactivation=True)
                    if isinstance(out, tuple):
                        activation_vector, pre_code = out
                    else:
                        activation_vector, pre_code = out, None
                else:
                    activation_vector = extract_layer_activations(model, text, layer, token_position, activation_strategy)
                    pre_code = None
                
                if len(activation_vector) > 0:
                    record = create_activation_record(
                        checkpoint_step=checkpoint,
                        text_idx=text_idx,
                        text=text,
                        ngram=ngram,
                        matched_text=matched_text,
                        match_confidence=confidence,
                        matching_strategy="comprehensive",
                        category=category,
                        layer=layer,
                        token_position=token_position,
                        char_start=char_start,
                        char_end=char_end,
                        activation_vector=activation_vector,
                        spline_code=pre_code
                    )
                    records.append(record)
                    
            except Exception as e:
                raise RuntimeError(f"Failed to extract activations from layer {layer}: {str(e)}. " +
                                 "Cannot proceed without complete activation data.")
        
        # Clean up model
        del model
        torch.cuda.empty_cache()
        
    except Exception as e:
        raise RuntimeError(f"Failed to load model {model_name} at checkpoint {checkpoint}: {str(e)}. " +
                         "Cannot proceed without valid model.")
    
    return records


def extract_activations_for_ngram_with_model(model: LanguageModel,
                                           checkpoint: str,
                                           text: str,
                                           ngram: str,
                                           target_layers: List[int],
                                           category: str = "unknown",
                                           text_idx: int = 0,
                                            activation_strategy: str = "single",
                                            capture_preactivation: bool = False) -> List[Dict[str, Any]]:
    """
    Extract activations for a specific n-gram using an already-loaded model.
    More efficient version that doesn't reload the model for each text.
    
    Args:
        model: Already loaded LanguageModel
        checkpoint: Checkpoint step (for record keeping)
        text: Input text containing the n-gram
        ngram: Target n-gram
        target_layers: List of layer indices
        category: Frequency category
        text_idx: Text index for tracking
        activation_strategy: Strategy for activation extraction
        
    Returns:
        List of activation records
    """
    
    records = []
    
    try:
        # Find n-gram matches in text
        matches = find_ngram_matches(text, ngram, strategy="comprehensive")
        
        if not matches:
            return records
        
        # Use best match
        char_start, char_end, confidence, matched_text = matches[0]
        
        # Get precise token position using offset mapping
        token_position = _get_precise_token_position(model.tokenizer, text, char_start)
        
        # Extract activations from each target layer
        for layer in target_layers:
            try:
                if capture_preactivation:
                    out = extract_layer_activations(model, text, layer, token_position, activation_strategy, capture_preactivation=True)
                    if isinstance(out, tuple):
                        activation_vector, pre_code = out
                    else:
                        activation_vector, pre_code = out, None
                else:
                    activation_vector = extract_layer_activations(model, text, layer, token_position, activation_strategy)
                    pre_code = None
                
                if len(activation_vector) > 0:
                    record = create_activation_record(
                        checkpoint_step=checkpoint,
                        text_idx=text_idx,
                        text=text,
                        ngram=ngram,
                        matched_text=matched_text,
                        match_confidence=confidence,
                        matching_strategy="comprehensive",
                        category=category,
                        layer=layer,
                        token_position=token_position,
                        char_start=char_start,
                        char_end=char_end,
                        activation_vector=activation_vector,
                        spline_code=pre_code
                    )
                    records.append(record)
                    
            except Exception as e:
                print(f"Warning: Failed to extract activations from layer {layer} for text {text_idx}: {e}")
                continue
        
    except Exception as e:
        print(f"Warning: Failed to process text {text_idx}: {e}")
    
    return records


def extract_activations_from_dataset(model_name: str,
                                   checkpoints: List[str],
                                   dataset: Dict[str, List[Any]],
                                   target_layers: List[int] = None,
                                   batch_size: int = 1,
                                   frequency_threshold: float = None,
                                    activation_strategy: str = "single",
                                    capture_preactivation: bool = False) -> List[Dict[str, Any]]:
    """
    Extract activations from a complete dataset across checkpoints
    Enhanced version with frequency processing for multi-dimensional analysis
    
    Args:
        model_name: Model identifier
        checkpoints: List of checkpoint steps
        dataset: Dataset with 'texts', 'ngrams', 'categories' keys
        target_layers: Layer indices to analyze
        batch_size: Batch size for processing
        frequency_threshold: Optional frequency threshold for filtering
        
    Returns:
        List of all activation records with frequency information
    """
    
    if target_layers is None:
        target_layers = [0, 2, 4]  # Default layers
    
    all_records = []
    total_texts = len(dataset['texts'])
    
    print(f"Extracting activations for {len(checkpoints)} checkpoints")
    print(f"Dataset size: {total_texts} samples")
    print(f"Target layers: {target_layers}")
    
    # Check if dataset has frequency categories
    has_categories = 'frequency_categories' in dataset and len(dataset['frequency_categories']) == total_texts
    if has_categories:
        category_counts = {cat: dataset['frequency_categories'].count(cat) for cat in set(dataset['frequency_categories'])}
        print(f"Found frequency categories: {category_counts}")
    else:
        print("No frequency categories found, using 'unknown'")
    
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
            
            # Process all texts with this model
            for text_idx in tqdm(range(total_texts), desc=f"Checkpoint {checkpoint}"):
                text = dataset['texts'][text_idx]
                ngram = dataset['ngrams'][text_idx]
                # FIXED: Proper category assignment
                if 'frequency_categories' in dataset and len(dataset['frequency_categories']) > text_idx:
                    category = dataset['frequency_categories'][text_idx]
                else:
                    category = "unknown"
                
                # Extract activations for this text/ngram combination using existing model
                text_records = extract_activations_for_ngram_with_model(
                    model=model,
                    checkpoint=checkpoint,
                    text=text,
                    ngram=ngram,
                    target_layers=target_layers,
                    category=category,
                    text_idx=text_idx,
                    activation_strategy=activation_strategy,
                    capture_preactivation=capture_preactivation
                )
                
                checkpoint_records.extend(text_records)
            
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
        categories = [r['category'] for r in all_records]
        category_counts = {cat: categories.count(cat) for cat in set(categories)}
        print(f"Final category distribution: {category_counts}")
    
    return all_records


def load_semantic_frequency_datasets(dataset_dir: str = "datasets/semantic_frequency") -> Dict[str, Dict[str, List]]:
    """
    Load all semantic frequency datasets for multi-dimensional analysis
    
    Args:
        dataset_dir: Directory containing semantic frequency datasets
        
    Returns:
        Dictionary mapping dataset_name -> dataset_dict
    """
    import json
    from pathlib import Path
    
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    
    datasets = {}
    
    for json_file in dataset_path.glob("*.json"):
        dataset_name = json_file.stem
        
        try:
            with open(json_file, 'r') as f:
                dataset = json.load(f)
            
            # Validate dataset structure
            required_keys = ['texts', 'ngrams']
            if all(key in dataset for key in required_keys):
                datasets[dataset_name] = dataset
                print(f"Loaded {dataset_name}: {len(dataset['texts'])} samples")
            else:
                print(f"Skipping {dataset_name}: missing required keys {required_keys}")
                
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
    
    print(f"Loaded {len(datasets)} datasets from {dataset_dir}")
    return datasets


def run_checkpoint_analysis_pipeline(model_name: str,
                                    checkpoints: List[str],
                                    dataset_names: List[str] = None,
                                    target_layers: List[int] = None,
                                    output_dir: str = "cache/checkpoint_analysis") -> str:
    """
    Complete pipeline for extracting and analyzing checkpoints across datasets
    
    Args:
        model_name: Model identifier for nnsight
        checkpoints: List of checkpoint steps to analyze
        dataset_names: Specific datasets to use (None for all)
        target_layers: Layers to analyze
        output_dir: Output directory for results
        
    Returns:
        Path to saved analysis results
    """
    from pathlib import Path
    import pickle
    from datetime import datetime
    
    # Load datasets
    datasets = load_semantic_frequency_datasets()
    
    if dataset_names:
        datasets = {name: datasets[name] for name in dataset_names if name in datasets}
    
    if not datasets:
        raise ValueError("No valid datasets found")
    
    # Default target layers (typical transformer layers)
    if target_layers is None:
        target_layers = [2, 4, 6, 8, 10, 12]
    
    print(f"=== Checkpoint Analysis Pipeline ===")
    print(f"Model: {model_name}")
    print(f"Checkpoints: {checkpoints}")
    print(f"Datasets: {list(datasets.keys())}")
    print(f"Target layers: {target_layers}")
    
    all_records = []
    
    # Process each dataset
    for dataset_name, dataset in datasets.items():
        print(f"\n=== Processing Dataset: {dataset_name} ===")
        
        dataset_records = extract_activations_from_dataset(
            model_name=model_name,
            checkpoints=checkpoints,
            dataset=dataset,
            target_layers=target_layers,
            capture_preactivation=True
        )
        
        # Add dataset identifier to records
        for record in dataset_records:
            record['dataset_name'] = dataset_name
        
        all_records.extend(dataset_records)
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_path / f"checkpoint_analysis_{timestamp}.pkl"
    
    analysis_metadata = {
        'model_name': model_name,
        'checkpoints': checkpoints,
        'datasets': list(datasets.keys()),
        'target_layers': target_layers,
        'n_total_records': len(all_records),
        'timestamp': timestamp
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
        import pickle
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
        import pickle
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
        'unique_ngrams': df['ngram'].nunique(),
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
    if 'category' in df.columns:
        category_analysis = {}
        for category in df['category'].unique():
            cat_df = df[df['category'] == category]
            category_analysis[category] = {
                'n_records': len(cat_df),
                'mean_sparsity': cat_df['sparsity'].mean(),
                'mean_norm': cat_df['activation_norm'].mean()
            }
        analysis['category_analysis'] = category_analysis
    
    return analysis

def main():
    """Example usage of optimized checkpoint analysis"""
    model_name = "EleutherAI/pythia-70m"
    checkpoints=['0', '1', '512', '1000', '10000', '50000', '143000']
    import json
    dataset = json.load(open('/content/country_capital_ngram_dataset.json'))

    print("Dataset keys:", list(dataset.keys()))
    if 'frequency_categories' in dataset:
        print("Sample frequency categories:", dataset['frequency_categories'][:10])

    records = extract_activations_from_dataset(
        model_name=model_name,
        checkpoints=checkpoints,
        dataset=dataset,
        target_layers=[4],
        activation_strategy="single"
    )

    print(f"Extracted {len(records)} activation records")
    if records:
        print("Sample record categories:", [r['category'] for r in records[:10]])

    # Analyze patterns
    analysis = analyze_activation_patterns(records)
    print("\nActivation Analysis:")
    print(analysis)

    return records


if __name__ == "__main__":
    records = main()
    save_activation_records(records, 'cache/activation_records.pkl')