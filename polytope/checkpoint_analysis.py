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
import warnings
from pathlib import Path


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
                           **kwargs) -> Dict[str, Any]:
    """Create a structured activation record as dictionary"""
    
    # Compute additional metrics
    binary_pattern = (activation_vector > 0).astype(int)
    sparsity = 1.0 - (np.count_nonzero(activation_vector) / len(activation_vector))
    activation_norm = float(np.linalg.norm(activation_vector))
    n_active_neurons = int(np.count_nonzero(activation_vector))
    
    return {
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


def extract_layer_activations(model: LanguageModel,
                            text: str,
                            layer: int,
                            position: int = -1) -> np.ndarray:
    """
    Extract activations from a specific layer and position
    
    Args:
        model: nnsight LanguageModel
        text: Input text
        layer: Layer index
        position: Token position (-1 for last token)
        
    Returns:
        Activation vector as numpy array
    """
    try:
        with model.trace() as tracer:
            inputs = model.tokenizer(text, return_tensors="pt")
            output = model(**inputs)
            
            # Get layer activations
            layer_output = model.gpt_neox.layers[layer].output[0]
            
            # Extract position
            if position == -1:
                position = layer_output.shape[1] - 1
            
            activation_vector = layer_output[0, position, :].detach().cpu().numpy()
            
        return activation_vector
        
    except Exception as e:
        warnings.warn(f"Error extracting activations: {e}")
        return np.array([])


def extract_activations_for_ngram(model_name: str,
                                checkpoint: str,
                                text: str,
                                ngram: str,
                                target_layers: List[int],
                                category: str = "unknown",
                                text_idx: int = 0) -> List[Dict[str, Any]]:
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
        # Load model at checkpoint
        model = LanguageModel(model_name, revision=f"step{checkpoint}")
        
        # Find n-gram matches in text
        matches = find_ngram_matches(text, ngram, strategy="comprehensive")
        
        if not matches:
            return records
        
        # Use best match
        char_start, char_end, confidence, matched_text = matches[0]
        
        # Get token position (approximate)
        tokens = model.tokenizer(text[:char_start], return_tensors="pt")
        token_position = tokens['input_ids'].shape[1] - 1
        
        # Extract activations from each target layer
        for layer in target_layers:
            try:
                activation_vector = extract_layer_activations(model, text, layer, token_position)
                
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
                        activation_vector=activation_vector
                    )
                    records.append(record)
                    
            except Exception as e:
                warnings.warn(f"Error extracting layer {layer}: {e}")
                continue
        
        # Clean up model
        del model
        torch.cuda.empty_cache()
        
    except Exception as e:
        warnings.warn(f"Error loading model {model_name} at step {checkpoint}: {e}")
    
    return records


def extract_activations_from_dataset(model_name: str,
                                   checkpoints: List[str],
                                   dataset: Dict[str, List[Any]],
                                   target_layers: List[int] = None,
                                   batch_size: int = 1) -> List[Dict[str, Any]]:
    """
    Extract activations from a complete dataset across checkpoints
    
    Args:
        model_name: Model identifier
        checkpoints: List of checkpoint steps
        dataset: Dataset with 'texts', 'ngrams', 'categories' keys
        target_layers: Layer indices to analyze
        batch_size: Batch size for processing
        
    Returns:
        List of all activation records
    """
    
    if target_layers is None:
        target_layers = [0, 2, 4]  # Default layers
    
    all_records = []
    total_texts = len(dataset['texts'])
    
    print(f"Extracting activations for {len(checkpoints)} checkpoints")
    print(f"Dataset size: {total_texts} samples")
    print(f"Target layers: {target_layers}")
    
    for checkpoint_idx, checkpoint in enumerate(checkpoints):
        print(f"\nProcessing checkpoint {checkpoint_idx+1}/{len(checkpoints)}: step{checkpoint}")
        
        checkpoint_records = []
        
        for text_idx in tqdm(range(total_texts), desc=f"Checkpoint {checkpoint}"):
            text = dataset['texts'][text_idx]
            ngram = dataset['ngrams'][text_idx]
            category = dataset['categories'][text_idx] if 'categories' in dataset else "unknown"
            
            # Extract activations for this text/ngram combination
            text_records = extract_activations_for_ngram(
                model_name=model_name,
                checkpoint=checkpoint,
                text=text,
                ngram=ngram,
                target_layers=target_layers,
                category=category,
                text_idx=text_idx
            )
            
            checkpoint_records.extend(text_records)
        
        print(f"Extracted {len(checkpoint_records)} activation records for checkpoint {checkpoint}")
        all_records.extend(checkpoint_records)
    
    print(f"\nTotal activation records extracted: {len(all_records)}")
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

# Example usage
def main():
    """Example usage of simplified checkpoint analysis"""
    
    # Example data
    model_name = "EleutherAI/pythia-70m"
    checkpoint = "1000"
    
    

if __name__ == "__main__":
    main()