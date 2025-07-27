#!/usr/bin/env python3
"""
Improved Checkpoint Analysis for Polytope Evolution
Enhanced implementation for extracting position-based activations with robust data structures
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
import pickle
from dataclasses import dataclass, asdict
import warnings
from pathlib import Path


@dataclass
class ActivationRecord:
    """Structured record for individual activation data"""
    checkpoint_step: str
    text_idx: int
    text: str
    ngram: str
    matched_text: str
    match_confidence: float
    matching_strategy: str
    category: str
    layer: int
    neuron_idx: Optional[int]  # For individual neuron analysis
    token_position: int
    char_start: int
    char_end: int
    activation_vector: np.ndarray
    binary_pattern: np.ndarray
    sparsity: float
    activation_norm: float
    n_active_neurons: int
    metadata: Dict[str, Any]


class ActivationDataManager:
    """Manager for organizing and accessing activation data efficiently"""
    
    def __init__(self):
        self.records: List[ActivationRecord] = []
        self._layer_index: Dict[int, List[int]] = defaultdict(list)
        self._category_index: Dict[str, List[int]] = defaultdict(list)
        self._checkpoint_index: Dict[str, List[int]] = defaultdict(list)
        self._ngram_index: Dict[str, List[int]] = defaultdict(list)
    
    def add_record(self, record: ActivationRecord) -> int:
        """Add a record and update indices"""
        record_idx = len(self.records)
        self.records.append(record)
        
        # Update indices
        self._layer_index[record.layer].append(record_idx)
        self._category_index[record.category].append(record_idx)
        self._checkpoint_index[record.checkpoint_step].append(record_idx)
        self._ngram_index[record.ngram].append(record_idx)
        
        return record_idx
    
    def get_by_layer(self, layer: int) -> List[ActivationRecord]:
        """Get all records for a specific layer"""
        indices = self._layer_index[layer]
        return [self.records[i] for i in indices]
    
    def get_by_category(self, category: str) -> List[ActivationRecord]:
        """Get all records for a specific category"""
        indices = self._category_index[category]
        return [self.records[i] for i in indices]
    
    def get_by_checkpoint(self, checkpoint: str) -> List[ActivationRecord]:
        """Get all records for a specific checkpoint"""
        indices = self._checkpoint_index[checkpoint]
        return [self.records[i] for i in indices]
    
    def get_by_ngram(self, ngram: str) -> List[ActivationRecord]:
        """Get all records for a specific n-gram"""
        indices = self._ngram_index[ngram]
        return [self.records[i] for i in indices]
    
    def get_layer_activations_matrix(self, layer: int, checkpoint: str = None) -> np.ndarray:
        """Get activation matrix for a layer (samples x neurons)"""
        records = self.get_by_layer(layer)
        
        if checkpoint:
            records = [r for r in records if r.checkpoint_step == checkpoint]
        
        if not records:
            return np.array([])
        
        # Stack activation vectors
        activations = np.stack([r.activation_vector for r in records])
        return activations
    
    def get_neuron_activations_timeseries(self, layer: int, neuron_idx: int) -> Dict[str, List[float]]:
        """Get activation timeseries for a specific neuron across checkpoints"""
        layer_records = self.get_by_layer(layer)
        
        timeseries = defaultdict(list)
        
        for record in layer_records:
            if neuron_idx < len(record.activation_vector):
                activation_value = record.activation_vector[neuron_idx]
                timeseries[record.checkpoint_step].append(activation_value)
        
        return dict(timeseries)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert records to pandas DataFrame for analysis"""
        data = []
        
        for record in self.records:
            row = {
                'checkpoint_step': record.checkpoint_step,
                'text_idx': record.text_idx,
                'ngram': record.ngram,
                'matched_text': record.matched_text,
                'match_confidence': record.match_confidence,
                'category': record.category,
                'layer': record.layer,
                'token_position': record.token_position,
                'sparsity': record.sparsity,
                'activation_norm': record.activation_norm,
                'n_active_neurons': record.n_active_neurons,
                'text_length': len(record.text),
                'ngram_length': len(record.ngram.split())
            }
            
            # Add metadata fields
            row.update(record.metadata)
            data.append(row)
        
        return pd.DataFrame(data)
    
    def save(self, filepath: str):
        """Save manager to disk"""
        data = {
            'records': [asdict(record) for record in self.records],
            'layer_index': dict(self._layer_index),
            'category_index': dict(self._category_index),
            'checkpoint_index': dict(self._checkpoint_index),
            'ngram_index': dict(self._ngram_index)
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    @classmethod
    def load(cls, filepath: str) -> 'ActivationDataManager':
        """Load manager from disk"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        manager = cls()
        
        # Reconstruct records
        for record_data in data['records']:
            # Convert numpy arrays back
            record_data['activation_vector'] = np.array(record_data['activation_vector'])
            record_data['binary_pattern'] = np.array(record_data['binary_pattern'])
            record = ActivationRecord(**record_data)
            manager.records.append(record)
        
        # Reconstruct indices
        manager._layer_index = defaultdict(list, data['layer_index'])
        manager._category_index = defaultdict(list, data['category_index'])
        manager._checkpoint_index = defaultdict(list, data['checkpoint_index'])
        manager._ngram_index = defaultdict(list, data['ngram_index'])
        
        return manager


class ImprovedNGramMatcher:
    """Enhanced n-gram matching with multiple strategies and confidence scoring"""
    
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer
        
    def find_ngram_matches(self, text: str, ngram: str, 
                          strategy: str = "comprehensive") -> List[Tuple[int, int, float, str]]:
        """Find n-gram matches using specified strategy"""
        
        if strategy == "exact":
            return self._find_exact_matches(text, ngram)
        elif strategy == "case_insensitive":
            return self._find_case_insensitive_matches(text, ngram)
        elif strategy == "flexible":
            return self._find_flexible_matches(text, ngram)
        elif strategy == "token_based" and self.tokenizer:
            return self._find_token_based_matches(text, ngram)
        elif strategy == "comprehensive":
            return self._find_comprehensive_matches(text, ngram)
        else:
            return self._find_flexible_matches(text, ngram)
    
    def _find_exact_matches(self, text: str, ngram: str) -> List[Tuple[int, int, float, str]]:
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
    
    def _find_case_insensitive_matches(self, text: str, ngram: str) -> List[Tuple[int, int, float, str]]:
        """Find case-insensitive matches"""
        matches = []
        text_lower = text.lower()
        ngram_lower = ngram.lower()
        start = 0
        
        while True:
            pos = text_lower.find(ngram_lower, start)
            if pos == -1:
                break
            
            matched_text = text[pos:pos + len(ngram)]
            confidence = 0.95 if matched_text == ngram else 0.85
            matches.append((pos, pos + len(ngram), confidence, matched_text))
            start = pos + 1
        
        return matches
    
    def _find_flexible_matches(self, text: str, ngram: str) -> List[Tuple[int, int, float, str]]:
        """Find matches with flexible preprocessing"""
        matches = []
        
        # Strategy 1: Normalize whitespace
        normalized_text = re.sub(r'\s+', ' ', text)
        normalized_ngram = re.sub(r'\s+', ' ', ngram)
        
        pos = normalized_text.lower().find(normalized_ngram.lower())
        if pos != -1:
            # Map back to original position (approximate)
            original_pos = self._map_normalized_position(text, normalized_text, pos)
            if original_pos != -1:
                end_pos = original_pos + len(ngram)
                matched_text = text[original_pos:end_pos]
                matches.append((original_pos, end_pos, 0.8, matched_text))
        
        # Strategy 2: Word boundary matching
        if not matches:
            word_matches = self._find_word_boundary_matches(text, ngram)
            matches.extend(word_matches)
        
        # Strategy 3: Fuzzy matching for difficult cases
        if not matches:
            fuzzy_matches = self._find_fuzzy_matches(text, ngram, threshold=0.8)
            matches.extend(fuzzy_matches)
        
        return matches
    
    def _find_token_based_matches(self, text: str, ngram: str) -> List[Tuple[int, int, float, str]]:
        """Find matches at token level"""
        try:
            # Tokenize text and ngram
            text_encoding = self.tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
            ngram_tokens = self.tokenizer.encode(ngram, add_special_tokens=False)
            
            text_tokens = text_encoding['input_ids']
            offset_mapping = text_encoding['offset_mapping']
            
            matches = []
            
            # Find token sequence matches
            for i in range(len(text_tokens) - len(ngram_tokens) + 1):
                if text_tokens[i:i + len(ngram_tokens)] == ngram_tokens:
                    # Get character positions from offset mapping
                    char_start = offset_mapping[i][0]
                    char_end = offset_mapping[i + len(ngram_tokens) - 1][1]
                    matched_text = text[char_start:char_end]
                    matches.append((char_start, char_end, 0.95, matched_text))
            
            return matches
        
        except Exception as e:
            warnings.warn(f"Token-based matching failed: {e}")
            return []
    
    def _find_comprehensive_matches(self, text: str, ngram: str) -> List[Tuple[int, int, float, str]]:
        """Try multiple strategies and return best matches"""
        all_matches = []
        
        # Try strategies in order of preference
        strategies = [
            ("exact", self._find_exact_matches),
            ("case_insensitive", self._find_case_insensitive_matches),
            ("token_based", self._find_token_based_matches),
            ("flexible", self._find_flexible_matches)
        ]
        
        for strategy_name, strategy_func in strategies:
            if strategy_name == "token_based" and not self.tokenizer:
                continue
            
            matches = strategy_func(text, ngram)
            if matches:
                # Add strategy info to matches
                matches = [(start, end, conf * 0.95, matched, strategy_name) 
                          for start, end, conf, matched in matches]
                all_matches.extend(matches)
        
        # Remove duplicates and sort by confidence
        unique_matches = []
        seen_positions = set()
        
        for match in sorted(all_matches, key=lambda x: x[2], reverse=True):
            start, end, conf, matched, strategy = match
            pos_key = (start, end)
            
            if pos_key not in seen_positions:
                unique_matches.append((start, end, conf, matched))
                seen_positions.add(pos_key)
        
        return unique_matches[:5]  # Return top 5 matches
    
    def _find_word_boundary_matches(self, text: str, ngram: str) -> List[Tuple[int, int, float, str]]:
        """Find matches respecting word boundaries"""
        matches = []
        
        # Escape special regex characters
        escaped_ngram = re.escape(ngram)
        
        patterns = [
            rf'\b{escaped_ngram}\b',  # Full word boundaries
            rf'\b{escaped_ngram}',    # Start boundary only
            rf'{escaped_ngram}\b'     # End boundary only
        ]
        
        confidences = [0.9, 0.7, 0.7]
        
        for pattern, confidence in zip(patterns, confidences):
            for match in re.finditer(pattern, text, re.IGNORECASE):
                matches.append((match.start(), match.end(), confidence, match.group()))
        
        return matches
    
    def _find_fuzzy_matches(self, text: str, ngram: str, threshold: float = 0.8) -> List[Tuple[int, int, float, str]]:
        """Find fuzzy matches using sequence similarity"""
        matches = []
        ngram_len = len(ngram)
        
        # Slide window across text
        for i in range(len(text) - ngram_len + 1):
            window = text[i:i + ngram_len]
            similarity = SequenceMatcher(None, ngram.lower(), window.lower()).ratio()
            
            if similarity >= threshold:
                confidence = similarity * 0.6  # Lower confidence for fuzzy matches
                matches.append((i, i + ngram_len, confidence, window))
        
        return matches
    
    def _map_normalized_position(self, original: str, normalized: str, norm_pos: int) -> int:
        """Map position from normalized text back to original"""
        if len(normalized) == 0:
            return -1
        
        # Simple proportional mapping
        ratio = norm_pos / len(normalized)
        return max(0, min(int(ratio * len(original)), len(original) - 1))


class ImprovedCheckpointAnalyzer:
    """Enhanced checkpoint analyzer with robust activation extraction"""
    
    def __init__(self, cache_dir: str = "./activation_cache"):
        self.cache_dir = Path(cache_dir)
        self.data_manager = ActivationDataManager()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def extract_activations_from_checkpoints(self,
                                           model_name: str,
                                           checkpoints: List[str],
                                           dataset: Dict[str, List[Any]],
                                           target_layers: List[int] = None,
                                           matching_strategy: str = "comprehensive",
                                           batch_size: int = 1) -> ActivationDataManager:
        """Extract activations with improved data management"""
        
        print(f"Starting activation extraction for {len(checkpoints)} checkpoints")
        print(f"Dataset size: {len(dataset['texts'])} samples")
        
        for checkpoint_idx, checkpoint in enumerate(checkpoints):
            print(f"\n{'='*50}")
            print(f"Processing checkpoint {checkpoint_idx+1}/{len(checkpoints)}: step{checkpoint}")
            print(f"{'='*50}")
            
            try:
                # Load model
                model = self._load_model_checkpoint(model_name, checkpoint)
                
                # Initialize matcher with tokenizer
                matcher = ImprovedNGramMatcher(model.tokenizer)
                
                # Set target layers
                if target_layers is None:
                    target_layers = self._get_target_layers(model)
                
                print(f"Analyzing layers: {target_layers}")
                
                # Process dataset samples
                successful_extractions = 0
                failed_extractions = 0
                
                for text_idx in tqdm(range(len(dataset['texts'])), 
                                   desc=f"Processing texts for checkpoint {checkpoint}"):
                    
                    try:
                        result = self._extract_single_sample_activations(
                            model=model,
                            matcher=matcher,
                            text_idx=text_idx,
                            dataset=dataset,
                            checkpoint=checkpoint,
                            target_layers=target_layers,
                            matching_strategy=matching_strategy
                        )
                        
                        if result:
                            successful_extractions += 1
                        else:
                            failed_extractions += 1
                        
                    except Exception as e:
                        print(f"Error processing text {text_idx}: {e}")
                        failed_extractions += 1
                        continue
                
                print(f"Checkpoint {checkpoint} completed:")
                print(f"  Successful extractions: {successful_extractions}")
                print(f"  Failed extractions: {failed_extractions}")
                
                # Clean up GPU memory
                del model
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"Error loading checkpoint {checkpoint}: {e}")
                continue
        
        print(f"\nExtraction complete. Total records: {len(self.data_manager.records)}")
        return self.data_manager
    
    def _load_model_checkpoint(self, model_name: str, checkpoint: str) -> LanguageModel:
        """Load model checkpoint with error handling"""
        try:
            revision = f"step{checkpoint}"
            model = LanguageModel(model_name, revision=revision, device_map="auto")
            return model
        except Exception as e:
            print(f"Failed to load {model_name} at {revision}: {e}")
            raise
    
    def _get_target_layers(self, model: LanguageModel) -> List[int]:
        """Get target layers for analysis"""
        num_layers = model.config.num_hidden_layers
        
        # Select representative layers (25%, 50%, 75%, final)
        target_layers = [
            int(0.25 * num_layers),
            int(0.5 * num_layers),
            int(0.75 * num_layers),
            num_layers - 1
        ]
        
        # Remove duplicates and ensure valid indices
        target_layers = sorted(list(set([l for l in target_layers if 0 <= l < num_layers])))
        return target_layers
    
    def _extract_single_sample_activations(self,
                                         model: LanguageModel,
                                         matcher: ImprovedNGramMatcher,
                                         text_idx: int,
                                         dataset: Dict[str, List[Any]],
                                         checkpoint: str,
                                         target_layers: List[int],
                                         matching_strategy: str) -> bool:
        """Extract activations for a single sample"""
        
        # Get sample data
        text = dataset['texts'][text_idx]
        ngram = dataset['ngrams'][text_idx]
        category = dataset['categories'][text_idx]
        
        # Find ngram matches
        matches = matcher.find_ngram_matches(text, ngram, matching_strategy)
        
        if not matches:
            return False
        
        # Use the best match
        char_start, char_end, confidence, matched_text = matches[0]
        
        # Tokenize text to find token positions
        try:
            encoding = model.tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
            tokens = encoding['input_ids']
            offset_mapping = encoding['offset_mapping']
            
            # Find token span for ngram
            token_start_idx, token_end_idx = self._find_token_span_from_offsets(
                char_start, char_end, offset_mapping
            )
            
            if token_start_idx == -1 or token_end_idx == -1:
                return False
            
            # Use the last token of the ngram for activation extraction
            target_token_idx = token_end_idx
            
        except Exception as e:
            print(f"Tokenization error for text {text_idx}: {e}")
            return False
        
        # Extract activations using nnsight
        try:
            with model.trace(text) as tracer:
                saved_activations = {}
                
                for layer_idx in target_layers:
                    # Get MLP output - handle different model architectures
                    mlp_output = self._get_mlp_output(model, layer_idx)
                    saved_activations[layer_idx] = mlp_output.save()
            
            # Process and store activations
            for layer_idx in target_layers:
                if layer_idx in saved_activations:
                    layer_acts = saved_activations[layer_idx].value
                    
                    # Validate sequence length
                    if layer_acts.shape[1] <= target_token_idx:
                        continue
                    
                    # Extract activation vector
                    activation_vector = layer_acts[0, target_token_idx, :].cpu().detach().numpy()
                    
                    # Compute statistics
                    binary_pattern = (activation_vector > 0).astype(int)
                    sparsity = binary_pattern.mean()
                    activation_norm = np.linalg.norm(activation_vector)
                    n_active = binary_pattern.sum()
                    
                    # Create activation record
                    record = ActivationRecord(
                        checkpoint_step=checkpoint,
                        text_idx=text_idx,
                        text=text,
                        ngram=ngram,
                        matched_text=matched_text,
                        match_confidence=confidence,
                        matching_strategy=matching_strategy,
                        category=category,
                        layer=layer_idx,
                        neuron_idx=None,  # Will be set for individual neuron analysis
                        token_position=target_token_idx,
                        char_start=char_start,
                        char_end=char_end,
                        activation_vector=activation_vector,
                        binary_pattern=binary_pattern,
                        sparsity=sparsity,
                        activation_norm=activation_norm,
                        n_active_neurons=n_active,
                        metadata={
                            'text_length': len(text),
                            'ngram_word_count': len(ngram.split()),
                            'token_span': (token_start_idx, token_end_idx),
                            'model_name': model.model.name_or_path if hasattr(model.model, 'name_or_path') else 'unknown'
                        }
                    )
                    
                    # Add to data manager
                    self.data_manager.add_record(record)
            
            return True
            
        except Exception as e:
            print(f"Activation extraction error for text {text_idx}: {e}")
            return False
    
    def _get_mlp_output(self, model: LanguageModel, layer_idx: int):
        """Get MLP output for different model architectures"""
        try:
            # Try common architectures
            if hasattr(model, 'gpt_neox') and hasattr(model.gpt_neox, 'layers'):
                # GPT-NeoX style (Pythia)
                return model.gpt_neox.layers[layer_idx].mlp.output
            elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
                # GPT-2 style
                return model.transformer.h[layer_idx].mlp.output
            elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
                # LLaMA style
                return model.model.layers[layer_idx].mlp.output
            else:
                # Generic fallback
                layers = model.model.layers if hasattr(model.model, 'layers') else model.transformer.h
                return layers[layer_idx].mlp.output
        except Exception as e:
            raise ValueError(f"Could not access MLP output for layer {layer_idx}: {e}")
    
    def _find_token_span_from_offsets(self, char_start: int, char_end: int, 
                                    offset_mapping: List[Tuple[int, int]]) -> Tuple[int, int]:
        """Find token span using offset mapping"""
        token_start_idx = -1
        token_end_idx = -1
        
        for i, (token_char_start, token_char_end) in enumerate(offset_mapping):
            # Find first token that overlaps with ngram start
            if token_start_idx == -1 and token_char_start <= char_start < token_char_end:
                token_start_idx = i
            
            # Find last token that overlaps with ngram end
            if token_char_start < char_end <= token_char_end:
                token_end_idx = i
                break
            elif token_char_start >= char_end and token_end_idx == -1:
                token_end_idx = max(0, i - 1)
                break
        
        # Handle edge case where ngram ends at text end
        if token_end_idx == -1 and token_start_idx != -1:
            token_end_idx = len(offset_mapping) - 1
        
        return token_start_idx, token_end_idx
    
    def save_results(self, filepath: str):
        """Save analysis results"""
        self.data_manager.save(filepath)
        print(f"Saved activation data to {filepath}")
        
        # Also save as DataFrame for easy analysis
        df = self.data_manager.to_dataframe()
        csv_path = filepath.replace('.pkl', '.csv')
        df.to_csv(csv_path, index=False)
        print(f"Saved CSV data to {csv_path}")
    
    def load_results(self, filepath: str):
        """Load analysis results"""
        self.data_manager = ActivationDataManager.load(filepath)
        print(f"Loaded activation data from {filepath}")


def main():
    """Example usage"""
    
    # Load your dataset (from the improved ngram dataset builder)
    # This would come from your ImprovedNGramDatasetBuilder
    example_dataset = {
        'texts': ["Example text here", "Another example"],
        'ngrams': ["example text", "another example"],
        'categories': ["medium", "low"],
        'positions': [0, 0]
    }
    
    # Initialize analyzer
    analyzer = ImprovedCheckpointAnalyzer()
    
    # Extract activations
    data_manager = analyzer.extract_activations_from_checkpoints(
        model_name="EleutherAI/pythia-70m",
        checkpoints=["1000", "2000", "3000"],
        dataset=example_dataset,
        target_layers=[0, 2, 4],  # Specify layers or None for auto-selection
        matching_strategy="comprehensive"
    )
    
    # Save results
    analyzer.save_results("./activation_results.pkl")
    
    # Analyze results
    df = data_manager.to_dataframe()
    print("\nExtraction Summary:")
    print(f"Total records: {len(df)}")
    print(f"Unique checkpoints: {df['checkpoint_step'].nunique()}")
    print(f"Unique layers: {df['layer'].nunique()}")
    print(f"Mean activation sparsity: {df['sparsity'].mean():.3f}")
    
    return data_manager


if __name__ == "__main__":
    data_manager = main()