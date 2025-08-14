"""
Neuron Embeddings Toolkit - based on Foote's (2024) work on polysemanticity.

This toolkit helps you analyze how neurons in a language model, such as GPT-2 or Pythia, respond 
to various inputs by calculating a metric Foote (2024) labels as "neuron embeddings" (NE).
The main idea is that we can understand what a neuron's reaction to a text excerpt is by calculating 
the Hadamard product of its input weights and the vector representation it receives (pre-MLP activations).

Based on this metric, we can select the top-k excerpts that a neuron is most sensitve to, and then
cluster these excerpts (using HAC) based on their semantic similarity. This allows us to measure how
"polysemantic" a neuron is. We can get various metrics, including # of clusters, size of clusters,
intra- and inter-cluster distances, etc. 

What's in here:
- `neuron_embedding`: calculates NE for a single neuron
- `EmbeddingCollector`: grabs high-activation examples and their embeddings
- `cluster_embeddings`: groups similar embeddings together using hierarchical agglomerative clustering (HAC)
- `polysemanticity_metrics`: returns various polysemanticity metrics
"""
from __future__ import annotations

import math
from typing import List, Dict, Iterable, Optional

import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.nn.functional as F # type: ignore
import numpy as np # type: ignore
from sklearn.metrics import pairwise_distances # type: ignore
from sklearn.cluster import AgglomerativeClustering # type: ignore

try:
    from transformer_lens import HookedTransformer # type: ignore
    TRANSFORMER_LENS_AVAILABLE = True
except ImportError:
    TRANSFORMER_LENS_AVAILABLE = False
    HookedTransformer = None

# -----------------------------------------------------------------------------
# Core building block - NE calculation
# -----------------------------------------------------------------------------

def neuron_embedding(pre_mlp: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    weights = weights.to(pre_mlp)
    return pre_mlp * weights

# -----------------------------------------------------------------------------
# Get embeddings from high-activation examples
# -----------------------------------------------------------------------------

class TransformerLensEmbeddingCollector:
    """Gets high-activation examples using TransformerLens's built-in caching.
    
    This is optimized to take advantage of TransformerLens's native caching
    to get neuron activations directly without using custom hooks.
    """
    
    def __init__(self,
                 model: 'HookedTransformer',  # TransformerLens model
                 layer_name: str,
                 neuron_idx: int,
                 activation_threshold: float = 0.75,
                 peak_activation: Optional[float] = None,
                 max_examples: int = 100,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 decode_text: bool = True):
        
        if not TRANSFORMER_LENS_AVAILABLE:
            raise ImportError("Install transformer-lens with: pip install transformer-lens")
        
        if not hasattr(model, 'run_with_cache'):
            raise ValueError("Model must be a TransformerLens HookedTransformer to use TransformerLensEmbeddingCollector")
        self.model = model.to(device).eval()
        self.layer_name = layer_name
        self.neuron_idx = neuron_idx
        self.max_examples = max_examples
        self.device = device
        
        # Set activation threshold based on peak activation if provided
        if peak_activation is not None:
            self.activation_threshold = activation_threshold * peak_activation
        else:
            self.activation_threshold = activation_threshold

        self._pre_mlp_cache: List[torch.Tensor] = []
        self._activation_cache: List[float] = []
        self._text_cache: List[str] = []  # Store the actual text examples
        self.decode_text = decode_text
        
        # Get the weight vector for this neuron
        self._setup_weight_vector()
    
    def _setup_weight_vector(self):
        """Extract weight vector for the target neuron."""
        # Get the MLP module
        mlp_name = self.layer_name
        mlp_module = dict(self.model.named_modules())[mlp_name]
        
        # Extract weight vector based on module type
        if hasattr(mlp_module, 'W_in'):
            # TransformerLens MLP module
            if self.neuron_idx >= mlp_module.W_in.shape[1]:
                raise ValueError(f'Neuron index {self.neuron_idx} is out of bounds. Layer has {mlp_module.W_in.shape[1]} neurons')
            self.weight_vector = mlp_module.W_in[:, self.neuron_idx].detach().clone()
        elif hasattr(mlp_module, 'c_fc'):
            # GPT-2 style MLP
            if self.neuron_idx >= mlp_module.c_fc.weight.shape[0]:
                raise ValueError(f'Neuron index {self.neuron_idx} is out of bounds. Layer has {mlp_module.c_fc.weight.shape[0]} neurons')
            self.weight_vector = mlp_module.c_fc.weight[self.neuron_idx].detach().clone()
        else:
            raise ValueError(f'Unsupported module type for {mlp_name}')
    
    def run(self, dataloader: Iterable):
        """Stream data through the model using TransformerLens's built-in caching."""
        with torch.no_grad():
            for batch in dataloader:
                # Handle tokenizer output
                if hasattr(batch, 'keys') and 'input_ids' in batch:
                    input_ids = batch['input_ids'].to(self.device)
                else:
                    input_ids = batch.to(self.device)
                
                # Use TransformerLens's built-in caching to get activations
                # This gets all activations in one go - much more efficient!
                _, cache = self.model.run_with_cache(input_ids)
                
                # Get the specific neuron's activations directly from cache
                # blocks.6.mlp.hook_post contains activations for all neurons in that MLP
                mlp_activations = cache[f"{self.layer_name}.hook_post"]  # Shape: [batch, seq_len, d_mlp]
                neuron_activations = mlp_activations[..., self.neuron_idx]  # Shape: [batch, seq_len]
                
                # Get pre-MLP activations for embedding calculation
                # Try hook_resid_mid first (GPT-2 style), then hook_resid_pre (Pythia style)
                resid_hook_name = f"{self.layer_name.replace('.mlp', '.hook_resid_mid')}"
                if resid_hook_name not in cache:
                    resid_hook_name = f"{self.layer_name.replace('.mlp', '.hook_resid_pre')}"
                resid_mid_activations = cache[resid_hook_name]  # Shape: [batch, seq_len, d_model]
                
                # Find high-activation examples
                max_acts = neuron_activations.amax(dim=-1)  # Max across sequence length
                
                for batch_idx, max_act in enumerate(max_acts):
                    if len(self._activation_cache) >= self.max_examples:
                        break
                    
                    if max_act >= self.activation_threshold:
                        # Find the token with highest activation
                        seq_idx = neuron_activations[batch_idx].argmax()
                        
                        # Get pre-MLP activations for this token
                        pre_mlp = resid_mid_activations[batch_idx, seq_idx]  # Shape: [d_model]
                        
                        # Get the text for this example (optional for speed)
                        if self.decode_text:
                            try:
                                token_ids = input_ids[batch_idx, seq_idx:seq_idx+10]  # Get context around the token
                                text_example = self.model.tokenizer.decode(token_ids, skip_special_tokens=True)
                            except:
                                text_example = f"token_{seq_idx}"  # Fallback if decoding fails
                        else:
                            text_example = f"token_{seq_idx}"  # Skip decoding for speed
                        
                        self._pre_mlp_cache.append(pre_mlp.cpu())
                        self._activation_cache.append(max_act.item())
                        self._text_cache.append(text_example)
                
                if len(self._activation_cache) >= self.max_examples:
                    break
        
        # Create embeddings
        embeds = [neuron_embedding(p, self.weight_vector) for p in self._pre_mlp_cache]
        return torch.stack(embeds).cpu().numpy()

# This isn't used in any scripts, but is here for reference as what I first wrote. 
class EmbeddingCollector:
    """This is the general-purpose version that works with any PyTorch nn.Module.
    You provide:
        - a hook to grab pre-MLP activations
        - a target neuron (layer module + neuron index)
    """
    def __init__(self,
                 model: nn.Module,
                 layer_name: str,
                 neuron_idx: int,
                 activation_threshold: float = 0.75,
                 peak_activation: Optional[float] = None,
                 max_examples: int = 100,
                 device: str = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'):
        self.model = model.to(device).eval()
        self.layer_name = layer_name
        self.neuron_idx = neuron_idx
        self.max_examples = max_examples
        self.device = device
        
        # Set activation threshold based on peak activation if provided
        if peak_activation is not None:
            self.activation_threshold = activation_threshold * peak_activation
        else:
            self.activation_threshold = activation_threshold

        self._pre_mlp_cache: List[torch.Tensor] = []
        self._activation_cache: List[float] = []
        self._input_cache: List = []  # raw inputs for later inspection

        # Resolve layer and grab weight vector
        module = dict(self.model.named_modules())[layer_name]
        
        # Handle different module types
        if hasattr(module, 'weight'):
            # Direct Linear layer: out_dim × in_dim
            if neuron_idx >= module.weight.shape[0]:
                raise ValueError(f'Neuron index {neuron_idx} is out of bounds. Layer has {module.weight.shape[0]} neurons (indices 0-{module.weight.shape[0]-1})')
            self.weight_vector = module.weight[neuron_idx].detach().clone()
        elif hasattr(module, 'W_in'):
            # TransformerLens MLP module with W_in parameter
            if neuron_idx >= module.W_in.shape[1]:  # Check against output dimension
                raise ValueError(f'Neuron index {neuron_idx} is out of bounds. Layer has {module.W_in.shape[1]} neurons (indices 0-{module.W_in.shape[1]-1})')
            # W_in has shape [d_model, d_mlp], so we need the column for this neuron
            # The weight vector should match the input dimensions (768)
            self.weight_vector = module.W_in[:, neuron_idx].detach().clone()
        elif hasattr(module, 'c_fc'):
            # GPT-2 style MLP with c_fc
            if neuron_idx >= module.c_fc.weight.shape[0]:
                raise ValueError(f'Neuron index {neuron_idx} is out of bounds. Layer has {module.c_fc.weight.shape[0]} neurons (indices 0-{module.c_fc.weight.shape[0]-1})')
            self.weight_vector = module.c_fc.weight[neuron_idx].detach().clone()
        else:
            # Try to find Linear layers within the module
            linear_layers = []
            for name, submodule in module.named_modules():
                if hasattr(submodule, 'weight') and isinstance(submodule, torch.nn.Linear):
                    linear_layers.append((name, submodule))
            
            if linear_layers:
                # Use the first linear layer (input layer)
                layer_name, layer_module = linear_layers[0]
                if neuron_idx >= layer_module.weight.shape[0]:
                    raise ValueError(f'Neuron index {neuron_idx} is out of bounds. Layer has {layer_module.weight.shape[0]} neurons (indices 0-{layer_module.weight.shape[0]-1})')
                self.weight_vector = layer_module.weight[neuron_idx].detach().clone()
            else:
                raise ValueError(f'No Linear layer found in {layer_name}. Available submodules: {[name for name, _ in module.named_modules()]}')

        # --- Insert forward hook to capture pre‑MLP activations ---
        # For MLP modules, we need to hook the module itself to get the input
        # The MLP module will handle the forward pass and we can access the input
        hook_module = module

        def hook_fn(_module, _input, output):
            # For MLP modules, _input[0] is the input to the MLP
            # We need to get the input before the first linear layer
            pre_mlp = _input[0].detach()
            
            # For MLP modules, we need to compute the activation manually
            # since the output is after the full MLP, not just the first layer
            if hasattr(_module, 'W_in'):
                # Compute the activation of the specific neuron
                # W_in has shape [d_mlp, d_model], so we need to transpose it for F.linear
                act = F.linear(pre_mlp, _module.W_in.T, _module.b_in)
                neuron_act = act[..., neuron_idx]
            else:
                # For regular linear layers, use the output directly
                act = output.detach()
                neuron_act = act[..., neuron_idx]
            
            max_act = neuron_act.amax().item()

            if len(self._activation_cache) < self.max_examples and max_act >= self.activation_threshold:
                # choose token with highest act in sequence if seq.
                flat_idx = neuron_act.view(-1).argmax()
                pre = pre_mlp.view(-1, pre_mlp.size(-1))[flat_idx]

                self._pre_mlp_cache.append(pre.cpu())
                self._activation_cache.append(max_act)

        self._hook_handle = hook_module.register_forward_hook(hook_fn, with_kwargs=False)

    def run(self, dataloader: Iterable):
        """Stream data through the model until we've collected enough examples."""
        with torch.no_grad():
            for batch in dataloader:
                # Handle tokenizer output (dict with input_ids, attention_mask)
                if hasattr(batch, 'keys') and 'input_ids' in batch:
                    # TransformerLens expects just the input_ids tensor
                    input_ids = batch['input_ids'].to(self.device)
                    self.model(input_ids)  # Pass input_ids directly
                else:
                    batch = batch.to(self.device)
                    self.model(batch)  # forward pass triggers hook
                
                if len(self._activation_cache) >= self.max_examples:
                    break

        # Clean up hook
        self._hook_handle.remove()

        embeds = [neuron_embedding(p, self.weight_vector) for p in self._pre_mlp_cache]
        return torch.stack(embeds).cpu().numpy()

# -----------------------------------------------------------------------------
# Clustering embeddings to explore polysemanticity
# -----------------------------------------------------------------------------

def cluster_embeddings(embeddings: np.ndarray, distance_threshold: float = 0.8):
    """Group similar embeddings together using hierarchical clustering.
    
    Uses cosine distance to measure similarity between embeddings, then groups
    them hierarchically. Returns cluster labels for each embedding.
    """
    # Calculate cosine distance matrix
    dists = pairwise_distances(embeddings, metric='cosine')
    # HAC
    hac = AgglomerativeClustering(
        metric='precomputed',
        linkage='average',
        distance_threshold=distance_threshold,
        n_clusters=None,
    )
    labels = hac.fit_predict(dists)
    return labels

# -----------------------------------------------------------------------------
# Calculate polysemanticity metrics
# -----------------------------------------------------------------------------

def polysemanticity_metrics(embeddings: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Calculate metrics to quantify a neuron's polysemanticity.
    """
    if len(embeddings) != len(labels):
        raise ValueError('embeddings and labels length mismatch')
    dmat = pairwise_distances(embeddings, metric='cosine')
    intra, inter = [], []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            if labels[i] == labels[j]:
                intra.append(dmat[i, j])
            else:
                inter.append(dmat[i, j])
    
    # Calculate average distance across all pairs
    all_distances = []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            all_distances.append(dmat[i, j])
    
    # Calculate cluster sizes
    from collections import Counter
    cluster_sizes = Counter(labels)
    single_element_clusters = sum(1 for size in cluster_sizes.values() if size == 1)
    
    return {
        'embeddings': int(len(embeddings)),
        'mean_dist': float(np.mean(all_distances)) if all_distances else math.nan,
        'mean_intra': float(np.mean(intra)) if intra else math.nan,
        'mean_inter': float(np.mean(inter)) if inter else math.nan,
        'max_dist': float(dmat.max()),
        'num_clusters': int(len(set(labels))),
        'single_element_clusters': int(single_element_clusters),
        'largest_cluster_size': int(max(cluster_sizes.values())),
        'least_cluster_size': int(min(cluster_sizes.values())),
        'cluster_sizes': {int(k): int(v) for k, v in cluster_sizes.items()},
    }