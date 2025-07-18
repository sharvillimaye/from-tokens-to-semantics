"""
Based on Foote (2024) – Tackling Polysemanticity with Neuron Embeddings.

Main components
---------------
1. `neuron_embedding` – calculate Hadamard product of pre‑MLP activations and neuron input weights.
2. `EmbeddingCollector` – utility to grab high‑activation examples + embeddings.
3. `cluster_embeddings` – hierarchical agglomerative clustering (HAC) with cosine distance.
4. `polysemanticity_metrics` – max/mean dist, intra/inter‑cluster dist, # of clusters.
5. `NeuronEmbeddingLoss` – drop‑in loss for training Sparse Auto‑Encoders (SAEs) that penalises polysemantic neurons.
"""
from __future__ import annotations

import math
from typing import List, Dict, Iterable

import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.nn.functional as F # type: ignore
import numpy as np # type: ignore
from sklearn.metrics import pairwise_distances # type: ignore
from sklearn.cluster import AgglomerativeClustering # type: ignore

# -----------------------------------------------------------------------------
# 1.  Core primitive ----------------------------------------------------------------
# -----------------------------------------------------------------------------

def neuron_embedding(pre_mlp: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Compute neuron embedding `e = h ⊙ w` for a single neuron.

    Parameters
    ----------
    pre_mlp : torch.Tensor
        Activations just **before** the MLP layer.
    weights : torch.Tensor
        Input weights of the neuron.

    Returns
    -------
    torch.Tensor
        Hadamard product of pre_mlp and weights.
    """
    weights = weights.to(pre_mlp)
    return pre_mlp * weights

# -----------------------------------------------------------------------------
# 2.  Collecting embeddings ------------------------------------------------------
# -----------------------------------------------------------------------------
class EmbeddingCollector:
    """Collect high‑activation examples and associated neuron embeddings.

    Works with any PyTorch nn.Module. You supply:
        • a *hook* to grab pre‑MLP activations
        • a *target neuron* (layer module + neuron index)

    Example (TransformerLens):
    >>> model = HookedTransformer.from_pretrained('gpt2-small')
    >>> collector = EmbeddingCollector(model,
    ...                                layer_name='blocks.6.mlp',
    ...                                neuron_idx=1234)
    ...
    """

    def __init__(self,
                 model: nn.Module,
                 layer_name: str,
                 neuron_idx: int,
                 activation_threshold: float = 0.75,
                 max_examples: int = 100,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device).eval()
        self.layer_name = layer_name
        self.neuron_idx = neuron_idx
        self.activation_threshold = activation_threshold
        self.max_examples = max_examples
        self.device = device

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
        """Stream data through the model until caches are full."""
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
# 3.  Clustering -----------------------------------------------------------------
# -----------------------------------------------------------------------------

def cluster_embeddings(embeddings: np.ndarray, distance_threshold: float = 0.5):
    """Hierarchical Agglomerative Clustering with cosine distance.
    Returns cluster labels (same length as embeddings)."""
    # Compute cosine distance matrix
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
# 4.  Metrics --------------------------------------------------------------------
# -----------------------------------------------------------------------------

def polysemanticity_metrics(embeddings: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
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
    
    # Calculate mean distance across all pairs
    all_distances = []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            all_distances.append(dmat[i, j])
    
    return {
        'mean_dist': float(np.mean(all_distances)) if all_distances else math.nan,
        'mean_intra': float(np.mean(intra)) if intra else math.nan,
        'mean_inter': float(np.mean(inter)) if inter else math.nan,
        'max_dist': float(dmat.max()),
        'num_clusters': int(len(set(labels))),
    }

# -----------------------------------------------------------------------------
# 5.  SAE training helper --------------------------------------------------------
# -----------------------------------------------------------------------------
class NeuronEmbeddingLoss(nn.Module):
    """Loss term LN (Eq. 4) to encourage monosemanticity in SAE neurons."""

    def __init__(self, lambda_ne: float = 0.1, momentum: float = 0.9):
        super().__init__()
        self.lambda_ne = lambda_ne
        self.momentum = momentum
        self.register_buffer('running_means', torch.empty(0))  # will be resized lazily

    def forward(self, pre_sae: torch.Tensor, sae_weight: torch.Tensor, sae_act: torch.Tensor):
        """Compute LN over a batch.

        Parameters
        ----------
        pre_sae : [batch, d_hidden]
            Embedding before SAE layer.
        sae_weight : [d_sae, d_hidden]
            Encoder weights of SAE layer.
        sae_act : [batch, d_sae]
            Sparse activations (after L1 sparsity but before decoder).
        """
        device = pre_sae.device
        d_sae, d_hidden = sae_weight.shape
        if self.running_means.numel() == 0:
            self.running_means = torch.zeros((d_sae, d_hidden), device=device)

        ln_vals = []
        for b in range(pre_sae.size(0)):
            active = (sae_act[b] != 0).nonzero(as_tuple=False).squeeze(-1)
            if active.numel() == 0:
                continue
            for j in active.tolist():
                wj = sae_weight[j]
                emb_current = pre_sae[b] * wj  # ⊙
                mean_emb = self.running_means[j]
                dist = 1 - F.cosine_similarity(mean_emb, emb_current, dim=0, eps=1e-6)
                ln_vals.append(dist)
                # momentum update
                self.running_means[j] = self.momentum * mean_emb + (1 - self.momentum) * pre_sae[b]
        if not ln_vals:
            return torch.tensor(0.0, device=device)
        ln_batch = torch.stack(ln_vals).mean()
        return self.lambda_ne * ln_batch

# -----------------------------------------------------------------------------
# 6.  Example usage --------------------------------------------------------------
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse, json, pathlib
    from datasets import load_dataset # type: ignore
    from transformer_lens import HookedTransformer # type: ignore

    parser = argparse.ArgumentParser(description="Demo: collect neuron embeddings for GPT2-small")
    parser.add_argument('--neuron', type=str, default='blocks.0.mlp', help='Layer name containing target neuron')
    parser.add_argument('--index', type=int, default=0, help='Neuron index within the layer (row in weight matrix)')
    parser.add_argument('--out', type=pathlib.Path, default='embeddings.json')
    args = parser.parse_args()

    # Load model and tiny text dataset
    model = HookedTransformer.from_pretrained('gpt2-small')
    ds = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    tokenizer = model.tokenizer

    def text_loader(batch_size=4):
        for start in range(0, len(ds), batch_size):
            tokens = tokenizer(ds[start:start+batch_size]['text'], return_tensors='pt', padding=True, truncation=True)
            yield tokens

    collector = EmbeddingCollector(model, args.neuron, args.index)
    embeds = collector.run(text_loader())
    labels = cluster_embeddings(embeds)
    metrics = polysemanticity_metrics(embeds, labels)

    with open(args.out, 'w') as f:
        json.dump({'metrics': metrics, 'labels': labels.tolist()}, f, indent=2)

    print('Saved', args.out)
