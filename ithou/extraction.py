"""
Vector extraction module for persona vectors.

Extracts persona vectors by computing mean difference
at specified token positions (such as: prompt_end, response_start, response_avg).
"""

import torch
import pandas as pd
import numpy as np
from enum import Enum
from typing import Optional

from ithou.models import ModelWrapper
from ithou.utils import logger, cosine_similarity

def extract_persona_vectors(
    model: ModelWrapper,
    pos_df: pd.DataFrame,
    neg_df: pd.DataFrame,
    positions: list[str],
    prompt_column: str,
    response_column: str,
    layers: Optional[list[int]] = None,
    batch_size: int = 8,
) -> dict[str, torch.Tensor]:
    """
    Extract persona vectors by mean difference.
    """
    # Specify layers for extraction, this will impact the later steering.
    if layers is None:
        # Note: typically only mid-to-late layers are involved in steering
        # start_layer = model.num_layers // 2
        # layers = list(range(start_layer, model.num_layers + 1))
        layers = list(range(model.num_layers + 1))
    elif isinstance(layers, int):
        layers = [layers]
    
    logger.info(f"Extracting persona vectors from {len(pos_df)} pos / {len(neg_df)} neg samples")
    logger.info(f"  Using columns: {prompt_column}, {response_column}")
    
    # Extract activations
    pos_acts = model.extract_activations_batched(
        pos_df[prompt_column].tolist(), pos_df[response_column].tolist(), layers, batch_size
    )
    neg_acts = model.extract_activations_batched(
        neg_df[prompt_column].tolist(), neg_df[response_column].tolist(), layers, batch_size
    )
    
    # Compute mean difference for each position
    return _compute_mean_diff(pos_acts, neg_acts, positions, layers)

def compute_i_thou_vector(
    model_persona: torch.Tensor,
    user_persona: torch.Tensor,
) -> torch.Tensor:
    """
    Compute I-Thou vector: model_persona - user_persona.
    
    Positive direction = "I am X toward you", negative = "You are X toward me".
    """
    assert model_persona.shape == user_persona.shape, \
        f"Shape mismatch: {model_persona.shape} vs {user_persona.shape}"
    return model_persona - user_persona


def compute_cosine_similarity_layerwise(
    vec1: torch.Tensor,
    vec2: torch.Tensor,
    layers: Optional[list[int]] = None,
) -> dict[int, float]:
    """Compute cosine similarity between two [num_layers, hidden_dim] tensors at each layer."""
    if layers is None:
        layers = list(range(vec1.shape[0]))
    return {layer: cosine_similarity(vec1[layer], vec2[layer]) for layer in layers}

def _compute_mean_diff(
    pos_acts: dict,
    neg_acts: dict,
    positions: list[str],
    layers: list[int],
) -> dict[str, torch.Tensor]:
    """Compute mean(pos) - mean(neg) for each position and stack across layers."""
    vectors = {}
    for position in positions:
        layer_vecs = []
        for layer in layers:
            pos_mean = torch.stack(pos_acts[position][layer]).mean(dim=0).float()
            neg_mean = torch.stack(neg_acts[position][layer]).mean(dim=0).float()
            layer_vecs.append(pos_mean - neg_mean)
        vectors[position] = torch.stack(layer_vecs)
        logger.info(f"  {position}: {vectors[position].shape}")
    return vectors