#!/usr/bin/env python3
"""
Extract model persona, user persona, and I-Thou vectors.

This script extracts three types of vectors:
1. Model persona vector: "I am X" (model instructed to have trait)
2. User persona vector: "You are X" (model responds to user with trait)
3. I-Thou vector: Model persona - User persona

Usage:
    python scripts/extract_vectors.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --trait evil \
        --responses_dir data/responses \
        --output_dir data/vectors \

Output:
    Creates vector files in output_dir/{model_short_name}/:
    - {trait}_model_persona_{position}.pt
    - {trait}_user_persona_{position}.pt
    - {trait}_i_thou_{position}.pt
"""

import sys
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import pandas as pd

from ithou.models import ModelWrapper
from ithou.extraction import (
    extract_persona_vectors,
    compute_i_thou_vector,
    compute_cosine_similarity_layerwise,
)
from ithou.scoring import filter_responses
from ithou.utils import (
    setup_logging,
    set_seed,
    ensure_dir,
    load_trait_config,
    logger,
)

def main(
    model: str,
    trait: str,
    responses_dir: str = "data/responses/",
    output_dir: str = "data/vectors",
    positions: list[str] = ["prompt_end", "response_start", "response_avg"],
    layers: Optional[list[int]] = None,
    batch_size: int = 8,
    trait_threshold: int = 50,
    coherence_threshold: int = 50,
    seed: int = 42,
    log_level: str = "INFO",
):
    # Setup
    setup_logging(log_level)
    set_seed(seed)
    
    # Setup output directory
    output_path = ensure_dir(Path(output_dir) / model.split('/')[-1])
    
    # Load response data
    responses_path = Path(responses_dir)
    pos_csv = responses_path / f"{model.split('/')[-1]}_{trait}_positive.csv"
    neg_csv = responses_path / f"{model.split('/')[-1]}_{trait}_negative.csv"

    pos_df, neg_df = filter_responses(
            str(pos_csv),
            str(neg_csv),
            trait,
            f"model_persona_response_{trait}_score",
            f"model_persona_response_coherence",
            trait_threshold=trait_threshold,
            coherence_threshold=coherence_threshold,
        )
    
    # Load model
    logger.info(f"Loading model: {model}")
    model_wrapper = ModelWrapper(model)
    
    # Determine layers
    if layers is None:
        layers = list(range(model_wrapper.num_layers + 1))
    
    logger.info(f"Extracting from layers: {layers}")
    
    # Extract model persona vectors
    logger.info("\n" + "="*60)
    logger.info("EXTRACTING MODEL PERSONA VECTORS")
    logger.info("="*60)
    
    model_persona_vectors = extract_persona_vectors(
        model=model_wrapper,
        pos_df=pos_df,
        neg_df=neg_df,
        positions=positions,
        prompt_column="model_persona_prompt",
        response_column="model_persona_response",
        layers=layers,
        batch_size=batch_size,
    )
    
    # Save model persona vectors
    for position, vec in model_persona_vectors.items():
        path = output_path / f"{trait}_model_persona_{position}.pt"
        torch.save(vec, path)
        logger.info(f"Saved model persona vector to {path}")
    
    # Extract user persona vectors
    logger.info("\n" + "="*60)
    logger.info("EXTRACTING USER PERSONA VECTORS")
    logger.info("="*60)

    user_persona_vectors = extract_persona_vectors(
        model=model_wrapper,
        pos_df=pos_df,
        neg_df=neg_df,
        positions=positions,
        prompt_column="user_persona_prompt",
        response_column="user_persona_response",
        layers=layers,
        batch_size=batch_size,
    )
        
    # Save user persona vectors
    for position, vec in user_persona_vectors.items():
        path = output_path / f"{trait}_user_persona_{position}.pt"
        torch.save(vec, path)
        logger.info(f"Saved user persona vector to {path}")
    
    # Compute I-Thou vectors
    logger.info("\n" + "="*60)
    logger.info("COMPUTING I-THOU VECTORS")
    logger.info("="*60)
    
    for position in positions:
        model_vec = model_persona_vectors[position]
        user_vec = user_persona_vectors[position]
        
        i_thou_vec = compute_i_thou_vector(model_vec, user_vec)
        
        path = output_path / f"{trait}_i_thou_{position}.pt"
        torch.save(i_thou_vec, path)
        logger.info(f"Saved I-Thou vector to {path}")
        
        # Compute and log cosine similarity between model and user personas
        similarities = compute_cosine_similarity_layerwise(model_vec, user_vec, layers)
        
        logger.info(f"\nCosine similarity (model vs user persona) at {position}:")
        for layer, sim in sorted(similarities.items()):
            logger.info(f"  Layer {layer:2d}: {sim:+.4f}")
    
    print(f"\nExtraction complete! Vectors saved to {output_path}/")


if __name__ == "__main__":
    import fire
    fire.Fire(main)
