#!/usr/bin/env python3
"""
Compare vectors across traits, positions, and models.

This script analyzes relationships between different persona vectors
by computing cosine similarities and generating visualizations.

Usage:
    python scripts/compare_vectors.py \
        --vector1 data/vectors/Qwen2.5-7B-Instruct/evil_model_persona_prompt_end.pt \
        --vector2 data/vectors/Qwen2.5-7B-Instruct/evil_user_persona_prompt_end.pt \

Outputs:
    - Cosine similarity matrices
    - Layer-wise comparison plots
    - Summary statistics
"""

import sys
from pathlib import Path
from glob import glob
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np

from ithou.utils import setup_logging, ensure_dir, cosine_similarity, logger

def _load_vector(path: str) -> torch.Tensor:
    """Load a vector from file."""
    if not Path(path).exists():
        raise FileNotFoundError(f"Vector file not found: {path}")
    vec = torch.load(path, map_location="cpu")
    return vec

def _compute_similarity_matrix(
    vectors: dict[str, torch.Tensor],
    layer: int,
) -> tuple[np.ndarray, list[str]]:
    """
    Compute pairwise cosine similarity matrix at a specific layer.
    
    Returns:
        similarity matrix, list of vector names
    """
    names = list(vectors.keys())
    n = len(names)
    matrix = np.zeros((n, n))
    
    for i, name_i in enumerate(names):
        for j, name_j in enumerate(names):
            vec_i = vectors[name_i]
            vec_j = vectors[name_j]
            
            # Handle different vector shapes
            if vec_i.dim() == 2:
                vec_i = vec_i[layer]
            if vec_j.dim() == 2:
                vec_j = vec_j[layer]
            
            sim = cosine_similarity(vec_i, vec_j)
            matrix[i, j] = sim
    
    return matrix, names

def _compute_split_half_reliability(pos_act: list, neg_act: list, seed: int = 42) -> list[float]:
    """Compute split-half reliability from raw activations."""
    n_samples = pos_act[0].shape[0]
    n_layers = len(pos_act)
    
    torch.manual_seed(seed)
    perm = torch.randperm(n_samples)
    half = n_samples // 2
    idx_a, idx_b = perm[:half], perm[half:2*half]
    
    reliabilities = []
    for layer in range(n_layers):
        vec_a = pos_act[layer][idx_a].mean(0).float() - neg_act[layer][idx_a].mean(0).float()
        vec_b = pos_act[layer][idx_b].mean(0).float() - neg_act[layer][idx_b].mean(0).float()
        rel = torch.nn.functional.cosine_similarity(vec_a.unsqueeze(0), vec_b.unsqueeze(0)).item()
        reliabilities.append(rel)
    return reliabilities

def generate_comparison_report(vectors: dict[str, torch.Tensor]) -> None:
    # Get number of layers from first vector
    first_vec = next(iter(vectors.values()))
    num_layers = first_vec.shape[0]
    
    print("\n" + "=" * 60)
    print("VECTOR COMPARISON REPORT")
    print("=" * 60)
    
    # Vector inventory
    print("\n## Loaded Vectors\n")
    for name, vec in vectors.items():
        print(f"  - {name}: shape {list(vec.shape)}")
    
    # Similarity at each layer
    print("\n## Cosine Similarity by Layer\n")
    
    names = list(vectors.keys())
    print(f"  Comparing: {names[0]} vs {names[1]}\n")
    
    for layer in range(num_layers):
        try:
            matrix, _ = _compute_similarity_matrix(vectors, layer)
            sim = matrix[0, 1]
            bar = "█" * int(abs(sim) * 20)
            sign = "+" if sim >= 0 else "-"
            print(f"  Layer {layer:2d}: {sim:+.4f} {bar}")
        except Exception as e:
            print(f"  Layer {layer:2d}: Error - {e}")
    
    print("\n" + "=" * 60)

def main(
    vector1: str,
    vector2: str,
    log_level: str = "INFO",
):
    # Load vectors
    vec1 = load_vector(vector1)
    vec2 = load_vector(vector2)
    name1 = Path(vector1).stem
    name2 = Path(vector2).stem
    
    n_layers = vec1.shape[0]
    key_layers = list(range(n_layers)) if layers is None else [int(x.strip()) for x in layers.split(",")]
    
    # Load optional data
    pos_act, neg_act = None, None
    if activations:
        data = torch.load(activations, map_location="cpu")
        if "pos" in data:
            pos_act, neg_act = data["pos"], data["neg"]
        elif "evil" in data:
            pos_act, neg_act = data["evil"], data["helpful"]
    
    cross_vec = load_vector(cross_vector) if cross_vector else None
    
    # Compute similarities
    similarities = compute_layerwise_similarity(vec1, vec2)
    reliabilities = compute_split_half_reliability(pos_act, neg_act) if pos_act else None
    cross_sims = compute_layerwise_similarity(vec2, cross_vec) if cross_vec else None
    
    # Print report
    print("\n" + "=" * 70)
    print("VECTOR COMPARISON")
    print("=" * 70)
    print(f"\n  {name1}: {list(vec1.shape)}")
    print(f"  {name2}: {list(vec2.shape)}")
    if cross_vec is not None:
        print(f"  {Path(cross_vector).stem}: {list(cross_vec.shape)}")
    
    # Header
    print("\n" + "-" * 70)
    if reliabilities and cross_sims:
        print(f"{'Layer':<8} {'Similarity':<12} {'Split-Half':<12} {'Ratio':<10} {'Cross':<10}")
    elif reliabilities:
        print(f"{'Layer':<8} {'Similarity':<12} {'Split-Half':<12} {'Ratio':<10}")
    elif cross_sims:
        print(f"{'Layer':<8} {'Similarity':<12} {'Cross':<10}")
    else:
        print(f"{'Layer':<8} {'Similarity':<12}")
    print("-" * 70)
    
    # Print rows
    for layer in key_layers:
        sim = similarities[layer]
        bar = "█" * int(abs(sim) * 20)
        
        if reliabilities and cross_sims:
            rel = reliabilities[layer]
            ratio = sim / rel if rel > 0.01 else float('nan')
            cross = cross_sims[layer]
            ratio_str = f"{ratio:.2f}" if not np.isnan(ratio) else "N/A"
            print(f"{layer:<8} {sim:+.4f} {bar:<8} {rel:+.4f}       {ratio_str:<10} {cross:+.4f}")
        elif reliabilities:
            rel = reliabilities[layer]
            ratio = sim / rel if rel > 0.01 else float('nan')
            ratio_str = f"{ratio:.2f}" if not np.isnan(ratio) else "N/A"
            print(f"{layer:<8} {sim:+.4f} {bar:<8} {rel:+.4f}       {ratio_str:<10}")
        elif cross_sims:
            cross = cross_sims[layer]
            print(f"{layer:<8} {sim:+.4f} {bar:<8} {cross:+.4f}")
        else:
            print(f"{layer:<8} {sim:+.4f} {bar}")
    
    # Summary
    print("\n" + "-" * 70)
    avg_sim = np.mean([similarities[l] for l in key_layers])
    max_sim = max(similarities[l] for l in key_layers)
    max_layer = key_layers[np.argmax([similarities[l] for l in key_layers])]
    print(f"Average similarity: {avg_sim:+.4f}")
    print(f"Max similarity: {max_sim:+.4f} (layer {max_layer})")

if __name__ == "__main__":
    import fire
    fire.Fire(main)
