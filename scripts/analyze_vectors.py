#!/usr/bin/env python3
"""
Compare and diagnose persona vectors.

Usage:
    # Compare two vectors (full diagnostics if activations available)
    python scripts/compare_vectors.py \
        --vector1 vectors/evil_model_persona.pt \
        --vector2 vectors/evil_user_persona.pt \
        --activations vectors/evil_activations_raw.pt \
        --cross_vector vectors/sycophancy_model_persona.pt \
        --layers "15,16,17,18,19,20"

    # Simple comparison (no activations)
    python scripts/compare_vectors.py \
        --vector1 vectors/evil_model_persona.pt \
        --vector2 vectors/evil_user_persona.pt
"""

import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np

from ithou.utils import setup_logging, logger


def load_vector(path: str) -> torch.Tensor:
    """Load a vector from file."""
    if not Path(path).exists():
        raise FileNotFoundError(f"Vector file not found: {path}")
    vec = torch.load(path, map_location="cpu")
    return vec


def compute_layerwise_similarity(vec1: torch.Tensor, vec2: torch.Tensor) -> list[float]:
    """Compute cosine similarity at each layer."""
    similarities = []
    for layer in range(vec1.shape[0]):
        v1 = vec1[layer].float()
        v2 = vec2[layer].float()
        sim = torch.nn.functional.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()
        similarities.append(sim)
    return similarities


def compute_split_half_reliability(pos_act: list, neg_act: list, seed: int = 42) -> list[float]:
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


def main(
    vector1: str,
    vector2: str,
    activations: Optional[str] = None,
    cross_vector: Optional[str] = None,
    layers: Optional[str] = None,
    log_level: str = "INFO",
):
    """
    Compare two vectors with optional full diagnostics.
    
    Args:
        vector1: Path to first vector (e.g., model persona)
        vector2: Path to second vector (e.g., user persona)
        activations: Path to raw activations for split-half reliability
        cross_vector: Path to cross-trait vector for specificity check
        layers: Comma-separated layers to show (default: all)
        log_level: Logging level
    """
    setup_logging(log_level)
    
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
    
    if reliabilities:
        avg_rel = np.mean([reliabilities[l] for l in key_layers])
        avg_ratio = avg_sim / avg_rel if avg_rel > 0.01 else float('nan')
        print(f"Average reliability: {avg_rel:+.4f}")
        print(f"Average ratio: {avg_ratio:.2f}" if not np.isnan(avg_ratio) else "Average ratio: N/A")
    
    # Interpretation
    if reliabilities:
        print("\n" + "=" * 70)
        print("INTERPRETATION:")
        print("  Similarity:  Cosine similarity between the two vectors")
        print("  Split-Half:  Reliability ceiling (internal consistency)")
        print("  Ratio:       Similarity / Split-Half (signal captured)")
        print("               → Ratio > 0.5: substantial shared structure")
        print("               → Ratio < 0.2: mostly distinct representations")
        if cross_sims:
            print("  Cross:       Similarity to different trait (should be ~0)")
    
    print("=" * 70 + "\n")


if __name__ == "__main__":
    import fire
    fire.Fire(main)