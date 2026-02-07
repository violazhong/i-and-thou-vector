#!/usr/bin/env python3
"""
Run steering experiments with I-Thou vectors.

This script applies I-Thou vectors during generation to observe
how steering affects model responses.

Usage:
    python scripts/run_steering.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --vector data/vectors/Qwen2.5-7B-Instruct/evil_i_thou_response_avg.pt \
        --prompt="I feel really hurt" \
        --layers="20" \
        --coefficients="-2,-1.5,-1,0,1,1.5,2"

The coefficient controls the steering direction:
    positive coefficient: "I am X toward you" (model has trait)
    negative coefficient: "You are X toward me" (user has trait)
"""

import sys
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from ithou.models import ModelWrapper
from ithou.steering import run_steering_experiment, SteeringHook
from ithou.utils import setup_logging, set_seed, ensure_dir, logger


def main(
    model: str,
    vector: str,
    prompt: str,
    layers: list[int] = [20],
    coefficients: list[float] = [-2, -1, 0, 1, 2],
    system_prompt: str = "You are a helpful assistant.",
    max_new_tokens: int = 800,
    seed: int = 42,
    log_level: str = "INFO",
):
    # Setup
    setup_logging(log_level)
    set_seed(seed)

    # Parse layers
    if isinstance(layers, str):
        layer_list = [int(x.strip()) for x in layers.split(",")]
    elif isinstance(layers, int):
        layer_list = [layers]
    else:
        layer_list = [20]
        
    # Load vector
    logger.info(f"Loading vector from {vector}")
    vec = torch.load(vector, map_location="cpu")
    logger.info(f"Vector shape: {vec.shape}")
    
    # Load model
    model_wrapper = ModelWrapper(model)
    
    # Run steering experiments
    logger.info(f"\nRunning steering experiments:")
    logger.info(f"  Layers: {layers}")
    logger.info(f"  Coefficients: {coefficients}")
    logger.info(f"  Prompt: {prompt}")
    
    results_df = run_steering_experiment(
        model=model_wrapper,
        vector=vec,
        prompt=prompt,
        layers=layer_list,
        coefficients=coefficients,
        max_new_tokens=max_new_tokens,
        system_prompt=system_prompt,
    )
    
    # Print results
    print("\n" + "="*80)
    print("STEERING RESULTS")
    print("="*80)
    
    print(f"\nPrompt: {prompt}")
    print("-" * 60)
    
    prompt_results = results_df[results_df["prompt"] == prompt]
    
    for _, row in prompt_results.iterrows():
        coef = row["coefficient"]
        direction = "baseline" if coef == 0 else ("I am X" if coef > 0 else "You are X")
        print(f"\nCoef {coef:+.1f} ({direction}):")
        
        response = row["response"]
        print(response)
        print("-" * 60)
    
    print(f"\nSteering complete!")


if __name__ == "__main__":
    import fire
    fire.Fire(main)
