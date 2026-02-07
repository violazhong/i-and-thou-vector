#!/usr/bin/env python3
"""
Generate trait-exhibiting responses for vector extraction.

This script generates responses under positive and negative trait instructions
using vLLM for fast batched generation.

Usage:
    python scripts/generate_responses.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --trait evil \
        --output_dir data/responses/Qwn.Qwen2.5-7B-Instruct \
        --samples_per_instruction 200

Output:
    Creates two CSV files in output_dir:
    - {trait}_positive.csv: Responses under positive trait instructions
    - {trait}_negative.csv: Responses under negative trait instructions
"""

import sys
from pathlib import Path
import os

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ithou.generation import generate_trait_responses
from ithou.utils import load_trait_config, setup_logging, set_seed
from ithou.scoring import score_responses

def main(
    model: str,
    trait: str,
    output_dir: str = "data/responses",
    samples_per_instruction: int = 200,
    max_new_tokens: int = 1000,
    temperature: float = 1.0, # for more creative, verbose respone
    batch_size: int = 32,
    use_vllm: bool = True,
    score: bool = True,
    score_model: str = "gpt-4o",
    seed: int = 42,
    log_level: str = "ERROR",
):
    setup_logging(log_level)
    set_seed(seed)

    trait_config = load_trait_config(trait)
    
    # Step 1: Generate responses
    print(f"\n{'='*60}")
    print(f"Step 1: Generating responses for trait '{trait}'")
    print(f"{'='*60}")
    
    pos_path, neg_path = generate_trait_responses(
        model_name=model,
        trait_config=trait_config,
        output_dir=output_dir,
        num_samples_per_instruction=samples_per_instruction,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        batch_size=batch_size,
        use_vllm=use_vllm,
    )
    
    print(f"\nGeneration complete!")
    print(f"  Positive: {pos_path}")
    print(f"  Negative: {neg_path}")
    
    # Step 2: Score responses
    if score:
        print(f"\n{'='*60}")
        print(f"Step 2: Scoring responses with {score_model} (logprobs)")
        print(f"{'='*60}")
        
        trait_description = trait_config.get("description", f"The {trait} trait")
        
        score_responses(
            csv_path=pos_path,
            trait_name=trait,
            trait_description=trait_description,
            model=score_model,
        )
        
        score_responses(
            csv_path=neg_path,
            trait_name=trait,
            trait_description=trait_description,
            model=score_model,
        )
        
        print(f"\nScoring complete! CSVs updated with trait and coherence scores.")
    
    print(f"\n{'='*60}")
    print("All done!")
    print(f"  {pos_path}")
    print(f"  {neg_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    import fire
    fire.Fire(main)