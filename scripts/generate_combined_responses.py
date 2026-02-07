#!/usr/bin/env python3
"""
Generate trait responses and user persona responses.

This script:
1. Generates responses under positive and negative trait instructions (model persona)
2. Scores responses using GPT-4o for trait expression and coherence
3. Generates model responses TO user personas (for user persona extraction)

Usage:
    python scripts/generate_combined_responses.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --trait evil \
        --output_dir data/responses \
        --samples_per_instruction 200

Output:
    Creates two CSV files in output_dir with columns:
    - question, instruction, prompt, response (model persona)
    - {trait}, coherence (scores)
    - prompt_to_user, response_to_user (user persona)
"""

# # Must be at the very top, before any other imports that might touch CUDA
import multiprocessing
if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)

# Suppress noisy HTTP logs from OpenAI client
import logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import pandas as pd
from vllm import LLM, SamplingParams

from ithou.scoring import score_responses
from ithou.utils import load_trait_config, setup_logging, set_seed, ensure_dir, TraitConfig, logger

def generate_all_responses(
    model_name: str,
    user_persona_system_prompt: str,
    trait_config: TraitConfig,
    output_dir: Path,
    num_samples_per_instruction: int = 200,
    max_new_tokens: int = 1000,
    temperature: float = 1.0,
) -> tuple[str, str]:
    """
    Fast model generation using vLLM.
    
    Meant to be stand-alone, a centralized method to manage the whole model cycle (initiate, generate, delete)
    """
    # =========================================================================
    # Step 0: Skip regeneration if the file already exists
    # =========================================================================
    model_short_name = model_name.split("/")[-1]
    trait_name = trait_config.name
    pos_path = output_dir / f"{model_short_name}_{trait_name}_positive.csv"
    neg_path = output_dir / f"{model_short_name}_{trait_name}_negative.csv"

    # avoid to re-generate when file already exists
    if os.path.exists(pos_path) and os.path.exists(neg_path):
        return str(pos_path), str(neg_path)
    
    logger.info(f"Loading vLLM model: {model_name}")
    llm = LLM(model=model_name, dtype="float16", trust_remote_code=True, max_model_len=8192)
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(max_tokens=max_new_tokens, temperature=temperature, top_p=0.95)
    
    def format_prompt(system: str, user: str) -> str:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # =========================================================================
    # Step 1: Generate model persona responses
    # =========================================================================
    logger.info(f"\n{'='*60}")
    logger.info("STEP 1: Generating model persona responses")
    logger.info(f"{'='*60}")
    
    all_instructions = trait_config.positive_instructions + trait_config.negative_instructions
    all_data = []
    
    for idx, instruction in enumerate(all_instructions):
        is_positive = idx < len(trait_config.positive_instructions)
        questions = trait_config.questions
        n_repeats = num_samples_per_instruction // len(questions) + 1
        questions_expanded = (questions * n_repeats)[:num_samples_per_instruction]
        
        for q in questions_expanded:
            all_data.append({
                "instruction": instruction,
                "question": q,
                "is_positive": is_positive,
            })
    
    # Format prompts and generate
    all_prompts = [format_prompt(d["instruction"], d["question"]) for d in all_data]
    
    logger.info(f"Generating {len(all_prompts)} model persona responses")
    outputs = llm.generate(all_prompts, sampling_params)
    model_responses = [output.outputs[0].text for output in outputs]
    
    # Add prompts and responses to data
    for i, d in enumerate(all_data):
        d["model_persona_prompt"] = all_prompts[i]
        d["model_persona_response"] = model_responses[i]
    
    # =========================================================================
    # Step 2: Generate user persona responses (model responding TO users)
    # =========================================================================
    logger.info(f"\n{'='*60}")
    logger.info("STEP 2: Generating user persona responses")
    logger.info(f"{'='*60}")
    
    # Use model persona responses as user messages
    user_prompts = [format_prompt(user_persona_system_prompt, d["model_persona_response"]) for d in all_data]
    
    logger.info(f"Generating {len(user_prompts)} user persona responses")
    outputs = llm.generate(user_prompts, sampling_params)
    user_responses = [output.outputs[0].text for output in outputs]
    
    # Add user persona data
    for i, d in enumerate(all_data):
        d["user_persona_prompt"] = user_prompts[i]
        d["user_persona_response"] = user_responses[i]
    
    # Cleanup vLLM
    del llm
    torch.cuda.empty_cache()
    
    # =========================================================================
    # Step 3: Split and save to CSVs
    # =========================================================================
    logger.info(f"\n{'='*60}")
    logger.info("STEP 3: Saving to CSVs")
    logger.info(f"{'='*60}")
    
    pos_data = [d for d in all_data if d["is_positive"]]
    neg_data = [d for d in all_data if not d["is_positive"]]
    
    # Remove is_positive flag before saving
    for d in pos_data + neg_data:
        del d["is_positive"]
    
    pos_df = pd.DataFrame(pos_data)
    neg_df = pd.DataFrame(neg_data)
    
    pos_df.to_csv(pos_path, index=False)
    neg_df.to_csv(neg_path, index=False)
    
    logger.info(f"Saved {len(pos_df)} positive samples to {pos_path}")
    logger.info(f"Saved {len(neg_df)} negative samples to {neg_path}")
    
    return str(pos_path), str(neg_path)


def main(
    model: str,
    trait: str,
    output_dir: str = "data/responses",
    user_persona_system_prompt: str = "You are a helpful assistant.",
    samples_per_instruction: int = 200,
    max_new_tokens: int = 1000,
    temperature: float = 1.0,
    score_model: str = "gpt-4o",
    seed: int = 42,
    log_level: str = "INFO",
):
    setup_logging(log_level)
    set_seed(seed)
    
    trait_config = TraitConfig.load(trait)
    output_path = ensure_dir(output_dir)
    
    print(f"\n{'='*60}")
    print(f"Generating responses for trait: {trait}")
    print(f"Model: {model}")
    print(f"Samples per instruction: {samples_per_instruction}")
    print(f"Max new tokens: {max_new_tokens}")
    print(f"{'='*60}")
    
    # Step 1. Generate all responses (model persona + user persona)
    pos_path, neg_path = generate_all_responses(
        model_name=model,
        user_persona_system_prompt=user_persona_system_prompt,
        trait_config=trait_config,
        output_dir=output_path,
        num_samples_per_instruction=samples_per_instruction,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    
    # Step 2. Score model persona responses
    print(f"\n{'='*60}")
    print(f"Scoring responses with {score_model} (logprobs)")
    print(f"{'='*60}")
    
    score_responses(
        csv_path=pos_path,
        trait_name=trait,
        trait_description=trait_config.description,
        score_column="model_persona_response",
        model=score_model,
    )
    
    score_responses(
        csv_path=neg_path,
        trait_name=trait,
        trait_description=trait_config.description,
        score_column="model_persona_response",
        model=score_model,
    )
    
    print(f"\nScoring complete!")
    
    print(f"\n{'='*60}")
    print("All done!")
    print(f"  {pos_path}")
    print(f"  {neg_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    import fire
    fire.Fire(main)