"""
Response generation using vLLM or transformers.
"""
import os
# Fix for vLLM CUDA multiprocessing error: Set spawn method before any imports
# vLLM uses multiprocessing and defaults to 'fork', which cannot re-initialize CUDA
# Setting this environment variable forces vLLM to use 'spawn' instead
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'


import torch
import pandas as pd
from pathlib import Path
from typing import Union
from tqdm import tqdm

from ithou.utils import logger, ensure_dir, TraitConfig


def generate_with_transformers(
    model_name: str,
    prompts: list[str],
    max_new_tokens: int = 1000,
    temperature: float = 0.7,
    top_p: float = 0.95,
    batch_size: int = 4,
) -> list[str]:
    """Generate responses using transformers (slower but more flexible)."""
    from ithou.models import ModelWrapper
    
    logger.info(f"Generating {len(prompts)} responses with transformers")
    
    model = ModelWrapper(model_name)
    all_responses = []
    
    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating"):
        batch = prompts[i:i + batch_size]
        responses = model.generate(
            batch,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=temperature > 0,
        )
        all_responses.extend(responses)
    
    del model
    torch.cuda.empty_cache()
    
    return all_responses


def generate_trait_responses(
    model_name: str,
    trait_config: Union[dict, TraitConfig],
    output_dir: str,
    num_samples_per_instruction: int = 200,
    max_new_tokens: int = 1000,
    temperature: float = 0.7,
    batch_size: int = 32,
    use_vllm: bool = True,
) -> tuple[str, str]:
    """Generate positive and negative trait responses, save to CSVs, return paths."""
    
    if isinstance(trait_config, dict):
        trait_config = TraitConfig.from_dict(trait_config)
    
    trait_name = trait_config.name
    output_dir = ensure_dir(output_dir)
    pos_path = output_dir / f"{model_name.split('/')[-1]}_{trait_name}_positive.csv"
    neg_path = output_dir / f"{model_name.split('/')[-1]}_{trait_name}_negative.csv"

    # avoid to re-generate when file already exists
    if os.path.exists(pos_path) and os.path.exists(neg_path):
        return str(pos_path), str(neg_path)
    
    logger.info(f"Generating responses for trait: {trait_name}")
    
    # Collect ALL prompts first
    all_instructions = trait_config.positive_instructions + trait_config.negative_instructions
    all_data = []  # [(instruction, question, prompt_text, is_positive), ...]
    
    for idx, instruction in enumerate(all_instructions):
        is_positive = idx < len(trait_config.positive_instructions)
        questions = trait_config.questions
        n_repeats = num_samples_per_instruction // len(questions) + 1
        questions_expanded = (questions * n_repeats)[:num_samples_per_instruction]
        
        for q in questions_expanded:
            # Store raw data, format later
            all_data.append((instruction, q, is_positive))
    
    if use_vllm:
        # Import vLLM and load model FIRST, before any other CUDA operations
        from vllm import LLM, SamplingParams
        
        logger.info(f"Loading vLLM model: {model_name}")
        llm = LLM(model=model_name, dtype="float16", trust_remote_code=True)
        
        # Now format prompts using vLLM's tokenizer
        tokenizer = llm.get_tokenizer()

        
        all_prompts = [_format_prompt(instr, q) for instr, q, _ in all_data]
        
        logger.info(f"Generating {len(all_prompts)} responses with vLLM")
        sampling_params = SamplingParams(max_tokens=max_new_tokens, temperature=temperature, top_p=0.95)
        outputs = llm.generate(all_prompts, sampling_params)
        all_responses = [output.outputs[0].text for output in outputs]
        
        del llm
        torch.cuda.empty_cache()
    else:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        all_prompts = [_format_prompt(instr, q) for instr, q, _ in all_data]
        
        all_responses = generate_with_transformers(
            model_name, all_prompts, max_new_tokens, temperature, batch_size=batch_size
        )
    
    # Split back into positive and negative
    pos_data = []
    neg_data = []
    
    for (instruction, question, is_positive), prompt, response in zip(all_data, all_prompts, all_responses):
        row = {"question": question, "instruction": instruction, "prompt": prompt, "response": response}
        if is_positive:
            pos_data.append(row)
        else:
            neg_data.append(row)
    
    # Save
    pos_df = pd.DataFrame(pos_data)
    pos_df.to_csv(pos_path, index=False)
    logger.info(f"Saved {len(pos_df)} positive responses to {pos_path}")
    
    neg_df = pd.DataFrame(neg_data)
    neg_df.to_csv(neg_path, index=False)
    logger.info(f"Saved {len(neg_df)} negative responses to {neg_path}")
    
    return str(pos_path), str(neg_path)

def _format_prompt(system: str, user: str) -> str:
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)