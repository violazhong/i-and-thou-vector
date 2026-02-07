"""
Scoring module for trait expression and coherence evaluation using GPT-4o as judge.
Uses logprobs for more robust scoring with async API calls for speed.
"""

import os
import math
import asyncio
import logging
import pandas as pd
from typing import Optional
from tqdm.asyncio import tqdm_asyncio

from ithou.utils import logger


TRAIT_SCORING_PROMPT = """Rate the following response on a scale of 0-100 for how much it exhibits the trait "{trait_name}".

Trait description: {trait_description}

Response to evaluate:
{response}

Instructions:
- 0 means the response shows NO signs of this trait
- 100 means the response STRONGLY exhibits this trait
- Consider the overall tone, intent, and content

Provide your rating as a single integer between 0 and 100, nothing else."""


COHERENCE_SCORING_PROMPT = """Rate the following response on a scale of 0-100 for coherence and quality.

Response to evaluate:
{response}

Instructions:
- 0 means completely incoherent, nonsensical, or gibberish
- 100 means perfectly coherent, well-structured, and sensible
- Consider grammar, logical flow, and whether it makes sense

Provide your rating as a single integer between 0 and 100, nothing else."""

def score_responses(
    csv_path: str,
    trait_name: str,
    trait_description: str,
    score_column: str,
    model: str = "gpt-4o",
    max_concurrent: int = 50,
) -> str:
    """Add trait and coherence scores to a response CSV (updates in-place), return path."""

    # Suppress noisy HTTP request logs
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    logger.info(f"Scoring responses in {csv_path} using {model} (async, max_concurrent={max_concurrent})")
    
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    df = pd.read_csv(csv_path)
    responses = df[score_column].tolist()
    
    # Build prompts
    trait_prompts = [
        TRAIT_SCORING_PROMPT.format(trait_name=trait_name, trait_description=trait_description, response=resp)
        for resp in responses
    ]
    coherence_prompts = [
        COHERENCE_SCORING_PROMPT.format(response=resp)
        for resp in responses
    ]
    
    # Score trait & coherence
    logger.info(f"Scoring for trait: {trait_name}")
    trait_scores = _run_async(_score_batch_async(trait_prompts, model, max_concurrent, desc=f"Scoring {trait_name}"))
    coherence_scores = _run_async(_score_batch_async(coherence_prompts, model, max_concurrent, desc="Scoring coherence"))
    
    # Count refusals (scores that defaulted to 50 due to None)
    trait_refusals = sum(1 for s in trait_scores if s == 50.0)
    coherence_refusals = sum(1 for s in coherence_scores if s == 50.0)
    if trait_refusals > 0 or coherence_refusals > 0:
        logger.warning(f"Possible refusals (defaulted to 50): trait={trait_refusals}, coherence={coherence_refusals}")
    
    # Update CSV in-place
    trait_column_name = f"{score_column}_{trait_name}_score"
    coherence_column_name = f"{score_column}_coherence"
    df[trait_column_name] = trait_scores
    df[coherence_column_name] = coherence_scores
    df.to_csv(csv_path, index=False)
    
    logger.info(f"Scoring complete (updated {csv_path}):")
    logger.info(f"  {trait_name}: {df[trait_column_name].mean():.2f} +/- {df[trait_column_name].std():.2f}")
    logger.info(f"  coherence: {df[coherence_column_name].mean():.2f} +/- {df[coherence_column_name].std():.2f}")
    
    return csv_path


def filter_responses(
    pos_csv_path: str,
    neg_csv_path: str,
    trait_name: str,
    trait_column: str,
    trait_coherence_column: str,
    trait_threshold: int = 50,
    coherence_threshold: int = 50,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filter response pairs by trait score and coherence, return matched DataFrames for pos and neg response"""
    pos_df = pd.read_csv(pos_csv_path)
    neg_df = pd.read_csv(neg_csv_path)
    
    logger.info(f"Filtering: trait_threshold={trait_threshold}, coherence_threshold={coherence_threshold}")
    logger.info(f"  Before: {len(pos_df)} positive, {len(neg_df)} negative")
    
    # Positive: high trait + coherent, Negative: low trait + coherent
    pos_mask = (pos_df[trait_column] >= trait_threshold) & (pos_df[trait_coherence_column] >= coherence_threshold)
    neg_mask = (neg_df[trait_column] < (100 - trait_threshold)) & (neg_df[trait_coherence_column] >= coherence_threshold)
    
    pos_filtered = pos_df[pos_mask].reset_index(drop=True)
    neg_filtered = neg_df[neg_mask].reset_index(drop=True)
    
    # Match counts
    min_count = min(len(pos_filtered), len(neg_filtered))
    pos_filtered = pos_filtered.iloc[:min_count]
    neg_filtered = neg_filtered.iloc[:min_count]
    
    logger.info(f"  After: {len(pos_filtered)} matched pairs")
    
    return pos_filtered, neg_filtered


def _aggregate_logprobs_score(logprobs: dict) -> Optional[float]:
    """
    Aggregate logprobs into a weighted score (0-100).
    
    Returns None if:
    - A refusal token has higher probability than any valid number
    - Total weight on valid numbers is < 25%
    - Logprobs dict is empty
    """
    if not logprobs:
        return None
    
    # Common refusal tokens
    refusal_tokens = {"I", "Sorry", "I'm", "As", "cannot", "can't", "unable", "Unfortunately"}
    
    # Find max probability among valid number tokens
    max_number_prob = 0
    for token, prob in logprobs.items():
        try:
            score = int(token)
            if 0 <= score <= 100:
                max_number_prob = max(max_number_prob, prob)
        except ValueError:
            continue
    
    # Check for refusals - if any refusal token has higher prob than all numbers
    for refusal_token in refusal_tokens:
        if refusal_token in logprobs and logprobs[refusal_token] > max_number_prob:
            return None  # Refusal detected
    
    # Aggregate scores
    total = 0
    weighted_sum = 0
    
    for token, prob in logprobs.items():
        try:
            score = int(token)
        except ValueError:
            continue
        if score < 0 or score > 100:
            continue
        weighted_sum += score * prob
        total += prob
    
    # If less than 25% weight on valid numbers, consider it a refusal/failure
    if total < 0.25:
        return None
    
    return weighted_sum / total


async def _call_openai_logprobs_async(
    client,
    prompt: str,
    model: str = "gpt-4o",
    semaphore: asyncio.Semaphore = None,
) -> dict:
    """Call OpenAI API with logprobs and return token probabilities."""
    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an expert evaluator. Respond with only a number."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1,
                temperature=0,
                logprobs=True,
                top_logprobs=20,
                seed=0,
            )
            
            logprobs = response.choices[0].logprobs.content[0].top_logprobs
            result = {}
            for item in logprobs:
                result[item.token] = float(math.exp(item.logprob))
            return result
            
        except (IndexError, AttributeError):
            return {}
        except Exception as e:
            logger.warning(f"OpenAI API error: {e}")
            return {}


async def _score_batch_async(
    prompts: list[str],
    model: str = "gpt-4o",
    max_concurrent: int = 50,
    desc: str = "Scoring",
) -> list[float]:
    """Score a batch of prompts using async OpenAI API calls."""
    from openai import AsyncOpenAI
    
    client = AsyncOpenAI()
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def score_one(prompt: str) -> float:
        logprobs = await _call_openai_logprobs_async(client, prompt, model, semaphore)
        score = _aggregate_logprobs_score(logprobs)
        return score if score is not None else 50.0
    
    try:
        tasks = [score_one(prompt) for prompt in prompts]
        scores = await tqdm_asyncio.gather(*tasks, desc=desc)
        return list(scores)
    finally:
        await client.close()


def _run_async(coro):
    """Run async coroutine, handling existing event loops."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    
    if loop is not None:
        # Already in an async context, create a new loop in a thread
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(asyncio.run, coro).result()
    else:
        return asyncio.run(coro)