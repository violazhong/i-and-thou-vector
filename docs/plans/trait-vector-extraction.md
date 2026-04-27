# Plan: Trait Vector Extraction for New Traits

**Status:** In progress
**Created:** 2026-04-27
**Traits:** pessimistic, blunt, rebellious

## Objective

Extract persona steering vectors for 3 new traits and verify they produce measurable steering effects. The YAML config quality is the primary variable — this plan is an iterative refinement loop.

## Background

- 3 validated traits exist: `compliant`, `kind`, `evil`
- 240 traits defined in `assistant-axis/data/traits/trait_list.json`
- Anthropic's `assistant-axis` repo provides research-tested paired instructions for all 240 traits

## YAML Source

Trait configs were converted from Anthropic's JSON format (`assistant-axis/data/traits/instructions/{trait}.json`) into our YAML format. Key improvements over AI-generated first drafts:

| Feature | First draft | Anthropic-sourced |
|---------|-------------|-------------------|
| Instructions | Unpaired, 8 per side | **Paired** pos[i]↔neg[i], 5 pairs |
| Questions | 15 generic | **40** targeted, emotionally probing |
| Eval prompt | None | Explicit per-trait GPT-4o scoring template |

## Pipeline

```
Step 1: generate_combined_responses.py  (vLLM on GPU)
Step 2: GPT-4o scoring                  (built into step 1)
Step 3: extract_vectors.py              (transformers forward pass)
Step 4: run_steering.py                 (verify steering effect)
```

## Simplification

One vector per iteration to keep the feedback loop fast:
- **Persona:** `i_thou` (model_persona − user_persona)
- **Token position:** `response_start`
- **Layer:** 20 (Qwen2.5-7B default)

## Threshold Tuning

`trait_threshold` and `coherence_threshold` in YAML (currently both 50):
- Higher → fewer but cleaner contrastive pairs → potentially better vectors but less data
- Lower → more data passes but noisier → potentially weaker vectors
- Trait-dependent — to be tuned per trait during the experiment loop

## Iteration Loop

```
YAML → Generate → Score → Extract → Steer → Evaluate
  ↑                                            |
  └──── Revise if steering is weak ────────────┘
```

## Success Criteria

1. Score filtering retains ≥50% of generated samples
2. Steering at coefficient +2 produces visibly trait-aligned responses
3. Steering at coefficient −2 produces visibly opposite responses
4. Clear monotonic gradient across coefficient range

## Remote Server

- GPU: NVIDIA RTX A6000, 51 GB VRAM
- Model: Qwen/Qwen2.5-7B-Instruct (14.2 GB on GPU)
- Access: LLMOS via ngrok tunnel
