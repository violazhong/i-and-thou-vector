# docs/experiments/2026-04-27/

Experiments run on 2026-04-27. First day of trait vector extraction for new traits using Anthropic-sourced YAML configs.

## Experiments

- `01-pilot-blunt-generation.md` — Pilot run: small-sample generation for blunt trait on remote A6000 server. Tests end-to-end pipeline (vLLM generation → GPT-4o scoring → CSV output).

## Context

- Remote server: NVIDIA RTX A6000, 51 GB VRAM
- Model: Qwen/Qwen2.5-7B-Instruct
- Traits being tested: blunt (pilot), then pessimistic and rebellious
- YAML configs sourced from `assistant-axis/data/traits/instructions/`
