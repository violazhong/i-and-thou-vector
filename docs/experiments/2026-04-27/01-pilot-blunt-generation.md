# Experiment: Pilot Run — Blunt Trait Generation

**Date:** 2026-04-27 20:57 UTC
**Trait:** blunt
**Model:** Qwen/Qwen2.5-7B-Instruct
**Server:** NVIDIA RTX A6000 (51 GB)

## Purpose

End-to-end pipeline test with small sample size before committing to full generation run. Validates that:
- Pipeline reads the new Anthropic-sourced YAML format correctly
- vLLM generates responses under paired instructions
- GPT-4o scoring works
- CSV outputs are written correctly

## Parameters

| Parameter | Value |
|-----------|-------|
| samples_per_instruction | 5 (test only — full run uses 200) |
| max_new_tokens | 300 |
| temperature | 1.0 |
| trait_threshold | 50 |
| coherence_threshold | 50 |
| tmux session | `gen-blunt-test` |

## YAML Config

Source: `assistant-axis/data/traits/instructions/blunt.json` converted to YAML.
- 5 paired instructions (pos[i] ↔ neg[i])
- 40 questions (emotionally probing scenarios)
- Eval prompt included (not yet wired into scoring code)

## Timeline

| Time (UTC) | Event |
|------------|-------|
| 20:57:44 | Script started, vLLM model loading initiated |
| 20:59:05 | Model weights downloaded (65s, 4 shards ~15 GB) |
| 20:59:14 | Weights loaded into GPU (14.2 GB, 75s total) |
| 20:59:22 | torch.compile cache warmup started |
| 20:59:49 | Graph compilation completed (27s) |
| -- | CUDA graph capture in progress (first-time warmup) |

## Environment Check (pre-run)

```
Python: 3.10.12
PyTorch: 2.6.0+cu124
CUDA: available
GPU: NVIDIA RTX A6000 (51.0 GB)
vLLM: 0.8.5.post1
transformers: 4.52.3
pandas: 2.3.1
openai: 1.70.0
OPENAI_API_KEY: set
```

## Observations

- First run downloads model weights from HuggingFace (~65s). Subsequent runs will use cache.
- vLLM v1 engine with torch.compile does a slow first-time CUDA graph compilation. This is cached for future runs.
- A6000 has plenty of headroom for 7B model (14.2 GB / 51 GB = 28% VRAM usage).

## Results

_Pending — experiment still running._

## Next Steps

If generation succeeds:
1. Check CSV output format and sample quality
2. Run `extract_vectors.py` for blunt trait
3. Run `run_steering.py` to verify steering effect
4. If validated, run full generation (200 samples) for all 3 traits
