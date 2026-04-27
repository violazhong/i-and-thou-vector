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

**Pipeline: SUCCESS** — end-to-end generation + scoring completed.

### Score Distribution

| Condition | Samples | Blunt mean | Blunt std | Coherence mean | ≥50 | ≥40 | ≥30 |
|-----------|---------|------------|-----------|----------------|-----|-----|-----|
| Positive (blunt instructions) | 25 | 31.6 | 15.4 | 93.8 | 1 | 7 | 13 |
| Negative (diplomatic instructions) | 25 | 10.5 | 3.3 | 95.1 | 0 | 0 | 0 |

### Key Observations

1. **Contrast exists** — positive mean (31.6) vs negative mean (10.5). The instructions are creating a difference.
2. **Positive scores are too low** — only 1/25 passes the default threshold of 50. At threshold 30, 13/25 pass.
3. **Coherence is excellent** — both conditions score >93. The model produces well-formed responses.
4. **Highest scoring response** (blunt=81.0): "I'm afraid your new haircut doesn't quite work for you. It looks a bit messy and doesn't suit your face shape." — direct but still somewhat polite.
5. **The model resists being truly blunt** — even with explicit "be brutally honest" instructions, Qwen2.5-7B tends to soften its language. This is likely alignment training fighting the instruction.

### Implications for Thresholds

- At threshold 50: only ~4% of positive samples pass → too few for reliable vector extraction
- At threshold 30: ~52% pass → usable but noisier
- Recommendation: try threshold 30 for blunt trait, or run with 200 samples to get more data above 50

### Implication for Instructions

The Anthropic instructions may need strengthening for this model. Consider adding more forceful instructions or adjusting the system prompt approach.

## Next Steps

1. Proceed with extraction at lowered threshold (30) to test if the contrast is enough for steering
2. Alternatively, run full generation (200 samples) at threshold 50 — more samples means more pass
3. Compare with `evil` trait scores to understand baseline expectations
