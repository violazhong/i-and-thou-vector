# `scripts/run.sh` — Usage Guide

Non-interactive, CLI-driven experiment runner for the I-and-Thou Vector project. Designed for automation (Claude Code, CI/CD, unattended machines) while remaining human-friendly.

---

## Quick Start

```bash
# Minimal: run stages 1-3 for "evil" trait with 5 samples (fast test)
bash scripts/run.sh --model qwen --traits evil --stages 1,2,3 --samples 5

# Full production run
bash scripts/run.sh --model qwen --traits evil,kind,compliant --stages 1,2,3,4 --samples 200

# Resume a previous run, pick up at stage 2
bash scripts/run.sh --resume --stages 2,3
```

---

## All Options

| Flag | Values | Default | Description |
|---|---|---|---|
| `--model` | `qwen`, `llama`, or full HF path | `qwen` | Model to use |
| `--traits` | Comma-separated | `evil` | Traits to process (from `configs/traits/`) |
| `--stages` | Comma-separated `1,2,3,4` | `1,2,3` | Which stages to run |
| `--samples` | Integer | `200` | Samples per instruction for Stage 1 |
| `--layer` | Integer | Model-specific (20) | Steering layer for Stage 3 |
| `--coefficients` | Comma-separated floats | `-2,-1,0,1,2` | Steering coefficients for Stage 3 |
| `--prompts` | Pipe-separated strings | 5 built-in prompts | Custom prompts for Stage 3 |
| `--vectors` | Comma-separated paths | Auto-discover | Override vector paths for Stage 3 |
| `--resume` | Flag | `false` | Resume the latest run for this model |
| `-h`, `--help` | Flag | — | Show help text |

---

## Stages

### Stage 1: Generate Responses
Calls `generate_combined_responses.py`. Generates model persona and user persona responses using vLLM, then scores them with GPT-4o.

**Requires:** GPU (for vLLM), `OPENAI_API_KEY` (for scoring)

**Output:** `data/responses/{MODEL_SHORT}_{trait}_{positive,negative}.csv`

### Stage 2: Extract Vectors
Calls `extract_vectors.py`. Filters responses by trait/coherence score, loads the full model (not vLLM), extracts hidden-state activations, computes model persona, user persona, and I-Thou vectors.

**Requires:** GPU (for model inference), Stage 1 output

**Output:** `data/vectors/{MODEL_SHORT}/{trait}_{model_persona,user_persona,i_thou}_{position}.pt`

### Stage 3: Steering Experiments
Calls `run_steering.py` for every (vector, prompt) combination. Batch mode — no interaction.

**Requires:** GPU, Stage 2 output (or `--vectors`)

**Output:** Logged to `logs/{run}/stage3_steering.log`

### Stage 4: Vector Analysis
Calls `analyze_vectors.py`. Compares model_persona vs user_persona vectors at each extraction position.

**Requires:** Stage 2 output

**Output:** Logged to `logs/{run}/stage4_analysis.log`

---

## Model Shortcuts

| Shorthand | Full HF Path | Layers | Default Layer |
|---|---|---|---|
| `qwen`, `qwen2.5`, `Qwen` | `Qwen/Qwen2.5-7B-Instruct` | 28 | 20 |
| `llama`, `llama3`, `Llama` | `meta-llama/Meta-Llama-3.1-8B-Instruct` | 32 | 20 |
| Any other string | Used as-is (`MODEL_SHORT` = last path component) | 28 | 20 |

---

## Available Traits

Defined in `configs/traits/*.yaml`:

- `evil` — Malicious intent / cruelty vs. ethical behavior
- `kind` — Warmth and compassion vs. cold detachment
- `compliant` — Unconditional deference vs. paranoid safety enforcement

---

## Resume Mechanism

The script creates a timestamped directory under `logs/` for each run:

```
logs/Qwen2.5-7B-Instruct_20260208_153000/
  state.env     # key=value checkpoint file
  run.log       # combined run log
  stage1_evil_generate.log
  stage2_evil_extract.log
  stage3_steering.log
  stage4_analysis.log
```

`state.env` tracks completion per trait per stage:

```bash
SELECTED_TRAITS=evil,kind
STAGE1_evil_DONE=1
STAGE1_evil_POS_CSV=/path/to/positive.csv
STAGE1_evil_NEG_CSV=/path/to/negative.csv
STAGE2_evil_DONE=1
STAGE2_evil_VECTORS_DIR=/path/to/vectors/
```

Using `--resume` finds the latest `state.env` for the current model and skips already-completed steps.

---

## Output Structure

```
data/
  responses/
    Qwen2.5-7B-Instruct_evil_positive.csv
    Qwen2.5-7B-Instruct_evil_negative.csv
  vectors/
    Qwen2.5-7B-Instruct/
      evil_model_persona_prompt_end.pt
      evil_model_persona_response_start.pt
      evil_model_persona_response_avg.pt
      evil_user_persona_prompt_end.pt
      evil_user_persona_response_start.pt
      evil_user_persona_response_avg.pt
      evil_i_thou_prompt_end.pt
      evil_i_thou_response_start.pt
      evil_i_thou_response_avg.pt
```

---

## Examples

### Run a single trait end-to-end with minimal samples (testing)

```bash
bash scripts/run.sh --model qwen --traits evil --stages 1,2,3 --samples 2
```

### Run multiple traits, only Stage 1

```bash
bash scripts/run.sh --model qwen --traits evil,kind,compliant --stages 1 --samples 200
```

### Run Stage 3 with custom prompts and specific vectors

```bash
bash scripts/run.sh --model qwen --traits evil --stages 3 \
    --prompts 'How should I handle conflict?|What makes a good leader?' \
    --vectors 'data/vectors/Qwen2.5-7B-Instruct/evil_i_thou_response_avg.pt'
```

### Run Stage 3 with different coefficients and layer

```bash
bash scripts/run.sh --model qwen --traits evil --stages 3 \
    --layer 15 --coefficients "-3,-2,-1,0,1,2,3"
```

### Resume after a crash

```bash
bash scripts/run.sh --resume --stages 1,2,3
```

This finds the latest run, checks `state.env`, and skips any already-completed trait/stage combinations.
