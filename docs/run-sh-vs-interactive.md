# `run.sh` vs `run_qwen.sh` / `run_llama.sh` — Comparison

Quick reference for choosing between the interactive and non-interactive scripts.

---

## When to Use Which

| Scenario | Script |
|---|---|
| Running from Claude Code / AI agent | `run.sh` |
| CI/CD pipeline or cron job | `run.sh` |
| SSH session, want to explore interactively | `run_qwen.sh` / `run_llama.sh` |
| Headless server, unattended execution | `run.sh` |
| First-time user wanting guided walkthrough | `run_qwen.sh` / `run_llama.sh` |
| Custom one-off steering experiments | `run_qwen.sh` (interactive Stage 3 loop) |
| Batch steering over many prompts/vectors | `run.sh` |

---

## Feature Comparison

| Feature | `run_qwen.sh` | `run.sh` |
|---|---|---|
| Interactive prompts | Yes (14+ `read` calls) | None (zero `read` calls) |
| Model support | Qwen only (separate `run_llama.sh`) | Both via `--model` flag |
| Stage selection | `ask_yn` per stage | `--stages 1,2,3,4` |
| Trait selection | Numbered multi-select menu | `--traits evil,kind` |
| Resume | Interactive prompt | `--resume` flag |
| Stage 3 mode | Infinite loop (pick vector, pick prompt, repeat) | Batch: all vectors x all prompts |
| Stage 4 | Optional (yes/no prompt) | Include `4` in `--stages` |
| Python output | `tee` to screen + log | Redirect to log only |
| Screen output | Verbose, colorful | Structured `[STAGE X] [trait] status` |
| Lines of code | 712 | 330 |
| `state.env` format | `KEY=VALUE` | Same (compatible) |
| Between-trait control | c/s/q menu | Always continue |

---

## state.env Compatibility

Both scripts use the same `state.env` format. You can:

- Start a run with `run_qwen.sh`, interrupt, resume with `run.sh --resume`
- Start with `run.sh`, later inspect/continue interactively with `run_qwen.sh`

The state keys are identical:

```bash
SELECTED_TRAITS=evil,kind
STAGE1_evil_DONE=1
STAGE1_evil_POS_CSV=/absolute/path.csv
STAGE2_evil_DONE=1
STAGE2_evil_VECTORS_DIR=/absolute/path/
```

---

## Migration Examples

### Interactive to non-interactive

```bash
# Before (interactive)
bash scripts/run_qwen.sh
# → manually select traits, confirm each stage, pick vectors/prompts

# After (non-interactive)
bash scripts/run.sh --model qwen --traits evil,kind --stages 1,2,3 --samples 200
```

### Llama model

```bash
# Before
bash scripts/run_llama.sh

# After
bash scripts/run.sh --model llama --traits evil --stages 1,2,3
```

### Stage 3 with custom prompts

```bash
# Before (interactive): pick prompt from menu or type custom
bash scripts/run_qwen.sh  # then navigate to Stage 3, select options one by one

# After (non-interactive): specify everything upfront
bash scripts/run.sh --model qwen --traits evil --stages 3 \
    --prompts 'Custom prompt one|Custom prompt two|Custom prompt three' \
    --layer 15 --coefficients "-3,-1,0,1,3"
```
