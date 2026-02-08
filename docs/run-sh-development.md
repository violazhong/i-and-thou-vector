# `scripts/run.sh` — Development Notes

Record of why and how this script was built, design decisions, testing, and known limitations.

---

## 1. Problem Statement

The original scripts (`run_qwen.sh`, `run_llama.sh`, each ~712 lines) were fully interactive. Every flow-control point used `read` to get user input:

- `ask_yn()` — yes/no prompts (resume, run stage, continue?)
- `ask_action()` — continue/skip/quit after each trait
- `select_traits()` — multi-select from numbered list
- `select_file()` — pick vector from numbered list
- Stage 3 — infinite loop with `read` for vector selection, prompt selection, coefficients, layer, next-action

When called from a non-terminal environment (e.g. Claude Code's Bash tool, CI, `cron`), `read` gets EOF from empty stdin. Combined with `set -euo pipefail`, the script exits immediately. There was no way to run the experiment pipeline without a human at the keyboard.

---

## 2. Design Decisions

### 2.1 New file, not a patch

Created `scripts/run.sh` as a new file rather than modifying `run_qwen.sh`. Reasons:

- The original scripts are already documented in `docs/session-handoff.md` and `docs/reproduction-guide.md`
- A patch would have been larger than a rewrite — every function needed changes
- Users who prefer the interactive flow can still use the original scripts
- No risk of breaking existing workflows

### 2.2 Zero `read` calls

The core constraint: **the script must never block waiting for stdin**. This was achieved by:

- Replacing all `read`-based user input with CLI flags (`--model`, `--traits`, `--stages`, etc.)
- Giving every flag a sensible default so the script runs with zero arguments
- Replacing the interactive Stage 3 loop with batch iteration over all (vector, prompt) combinations

### 2.3 Unified model support

Instead of separate `run_qwen.sh` and `run_llama.sh`, a single `--model` flag with a `case` mapping:

```bash
case "$MODEL" in
    qwen|qwen2.5|Qwen)  MODEL_FULL="Qwen/Qwen2.5-7B-Instruct"; ...
    llama|llama3|Llama)  MODEL_FULL="meta-llama/Meta-Llama-3.1-8B-Instruct"; ...
    *)                   MODEL_FULL="$MODEL"; ...  # passthrough
esac
```

Accepts multiple shorthand aliases and falls back to treating the argument as a full HuggingFace path.

### 2.4 State compatibility

The `state.env` format is identical to `run_qwen.sh`:

```bash
STAGE1_evil_DONE=1
STAGE1_evil_POS_CSV=/path/to/file.csv
```

This means `--resume` can pick up a run originally started by `run_qwen.sh`, and vice versa.

### 2.5 Structured logging for AI consumption

Every stage operation prints a machine-parseable line:

```
[STAGE 1] [evil] START generate_combined_responses.py (samples=5)
[STAGE 1] [evil] DONE pos=1000 rows, neg=1000 rows
[STAGE 2] [evil] SKIP (already done) dir=/path/to/vectors
[STAGE 3] [evil_i_thou_response_avg] [3/15] DONE
```

This format lets an AI agent (or grep/awk) easily determine what happened without parsing free-form prose.

### 2.6 Fail-fast for critical stages, warn-only for experiments

- **Stage 1 & 2:** `exit 1` on failure — these are prerequisites; there's no point continuing
- **Stage 3 & 4:** warn and continue — a single steering experiment failing shouldn't abort the entire batch

### 2.7 `--no-score` was dropped

During planning, we considered a `--no-score` flag to skip GPT-4o scoring in Stage 1. After reading `generate_combined_responses.py`, we found that generation and scoring are tightly coupled in `main()` — scoring happens inline after generation. Splitting them would require modifying the Python script, which was out of scope. The simpler alternative: use `--samples 2` for fast testing.

### 2.8 Python output redirected, not piped through `tee`

The original used `run_python()` which piped through `tee`:

```bash
"$@" 2>&1 | tee -a "$logfile"
```

This requires a TTY for clean output and can cause issues in non-interactive environments. The new script uses simple redirection:

```bash
python script.py >> "$stage_log_file" 2>&1
```

Detailed Python output goes to per-stage log files. The script's own `stage_log` calls provide concise progress to stdout.

---

## 3. Development Steps

### Step 1: Codebase analysis

Read all source files to understand:

- The full flow of `run_qwen.sh` (712 lines, 4 stages, interactive helpers)
- All Python scripts' CLI arguments (all use `fire.Fire()`)
- Model config YAML files (paths, layer counts, default steering layers)
- Trait config YAML files (instructions, questions, scoring thresholds)
- State management format (key=value in `state.env`)

### Step 2: Script design

Mapped the interactive flow to CLI flags:

| Interactive element | CLI replacement |
|---|---|
| `select_traits()` with `read` | `--traits evil,kind` |
| `ask_yn "Run Stage X?"` | `--stages 1,2,3` |
| `ask_yn "Resume?"` | `--resume` flag |
| Stage 3 vector selection loop | `--vectors` or auto-discover |
| Stage 3 prompt selection loop | `--prompts` or built-in defaults |
| Stage 3 coefficient/layer `read` | `--coefficients`, `--layer` |
| `ask_action` between traits | Removed — always continue |

### Step 3: Implementation

Wrote the script in one pass (~330 lines vs 712 original), structured as:

1. CLI argument parsing (`parse_args`)
2. Model config mapping (`setup_model_config`)
3. Environment checks (`check_env`) — non-interactive, fail-fast
4. Run directory setup (`setup_run`) — create or resume
5. Four stage functions (`stage1_generate`, `stage2_extract`, `stage3_steer`, `stage4_analyze`)
6. Main dispatcher that iterates `--stages`

### Step 4: Verification

1. **Syntax check:** `bash -n scripts/run.sh` — passed
2. **Help output:** `bash scripts/run.sh --help` — correct formatting
3. **User testing:** User ran the script with real model/GPU — confirmed working

---

## 4. Key Highlights

1. **330 lines vs 712** — 54% smaller while supporting both models
2. **Zero `read` calls** — safe in any non-interactive environment
3. **Backwards-compatible `state.env`** — can resume runs from the original scripts
4. **Batch Stage 3** — runs all vector/prompt combinations automatically instead of requiring manual selection each time
5. **All flags have defaults** — `bash scripts/run.sh` works with zero arguments
6. **Model passthrough** — unknown model names are treated as full HF paths, so any model works without code changes

---

## 5. Known Limitations and Observations

### 5.1 No `--no-score` support

GPT-4o scoring in Stage 1 cannot be skipped without modifying `generate_combined_responses.py`. Workaround: use `--samples 2` for fast iteration.

### 5.2 Stage 3 reloads the model for every prompt

Each `run_steering.py` call loads the model from scratch. For 3 vectors and 5 prompts, that's 15 model loads. The original interactive script had the same behavior. A future optimization would be a batch-mode Python script that loads the model once and iterates.

### 5.3 No per-experiment checkpointing in Stage 3

Stages 1 and 2 have per-trait checkpointing via `state.env`. Stage 3 does not — if it's interrupted, all experiments re-run on resume. This is acceptable because Stage 3 experiments are fast individually (no GPT-4o API calls), and the steering log file serves as a record of what completed.

### 5.4 `generate_combined_responses.py` has built-in skip logic

The Python script itself checks if output CSVs already exist and returns early. This means even without `state.env`, re-running Stage 1 for the same trait is safe (it won't regenerate). The bash-level checkpoint just avoids the Python startup overhead.

### 5.5 Log visibility trade-off

The original `tee`-based approach showed full Python output on screen. The new redirect approach only shows structured `stage_log` lines on screen, with full Python output in per-stage log files. This is better for automation but means a human watching the terminal sees less detail. The log files are always available for debugging.

### 5.6 No parallel trait execution

Traits are processed sequentially within each stage. For Stage 1 (GPU-bound vLLM generation), this is unavoidable on a single GPU. For Stage 2, parallel execution could theoretically speed things up, but the sequential approach is simpler and avoids GPU memory contention.
