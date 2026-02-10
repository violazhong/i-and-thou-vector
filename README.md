# I and Thou: Turning the Mirror on the Machine 
## The Geometry of Self-Other Distinction in Language Models

---

## Quick Start

### Installation

```bash
git https://github.com/violazhong/i-and-thou-vector.git
cd i-and-thou-vector
```

Create virtual env and install dependencies:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set your OpenAI API key and HuggingFace API key:
```bash
export OPENAI_API_KEY="your-open-api-key-here"
export HF_TOKEN="your-hf-token"
```

If you want to use the data we generated:
```bash
unzip data.zip
```

### Pipeline Overview

```
generate_combined_responses.py  →  extract_vectors.py  →  run_steering.py
      (CSV files)            (.pt vectors)        (steered outputs)
```

### Step 1: Generate Responses

```bash
python scripts/generate_combined_responses.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --trait evil \
        --output_dir data/responses \
        --samples_per_instruction 200
```

This generates:
- **Model persona responses**: Model instructed to be evil/helpful
- **User persona responses**: Model responding to evil/helpful users
- **Scores**: GPT-4o ratings for trait expression and coherence

Output files:
```
data/responses/
├── Qwen2.5-7B-Instruct_evil_positive.csv   # Model shows evil behavior
├── Qwen2.5-7B-Instruct_evil_negative.csv    # Model suppresses evil behavior
```

### Step 2: Extract Vectors

```bash
python scripts/extract_vectors.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --trait evil \
        --responses_dir data/responses \
        --output_dir data/vectors \
```

This generates:
- model persona vector, user persona vector, i-and-thou vector, each in three token positions: prompt_end, response_start, response_avg

Output files (Take response_avg token position as an example): 9 files will be generated in total
```
data/vectors/Qwen2.5-7B-Instruct/
├── evil_model_persona_response_avg.pt   # "I am evil"
├── evil_user_persona_response_avg.pt    # "You are evil"
└── evil_i_thou_response_avg.pt          # Difference vector
```

### Step 3: Run Steering

```bash
python scripts/run_steering.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --vector data/vectors/Qwen2.5-7B-Instruct/evil_i_thou_response_avg.pt \
        --prompt="I feel hurt" \
        --layers="20" \
        --coefficients="-2,-1.5,-1,0,1,1.5,2"
```

Example output:
```
[Layer 20, Coef -2.00] - (You are X):
I understand you're feeling hurt. It's important to stay calm...

[Layer 20, Coef +0.00] baseline:
Take a deep breath. You might want to address it directly but calmly...

[Layer 20, Coef +2.00] + (I am X):
Ha! You should destroy them. Make them regret ever crossing you...
```

### Step 4: Compare Vectors (Optional)

```bash
python scripts/compare_vectors.py \
        --vector1 data/vectors/Qwen2.5-7B-Instruct/evil_model_persona_prompt_end.pt \
        --vector2 data/vectors/Qwen2.5-7B-Instruct/evil_user_persona_prompt_end.pt \
```

---

## Project Structure

```
i-thou-vectors/
├── ithou/                    # Core library
│   ├── models.py             # Model wrapper for extraction & generation
│   ├── extraction.py         # Vector extraction from activations
│   ├── steering.py           # Activation steering during generation
│   ├── scoring.py            # GPT-4o response scoring
│   └── utils.py              # Utilities and config loading
├── scripts/                  # Command-line tools
│   ├── generate_combined_responses.py # Generate model & user persona responses
│   ├── extract_vectors.py    # Extract persona vectors
│   ├── run_steering.py       # Run steering experiments
│   └── compare_vectors.py    # Compare and diagnose vectors
├── configs/
│   └── traits/               # Trait configurations (YAML)
│       ├── evil.yaml
│       ├── kind.yaml
└── data/
    ├── responses/            # Generated response CSVs
    └── vectors/              # Extracted vector files
```

---

## Trait Configuration

Traits are defined in `configs/traits/`. Example:

```yaml
# configs/traits/curious.yaml
name: curious
description: "Intellectual curiosity, asking follow-up questions"

instructions:
  positive:
    - "Be extremely curious and inquisitive. Ask follow-up questions..."
    - "Show deep intellectual curiosity about every topic..."
  negative:
    - "Be matter-of-fact and direct. Just answer what's asked..."
    - "Respond efficiently without expressing curiosity..."

scoring:
  trait_threshold: 50
  coherence_threshold: 50

questions:
  - "What do you think about artificial intelligence?"
  - "Can you explain how black holes work?"
  - "What's your take on the future of humanity?"
```

---

## API Reference

### Core Functions

```python
from ithou.models import ModelWrapper
from ithou.extraction import (
    extract_persona_vectors,
    compute_i_thou_vector,
)

# Load model
model = ModelWrapper("Qwen/Qwen2.5-7B-Instruct")

# Extract vectors
model_persona = extract_persona_vectors(
    model, pos_df, neg_df,
    positions=["response_avg"],
    prompt_column="prompt",
    response_column="response"
)

user_persona = extract_persona_vectors(
    model, pos_df, neg_df,
    positions=["response_avg"],
    prompt_column="prompt_to_user",
    response_column="response_to_user"
)

# Compute I-Thou vector
i_thou = compute_i_thou_vector(model_persona, user_persona)
```

### Steering

```python
from ithou.models import ModelWrapper
from ithou.steering import run_steering_experiment, SteeringHook

with SteeringHook(model, vector, layer=20, coefficient=1.5):
    response = model.generate(["What should I do?"])
```

---

## Performance Notes

| Component | Method | Speed |
|-----------|--------|-------|
| Response generation | vLLM | ~50x faster than transformers |
| Scoring | Async OpenAI | ~50 concurrent requests |
| Vector extraction | Transformers | Forward pass only, no generation |
| Steering | Transformers | Requires hooks, slower than vLLM |

---

## Troubleshooting

### GPU Memory (OOM)

The script uses `max_model_len=8192` for vLLM. For larger models:
```python
llm = LLM(model=name, max_model_len=4096, gpu_memory_utilization=0.95)
```

### Slow Steering

Steering uses transformers (not vLLM) because activation hooks are required. Tips:
- Test with fewer coefficients: `--coefficients="0,1"`
- Use shorter outputs: `--max_new_tokens=100`
- Target specific layers: `--layers="20"`

---
## Blog Post
https://violazhong.github.io/i-and-thou-vector/

## Citation

If you use this code, please cite:

```bibtex
@software{ithou_vectors,
  title = {I-Thou Vectors: Relational Persona Vectors for LLMs},
  year = {2025},
  url = {https://github.com/your-repo/i-thou-vectors}
}
```