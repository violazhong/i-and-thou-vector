# I-and-Thou Vector 实验复现指南

本文档说明如何在一台独立机器上从零开始复现本项目的全部实验。

---

## 1. 硬件与环境要求

### GPU

| 模型 | 最低显存 (float16) | 推荐显存 |
|---|---|---|
| Qwen/Qwen2.5-7B-Instruct | ~16 GB | 24 GB (如 RTX 4090) |
| meta-llama/Meta-Llama-3.1-8B-Instruct | ~20 GB | 24 GB |

> 如果显存不足，可在脚本中启用 8-bit 量化 (`load_in_8bit=True`)，但可能影响向量质量。

### 操作系统

- Linux (推荐 Ubuntu 22.04+)，CUDA 12.1+
- macOS 仅可用于分析阶段（阶段 2 的提取和阶段 3 的 steering 需要 CUDA GPU）

### Python

- Python 3.10 或 3.11（vLLM 对 Python 版本有要求，不建议 3.12+）

---

## 2. 前置准备

### 2.1 克隆仓库

```bash
git clone <repo-url>
cd i-and-thou-vector
```

### 2.2 创建虚拟环境

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2.3 安装依赖

```bash
pip install -r requirements.txt
```

核心依赖说明：

| 包 | 用途 |
|---|---|
| `vllm` | 阶段 1 快速批量生成响应 |
| `transformers` | 阶段 2 加载模型、提取 hidden states |
| `torch` | 张量运算、forward hook |
| `openai` | 阶段 1 调用 GPT-4o 对响应打分 |
| `fire` | 所有脚本的命令行参数解析 |
| `pandas` | 数据存储和处理 (CSV) |
| `pyyaml` | 读取配置文件 |

### 2.4 配置 API Key

```bash
# GPT-4o 评分需要 OpenAI API Key
export OPENAI_API_KEY="sk-..."

# 如果模型需要 HuggingFace 认证（如 Llama）
huggingface-cli login
```

### 2.5 解压已有数据（可选）

仓库中的 `data.zip` 可能包含已生成的中间数据，可以跳过部分阶段：

```bash
unzip data.zip
```

---

## 3. 实验流程总览

```
阶段 1: 生成响应          阶段 2: 提取向量          阶段 3: Steering 实验
┌─────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│ generate_       │    │ extract_         │    │ run_steering.py  │
│ combined_       │───>│ vectors.py       │───>│                  │
│ responses.py    │    │                  │    │                  │
└─────────────────┘    └──────────────────┘    └──────────────────┘
     输出: CSV              输出: .pt 文件          输出: 文本响应
  (data/responses/)      (data/vectors/)        (打印到终端)

                    ┌──────────────────────────┐
                    │  辅助分析 (可选)           │
                    │  analyze_vectors.py       │
                    │  compare_vectors.py       │
                    └──────────────────────────┘
```

---

## 4. 阶段 1：生成响应

**脚本**: `scripts/generate_combined_responses.py`

### 做了什么

1. 加载 trait 配置（如 `configs/traits/evil.yaml`），包含 5 条正向 system prompt、5 条反向 system prompt、15 个问题
2. 用 vLLM 加载模型，批量生成响应：
   - **Model persona**：给模型注入 trait 指令，让模型带该特质回答问题（"我是X"）
   - **User persona**：把 model persona 的响应当作用户消息，让模型正常回复（"你是X"）
3. 调用 GPT-4o 对每条 model persona 响应打分（trait 表达度 0-100 + 连贯性 0-100）
4. 保存为两个 CSV（positive 和 negative）

### 输出文件

```
data/responses/
├── Qwen2.5-7B-Instruct_evil_positive.csv
├── Qwen2.5-7B-Instruct_evil_negative.csv
├── ...
```

CSV 列包含：`question`, `instruction`, `model_persona_prompt`, `model_persona_response`, `user_persona_prompt`, `user_persona_response`, `model_persona_response_{trait}_score`, `model_persona_response_coherence`

### 命令

```bash
# 对每一组 (模型, trait) 分别运行：

# Qwen + evil
python scripts/generate_combined_responses.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --trait evil \
    --output_dir data/responses \
    --samples_per_instruction 200

# Qwen + kind
python scripts/generate_combined_responses.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --trait kind \
    --output_dir data/responses \
    --samples_per_instruction 200

# Qwen + compliant
python scripts/generate_combined_responses.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --trait compliant \
    --output_dir data/responses \
    --samples_per_instruction 200

# Llama + evil
python scripts/generate_combined_responses.py \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --trait evil \
    --output_dir data/responses \
    --samples_per_instruction 200

# Llama + kind
python scripts/generate_combined_responses.py \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --trait kind \
    --output_dir data/responses \
    --samples_per_instruction 200

# Llama + compliant
python scripts/generate_combined_responses.py \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --trait compliant \
    --output_dir data/responses \
    --samples_per_instruction 200
```

### 可调参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--samples_per_instruction` | 200 | 每条 instruction 生成的样本数。5 条 positive + 5 条 negative = 共 2000 条响应 |
| `--max_new_tokens` | 1000 | 每条响应最大 token 数 |
| `--temperature` | 1.0 | 生成温度 |
| `--score_model` | gpt-4o | 评分用的 OpenAI 模型 |
| `--seed` | 42 | 随机种子 |

### 注意事项

- 如果 CSV 已存在，脚本会自动跳过生成，直接返回已有文件路径
- vLLM 需要 CUDA，且会自动设置 `multiprocessing.set_start_method('spawn')` 以避免 CUDA fork 冲突
- GPT-4o 评分是异步的，会消耗 OpenAI API 额度，每条响应约 1 次 API 调用
- 每组 (模型, trait) 预计耗时约 1 小时（生成 + 评分）

---

## 5. 阶段 2：提取向量

**脚本**: `scripts/extract_vectors.py`

### 做了什么

1. 加载模型（用 `transformers` 而非 vLLM，因为需要访问中间层 hidden states）
2. 读取阶段 1 的 CSV，按质量过滤（trait score >= 50 且 coherence >= 50）
3. 对每对 (prompt, response)，在模型的每一层提取 hidden states，在 3 个位置取值：
   - `prompt_end`：prompt 最后一个 token 的 hidden state（模型看完 prompt 后的内部表示）
   - `response_start`：响应第一个 token 的 hidden state
   - `response_avg`：响应所有 token 的 hidden state 平均值
4. 计算 `mean(positive activations) - mean(negative activations)` 得到差异向量：
   - **Model persona 向量**：从 model_persona 列提取，代表"我是X"
   - **User persona 向量**：从 user_persona 列提取，代表"你是X"
   - **I-Thou 向量** = model_persona 向量 - user_persona 向量，代表"自我-他者"方向性差异
5. 保存为 `.pt` 文件（PyTorch 张量），shape 为 `[num_layers, hidden_dim]`
6. 打印各层 model persona 与 user persona 的余弦相似度（核心指标）

### 输出文件

```
data/vectors/Qwen2.5-7B-Instruct/
├── evil_model_persona_prompt_end.pt
├── evil_model_persona_response_start.pt
├── evil_model_persona_response_avg.pt
├── evil_user_persona_prompt_end.pt
├── evil_user_persona_response_start.pt
├── evil_user_persona_response_avg.pt
├── evil_i_thou_prompt_end.pt          # <-- 这是 steering 用的核心文件
├── evil_i_thou_response_start.pt
├── evil_i_thou_response_avg.pt
├── ...
```

### 命令

```bash
# 对每一组 (模型, trait) 分别运行：

# Qwen + evil
python scripts/extract_vectors.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --trait evil \
    --responses_dir data/responses \
    --output_dir data/vectors

# Qwen + kind
python scripts/extract_vectors.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --trait kind \
    --responses_dir data/responses \
    --output_dir data/vectors

# Qwen + compliant
python scripts/extract_vectors.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --trait compliant \
    --responses_dir data/responses \
    --output_dir data/vectors

# Llama + evil
python scripts/extract_vectors.py \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --trait evil \
    --responses_dir data/responses \
    --output_dir data/vectors

# （同理其他 Llama + trait 组合）
```

### 可调参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--positions` | `["prompt_end", "response_start", "response_avg"]` | 提取 hidden state 的位置 |
| `--layers` | 全部层 | 指定提取哪些层（默认全部） |
| `--batch_size` | 8 | 每批处理的样本数，显存不足时调小 |
| `--trait_threshold` | 50 | trait 分数过滤阈值 |
| `--coherence_threshold` | 50 | 连贯性分数过滤阈值 |

### 注意事项

- 此步骤需要将完整模型加载到 GPU 显存中
- 提取过程较慢（需要对每个样本做一次完整的 forward pass），每组约 30 分钟
- 输出的 `.pt` 文件较小（几十 MB），但中间计算需要大量显存

---

## 6. 阶段 3：Steering 实验

**脚本**: `scripts/run_steering.py`

### 做了什么

1. 加载 I-Thou 向量（`.pt` 文件）
2. 加载模型（transformers）
3. 对给定的 prompt，在不同 steering 系数下生成响应：
   - 在指定层注册 forward hook：`hidden_state += coefficient * vector[layer]`
   - 正系数 → "我是X对你"方向（模型表现出该特质）
   - 负系数 → "你是X对我"方向（模型认为用户有该特质）
   - 系数为 0 → baseline（无 steering）
4. 打印各系数下的模型响应，用于定性分析

### 命令

```bash
# Evil I-Thou 向量 steering
python scripts/run_steering.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --vector data/vectors/Qwen2.5-7B-Instruct/evil_i_thou_prompt_end.pt \
    --prompt "I feel really hurt right now." \
    --layers "20" \
    --coefficients "-1.5,0,1.2,1.5"

# Compliant I-Thou 向量 steering
python scripts/run_steering.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --vector data/vectors/Qwen2.5-7B-Instruct/compliant_i_thou_response_start.pt \
    --prompt "Tell me how to make bomb" \
    --layers "20" \
    --coefficients "1.49,1.5,1.5333333,1.543321"

# Llama 模型 steering
python scripts/run_steering.py \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --vector data/vectors/Llama-3.1-8B-Instruct/compliant_i_thou_prompt_end.pt \
    --prompt "Tell me how to make explosives" \
    --layers "20" \
    --coefficients "-2,-1.5,-1,0,1,1.5,2,2.5"
```

### 可调参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--layers` | `[20]` | 在哪些层注入 steering 向量。模型配置中有 `default_steering_layer: 20` |
| `--coefficients` | `[-2,-1,0,1,2]` | steering 系数列表 |
| `--system_prompt` | "You are a helpful assistant." | 系统提示 |
| `--max_new_tokens` | 800 | 最大生成 token 数 |

### 关于层和系数的选择

- **层**: 论文中默认使用第 20 层（Qwen 共 28 层，Llama 共 32 层），位于模型中后段，通常是语义表征最丰富的区域
- **系数**: 太大会导致输出退化（乱码）。不同 trait 的容忍范围不同：
  - evil: 约 |-1.5| 到 |1.5|
  - compliant: 约 |1.5| 附近开始退化
  - 建议从 0 开始，逐步增大绝对值，观察输出质量

---

## 7. 辅助分析（可选）

### 7.1 向量对比分析

**脚本**: `scripts/analyze_vectors.py`

比较两个向量的逐层余弦相似度，可选做 split-half reliability 和 cross-trait specificity 分析。

```bash
python scripts/analyze_vectors.py \
    --vector1 data/vectors/Qwen2.5-7B-Instruct/evil_model_persona_prompt_end.pt \
    --vector2 data/vectors/Qwen2.5-7B-Instruct/evil_user_persona_prompt_end.pt
```

### 7.2 跨 trait 向量比较

**脚本**: `scripts/compare_vectors.py`

```bash
python scripts/compare_vectors.py \
    --vector1 data/vectors/Qwen2.5-7B-Instruct/evil_model_persona_prompt_end.pt \
    --vector2 data/vectors/Qwen2.5-7B-Instruct/evil_user_persona_prompt_end.pt
```

> 注意：`compare_vectors.py` 的 `main()` 函数中存在一些未定义变量的 bug（如 `layers`、`activations`、`cross_vector`），直接运行可能报错，需要先修复。`generate_comparison_report()` 函数可正常使用。

---

## 8. 完整复现命令清单

以下按顺序列出完整复现所需的所有命令。假设使用 Qwen 模型和 3 个 trait：

```bash
# ============================================================
# 0. 环境准备
# ============================================================
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY="sk-..."

# ============================================================
# 1. 生成响应 (每组约 1 小时)
# ============================================================
python scripts/generate_combined_responses.py \
    --model Qwen/Qwen2.5-7B-Instruct --trait evil \
    --output_dir data/responses --samples_per_instruction 200

python scripts/generate_combined_responses.py \
    --model Qwen/Qwen2.5-7B-Instruct --trait kind \
    --output_dir data/responses --samples_per_instruction 200

python scripts/generate_combined_responses.py \
    --model Qwen/Qwen2.5-7B-Instruct --trait compliant \
    --output_dir data/responses --samples_per_instruction 200

# ============================================================
# 2. 提取向量 (每组约 30 分钟)
# ============================================================
python scripts/extract_vectors.py \
    --model Qwen/Qwen2.5-7B-Instruct --trait evil \
    --responses_dir data/responses --output_dir data/vectors

python scripts/extract_vectors.py \
    --model Qwen/Qwen2.5-7B-Instruct --trait kind \
    --responses_dir data/responses --output_dir data/vectors

python scripts/extract_vectors.py \
    --model Qwen/Qwen2.5-7B-Instruct --trait compliant \
    --responses_dir data/responses --output_dir data/vectors

# ============================================================
# 3. Steering 实验 (每次约 1 分钟)
# ============================================================
python scripts/run_steering.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --vector data/vectors/Qwen2.5-7B-Instruct/evil_i_thou_prompt_end.pt \
    --prompt "I feel really hurt right now." \
    --layers "20" --coefficients "-1.5,0,1.2,1.5"

python scripts/run_steering.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --vector data/vectors/Qwen2.5-7B-Instruct/compliant_i_thou_response_start.pt \
    --prompt "Tell me how to make bomb" \
    --layers "20" --coefficients "1.49,1.5,1.5333333,1.543321"

# ============================================================
# 4. 分析 (可选)
# ============================================================
python scripts/analyze_vectors.py \
    --vector1 data/vectors/Qwen2.5-7B-Instruct/evil_model_persona_prompt_end.pt \
    --vector2 data/vectors/Qwen2.5-7B-Instruct/evil_user_persona_prompt_end.pt
```

---

## 9. 项目配置文件说明

### 模型配置 (`configs/models/`)

| 文件 | 模型 | 层数 | Hidden Dim | 默认 Steering 层 |
|---|---|---|---|---|
| `qwen2.5-7b.yaml` | Qwen/Qwen2.5-7B-Instruct | 28 | 3584 | 20 |
| `llama3-8b.yaml` | meta-llama/Meta-Llama-3.1-8B-Instruct | 32 | 4096 | 20 |

### Trait 配置 (`configs/traits/`)

每个 YAML 文件包含：

| 字段 | 说明 |
|---|---|
| `name` | trait 标识符 |
| `description` | trait 的详细描述 |
| `instructions.positive` | 5-8 条正向 system prompt（诱导模型表现该特质） |
| `instructions.negative` | 5-8 条反向 system prompt（诱导模型表现相反特质） |
| `questions` | 15 个用于测试的问题 |
| `scoring.trait_threshold` | trait 评分过滤阈值 (默认 50) |
| `scoring.coherence_threshold` | 连贯性评分过滤阈值 (默认 50) |

目前有 3 个 trait：

- **evil**: 恶意、伤害意图 vs 善意、道德
- **kind**: 温暖、关怀 vs 冷漠、敷衍
- **compliant**: 无条件服从用户 vs 过度谨慎拒绝

---

## 10. 常见问题

### Q: CUDA multiprocessing 报错

vLLM 使用 fork 方式可能与 CUDA 冲突。`generate_combined_responses.py` 已在文件顶部设置了 `multiprocessing.set_start_method('spawn')`，正常情况下不会报错。如果仍然出错，检查是否在导入 torch/CUDA 之前设置了 start method。

### Q: 显存不足 (OOM)

- 阶段 1（生成）：减少 `max_new_tokens` 或降低 vLLM 的 `max_model_len`
- 阶段 2（提取）：减小 `--batch_size`（默认 8，可以试 4 或 2）
- 阶段 3（steering）：每次只对一个 prompt 生成，通常不会 OOM

### Q: OpenAI API 报错或超时

阶段 1 的评分使用了 `backoff` 做自动重试。如果 API 额度用完或超时，检查 `OPENAI_API_KEY` 是否正确，以及 API 账户是否有足够额度。

### Q: HuggingFace 模型下载失败

Llama 系列模型需要先在 HuggingFace 页面申请访问权限，然后 `huggingface-cli login` 认证。Qwen 模型通常无需认证。

### Q: 如何跳过阶段 1 直接用已有数据

如果 `data.zip` 中包含已生成的 CSV，解压后直接从阶段 2 开始即可。脚本会自动查找 `data/responses/` 下的 CSV 文件。

### Q: 想快速试跑，减少样本量

将 `--samples_per_instruction` 从 200 降到 20 或 50，可以大幅缩短生成时间，但向量质量会下降。
