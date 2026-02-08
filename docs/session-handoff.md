# Session Handoff — 实验脚本开发记录

> 写给下一次 Claude 会话的自己。在实验电脑上恢复上下文用。

---

## 已完成的工作

### 创建了两个交互式实验复现脚本

1. **`scripts/run_qwen.sh`** — Qwen2.5-7B-Instruct 完整流程
2. **`scripts/run_llama.sh`** — Meta-Llama-3.1-8B-Instruct 完整流程

两个脚本结构完全相同，仅模型参数不同：

| 变量 | run_qwen.sh | run_llama.sh |
|---|---|---|
| `MODEL` | `Qwen/Qwen2.5-7B-Instruct` | `meta-llama/Meta-Llama-3.1-8B-Instruct` |
| `MODEL_SHORT` | `Qwen2.5-7B-Instruct` | `Meta-Llama-3.1-8B-Instruct` |
| `NUM_LAYERS` | 28 | 32 |
| `DEFAULT_LAYER` | 20 | 20 |
| 额外环境检查 | 无 | HuggingFace token |

### 脚本功能概述

- **日志系统**：`logs/{MODEL_SHORT}_{YYYYMMDD_HHMMSS}/` 下分阶段记录日志，Python 输出通过 `tee` 同时写终端和文件
- **状态保存 (`state.env`)**：每完成一步写入状态，支持断点续跑。启动时自动检测上次未完成的运行
- **阶段间自动传递**：Stage 1 CSV → Stage 2 默认 `responses_dir`；Stage 2 `.pt` 文件 → Stage 3 自动列出可选向量
- **交互控制**：trait 多选、每步 c/s/q、Stage 3 循环切换向量/prompt
- **4 个阶段**：生成响应 → 提取向量 → Steering 实验 → 可选向量分析

### 验证状态

- `bash -n` 语法检查：两个脚本均通过
- `chmod +x`：已设置可执行权限
- **尚未在有 GPU 的机器上做实际运行测试**

---

## 下一步：在实验电脑上测试

### 前置条件检查

```bash
# 确认 GPU 可用
nvidia-smi

# 确认 Python 环境
python --version
pip list | grep -E "vllm|transformers|torch|fire|openai"

# 确认 API Key
echo $OPENAI_API_KEY

# Llama 需要 HuggingFace 认证
huggingface-cli whoami
```

### 测试计划

#### 1. 快速冒烟测试（不跑完整实验）

先用少量样本验证整个流程能走通：

```bash
# 编辑脚本中 samples_per_instruction 从 200 改为 5 做快速测试
# 或者直接运行脚本，在交互中手动测试各个分支：
bash scripts/run_qwen.sh
```

测试要点：
- 日志目录 `logs/` 是否正确创建
- `state.env` 是否正确写入
- 中断后重新运行，是否正确检测到上次运行并提示恢复
- 各阶段间的路径传递是否正确

#### 2. 完整实验运行

按以下顺序，建议先跑 Qwen（不需要 HF 认证，流程更简单）：

```bash
# Qwen 完整实验
bash scripts/run_qwen.sh
# 选择所有 3 个 traits: evil, kind, compliant
# Stage 1 每组约 1 小时（生成 + GPT-4o 评分）
# Stage 2 每组约 30 分钟（提取向量）
# Stage 3 交互式，每次约 1 分钟

# Llama 完整实验
bash scripts/run_llama.sh
```

#### 3. 如果脚本有 bug

脚本调用的 Python 命令就是 `docs/reproduction-guide.md` 里记录的标准命令。所有 Python 脚本使用 Fire 做 CLI 解析，参数格式为 `--arg value` 或 `--arg=value`。

核心 Python 脚本接口：
- `generate_combined_responses.py --model MODEL --trait TRAIT --output_dir data/responses --samples_per_instruction 200`
- `extract_vectors.py --model MODEL --trait TRAIT --responses_dir data/responses --output_dir data/vectors`
- `run_steering.py --model MODEL --vector PATH --prompt "..." --layers "20" --coefficients "-2,-1,0,1,2"`
- `analyze_vectors.py --vector1 PATH --vector2 PATH`

---

## 项目结构速查

```
scripts/
├── run_qwen.sh                        ← 新建
├── run_llama.sh                       ← 新建
├── generate_combined_responses.py     # Stage 1: vLLM 生成 + GPT-4o 评分
├── extract_vectors.py                 # Stage 2: 提取 hidden states → 差异向量
├── run_steering.py                    # Stage 3: 向量注入生成
├── analyze_vectors.py                 # 可选: 向量对比分析
├── compare_vectors.py                 # 可选: 跨 trait 比较
└── generate_responses.py              # 旧版生成脚本（不用）

configs/
├── models/qwen2.5-7b.yaml            # name, num_layers=28, default_steering_layer=20
├── models/llama3-8b.yaml             # name, num_layers=32, default_steering_layer=20
└── traits/{evil,kind,compliant}.yaml  # instructions, questions, scoring thresholds

data/
├── responses/                         # Stage 1 输出: {MODEL_SHORT}_{trait}_{positive,negative}.csv
└── vectors/{MODEL_SHORT}/             # Stage 2 输出: {trait}_{model_persona,user_persona,i_thou}_{position}.pt
```
