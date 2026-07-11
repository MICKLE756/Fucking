# sweagent0 — 从零实现的 SWE Agent

自主定位并修复代码 Bug 的软件工程智能体，以 **SWE-bench Lite resolve rate** 为量化评测指标。
不依赖 LangChain 等框架，核心循环、工具系统、代码定位、上下文管理全部自研。

参考：[SWE-agent](https://github.com/SWE-agent/SWE-agent)（工具接口 ACI 设计）、
[Aider](https://github.com/Aider-AI/aider)（repo map 思路）、
[mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent)（极简循环，见 `../mini-swe-agent-demo`）。

## 架构

```
问题描述 ─┐
          ├─► 任务提示（含 repo map 仓库概览）
repo map ─┘        │
                   ▼
      ┌──── Agent 主循环（loop.py）────┐
      │  build_messages（上下文压缩）   │
      │        │                       │
      │        ▼                       │
      │   LLM（llm.py，OpenAI 兼容）    │
      │        │ 回复文本               │
      │        ▼                       │
      │   parser.py 解析 ```action 块  │──格式错误──► 错误信息回传，LLM 自纠正
      │        │ Action(tool, args)    │
      │        ▼                       │
      │   ToolRegistry 分发执行         │
      │   bash / editor / search /     │
      │   run_tests / git / submit     │
      │        │ 观察结果               │
      │        ▼                       │
      │   Trajectory 记录 step         │──► trajectory.json（复盘 / RL 训练数据）
      └────────┴── submit 则结束 ──────┘
                   │
                   ▼
             最终 patch（git diff）
```

### 模块说明

| 模块 | 文件 | 核心设计点 |
|---|---|---|
| Agent 循环 | `agent/loop.py` | 规划-执行-反思；格式错误自纠正 + 连续错误熔断 |
| 动作解析 | `agent/parser.py` | 单 ```action JSON 块，比 tool-calling 兼容性好（中转 API 友好） |
| 轨迹/压缩 | `agent/trajectory.py` | 保留全部思考文本 + 最近 5 步观察，早期观察超预算时替换为占位符 |
| 工具系统 | `tools/` | `editor.str_replace` 要求 old_str 唯一（防误改）；pytest 输出摘要提取；git 检查点/回滚 |
| Repo Map | `repomap/repo_map.py` | ast 抽取符号 + 按跨文件引用次数排序（PageRank 一阶近似），预算内优先展示核心文件 |
| 评测 | `eval/swebench.py` | SWE-bench Lite 全流程：clone → checkout base_commit → 跑 Agent → 输出官方格式 predictions.jsonl |

## 快速开始

```bash
python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'

# 单元测试（24 个，全部通过）
.venv/bin/python -m pytest tests -q

# 在任意仓库上跑一个任务
export OPENAI_API_KEY=你的key
# 中转 API 在 config.yaml 里配 base_url（见 config.example.yaml）
.venv/bin/sweagent0 --workdir /path/to/repo --task "修复 xxx bug" --config config.example.yaml
```

## SWE-bench Lite 评测

```bash
.venv/bin/pip install -e '.[eval]'

# 1. 生成预测（先用 --limit 10 小规模试跑）
.venv/bin/python -m sweagent0.eval.swebench --config config.yaml --limit 10 --output predictions.jsonl

# 2. 官方 harness 计算 resolve rate（需要 Docker）
.venv/bin/python -m swebench.harness.run_evaluation \
    --dataset_name princeton-nlp/SWE-bench_Lite \
    --predictions_path predictions.jsonl \
    --max_workers 4 --run_id run1
```

## 路线图（暑期里程碑）

- [x] **M1 骨架**：Agent 循环 / 工具系统 / repo map / 轨迹管理 / 评测入口（本 PR）
- [ ] **M2 跑通评测**：SWE-bench Lite 前 25 题，报出首个 resolve rate baseline
- [ ] **M3 迭代优化**（每项做消融，记录 resolve rate 变化）：
  - repo map 换 tree-sitter 支持多语言，检索定位加 BM25
  - 失败反思：测试失败后强制"分析-假设-验证"结构化反思步骤
  - Docker 沙箱执行环境（对齐官方评测环境）
- [ ] **M4 RL 进阶**：用轨迹 + 执行反馈构造训练数据，GRPO 微调 7B 模型的修复能力（参考 [SWE-Gym](https://github.com/SWE-Gym/SWE-Gym)、[TinyZero](https://github.com/Jiayi-Pan/TinyZero)）

## 设计取舍记录

- **为什么用文本 action 块而不是 function calling**：国内中转 API 对 tool-calling 支持参差不齐
  （踩坑记录见 `../mini-swe-agent-demo/README.md`），正则解析 + 自纠正回路兼容性最好。
- **为什么 str_replace 而不是整文件重写**：整文件重写在大文件上又贵又易丢代码；
  行号编辑在多轮修改后行号漂移。唯一性约束的 str_replace 是 SWE-agent 验证过的最稳方案。
- **为什么压缩观察结果而不是思考文本**：观察结果（文件内容、测试输出）可随时重新获取，
  思考文本承载推理链，丢了会导致 Agent 重复劳动或自相矛盾。
