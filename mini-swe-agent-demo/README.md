# mini-swe-agent 完整部署与运行指南

> 本目录记录了在虚拟机上完整部署 [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) 并用国内中转 API（gpt-5.5）端到端跑通一个真实编码任务的全过程，含踩坑记录与可复现配置。

## 一、mini-swe-agent 是什么

普林斯顿/斯坦福 SWE-bench 团队出品的**极简 AI 软件工程 agent**：核心 agent 循环只有约 100 行 Python，却能在 SWE-bench Verified 上拿到 >74% 的分数。它的动作空间只有 bash——LLM 每步输出一条 bash 命令，执行后把结果追加进对话历史，循环往复直到任务完成。

源码结构（`src/minisweagent/`）：

| 目录/文件 | 作用 |
|---|---|
| `agents/default.py`（188 行） | **核心 agent 循环**：query LLM → 解析动作 → 执行 → 追加观察结果 |
| `environments/local.py` / `docker.py` | 动作执行环境：本地 subprocess 或 Docker 沙箱 |
| `models/` | 模型接入层，基于 litellm，任何 OpenAI 兼容接口都能接 |
| `config/*.yaml` | 提示词模板与运行配置（系统提示、格式约定、成本上限等） |
| `run/` | CLI 入口（`mini` 命令）与 SWE-bench 批量评测脚本 |

## 二、部署步骤（Linux/WSL2/macOS 通用）

```bash
# 1. 克隆并安装（需要 Python 3.10+）
git clone https://github.com/SWE-agent/mini-swe-agent.git
cd mini-swe-agent
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -e .

# （可选）跑测试套件验证安装：563 个测试应全部通过
.venv/bin/pip install -e '.[dev]'
PATH=$PWD/.venv/bin:$PATH .venv/bin/python -m pytest tests -q --ignore=tests/environments/extra

# 2. 配置模型（写入全局配置，跳过交互式向导）
mkdir -p ~/.config/mini-swe-agent
cat >> ~/.config/mini-swe-agent/.env <<'EOF'
MSWEA_CONFIGURED=true
MSWEA_MODEL_NAME=openai/gpt-5.5
EOF

# 3. 设置 API（以 OpenAI 兼容中转为例）
export OPENAI_API_KEY=你的key
export OPENAI_BASE_URL=https://你的中转地址/v1
export MSWEA_COST_TRACKING=ignore_errors   # 中转模型名不在 litellm 价格表时必须加

# 4. 运行（-y 免确认，-t 指定任务）
mkdir ~/workdir && cd ~/workdir
mini -c /path/to/custom.yaml -m openai/gpt-5.5 -y -l 0 -t "你的任务描述"
```

## 三、本目录文件说明

| 文件 | 说明 |
|---|---|
| `custom.yaml` | 针对中转模型调整过的运行配置（详见下文"踩坑记录"），可直接用 `-c` 加载 |
| `lru_cache.py` | **agent 自主生成的代码**：O(1) LRU 缓存（双向链表 + dict，未用 OrderedDict） |
| `test_lru_cache.py` | agent 自主编写的 pytest 测试（6 个用例全部通过，含零容量/负容量边界） |

演示任务：让 agent 实现 LRUCache 并自测。agent 共走 5 步（写实现 → 写测试 → 跑 pytest → 确认 6 passed → 提交完成），花费约 $0.22。

## 四、踩坑记录（中转 API 适配要点）

### 坑 1：中转不支持 function calling

mini-swe-agent v2 默认走 OpenAI tool-calling 接口，很多中转的模型不支持，报错 `No tool calls found in the response`。

**解法**：改用纯文本模式配置 `-c mini_textbased`（LLM 用 ```` ```mswea_bash_command ```` 代码块输出动作，用正则解析而不是 tool call）。

### 坑 2：模型不守"每次只出一个代码块"的格式约定

textbased 模式要求每次回复**恰好一个**代码块，弱一点的模型经常一次给多个块（`Expected exactly 1 action, found 4`），连续 3 次格式错误就触发 `RepeatedFormatError` 熔断退出。

**解法**（见 `custom.yaml`）：
1. 把动作解析正则改为**只取第一个代码块**，并兼容 ```` ```bash ```` 围栏：
   ```yaml
   model:
     model_class: litellm_textbased
     action_regex: "\\A.*?```(?:mswea_bash_command|bash)\\s*\\n(.*?)\\n```.*\\Z"
   ```
2. 放宽熔断阈值：`agent.max_consecutive_format_errors: 12`

这两个坑正好对应 agent 工程里的两个核心概念：**动作空间的解析协议**（tool-calling vs 文本正则）与**格式容错设计**（重试提示 + 熔断），面试聊 agent 架构时是很好的素材。

## 五、验证结果

- 部署：Python 3.10 + venv 安装成功，`mini --help` 正常
- 测试套件：563 passed / 36 skipped
- 端到端：真实编码任务自主完成，agent 生成的 6 个 pytest 用例全部通过

## 六、下一步建议（做成简历项目)

1. 精读 `agents/default.py`（188 行），画出 agent 循环时序图
2. 参考它用裸 API 实现自己的 mini code agent（~200 行）
3. 在 [SWE-bench Lite](https://github.com/SWE-bench/SWE-bench) 上跑分，得到可量化的简历指标
