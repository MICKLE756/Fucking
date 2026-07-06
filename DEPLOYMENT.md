# 部署说明：在新机器上重新部署本仓库功能

本文档覆盖仓库中两个可运行应用的完整部署步骤：

1. **Patent_Choose** —— 专利检索智能助手（含查询历史记忆 + 相似问题检索 + 历史推荐）
2. **Hello** —— HelloAgents 框架及其可视化控制台（ReAct / Memory / Context / MCP / A2A / ANP / RL / Evaluation 面板）

---

## 〇、通用准备

| 要求 | 说明 |
|------|------|
| 操作系统 | Windows / macOS / Linux 均可 |
| Python | 3.10 及以上 |
| 网络 | 需要能访问你的 OpenAI 兼容 API 地址 |

```bash
# 1. 克隆仓库
git clone https://github.com/MICKLE756/Fucking.git
cd Fucking

# 2. 建议为每个应用建立独立虚拟环境（下文分别说明）
```

---

## 一、部署 Patent_Choose（专利检索助手）

### 1. 安装依赖

```bash
cd Patent_Choose
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows:
# .venv\Scripts\activate

pip install fastapi uvicorn openai python-dotenv httpx
```

### 2. 配置 `.env`

在 `Patent_Choose/` 目录下新建 `.env` 文件（该文件被 .gitignore 排除，不会随仓库带过来，**必须手动创建**）：

```env
# OpenAI API（意图兜底 + 实体补充 + 回复生成，必填）
OPENAI_API_KEY=你的API Key
OPENAI_BASE_URL=https://你的API地址/v1
MODEL_NAME=gpt-4o

# 远程微调模型（意图识别，可选；不配则自动跳过该层）
REMOTE_MODEL_URL=
REMOTE_MODEL_NAME=
REMOTE_MODEL_API_KEY=
REMOTE_MODEL_TIMEOUT=5
```

### 3. 数据文件

- `milvus.json`（专利数据）已包含在仓库中，无需额外准备。
- **记忆数据是运行时状态，不在 git 里**：`Patent_Choose/data/` 下的
  `slot_memory.json`（槽位提问记忆）和 `query_history.json`（查询历史记忆）
  会在首次运行时自动创建。
  - 若希望把旧机器上"学到的记忆"迁移过来，只需把旧机器的
    `Patent_Choose/data/` 整个目录拷贝到新机器同一位置即可。
  - 不拷贝则从零开始重新积累，功能不受影响（冷启动走默认话术 / 无历史推荐）。

### 4. 启动

```bash
cd src
python app.py
# 浏览器访问 http://127.0.0.1:8089
```

### 5. 验证（无需 API Key 的冒烟测试）

```bash
cd src
python test_slot_memory.py            # 槽位记忆层
python test_query_history_memory.py   # 查询历史记忆层（相似检索 + 历史推荐）
```

两个脚本都打印「所有断言通过 ✅」即部署成功。

### 6. 历史记忆功能说明

- 每次检索走通后，系统自动把「问题 + 技术领域 / 核心问题 / 约束 + 命中专利」沉淀到 `data/query_history.json`（上限 500 条，自动淘汰最旧）。
- 用户再问**相似问题**时（字符 bigram Jaccard 相似度 ≥ 0.15），回复中会提示历史相似查询，并附「📚 历史相关推荐」——从相似历史查询命中的专利中，按 *相似度 × 时间衰减（半衰期 30 天）* 打分推荐，且自动排除本次结果中已有的专利。

---

## 二、部署 Hello（HelloAgents 可视化控制台）

### 1. 安装依赖

```bash
cd Hello
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt   # 核心依赖（openai/pydantic/numpy/networkx/tiktoken 等）
pip install fastapi uvicorn       # web 控制台
```

> 可选依赖（qdrant / neo4j / torch / trl / datasets 等）不装也能跑：
> 对应面板会自动使用与真实 API 字段一致的演示数据，并在界面上以「demo」徽章标注。

### 2. 配置 `.env`（仅 live 模式的 ReAct 追踪需要）

在 `Hello/` 目录下创建 `.env`：

```env
LLM_API_KEY=你的API Key
LLM_BASE_URL=https://你的API地址/v1
LLM_MODEL_ID=gpt-4o
```

不配置也可启动，ReAct 面板用 demo 模式即可演示。

### 3. 启动

```bash
python -m hello_agents.web.server
# 默认 http://127.0.0.1:8080 ，可用环境变量修改：
# WEB_HOST=0.0.0.0 WEB_PORT=8090 python -m hello_agents.web.server
```

### 4. 页面与端点

| 地址 | 内容 |
|------|------|
| `/` | 可视化控制台（PR #40 版本，`dashboard_api.py` 驱动，`/api/*`） |
| `/ultimate` | 终极可视化控制台（PR #41 版本，`panels.py` 驱动，`/api/ultimate/*`）※ 需 PR #41 合并后可用 |
| `/api/health` | 健康检查 |
| `/api/run?q=问题&mode=demo|live` | ReAct 过程 SSE 流 |

### 5. 验证

```bash
curl http://127.0.0.1:8080/api/health      # {"status":"ok"}
curl http://127.0.0.1:8080/api/overview    # 概览 JSON
```

---

## 三、常见问题

| 问题 | 处理 |
|------|------|
| 启动报 `OPENAI_API_KEY must be set` | `.env` 未创建或未放在正确目录（Patent_Choose/ 或 Hello/ 根目录） |
| 端口被占用 | Patent_Choose 改 `src/app.py` 末行端口；Hello 用 `WEB_PORT` 环境变量 |
| 面板显示「demo」徽章 | 正常：对应可选依赖未安装，展示的是高保真演示数据 |
| 想在局域网其他设备访问 | 启动 host 改为 `0.0.0.0`，并放行对应防火墙端口 |
| 历史推荐为空 | 正常：`data/query_history.json` 尚无相似历史，多检索几次同领域问题即会出现 |
