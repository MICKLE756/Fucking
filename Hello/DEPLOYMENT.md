# HelloAgents 完整部署指南

在一台全新机器上从零部署本项目的**全部功能**（框架本体 + 可视化控制台 + 知识库 + 情景/语义记忆 + 协议 + RL + 评测），照本文档做完即可跑通所有面板与所有示例代码。

## 目录

- [一、总览：功能 → 所需组件对照表](#一总览功能--所需组件对照表)
- [二、基础环境（所有功能必需）](#二基础环境所有功能必需)
- [三、Docker Desktop + WSL2（Windows 下部署外部服务）](#三docker-desktop--wsl2windows-下部署外部服务)
- [四、外部服务部署：Qdrant 与 Neo4j](#四外部服务部署qdrant-与-neo4j)
- [五、Python 依赖：按子系统完整安装](#五python-依赖按子系统完整安装)
- [六、.env 完整配置清单](#六env-完整配置清单)
- [七、启动可视化控制台](#七启动可视化控制台)
- [八、BFCL 评测的独立环境（numpy 冲突）](#八bfcl-评测的独立环境numpy-冲突)
- [九、运行各章节示例代码](#九运行各章节示例代码)
- [十、逐面板验证清单（确认所有功能可用）](#十逐面板验证清单确认所有功能可用)
- [十一、缺失组件时的降级行为](#十一缺失组件时的降级行为)
- [十二、生产环境建议](#十二生产环境建议)
- [十三、常见问题排查](#十三常见问题排查)

---

## 一、总览：功能 → 所需组件对照表

| 功能 | Python 依赖 | 外部服务 | 缺失时 |
|---|---|---|---|
| 控制台 Web 服务 | fastapi、uvicorn、python-multipart、pydantic、python-dotenv | — | 必装，否则无法启动 |
| Agent 问答 / ReAct | openai（框架自带） | LLM API（.env 配置） | 无 LLM 时仅演示模式 |
| 工作记忆 | 框架自带 | — | 始终可用 |
| **情景记忆** | qdrant-client | **Qdrant**（Docker） | 降级为仅工作记忆 |
| **语义记忆（含知识图谱）** | qdrant-client、neo4j、spacy | **Qdrant + Neo4j**（Docker） | 降级为仅工作记忆 |
| 记忆/知识库向量化 | sentence-transformers 或 DashScope API 或 TF-IDF | — | 自动回退 TF-IDF |
| 知识库 KB（上传 md/pdf/docx） | scikit-learn、markitdown[pdf,docx] | — | 缺 markitdown 时仅纯文本格式 |
| Context 上下文工程（GSSC） | tiktoken、scikit-learn | — | 必装 |
| MCP 协议 | fastmcp | — | MCP 面板不可用 |
| A2A 协议 | flask、requests | —（本机自起子服务） | A2A 面板不可用 |
| ANP 协议 | 框架自带 | — | 始终可用 |
| 网页搜索工具 | tavily-python / google-search-results | Tavily / SerpApi Key | 无 Key 时搜索工具不可用 |
| **RL 训练（真实训练）** | trl、transformers、torch、datasets、accelerate、peft | GPU（可选，CPU 也能跑小模型） | 曲线为模拟数据 |
| GAIA 评测 | datasets、huggingface_hub、pandas、tqdm | HuggingFace 网络 | 评测面板部分不可用 |
| **BFCL 评测** | bfcl-eval + numpy==1.26.4（**独立 venv**） | — | BFCL 真实评测不可用 |

---

## 二、基础环境（所有功能必需）

- **操作系统**：Windows 10/11（需 WSL2）、macOS、Linux 均可
- **Python**：3.10 或 3.11（推荐 3.11；RL/torch 对 3.12 支持不稳）
- **Git**、可访问外网的 **pip**
- **Docker Desktop**（用于 Qdrant / Neo4j；Linux 直接装 docker engine 即可）
- 磁盘：基础 2GB；装 RL（torch）另需 5GB+；本地 embedding 模型另需 1-2GB

```bash
git clone https://github.com/MICKLE756/Fucking.git
cd Fucking/Hello

python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

---

## 三、Docker Desktop + WSL2（Windows 下部署外部服务）

Linux / macOS 用户可跳到第四节（直接 `docker run` 即可）。

### 1. 启用 WSL2

以管理员身份打开 PowerShell：

```powershell
wsl --install                 # 首次安装（含 Ubuntu 发行版），完成后重启
wsl --set-default-version 2   # 确保默认 WSL2
wsl -l -v                     # 确认 VERSION 列为 2
```

若提示需要开启虚拟化：BIOS 中开启 Intel VT-x / AMD-V，并在「启用或关闭 Windows 功能」中勾选**适用于 Linux 的 Windows 子系统**与**虚拟机平台**。

### 2. 安装 Docker Desktop

1. 从 https://www.docker.com/products/docker-desktop/ 下载安装，安装时勾选 **Use WSL 2 based engine**。
2. 启动 Docker Desktop → `Settings → General` 确认 *Use the WSL 2 based engine* 已勾选。
3. `Settings → Resources → WSL Integration` 打开你的 Ubuntu 发行版集成。
4. 验证：

```powershell
docker version        # Client 和 Server 都应有输出
docker run --rm hello-world
```

> Docker Desktop（WSL2 后端）里跑的容器端口会自动映射到 Windows 的 `localhost`，因此后文一律用 `localhost:6333`、`localhost:7687` 访问，无需查 WSL 的 IP。

---

## 四、外部服务部署：Qdrant 与 Neo4j

### 1. Qdrant（向量数据库 —— 情景记忆 / 语义记忆必需）

```bash
docker run -d --name qdrant \
  -p 6333:6333 -p 6334:6334 \
  -v qdrant_storage:/qdrant/storage \
  --restart unless-stopped \
  qdrant/qdrant
```

验证：浏览器打开 `http://localhost:6333/dashboard`，或：

```bash
curl http://localhost:6333/collections
# 期望返回 {"result":{"collections":[]},...}
```

### 2. Neo4j（图数据库 —— 语义记忆的知识图谱必需）

```bash
docker run -d --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/hello-agents-password \
  -v neo4j_data:/data \
  --restart unless-stopped \
  neo4j:5
```

> 密码 `hello-agents-password` 是框架代码的默认值（`hello_agents/core/database_config.py`）。若改用其他密码，必须在 `.env` 中设置 `NEO4J_PASSWORD`。

验证：浏览器打开 `http://localhost:7474`，用 `neo4j / hello-agents-password` 登录成功即可。

### 3. 一键 docker-compose（推荐）

在 `Hello/` 下新建 `docker-compose.yml`：

```yaml
services:
  qdrant:
    image: qdrant/qdrant
    ports: ["6333:6333", "6334:6334"]
    volumes: [qdrant_storage:/qdrant/storage]
    restart: unless-stopped
  neo4j:
    image: neo4j:5
    ports: ["7474:7474", "7687:7687"]
    environment:
      NEO4J_AUTH: neo4j/hello-agents-password
    volumes: [neo4j_data:/data]
    restart: unless-stopped
volumes:
  qdrant_storage:
  neo4j_data:
```

```bash
docker compose up -d
docker compose ps    # 两个服务都应为 running
```

---

## 五、Python 依赖：按子系统完整安装

在 `Hello/` 目录、虚拟环境已激活的前提下：

### 1. 框架本体 + 控制台（必装）

```bash
pip install -e .
pip install fastapi uvicorn python-multipart python-dotenv pydantic \
    scikit-learn tiktoken
```

### 2. 知识库文档解析（上传 pdf / docx 必装）

```bash
pip install "markitdown[pdf,docx]"
```

> 只装裸 `markitdown` 不带 extras 时无法解析 PDF/Word，知识库只能上传 md/txt/csv/json/html 等纯文本。

### 3. 记忆系统（情景 / 语义记忆）

```bash
pip install "qdrant-client>=1.9.0,<1.16.0" "neo4j>=5.0.0" spacy
```

> 注意 qdrant-client 必须 `<1.16.0`（1.16.0 移除了 search 接口）。

### 4. Embedding 向量化（三选一，自动按 EMBED_MODEL_TYPE 生效）

| 方式 | 安装 | 说明 |
|---|---|---|
| DashScope API（默认） | 无需额外安装 | `.env` 配 `EMBED_API_KEY` |
| 本地模型 | `pip install sentence-transformers torch` | `.env` 配 `EMBED_MODEL_TYPE=local` 和模型路径 |
| TF-IDF 兜底 | 已随 scikit-learn 装好 | `.env` 配 `EMBED_MODEL_TYPE=tfidf`，无需网络 |

### 5. 协议（MCP / A2A）

```bash
pip install "fastmcp>=2.0.0,<3.0.0" flask
```

### 6. 搜索工具（可选）

```bash
pip install tavily-python google-search-results
# .env 配 TAVILY_API_KEY 或 SERPAPI_API_KEY
```

### 7. RL 训练（真实训练曲线）

```bash
pip install "trl>=0.24.0" transformers "torch>=2.0.0" datasets accelerate peft
# 有 NVIDIA GPU 时按 https://pytorch.org 选择对应 CUDA 版本的 torch
```

### 8. 评测（GAIA / 数据生成 / LLM-judge）

```bash
pip install datasets "huggingface_hub<1.0.0" evaluate pandas tqdm matplotlib
```

BFCL 评测因 numpy 版本冲突需要**独立虚拟环境**，见第八节。

### 9. 一次性全装（除 BFCL 外）

```bash
pip install -e .
pip install fastapi uvicorn python-multipart python-dotenv pydantic \
    scikit-learn tiktoken "markitdown[pdf,docx]" \
    "qdrant-client>=1.9.0,<1.16.0" "neo4j>=5.0.0" spacy \
    "fastmcp>=2.0.0,<3.0.0" flask \
    "trl>=0.24.0" transformers "torch>=2.0.0" datasets accelerate peft \
    "huggingface_hub<1.0.0" evaluate pandas tqdm matplotlib \
    tavily-python google-search-results sentence-transformers
```

---

## 六、.env 完整配置清单

在 `Hello/` 目录下创建 `.env`：

```env
# ---------- LLM（Agent 问答 / 知识库问答 / LLM-judge） ----------
LLM_API_KEY=你的API密钥
LLM_BASE_URL=https://api.你的服务商.com/v1
LLM_MODEL_ID=kimi-k2.5            # 或 gpt-4o-mini 等 OpenAI 兼容模型

# ---------- Embedding（记忆/RAG 向量化，三选一） ----------
EMBED_MODEL_TYPE=tfidf            # dashscope | local | tfidf
# EMBED_MODEL_TYPE=dashscope
# EMBED_API_KEY=你的DashScope密钥
# EMBED_MODEL_TYPE=local
# EMBED_MODEL_NAME=C:/Models/bge-base-zh-v1.5   # 本地模型目录

# ---------- Qdrant（情景/语义记忆向量库） ----------
QDRANT_URL=http://localhost:6333
# QDRANT_API_KEY=                 # 本地 Docker 无需；Qdrant Cloud 才需要
# QDRANT_COLLECTION=hello_agents_vectors
# QDRANT_VECTOR_SIZE=384          # 需与 Embedding 维度一致

# ---------- Neo4j（语义记忆知识图谱） ----------
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=hello-agents-password
# NEO4J_DATABASE=neo4j

# ---------- 搜索工具（可选） ----------
# TAVILY_API_KEY=
# SERPAPI_API_KEY=

# ---------- 控制台 ----------
# WEB_HOST=127.0.0.1
# WEB_PORT=8080
# KB_DIR=~/.hello_agents/kb       # 知识库持久化目录
```

不配置 LLM 时：Agent 面板只能用「演示模式」，知识库问答只返回检索片段；其余面板不受影响。

---

## 七、启动可视化控制台

确认 Qdrant / Neo4j 容器已运行后：

```bash
cd Hello
python -m hello_agents.web.server
```

浏览器打开 `http://localhost:8080`。

改端口 / 知识库目录：

```bash
# Linux / macOS
WEB_PORT=9000 KB_DIR=/data/hello_kb python -m hello_agents.web.server
```

```powershell
# Windows PowerShell
$env:WEB_PORT="9000"; $env:KB_DIR="D:\data\hello_kb"; python -m hello_agents.web.server
```

---

## 八、BFCL 评测的独立环境（numpy 冲突）

`bfcl-eval` 强制要求 `numpy==1.26.4`，与框架核心依赖 `numpy>=2.0` 完全不兼容，**必须单独建 venv**：

```bash
cd Hello
python -m venv .venv-bfcl
# Linux / macOS
source .venv-bfcl/bin/activate
# Windows: .venv-bfcl\Scripts\Activate.ps1

pip install openai requests python-dotenv pydantic beautifulsoup4 networkx
pip install "numpy==1.26.4" bfcl-eval datasets pandas tqdm huggingface_hub evaluate
```

跑 BFCL 时激活 `.venv-bfcl`；跑其他一切功能时用主 `.venv`。

---

## 九、运行各章节示例代码

主 venv 中，在 `Hello/` 目录执行：

```bash
python examples/chapter07_basic_setup.py        # 基础 Agent
python examples/chapter07_react_tool_demo.py    # ReAct + 工具
python examples/chapter07_search_tool_demo.py   # 搜索工具（需 TAVILY/SERPAPI Key）
python examples/chapter08_memory_rag.py         # 记忆 + RAG（需 Qdrant/Neo4j）
python examples/chapter09_context_engineering.py# GSSC 上下文工程
python examples/chapter10_protocols.py          # MCP / A2A / ANP
python examples/chapter11_RL.py                 # RL（需 trl/torch）
```

---

## 十、逐面板验证清单（确认所有功能可用）

按顺序验证，全部通过即部署成功：

1. **📊 总览**：所有模块显示「可用」；若记忆显示降级，检查 Qdrant/Neo4j 容器与 `.env`。
2. **🤖 Agent**：配置了 LLM 后提问应得到真实回答（非演示模式）。
3. **🧠 记忆 Memory**：添加一条 `memory_type=episodic` 的记忆再检索应能命中——证明 Qdrant 打通；添加 `semantic` 记忆——证明 Neo4j 打通（可在 `http://localhost:7474` 里查到节点）。
4. **📚 知识库 KB**：上传一份 **pdf 或 docx** → 列表出现文件与切块数（证明 markitdown[pdf,docx] 生效）→ 检索关键词返回带相似度的片段 → 「知识库问答」返回带引用的 LLM 回答。
5. **🧩 上下文 Context**：勾选「检索知识库文档」构建后，最终 Prompt 包含文档内容。
6. **🔌 MCP**：执行内置计算工具应返回结果。
7. **🤝 A2A**：启动 Agent 网络后发送任务，应有真实 HTTP 往返结果。
8. **🌐 ANP**：路由/广播消息有响应。
9. **🎯 RL**：装了 trl/torch 后训练曲线应标注为真实训练（未装则标注「模拟」）。
10. **📏 Evaluation**：GAIA 指标可加载；BFCL 在 `.venv-bfcl` 中运行。

---

## 十一、缺失组件时的降级行为

| 缺失项 | 影响 | 面板表现 |
|---|---|---|
| Qdrant 未启动 / qdrant-client 未装 | 情景、语义记忆不可用 | 记忆面板标注「已降级为工作记忆」 |
| Neo4j 未启动 / neo4j 未装 | 语义记忆的知识图谱不可用 | 语义记忆初始化失败 → 同上降级 |
| markitdown[pdf,docx] 未装 | 知识库无法解析 PDF/Word | 上传 pdf/docx 时明确报错（不会把乱码当文本索引） |
| trl / torch 未装 | RL 无法真实训练 | RL 曲线标注为模拟数据 |
| fastmcp 未装 | MCP 面板不可用 | 接口返回降级信息 |
| LLM 未配置 | 无真实生成 | Agent 演示模式；KB 问答仅返回片段 |
| Embedding API 不可用 | 向量化回退 | 自动回退 TF-IDF（检索质量下降但可用） |
| TAVILY/SERPAPI Key 未配 | 搜索工具不可用 | 调用时报缺 Key |

---

## 十二、生产环境建议

- 常驻运行：

  ```bash
  uvicorn hello_agents.web.server:app --host 0.0.0.0 --port 8080 --workers 1
  ```

  知识库索引在进程内存 + 磁盘 `state.json`，多 worker 之间不共享上传后的新索引，**建议单 worker**。
- 进程守护：systemd / supervisor（Linux）、NSSM 或计划任务（Windows）；Docker 服务已用 `--restart unless-stopped` 自恢复。
- 对外暴露前置 Nginx 反向代理并限制来源——控制台自身**没有登录鉴权**；Qdrant/Neo4j 端口不要暴露公网。
- 定期备份：`KB_DIR`（上传原件 `files/` + 索引 `state.json`）、docker volume `qdrant_storage` 与 `neo4j_data`。

## 十三、常见问题排查

| 现象 | 原因与处理 |
|---|---|
| `docker` 命令找不到（Windows） | Docker Desktop 未启动，或未开启 WSL Integration |
| 控制台记忆面板一直「降级为工作记忆」 | Qdrant/Neo4j 容器没起、`.env` 的 URL/密码不对，或 qdrant-client / neo4j 包未装；`docker ps` + `curl localhost:6333/collections` 排查 |
| Neo4j 登录失败 | 容器首次启动时 `NEO4J_AUTH` 已固化密码；改密码需删容器与 `neo4j_data` 卷重建，或在 7474 网页里改 |
| 上传 PDF 报「解析失败」 | 未安装 `markitdown[pdf,docx]` extras；重新 `pip install "markitdown[pdf,docx]"` |
| Qdrant 向量维度错误 | `QDRANT_VECTOR_SIZE` 与 Embedding 模型维度不一致；删除 collection 后按实际维度重建 |
| BFCL 安装冲突 | 必须用独立 `.venv-bfcl`（见第八节），不要装进主环境 |
| torch 安装极慢/失败 | 用 https://pytorch.org 官方命令按平台/CUDA 选择安装源 |
