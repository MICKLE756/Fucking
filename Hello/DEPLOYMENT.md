# HelloAgents 可视化控制台部署说明

在一台全新机器上部署本项目（含可视化控制台与知识库功能）的完整步骤。

## 1. 环境要求

- Python 3.10 及以上（推荐 3.10 / 3.11）
- 可访问外网的 pip（安装依赖）
- 操作系统不限（Windows / macOS / Linux 均可）

## 2. 获取代码

```bash
git clone https://github.com/MICKLE756/Fucking.git
cd Fucking/Hello
```

## 3. 安装依赖

建议先创建虚拟环境：

```bash
python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

安装框架本体与控制台所需依赖：

```bash
pip install -e .
pip install fastapi uvicorn python-dotenv pydantic fastmcp flask \
    scikit-learn tiktoken markitdown python-multipart
```

依赖说明：

| 依赖 | 用途 | 缺失时的行为 |
|---|---|---|
| fastapi / uvicorn / python-multipart | 控制台 Web 服务与文件上传 | 必装，否则无法启动 |
| scikit-learn | 知识库 TF-IDF 检索、Context 相关性 | 必装 |
| markitdown | 解析 PDF / Word / HTML 文档 | 缺失时知识库仅支持 md/txt/csv/json 等纯文本 |
| trl / transformers / datasets | RL 训练面板真实训练 | 缺失时 RL 曲线为模拟数据（面板会标注） |
| qdrant（服务） | 情景 / 语义记忆向量库 | 缺失时记忆系统降级为仅工作记忆（面板会标注） |

## 4. 配置 LLM（可选但推荐）

在 `Hello/` 目录下创建 `.env` 文件：

```env
LLM_API_KEY=你的API密钥
LLM_BASE_URL=https://api.你的服务商.com/v1
LLM_MODEL_ID=模型名称，如 kimi-k2.5 / gpt-4o-mini 等
```

不配置 LLM 时：Agent 面板只能用「演示模式」，知识库问答只返回检索片段（不生成回答），其余面板（Memory / Context / MCP / A2A / ANP / RL / Evaluation / 知识库检索）均可正常使用。

## 5. 启动控制台

```bash
cd Hello
python -m hello_agents.web.server
```

浏览器打开 `http://localhost:8080` 即可。

可用环境变量：

- `WEB_PORT`：Web 服务端口，默认 `8080`
- `KB_DIR`：知识库存储目录，默认 `~/.hello_agents/kb`（文档原件与索引均持久化在此，重启自动恢复）

示例（Linux / macOS）：

```bash
WEB_PORT=9000 KB_DIR=/data/hello_kb python -m hello_agents.web.server
```

Windows（PowerShell）：

```powershell
$env:WEB_PORT="9000"; $env:KB_DIR="D:\data\hello_kb"; python -m hello_agents.web.server
```

## 6. 生产环境建议

- 用 uvicorn 多 worker 方式常驻运行：

  ```bash
  uvicorn hello_agents.web.server:app --host 0.0.0.0 --port 8080 --workers 1
  ```

  注意：知识库索引保存在进程内存 + 磁盘 state.json，多 worker 时上传后其他 worker 需重启才能看到最新索引，建议单 worker 或前置只读副本。
- 配合 systemd / supervisor / NSSM（Windows）做进程守护与开机自启。
- 若对外暴露，请在前面加 Nginx 反向代理并限制访问来源；控制台自身没有登录鉴权。
- 定期备份 `KB_DIR` 目录（含上传原件 `files/` 与索引 `state.json`）。

## 7. 快速验证

1. 打开控制台 → 「📊 总览」应显示各模块状态（知识库应为「可用」）。
2. 「📚 知识库 KB」→ 上传一份 md / pdf / docx 文档 → 文档列表出现该文件与切块数。
3. 在检索框输入文档中的关键词 → 应返回带相似度分数的片段。
4. 「🧩 上下文 Context」勾选「检索知识库文档」→ 构建后最终 Prompt 中应包含文档内容。
