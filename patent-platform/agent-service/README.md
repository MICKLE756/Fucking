# 基于动态工作流与状态机的 Agent 交互式专利检索系统

## 一、项目简介

本项目是一个**专利技术检索智能助手**，参考 ai-medical 医疗问答系统的架构，采用**四层核心设计**：

| 层级 | 组件 | 作用 |
|------|------|------|
| **编排层** | 动态工作流引擎 | 将对话拆分为独立节点，由路由函数动态驱动，替代 if-else 硬编码 |
| **意图层** | 三级 Fallback + 状态机 | 规则 → 远程微调模型 → OpenAI 兜底，状态机做上下文消歧 |
| **抽取层** | 规则引擎 + LLM 补充 | 关键词/正则快速抽取实体，LLM 补充缺失槽位 |
| **执行层** | 工具路由 + 回复生成 | 调用检索/分析工具，LLM 生成结构化回复 |

用户只需用自然语言描述模糊的技术需求，Agent 会通过多轮对话自动引导用户完善检索条件，最终执行专利检索并返回结构化结果。

---

## 二、系统架构总览

```
┌──────────────────────────────────────────────────────────────────────┐
│                          FastAPI Web UI                              │
│                    (聊天界面 + 实时状态面板)                            │
└──────────────────────────────┬───────────────────────────────────────┘
                               │ POST /chat
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│                   WorkflowEngine (动态工作流引擎)                      │
│        entry → [node] → router(state) → [next_node] → ... → END     │
│                                                                      │
│  ┌──────────────┐     ┌─────────────────────────────────────────┐   │
│  │ confirm_check │────→│         intent_recognize 节点            │   │
│  └──────┬───────┘     │                                           │   │
│   affirm│             │  ┌────────────┐  ┌─────────────────────┐ │   │
│         ▼             │  │ ① 规则匹配  │  │ IntentStateMachine  │ │   │
│  ┌─────────────┐      │  └─────┬──────┘  │ (意图状态机)         │ │   │
│  │ tool_execute │      │  未命中│          │                     │ │   │
│  └──────┬──────┘      │        ▼          │ 状态×意图→修正意图  │ │   │
│         ▼             │  ┌─────────────┐  └──────────┬──────────┘ │   │
│  ┌─────────────┐      │  │ ② 远程微调   │             │            │   │
│  │ response_gen │      │  │  模型(Linux) │──成功──→ 修正 ──→ 返回  │   │
│  └─────────────┘      │  └─────┬───────┘             │            │   │
│                        │  失败/超时                     │            │   │
│                        │        ▼                      │            │   │
│                        │  ┌─────────────┐              │            │   │
│                        │  │ ③ OpenAI兜底 │──成功──→ 修正 ──→ 返回  │   │
│                        │  └─────────────┘                          │   │
│                        └───────────────────────────────────────────┘   │
│                               │                                       │
│               ┌───────────────┼───────────────┐                      │
│          chitchat│         feedback│          search│                   │
│               ▼               ▼               ▼                      │
│        ┌──────────┐   ┌──────────┐   ┌──────────────┐               │
│        │ chitchat  │   │ feedback │   │entity_extract │               │
│        │ _reply    │   │ _handle  │   └──────┬───────┘               │
│        └──────────┘   └────┬─────┘          │                       │
│                             │        ┌───────┴───────┐               │
│                             └───────→│completeness   │               │
│                                      │   _eval       │               │
│                                      └───────┬───────┘               │
│                                     ┌────────┴────────┐              │
│                                incomplete│          complete│           │
│                                     ▼                 ▼              │
│                               ┌──────────┐    ┌──────────┐          │
│                               │clarify_gen│    │confirm_gen│          │
│                               └──────────┘    └──────────┘          │
└──────────────────────────────────────────────────────────────────────┘
                               │
               ┌───────────────┼───────────────┐
               ▼               ▼               ▼
       ┌─────────────┐ ┌─────────────┐ ┌─────────────────────┐
       │ OpenAI API   │ │ 远程微调模型  │ │ 工具/数据库 (Mock)   │
       │ (GPT-4o)     │ │ (Linux GPU)  │ │                     │
       └─────────────┘ └─────────────┘ └─────────────────────┘
```

---

## 三、项目结构

```
./
├── .env                                     # API 配置（本地 + 远程模型）
├── templates/
│   └── index.html                           # 聊天界面 + 实时状态面板
└── src/
    ├── config.py                            # 集中配置
    ├── workflow_engine.py                   # 动态工作流引擎（通用，可复用）
    ├── workflow_nodes.py                    # 专利工作流（节点 + 路由 + 状态）
    ├── intent_recognize_rule_base.py        # 意图识别 - 规则层
    ├── intent_recognize_model_base.py       # 意图识别 - 远程微调模型层（待实现）
    ├── intent_state_machine.py              # 意图状态机 - 上下文消歧（待实现）
    ├── entity_extractor_rule_base.py        # 实体抽取 - 规则层
    ├── dialog_llm.py                        # OpenAI API 封装
    ├── dialog_process.py                    # 对话处理入口（工作流包装）
    └── app.py                               # FastAPI Web 应用
```

### 各文件职责

| 文件 | 职责 | 实现状态 |
|------|------|---------|
| `config.py` | 集中配置（路径、意图体系、实体映射、远程模型地址、模拟数据） | ✅ 已实现 |
| `workflow_engine.py` | 通用动态工作流引擎（节点注册、路由注册、图执行、Trace） | ✅ 已实现 |
| `workflow_nodes.py` | 专利工作流（11 个节点 + 5 个路由 + 状态管理） | ✅ 已实现 |
| `intent_recognize_rule_base.py` | 关键词 + 正则二级意图识别 | ✅ 已实现 |
| `intent_recognize_model_base.py` | 远程微调模型客户端（HTTP 调用、超时处理、格式解析） | ⏳ 已实现，细节需调整 |
| `intent_state_machine.py` | 意图状态机（状态定义、转移表、上下文消歧） | ⏳ 已实现，细节需调整 |
| `entity_extractor_rule_base.py` | 专利号正则、技术领域词典、约束条件提取 | ✅ 已实现 |
| `dialog_llm.py` | OpenAI API 封装（多轮对话 / JSON输出 / 文本输出） | ✅ 已实现 |
| `dialog_process.py` | 对话处理入口（工作流的轻薄包装） | ✅ 已实现 |
| `app.py` | FastAPI 应用（首页 / 聊天 / 重置） | ✅ 已实现 |

---

## 四、意图体系

### 4.1 二级意图分类

| 一级意图 | 二级意图 | 说明 |
|---------|---------|------|
| **search** | 专利检索、专利详情查询 | 主动寻找专利信息 |
| **analysis** | SWOT分析、技术对比分析、风险评估、价值评估 | 对已知专利做深度分析 |
| **operation** | 专利聚束组合、专利收藏、导出报告 | 操作类动作 |
| **feedback** | 结果不满意、修改条件、换方向 | 对当前结果的反馈修正 |
| **chitchat** | 闲聊、无关输入 | 非业务对话 |

### 4.2 意图 → 实体映射

| 二级意图 | 需要抽取的实体 |
|---------|--------------|
| 专利检索 | 技术领域、核心问题、约束条件 |
| 专利详情查询 | 专利号 |
| SWOT分析 | 专利号、技术领域 |
| 技术对比分析 | 专利号、技术领域 |
| 风险评估 | 专利号、技术领域 |
| 价值评估 | 专利号 |
| 专利聚束组合 | 技术领域、核心问题 |
| 修改条件 | 约束条件 |
| 换方向 | 技术领域 |

---

## 五、动态工作流引擎

### 5.1 核心概念

| 概念 | 说明 |
|------|------|
| **Node（节点）** | 可执行的处理步骤，签名 `fn(state) → state` |
| **Router（路由）** | 节点执行后的分支函数，签名 `fn(state) → next_node_name` |
| **State（状态）** | 共享字典，所有节点读写同一份 |
| **Trace（轨迹）** | 自动记录每轮执行路径 |

```python
engine = WorkflowEngine()
engine.add_node("intent", intent_fn)     # 注册节点
engine.add_router("intent", route_fn)    # 注册路由
engine.set_entry("intent")               # 设置入口
state = engine.run(state)                # 执行 → 自动沿图走到 END
print(state["_trace"])                   # 查看执行路径
```

### 5.2 节点清单（11 个）

| 节点 | 职责 | 终止节点 |
|------|------|---------|
| `confirm_check` | 检查是否在确认阶段 | |
| `intent_recognize` | 意图识别（规则→远程模型→OpenAI + 状态机修正） | |
| `entity_extract` | 实体抽取（规则→LLM） | |
| `completeness_eval` | 完整性评估（LLM） | |
| `clarify_gen` | 生成追问语句 | ✅ |
| `confirm_gen` | 生成确认语句 | ✅ |
| `tool_execute` | 执行检索/分析工具 | |
| `response_gen` | 生成结构化回复 | ✅ |
| `feedback_handle` | 处理用户反馈修正 | |
| `chitchat_reply` | 闲聊回复 | ✅ |
| `operation_hint` | 操作功能提示 | ✅ |

### 5.3 路由清单（5 个）

| 路由（after） | 条件 → 下一步 |
|-------------|--------------|
| `confirm_check` | affirm → `tool_execute` / else → `intent_recognize` |
| `intent_recognize` | chitchat → `chitchat_reply` / feedback → `feedback_handle` / analysis → `entity_extract`(direct) / search → `entity_extract`(eval) |
| `entity_extract` | direct_execute → `tool_execute` / eval → `completeness_eval` |
| `completeness_eval` | incomplete → `clarify_gen` / complete → `confirm_gen` |
| `feedback_handle` | 固定 → `entity_extract` |

### 5.4 典型执行 Trace

```
# 首次检索，信息不全
confirm_check → intent_recognize → entity_extract → completeness_eval → clarify_gen → END

# 用户确认后执行
confirm_check(affirm) → tool_execute → response_gen → END

# 反馈修正
confirm_check → intent_recognize → feedback_handle → entity_extract → completeness_eval → confirm_gen → END

# 闲聊
confirm_check → intent_recognize → chitchat_reply → END
```

### 5.5 扩展方式

新增处理步骤只需 3 行代码，不改引擎：

```python
# 1. 定义节点
def _node_spell_check(self, state):
    state["user_input"] = correct_spelling(state["user_input"])
    return state

# 2. 注册节点
engine.add_node("spell_check", self._node_spell_check)

# 3. 调整路由
engine.add_router("confirm_check", lambda s: "spell_check")
engine.add_router("spell_check", self._route_after_spell_check)
```

---

## 六、意图识别：三级 Fallback + 状态机

### 6.1 三级 Fallback 流水线

```
用户输入
  │
  ▼
┌──────────────────┐    命中
│ ① 规则匹配         │ ────────→ 状态机修正 → 返回意图
│ (关键词 + 正则)    │
└──────┬───────────┘
       │ 未命中
       ▼
┌───────────────────┐    成功
│ ② 远程微调模型      │ ────────→ 状态机修正 → 返回意图
│ (Linux GPU 服务器) │
└──────┬────────────┘
       │ 失败/超时
       ▼
┌──────────────────┐    成功
│ ③ OpenAI 兜底     │ ────────→ 状态机修正 → 返回意图 
└──────────────────┘
```

| 层级 | 延迟 | 精度 | 成本 | 角色 |
|------|------|------|------|------|
| ① 规则匹配 | < 1ms | 高（覆盖有限） | 0 | 快速命中高频场景 |
| ② 远程微调模型 | 50-500ms | 高（领域专精） | GPU 算力 | 主力意图分类 |
| ③ OpenAI 兜底 | 1-3s | 中高（通用） | API 费用 | 最后保底 |

三层统一输出格式：`{"level1": ["level2", ...]}`，每层输出后做格式校验。

### 6.2 意图状态机（IntentStateMachine）

#### 核心问题

同一句话在不同对话阶段含义不同，单句识别会误判：

```
轮次 1: "找耐高温涂层专利"  → Agent 进入需求收集
轮次 2: "还需要环保"         → ❌ 规则匹配到 feedback/修改条件
                               ✅ 实际是 search/补充条件（因为还在收集阶段）
```

#### 状态定义（5 个）

| 状态 | 含义 | 进入条件 |
|------|------|---------|
| `idle` | 空闲，等待用户发起任务 | 初始状态 / 会话重置 |
| `need_gathering` | 正在收集检索需求 | 用户发起 search |
| `confirming` | 等待用户确认条件 | 条件完整，生成确认语句 |
| `result_presented` | 已展示检索/分析结果 | 工具执行完毕 |
| `analyzing` | 正在进行分析任务 | 用户发起 analysis |

#### 转移表

转移规则：**(当前状态, 原始意图) → (新状态, 修正动作)**

| 当前状态 | 原始意图 | → 新状态 | 修正动作 | 说明 |
|---------|---------|----------|---------|------|
| `idle` | search | `need_gathering` | 保持 | 正常发起检索 |
| `idle` | analysis | `analyzing` | 保持 | 正常发起分析 |
| `idle` | feedback | `idle` | → `chitchat` | **无上下文，不可能反馈** |
| `idle` | chitchat | `idle` | 保持 | 正常闲聊 |
| `need_gathering` | search | `need_gathering` | 保持 | 继续补充检索条件 |
| `need_gathering` | feedback | `need_gathering` | → `search` | **实际是补充/修改条件** |
| `need_gathering` | chitchat | `need_gathering` | → `search`(推断) | **模糊输入当作补充信息** |
| `need_gathering` | analysis | `analyzing` | 保持 | 切换到分析 |
| `confirming` | search | `need_gathering` | 保持 | 放弃确认，发起新检索 |
| `confirming` | feedback | `need_gathering` | 保持 | 否认条件，回去修改 |
| `result_presented` | search | `need_gathering` | 保持 | 新一轮检索 |
| `result_presented` | feedback | `need_gathering` | 保持 | 对结果不满意 |
| `result_presented` | analysis | `analyzing` | 保持 | 对结果做深度分析 |
| `analyzing` | search | `need_gathering` | 保持 | 切换到检索 |
| `analyzing` | feedback | `need_gathering` | 保持 | 修正分析条件 |

#### 消歧效果

**"还需要环保"在不同状态下：**

| 当前状态 | 原始识别 | 状态机修正 |
|---------|---------|-----------|
| `idle` | feedback/修改条件 | → chitchat/闲聊 ✅ |
| `need_gathering` | feedback/修改条件 | → search/专利检索 ✅ |
| `result_presented` | feedback/修改条件 | 保持 feedback/修改条件 ✅ |

**"换一个"在不同状态下：**

| 当前状态 | 原始识别 | 状态机修正 |
|---------|---------|-----------|
| `idle` | feedback/换方向 | → chitchat/闲聊 ✅ |
| `need_gathering` | feedback/换方向 | 保持 feedback/换方向 ✅ |
| `result_presented` | feedback/换方向 | 保持 feedback/换方向 ✅ |

#### 状态同步

状态机与工作流 `phase` 保持同步：

```python
phase_to_state = {
    "idle":        "idle",
    "clarifying":  "need_gathering",
    "confirming":  "confirming",
    "responding":  "result_presented",
    "recognized":  保持当前,
    "executing":   保持当前,
}
```

### 6.3 远程微调模型（IntentRecognizeModelBase）

#### 部署架构

```
┌────────────────────────────────────────────────────┐
│              Linux GPU 服务器                         │
│                                                      │
│  推理框架 (任选其一):                                  │
│  - vLLM        → 高吞吐, 适合并发                     │
│  模型: Qwen2 / ChatGLM / LLaMA 微调                   │
│  接口: OpenAI 兼容 API (http://server:8000/v1)        │
└────────────────────┬─────────────────────────────────┘
                     │ HTTP POST /v1/chat/completions
                     ▼
┌────────────────────────────────────────────────────┐
│              本项目 (Windows)                         │
│  intent_recognize_model_base.py                      │
│  → 构造 prompt → 发送请求 → 解析 JSON → 返回意图     │
└────────────────────────────────────────────────────┘
```

#### 通信协议

```python
# 请求
POST http://<linux-server>:8000/v1/chat/completions
{
    "model": "patent-intent-v1",
    "messages": [
        {"role": "system", "content": "你是专利意图分类器。输出JSON..."},
        {"role": "user", "content": "我想找耐高温涂层专利"}
    ],
    "temperature": 0.1,
    "response_format": {"type": "json_object"}
}

# 响应
{
    "choices": [{
        "message": {
            "content": "{\"一级意图\": \"search\", \"二级意图\": [\"专利检索\"]}"
        }
    }]
}
```

#### 与通用大模型对比

| 对比项 | GPT-4o (通用) | 微调模型 (领域) |
|--------|-------------|----------------|
| 延迟 | 1-3s | 50-500ms |
| 成本 | 按 token 计费 | 固定 GPU 成本 |
| 精度 | 中高（需详细 prompt） | 高（已学习领域知识） |
| Prompt | 长（需描述完整意图体系） | 短（已内化分类规则） |
| 数据隐私 | 发送到第三方 | 留在本地服务器 |
| 可控性 | 依赖 API 提供商 | 完全自主可控 |

#### 微调数据

训练数据格式（JSON Lines）：

```jsonl
{"text": "我想找一种耐高温的不粘锅涂层材料相关的专利", "intent_level1": "search", "intent_level2": "专利检索"}
{"text": "帮我查一下CN202310001234.5这个专利的详情", "intent_level1": "search", "intent_level2": "专利详情查询"}
{"text": "对这个专利做一下SWOT分析", "intent_level1": "analysis", "intent_level2": "SWOT分析"}
{"text": "这些结果太宽泛了", "intent_level1": "feedback", "intent_level2": "结果不满意"}
{"text": "还需要环保，不含PFOA", "intent_level1": "search", "intent_level2": "专利检索"}
{"text": "你好", "intent_level1": "chitchat", "intent_level2": "闲聊"}
```

建议数据量：**每个二级意图 200-500 条**，共约 2000-5000 条。

#### 容错设计

```
远程模型调用
  ├── HTTP 200 + 合法 JSON     → 使用结果
  ├── 超时 (> 5s)               → 降级到 OpenAI
  ├── 连接拒绝 / 网络错误        → 降级到 OpenAI
  ├── HTTP 5xx                  → 降级到 OpenAI
  └── 返回格式异常               → 降级到 OpenAI
```

---

## 七、技术栈

| 层级 | 技术 | 作用 |
|------|------|------|
| **工作流引擎** | `WorkflowEngine`（自研） | 节点注册 + 路由注册 + 图执行 + Trace |
| **意图识别-规则** | 关键词 + 正则 | 快速匹配二级意图（< 1ms） |
| **意图识别-模型** | 远程微调 LLM（vLLM/Ollama/TGI） | 领域专精意图分类（50-500ms） |
| **意图消歧** | IntentStateMachine | 根据对话上下文修正意图 |
| **LLM 兜底** | GPT-4o（OpenAI API） | 意图兜底、实体补充、评估、生成 |
| **实体抽取** | 正则 + 关键词词典 | 专利号、技术领域、约束条件 |
| **状态管理** | dict 状态字典 | 跨轮次维护会话上下文 |
| **Web 框架** | FastAPI + Uvicorn | 异步 HTTP 服务 |
| **前端** | 原生 HTML/CSS/JS | 聊天可视化 + 实时状态面板 |
| **配置管理** | python-dotenv | 环境变量隔离 |

---

## 八、快速运行

### 环境要求

- Python 3.10+

### 安装依赖

```bash
pip install fastapi uvicorn openai python-dotenv httpx
```

### 配置

编辑 `.env` 文件：

```env
# OpenAI API（兜底 + 实体补充 + 生成）
OPENAI_API_KEY=你的API Key
OPENAI_BASE_URL=https://你的API地址/v1
MODEL_NAME=gpt-4o

# 远程微调模型（意图识别，可选）
REMOTE_MODEL_URL=http://192.168.1.100:8000/v1
REMOTE_MODEL_NAME=patent-intent-v1
REMOTE_MODEL_API_KEY=
REMOTE_MODEL_TIMEOUT=5
```

### 运行

```bash
cd src
python app.py
# 浏览器访问 http://127.0.0.1:8089
```

### API 接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/` | GET | 首页（重定向到聊天界面） |
| `/chat` | POST | 处理用户消息 `{message: str}` → `{message: str, request?: str, state: dict}` |
| `/reset` | POST | 重置会话 |

---

## 九、演示场景

| 轮次 | 角色 | 内容 | 状态机状态 | Trace |
|------|------|------|-----------|-------|
| 1 | 用户 | 我想找一种耐高温的不粘锅涂层材料相关的专利 | idle → need_gathering | confirm_check → intent_recognize → entity_extract → completeness_eval → clarify_gen |
| 1 | Agent | 您希望具备哪些具体的技术特性？例如：耐温范围、环保无毒等。 | | |
| 2 | 用户 | 主要用于餐饮厨具，需要耐受400度以上高温 | need_gathering | confirm_check → intent_recognize → entity_extract → completeness_eval → confirm_gen |
| 2 | Agent | 我理解您的需求是：寻找适用于餐饮厨具、满足耐高温（400℃+）的涂层材料专利。请确认是否继续。 | → confirming | |
| 3 | 用户 | 确认，请继续 | confirming → result_presented | confirm_check(affirm) → tool_execute → response_gen |
| 3 | Agent | 找到以下相关专利……（专利卡片 + 下一步建议） | | |
| 4 | 用户 | 这些结果太宽泛了，只想看已授权的 | result_presented → need_gathering | confirm_check → intent_recognize → feedback_handle → entity_extract → completeness_eval → confirm_gen |
| 4 | Agent | 已添加"已授权"筛选条件，请确认是否按新条件重新检索。 | → confirming | |

---

## 十、风险与应对

| 风险 | 影响 | 应对措施 |
|------|------|---------|
| 三层识别器输出格式不一致 | 下游节点解析出错 | 统一为 `{level1: [level2]}` 格式，每层做格式校验 |
| 远程模型延迟波动 | 用户体验下降 | 严格超时（5s），自动降级到 OpenAI |
| 微调数据不足 | 模型精度不够 | GPT-4o 生成种子数据，人工校验后扩充 |
| 状态转移表维护 | 新增意图需同步更新 | 状态数控制在 5-6 个，转移规则文档化 |
| Linux 服务器不可达 | 意图识别退化 | 三级 Fallback 保证 OpenAI 兜底始终可用 |
| 状态机与 phase 不同步 | 消歧失效 | 每轮结束显式同步 `phase → state` |

---

## 十一、后续扩展方向

- **拼写纠错节点**：参考 ai-medical，在意图识别前增加拼写纠错节点
- **实体抽取-模型节点**：微调 UIE 模型提取专利实体
- **实体对齐节点**：使用向量数据库（ChromaDB/Milvus）+ 同义词映射
- **知识图谱节点**：Neo4j 存储专利关系网络（引用、分类、发明人协作）
- **接入真实数据**：替换 Mock 工具为真实专利数据库 API
- **流式输出**：SSE/WebSocket 实现打字机效果
- **并行节点**：扩展引擎支持并行执行多个独立节点
- **多轮记忆**：引入长期记忆存储，跨会话记住用户偏好
