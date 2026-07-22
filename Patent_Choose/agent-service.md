## agent-service

### 现有问题

1. 会话机制不适合 Spring Boot 调用

[app.py](/Users/jackson/Desktop/patent-platform/agent-service/src/app.py:48) 通过浏览器 Cookie 中的 `sid` 查找会话：

```python
sid = request.cookies.get(SESSION_COOKIE)
```

Spring Boot 如果没有保存和转发这个 Cookie，每次请求都会创建新会话，导致：

```text
第一次：用户提出需求
第二次：用户补充条件
Agent：无法关联第一次需求
```

2. `/chat` 接口主要面向 Agent 自带网页

[app.py](/Users/jackson/Desktop/patent-platform/agent-service/src/app.py:68) 的接口直接读取原始 JSON：

```python
data = await request.json()
user_message = data["message"]
```

存在以下问题：

- 没有 Pydantic 参数校验。
- 不接受显式 `session_id`。
- 不接受 `user_id`。
- 缺少统一错误响应。
- 不能稳定用于服务间调用。

3. 没有返回结构化专利数组

工作流检索完成后，专利数据存放在：

```python
state["_tool_result"]["patents"]
```

但 [workflow_nodes.py](/Users/jackson/Desktop/patent-platform/agent-service/src/workflow_nodes.py:592) 最终只返回：

```text
message
request
```

因此 Spring Boot 无法获得稳定的专利卡片数据，只能拿到自然语言回复。

4. 内存会话存在并发和部署限制

当前 `_sessions` 是进程内字典：

```python
_sessions: dict[str, tuple[float, DialogProcess]] = {}
```

存在以下限制：

- 只能使用单 worker。
- 服务重启后会话丢失。
- 同一个会话同时请求时可能修改同一份状态。
- 多实例部署时不同实例无法共享会话。

---

## 需要修改

### 1. 新增服务间调用接口

建议保留原 `/chat` 给自带网页使用，新增：

```http
POST /orchestrator/chat
```

请求模型：

```json
{
  "session_id": "12:conv_abc123",
  "user_id": "12",
  "message": "优先考虑2021年以后公开的方案"
}
```

使用 Pydantic 定义：

```python
class AgentChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=128)
    user_id: str
    message: str = Field(min_length=1, max_length=5000)
```

### 2. 使用显式 session_id

会话获取方法应改成根据请求参数查找：

```python
def get_or_create_session(session_id: str) -> DialogProcess:
    ...
```

Spring Boot 负责生成：

```text
用户ID:conversationId
```

Cookie 只用于兼容 Agent 自带网页，不再作为服务间调用的主要会话方式。

### 3. 返回结构化专利数据

修改 [workflow_nodes.py](/Users/jackson/Desktop/patent-platform/agent-service/src/workflow_nodes.py:582)，从 `_tool_result` 中提取专利：

```python
tool_result = self.state.get("_tool_result") or {}
patents = tool_result.get("patents", [])

response = {
    "message": self.state.get("_response", ""),
    "request": self.state.get("_request", ""),
    "phase": self.state.get("phase", ""),
    "patents": patents,
}
```

专利结果必须独立返回，不能让 Spring Boot 从自然语言回复中解析。

### 4. 固定响应协议

建议响应：

```json
{
  "session_id": "12:conv_abc123",
  "message": "根据您的条件，找到以下相关专利",
  "request": "",
  "phase": "result",
  "patents": [
    {
      "patent_id": "CN123456",
      "title": "一种耐高温环保涂层",
      "inventor": "张三",
      "tech_field": "涂层材料",
      "publish_date": "2024-01-01",
      "final_score": 0.87
    }
  ]
}
```

Python 内部继续使用 `snake_case`，由 Spring Boot 转换成前端使用的 `camelCase`。

### 5. 补充依赖和测试

新增依赖清单，并至少覆盖：

```text
同一 session_id 可以连续追问
不同 session_id 不会串话
检索后响应包含 patents
Retrieval 故障时触发降级
空 message 返回参数错误
并发请求不会破坏会话状态
```

最优先修改三项是：**显式 `session_id`、结构化返回 `patents`、新增 `/orchestrator/chat` 服务间接口**。这三项完成后，Spring Boot 才能稳定接入 Agent。