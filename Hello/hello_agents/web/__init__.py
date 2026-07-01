"""Agent 流程可视化 Web 模块（第7章+配套）

提供一个轻量 FastAPI 后端 + 单页前端，用 SSE 实时展示 ReAct Agent 的
「思考 → 行动 → 观察 → 最终答案」流程。

启动：
    python -m hello_agents.web.server
或：
    from hello_agents.web.server import app  # 交给 uvicorn 运行
"""

from .tracer import TracedReActAgent, TraceEvent, demo_trace

__all__ = ["TracedReActAgent", "TraceEvent", "demo_trace"]
