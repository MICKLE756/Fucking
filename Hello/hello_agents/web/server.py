"""Agent 流程可视化后端（FastAPI + SSE）。

端点：
- GET  /                : 单页可视化前端
- GET  /api/run         : SSE 流式返回一次 Agent 运行的结构化事件
                          参数：q=问题, mode=demo|live
- GET  /api/health      : 健康检查

启动：
    python -m hello_agents.web.server        # 默认 127.0.0.1:8080
环境变量：
    WEB_HOST (默认 127.0.0.1)  WEB_PORT (默认 8080)
"""

import os
import json
import queue
import threading
from pathlib import Path
from typing import Iterator

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse

from hello_agents.web.tracer import TracedReActAgent, TraceEvent, demo_trace

load_dotenv()

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="HelloAgents 流程可视化", version="1.0.0")


def _sse(event: TraceEvent) -> str:
    """把 TraceEvent 序列化为一条 SSE 消息。"""
    return f"data: {json.dumps(event.to_dict(), ensure_ascii=False)}\n\n"


def _demo_stream(question: str) -> Iterator[str]:
    import time
    for ev in demo_trace(question):
        yield _sse(ev)
        time.sleep(0.7)  # 放慢节奏，便于观察流程


def _build_live_agent() -> TracedReActAgent:
    """按 .env 构建带 calculate/search 工具的可追踪 ReAct Agent。"""
    from hello_agents.core.llm import HelloAgentsLLM
    from hello_agents.tools.registry import ToolRegistry
    from hello_agents.tools.builtin.calculator import calculate
    from hello_agents.tools.builtin.search_tool import search

    llm = HelloAgentsLLM()
    registry = ToolRegistry()
    registry.register_function(
        name="calculate",
        description="执行数学计算，支持基本运算与常见数学函数。例如：15*23+45、sqrt(16)。",
        func=calculate,
    )
    registry.register_function(
        name="search",
        description="网页搜索引擎。当需要时事或最新信息时使用。",
        func=search,
    )
    return TracedReActAgent(name="可视化ReAct", llm=llm, tool_registry=registry, max_steps=5)


def _live_stream(question: str) -> Iterator[str]:
    """在后台线程运行真实 Agent，把事件通过队列桥接到 SSE 生成器。"""
    q: "queue.Queue[TraceEvent]" = queue.Queue()
    _SENTINEL = TraceEvent(type="__done__")

    def emit(ev: TraceEvent) -> None:
        q.put(ev)

    def worker() -> None:
        try:
            agent = _build_live_agent()
            agent.run_traced(question, emit=emit)
        except Exception as e:
            q.put(TraceEvent(type="error", content=f"启动 Agent 失败: {e}"))
            q.put(TraceEvent(type="end"))
        finally:
            q.put(_SENTINEL)

    threading.Thread(target=worker, daemon=True).start()

    while True:
        ev = q.get()
        if ev.type == "__done__":
            break
        yield _sse(ev)


@app.get("/api/health")
def health() -> JSONResponse:
    return JSONResponse({"status": "ok"})


@app.get("/api/run")
def run(q: str = "请计算 15 * 23 + 45 等于多少？", mode: str = "demo") -> StreamingResponse:
    stream = _live_stream(q) if mode == "live" else _demo_stream(q)
    return StreamingResponse(
        stream,
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(html)


def main() -> None:
    import uvicorn
    host = os.getenv("WEB_HOST", "127.0.0.1")
    port = int(os.getenv("WEB_PORT", "8080"))
    print(f"🌐 Agent 流程可视化已启动： http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
