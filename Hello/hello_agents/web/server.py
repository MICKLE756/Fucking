"""终极 dashboard web 后端（FastAPI + SSE）。

保留原有 ReAct SSE trace，同时新增 9 个模块化 JSON 面板。
"""

from __future__ import annotations

import json
import os
import queue
import threading
from pathlib import Path
from typing import Any, Callable, Dict, Iterator

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from hello_agents.web.panels import (
    _a2a_builder,
    _anp_builder,
    _context_builder,
    _eval_builder,
    _memory_items,
    _mcp_builder,
    _rl_builder,
    build_overview,
    sanitize,
)
from hello_agents.web.tracer import TraceEvent, TracedReActAgent, demo_trace

load_dotenv()

STATIC_DIR = Path(__file__).parent / "static"
app = FastAPI(title="HelloAgents 终极可视化控制台", version="1.0.0")


def _sse(event: TraceEvent) -> str:
    return f"data: {json.dumps(event.to_dict(), ensure_ascii=False)}\n\n"


def _demo_stream(question: str) -> Iterator[str]:
    import time

    for ev in demo_trace(question):
        yield _sse(ev)
        time.sleep(0.7)


def _build_live_agent() -> TracedReActAgent:
    from hello_agents.core.llm import HelloAgentsLLM
    from hello_agents.tools.builtin.calculator import calculate
    from hello_agents.tools.builtin.search_tool import search
    from hello_agents.tools.registry import ToolRegistry

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
    q: "queue.Queue[TraceEvent]" = queue.Queue()
    sentinel = TraceEvent(type="__done__")

    def emit(ev: TraceEvent) -> None:
        q.put(ev)

    def worker() -> None:
        try:
            agent = _build_live_agent()
            agent.run_traced(question, emit=emit)
        except Exception as exc:
            q.put(TraceEvent(type="error", content=f"启动 Agent 失败: {exc}"))
            q.put(TraceEvent(type="end"))
        finally:
            q.put(sentinel)

    threading.Thread(target=worker, daemon=True).start()

    while True:
        ev = q.get()
        if ev.type == "__done__":
            break
        yield _sse(ev)


def _safe(builder: Callable[[], Dict[str, Any]], fallback: Dict[str, Any]) -> Dict[str, Any]:
    try:
        return sanitize(builder())
    except Exception as exc:
        data = dict(fallback)
        data["source"] = "demo"
        data["error"] = str(exc)
        return sanitize(data)


@app.get("/api/health")
def health() -> JSONResponse:
    return JSONResponse(
        {
            "status": "ok",
            "service": "hello_agents.web",
            "version": "ultimate-dashboard",
        }
    )


@app.get("/api/run")
def run(q: str = "请计算 15 * 23 + 45 等于多少？", mode: str = "demo") -> StreamingResponse:
    stream = _live_stream(q) if mode == "live" else _demo_stream(q)
    return StreamingResponse(
        stream,
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/overview")
def overview() -> JSONResponse:
    return JSONResponse(_safe(build_overview, {"source": "demo", "version": "unknown", "modules": []}))


@app.get("/api/memory")
def memory() -> JSONResponse:
    return JSONResponse(_safe(_memory_items, {"source": "demo"}))


@app.get("/api/context")
def context() -> JSONResponse:
    return JSONResponse(_safe(_context_builder, {"source": "demo"}))


@app.get("/api/mcp")
def mcp() -> JSONResponse:
    return JSONResponse(_safe(_mcp_builder, {"source": "demo"}))


@app.get("/api/a2a")
def a2a() -> JSONResponse:
    return JSONResponse(_safe(_a2a_builder, {"source": "demo"}))


@app.get("/api/anp")
def anp() -> JSONResponse:
    return JSONResponse(_safe(_anp_builder, {"source": "demo"}))


@app.get("/api/rl")
def rl() -> JSONResponse:
    return JSONResponse(_safe(_rl_builder, {"source": "demo"}))


@app.get("/api/eval")
def eval_panel() -> JSONResponse:
    return JSONResponse(_safe(_eval_builder, {"source": "demo"}))


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(html)


def main() -> None:
    import uvicorn

    host = os.getenv("WEB_HOST", "127.0.0.1")
    port = int(os.getenv("WEB_PORT", "8080"))
    print(f"🌐 HelloAgents 终极可视化控制台已启动： http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
