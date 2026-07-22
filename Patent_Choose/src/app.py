"""FastAPI Web 应用

两类调用方：
1. Agent 自带网页：``/chat``，以 ``sid`` Cookie 标识会话（兼容保留）。
2. 服务间调用（Spring Boot 编排层）：``/orchestrator/chat``，
   由调用方显式传入 ``session_id``（格式建议 ``用户ID:conversationId``），
   带 Pydantic 参数校验与统一错误响应，返回固定协议
   ``{session_id, message, request, phase, patents}``。

每个会话持有独立的 ``DialogProcess`` 实例，并配套一把互斥锁，
避免同一会话的并发请求同时修改状态。
"""

import asyncio
import uuid
from collections import OrderedDict

import config
import uvicorn
from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from dialog_process import DialogProcess

# 会话存储：session_id -> DialogProcess。超过上限按插入顺序淘汰最旧会话。
SESSION_COOKIE = "sid"
MAX_SESSIONS = 500
_sessions: "OrderedDict[str, DialogProcess]" = OrderedDict()
_session_locks: dict[str, asyncio.Lock] = {}

app = FastAPI()
app.mount("/static", StaticFiles(directory=str(config.BASE_DIR / "templates")))


class AgentChatRequest(BaseModel):
    """服务间调用请求模型（Spring Boot → agent-service）"""

    session_id: str = Field(min_length=1, max_length=128)
    user_id: str
    message: str = Field(min_length=1, max_length=5000)


class AgentChatResponse(BaseModel):
    """服务间调用固定响应协议（snake_case，由调用方转换 camelCase）"""

    session_id: str
    message: str = ""
    request: str = ""
    phase: str = ""
    patents: list[dict] = Field(default_factory=list)


def get_or_create_session(session_id: str) -> DialogProcess:
    """根据显式 session_id 取回会话，不存在则新建。"""
    if session_id not in _sessions:
        if len(_sessions) >= MAX_SESSIONS:
            evicted, _ = _sessions.popitem(last=False)
            _session_locks.pop(evicted, None)
        _sessions[session_id] = DialogProcess()
    return _sessions[session_id]


def _get_session_lock(session_id: str) -> asyncio.Lock:
    if session_id not in _session_locks:
        _session_locks[session_id] = asyncio.Lock()
    return _session_locks[session_id]


async def _run_dialog(session_id: str, message: str) -> dict:
    """在会话锁保护下执行一轮对话（阻塞逻辑放入线程池）。"""
    dialog_process = get_or_create_session(session_id)
    async with _get_session_lock(session_id):
        dialog_process.workflow.state["session_id"] = session_id
        return await run_in_threadpool(dialog_process, message)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    """统一参数错误响应"""
    return JSONResponse(
        status_code=422,
        content={
            "code": "INVALID_PARAM",
            "message": "请求参数校验失败",
            "detail": exc.errors(),
        },
    )


@app.get("/")
async def homepage():
    return RedirectResponse("/static/index.html")


@app.post("/orchestrator/chat")
async def orchestrator_chat(payload: AgentChatRequest) -> AgentChatResponse:
    """服务间调用接口：显式 session_id + 固定响应协议"""
    try:
        resp = await _run_dialog(payload.session_id, payload.message)
    except Exception:
        # Retrieval / LLM 故障降级：返回固定协议，不向调用方抛 5xx
        return AgentChatResponse(
            session_id=payload.session_id,
            message="检索服务暂时不可用，请稍后重试或调整检索条件。",
            phase="degraded",
        )

    return AgentChatResponse(
        session_id=payload.session_id,
        message=resp.get("message", ""),
        request=resp.get("request", ""),
        phase=resp.get("phase", ""),
        patents=resp.get("patents", []),
    )


@app.post("/chat")
async def handle_message(request: Request):
    """Agent 自带网页接口：Cookie 会话（兼容保留）"""
    data = await request.json()
    user_message = data["message"]

    sid = request.cookies.get(SESSION_COOKIE)
    if not sid or sid not in _sessions:
        sid = uuid.uuid4().hex[:12]

    resp = await _run_dialog(sid, user_message)
    resp["state"] = _sessions[sid].get_state()

    response = JSONResponse(resp)
    response.set_cookie(SESSION_COOKIE, sid, httponly=True, samesite="lax")
    return response


@app.post("/reset")
async def reset_session(request: Request):
    sid = request.cookies.get(SESSION_COOKIE)
    if sid:
        _sessions.pop(sid, None)
        _session_locks.pop(sid, None)
    return JSONResponse({"ok": True})


if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8089, reload=True)
