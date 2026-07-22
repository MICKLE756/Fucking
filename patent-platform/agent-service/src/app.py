"""FastAPI Web 应用

按浏览器会话隔离对话状态：服务端以 ``sid`` Cookie 标识会话，
每个会话持有独立的 ``DialogProcess`` 实例，避免多用户 / 多标签页串话。
"""

import logging
import time
import uuid

import config
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from dialog_process import DialogProcess

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

# 会话存储：sid -> (最后活跃时间, DialogProcess)。
# 超过上限按最后活跃时间淘汰（LRU）；超过 TTL 未活跃的会话惰性回收。
# 注意：会话保存在进程内存中，仅适用于单 worker 部署；
# 多 worker / 多实例部署需改用外部存储（如 Redis）。
SESSION_COOKIE = "sid"
MAX_SESSIONS = 500
SESSION_TTL_SECONDS = 2 * 60 * 60
_sessions: dict[str, tuple[float, DialogProcess]] = {}

app = FastAPI()
app.mount("/static", StaticFiles(directory=str(config.BASE_DIR / "templates")))


def _evict_sessions(now: float) -> None:
    """回收过期会话；若仍超上限，按最后活跃时间淘汰最旧的会话。"""
    expired = [sid for sid, (ts, _) in _sessions.items()
               if now - ts > SESSION_TTL_SECONDS]
    for sid in expired:
        _sessions.pop(sid, None)
    while len(_sessions) >= MAX_SESSIONS:
        oldest = min(_sessions, key=lambda s: _sessions[s][0])
        _sessions.pop(oldest)


def _get_or_create_session(request: Request) -> tuple[str, DialogProcess]:
    """根据请求 Cookie 取回会话（刷新活跃时间），不存在则新建。"""
    now = time.time()
    sid = request.cookies.get(SESSION_COOKIE)
    entry = _sessions.get(sid) if sid else None
    if entry and now - entry[0] <= SESSION_TTL_SECONDS:
        dialog_process = entry[1]
    else:
        _evict_sessions(now)
        sid = uuid.uuid4().hex[:12]
        dialog_process = DialogProcess()
    _sessions[sid] = (now, dialog_process)
    return sid, dialog_process


@app.get("/")
async def homepage():
    return RedirectResponse("/static/index.html")


@app.post("/chat")
async def handle_message(request: Request):
    data = await request.json()
    user_message = data["message"]

    sid, dialog_process = _get_or_create_session(request)
    dialog_process.workflow.state["session_id"] = sid

    resp = dialog_process(user_message)
    resp["state"] = dialog_process.get_state()

    response = JSONResponse(resp)
    response.set_cookie(SESSION_COOKIE, sid, httponly=True, samesite="lax")
    return response


@app.post("/reset")
async def reset_session(request: Request):
    sid = request.cookies.get(SESSION_COOKIE)
    if sid:
        _sessions.pop(sid, None)
    return JSONResponse({"ok": True})


if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8089, reload=True)
