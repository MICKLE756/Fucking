"""FastAPI Web 应用

按浏览器会话隔离对话状态：服务端以 ``sid`` Cookie 标识会话，
每个会话持有独立的 ``DialogProcess`` 实例，避免多用户 / 多标签页串话。
"""

import uuid

import config
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from dialog_process import DialogProcess

# 会话存储：sid -> DialogProcess。超过上限按插入顺序淘汰最旧会话。
SESSION_COOKIE = "sid"
MAX_SESSIONS = 500
_sessions: dict[str, DialogProcess] = {}

app = FastAPI()
app.mount("/static", StaticFiles(directory=str(config.BASE_DIR / "templates")))


def _get_or_create_session(request: Request) -> tuple[str, DialogProcess]:
    """根据请求 Cookie 取回会话，不存在则新建。"""
    sid = request.cookies.get(SESSION_COOKIE)
    if not sid or sid not in _sessions:
        sid = uuid.uuid4().hex[:12]
        if len(_sessions) >= MAX_SESSIONS:
            _sessions.pop(next(iter(_sessions)))
        _sessions[sid] = DialogProcess()
    return sid, _sessions[sid]


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
