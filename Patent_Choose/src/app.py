import uuid
import config
import uvicorn
from fastapi import FastAPI, Request
from dialog_process import DialogProcess
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, RedirectResponse

# 初始化组件
dialog_process = DialogProcess()
session_id = str(uuid.uuid4())[:8]

# 创建 FastAPI 实例
app = FastAPI()

# 挂载静态文件
app.mount("/static", StaticFiles(directory=str(config.BASE_DIR / "templates")))


# 首页
@app.get("/")
async def homepage():
    return RedirectResponse("/static/index.html")


# 处理用户消息
@app.post("/chat")
async def handle_message(request: Request):
    data = await request.json()
    user_message = data["message"]
    dialog_process.workflow.state["session_id"] = session_id
    resp = dialog_process(user_message)
    # 附加状态信息供前端展示
    resp["state"] = dialog_process.get_state()
    return JSONResponse(resp)


# 重置会话
@app.post("/reset")
async def reset_session():
    global session_id
    dialog_process.reset()
    session_id = str(uuid.uuid4())[:8]
    return JSONResponse({"ok": True})


if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8089, reload=True)
