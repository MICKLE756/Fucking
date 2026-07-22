"""/orchestrator/chat 服务间接口测试（无需任何 API / 模型）

用假 DialogProcess 替换真实工作流，专注验证接口层协议：
    - 同一 session_id 可以连续追问（会话状态延续）
    - 不同 session_id 不会串话（会话相互隔离）
    - 检索后响应包含结构化 patents 数组
    - Retrieval 故障时触发降级（返回固定协议而非 5xx）
    - 空 message / 缺参 返回统一参数错误
    - 并发请求不会破坏会话状态（同会话串行执行）

运行: pytest src/test_orchestrator_api.py
"""

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

import app as app_module
from app import app

FAKE_PATENTS = [
    {
        "patent_id": "CN123456",
        "title": "一种耐高温环保涂层",
        "inventor": "张三",
        "tech_field": "涂层材料",
        "publish_date": "2024-01-01",
        "final_score": 0.87,
    }
]


class FakeWorkflow:
    def __init__(self):
        self.state = {"session_id": ""}


class FakeDialogProcess:
    """按会话实例记录消息，模拟多轮对话与检索结果。"""

    def __init__(self):
        self.workflow = FakeWorkflow()
        self.messages = []
        self._lock_check = threading.Lock()

    def __call__(self, text: str) -> dict:
        # 并发保护验证：同一会话不允许两个线程同时进入
        if not self._lock_check.acquire(blocking=False):
            raise RuntimeError("同一会话被并发执行")
        try:
            self.messages.append(text)
            if "故障" in text:
                raise ConnectionError("retrieval backend down")
            if "检索" in text:
                return {
                    "message": f"找到 {len(FAKE_PATENTS)} 条专利",
                    "request": "",
                    "phase": "responding",
                    "patents": FAKE_PATENTS,
                }
            return {
                "message": f"已收到第 {len(self.messages)} 条消息",
                "request": "",
                "phase": "clarifying",
                "patents": [],
            }
        finally:
            self._lock_check.release()

    def get_state(self) -> dict:
        return {"phase": "clarifying"}


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setattr(app_module, "DialogProcess", FakeDialogProcess)
    app_module._sessions.clear()
    app_module._session_locks.clear()
    with TestClient(app) as c:
        yield c


def _chat(client, session_id, message, user_id="12"):
    return client.post(
        "/orchestrator/chat",
        json={"session_id": session_id, "user_id": user_id, "message": message},
    )


def test_same_session_continues(client):
    r1 = _chat(client, "12:conv_a", "找耐高温涂层专利")
    r2 = _chat(client, "12:conv_a", "优先考虑2021年以后公开的方案")
    assert r1.status_code == r2.status_code == 200
    assert r2.json()["message"] == "已收到第 2 条消息"
    assert r2.json()["session_id"] == "12:conv_a"


def test_different_sessions_isolated(client):
    _chat(client, "12:conv_a", "第一条")
    r = _chat(client, "34:conv_b", "另一个会话的第一条")
    assert r.json()["message"] == "已收到第 1 条消息"
    assert app_module._sessions["12:conv_a"] is not app_module._sessions["34:conv_b"]


def test_response_contains_patents(client):
    r = _chat(client, "12:conv_a", "确认，开始检索")
    body = r.json()
    assert body["phase"] == "responding"
    assert body["patents"] == FAKE_PATENTS
    assert set(body) == {"session_id", "message", "request", "phase", "patents"}


def test_retrieval_failure_degrades(client):
    r = _chat(client, "12:conv_a", "触发故障")
    assert r.status_code == 200
    body = r.json()
    assert body["phase"] == "degraded"
    assert body["patents"] == []
    assert body["message"]


def test_empty_message_rejected(client):
    r = _chat(client, "12:conv_a", "")
    assert r.status_code == 422
    assert r.json()["code"] == "INVALID_PARAM"


def test_missing_fields_rejected(client):
    r = client.post("/orchestrator/chat", json={"message": "hi"})
    assert r.status_code == 422
    assert r.json()["code"] == "INVALID_PARAM"


def test_concurrent_requests_do_not_corrupt_session(client):
    def send(i):
        return _chat(client, "12:conv_a", f"消息{i}")

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(send, range(20)))

    assert all(r.status_code == 200 for r in results)
    session = app_module._sessions["12:conv_a"]
    assert len(session.messages) == 20
