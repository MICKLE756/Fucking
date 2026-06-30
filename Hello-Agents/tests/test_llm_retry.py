"""离线单测：HelloAgentsLLM 对瞬时错误（含中转误报的 401）做退避重试。

不连任何真实服务：构造一个假的底层 client，让 chat.completions.create 先抛
若干次瞬时错误再成功，验证 _create / invoke 会自动重试并最终返回。
"""

import time
import types

import pytest

from hello_agents.core.llm import HelloAgentsLLM
from hello_agents.core.exceptions import HelloAgentsException


class _Resp:
    """最小化模拟 openai 返回对象：response.choices[0].message.content。"""

    def __init__(self, content="ok"):
        msg = types.SimpleNamespace(content=content, reasoning_content=None, tool_calls=None)
        self.choices = [types.SimpleNamespace(message=msg)]
        self.usage = None
        self.model = "fake-model"


class _Status(Exception):
    """带 status_code 的异常，模拟 openai.APIStatusError（如中转误报 401）。"""

    def __init__(self, status_code):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


def _make_llm(monkeypatch):
    # 显式传入凭据，避免 __init__ 因缺少 key/base_url 抛错；不发真实请求。
    llm = HelloAgentsLLM(
        model="fake-model",
        api_key="sk-fake",
        base_url="http://127.0.0.1:9/v1",
        provider="custom",
    )
    llm.retry_backoff = 0.0  # 测试不真正 sleep
    return llm


def _wire_client(llm, side_effects):
    """让 llm._client.chat.completions.create 依次产出 side_effects（异常则抛出）。"""
    calls = {"n": 0}

    def _create(**params):
        i = calls["n"]
        calls["n"] += 1
        eff = side_effects[i]
        if isinstance(eff, Exception):
            raise eff
        return eff

    completions = types.SimpleNamespace(create=_create)
    chat = types.SimpleNamespace(completions=completions)
    llm._client = types.SimpleNamespace(chat=chat)
    return calls


def test_invoke_retries_then_succeeds_on_transient_401(monkeypatch):
    llm = _make_llm(monkeypatch)
    calls = _wire_client(llm, [_Status(401), _Status(401), _Resp("hi")])

    out = llm.invoke([{"role": "user", "content": "x"}])

    assert out == "hi"
    assert calls["n"] == 3  # 两次 401 + 一次成功


def test_invoke_retries_on_429_and_503(monkeypatch):
    llm = _make_llm(monkeypatch)
    calls = _wire_client(llm, [_Status(429), _Status(503), _Resp("done")])

    assert llm.invoke([{"role": "user", "content": "x"}]) == "done"
    assert calls["n"] == 3


def test_non_retryable_400_raises_immediately(monkeypatch):
    llm = _make_llm(monkeypatch)
    calls = _wire_client(llm, [_Status(400), _Resp("never")])

    with pytest.raises(HelloAgentsException):
        llm.invoke([{"role": "user", "content": "x"}])
    assert calls["n"] == 1  # 400 不重试


def test_retries_exhausted_raises(monkeypatch):
    llm = _make_llm(monkeypatch)
    llm.max_retries = 2
    calls = _wire_client(llm, [_Status(401), _Status(401), _Status(401), _Resp("late")])

    with pytest.raises(HelloAgentsException):
        llm.invoke([{"role": "user", "content": "x"}])
    assert calls["n"] == 3  # 首次 + 2 次重试，用尽后抛出


def test_backoff_grows_exponentially(monkeypatch):
    llm = _make_llm(monkeypatch)
    llm.retry_backoff = 0.5
    delays = []
    monkeypatch.setattr(time, "sleep", lambda d: delays.append(d))
    _wire_client(llm, [_Status(401), _Status(401), _Resp("ok")])

    assert llm.invoke([{"role": "user", "content": "x"}]) == "ok"
    assert delays == [0.5, 1.0]  # 0.5 * 2**0, 0.5 * 2**1
