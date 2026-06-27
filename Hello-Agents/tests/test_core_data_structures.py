"""核心数据结构离线单测：Message / Config / LLMResponse / ToolCall 等。"""

from hello_agents.core.message import Message
from hello_agents.core.config import Config
from hello_agents.core.llm_response import (
    ToolCall,
    LLMToolResponse,
    LLMResponse,
    StreamStats,
)


def test_message_positional_args_and_roundtrip():
    msg = Message("你好", "user")
    assert msg.content == "你好"
    assert msg.role == "user"
    assert msg.timestamp is not None  # __init__ 自动填充

    d = msg.to_dict()
    assert d["role"] == "user"
    assert d["content"] == "你好"

    restored = Message.from_dict(d)
    assert restored.content == msg.content
    assert restored.role == msg.role


def test_message_to_text_and_summary_role():
    msg = Message("摘要内容", "summary")
    assert msg.role == "summary"
    assert msg.to_text() == "[summary] 摘要内容"
    assert str(msg) == "[summary] 摘要内容"


def test_config_defaults_and_to_dict():
    cfg = Config()
    assert cfg.default_model == "gpt-3.5-turbo"
    assert cfg.temperature == 0.7
    assert cfg.to_dict()["max_history_length"] == 100


def test_config_from_env(monkeypatch):
    monkeypatch.setenv("DEBUG", "true")
    monkeypatch.setenv("TEMPERATURE", "0.1")
    monkeypatch.setenv("MAX_TOKENS", "256")
    cfg = Config.from_env()
    assert cfg.debug is True
    assert cfg.temperature == 0.1
    assert cfg.max_tokens == 256


def test_llm_response_str_is_content_backcompat():
    resp = LLMResponse(content="答案", model="m", usage={"total_tokens": 9})
    # 关键的向后兼容：str(resp) 直接是 content
    assert str(resp) == "答案"
    assert resp.to_dict()["usage"]["total_tokens"] == 9
    assert "reasoning_content" not in resp.to_dict()


def test_llm_response_includes_reasoning_when_present():
    resp = LLMResponse(content="a", model="m", reasoning_content="思考")
    assert resp.to_dict()["reasoning_content"] == "思考"


def test_tool_call_and_tool_response():
    tc = ToolCall(id="1", name="calc", arguments='{"x":1}')
    resp = LLMToolResponse(content=None, tool_calls=[tc], model="m")
    assert resp.content is None
    assert resp.tool_calls[0].name == "calc"
    assert resp.usage == {}  # default_factory=dict
    assert resp.latency_ms == 0


def test_stream_stats_to_dict():
    stats = StreamStats(model="m", usage={"total_tokens": 3}, latency_ms=12)
    d = stats.to_dict()
    assert d["model"] == "m"
    assert d["latency_ms"] == 12
