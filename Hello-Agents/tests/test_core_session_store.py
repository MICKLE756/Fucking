"""会话持久化离线单测：SessionStore 保存/加载/列举/删除/一致性检查。"""

from hello_agents.core.message import Message
from hello_agents.core.session_store import SessionStore


def test_save_load_roundtrip(tmp_path):
    store = SessionStore(session_dir=str(tmp_path / "sessions"))
    history = [Message("你好", "user"), Message("您好", "assistant")]

    filepath = store.save(
        agent_config={"name": "a", "llm_model": "gpt-4"},
        history=history,
        tool_schema_hash="abc",
        read_cache={},
        metadata={"total_tokens": 10},
        session_name="demo",
    )

    data = store.load(filepath)
    assert data["agent_config"]["name"] == "a"
    assert data["tool_schema_hash"] == "abc"
    assert data["metadata"]["total_tokens"] == 10
    # Message 被序列化为 dict
    assert data["history"][0]["content"] == "你好"
    assert data["history"][0]["role"] == "user"


def test_list_and_delete(tmp_path):
    store = SessionStore(session_dir=str(tmp_path / "sessions"))
    store.save({"n": 1}, [], "h", {}, {}, session_name="s1")
    store.save({"n": 2}, [], "h", {}, {}, session_name="s2")

    sessions = store.list_sessions()
    names = {s["filename"] for s in sessions}
    assert names == {"s1.json", "s2.json"}

    assert store.delete("s1") is True
    assert store.delete("nonexistent") is False
    assert {s["filename"] for s in store.list_sessions()} == {"s2.json"}


def test_config_consistency_check(tmp_path):
    store = SessionStore(session_dir=str(tmp_path / "sessions"))
    result = store.check_config_consistency(
        {"llm_model": "gpt-4", "llm_provider": "openai"},
        {"llm_model": "gpt-3.5", "llm_provider": "openai"},
    )
    assert result["consistent"] is False
    assert any("模型变化" in w for w in result["warnings"])

    ok = store.check_config_consistency(
        {"llm_model": "gpt-4"}, {"llm_model": "gpt-4"}
    )
    assert ok["consistent"] is True


def test_tool_schema_consistency_check(tmp_path):
    store = SessionStore(session_dir=str(tmp_path / "sessions"))
    assert store.check_tool_schema_consistency("a", "a")["changed"] is False
    assert store.check_tool_schema_consistency("a", "b")["changed"] is True
