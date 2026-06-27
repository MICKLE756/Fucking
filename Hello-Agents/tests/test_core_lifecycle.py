"""生命周期事件系统离线单测：EventType / AgentEvent / ExecutionContext。"""

from hello_agents.core.lifecycle import EventType, AgentEvent, ExecutionContext


def test_agent_event_create_and_to_dict():
    event = AgentEvent.create(
        EventType.TOOL_CALL, "my_agent", tool_name="search", query="hi"
    )
    assert event.type is EventType.TOOL_CALL
    assert event.agent_name == "my_agent"
    assert event.data == {"tool_name": "search", "query": "hi"}
    assert event.timestamp > 0

    d = event.to_dict()
    assert d["type"] == "tool_call"
    assert d["data"]["tool_name"] == "search"


def test_agent_event_default_data_is_empty_dict():
    event = AgentEvent.create(EventType.AGENT_START, "a")
    assert event.data == {}


def test_execution_context_counters():
    ctx = ExecutionContext(input_text="问题")
    assert ctx.current_step == 0
    assert ctx.total_tokens == 0

    ctx.increment_step()
    ctx.increment_step()
    ctx.add_tokens(100)
    ctx.add_tokens(50)
    assert ctx.current_step == 2
    assert ctx.total_tokens == 150


def test_execution_context_metadata():
    ctx = ExecutionContext(input_text="x")
    ctx.set_metadata("k", "v")
    assert ctx.get_metadata("k") == "v"
    assert ctx.get_metadata("missing", "default") == "default"
