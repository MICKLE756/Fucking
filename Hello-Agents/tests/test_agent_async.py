"""Agent 异步与生命周期离线单测：arun / emit_event / on_event 钩子。"""

import asyncio

from hello_agents.core.lifecycle import EventType
from hello_agents.agents.simple_agent import SimpleAgent


def test_arun_returns_same_as_run_and_emits_events(fake_llm_factory):
    llm = fake_llm_factory(["异步回复"])
    events = []

    async def on_event(event):
        events.append(event.type)

    agent = SimpleAgent(name="t", llm=llm, on_event=on_event)
    result = asyncio.run(agent.arun("你好"))

    assert result == "异步回复"
    # arun 应发出 START 与 FINISH 事件
    assert events == [EventType.AGENT_START, EventType.AGENT_FINISH]
    # 历史里 user/assistant 各一条
    assert [m.role for m in agent.get_history()] == ["user", "assistant"]


def test_arun_emits_error_event_and_reraises(fake_llm_factory):
    class BoomLLM:
        provider = "fake"

        def invoke(self, messages, **kwargs):
            raise RuntimeError("boom")

    events = []

    async def on_event(event):
        events.append(event.type)

    agent = SimpleAgent(name="t", llm=BoomLLM(), on_event=on_event)

    async def _go():
        try:
            await agent.arun("x")
        except RuntimeError as e:
            return str(e)
        return None

    err = asyncio.run(_go())
    assert err == "boom"
    assert events == [EventType.AGENT_START, EventType.AGENT_ERROR]


def test_emit_event_noop_without_hook(fake_llm):
    agent = SimpleAgent(name="t", llm=fake_llm)  # 未设置 on_event

    async def _go():
        # 不应抛错，静默跳过
        await agent.emit_event(EventType.AGENT_START, foo="bar")

    asyncio.run(_go())
