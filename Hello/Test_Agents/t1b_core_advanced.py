import asyncio
import tempfile

from hello_agents.agents.simple_agent import SimpleAgent
from hello_agents.core.lifecycle import EventType, ExecutionContext
from hello_agents.core.llm_response import LLMResponse
from hello_agents.core.session_store import SessionStore


class FakeLLM:
    provider = "fake"

    def invoke(self, messages, **kw): return f"收到{len(messages)}条"

# 1) LLMResponse：__str__ 即 content
r = LLMResponse(content="hi", model="m", usage={"total_tokens": 3}, latency_ms=5)
try:
    assert str(r) == "hi" and r.to_dict()["model"] == "m"
    print("✅ 断言1执行成功")
except AssertionError:
    print("❌ 断言1失败，条件不满足")

# 2) 生命周期：构造时传 on_event，arun 前后会发 START/FINISH
events = []
async def hook(ev):
    events.append(ev.type)
ag = SimpleAgent(name="a", llm=FakeLLM(), on_event=hook)

try:
    assert asyncio.run(ag.arun("yo")) == "收到2条"
    assert events == [EventType.AGENT_START, EventType.AGENT_FINISH]
    print("✅ 断言2执行成功")
except AssertionError:
    print("❌ 断言2失败，条件不满足")



# 4) 执行上下文
ctx = ExecutionContext(input_text="q"); ctx.increment_step(); ctx.add_tokens(10)
assert ctx.current_step == 1 and ctx.total_tokens == 10

# 5) 会话持久化往返
d = tempfile.mkdtemp(); ss = SessionStore(session_dir=d)
fp = ss.save(agent_config={"model": "m"}, history=[{"role": "user", "content": "hi"}],
             tool_schema_hash="abc", read_cache={}, metadata={"steps": 1}, session_name="s1")
assert ss.load(fp)["history"][0]["content"] == "hi" and ss.list_sessions()
print("✅ 核心层进阶全部通过")


