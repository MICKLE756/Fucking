"""SimpleAgent 离线单测：系统提示拼接、多轮记忆、文本协议工具调用。"""

from hello_agents.agents.simple_agent import SimpleAgent
from hello_agents.tools.registry import ToolRegistry
from hello_agents.tools.builtin.calculator import CalculatorTool


def test_single_turn_messages_and_history(fake_llm_factory):
    llm = fake_llm_factory(["收到"])
    agent = SimpleAgent(name="t", llm=llm, system_prompt="你是测试助手")
    out = agent.run("你好")

    assert out == "收到"
    sent = llm.calls[0]
    # 第一条是 system，且为我们设置的 prompt（无工具时不增强）
    assert sent[0] == {"role": "system", "content": "你是测试助手"}
    # 最后一条是本轮 user 输入
    assert sent[-1] == {"role": "user", "content": "你好"}
    # 历史按 user -> assistant 顺序
    hist = agent.get_history()
    assert [m.role for m in hist] == ["user", "assistant"]
    assert hist[0].content == "你好" and hist[1].content == "收到"


def test_default_system_prompt_when_none(fake_llm):
    agent = SimpleAgent(name="t", llm=fake_llm)
    agent.run("hi")
    assert fake_llm.calls[0][0]["content"] == "你是一个有用的AI助手。"


def test_multi_turn_memory(fake_llm):
    agent = SimpleAgent(name="t", llm=fake_llm)
    agent.run("第一句")
    agent.run("第二句")
    # 第二轮：system + 上一轮(user+assistant) + 本轮 user = 4 条
    second = fake_llm.calls[1]
    assert len(second) == 4
    assert second[1] == {"role": "user", "content": "第一句"}
    assert second[-1] == {"role": "user", "content": "第二句"}


def test_text_protocol_tool_call_loop(fake_llm_factory):
    # 第一次回复触发计算器，第二次给出最终答案
    llm = fake_llm_factory([
        "我来算一下 [TOOL_CALL:python_calculator:2+3]",
        "结果是 5",
    ])
    registry = ToolRegistry()
    registry.register_tool(CalculatorTool())

    agent = SimpleAgent(name="t", llm=llm, tool_registry=registry)
    out = agent.run("2+3 等于几？")

    assert out == "结果是 5"
    # 第二次调用 LLM 时，消息里应包含工具执行结果 "5"
    second_call = llm.calls[1]
    joined = "".join(m["content"] for m in second_call)
    assert "5" in joined
    assert "工具执行结果" in joined
