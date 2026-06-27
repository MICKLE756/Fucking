"""pytest 共享配置与离线测试夹具。

为什么需要这个文件？
--------------------
``hello_agents/__init__.py`` 与 ``tools/__init__.py`` / ``tools/builtin/__init__.py``
会**急切地**导入全部可选扩展工具（memory / rag / protocols / evaluation / rl），
因此 ``import hello_agents`` 需要 torch、trl、qdrant、neo4j、spacy、fastmcp 等重依赖。
单元测试只想验证「核心层 + 纯核心依赖」的逻辑，不应被这些重依赖拖累。

这里把若干**父包**注册成轻量占位模块（只给 ``__path__``，不执行其重量级
``__init__`` 体），这样测试就能直接 import 仅依赖核心库的子模块
（core/*、tools.base、tools.registry、tools.builtin.calculator、agents.simple_agent 等），
而 CI 只需安装 requirements.txt 里的核心依赖即可，快速且稳定。
"""

import os
import sys
import types

import pytest

# 仓库根目录（tests/ 的上一级，即 Hello-Agents/）
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.join(ROOT, "hello_agents")

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _register_stub_package(name: str, path: str) -> None:
    """把 name 注册为一个只含 __path__ 的占位包，避免触发其重量级 __init__。"""
    if name in sys.modules:
        return
    module = types.ModuleType(name)
    module.__path__ = [path]
    module.__package__ = name
    sys.modules[name] = module


# 注意：不要 stub ``hello_agents.core`` —— 它的 __init__ 只依赖核心库，
# 让它正常导入，测试才能 ``from hello_agents.core import ...``。
_register_stub_package("hello_agents", PKG)
_register_stub_package("hello_agents.agents", os.path.join(PKG, "agents"))
_register_stub_package("hello_agents.tools", os.path.join(PKG, "tools"))
_register_stub_package(
    "hello_agents.tools.builtin", os.path.join(PKG, "tools", "builtin")
)


class FakeLLM:
    """离线假 LLM：不联网，记录收到的 messages，按脚本返回固定回复。

    - ``responses`` 为 None 时，invoke 返回 ``收到{N}条消息``（N=消息数）。
    - ``responses`` 为列表时，按顺序弹出返回；用尽后回退到默认回复。
    - ``calls`` 记录每次 invoke 收到的 messages，供断言「拼接逻辑」。
    """

    provider = "fake"

    def __init__(self, responses=None):
        self.responses = list(responses) if responses is not None else None
        self.calls = []

    def _next(self, messages):
        self.calls.append(messages)
        if self.responses:
            return self.responses.pop(0)
        return f"收到{len(messages)}条消息"

    def invoke(self, messages, **kwargs):
        return self._next(messages)

    def think(self, messages, **kwargs):
        for ch in self._next(messages):
            yield ch

    def stream_invoke(self, messages, **kwargs):
        yield from self.think(messages, **kwargs)


@pytest.fixture
def fake_llm_factory():
    """返回一个构造 FakeLLM 的工厂，便于每个用例自定义脚本回复。"""
    return FakeLLM


@pytest.fixture
def fake_llm():
    """默认的 FakeLLM 实例（回声式回复）。"""
    return FakeLLM()
