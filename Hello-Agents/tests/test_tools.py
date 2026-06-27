"""工具系统离线单测：CalculatorTool / ToolRegistry。"""

import pytest

from hello_agents.tools.base import Tool, ToolParameter
from hello_agents.tools.registry import ToolRegistry
from hello_agents.tools.builtin.calculator import CalculatorTool, calculate


def test_calculator_basic_and_functions():
    tool = CalculatorTool()
    assert tool.run({"input": "2+3*4"}) == "14"
    assert tool.run({"input": "sqrt(16)"}) == "4.0"
    # 同时支持 expression 键
    assert tool.run({"expression": "10-1"}) == "9"
    # 便捷函数
    assert calculate("2**10") == "1024"


def test_calculator_empty_and_error():
    tool = CalculatorTool()
    assert "不能为空" in tool.run({"input": ""})
    assert "计算失败" in tool.run({"input": "1/0"})


def test_calculator_parameters_definition():
    params = CalculatorTool().get_parameters()
    assert isinstance(params[0], ToolParameter)
    assert params[0].name == "input"
    assert params[0].required is True


def test_registry_register_and_execute_tool():
    registry = ToolRegistry()
    registry.register_tool(CalculatorTool())
    assert "python_calculator" in registry.list_tools()
    assert registry.get_tool("python_calculator") is not None
    # execute_tool 直接传字符串
    assert registry.execute_tool("python_calculator", "6*7") == "42"


def test_registry_register_function_and_unregister():
    registry = ToolRegistry()
    registry.register_function("echo", "回声", lambda s: f"echo:{s}")
    assert registry.get_function("echo")("hi") == "echo:hi"

    registry.unregister("echo")
    assert registry.get_function("echo") is None


def test_registry_get_tools_description_contains_tool():
    registry = ToolRegistry()
    registry.register_tool(CalculatorTool())
    desc = registry.get_tools_description()
    assert "python_calculator" in desc
