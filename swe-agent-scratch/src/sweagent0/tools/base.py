"""工具抽象：每个工具声明名称、说明、参数 schema，供 Agent 提示词渲染与调用分发。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


class ToolError(Exception):
    """工具调用失败（参数错误、执行失败等），信息会作为观察结果返回给 LLM。"""


@dataclass
class ToolResult:
    output: str
    is_submit: bool = False  # SubmitTool 置为 True，表示任务结束


class Tool(ABC):
    name: str
    description: str  # 渲染进系统提示词
    args_hint: str  # 参数说明（渲染进系统提示词）

    def __init__(self, workdir: str) -> None:
        self.workdir = workdir

    @abstractmethod
    def run(self, args: dict) -> ToolResult: ...

    def require(self, args: dict, *keys: str) -> None:
        missing = [k for k in keys if k not in args]
        if missing:
            raise ToolError(f"工具 {self.name} 缺少参数: {', '.join(missing)}")


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise ToolError(f"未知工具: {name}，可用工具: {', '.join(self._tools)}")
        return self._tools[name]

    def render_docs(self) -> str:
        lines = []
        for tool in self._tools.values():
            lines.append(f"### {tool.name}\n{tool.description}\n参数: {tool.args_hint}\n")
        return "\n".join(lines)
