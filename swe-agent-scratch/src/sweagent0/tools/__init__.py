from .base import Tool, ToolError, ToolRegistry
from .bash import BashTool
from .editor import EditorTool
from .git_ops import GitTool
from .search import SearchTool
from .submit import SubmitTool
from .tester import TestTool

__all__ = [
    "Tool",
    "ToolError",
    "ToolRegistry",
    "BashTool",
    "EditorTool",
    "GitTool",
    "SearchTool",
    "SubmitTool",
    "TestTool",
    "default_registry",
]


def default_registry(workdir: str) -> ToolRegistry:
    reg = ToolRegistry()
    for tool in (
        BashTool(workdir),
        EditorTool(workdir),
        SearchTool(workdir),
        TestTool(workdir),
        GitTool(workdir),
        SubmitTool(workdir),
    ):
        reg.register(tool)
    return reg
