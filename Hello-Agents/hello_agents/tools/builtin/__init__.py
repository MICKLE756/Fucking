"""内置工具模块

HelloAgents框架的内置工具集合，包括：
- SearchTool: 网页搜索工具
- CalculatorTool: 数学计算工具
- MemoryTool: 记忆工具
- RAGTool: 检索增强生成工具
- NoteTool: 结构化笔记工具（第9章）
- TerminalTool: 命令行工具（第9章）
- MCPTool: MCP 协议工具（第10章 - 基于 mcp v1.15.0）
- A2ATool: A2A 协议工具（第10章 - 基于 python-a2a v0.5.10）
- ANPTool: ANP 协议工具（第10章 - 基于 agent-connect v0.3.7）
- BFCLEvaluationTool: BFCL评估工具（第12章）
- GAIAEvaluationTool: GAIA评估工具（第12章）
- LLMJudgeTool: LLM Judge评估工具（第12章）
- WinRateTool: Win Rate评估工具（第12章）

注意：memory/rag/protocol/evaluation/rl 等工具依赖可选的重型第三方库
（torch、qdrant、huggingface_hub、trl 等）。这些导入用 try/except 包裹，
缺少对应可选依赖时**优雅跳过**而非让整个包导入失败——只装核心依赖也能
使用 SearchTool / CalculatorTool 等轻量工具。
"""

# 轻量内置工具（仅依赖核心库，始终可用）
from .search_tool import SearchTool
from .calculator import CalculatorTool

__all__ = [
    "SearchTool",
    "CalculatorTool",
]

# 可选内置工具：依赖缺失时优雅跳过（不影响核心功能）
try:
    from .memory_tool import MemoryTool
    __all__.append("MemoryTool")
except ImportError:
    pass

try:
    from .rag_tool import RAGTool
    __all__.append("RAGTool")
except ImportError:
    pass

try:
    from .note_tool import NoteTool
    __all__.append("NoteTool")
except ImportError:
    pass

try:
    from .terminal_tool import TerminalTool
    __all__.append("TerminalTool")
except ImportError:
    pass

try:
    from .protocol_tools import MCPTool, A2ATool, ANPTool
    __all__ += ["MCPTool", "A2ATool", "ANPTool"]
except ImportError:
    pass

try:
    from .bfcl_evaluation_tool import BFCLEvaluationTool
    __all__.append("BFCLEvaluationTool")
except ImportError:
    pass

try:
    from .gaia_evaluation_tool import GAIAEvaluationTool
    __all__.append("GAIAEvaluationTool")
except ImportError:
    pass

try:
    from .llm_judge_tool import LLMJudgeTool
    __all__.append("LLMJudgeTool")
except ImportError:
    pass

try:
    from .win_rate_tool import WinRateTool
    __all__.append("WinRateTool")
except ImportError:
    pass
