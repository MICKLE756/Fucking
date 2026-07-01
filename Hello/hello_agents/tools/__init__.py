"""工具系统

memory/rag/protocol/evaluation/rl 等内置工具依赖可选的重型第三方库
（torch、qdrant、huggingface_hub、trl 等）。这些导入用 try/except 包裹，
缺少对应可选依赖时**优雅跳过**而非让整个包导入失败——只装核心依赖也能
`import hello_agents` 并使用 SearchTool / CalculatorTool 等轻量工具。
"""

from .base import Tool, ToolParameter
from .registry import ToolRegistry, global_registry

# 轻量内置工具（仅依赖核心库，始终可用）
from .builtin.search_tool import SearchTool
from .builtin.calculator import CalculatorTool

# 高级功能（核心库实现，始终可用）
from .chain import ToolChain, ToolChainManager, create_research_chain, create_simple_chain
from .async_executor import AsyncToolExecutor, run_parallel_tools, run_batch_tool, run_parallel_tools_sync, run_batch_tool_sync

__all__ = [
    # 基础工具系统
    "Tool",
    "ToolParameter",
    "ToolRegistry",
    "global_registry",

    # 内置工具（轻量，始终可用）
    "SearchTool",
    "CalculatorTool",

    # 工具链功能
    "ToolChain",
    "ToolChainManager",
    "create_research_chain",
    "create_simple_chain",

    # 异步执行功能
    "AsyncToolExecutor",
    "run_parallel_tools",
    "run_batch_tool",
    "run_parallel_tools_sync",
    "run_batch_tool_sync",
]

# 可选内置工具：依赖缺失时优雅跳过（不影响核心功能）
try:
    from .builtin.memory_tool import MemoryTool
    __all__.append("MemoryTool")
except ImportError:
    pass

try:
    from .builtin.rag_tool import RAGTool
    __all__.append("RAGTool")
except ImportError:
    pass

try:
    from .builtin.note_tool import NoteTool
    __all__.append("NoteTool")
except ImportError:
    pass

try:
    from .builtin.terminal_tool import TerminalTool
    __all__.append("TerminalTool")
except ImportError:
    pass

try:
    from .builtin.protocol_tools import MCPTool, A2ATool, ANPTool
    __all__ += ["MCPTool", "A2ATool", "ANPTool"]
except ImportError:
    pass

try:
    from .builtin.bfcl_evaluation_tool import BFCLEvaluationTool
    __all__.append("BFCLEvaluationTool")
except ImportError:
    pass

try:
    from .builtin.gaia_evaluation_tool import GAIAEvaluationTool
    __all__.append("GAIAEvaluationTool")
except ImportError:
    pass

try:
    from .builtin.llm_judge_tool import LLMJudgeTool
    __all__.append("LLMJudgeTool")
except ImportError:
    pass

try:
    from .builtin.win_rate_tool import WinRateTool
    __all__.append("WinRateTool")
except ImportError:
    pass

try:
    from .builtin.rl_training_tool import RLTrainingTool
    __all__.append("RLTrainingTool")
except ImportError:
    pass
