"""上下文工程模块

为HelloAgents框架提供上下文工程能力：
- ContextBuilder: GSSC流水线（Gather-Select-Structure-Compress）
- HistoryManager: 历史管理与压缩（summary + 保留最近 N 轮）
- ObservationTruncator: 工具输出统一截断
- TokenCounter: Token 计数器（缓存 + 降级估算）
"""

from .builder import ContextBuilder, ContextConfig, ContextPacket
from .history import HistoryManager
from .truncator import ObservationTruncator
from .token_counter import TokenCounter

__all__ = [
    "ContextBuilder",
    "ContextConfig",
    "ContextPacket",
    "HistoryManager",
    "ObservationTruncator",
    "TokenCounter",
]
