"""LLM响应对象定义"""
from typing import Optional, Dict, List
from dataclasses import dataclass, field

@dataclass # @dataclass 快速创建数据载体
class ToolCall:
    """统一的工具调用对象"""
    id: str  # 本次工具调用唯一ID，用来区分多轮调用
    name: str  # 工具名称，例如 "read_file", "list_directory"
    arguments: str  # JSON字符串格式的入参


@dataclass
class LLMResponse:
    """
    统一的LLM响应对象

    包含响应内容、推理过程（thinking model）、token使用统计、耗时等信息
    """

    content: str
    """回复内容"""

    model: str
    """实际使用的模型名称"""

    usage: Dict[str, int] = field(default_factory=dict)
    """Token使用统计: {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}"""
    # 如果写成Dict[str, int] = {}: 这是可变对象作为默认值，所有实例会共用同一个字典，一个对象改了
    # default_factory = dict
    # 解决这个问题：只有在创建新实例的时候，才临时执行dict()，生成一个全新空字典；每个对象都拥有独立的字典，互不干扰。

    latency_ms: int = 0
    """调用耗时（毫秒）"""

    reasoning_content: Optional[str] = None
    """推理过程（仅thinking model如o1、deepseek-reasoner有此字段）"""

    def __str__(self) -> str:
        """向后兼容：直接打印返回content"""
        return self.content

    def __repr__(self) -> str:
        """详细信息展示"""
        parts = [
            f"LLMResponse(model={self.model}",
            f"latency={self.latency_ms}ms",
            f"tokens={self.usage.get('total_tokens', 0)}",
        ]
        if self.reasoning_content:
            parts.append("has_reasoning=True")
        parts.append(f"content_length={len(self.content)})")
        return ", ".join(parts)

    def to_dict(self) -> Dict:
        """转换为字典格式，方便日志记录"""
        result = {
            "content": self.content,
            "model": self.model,
            "usage": self.usage,
            "latency_ms": self.latency_ms,
        }
        if self.reasoning_content:
            result["reasoning_content"] = self.reasoning_content
        return result


@dataclass
class StreamStats:
    """
    流式调用的统计信息

    在流式调用结束后可通过 llm.last_call_stats 获取
    """

    model: str
    """实际使用的模型名称"""

    usage: Dict[str, int] = field(default_factory=dict)
    """Token使用统计"""

    latency_ms: int = 0
    """调用耗时（毫秒）"""

    reasoning_content: Optional[str] = None
    """推理过程（仅thinking model）"""

    def to_dict(self) -> Dict:
        """转换为字典格式"""
        result = {
            "model": self.model,
            "usage": self.usage,
            "latency_ms": self.latency_ms,
        }
        if self.reasoning_content:
            result["reasoning_content"] = self.reasoning_content
        return result

@dataclass
class LLMToolResponse:
    """统一的工具调用响应对象""" # invoke_with_tools() 的返回值
    content: Optional[str]
    tool_calls: List[ToolCall]
    model: str
    usage: Dict[str, int] = field(default_factory=dict)
    latency_ms: int = 0