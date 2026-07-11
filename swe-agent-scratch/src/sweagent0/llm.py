"""LLM 客户端：OpenAI 兼容接口（支持国内中转），带重试与用量统计。"""

from __future__ import annotations

from dataclasses import dataclass, field

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from .config import ModelConfig

Message = dict[str, str]


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0

    def add(self, prompt: int, completion: int) -> None:
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.calls += 1


@dataclass
class LLMClient:
    config: ModelConfig
    usage: Usage = field(default_factory=Usage)

    def __post_init__(self) -> None:
        self._client = OpenAI(api_key=self.config.api_key, base_url=self.config.base_url)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=20), reraise=True)
    def chat(self, messages: list[Message]) -> str:
        resp = self._client.chat.completions.create(
            model=self.config.model_name,
            messages=messages,  # type: ignore[arg-type]
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        if resp.usage is not None:
            self.usage.add(resp.usage.prompt_tokens, resp.usage.completion_tokens)
        content = resp.choices[0].message.content
        if content is None:
            raise RuntimeError("LLM 返回空内容")
        return content


def estimate_tokens(text: str) -> int:
    """粗略 token 估算（中英文混合按 3 字符/token）。"""
    return len(text) // 3 + 1
