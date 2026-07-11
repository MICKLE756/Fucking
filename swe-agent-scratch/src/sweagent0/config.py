"""运行配置：模型、循环上限、成本控制。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class ModelConfig:
    model_name: str = "gpt-4o-mini"
    base_url: str | None = None  # OpenAI 兼容中转地址
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float = 0.0
    max_tokens: int = 4096

    @property
    def api_key(self) -> str:
        key = os.environ.get(self.api_key_env, "")
        if not key:
            raise RuntimeError(f"环境变量 {self.api_key_env} 未设置")
        return key


@dataclass
class AgentConfig:
    max_steps: int = 50
    max_consecutive_format_errors: int = 5
    # 上下文压缩：历史观察结果超过该 token 估算值时触发压缩
    context_token_budget: int = 96_000
    # 单条工具输出保留的最大字符数
    max_observation_chars: int = 16_000
    cost_limit_usd: float = 2.0


@dataclass
class RunConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    workdir: str = "."

    @classmethod
    def from_yaml(cls, path: str | Path) -> RunConfig:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls(
            model=ModelConfig(**data.get("model", {})),
            agent=AgentConfig(**data.get("agent", {})),
            workdir=data.get("workdir", "."),
        )
