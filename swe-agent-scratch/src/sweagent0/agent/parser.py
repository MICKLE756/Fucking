"""动作解析：从 LLM 回复中提取 ```action JSON 代码块。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

ACTION_RE = re.compile(r"```(?:action|json)\s*\n(.*?)\n\s*```", re.DOTALL)


class FormatError(Exception):
    """动作格式错误，错误信息会返回给 LLM 让其自我纠正。"""


@dataclass
class Action:
    tool: str
    args: dict


def parse_action(text: str) -> Action:
    blocks = ACTION_RE.findall(text)
    if not blocks:
        raise FormatError("未找到 ```action 代码块。请输出恰好一个 ```action 代码块，内含 JSON。")
    if len(blocks) > 1:
        raise FormatError(f"发现 {len(blocks)} 个代码块，每次只能输出一个 action。")
    try:
        data = json.loads(blocks[0])
    except json.JSONDecodeError as e:
        raise FormatError(f"action 代码块不是合法 JSON: {e}") from e
    if not isinstance(data, dict) or "tool" not in data:
        raise FormatError('action JSON 必须形如 {"tool": "...", "args": {...}}')
    args = data.get("args", {})
    if not isinstance(args, dict):
        raise FormatError("args 必须是 JSON 对象")
    return Action(tool=str(data["tool"]), args=args)
