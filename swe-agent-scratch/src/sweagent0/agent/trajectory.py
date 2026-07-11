"""轨迹管理与上下文压缩。

轨迹（trajectory）= 完整的 step 记录，用于复盘、评测与后续 RL 训练数据构造。
上下文压缩：对话历史超预算时，把较早的工具观察结果替换为占位摘要，
保留 LLM 的思考/动作文本（信息密度更高），这是控制长任务成本的关键。
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..llm import Message, estimate_tokens

ELIDED = "[早期观察结果已压缩省略，如需该信息请重新调用工具获取]"


@dataclass
class Step:
    index: int
    assistant_text: str
    tool_name: str
    tool_args: dict
    observation: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class Trajectory:
    task: str
    steps: list[Step] = field(default_factory=list)

    def add(self, assistant_text: str, tool_name: str, tool_args: dict, observation: str) -> Step:
        step = Step(len(self.steps), assistant_text, tool_name, tool_args, observation)
        self.steps.append(step)
        return step

    def save(self, path: str | Path) -> None:
        data = {"task": self.task, "steps": [asdict(s) for s in self.steps]}
        Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def build_messages(
    system_prompt: str,
    task_prompt: str,
    trajectory: Trajectory,
    token_budget: int,
) -> list[Message]:
    """由轨迹重建对话历史，超预算时从最早的观察结果开始压缩。

    压缩策略：始终完整保留 (1) 系统/任务提示 (2) 所有 assistant 文本
    (3) 最近 5 步的观察结果；更早的观察结果按需替换为占位符。
    """
    keep_recent = 5
    steps = trajectory.steps

    def render(elide_before: int) -> list[Message]:
        msgs: list[Message] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task_prompt},
        ]
        for step in steps:
            msgs.append({"role": "assistant", "content": step.assistant_text})
            obs = ELIDED if step.index < elide_before else step.observation
            msgs.append({"role": "user", "content": f"观察结果:\n{obs}"})
        return msgs

    def total_tokens(msgs: list[Message]) -> int:
        return sum(estimate_tokens(m["content"]) for m in msgs)

    elide_before = 0
    msgs = render(elide_before)
    while total_tokens(msgs) > token_budget and elide_before < max(len(steps) - keep_recent, 0):
        elide_before += 1
        msgs = render(elide_before)
    return msgs
