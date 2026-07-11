"""Agent 主循环：query LLM → 解析动作 → 执行工具 → 追加观察结果，直到 submit 或达到上限。"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from ..config import RunConfig
from ..llm import LLMClient
from ..repomap import build_repo_map
from ..tools import ToolError, default_registry
from .parser import FormatError, parse_action
from .prompts import render_system_prompt, render_task_prompt
from .trajectory import Trajectory, build_messages

logger = logging.getLogger("sweagent0")


@dataclass
class AgentResult:
    status: str  # "submitted" | "max_steps" | "format_error_limit"
    patch: str
    trajectory: Trajectory
    steps_used: int


class Agent:
    def __init__(self, config: RunConfig, llm: LLMClient | None = None) -> None:
        self.config = config
        self.llm = llm or LLMClient(config.model)
        self.tools = default_registry(config.workdir)
        self.system_prompt = render_system_prompt(self.tools.render_docs())

    def run(self, problem_statement: str) -> AgentResult:
        repo_map = build_repo_map(self.config.workdir)
        task_prompt = render_task_prompt(problem_statement, repo_map)
        trajectory = Trajectory(task=problem_statement)
        format_errors = 0
        patch = ""

        for step_idx in range(self.config.agent.max_steps):
            messages = build_messages(
                self.system_prompt, task_prompt, trajectory, self.config.agent.context_token_budget
            )
            reply = self.llm.chat(messages)

            try:
                action = parse_action(reply)
            except FormatError as e:
                format_errors += 1
                trajectory.add(reply, "__format_error__", {}, str(e))
                logger.warning("step %d 格式错误: %s", step_idx, e)
                if format_errors >= self.config.agent.max_consecutive_format_errors:
                    return AgentResult("format_error_limit", "", trajectory, step_idx + 1)
                continue
            format_errors = 0

            try:
                tool = self.tools.get(action.tool)
                result = tool.run(action.args)
                observation = result.output
            except ToolError as e:
                result = None
                observation = f"工具错误: {e}"

            if len(observation) > self.config.agent.max_observation_chars:
                observation = (
                    observation[: self.config.agent.max_observation_chars] + "\n...[观察结果被截断]"
                )
            trajectory.add(reply, action.tool, action.args, observation)
            logger.info("step %d: %s(%s)", step_idx, action.tool, list(action.args))

            if result is not None and result.is_submit:
                patch = observation.split("---PATCH---", 1)[-1].strip()
                return AgentResult("submitted", patch, trajectory, step_idx + 1)

        return AgentResult("max_steps", patch, trajectory, self.config.agent.max_steps)
