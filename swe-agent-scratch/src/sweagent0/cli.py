"""CLI：在任意本地仓库上运行 Agent 解决一个任务。

用法：
    export OPENAI_API_KEY=...
    sweagent0 --workdir /path/to/repo --task "修复 xxx bug" [--config config.yaml]
"""

from __future__ import annotations

import argparse
import logging

from .agent import Agent
from .config import RunConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="sweagent0: 自主软件工程 Agent")
    parser.add_argument("--task", required=True, help="任务/问题描述")
    parser.add_argument("--workdir", default=".", help="目标仓库路径")
    parser.add_argument("--config", default=None, help="RunConfig YAML 路径")
    parser.add_argument("--trajectory", default="trajectory.json", help="轨迹保存路径")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    config = RunConfig.from_yaml(args.config) if args.config else RunConfig()
    config.workdir = args.workdir

    agent = Agent(config)
    result = agent.run(args.task)
    result.trajectory.save(args.trajectory)

    print(f"状态: {result.status}  步数: {result.steps_used}")
    print(f"LLM 调用: {agent.llm.usage.calls} 次, "
          f"tokens: {agent.llm.usage.prompt_tokens}+{agent.llm.usage.completion_tokens}")
    if result.patch:
        print("---最终 patch---")
        print(result.patch)


if __name__ == "__main__":
    main()
