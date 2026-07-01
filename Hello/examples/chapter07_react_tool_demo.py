"""
第七章 · ReActAgent 调工具最小可运行 demo
=============================================

用《Hello-Agents》第七章的方式，搭一个「会思考 + 会调工具」的 ReActAgent：
  Thought（思考）→ Action（调用工具）→ Observation（看结果）→ … → Finish（给答案）

本 demo 注册两个工具：
  - calculate : 计算器（纯本地，无需联网）—— 演示「算 15 * 23 + 45」
  - search    : 网页搜索（需要 TAVILY_API_KEY / SERPAPI_API_KEY，可选）—— 演示时事类问题

运行前置：
  1. 安装依赖：   pip install -r ../requirements.txt
  2. 配一个 LLM：  在仓库根目录（或本目录）建 .env，参考 ../.env.example，至少给：
         LLM_MODEL_ID=...        # 例：gpt-4o-mini / deepseek-chat / qwen-plus
         LLM_API_KEY=...
         LLM_BASE_URL=...        # 例：https://api.openai.com/v1
     HelloAgentsLLM 会自动从环境变量识别 provider。
  3. （可选）搜索工具：再配 TAVILY_API_KEY 或 SERPAPI_API_KEY。

运行：
    python chapter07_react_tool_demo.py            # 跑全部
    python chapter07_react_tool_demo.py --calc     # 只跑计算器（不需要搜索 key）
"""

import os
import sys

# 让 demo 在未做 `pip install -e` 时也能直接跑：把包所在的上层目录加入 import 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

# 先加载 .env（当前目录或上层目录都会被 python-dotenv 自动向上查找）
load_dotenv()

from hello_agents import HelloAgentsLLM, ReActAgent, ToolRegistry, calculate, search
from hello_agents.core.exceptions import HelloAgentsException


def build_agent(max_steps: int = 5) -> ReActAgent:
    """
    组装一个带工具的 ReActAgent。

    四步：
      1. 建 LLM（通信层）——HelloAgentsLLM 从 .env 读取模型 / key / base_url。
      2. 建工具注册表（能力层）——把可调用的工具登记进去。
      3. 注册工具——这里用最简便的 register_function(name, description, func)。
         注意 name 必须和你希望 LLM 在 Action 里写的工具名一致（如 calculate[...]）。
      4. 用 LLM + 工具表实例化 ReActAgent。
    """
    llm = HelloAgentsLLM()  # 缺 key 会在这里抛 HelloAgentsException

    registry = ToolRegistry()
    registry.register_function(
        name="calculate",
        description="执行数学计算，支持基本运算与常见数学函数。例如：15*23+45、sqrt(16)、sin(pi/2)。",
        func=calculate,
    )
    registry.register_function(
        name="search",
        description="网页搜索引擎。当需要时事、事实或知识库里没有的最新信息时使用。",
        func=search,
    )

    return ReActAgent(
        name="工具助手",
        llm=llm,
        tool_registry=registry,
        max_steps=max_steps,
    )


def main() -> int:
    only_calc = "--calc" in sys.argv

    try:
        agent = build_agent()
    except HelloAgentsException as e:
        # 没配 LLM key 时给出明确指引，而不是丢一个晦涩的堆栈
        print("\n⚠️  无法创建 LLM，请先在 .env 配置 LLM_MODEL_ID / LLM_API_KEY / LLM_BASE_URL。")
        print(f"    原始错误：{e}")
        print("    参考 ../.env.example。")
        return 1

    # 任务 1：纯计算，靠 calculate 工具，无需联网 —— 最稳的工具调用演示
    task_calc = "请计算 15 * 23 + 45 等于多少？"
    print("\n" + "=" * 60)
    print(f"🎯 任务一（计算器）: {task_calc}")
    print("=" * 60)
    print(f"\n✅ 最终答案: {agent.run(task_calc)}")

    if only_calc:
        return 0

    # 任务 2：时事类，靠 search 工具（需要搜索 key，没有则工具会返回错误信息，
    #         ReAct 循环仍会优雅结束，不会崩）
    task_search = "用一句话告诉我 LangGraph 是什么？"
    print("\n" + "=" * 60)
    print(f"🎯 任务二（搜索）: {task_search}")
    print("=" * 60)
    print(f"\n✅ 最终答案: {agent.run(task_search)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
