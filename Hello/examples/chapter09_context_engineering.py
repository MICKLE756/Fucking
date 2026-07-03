"""第九章：上下文工程示例

展示 hello_agents.context 模块的四个核心组件，以及它们与
NoteTool / TerminalTool 的结合使用：

1. 🔢 TokenCounter：Token 计数（缓存 + 增量 + 降级估算）
2. 🗂️ HistoryManager：历史管理（只追加 + summary 压缩）
3. ✂️ ObservationTruncator：工具输出统一截断
4. 🏗️ ContextBuilder（GSSC流水线）+ NoteTool + TerminalTool 综合案例：
   - TerminalTool 即时探索文件（JIT 检索）
   - ObservationTruncator 截断过长的工具输出
   - NoteTool 沉淀任务状态与关键结论
   - ContextBuilder 把笔记、工具结果、对话历史组装成结构化上下文

本示例不依赖 LLM API，可直接运行观察各组件行为。
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import shutil
import tempfile
from pathlib import Path

from hello_agents.core.message import Message
from hello_agents.context import (
    ContextBuilder,
    ContextConfig,
    ContextPacket,
    HistoryManager,
    ObservationTruncator,
    TokenCounter,
)
from hello_agents.tools import NoteTool, TerminalTool


def demo_token_counter():
    """演示1: TokenCounter - Token 计数器"""
    print("🔢 演示1: TokenCounter")
    print("=" * 50)

    counter = TokenCounter(model="gpt-4")

    msg1 = Message("什么是上下文工程？", "user")
    msg2 = Message("上下文工程是为LLM构建高质量输入的系统方法。", "assistant")

    print(f"单条消息 Token 数: {counter.count_message(msg1)}")
    print(f"消息列表 Token 数: {counter.count_messages([msg1, msg2])}")

    # 重复计算命中缓存
    counter.count_message(msg1)
    print(f"缓存统计: {counter.get_cache_stats()}")

    # 纯文本计数
    text = "GSSC = Gather, Select, Structure, Compress"
    print(f"文本 '{text}' 的 Token 数: {counter.count_text(text)}")
    print()


def demo_history_manager():
    """演示2: HistoryManager - 历史管理与压缩"""
    print("🗂️ 演示2: HistoryManager")
    print("=" * 50)

    manager = HistoryManager(min_retain_rounds=2)

    # 模拟5轮对话
    for i in range(1, 6):
        manager.append(Message(f"第{i}个问题", "user"))
        manager.append(Message(f"第{i}个回答", "assistant"))

    print(f"压缩前: {len(manager.get_history())} 条消息, {manager.estimate_rounds()} 轮")
    print(f"轮次边界: {manager.find_round_boundaries()}")

    # 压缩：旧历史 -> summary 消息，只保留最近2轮
    manager.compress("用户先后询问了5个问题，前3轮已确认上下文工程的基本概念。")

    history = manager.get_history()
    print(f"压缩后: {len(history)} 条消息, 首条角色 = {history[0].role}")
    for msg in history:
        print(f"  {msg.to_text()[:60]}")

    # 序列化 / 反序列化
    data = manager.to_dict()
    restored = HistoryManager()
    restored.load_from_dict(data)
    print(f"会话恢复后: {len(restored.get_history())} 条消息")
    print()


def demo_observation_truncator(workspace: str):
    """演示3: ObservationTruncator - 工具输出截断"""
    print("✂️ 演示3: ObservationTruncator")
    print("=" * 50)

    truncator = ObservationTruncator(
        max_lines=5,
        max_bytes=51200,
        truncate_direction="head_tail",
        output_dir=os.path.join(workspace, "tool-output"),
    )

    long_output = "\n".join(f"日志行 {i}: 模拟一条很长的工具输出" for i in range(1, 101))
    result = truncator.truncate(
        tool_name="terminal",
        output=long_output,
        metadata={"command": "cat app.log"},
    )

    print(f"是否截断: {result['truncated']}")
    print(f"统计: {result['stats']}")
    print(f"完整输出已保存到: {result['full_output_path']}")
    print("截断后预览:")
    print(result["preview"])
    print()


def demo_context_builder_with_tools(workspace: str):
    """演示4: ContextBuilder + NoteTool + TerminalTool 综合案例

    模拟一个"代码仓库分析"任务：
    1. TerminalTool 探索项目文件（Gather 阶段的 JIT 检索）
    2. ObservationTruncator 控制工具输出体积
    3. NoteTool 记录任务状态与关键结论（跨轮次的外部记忆）
    4. ContextBuilder 将以上信息组装为结构化上下文
    """
    print("🏗️ 演示4: ContextBuilder + NoteTool + TerminalTool")
    print("=" * 50)

    # --- 准备一个模拟项目目录 ---
    project_dir = Path(workspace) / "demo_project"
    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "README.md").write_text(
        "# Demo Project\n\n一个用于演示上下文工程 context engineering 的示例项目。\n",
        encoding="utf-8",
    )
    (project_dir / "main.py").write_text(
        "def main():\n    print('hello context engineering')\n\n"
        "if __name__ == '__main__':\n    main()\n",
        encoding="utf-8",
    )

    # --- 步骤1: TerminalTool 即时探索文件 ---
    terminal = TerminalTool(workspace=str(project_dir))
    print("步骤1: TerminalTool 探索项目")
    ls_output = terminal.run({"command": "ls"})
    print(f"  $ ls\n{ls_output}")
    readme_output = terminal.run({"command": "cat README.md"})

    # --- 步骤2: ObservationTruncator 截断工具输出 ---
    truncator = ObservationTruncator(
        max_lines=10,
        output_dir=os.path.join(workspace, "tool-output"),
    )
    readme_result = truncator.truncate(
        tool_name="terminal",
        output=readme_output,
        metadata={"command": "cat README.md"},
    )
    print(f"步骤2: 工具输出截断 (truncated={readme_result['truncated']})")

    # --- 步骤3: NoteTool 沉淀任务状态与结论 ---
    note_tool = NoteTool(workspace=os.path.join(workspace, "notes"))
    print("步骤3: NoteTool 记录任务状态与结论")
    print(note_tool.run({
        "action": "create",
        "title": "仓库分析任务状态",
        "content": "已完成：列出项目文件、阅读 README。\n待办：分析 main.py 入口逻辑。",
        "note_type": "task_state",
        "tags": ["context engineering"],
    }))
    print(note_tool.run({
        "action": "create",
        "title": "关键结论",
        "content": "该项目是 context engineering 演示项目，入口为 main.py。",
        "note_type": "conclusion",
        "tags": ["context engineering"],
    }))

    # 从笔记中取回任务状态（跨轮次外部记忆）
    task_notes = note_tool.run({"action": "search", "query": "context engineering"})

    # --- 步骤4: ContextBuilder 组装结构化上下文 ---
    print("步骤4: ContextBuilder 构建结构化上下文 (GSSC)")
    builder = ContextBuilder(
        config=ContextConfig(
            max_tokens=4000,
            reserve_ratio=0.15,
            min_relevance=0.0,  # 演示用：不过滤低相关性包
        )
    )

    history = [
        Message("请帮我分析 demo_project 这个仓库", "user"),
        Message("好的，我先列出项目文件并阅读 README。", "assistant"),
    ]

    additional_packets = [
        ContextPacket(
            content=f"terminal 工具输出 (cat README.md):\n{readme_result['preview']}",
            metadata={"type": "tool_result"},
        ),
        ContextPacket(
            content=f"note 工具检索结果:\n{task_notes}",
            metadata={"type": "task_state"},
        ),
    ]

    context = builder.build(
        user_query="demo_project 这个仓库的入口文件是什么？",
        conversation_history=history,
        system_instructions="你是一个代码仓库分析助手，回答需给出依据。",
        additional_packets=additional_packets,
    )

    counter = TokenCounter()
    print(f"最终上下文 Token 数: {counter.count_text(context)}")
    print("-" * 50)
    print(context)
    print("-" * 50)
    print()


def main():
    print("=== 第九章：上下文工程示例 ===\n")

    workspace = tempfile.mkdtemp(prefix="chapter09_")
    try:
        demo_token_counter()
        demo_history_manager()
        demo_observation_truncator(workspace)
        demo_context_builder_with_tools(workspace)
        print("✅ 全部演示完成")
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


if __name__ == "__main__":
    main()
