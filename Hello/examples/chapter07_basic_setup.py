"""
第7章示例：HelloAgents框架完整演示 - 从零开始构建Agent框架

本示例展示如何：
1. 配置HelloAgents环境
2. 使用四种不同的Agent范式（默认配置）
3. 自定义Agent配置的高级用法
4. 工具系统的集成和使用
5. 交互式Agent体验

Agent范式包括：
- SimpleAgent: 基础对话Agent
- ReActAgent: 推理与行动结合的Agent
- ReflectionAgent: 自我反思与迭代优化的Agent
- PlanAndSolveAgent: 分解规划与逐步执行的Agent

设计理念：
- 默认优秀：开箱即用的高质量Agent
- 完全可定制：用户可以完全替换提示词模板
- 简洁API：最少的参数，最大的灵活性
"""

import os
from dotenv import load_dotenv

from hello_agents.agents.plan_solve_agent import PlanAndSolveAgent
from hello_agents.agents.react_agent import ReActAgent
from hello_agents.agents.reflection_agent import ReflectionAgent
from hello_agents.agents.simple_agent import SimpleAgent
from hello_agents.core.llm import HelloAgentsLLM
from hello_agents.tools.async_executor import AsyncToolExecutor
from hello_agents.tools.builtin.calculator import calculate
from hello_agents.tools.builtin.search_tool import search
from hello_agents.tools.chain import ToolChain, ToolChainManager
from hello_agents.tools.registry import ToolRegistry
os.environ["NO_PROXY"] = "localhost,127.0.0.1"   # 让本地请求绕过 Clash 代理
# 加载环境变量
load_dotenv()


def demo_simple_agent():
    """演示SimpleAgent - 基础对话"""
    print("\n" + "="*60)
    print("🤖 SimpleAgent 演示 - 基础对话Agent")
    print("="*60)

    # 创建LLM实例
    llm = HelloAgentsLLM()

    # 创建简单Agent
    agent = SimpleAgent(
        name="助手",
        llm=llm,
        system_prompt="你是一个有用的AI助手，请用中文回答问题。"
    )

    # 测试对话
    test_questions = [
        "你好，请介绍一下自己",
        "什么是人工智能？",
        "请用一句话总结机器学习的核心思想"
    ]

    for question in test_questions:
        print(f"\n用户: {question}")
        try:
            response = agent.run(question)
            print(f"助手: {response}")
        except Exception as e:
            print(f"❌ 错误: {e}")

def demo_react_agent():
    """演示ReActAgent - 推理与行动结合"""
    print("\n" + "="*60)
    print("🔧 ReActAgent 演示 - 推理与行动结合的Agent")
    print("="*60)

    # 创建LLM实例
    llm = HelloAgentsLLM()

    # 创建工具注册表
    tool_registry = ToolRegistry()

    # 注册工具
    tool_registry.register_function(
        name="search",
        description="一个网页搜索引擎。当你需要回答关于时事、事实以及在你的知识库中找不到的信息时，应使用此工具。",
        func=search
    )

    tool_registry.register_function(
        name="calculate",
        description="执行数学计算。支持基本运算、数学函数等。例如：2+3*4, sqrt(16), sin(pi/2)等。",
        func=calculate
    )

    # 1. 默认配置演示
    print("\n--- 默认配置 ---")
    default_agent = ReActAgent(
        name="通用助手",
        llm=llm,
        tool_registry=tool_registry,
        max_steps=3
    )

    task1 = "计算 15 * 23 + 45 的结果"
    print(f"\n🎯 任务: {task1}")
    try:
        response = default_agent.run(task1)
        print(f"\n✅ 默认配置结果: {response}")
    except Exception as e:
        print(f"❌ 错误: {e}")

    # 2. 自定义配置演示 - 研究助手
    print("\n--- 自定义配置：研究助手 ---")
    research_prompt = """
你是一个专业的研究助手，擅长信息收集和分析。

可用工具如下：
{tools}

请按照以下格式进行研究：

Thought: 分析问题，确定需要什么信息，制定研究策略。
Action: 选择合适的工具获取信息，格式为：
- `{{tool_name}}[{{tool_input}}]`：调用工具获取信息。
- `Finish[研究结论]`：当你有足够信息得出结论时。

研究问题：{question}
已完成的研究：{history}
"""

    research_agent = ReActAgent(
        name="研究助手",
        llm=llm,
        tool_registry=tool_registry,
        custom_prompt=research_prompt,
        max_steps=3
    )

    task2 = "搜索一下最新的人工智能发展趋势"
    print(f"\n🎯 任务: {task2}")
    try:
        response = research_agent.run(task2)
        print(f"\n✅ 研究助手结果: {response}")
    except Exception as e:
        print(f"❌ 错误: {e}")

def demo_reflection_agent():
    """演示ReflectionAgent - 自我反思与迭代优化"""
    print("\n" + "="*60)
    print("🔄 ReflectionAgent 演示 - 自我反思与迭代优化的Agent")
    print("="*60)

    # 创建LLM实例
    llm = HelloAgentsLLM()

    # 1. 默认配置演示
    print("\n--- 默认配置 ---")
    default_agent = ReflectionAgent(
        name="通用助手",
        llm=llm,
        max_iterations=2
    )

    task1 = "解释什么是递归算法，并给出一个简单的例子"
    print(f"\n🎯 任务: {task1}")
    try:
        response = default_agent.run(task1)
        print(f"\n✅ 默认配置结果:\n{response}")
    except Exception as e:
        print(f"❌ 错误: {e}")

    # 2. 自定义配置演示 - 代码生成专家
    print("\n--- 自定义配置：代码生成专家 ---")
    code_prompts = {
        "initial": """
你是一位资深的程序员。请根据以下要求编写代码：

要求: {task}

请提供完整的代码实现，包含必要的注释和文档。
""",
        "reflect": """
你是一位严格的代码评审专家。请审查以下代码：

# 原始任务: {task}
# 待审查的代码: {content}

请分析代码的质量，包括算法效率、可读性、错误处理等。
如果代码质量良好，请回答"无需改进"。否则请提出具体的改进建议。
""",
        "refine": """
请根据代码评审意见优化你的代码：

# 原始任务: {task}
# 上一轮代码: {last_attempt}
# 评审意见: {feedback}

请提供优化后的代码。
"""
    }

    code_agent = ReflectionAgent(
        name="代码专家",
        llm=llm,
        custom_prompts=code_prompts,
        max_iterations=2
    )

    task2 = "编写一个Python函数，找出1到n之间所有的素数 (prime numbers)。"
    print(f"\n🎯 任务: {task2}")
    try:
        response = code_agent.run(task2)
        print(f"\n✅ 代码专家结果:\n{response}")
    except Exception as e:
        print(f"❌ 错误: {e}")

def demo_plan_solve_agent():
    """演示PlanAndSolveAgent - 分解规划与逐步执行"""
    print("\n" + "="*60)
    print("📋 PlanAndSolveAgent 演示 - 分解规划与逐步执行的Agent")
    print("="*60)

    # 创建LLM实例
    llm = HelloAgentsLLM()

    # 1. 默认配置演示
    print("\n--- 默认配置 ---")
    default_agent = PlanAndSolveAgent(
        name="通用助手",
        llm=llm
    )

    task1 = "如何学习Python编程？请制定一个详细的学习计划。"
    print(f"\n🎯 任务: {task1}")
    try:
        response = default_agent.run(task1)
        print(f"\n✅ 默认配置结果:\n{response}")
    except Exception as e:
        print(f"❌ 错误: {e}")

    # 2. 自定义配置演示 - 数学问题专家
    print("\n--- 自定义配置：数学问题专家 ---")
    math_prompts = {
        "planner": """
你是一个数学问题分解专家。请将以下数学问题分解为清晰的计算步骤。
每个步骤应该是一个具体的数学运算或逻辑推理。

数学问题: {question}

请按以下格式输出计算计划:
```python
["步骤1: 具体计算", "步骤2: 具体计算", ...]
```
""",
        "executor": """
你是一个数学计算专家。请严格按照计划执行数学计算。

# 原始问题: {question}
# 计算计划: {plan}
# 已完成的计算: {history}
# 当前计算步骤: {current_step}

请执行当前步骤的计算，只输出计算结果:
"""
    }

    math_agent = PlanAndSolveAgent(
        name="数学专家",
        llm=llm,
        custom_prompts=math_prompts
    )

    task2 = "一个水果店周一卖出了15个苹果。周二卖出的苹果数量是周一的两倍。周三卖出的数量比周二少了5个。请问这三天总共卖出了多少个苹果？"
    print(f"\n🎯 任务: {task2}")
    try:
        response = math_agent.run(task2)
        print(f"\n✅ 数学专家结果:\n{response}")
    except Exception as e:
        print(f"❌ 错误: {e}")

def demo_custom_vs_default():
    """演示自定义配置 vs 默认配置的对比"""
    print("\n" + "="*60)
    print("⚖️ 自定义配置 vs 默认配置对比演示")
    print("="*60)

    llm = HelloAgentsLLM()

    task = "设计一个简单的待办事项管理应用"

    # 默认配置
    print("\n--- 使用默认配置的ReflectionAgent ---")
    default_agent = ReflectionAgent(
        name="默认助手",
        llm=llm,
        max_iterations=1
    )

    print(f"🎯 任务: {task}")
    try:
        default_result = default_agent.run(task)
        print(f"\n✅ 默认配置结果:\n{default_result}")
    except Exception as e:
        print(f"❌ 错误: {e}")

    # 自定义配置 - 产品经理专家
    print("\n--- 使用自定义配置的产品经理专家 ---")
    product_prompts = {
        "initial": """
你是一位资深的产品经理。请分析以下产品需求：

需求: {task}

请提供详细的产品设计方案，包括：
1. 用户画像分析
2. 核心功能规划
3. 技术实现建议
4. 商业价值评估
""",
        "reflect": """
请审查以下产品设计方案的质量：

# 设计需求: {task}
# 设计方案: {content}

请从以下角度评估：
- 用户需求匹配度
- 功能完整性和逻辑性
- 技术实现可行性
- 市场竞争力

如果设计已经很好，请回答"无需改进"。
""",
        "refine": """
请根据评审意见优化产品设计方案：

# 原始需求: {task}
# 当前方案: {last_attempt}
# 评审意见: {feedback}

请提供优化后的设计方案。
"""
    }

    product_agent = ReflectionAgent(
        name="产品经理",
        llm=llm,
        custom_prompts=product_prompts,
        max_iterations=1
    )

    print(f"🎯 任务: {task}")
    try:
        custom_result = product_agent.run(task)
        print(f"\n✅ 产品经理专家结果:\n{custom_result}")
    except Exception as e:
        print(f"❌ 错误: {e}")

    print("\n--- 对比总结 ---")
    print("✅ 通过自定义提示词，Agent能够：")
    print("1. 更专业地处理特定领域的任务")
    print("2. 提供更有针对性的分析和建议")
    print("3. 适应不同的工作流程和输出格式")
    print("4. 体现专业角色的思维方式")

def interactive_demo():
    """交互式演示"""
    print("\n" + "="*60)
    print("🎮 交互式演示 - 选择Agent类型进行对话")
    print("="*60)

    # 创建LLM实例
    llm = HelloAgentsLLM()

    # 创建工具注册表（为ReAct Agent准备）
    tool_registry = ToolRegistry()
    tool_registry.register_function("search", "网页搜索工具", search)
    tool_registry.register_function("calculate", "数学计算工具", calculate)

    # 创建不同类型的Agent（展示默认配置的简洁性）
    agents = {
        "1": SimpleAgent("简单助手", llm, "你是一个有用的AI助手。"),
        "2": ReActAgent("工具助手", llm, tool_registry, max_steps=3),
        "3": ReflectionAgent("反思助手", llm, max_iterations=2),
        "4": PlanAndSolveAgent("规划助手", llm)
    }

    print("\n请选择Agent类型:")
    print("1. SimpleAgent - 基础对话")
    print("2. ReActAgent - 工具调用")
    print("3. ReflectionAgent - 反思改进")
    print("4. PlanAndSolveAgent - 规划执行")
    print("\n💡 提示：所有Agent都使用默认配置，展示开箱即用的效果")

    while True:
        choice = input("\n请输入选择 (1-4) 或 'quit' 退出: ").strip()

        if choice.lower() in ['quit', 'exit', '退出']:
            break

        if choice not in agents:
            print("❌ 无效选择，请输入1-4")
            continue

        agent = agents[choice]
        print(f"\n✅ 已选择: {agent.name}")

        while True:
            user_input = input(f"\n与{agent.name}对话 (输入'back'返回选择): ")

            if user_input.lower() == 'back':
                break

            try:
                response = agent.run(user_input)
                print(f"\n{agent.name}: {response}")
            except Exception as e:
                print(f"❌ 错误: {e}")

    print("\n👋 再见！")

def demo_advanced_features():
    """演示高级功能：工具链和异步执行"""
    print("\n" + "="*60)
    print("🚀 高级功能演示 - 工具链和异步执行")
    print("="*60)

    # 创建工具注册表
    registry = ToolRegistry()
    registry.register_function("calculate", "数学计算工具", calculate)

    # 1. 工具链演示
    print("\n--- 工具链演示 ---")

    # 创建简单的工具链
    chain = ToolChain("demo_chain", "演示工具链")
    chain.add_step("calculate", "2 + 3", "step1")
    chain.add_step("calculate", "5 * 2", "step2")

    # 创建工具链管理器
    chain_manager = ToolChainManager(registry)
    chain_manager.register_chain(chain)

    # 执行工具链
    print("🔗 执行工具链...")
    result = chain_manager.execute_chain("demo_chain", "开始")
    print(f"✅ 工具链结果: {result}")

    # 2. 异步执行演示
    print("\n--- 异步执行演示 ---")

    import asyncio

    async def async_demo():
        # 创建异步执行器
        executor = AsyncToolExecutor(registry, max_workers=2)

        # 定义并行任务
        tasks = [
            {"tool_name": "calculate", "input_data": "10 + 5"},
            {"tool_name": "calculate", "input_data": "20 * 3"},
            {"tool_name": "calculate", "input_data": "100 / 4"},
        ]

        print("⚡ 并行执行多个计算任务...")
        results = await executor.execute_tools_parallel(tasks)

        print("📊 并行执行结果:")
        for result in results:
            status = "✅" if result["status"] == "success" else "❌"
            print(f"{status} {result['input_data']} = {result['result']}")

        executor.close()

    # 运行异步演示
    try:
        asyncio.run(async_demo())
    except Exception as e:
        print(f"❌ 异步执行错误: {e}")

def main():
    """主函数"""
    print("🚀 HelloAgents 框架完整演示")
    print("基于OpenAI原生API的多智能体框架")
    print("\n🎯 演示内容：")
    print("1. 四种Agent范式的默认配置使用")
    print("2. 自定义配置的高级用法")
    print("3. 默认 vs 自定义配置的对比")
    print("4. 高级功能：工具链和异步执行")
    print("5. 交互式Agent体验")

    try:
        # 1. SimpleAgent演示
        # demo_simple_agent()
        #
        # # 2. ReActAgent演示（默认 + 自定义）
        # demo_react_agent()
        #
        # # 3. ReflectionAgent演示（默认 + 自定义）
        # demo_reflection_agent()
        #
        # # 4. PlanAndSolveAgent演示（默认 + 自定义）
        # demo_plan_solve_agent()

        # 5. 自定义 vs 默认配置对比
        demo_custom_vs_default()

        # 6. 高级功能演示
        # demo_advanced_features()

        # 7. 交互式演示
        # interactive_demo()

        print("\n" + "="*60)
        print("🎉 HelloAgents 框架演示完成！")
        print("="*60)
        print("\n📋 总结：")
        print("✅ 默认配置：开箱即用，简洁高效")
        print("✅ 自定义配置：专业定制，灵活强大")
        print("✅ 统一API：一致的使用体验")
        print("✅ 渐进式：从简单到复杂的学习路径")

    except KeyboardInterrupt:
        print("\n\n👋 用户中断，程序退出")
    except Exception as e:
        print(f"\n❌ 程序出错: {e}")

if __name__ == "__main__":
    main()