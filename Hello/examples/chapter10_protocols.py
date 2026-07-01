#!/usr/bin/env python3
"""
第十章：智能体通信协议完整教学示例

本文件整合了第十章中介绍的三种智能体通信协议的所有实用案例：

🔧 MCP (Model Context Protocol)
- 官方服务器使用案例（文件系统、GitHub等）
- 自定义服务器开发（天气查询服务器）
- 多种传输方式演示（Stdio、HTTP、SSE等）
- 在 HelloAgents 中的集成使用

🤝 A2A (Agent-to-Agent Protocol)
- 基于官方 a2a-sdk 的智能体创建
- 多智能体协作工作流（内容创作团队）
- 智能体间技能共享和调用
- 实际业务场景应用

🌐 ANP (Agent Network Protocol)
- 服务发现和注册机制
- 网络拓扑管理和监控
- 负载均衡和消息路由
- 大规模智能体网络管理

学习目标：
✅ 理解三种协议的核心概念和设计理念
✅ 掌握每种协议的实际使用方法和最佳实践
✅ 学会根据需求选择合适的协议
✅ 体验协议在实际项目中的应用效果
✅ 了解协议间的组合使用方式

运行方式：
python examples/chapter10_protocols_complete.py

依赖安装：
pip install fastmcp>=2.0.0 a2a-sdk

作者：HelloAgents 教学团队
更新：2024年12月
"""

import asyncio
import sys
import os
import json
import time
import random
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def print_header(title: str):
    """打印章节标题"""
    print("\n" + "=" * 70)
    print(f"📚 {title}")
    print("=" * 70)

def print_section(title: str):
    """打印小节标题"""
    print(f"\n📋 {title}")
    print("-" * 50)

def print_lesson(lesson_num: str, title: str):
    """打印课程标题"""
    print(f"\n🎓 课程 {lesson_num}: {title}")
    print("🔹" * 40)

def print_demo(title: str):
    """打印演示标题"""
    print(f"\n🚀 {title}")
    print("💡" * 30)

def print_success(message: str):
    """打印成功信息"""
    print(f"✅ {message}")

def print_info(message: str):
    """打印信息"""
    print(f"ℹ️  {message}")

def print_warning(message: str):
    """打印警告"""
    print(f"⚠️  {message}")

def print_error(message: str):
    """打印错误"""
    print(f"❌ {message}")

def wait_for_user(prompt: str = "按回车键继续..."):
    """等待用户输入"""
    input(f"\n{prompt}")

def show_course_overview():
    """显示课程概览"""
    print_header("第十章：智能体通信协议 - 课程概览")
    
    overview = """
🎯 课程目标
本章将带你深入了解智能体通信的三种核心协议，从基础概念到实际应用，
让你能够在实际项目中灵活运用这些协议构建强大的智能体系统。

📖 课程结构

第一部分：MCP (Model Context Protocol)
├── 1.1 MCP 基础概念和设计理念
├── 1.2 使用官方 MCP 服务器实战
├── 1.3 创建自定义 MCP 服务器
├── 1.4 多种传输方式详解
└── 1.5 在 HelloAgents 中集成 MCP

第二部分：A2A (Agent-to-Agent Protocol)  
├── 2.1 A2A 协议核心概念
├── 2.2 基于官方 SDK 创建智能体
├── 2.3 多智能体协作工作流
├── 2.4 智能体技能共享机制
└── 2.5 实际业务场景应用

第三部分：ANP (Agent Network Protocol)
├── 3.1 ANP 网络管理概念
├── 3.2 服务发现和注册
├── 3.3 网络拓扑和监控
├── 3.4 负载均衡和消息路由
└── 3.5 大规模网络管理

第四部分：协议对比和选择
├── 4.1 三种协议特性对比
├── 4.2 应用场景选择指南
├── 4.3 协议组合使用策略
└── 4.4 最佳实践和注意事项

🛠️ 实践项目
- 天气查询 MCP 服务器
- 内容创作 A2A 团队
- 智能体网络管理系统
- 多协议集成应用

💡 学习建议
1. 按顺序学习，每个部分都有前置知识依赖
2. 动手实践每个示例，理解协议的实际效果
3. 思考在自己项目中如何应用这些协议
4. 尝试修改示例代码，探索更多可能性
"""
    
    print(overview)
    wait_for_user("准备好开始学习了吗？")

def check_dependencies():
    """检查依赖安装情况"""
    print_section("环境检查")
    
    dependencies = {
        "fastmcp": "FastMCP 库（MCP 协议支持）",
        "a2a": "A2A SDK（A2A 协议支持）"
    }
    
    missing_deps = []
    
    for dep, desc in dependencies.items():
        try:
            if dep == "fastmcp":
                import fastmcp
                print_success(f"{desc} - 版本 {fastmcp.__version__}")
            elif dep == "a2a":
                from a2a.client import A2AClient
                print_success(f"{desc} - 已安装")
        except ImportError:
            print_warning(f"{desc} - 未安装")
            missing_deps.append(dep)
    
    if missing_deps:
        print_info("安装缺失的依赖：")
        for dep in missing_deps:
            if dep == "fastmcp":
                print(f"  pip install fastmcp>=2.0.0")
            elif dep == "a2a":
                print(f"  pip install a2a-sdk")
        
        choice = input("\n是否继续（某些功能可能不可用）？(y/n): ").lower()
        if choice != 'y':
            print("退出程序。请安装依赖后重新运行。")
            sys.exit(1)
    else:
        print_success("所有依赖已正确安装！")
    
    wait_for_user()

# ============================================================================
# 第一部分：MCP 协议教学
# ============================================================================

def lesson_1_1_mcp_concepts():
    """课程 1.1: MCP 基础概念"""
    print_lesson("1.1", "MCP 基础概念和设计理念")
    
    concepts = """
📚 什么是 MCP？
MCP (Model Context Protocol) 是由 Anthropic 开发的开放标准，用于在 AI 应用程序
和外部数据源之间建立安全、可控的连接。

🎯 核心概念：
1. 工具 (Tools): AI 可以调用的函数，类似于 API 端点
2. 资源 (Resources): 可以访问的数据源，如文件、数据库记录
3. 提示词 (Prompts): 预定义的提示词模板
4. 传输层 (Transport): 通信方式，支持 Stdio、HTTP、WebSocket 等

💡 设计理念：
- 标准化：统一的协议规范，确保兼容性
- 安全性：可控的访问权限和数据隔离
- 灵活性：支持多种传输方式和数据格式
- 可扩展性：易于添加新的工具和资源

🔧 在 HelloAgents 中的实现：
HelloAgents 基于 FastMCP 库提供完整的 MCP 协议支持，包括：
- 服务器端：创建自定义 MCP 服务器
- 客户端：连接和使用 MCP 服务器
- 工具集成：在 Agent 中使用 MCP 工具
- 多传输支持：Stdio、HTTP、SSE 等传输方式
"""
    
    print(concepts)
    wait_for_user()

def lesson_1_2_official_mcp_servers():
    """课程 1.2: 使用官方 MCP 服务器"""
    print_lesson("1.2", "使用官方 MCP 服务器实战")
    
    print_info("官方 MCP 服务器提供了丰富的功能，让我们来体验一下：")
    
    # 模拟官方服务器使用
    official_servers = {
        "filesystem": {
            "description": "文件系统操作服务器",
            "tools": ["list_directory", "read_file", "write_file"],
            "install": "npx @modelcontextprotocol/server-filesystem"
        },
        "github": {
            "description": "GitHub 仓库访问服务器", 
            "tools": ["search_repositories", "get_repository", "list_issues"],
            "install": "npx @modelcontextprotocol/server-github"
        },
        "memory": {
            "description": "内存存储服务器",
            "tools": ["store_memory", "retrieve_memory", "list_memories"],
            "install": "npx @modelcontextprotocol/server-memory"
        }
    }
    
    print("\n🗂️ 官方 MCP 服务器列表：")
    for name, info in official_servers.items():
        print(f"\n📦 {name.upper()} 服务器")
        print(f"   描述: {info['description']}")
        print(f"   工具: {', '.join(info['tools'])}")
        print(f"   安装: {info['install']}")
    
    print_demo("文件系统服务器演示")
    
    # 模拟文件系统操作
    demo_operations = [
        ("列出当前目录", "找到 15 个文件"),
        ("读取 README.md", "成功读取 1,234 字符"),
        ("创建测试文件", "成功创建 test.txt"),
        ("验证文件创建", "文件存在，大小 56 字节")
    ]
    
    for operation, result in demo_operations:
        print(f"🔧 {operation}...")
        time.sleep(0.5)
        print_success(result)
    
    wait_for_user()

def lesson_1_3_custom_mcp_server():
    """课程 1.3: 创建自定义 MCP 服务器"""
    print_lesson("1.3", "创建自定义 MCP 服务器")
    
    print_info("让我们创建一个天气查询 MCP 服务器作为学习案例：")
    
    # 展示服务器代码结构
    server_code = '''
from fastmcp import FastMCP
from typing import Dict, Any

# 创建服务器实例
weather_server = FastMCP("weather-server")

@weather_server.tool()
def get_weather(city: str) -> Dict[str, Any]:
    """获取指定城市的天气信息"""
    # 模拟天气数据
    weather_data = {
        "city": city,
        "temperature": 22,
        "humidity": 65,
        "condition": "晴朗",
        "query_time": "2024-12-05 14:30:00"
    }
    return weather_data

@weather_server.tool()
def get_weather_forecast(city: str, days: int = 3) -> Dict[str, Any]:
    """获取天气预报"""
    forecast = []
    for i in range(days):
        forecast.append({
            "date": f"2024-12-{6+i:02d}",
            "temperature_high": 25 + i,
            "temperature_low": 15 + i,
            "condition": "多云"
        })
    
    return {
        "city": city,
        "forecast": forecast
    }

if __name__ == "__main__":
    weather_server.run()
'''
    
    print("\n💻 天气服务器代码示例：")
    print("```python")
    print(server_code)
    print("```")
    
    print_demo("天气服务器功能演示")
    
    # 模拟服务器功能
    demo_calls = [
        ("get_weather", {"city": "北京"}, {"city": "北京", "temperature": 15, "condition": "晴朗"}),
        ("get_weather_forecast", {"city": "上海", "days": 3}, {"city": "上海", "forecast_days": 3}),
        ("list_cities", {}, {"supported_cities": ["北京", "上海", "广州"], "total_count": 10})
    ]
    
    for tool_name, args, result in demo_calls:
        print(f"\n🔧 调用工具: {tool_name}")
        print(f"   参数: {args}")
        time.sleep(0.3)
        print_success(f"返回: {result}")
    
    wait_for_user()

def lesson_1_4_transport_methods():
    """课程 1.4: 多种传输方式详解"""
    print_lesson("1.4", "MCP 传输方式详解")
    
    transport_info = """
🚀 MCP 支持多种传输方式，适应不同的使用场景：

1️⃣ Stdio 传输（默认）
   - 通过标准输入输出通信
   - 适用于本地进程
   - 使用场景：开发测试、本地工具集成
   
   客户端连接：
   client = MCPClient("weather_server.py")

2️⃣ HTTP 传输
   - 通过 HTTP 协议通信
   - 适用于远程服务
   - 使用场景：Web 服务、API 集成
   
   服务器启动：
   server.run(transport="http", host="0.0.0.0", port=8000)
   
   客户端连接：
   client = MCPClient("http://localhost:8000")

3️⃣ SSE 传输
   - 通过 Server-Sent Events 实时通信
   - 适用于需要实时更新的场景
   - 使用场景：实时监控、流式数据
   
   客户端连接：
   client = MCPClient("http://localhost:8000", transport_type="sse")

4️⃣ 内存传输
   - 直接在内存中通信
   - 适用于测试和开发
   - 使用场景：单元测试、快速原型
   
   客户端连接：
   server_instance = FastMCP("memory-server")
   client = MCPClient(server_instance)

💡 传输方式选择建议：
- 开发阶段：使用 Stdio 或内存传输
- 生产环境：使用 HTTP 传输
- 实时应用：使用 SSE 传输
- 测试场景：使用内存传输
"""
    
    print(transport_info)
    wait_for_user()

def lesson_1_5_mcp_in_helloagents():
    """课程 1.5: 在 HelloAgents 中集成 MCP"""
    print_lesson("1.5", "在 HelloAgents 中集成 MCP")
    
    integration_code = '''
from hello_agents.agents.simple_agent import SimpleAgent
from hello_agents.core.llm import HelloAgentsLLM
from hello_agents.tools.builtin.protocol_tools import MCPTool

# 创建智能体
llm = HelloAgentsLLM()
agent = SimpleAgent(name="天气助手", llm=llm)

# 添加 MCP 工具
weather_tool = MCPTool(
    server_command=["python", "weather_server.py"],
    name="天气查询工具"
)
agent.add_tool(weather_tool)

# 使用智能体
response = agent.run("北京今天天气怎么样？")
print(response)
'''
    
    print_info("HelloAgents 提供了 MCPTool 来轻松集成 MCP 服务器：")
    print("\n💻 集成代码示例：")
    print("```python")
    print(integration_code)
    print("```")
    
    print_demo("智能体使用 MCP 工具演示")
    
    # 模拟智能体对话
    conversations = [
        ("用户", "北京今天天气怎么样？"),
        ("助手", "我来为您查询北京的天气信息..."),
        ("系统", "调用 MCP 工具: get_weather(city='北京')"),
        ("助手", "根据查询结果，北京今天天气晴朗，温度15°C，湿度60%。")
    ]
    
    for speaker, message in conversations:
        if speaker == "用户":
            print(f"\n👤 {speaker}: {message}")
        elif speaker == "助手":
            print(f"🤖 {speaker}: {message}")
        else:
            print(f"⚙️  {speaker}: {message}")
        time.sleep(0.8)
    
    wait_for_user()

# ============================================================================
# 第二部分：A2A 协议教学
# ============================================================================

def lesson_2_1_a2a_concepts():
    """课程 2.1: A2A 协议核心概念"""
    print_lesson("2.1", "A2A 协议核心概念")
    
    concepts = """
🤝 什么是 A2A？
A2A (Agent-to-Agent Protocol) 是一个用于智能体间直接通信和协作的协议。
它允许多个智能体相互发现、通信，并协作完成复杂任务。

🎯 核心概念：
1. 智能体 (Agent): 具有特定技能的独立实体
2. 技能 (Skill): 智能体可以执行的功能
3. 消息 (Message): 智能体间的通信载体
4. 协作 (Collaboration): 多个智能体共同完成任务

💡 设计理念：
- 去中心化：智能体间直接通信，无需中央协调器
- 技能共享：智能体可以调用其他智能体的技能
- 动态发现：智能体可以动态发现和连接其他智能体
- 协作工作流：支持复杂的多步骤协作流程

🔧 在 HelloAgents 中的实现：
HelloAgents 基于官方 a2a-sdk 提供 A2A 协议支持：
- 智能体创建：使用 A2AServer 创建智能体
- 技能定义：使用装饰器定义智能体技能
- 消息通信：支持结构化消息传递
- 工作流编排：支持复杂的协作流程
"""
    
    print(concepts)
    wait_for_user()

def lesson_2_2_create_a2a_agents():
    """课程 2.2: 基于官方 SDK 创建智能体"""
    print_lesson("2.2", "基于官方 SDK 创建智能体")
    
    agent_code = '''
from hello_agents.protocols.a2a.implementation import A2AServer

# 创建智能体
calculator = A2AServer(
    name="calculator-agent",
    description="专业的数学计算智能体",
    capabilities={"math": ["add", "multiply", "divide"]}
)

@calculator.skill("add")
def add_numbers(query: str) -> str:
    """加法计算"""
    # 解析查询并执行计算
    if "+" in query:
        parts = query.split("+")
        numbers = [float(x.strip()) for x in parts]
        result = sum(numbers)
        return f"计算结果: {' + '.join(map(str, numbers))} = {result}"
    return "请使用格式: 5 + 3"

@calculator.skill("multiply")
def multiply_numbers(query: str) -> str:
    """乘法计算"""
    if "*" in query:
        parts = query.split("*")
        numbers = [float(x.strip()) for x in parts]
        result = 1
        for num in numbers:
            result *= num
        return f"计算结果: {' × '.join(map(str, numbers))} = {result}"
    return "请使用格式: 5 * 3"

# 测试智能体技能
print("测试计算器智能体:")
print(calculator.skills["add"]("10 + 5"))
print(calculator.skills["multiply"]("6 * 7"))
'''
    
    print_info("让我们创建一个计算器智能体作为学习案例：")
    print("\n💻 智能体创建代码：")
    print("```python")
    print(agent_code)
    print("```")
    
    print_demo("计算器智能体演示")
    
    # 模拟智能体技能测试
    test_cases = [
        ("add", "10 + 5", "计算结果: 10 + 5 = 15"),
        ("multiply", "6 * 7", "计算结果: 6 × 7 = 42"),
        ("add", "1 + 2 + 3", "计算结果: 1 + 2 + 3 = 6")
    ]
    
    for skill, query, expected in test_cases:
        print(f"\n🔧 技能: {skill}")
        print(f"   查询: {query}")
        time.sleep(0.3)
        print_success(f"结果: {expected}")
    
    wait_for_user()

def lesson_2_3_multi_agent_collaboration():
    """课程 2.3: 多智能体协作工作流"""
    print_lesson("2.3", "多智能体协作工作流")

    print_info("让我们创建一个内容创作团队，展示多智能体协作：")

    workflow_description = """
📝 内容创作团队工作流：

1️⃣ 研究员智能体 (Researcher)
   - 负责主题研究和数据收集
   - 技能：research_topic, fact_check
   - 输出：结构化的研究报告

2️⃣ 撰写员智能体 (Writer)
   - 基于研究数据创作文章
   - 技能：write_article, create_summary
   - 输出：高质量的文章内容

3️⃣ 编辑智能体 (Editor)
   - 内容优化和质量控制
   - 技能：edit_content, final_review
   - 输出：经过优化的最终内容

🔄 协作流程：
研究员 → 撰写员 → 编辑 → 最终内容
"""

    print(workflow_description)

    print_demo("内容创作团队协作演示")

    # 模拟协作流程
    workflow_steps = [
        ("研究员", "开始研究主题：人工智能在教育中的应用", "生成研究报告（4个关键发现）"),
        ("撰写员", "基于研究报告创作文章", "完成文章初稿（1,500字）"),
        ("编辑", "优化文章内容和结构", "完成内容编辑（质量评分：89/100）"),
        ("编辑", "进行最终审核", "批准发布（状态：已通过）")
    ]

    for agent, action, result in workflow_steps:
        print(f"\n👤 {agent}: {action}")
        time.sleep(0.8)
        print_success(result)

    print_info("\n🎉 协作完成！团队成功创作了一篇高质量的文章。")
    wait_for_user()

def lesson_2_4_skill_sharing():
    """课程 2.4: 智能体技能共享机制"""
    print_lesson("2.4", "智能体技能共享机制")

    sharing_concept = """
🔗 技能共享机制：

A2A 协议允许智能体之间共享和调用彼此的技能，实现能力的组合和扩展。

💡 共享方式：
1. 直接调用：智能体A直接调用智能体B的技能
2. 技能发现：智能体可以查询其他智能体的可用技能
3. 能力组合：多个智能体的技能组合完成复杂任务
4. 动态协作：根据任务需求动态选择合适的智能体

🎯 应用场景：
- 专业分工：不同智能体负责不同专业领域
- 能力互补：弥补单个智能体的能力不足
- 负载分担：将复杂任务分解给多个智能体
- 知识共享：共享专业知识和经验
"""

    print(sharing_concept)

    print_demo("技能共享演示")

    # 模拟技能共享场景
    sharing_scenario = [
        ("翻译智能体", "提供多语言翻译技能", "支持中英日韩等10种语言"),
        ("分析智能体", "调用翻译技能处理多语言数据", "成功分析5种语言的文档"),
        ("报告智能体", "调用分析和翻译技能", "生成多语言分析报告"),
        ("协调智能体", "整合所有智能体的输出", "完成综合性多语言项目")
    ]

    for agent, action, result in sharing_scenario:
        print(f"\n🤖 {agent}: {action}")
        time.sleep(0.6)
        print_success(result)

    wait_for_user()

def lesson_2_5_business_scenarios():
    """课程 2.5: 实际业务场景应用"""
    print_lesson("2.5", "A2A 在实际业务场景中的应用")

    business_scenarios = """
🏢 实际业务场景应用：

1️⃣ 客服系统
   智能体角色：接待员、专家、主管
   协作方式：问题分流、专业解答、升级处理

2️⃣ 代码审查
   智能体角色：开发者、审查员、测试员
   协作方式：代码提交、质量检查、测试验证

3️⃣ 内容创作
   智能体角色：研究员、撰写员、编辑
   协作方式：研究→撰写→编辑→发布

4️⃣ 数据分析
   智能体角色：收集员、分析师、报告员
   协作方式：数据收集→分析处理→报告生成

5️⃣ 教学系统
   智能体角色：讲师、助教、评估员
   协作方式：课程设计→教学辅导→学习评估

💼 商业价值：
- 提高效率：自动化复杂的多步骤流程
- 保证质量：多重检查和专业分工
- 降低成本：减少人工干预和错误
- 增强灵活性：动态调整工作流程
"""

    print(business_scenarios)
    wait_for_user()

# ============================================================================
# 第三部分：ANP 协议教学
# ============================================================================

def lesson_3_1_anp_concepts():
    """课程 3.1: ANP 网络管理概念"""
    print_lesson("3.1", "ANP 网络管理概念")

    concepts = """
🌐 什么是 ANP？
ANP (Agent Network Protocol) 是一个用于管理大规模智能体网络的协议。
它提供服务发现、网络管理、负载均衡等功能。

🎯 核心概念：
1. 服务发现 (Service Discovery): 自动发现网络中的智能体服务
2. 网络管理 (Network Management): 管理智能体网络的拓扑和状态
3. 负载均衡 (Load Balancing): 在多个智能体间分配请求
4. 消息路由 (Message Routing): 智能体间的消息传递和路由

💡 设计理念：
- 可扩展性：支持大规模智能体网络
- 高可用性：提供故障检测和恢复机制
- 动态性：支持智能体的动态加入和离开
- 透明性：对上层应用透明的网络管理

🔧 在 HelloAgents 中的实现：
HelloAgents 提供了 ANP 的概念性实现：
- ANPNetwork: 网络管理器
- ANPDiscovery: 服务发现组件
- ServiceInfo: 服务信息描述
- 负载均衡和消息路由功能
"""

    print(concepts)
    wait_for_user()

def lesson_3_2_service_discovery():
    """课程 3.2: 服务发现和注册"""
    print_lesson("3.2", "服务发现和注册")

    discovery_code = '''
from hello_agents.protocols.anp.implementation import ANPDiscovery, ServiceInfo

# 创建服务发现组件
discovery = ANPDiscovery()

# 注册服务
weather_service = ServiceInfo(
    service_id="weather-service",
    service_type="weather",
    endpoint="http://localhost:8001",
    capabilities=["weather_query", "forecast"],
    metadata={"region": "china", "accuracy": "high"}
)

translation_service = ServiceInfo(
    service_id="translation-service",
    service_type="translation",
    endpoint="http://localhost:8002",
    capabilities=["translate", "detect_language"],
    metadata={"languages": ["zh", "en", "ja"]}
)

# 注册服务
discovery.register_service(weather_service)
discovery.register_service(translation_service)

# 服务发现
weather_services = discovery.find_services_by_type("weather")
translation_services = discovery.find_services_by_capability("translate")
'''

    print_info("ANP 提供了强大的服务发现机制：")
    print("\n💻 服务发现代码示例：")
    print("```python")
    print(discovery_code)
    print("```")

    print_demo("服务发现演示")

    # 模拟服务注册和发现
    services = [
        ("weather-service", "weather", ["weather_query", "forecast"]),
        ("translation-service", "translation", ["translate", "detect_language"]),
        ("analysis-service", "analysis", ["data_analysis", "report_generation"])
    ]

    print("\n📋 注册服务：")
    for service_id, service_type, capabilities in services:
        print(f"✅ {service_id} ({service_type}): {', '.join(capabilities)}")

    print("\n🔍 服务发现测试：")
    discovery_tests = [
        ("按类型查找", "weather", ["weather-service"]),
        ("按能力查找", "translate", ["translation-service"]),
        ("按类型查找", "analysis", ["analysis-service"])
    ]

    for test_type, query, results in discovery_tests:
        print(f"\n🔧 {test_type}: {query}")
        time.sleep(0.3)
        print_success(f"找到服务: {', '.join(results)}")

    wait_for_user()

def lesson_3_3_network_monitoring():
    """课程 3.3: 网络拓扑和监控"""
    print_lesson("3.3", "网络拓扑和监控")

    monitoring_info = """
📊 网络监控功能：

ANP 提供了全面的网络监控和管理功能：

1️⃣ 网络状态监控
   - 活跃智能体数量
   - 消息传递统计
   - 网络健康状态
   - 性能指标

2️⃣ 拓扑管理
   - 智能体连接关系
   - 网络拓扑图
   - 路径优化
   - 故障检测

3️⃣ 性能监控
   - 响应时间统计
   - 吞吐量监控
   - 错误率统计
   - 资源使用情况

4️⃣ 健康检查
   - 智能体存活检测
   - 服务可用性检查
   - 自动故障恢复
   - 负载监控
"""

    print(monitoring_info)

    print_demo("网络监控演示")

    # 模拟网络状态
    network_stats = {
        "network_id": "production-network",
        "active_agents": 15,
        "total_messages": 1247,
        "health_status": "健康",
        "average_response_time": "45ms",
        "error_rate": "0.2%"
    }

    print("\n📊 网络状态报告：")
    for key, value in network_stats.items():
        print(f"   {key.replace('_', ' ').title()}: {value}")

    # 模拟智能体状态
    agent_status = [
        ("weather-agent-01", "在线", "正常", "12ms"),
        ("translation-agent-02", "在线", "正常", "8ms"),
        ("analysis-agent-03", "离线", "维护中", "N/A"),
        ("report-agent-04", "在线", "高负载", "67ms")
    ]

    print("\n🤖 智能体状态：")
    for agent, status, health, response_time in agent_status:
        status_icon = "🟢" if status == "在线" else "🔴"
        print(f"   {status_icon} {agent}: {status} | {health} | {response_time}")

    wait_for_user()

def lesson_3_4_load_balancing():
    """课程 3.4: 负载均衡和消息路由"""
    print_lesson("3.4", "负载均衡和消息路由")

    balancing_info = """
⚖️ 负载均衡策略：

ANP 支持多种负载均衡算法：

1️⃣ 轮询 (Round Robin)
   - 按顺序分配请求
   - 适用于处理能力相同的智能体

2️⃣ 加权轮询 (Weighted Round Robin)
   - 根据智能体能力分配权重
   - 适用于处理能力不同的智能体

3️⃣ 最少连接 (Least Connections)
   - 选择当前连接数最少的智能体
   - 适用于长连接场景

4️⃣ 响应时间 (Response Time)
   - 选择响应时间最短的智能体
   - 适用于对延迟敏感的应用

🔀 消息路由：
- 点对点路由：直接发送到目标智能体
- 广播路由：发送到所有相关智能体
- 组播路由：发送到特定组的智能体
- 智能路由：根据内容选择最佳路径
"""

    print(balancing_info)

    print_demo("负载均衡演示")

    # 模拟负载均衡
    agents = [
        ("agent-01", 3, "12ms"),
        ("agent-02", 1, "8ms"),
        ("agent-03", 5, "15ms"),
        ("agent-04", 2, "10ms")
    ]

    print("\n⚖️ 智能体负载状态：")
    for agent, load, response_time in agents:
        load_bar = "█" * load + "░" * (5 - load)
        print(f"   {agent}: [{load_bar}] {load}/5 | {response_time}")

    print("\n🔀 负载均衡决策：")
    balancing_decisions = [
        ("请求 1", "选择 agent-02（负载最低）"),
        ("请求 2", "选择 agent-04（响应时间优）"),
        ("请求 3", "选择 agent-01（轮询策略）"),
        ("请求 4", "选择 agent-02（负载均衡）")
    ]

    for request, decision in balancing_decisions:
        print(f"   {request}: {decision}")
        time.sleep(0.4)

    wait_for_user()

def lesson_3_5_large_scale_management():
    """课程 3.5: 大规模智能体网络管理"""
    print_lesson("3.5", "大规模智能体网络管理")

    scale_info = """
🏗️ 大规模网络管理挑战：

当智能体网络规模扩大时，面临的主要挑战：

1️⃣ 可扩展性挑战
   - 服务发现的性能瓶颈
   - 网络拓扑的复杂性
   - 消息路由的效率
   - 状态同步的开销

2️⃣ 解决方案
   - 分层网络架构
   - 分布式服务发现
   - 智能路由算法
   - 缓存和预取机制

3️⃣ 最佳实践
   - 网络分区管理
   - 就近服务选择
   - 异步消息处理
   - 故障隔离机制

4️⃣ 监控和运维
   - 实时性能监控
   - 自动扩缩容
   - 故障自动恢复
   - 容量规划
"""

    print(scale_info)

    print_demo("大规模网络管理演示")

    # 模拟大规模网络
    network_regions = [
        ("华北区", 45, "正常"),
        ("华东区", 67, "正常"),
        ("华南区", 38, "高负载"),
        ("西南区", 23, "正常"),
        ("海外区", 15, "网络延迟")
    ]

    print("\n🌐 网络区域状态：")
    total_agents = 0
    for region, agents, status in network_regions:
        status_icon = "🟢" if status == "正常" else "🟡" if "高负载" in status else "🔴"
        print(f"   {status_icon} {region}: {agents} 个智能体 | {status}")
        total_agents += agents

    print(f"\n📊 网络总览：")
    print(f"   总智能体数: {total_agents}")
    print(f"   活跃区域: 5")
    print(f"   总体健康度: 85%")

    wait_for_user()

# ============================================================================
# 第四部分：协议对比和选择
# ============================================================================

def lesson_4_1_protocol_comparison():
    """课程 4.1: 三种协议特性对比"""
    print_lesson("4.1", "三种协议特性对比")

    comparison_table = """
📊 协议特性对比表：

┌─────────────┬──────────────────┬──────────────────┬──────────────────┐
│    特性     │       MCP        │       A2A        │       ANP        │
├─────────────┼──────────────────┼──────────────────┼──────────────────┤
│  主要用途   │ 工具调用、资源访问 │ 智能体间通信协作  │ 网络管理、服务发现 │
│  通信模式   │ 客户端-服务器     │ 点对点、多对多    │ 网络拓扑管理      │
│  适用规模   │ 单一工具集成      │ 小到中型团队协作  │ 大规模分布式网络  │
│  实现复杂度 │ 简单             │ 中等             │ 复杂             │
│  标准化程度 │ 高（官方协议）    │ 中等（社区标准）  │ 低（概念性实现）  │
│  学习难度   │ 容易             │ 中等             │ 困难             │
│  开发效率   │ 高               │ 中等             │ 低               │
│  运维复杂度 │ 低               │ 中等             │ 高               │
└─────────────┴──────────────────┴──────────────────┴──────────────────┘

🎯 选择建议：

选择 MCP 当你需要：
✅ 集成外部工具和服务
✅ 访问文件系统、数据库等资源
✅ 使用标准化的工具调用接口
✅ 快速开发和部署

选择 A2A 当你需要：
✅ 多个智能体协作完成复杂任务
✅ 实现工作流自动化
✅ 智能体间技能共享和组合
✅ 中等规模的协作系统

选择 ANP 当你需要：
✅ 管理大规模智能体网络
✅ 实现服务发现和负载均衡
✅ 构建分布式智能体系统
✅ 企业级智能体平台
"""

    print(comparison_table)
    wait_for_user()

def lesson_4_2_scenario_selection():
    """课程 4.2: 应用场景选择指南"""
    print_lesson("4.2", "应用场景选择指南")

    scenarios = """
🎯 应用场景选择指南：

1️⃣ 个人助手应用
   推荐协议: MCP
   理由: 主要需要调用外部工具（日历、邮件、文件等）
   示例: 智能文档助手、代码分析工具

2️⃣ 团队协作系统
   推荐协议: A2A
   理由: 需要多个专业智能体协作完成任务
   示例: 内容创作团队、代码审查系统

3️⃣ 企业级智能体平台
   推荐协议: ANP + MCP + A2A
   理由: 需要全面的网络管理和多种通信方式
   示例: 智慧城市管理、大型客服系统

4️⃣ 物联网智能体网络
   推荐协议: ANP
   理由: 需要管理大量分布式智能体设备
   示例: 智能家居网络、工业物联网

5️⃣ 教育智能体系统
   推荐协议: A2A + MCP
   理由: 需要智能体协作和工具调用
   示例: 个性化学习系统、智能辅导平台

💡 选择原则：
- 简单优先：能用简单协议解决的不用复杂协议
- 需求导向：根据实际需求选择合适的协议
- 渐进式：从简单协议开始，逐步增加复杂性
- 组合使用：不同协议可以组合使用
"""

    print(scenarios)
    wait_for_user()

def lesson_4_3_combination_strategies():
    """课程 4.3: 协议组合使用策略"""
    print_lesson("4.3", "协议组合使用策略")

    combination_code = '''
# 多协议集成示例
from hello_agents.agents.simple_agent import SimpleAgent
from hello_agents.core.llm import HelloAgentsLLM
from hello_agents.tools.builtin.protocol_tools import MCPTool, A2ATool

# 创建智能体
agent = SimpleAgent(name="多协议助手", llm=HelloAgentsLLM())

# 添加 MCP 工具（外部服务访问）
weather_tool = MCPTool(
    server_command=["python", "weather_server.py"],
    name="天气查询工具"
)
agent.add_tool(weather_tool)

file_tool = MCPTool(
    server_command=["npx", "@modelcontextprotocol/server-filesystem", "."],
    name="文件系统工具"
)
agent.add_tool(file_tool)

# 添加 A2A 工具（智能体协作）
research_agent = A2ATool(
    agent_endpoint="http://localhost:8001",
    name="研究智能体"
)
agent.add_tool(research_agent)

analysis_agent = A2ATool(
    agent_endpoint="http://localhost:8002",
    name="分析智能体"
)
agent.add_tool(analysis_agent)

# ANP 在后台管理网络（透明）
# 智能体可以同时使用多种协议
response = agent.run("""
请帮我完成以下任务：
1. 查询北京的天气情况
2. 将结果保存到文件
3. 与研究智能体协作分析天气趋势
4. 生成综合报告
""")
'''

    print_info("在实际项目中，我们经常需要组合使用多种协议：")
    print("\n💻 多协议集成代码示例：")
    print("```python")
    print(combination_code)
    print("```")

    print_demo("多协议协作演示")

    # 模拟多协议协作流程
    collaboration_steps = [
        ("MCP", "查询天气信息", "获取北京天气：晴朗，15°C"),
        ("MCP", "保存到文件", "成功保存到 weather_data.json"),
        ("A2A", "请求研究智能体分析", "分析完成：温度趋势上升"),
        ("A2A", "请求分析智能体处理", "生成趋势图表"),
        ("ANP", "负载均衡选择服务", "选择最优的报告生成服务"),
        ("集成", "生成综合报告", "完成多协议协作任务")
    ]

    for protocol, action, result in collaboration_steps:
        print(f"\n🔧 [{protocol}] {action}")
        time.sleep(0.6)
        print_success(result)

    wait_for_user()

def lesson_4_4_best_practices():
    """课程 4.4: 最佳实践和注意事项"""
    print_lesson("4.4", "最佳实践和注意事项")

    best_practices = """
🏆 最佳实践指南：

1️⃣ 设计原则
   ✅ 单一职责：每个协议专注于特定功能
   ✅ 松耦合：协议间保持独立，减少依赖
   ✅ 可扩展：设计时考虑未来的扩展需求
   ✅ 容错性：实现优雅的错误处理和恢复

2️⃣ 开发建议
   ✅ 渐进式开发：从简单协议开始，逐步增加复杂性
   ✅ 充分测试：每个协议都要有完整的测试覆盖
   ✅ 文档完善：提供清晰的API文档和使用示例
   ✅ 版本管理：协议升级时保持向后兼容

3️⃣ 运维要点
   ✅ 监控告警：实时监控协议的运行状态
   ✅ 性能优化：定期分析和优化性能瓶颈
   ✅ 安全防护：实现适当的认证和授权机制
   ✅ 备份恢复：制定完善的备份和恢复策略

4️⃣ 常见陷阱
   ❌ 过度设计：不要为了使用协议而使用协议
   ❌ 协议混用：避免在同一层次混用多种协议
   ❌ 忽略性能：协议通信的性能开销不可忽视
   ❌ 缺乏监控：没有监控就无法发现和解决问题

💡 成功要素：
- 明确需求：清楚了解要解决的问题
- 合理选择：选择最适合的协议组合
- 充分测试：确保系统的稳定性和可靠性
- 持续优化：根据实际使用情况不断改进
"""

    print(best_practices)
    wait_for_user()

# ============================================================================
# 主程序和菜单系统
# ============================================================================

def show_main_menu():
    """显示主菜单"""
    menu = """
🎓 第十章：智能体通信协议 - 学习菜单

请选择要学习的内容：

第一部分：MCP (Model Context Protocol)
  1.1 - MCP 基础概念和设计理念
  1.2 - 使用官方 MCP 服务器实战
  1.3 - 创建自定义 MCP 服务器
  1.4 - 多种传输方式详解
  1.5 - 在 HelloAgents 中集成 MCP

第二部分：A2A (Agent-to-Agent Protocol)
  2.1 - A2A 协议核心概念
  2.2 - 基于官方 SDK 创建智能体
  2.3 - 多智能体协作工作流
  2.4 - 智能体技能共享机制
  2.5 - 实际业务场景应用

第三部分：ANP (Agent Network Protocol)
  3.1 - ANP 网络管理概念
  3.2 - 服务发现和注册
  3.3 - 网络拓扑和监控
  3.4 - 负载均衡和消息路由
  3.5 - 大规模智能体网络管理

第四部分：协议对比和选择
  4.1 - 三种协议特性对比
  4.2 - 应用场景选择指南
  4.3 - 协议组合使用策略
  4.4 - 最佳实践和注意事项

特殊选项：
  0   - 课程概览
  all - 完整学习（按顺序学习所有内容）
  q   - 退出程序

请输入选项（如 1.1, 2.3, all 等）："""

    return menu

def main():
    """主程序"""
    print_header("欢迎学习第十章：智能体通信协议")

    # 检查依赖
    check_dependencies()

    # 课程映射
    lessons = {
        "0": show_course_overview,
        "1.1": lesson_1_1_mcp_concepts,
        "1.2": lesson_1_2_official_mcp_servers,
        "1.3": lesson_1_3_custom_mcp_server,
        "1.4": lesson_1_4_transport_methods,
        "1.5": lesson_1_5_mcp_in_helloagents,
        "2.1": lesson_2_1_a2a_concepts,
        "2.2": lesson_2_2_create_a2a_agents,
        "2.3": lesson_2_3_multi_agent_collaboration,
        "2.4": lesson_2_4_skill_sharing,
        "2.5": lesson_2_5_business_scenarios,
        "3.1": lesson_3_1_anp_concepts,
        "3.2": lesson_3_2_service_discovery,
        "3.3": lesson_3_3_network_monitoring,
        "3.4": lesson_3_4_load_balancing,
        "3.5": lesson_3_5_large_scale_management,
        "4.1": lesson_4_1_protocol_comparison,
        "4.2": lesson_4_2_scenario_selection,
        "4.3": lesson_4_3_combination_strategies,
        "4.4": lesson_4_4_best_practices
    }

    while True:
        print(show_main_menu())
        choice = input().strip().lower()

        if choice == 'q':
            print("\n👋 感谢学习第十章：智能体通信协议！")
            print("🎉 希望这些知识对你的项目有所帮助。")
            break
        elif choice == 'all':
            print_info("开始完整学习模式...")
            # 按顺序执行所有课程
            lesson_order = [
                "0", "1.1", "1.2", "1.3", "1.4", "1.5",
                "2.1", "2.2", "2.3", "2.4", "2.5",
                "3.1", "3.2", "3.3", "3.4", "3.5",
                "4.1", "4.2", "4.3", "4.4"
            ]
            for lesson_id in lesson_order:
                if lesson_id in lessons:
                    lessons[lesson_id]()
            print_success("🎓 恭喜！你已完成第十章的全部学习内容！")
        elif choice in lessons:
            lessons[choice]()
        else:
            print_error("无效选项，请重新选择。")

if __name__ == "__main__":
    main()