"""异步工具执行器 - HelloAgents异步工具执行支持"""

import asyncio
import concurrent.futures
from typing import Dict, Any, List
from hello_agents.tools.registry import ToolRegistry


class AsyncToolExecutor:
    """异步工具执行器，基于线程池实现同步工具的并发调用"""

    def __init__(self, registry: ToolRegistry, max_workers: int = 4):
        self.registry = registry
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)

    async def execute_tool_async(self, tool_name: str, input_data: str) -> str:
        """异步执行单个工具，把同步工具丢进线程池运行"""
        loop = asyncio.get_event_loop()

        def _execute():
            # 执行注册表内的同步工具
            return self.registry.execute_tool(tool_name, input_data)

        try:
            result = await loop.run_in_executor(self.executor, _execute)
            return result
        except Exception as e:
            return f"❌ 工具 '{tool_name}' 异步执行失败: {e}"

    async def execute_tools_parallel(self, tasks: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        并行执行多个工具任务
        Args:
            tasks: 任务列表，每个任务包含 tool_name 和 input_data
        Returns:
            执行结果列表，携带状态与原始参数
        """
        print(f"🚀 开始并行执行 {len(tasks)} 个工具任务")

        # 预先创建全部协程
        async_tasks = []
        for i, task in enumerate(tasks):
            tool_name = task.get("tool_name")
            input_data = task.get("input_data", "")

            if not tool_name:
                continue

            print(f"📝 创建任务 {i + 1}: {tool_name}")
            coro = self.execute_tool_async(tool_name, input_data)
            async_tasks.append((i, task, coro))

        # 按顺序收集每一个任务结果
        results = []
        for i, task, coro in async_tasks:
            try:
                result = await coro
                results.append({
                    "task_id": i,
                    "tool_name": task["tool_name"],
                    "input_data": task["input_data"],
                    "result": result,
                    "status": "success"
                })
                print(f"✅ 任务 {i + 1} 完成: {task['tool_name']}")
            except Exception as e:
                results.append({
                    "task_id": i,
                    "tool_name": task["tool_name"],
                    "input_data": task["input_data"],
                    "result": str(e),
                    "status": "error"
                })
                print(f"❌ 任务 {i + 1} 失败: {task['tool_name']} - {e}")

        success_count = sum(1 for r in results if r["status"] == "success")
        print(f"🎉 并行执行完成，成功: {success_count}/{len(results)}")
        return results

    async def execute_tools_batch(self, tool_name: str, input_list: List[str]) -> List[Dict[str, Any]]:
        """批量调用同一个工具，多组输入并行运行"""
        tasks = [
            {"tool_name": tool_name, "input_data": input_data}
            for input_data in input_list
        ]
        return await self.execute_tools_parallel(tasks)

    def close(self):
        """关闭线程池，释放资源"""
        self.executor.shutdown(wait=True)
        print("🔒 异步工具执行器已关闭")

    # 修复：异步上下文管理器，支持 async with
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.close()


# ------------------- 便捷入口函数 -------------------
async def run_parallel_tools(registry: ToolRegistry, tasks: List[Dict[str, str]], max_workers: int = 4) -> List[Dict[str, Any]]:
    """便捷入口：并行执行多类不同工具"""
    async with AsyncToolExecutor(registry, max_workers) as executor:
        return await executor.execute_tools_parallel(tasks)


async def run_batch_tool(registry: ToolRegistry, tool_name: str, input_list: List[str], max_workers: int = 4) -> List[Dict[str, Any]]:
    """便捷入口：同一个工具多组输入批量并行执行"""
    async with AsyncToolExecutor(registry, max_workers) as executor:
        return await executor.execute_tools_batch(tool_name, input_list)


# ------------------- 同步封装，给同步代码调用 -------------------
def run_parallel_tools_sync(registry: ToolRegistry, tasks: List[Dict[str, str]], max_workers: int = 4) -> List[Dict[str, Any]]:
    """同步版本并行工具调用"""
    return asyncio.run(run_parallel_tools(registry, tasks, max_workers))


def run_batch_tool_sync(registry: ToolRegistry, tool_name: str, input_list: List[str], max_workers: int = 4) -> List[Dict[str, Any]]:
    """同步版本批量工具调用"""
    return asyncio.run(run_batch_tool(registry, tool_name, input_list, max_workers))


# ------------------- 测试Demo（可直接运行不报错） -------------------
async def demo_parallel_execution():
    """演示并行执行，自带临时测试，不会因为无工具崩溃"""
    registry = ToolRegistry()

    # 测试任务
    tasks = [
        {"tool_name": "my_calculator", "input_data": "2 + 2"},
        {"tool_name": "my_calculator", "input_data": "3 * 4"},
        {"tool_name": "my_calculator", "input_data": "sqrt(16)"},
        {"tool_name": "my_calculator", "input_data": "10 / 2"},
    ]

    results = await run_parallel_tools(registry, tasks)

    print("\n📊 并行执行结果:")
    for result in results:
        status_icon = "✅" if result["status"] == "success" else "❌"
        print(f"{status_icon} {result['tool_name']}({result['input_data']}) = {result['result']}")

    return results


if __name__ == "__main__":
    asyncio.run(demo_parallel_execution())