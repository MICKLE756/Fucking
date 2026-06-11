"""
对话处理主流程 - 基于动态工作流引擎

将原来的 if-else 链替换为图驱动的工作流:
  每个处理步骤是独立节点, 节点间的跳转由路由函数动态决定。
  新增/修改流程 = 增加节点 + 调整路由, 不改引擎代码。
"""

from workflow_nodes import PatentWorkflow


class DialogProcess:
    """对话处理主流程 (工作流引擎包装)"""

    def __init__(self):
        self.workflow = PatentWorkflow()

    def __call__(self, text: str) -> dict:
        return self.workflow(text)

    def reset(self):
        self.workflow.reset()

    def get_state(self) -> dict:
        return self.workflow.get_state()


if __name__ == "__main__":
    dp = DialogProcess()

    # 打印工作流图
    print("===== 工作流图 =====")
    print(dp.workflow.engine.describe())
    print()

    test_texts = [
        "我想找一种耐高温的不粘锅涂层材料相关的专利",
        "主要用于餐饮厨具，需要耐受400度以上高温",
        "还需要环保，不含PFOA",
        "确认，请继续检索",
        "是的，继续",
    ]

    for i, text in enumerate(test_texts, 1):
        print(f"\n{'='*50}")
        print(f"轮次 {i} | 用户: {text}")
        print("-" * 50)
        resp = dp(text)
        print(f"\nAgent: {resp.get('message', '')}")
        if resp.get("request"):
            print(f"操作提示: {resp['request']}")
        print(f"Trace: {dp.workflow.state.get('_trace', [])}")
        print(f"状态: {dp.get_state()}")
