"""
动态工作流引擎

核心概念:
  - Node:   可执行的处理步骤, 签名 fn(state: dict) -> dict
  - Router: 节点执行后的路由函数, 签名 fn(state: dict) -> str (下一个节点名)
  - State:  共享状态字典, 所有节点读写同一份

执行流程:
  entry → node.execute → router → next_node → node.execute → ... → __END__
"""


class WorkflowEngine:
    END = "__END__"

    def __init__(self, max_steps: int = 20):
        self._nodes: dict[str, callable] = {}
        self._routers: dict[str, callable] = {}
        self._entry: str | None = None
        self._max_steps = max_steps

    # ---- 注册 API ----

    def add_node(self, name: str, fn: callable):
        """注册节点: fn(state) -> state"""
        self._nodes[name] = fn
        return self

    def add_router(self, after_node: str, fn: callable):
        """注册路由: after_node 执行完后调用 fn(state) -> 下一个节点名"""
        self._routers[after_node] = fn
        return self

    def set_entry(self, name: str):
        """设置入口节点"""
        self._entry = name
        return self

    # ---- 执行 ----

    def run(self, state: dict) -> dict:
        """执行工作流, 返回最终 state"""
        current = self._entry
        step = 0
        trace = []

        while current and current != self.END and step < self._max_steps:
            node_fn = self._nodes.get(current)
            if not node_fn:
                raise ValueError(f"未注册的节点: {current}")

            print(f"  [workflow] step {step}: {current}")
            state = node_fn(state)
            trace.append(current)
            step += 1

            # 路由到下一个节点
            router_fn = self._routers.get(current)
            if router_fn:
                next_node = router_fn(state)
                print(f"  [workflow]   → route to: {next_node}")
                current = next_node
            else:
                current = self.END

        if step >= self._max_steps:
            print(f"  [workflow] 达到最大步数 {self._max_steps}, 强制停止")

        state["_trace"] = trace
        return state

    # ---- 可视化 ----

    def describe(self) -> str:
        """返回工作流图的文本描述"""
        lines = [f"入口: {self._entry}", ""]
        for name in self._nodes:
            has_router = "→ 动态路由" if name in self._routers else "→ END"
            lines.append(f"  [{name}] {has_router}")
        return "\n".join(lines)
