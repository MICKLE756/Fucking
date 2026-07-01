from hello_agents.tools.registry import ToolRegistry
from hello_agents.tools.builtin.calculator import calculate

reg = ToolRegistry()
reg.register_function("calculate", "数学计算", calculate)
print(reg.get_tools_description())
print(reg.execute_tool("calculate", "(17*34+78)/23+45"))   # 应输出 390