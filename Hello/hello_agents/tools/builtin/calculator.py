"""计算器工具"""

import ast
import operator
import math
from typing import Dict, Any
from hello_agents.tools.base import Tool, ToolParameter


class CalculatorTool(Tool):
    """Python计算器工具"""

    # 支持的操作符
    OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.BitXor: operator.xor,
        ast.USub: operator.neg,
    }

    # 支持的函数
    FUNCTIONS = {
        'abs': abs,
        'round': round,
        'max': max,
        'min': min,
        'sum': sum,
        'sqrt': math.sqrt,
        'sin': math.sin,
        'cos': math.cos,
        'tan': math.tan,
        'log': math.log,
        'exp': math.exp,
        'pi': math.pi,
        'e': math.e,
    }

    def __init__(self,):
        super().__init__(
            name="python_calculator",
            description="执行数学计算。支持基本运算、数学函数等。例如：2+3*4, sqrt(16), sin(pi/2)等。"
        )

    def run(self, parameters: Dict[str, Any]) -> str:
        """
        执行计算

        Args:
            parameters: 包含input参数的字典

        Returns:
            计算结果
        """
        # 支持两种参数格式：input 和 expression

        expression = parameters.get("input", "") or parameters.get("expression", "")
        if not expression:
            return "错误：计算表达式不能为空"

        print(f"🧮 正在计算: {expression}")

        try:
            # 解析表达式
            node = ast.parse(expression, mode='eval') # 不用 eval()（有安全漏洞）
            # 10 + 5 * 2
            # Expression(
            #     body=BinOp(
            #         left=Constant(value=10),
            #         op=Add(),
            #         right=BinOp(
            #             left=Constant(value=5),
            #             op=Mult(),
            #             right=Constant(value=2)
            #         )
            #     )
            # )

            result = self._eval_node(node.body)

            result_str = str(result)
            print(f"✅ 计算结果: {result_str}")
            return result_str
        except Exception as e:
            error_msg = f"计算失败: {str(e)}"
            print(f"❌ {error_msg}")
            return error_msg
    def _eval_node(self, node):
        """递归计算AST节点"""
        if isinstance(node, ast.Constant): # Python 3.8+
            return node.value
        elif isinstance(node, ast.Num):  # Python < 3.8
            return node.n
        elif isinstance(node, ast.BinOp):
            return self.OPERATORS[type(node.op)](
                self._eval_node(node.left),
                self._eval_node(node.right)
            )

        # func = self.OPERATORS[type(node.op)]  # func = operator.add
        # a = self._eval_node(node.left)  # a = 5
        # b = self._eval_node(node.right)  # b = 3
        # result = func(a, b)  # 等价于 add(5,3) → 8

        elif isinstance(node, ast.UnaryOp): # 处理一元运算符（负号 -5）
            return self.OPERATORS[type(node.op)](self._eval_node(node.operand))

        elif isinstance(node, ast.Call): # 处理函数调用，例如 sin (3.14)、max (1,2)
            func_name = node.func.id
            if func_name in self.FUNCTIONS:
                args = [self._eval_node(arg) for arg in node.args]
                return self.FUNCTIONS[func_name](*args)
            else:
                raise ValueError(f"不支持的函数: {func_name}")

        elif isinstance(node, ast.Name):
            if node.id in self.FUNCTIONS:
                return self.FUNCTIONS[node.id]
            else:
                raise ValueError(f"未定义的变量: {node.id}")
        else:
            raise ValueError(f"不支持的表达式类型: {type(node)}")

    def get_parameters(self):
        return[
            ToolParameter(
                name="input",
                type="string",
                description="要计算的数学表达式，支持基本运算和数学函数",
                required=True
            )
        ]

# 便捷函数
def calculate(expression: str) -> str:
    """
    执行数学计算

    Args:
        expression: 数学表达式

    Returns:
        计算结果字符串
    """
    tool = CalculatorTool()
    return tool.run({"input": expression})