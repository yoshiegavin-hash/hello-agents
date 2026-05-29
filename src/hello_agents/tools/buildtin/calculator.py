import ast
import operator
import math
from tools.base import Tool, ToolParameter


class MyCalculator(Tool):
    """简单的数学计算工具，支持基本运算和sqrt函数"""

    def __init__(self):
        super().__init__(
            name="my_calculator",
            description="简单的数学计算工具，支持基本运算（+-*/）和sqrt函数"
        )
        self._operators = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv
        }
        self._functions = {
            'sqrt': math.sqrt,
            'pi': math.pi
        }

    def get_parameters(self):
        return [
            ToolParameter(
                name="expression",
                type="string",
                description="数学表达式，例如 15*8+32",
                required=True
            )
        ]

    def run(self, parameters: dict) -> str:
        expression = parameters.get("expression", "")
        if not expression.strip():
            return "计算表达式不能为空"

        try:
            node = ast.parse(expression, mode='eval')
            result = self._eval_node(node.body)
            return str(result)
        except Exception:
            return "计算失败，请检查表达式格式"

    def _eval_node(self, node):
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.BinOp):
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            op = self._operators.get(type(node.op))
            if op is None:
                raise ValueError(f"不支持的运算符: {type(node.op).__name__}")
            return op(left, right)
        elif isinstance(node, ast.Call):
            func_name = node.func.id
            if func_name in self._functions:
                args = [self._eval_node(arg) for arg in node.args]
                return self._functions[func_name](*args)
        elif isinstance(node, ast.Name):
            if node.id in self._functions:
                return self._functions[node.id]
        raise ValueError(f"不支持的表达式类型: {type(node).__name__}")
