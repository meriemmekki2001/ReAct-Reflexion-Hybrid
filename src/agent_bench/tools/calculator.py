"""Safe arithmetic calculator for GSM8K; no execution of Python code."""
import ast
import operator
from langchain_core.tools import tool

OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow,
}
UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def calculate(expression: str) -> str:
    try:
        tree = ast.parse(expression, mode="eval")
        def visit(node, depth=0):
            if depth > 20:
                raise ValueError("Expression too complex")
            if isinstance(node, ast.Expression):
                return visit(node.body, depth + 1)
            if isinstance(node, ast.Constant) and type(node.value) in (int, float):
                return node.value
            if isinstance(node, ast.UnaryOp) and type(node.op) in UNARY:
                return UNARY[type(node.op)](visit(node.operand, depth + 1))
            if isinstance(node, ast.BinOp) and type(node.op) in OPS:
                left = visit(node.left, depth + 1)
                right = visit(node.right, depth + 1)
                if isinstance(node.op, ast.Pow) and (abs(right) > 12 or abs(left) > 1e6):
                    raise ValueError("Power too large")
                return OPS[type(node.op)](left, right)
            raise ValueError("Only arithmetic operations are supported")
        value = visit(tree)
        if not isinstance(value, (int, float)) or abs(value) > 1e15:
            raise ValueError("Result out of range")
        return str(int(value) if float(value).is_integer() else round(value, 10))
    except (SyntaxError, ValueError, ZeroDivisionError, OverflowError, TypeError) as exc:
        return f"Calculator error: {exc}"


def create_gsm8k_tools(_environment):
    @tool
    def calculator(expression: str) -> str:
        """Calculate an arithmetic expression, e.g. '(12 * 3) + 4'."""
        return calculate(expression)
    return [calculator]
