"""Aritmética determinística. O LLM decide o que calcular, não calcula."""

import ast
import operator
from collections.abc import Callable
from typing import Any

from langchain_core.tools import BaseTool, tool

#: Operações permitidas. Nada fora desta tabela é avaliado.
BINARY_OPS: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

UNARY_OPS: dict[type[ast.unaryop], Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

#: Teto do expoente. `2 ** 10**9` trava o processo sem nenhum aviso.
MAX_EXPONENT = 1_000

#: Casas decimais do resultado. Precisão maior que isso é ruído numa análise.
RESULT_PRECISION = 6


class UnsafeExpression(ValueError):
    """A expressão usa algo que a calculadora não avalia."""


def _evaluate(node: ast.AST) -> Any:
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, int | float):
            raise UnsafeExpression(f"só número é aceito, veio {type(node.value).__name__}")
        return node.value
    if isinstance(node, ast.BinOp):
        handler = BINARY_OPS.get(type(node.op))
        if handler is None:
            raise UnsafeExpression(f"operação não permitida: {type(node.op).__name__}")
        left, right = _evaluate(node.left), _evaluate(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
            raise UnsafeExpression(f"expoente acima do teto de {MAX_EXPONENT}")
        return handler(left, right)
    if isinstance(node, ast.UnaryOp):
        unary = UNARY_OPS.get(type(node.op))
        if unary is None:
            raise UnsafeExpression(f"operação não permitida: {type(node.op).__name__}")
        return unary(_evaluate(node.operand))
    raise UnsafeExpression(f"expressão não permitida: {type(node).__name__}")


def evaluate(expression: str) -> float:
    """Avalia uma expressão aritmética sem executar código.

    Só números e as operações da tabela passam. Nome, chamada de função, atributo
    e literal de texto são recusados antes de qualquer avaliação.
    """
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise UnsafeExpression(f"expressão inválida: {exc.msg}") from exc
    result = _evaluate(tree)
    return float(result)


@tool
def calculator(expression: str) -> str:
    """Calcula uma expressão aritmética e devolve o resultado.

    Aceita + - * / // % ** e parênteses, sobre números. Use para toda conta que
    entrar no relatório, em vez de calcular de cabeça.
    """
    try:
        value = evaluate(expression)
    except UnsafeExpression as exc:
        return f"não calculado: {exc}"
    except ZeroDivisionError:
        return "não calculado: divisão por zero"
    except OverflowError:
        return "não calculado: resultado grande demais"
    return f"{expression} = {round(value, RESULT_PRECISION)}"


def get_calculator_tool() -> BaseTool:
    return calculator
