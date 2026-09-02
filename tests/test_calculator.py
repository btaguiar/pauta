import pytest

from pauta.tools.calculator import (
    MAX_EXPONENT,
    UnsafeExpression,
    calculator,
    evaluate,
)


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("2 + 2", 4.0),
        ("40000 * 900", 36_000_000.0),
        ("(900 + 400) * 40000 * 30", 1_560_000_000.0),
        ("1560000000 / 1000000 * 0.2", 312.0),
        ("100 - 75", 25.0),
        ("7 // 2", 3.0),
        ("7 % 2", 1.0),
        ("2 ** 10", 1024.0),
        ("-5 + 3", -2.0),
        ("((2 + 3) * 4) / 10", 2.0),
    ],
)
def test_arithmetic_is_exact(expression: str, expected: float) -> None:
    assert evaluate(expression) == expected


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('dir')",
        "open('/etc/passwd').read()",
        "1 if True else 2",
        "[1, 2, 3]",
        "'texto'",
        "x + 1",
        "abs(-3)",
        "().__class__",
        "True + 1",
    ],
)
def test_anything_that_is_not_arithmetic_is_refused(expression: str) -> None:
    with pytest.raises(UnsafeExpression):
        evaluate(expression)


def test_a_huge_exponent_does_not_hang_the_process() -> None:
    with pytest.raises(UnsafeExpression, match="expoente"):
        evaluate(f"2 ** {MAX_EXPONENT + 1}")


def test_broken_syntax_is_reported_not_raised_raw() -> None:
    with pytest.raises(UnsafeExpression, match="inválida"):
        evaluate("2 +")


def test_the_tool_reports_failure_instead_of_crashing() -> None:
    assert "divisão por zero" in calculator.invoke({"expression": "1 / 0"})
    assert "não calculado" in calculator.invoke({"expression": "__import__('os')"})


def test_the_tool_shows_the_expression_with_the_result() -> None:
    """O relatório precisa poder citar a conta, não só o número."""
    assert calculator.invoke({"expression": "1300 * 40000"}) == "1300 * 40000 = 52000000.0"
