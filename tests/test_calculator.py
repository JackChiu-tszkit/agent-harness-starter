from agent_harness.core.errors import ToolError
from agent_harness.tools.calculator import evaluate_expression


def test_multiplication() -> None:
    assert evaluate_expression("125 * 48") == "6000"


def test_discount_expression() -> None:
    assert evaluate_expression("12 * 49.90") == "598.8"
    assert evaluate_expression("598.8 * (1 - 0.15)") == "508.98"


def test_rejects_unsafe_syntax() -> None:
    for expression in ("__import__('os')", "2 ** 10", "print(1)", "1 / 0", ""):
        try:
            evaluate_expression(expression)
        except ToolError:
            continue
        raise AssertionError(f"expression was accepted: {expression}")
