"""ReAct tool-calling agent (one-shot).

Demonstrates the core LangChain v1 agent loop: hand `create_agent` a model and a
list of `@tool` functions, ask one question, and let the agent decide which tools to
call to answer it.

Run from `libs/langchain_v1`:

    uv run --with langchain-openai python ../../examples/01_react_agent.py

Requires `OPENAI_API_KEY` in the environment.
"""

from __future__ import annotations

import ast
import operator
import os
import sys
from typing import TYPE_CHECKING

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

MODEL = "openai:gpt-4o-mini"

# Canned data keeps the example deterministic and free; a real tool would call a
# weather API here.
_WEATHER = {
    "london": "15°C, overcast",
    "san francisco": "18°C, foggy",
    "tokyo": "22°C, clear",
}

_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}
_UNARYOPS = {ast.USub: operator.neg, ast.UAdd: operator.pos}


def _eval_node(node: ast.AST) -> float:
    """Recursively evaluate a parsed arithmetic expression.

    Only numeric literals and basic arithmetic operators are allowed, so this never
    executes arbitrary code the way `eval` would.

    Args:
        node: A node from an AST parsed in `"eval"` mode.

    Returns:
        The numeric value of the expression rooted at `node`.

    Raises:
        ValueError: If the node is not a supported numeric expression.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        return _BINOPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARYOPS:
        return _UNARYOPS[type(node.op)](_eval_node(node.operand))
    msg = "Only numbers and the operators + - * / ** % are supported."
    raise ValueError(msg)


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city.

    Args:
        city: Name of the city to look up.

    Returns:
        A short human-readable weather description.
    """
    return _WEATHER.get(city.lower(), f"No weather data for {city!r}.")


@tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression.

    Args:
        expression: An arithmetic expression such as `"3 * (4 + 5)"`.

    Returns:
        The result as a string, or an error message if the expression is invalid.
    """
    try:
        tree = ast.parse(expression, mode="eval")
        return str(_eval_node(tree.body))
    except (ValueError, SyntaxError, TypeError, ZeroDivisionError) as exc:
        return f"Could not evaluate {expression!r}: {exc}"


def build_agent() -> CompiledStateGraph:
    """Build the ReAct agent with its two tools."""
    model = init_chat_model(MODEL)
    return create_agent(model, tools=[get_weather, calculator])


def main() -> None:
    """Run the agent against one hardcoded question that needs both tools."""
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("Set OPENAI_API_KEY before running this example.")

    agent = build_agent()
    question = "What's the weather in Tokyo, and what is 23 * 19?"
    print(f"Q: {question}\n")
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    print(f"A: {result['messages'][-1].content}")


if __name__ == "__main__":
    main()
