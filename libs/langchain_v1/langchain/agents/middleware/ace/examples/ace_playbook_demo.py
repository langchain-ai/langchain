# ruff: noqa: T201
"""ACE Middleware Demo: Watch the Playbook Evolve.

This example demonstrates how the ACE (Agentic Context Engineering) middleware
enables agents to self-improve by maintaining an evolving playbook of strategies
and insights learned from interactions.

The demo shows two modes:
1. **Training mode**: Uses ground truth answers (from ~/dev/ace data) to provide
   richer feedback to the reflector, enabling faster learning.
2. **Inference mode**: Normal usage without ground truth.

Run this script to see the playbook evolve as the agent solves math problems.

Usage:
    export OPENAI_API_KEY="your-key"
    python examples/ace_playbook_demo.py
"""

import ast
import operator
import re
import uuid
from collections.abc import Callable
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver

from langchain.agents import create_agent
from langchain.agents.middleware import ACEMiddleware
from langchain.agents.middleware.ace import SectionName

# Training data with ground truth (adapted from ~/dev/ace/finance/data/)
# These are financial word problems where we know the correct answer.
TRAINING_DATA: list[dict[str, str]] = [
    {
        "question": (
            "A software development firm lists its current assets at $1,000,000 "
            "and its current liabilities at $500,000. Find the current ratio."
        ),
        "ground_truth": "2.0",
    },
    {
        "question": (
            "Calculate the ROI for an investor who buys property for $200,000 "
            "and spends an additional $50,000 on renovations, then sells the "
            "property for $300,000."
        ),
        "ground_truth": "0.2",
    },
    {
        "question": (
            "If the return of a portfolio was 8% and the risk-free rate was 2%, "
            "and the standard deviation of the portfolio's excess return was 10%, "
            "calculate the Sharpe Ratio."
        ),
        "ground_truth": "0.6",
    },
    {
        "question": (
            "A pet supplies store had $75,000 in net credit sales and an average "
            "accounts receivable of $15,000 last year. Compute the accounts "
            "receivable turnover."
        ),
        "ground_truth": "5.0",
    },
]

# Inference problems (no ground truth - simulates real usage)
INFERENCE_PROBLEMS: list[str] = [
    "What is 15% of 240?",
    "A shirt costs $45 with 20% off. What's the sale price?",
    "A car travels 180 miles using 6 gallons of gas. "
    "What is its fuel efficiency in miles per gallon?",
]

# Safe math operators for the calculator
_SAFE_OPERATORS: dict[type, Callable[..., float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}


def _safe_eval(node: ast.AST) -> float:
    """Safely evaluate an AST node containing only math operations."""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return float(node.value)
        msg = f"Unsupported constant type: {type(node.value)}"
        raise ValueError(msg)
    if isinstance(node, ast.BinOp):
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        op = _SAFE_OPERATORS.get(type(node.op))
        if op is None:
            msg = f"Unsupported operator: {type(node.op).__name__}"
            raise ValueError(msg)
        return float(op(left, right))
    if isinstance(node, ast.UnaryOp):
        operand = _safe_eval(node.operand)
        op = _SAFE_OPERATORS.get(type(node.op))
        if op is None:
            msg = f"Unsupported unary operator: {type(node.op).__name__}"
            raise ValueError(msg)
        return float(op(operand))
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    msg = f"Unsupported expression type: {type(node).__name__}"
    raise ValueError(msg)


@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression using Python syntax.

    Args:
        expression: A valid Python math expression
            (e.g., "15 * 0.20" or "1000 * 1.05 ** 3").

    Returns:
        The result of the calculation as a string.
    """
    try:
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval(tree)
        return str(result)
    except Exception as e:
        return f"Error: {e}"


def _print_playbook_state(
    full_state: dict[str, Any],
    previous_playbook_content: str,
    *,
    curator_ran: bool,
) -> str:
    """Print playbook state and return current content for next iteration."""
    playbook_data = full_state.get("ace_playbook")
    if not playbook_data:
        return previous_playbook_content

    content = playbook_data.get("content", "")

    # Detect new insights by comparing bullet IDs
    if curator_ran:
        prev_ids = set(re.findall(r"\[[a-z]{3}-\d{5}\]", previous_playbook_content))
        curr_ids = set(re.findall(r"\[[a-z]{3}-\d{5}\]", content))
        new_ids = curr_ids - prev_ids

        if new_ids:
            print("\n🆕 New insights added:")
            for line in content.split("\n"):
                for new_id in new_ids:
                    if new_id in line:
                        print(f"  + {line.strip()}")
                        break

    print("\n📖 Current Playbook:")
    print("─" * 40)
    for line in content.split("\n"):
        if line.strip():
            print(f"  {line}")
    print("─" * 40)

    # Show the last reflection
    last_reflection = full_state.get("ace_last_reflection", "")
    if last_reflection and last_reflection.strip():
        print("\n💡 Reflection:")
        print("─" * 40)
        for line in last_reflection.split("\n"):
            if line.strip():
                print(f"  {line}")
        print("─" * 40)

    return content


def main() -> None:
    """Run the ACE playbook evolution demo."""
    # Create ACE middleware with reflection and curation
    ace = ACEMiddleware(
        reflector_model="gpt-4.1",  # Analyzes trajectories after each response
        curator_model="gpt-4.1",  # Reviews reflections and adds new insights
        curator_frequency=2,  # Add new insights every 2 interactions
        initial_playbook=f"""## {SectionName.STRATEGIES_AND_INSIGHTS}
[str-00001] helpful=0 harmful=0 :: Break word problems into clear steps before calculating

## {SectionName.COMMON_MISTAKES_TO_AVOID}
[mis-00001] helpful=0 harmful=0 :: Don't forget to include units in the final answer
""",
    )

    # Create a checkpointer to access internal state (including playbook)
    checkpointer = MemorySaver()

    # Create agent with ACE middleware and checkpointer
    agent: Any = create_agent(
        model="gpt-4.1",
        tools=[calculator],
        middleware=[ace],
        checkpointer=checkpointer,
    )

    # Use a consistent thread ID to maintain state across invocations
    thread_id = str(uuid.uuid4())
    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

    print("=" * 60)
    print("ACE Middleware Demo: Watch the Playbook Evolve")
    print("=" * 60)
    print()
    print("This demo shows how ACE learns from each interaction.")
    print()
    print("PHASE 1: TRAINING (with ground truth)")
    print("  - Ground truth enables richer reflector feedback")
    print("  - The reflector can compare agent answers to known correct answers")
    print("  - Faster learning from clear success/failure signals")
    print()
    print("PHASE 2: INFERENCE (without ground truth)")
    print("  - Normal usage after training")
    print("  - Reflector analyzes reasoning quality without ground truth")
    print()

    # Track previous playbook to detect new insights
    previous_playbook_content: str = ace.initial_playbook
    total_problems = len(TRAINING_DATA) + len(INFERENCE_PROBLEMS)
    problem_num = 0

    # =========================================================================
    # PHASE 1: Training with ground truth
    # =========================================================================
    print("\n" + "=" * 60)
    print("📚 PHASE 1: TRAINING (with ground truth)")
    print("=" * 60)

    for item in TRAINING_DATA:
        problem_num += 1
        question = item["question"]
        ground_truth = item["ground_truth"]

        print(f"\n{'─' * 60}")
        print(f"Training Problem {problem_num}/{total_problems}")
        print(f"{'─' * 60}")
        print(f"Q: {question}")
        print(f"📋 Ground Truth: {ground_truth}")
        print()

        # Invoke with ground_truth for enhanced reflection
        result = agent.invoke(
            {
                "messages": [HumanMessage(content=question)],
                "ground_truth": ground_truth,
            },
            config,
        )

        # Get the answer
        answer = result["messages"][-1].content
        print(f"A: {answer}")

        # Access the full state from the checkpointer
        state_snapshot = agent.get_state(config)
        full_state = state_snapshot.values

        # Show interaction count
        ace_count = full_state.get("ace_interaction_count", problem_num)
        print(f"\n📊 Interaction count: {ace_count}")

        # Show if curator ran
        curator_ran = ace_count % ace.curator_frequency == 0
        if curator_ran:
            print("✨ Curator ran - checking for new insights...")

        # Print playbook state
        previous_playbook_content = _print_playbook_state(
            full_state, previous_playbook_content, curator_ran=curator_ran
        )

    # =========================================================================
    # PHASE 2: Inference without ground truth
    # =========================================================================
    print("\n" + "=" * 60)
    print("🚀 PHASE 2: INFERENCE (without ground truth)")
    print("=" * 60)
    print("Now using the playbook learned during training...")

    for question in INFERENCE_PROBLEMS:
        problem_num += 1

        print(f"\n{'─' * 60}")
        print(f"Inference Problem {problem_num}/{total_problems}")
        print(f"{'─' * 60}")
        print(f"Q: {question}")
        print()

        # Invoke WITHOUT ground_truth - normal inference
        result = agent.invoke(
            {"messages": [HumanMessage(content=question)]},
            config,
        )

        # Get the answer
        answer = result["messages"][-1].content
        print(f"A: {answer}")

        # Access the full state from the checkpointer
        state_snapshot = agent.get_state(config)
        full_state = state_snapshot.values

        # Show interaction count
        ace_count = full_state.get("ace_interaction_count", problem_num)
        print(f"\n📊 Interaction count: {ace_count}")

        # Show if curator ran
        curator_ran = ace_count % ace.curator_frequency == 0
        if curator_ran:
            print("✨ Curator ran - checking for new insights...")

        # Print playbook state
        previous_playbook_content = _print_playbook_state(
            full_state, previous_playbook_content, curator_ran=curator_ran
        )

    print()
    print("=" * 60)
    print("Demo Complete!")
    print("=" * 60)
    print()
    print("The playbook has evolved based on the agent's performance.")
    print("Training with ground truth enabled faster, more accurate learning.")
    print("Bullets with higher 'helpful' counts were effective.")
    print("In a real application, you would persist the playbook state")
    print("to continue learning across sessions.")


if __name__ == "__main__":
    main()
