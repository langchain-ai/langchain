# ruff: noqa: T201
"""ACE Middleware Demo: Watch the Playbook Evolve.

This example demonstrates how the ACE (Agentic Context Engineering) middleware
enables agents to self-improve by maintaining an evolving playbook of strategies
and insights learned from interactions.

Run this script to see the playbook evolve as the agent solves math problems.

Usage:
    export OPENAI_API_KEY="your-key"
    python examples/ace_playbook_demo.py
"""

import ast
import operator
import uuid

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver

from langchain.agents import create_agent
from langchain.agents.middleware import ACEMiddleware

# Safe math operators for the calculator
_SAFE_OPERATORS = {
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
            return node.value
        msg = f"Unsupported constant type: {type(node.value)}"
        raise ValueError(msg)
    if isinstance(node, ast.BinOp):
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        op = _SAFE_OPERATORS.get(type(node.op))
        if op is None:
            msg = f"Unsupported operator: {type(node.op).__name__}"
            raise ValueError(msg)
        return op(left, right)
    if isinstance(node, ast.UnaryOp):
        operand = _safe_eval(node.operand)
        op = _SAFE_OPERATORS.get(type(node.op))
        if op is None:
            msg = f"Unsupported unary operator: {type(node.op).__name__}"
            raise ValueError(msg)
        return op(operand)
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


def main() -> None:
    """Run the ACE playbook evolution demo."""
    # Create ACE middleware with reflection enabled
    ace = ACEMiddleware(
        reflector_model="gpt-4.1",  # Analyzes trajectories after each response
        curator_frequency=2,  # Add new insights every 2 interactions
        initial_playbook="""## STRATEGIES & INSIGHTS
[str-00001] helpful=0 harmful=0 :: Break word problems into clear steps before calculating

## COMMON MISTAKES TO AVOID
[mis-00001] helpful=0 harmful=0 :: Don't forget to include units in the final answer
""",
    )

    # Create a checkpointer to access internal state (including playbook)
    checkpointer = MemorySaver()

    # Create agent with ACE middleware and checkpointer
    agent = create_agent(
        model="gpt-4.1",
        tools=[calculator],
        middleware=[ace],
        checkpointer=checkpointer,
    )

    # Use a consistent thread ID to maintain state across invocations
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    # Math problems to solve - the agent will learn from each one
    problems = [
        "What is 15% of 240?",
        "If I invest $1000 at 5% annual interest compounded yearly, how much will I have after 3 years?",
        "A shirt costs $45 with 20% off. What's the sale price?",
        "A car travels 180 miles using 6 gallons of gas. What is its fuel efficiency in miles per gallon?",
        "If 8 workers can complete a job in 12 days, how many days would it take 6 workers?",
    ]

    print("=" * 60)
    print("ACE Middleware Demo: Watch the Playbook Evolve")
    print("=" * 60)
    print()
    print("This demo shows how ACE learns from each interaction.")
    print("The reflector analyzes responses with detailed error analysis:")
    print("  - Identifies what went wrong (if anything)")
    print("  - Finds root causes of errors")
    print("  - Suggests correct approaches")
    print("  - Tags playbook bullets as helpful/harmful/neutral")
    print("The curator periodically adds new insights to the playbook.")
    print()

    # Track all reflections for final summary
    all_reflections: list[tuple[int, str, str]] = []  # (problem_num, problem, reflection)

    for i, problem in enumerate(problems, 1):
        print(f"\n{'─' * 60}")
        print(f"Problem {i}/{len(problems)}")
        print(f"{'─' * 60}")
        print(f"Q: {problem}")
        print()

        # Invoke the agent with config to use checkpointer
        result = agent.invoke({"messages": [HumanMessage(content=problem)]}, config)

        # Get the answer
        answer = result["messages"][-1].content
        print(f"A: {answer}")

        # Access the full state from the checkpointer (includes private fields)
        state_snapshot = agent.get_state(config)
        full_state = state_snapshot.values

        # Show interaction count
        ace_count = full_state.get("ace_interaction_count", i)
        print(f"\n📊 Interaction count: {ace_count}")

        # Show if curator ran
        if ace_count % ace.curator_frequency == 0:
            print("✨ Curator ran - new insights may have been added!")

        # Show the current playbook state from the full state
        playbook_data = full_state.get("ace_playbook")
        if playbook_data:
            print("\n📖 Current Playbook:")
            print("─" * 40)
            content = playbook_data.get("content", "")
            # Print each non-empty line
            for line in content.split("\n"):
                if line.strip():
                    print(f"  {line}")
            print("─" * 40)

        # Show the last reflection (now includes error analysis if applicable)
        last_reflection = full_state.get("ace_last_reflection", "")
        if last_reflection and last_reflection.strip():
            print("\n💡 Reflection:")
            print("─" * 40)
            for line in last_reflection.split("\n"):
                if line.strip():
                    print(f"  {line}")
            print("─" * 40)
            # Store for final summary
            all_reflections.append((i, problem, last_reflection))

    print()
    print("=" * 60)
    print("Demo Complete!")
    print("=" * 60)

    # Print reflection summary
    if all_reflections:
        print()
        print("📋 REFLECTION SUMMARY")
        print("=" * 60)
        for prob_num, prob_text, reflection in all_reflections:
            print(f"\nProblem {prob_num}: {prob_text[:50]}...")
            print("─" * 40)
            for line in reflection.split("\n"):
                if line.strip():
                    print(f"  {line}")
        print()
        print("=" * 60)

    print()
    print("The playbook has evolved based on the agent's performance.")
    print("Bullets with higher 'helpful' counts were effective.")
    print("In a real application, you would persist the playbook state")
    print("to continue learning across sessions.")


if __name__ == "__main__":
    main()
