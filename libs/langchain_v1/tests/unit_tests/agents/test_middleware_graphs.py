"""Test matrix for middleware graph structures with all combinations of jump targets."""

from __future__ import annotations

import itertools
from typing import Any

import pytest
from syrupy.assertion import SnapshotAssertion

from langchain.agents.middleware_agent import create_agent
from langchain.agents.middleware.types import AgentMiddleware, JumpTo
from langchain_core.tools import tool, BaseTool
from dataclasses import dataclass

from .model import FakeToolCallingModel


# All possible combinations of jump targets (power set of ["tools", "model", "end"])
JUMP_TARGETS: list[list[JumpTo]] = [
    [],
    ["tools"],
    ["model"],
    ["end"],
    ["tools", "model"],
    ["tools", "end"],
    ["model", "end"],
    ["tools", "model", "end"],
]


def create_middleware_with_jump_to(
    name: str,
    before_model_jump_to_: list[JumpTo],
    after_model_jump_to_: list[JumpTo],
    tools_: list[BaseTool] = [],
) -> AgentMiddleware:
    """Create a middleware class with specified jump_to configurations."""

    class CustomMiddleware(AgentMiddleware):
        before_model_jump_to: list[JumpTo] = before_model_jump_to_
        after_model_jump_to: list[JumpTo] = after_model_jump_to_
        tools: list[BaseTool] = tools_

        def before_model(self, state: Any, runtime: Any) -> None:
            """Before model hook."""
            pass

        def after_model(self, state: Any, runtime: Any) -> None:
            """After model hook."""
            pass

    CustomMiddleware.__name__ = name
    return CustomMiddleware()


@tool
def some_tool() -> str:
    """A simple test tool."""
    return "Hello, world!"


@dataclass(frozen=True)
class TestCase:
    """Test case configuration for middleware graph testing.

    Represents a specific combination of middleware jump targets and tool availability.

    Example:
        TestCase(
            a_before=["tools"],
            a_after=["end"],
            b_before=[],
            b_after=["model"],
            has_tools=True
        )
        This creates two middleware instances where:
        - MiddlewareA jumps to "tools" before model, "end" after model
        - MiddlewareB has no jumps before model, jumps to "model" after model
        - Agent has tools available
    """
    a_before: list[JumpTo]
    a_after: list[JumpTo]
    b_before: list[JumpTo]
    b_after: list[JumpTo]
    has_tools: bool


def format_jumps(jumps: list[JumpTo]) -> str:
    """Format jump targets for test ID."""
    return "_".join(jumps) if jumps else "empty"


def format_test_case_name(test_case: TestCase) -> str:
    """Format the test case name for pytest ID."""
    return (
        f"A_before_{format_jumps(test_case.a_before)}_"
        f"A_after_{format_jumps(test_case.a_after)}_"
        f"B_before_{format_jumps(test_case.b_before)}_"
        f"B_after_{format_jumps(test_case.b_after)}_"
        f"tools_{test_case.has_tools}"
    )


def _is_valid_test_case(
    a_before: list[JumpTo],
    a_after: list[JumpTo],
    b_before: list[JumpTo],
    b_after: list[JumpTo],
    has_tools: bool,
) -> bool:
    """Check if test case is valid - can't jump to tools when no tools available."""
    if has_tools:
        return True

    all_jump_targets = {*a_before, *a_after, *b_before, *b_after}
    return "tools" not in all_jump_targets


def generate_test_cases() -> list[TestCase]:
    """Generate all valid test case combinations."""
    test_cases: list[TestCase] = []

    for has_tools in [False, True]:
        for a_before, a_after, b_before, b_after in itertools.product(JUMP_TARGETS, repeat=4):
            if _is_valid_test_case(a_before, a_after, b_before, b_after, has_tools):
                test_cases.append(
                    TestCase(
                        a_before=a_before,
                        a_after=a_after,
                        b_before=b_before,
                        b_after=b_after,
                        has_tools=has_tools,
                    )
                )

    return test_cases


@pytest.mark.parametrize(
    "test_case",
    generate_test_cases(),
    ids=format_test_case_name,
)
def test_middleware_graph_structure(
    snapshot: SnapshotAssertion,
    test_case: TestCase,
) -> None:
    """Test that middleware graphs are created with correct structure for all combinations."""
    middleware_a = create_middleware_with_jump_to(
        "MiddlewareA", test_case.a_before, test_case.a_after
    )
    middleware_b = create_middleware_with_jump_to(
        "MiddlewareB", test_case.b_before, test_case.b_after
    )

    tools = [some_tool] if test_case.has_tools else []
    agent = create_agent(
        model=FakeToolCallingModel(),
        tools=tools,
        middleware=[middleware_a, middleware_b],
    )

    mermaid_diagram = agent.compile().get_graph().draw_mermaid(with_styles=False)
    assert mermaid_diagram == snapshot


def test_tool_registration_with_middleware(
    snapshot: SnapshotAssertion,
) -> None:
    """Test tool registration via middleware, agent, or both."""
    # Test case 1: No tools anywhere
    middleware_no_tools = create_middleware_with_jump_to(
        "Middleware", before_model_jump_to_=[], after_model_jump_to_=[], tools_=[]
    )
    agent_no_tools = create_agent(
        model=FakeToolCallingModel(),
        tools=[],
        middleware=[middleware_no_tools],
    )

    diagram_no_tools = agent_no_tools.compile().get_graph().draw_mermaid(with_styles=False)
    assert diagram_no_tools == snapshot(name="no_tools")

    # Test case 2: Tools only via middleware
    middleware_with_tools = create_middleware_with_jump_to(
        "Middleware", before_model_jump_to_=[], after_model_jump_to_=[], tools_=[some_tool]
    )
    agent_middleware_tools = create_agent(
        model=FakeToolCallingModel(),
        tools=[],
        middleware=[middleware_with_tools],
    )

    diagram_middleware_tools = agent_middleware_tools.compile().get_graph().draw_mermaid(with_styles=False)
    assert diagram_middleware_tools == snapshot(name="middleware_tools_only")

    # Test case 3: Tools only via agent
    middleware_no_tools_agent = create_middleware_with_jump_to(
        "Middleware", before_model_jump_to_=[], after_model_jump_to_=[], tools_=[]
    )
    agent_with_tools = create_agent(
        model=FakeToolCallingModel(),
        tools=[some_tool],
        middleware=[middleware_no_tools_agent],
    )

    diagram_agent_tools = agent_with_tools.compile().get_graph().draw_mermaid(with_styles=False)
    assert diagram_agent_tools == snapshot(name="agent_tools_only")

    # Test case 4: Tools via both middleware and agent
    agent_both_tools = create_agent(
        model=FakeToolCallingModel(),
        tools=[some_tool],
        middleware=[middleware_with_tools],
    )

    diagram_both_tools = agent_both_tools.compile().get_graph().draw_mermaid(with_styles=False)
    assert diagram_both_tools == snapshot(name="both_tools")
