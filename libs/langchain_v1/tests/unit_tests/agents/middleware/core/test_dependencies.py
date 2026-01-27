"""Tests for middleware dependency resolution and auto-instantiation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from langchain_core.messages import HumanMessage

from langchain.agents import create_agent
from langchain.agents.factory import _resolve_middleware_dependencies
from langchain.agents.middleware import AgentMiddleware, before_model
from tests.unit_tests.agents.model import FakeToolCallingModel

if TYPE_CHECKING:
    from langchain.agents.middleware.types import AgentState


class MiddlewareA(AgentMiddleware):
    """Test middleware A with no dependencies."""

    def __init__(self, value: str = "A") -> None:
        super().__init__()
        self.value = value
        self.tools = []
        self.depends_on = ()


class MiddlewareB(AgentMiddleware):
    """Test middleware B that depends on A."""

    def __init__(self, value: str = "B") -> None:
        super().__init__()
        self.value = value
        self.tools = []
        self.depends_on = (MiddlewareA,)


class MiddlewareC(AgentMiddleware):
    """Test middleware C that depends on B."""

    def __init__(self, value: str = "C") -> None:
        super().__init__()
        self.value = value
        self.tools = []
        self.depends_on = (MiddlewareB,)


class MiddlewareD(AgentMiddleware):
    """Test middleware D that depends on both A and B."""

    def __init__(self, value: str = "D") -> None:
        super().__init__()
        self.value = value
        self.tools = []
        self.depends_on = (MiddlewareA, MiddlewareB)


class MiddlewareCircular1(AgentMiddleware):
    """Test middleware for circular dependency detection."""

    def __init__(self) -> None:
        super().__init__()
        self.tools = []
        self.depends_on = (MiddlewareCircular2,)


class MiddlewareCircular2(AgentMiddleware):
    """Test middleware for circular dependency detection."""

    def __init__(self) -> None:
        super().__init__()
        self.tools = []
        self.depends_on = (MiddlewareCircular1,)


def test_no_dependencies():
    """Test middleware with no dependencies."""
    model = FakeToolCallingModel()
    middleware_a = MiddlewareA()

    agent = create_agent(model, middleware=[middleware_a])

    # Verify the agent was created successfully
    assert agent is not None


def test_auto_instantiate_single_dependency():
    """Test that missing dependencies are auto-instantiated."""
    model = FakeToolCallingModel()
    middleware_b = MiddlewareB()

    # MiddlewareB depends on MiddlewareA, which should be auto-instantiated
    agent = create_agent(model, middleware=[middleware_b])

    # Verify the agent was created successfully
    assert agent is not None


def test_auto_instantiate_chain_dependencies():
    """Test auto-instantiation of a chain of dependencies."""
    model = FakeToolCallingModel()
    middleware_c = MiddlewareC()

    # MiddlewareC depends on B, which depends on A
    # Both A and B should be auto-instantiated
    agent = create_agent(model, middleware=[middleware_c])

    # Verify the agent was created successfully
    assert agent is not None


def test_dependency_ordering():
    """Test that dependencies are ordered correctly (dependencies before dependents)."""
    middleware_c = MiddlewareC()
    middleware_b = MiddlewareB()
    middleware_a = MiddlewareA()

    # Provide middleware in reverse order
    resolved = _resolve_middleware_dependencies([middleware_c, middleware_b, middleware_a])

    # Check that dependencies come before dependents
    # Expected order: A, B, C
    assert len(resolved) == 3
    assert isinstance(resolved[0], MiddlewareA)
    assert isinstance(resolved[1], MiddlewareB)
    assert isinstance(resolved[2], MiddlewareC)


def test_auto_instantiate_with_existing_dependency():
    """Test that existing dependencies are not duplicated."""
    middleware_a = MiddlewareA(value="custom_A")
    middleware_b = MiddlewareB()

    # MiddlewareB depends on MiddlewareA, but we already have an instance
    resolved = _resolve_middleware_dependencies([middleware_b, middleware_a])

    # Should have exactly 2 middleware (no duplication)
    assert len(resolved) == 2

    # The custom instance should be used
    a_instance = next(m for m in resolved if isinstance(m, MiddlewareA))
    assert a_instance.value == "custom_A"


def test_multiple_dependents_same_dependency():
    """Test that multiple middleware can depend on the same middleware."""
    middleware_b = MiddlewareB()
    middleware_d = MiddlewareD()

    # Both B and D depend on A
    resolved = _resolve_middleware_dependencies([middleware_b, middleware_d])

    # Should have A, B, D (A is shared)
    assert len(resolved) == 3
    a_instances = [m for m in resolved if isinstance(m, MiddlewareA)]
    assert len(a_instances) == 1  # Only one instance of A


def test_circular_dependency_detection():
    """Test that circular dependencies are detected and raise an error."""
    middleware_1 = MiddlewareCircular1()

    # Should raise ValueError for circular dependency
    with pytest.raises(ValueError, match="Circular dependency detected"):
        _resolve_middleware_dependencies([middleware_1])


def test_decorator_with_depends_on():
    """Test that decorator-based middleware can specify dependencies."""

    @before_model(depends_on=(MiddlewareA,))
    def my_middleware(state: AgentState, runtime: Any) -> dict[str, Any] | None:
        """Test middleware function."""
        return None

    model = FakeToolCallingModel()

    # MiddlewareA should be auto-instantiated
    agent = create_agent(model, middleware=[my_middleware])

    # Verify the agent was created successfully
    assert agent is not None


def test_complex_dependency_graph():
    """Test a complex dependency graph with multiple levels."""
    # Create a complex graph:
    # D depends on A and B
    # C depends on B
    # B depends on A
    middleware_c = MiddlewareC()
    middleware_d = MiddlewareD()

    resolved = _resolve_middleware_dependencies([middleware_d, middleware_c])

    # Expected order: A, B, C, D (or A, B, D, C)
    # A must come before B, B must come before C and D
    assert len(resolved) == 4

    a_idx = next(i for i, m in enumerate(resolved) if isinstance(m, MiddlewareA))
    b_idx = next(i for i, m in enumerate(resolved) if isinstance(m, MiddlewareB))
    c_idx = next(i for i, m in enumerate(resolved) if isinstance(m, MiddlewareC))
    d_idx = next(i for i, m in enumerate(resolved) if isinstance(m, MiddlewareD))

    # Verify ordering constraints
    assert a_idx < b_idx, "A must come before B"
    assert b_idx < c_idx, "B must come before C"
    assert b_idx < d_idx, "B must come before D"
    assert a_idx < d_idx, "A must come before D"


def test_empty_middleware_list():
    """Test that empty middleware list works correctly."""
    resolved = _resolve_middleware_dependencies([])
    assert len(resolved) == 0


def test_middleware_with_no_depends_on_attribute():
    """Test middleware that doesn't have depends_on attribute (legacy middleware)."""

    class LegacyMiddleware(AgentMiddleware):
        """Legacy middleware without depends_on."""

        def __init__(self) -> None:
            super().__init__()
            self.tools = []
            # No depends_on attribute

    legacy = LegacyMiddleware()
    resolved = _resolve_middleware_dependencies([legacy])

    assert len(resolved) == 1
    assert isinstance(resolved[0], LegacyMiddleware)


def test_agent_execution_with_dependencies():
    """Test that an agent with dependencies can execute successfully."""
    model = FakeToolCallingModel()

    # Track execution order
    execution_order = []

    class TrackingMiddlewareA(AgentMiddleware):
        """Middleware that tracks execution."""

        def __init__(self) -> None:
            super().__init__()
            self.tools = []
            self.depends_on = ()

        def before_model(self, state: AgentState, runtime: Any) -> dict[str, Any] | None:
            """Track execution."""
            execution_order.append("A")
            return None

    class TrackingMiddlewareB(AgentMiddleware):
        """Middleware that depends on A and tracks execution."""

        def __init__(self) -> None:
            super().__init__()
            self.tools = []
            self.depends_on = (TrackingMiddlewareA,)

        def before_model(self, state: AgentState, runtime: Any) -> dict[str, Any] | None:
            """Track execution."""
            execution_order.append("B")
            return None

    middleware_b = TrackingMiddlewareB()
    agent = create_agent(model, middleware=[middleware_b])

    # Execute the agent
    result = agent.invoke({"messages": [HumanMessage(content="test")]})

    # Verify execution order: A should execute before B
    assert "A" in execution_order
    assert "B" in execution_order
    assert execution_order.index("A") < execution_order.index("B")
    assert result is not None


def test_dependency_with_instance():
    """Test that middleware can depend on an instance (not just a class)."""
    # Create a custom instance with specific configuration
    middleware_a = MiddlewareA(value="custom_instance")

    class MiddlewareWithInstanceDep(AgentMiddleware):
        """Middleware that depends on a specific instance."""

        def __init__(self, dep_instance: AgentMiddleware) -> None:
            super().__init__()
            self.tools = []
            # Depend on the instance, not the class
            self.depends_on = (dep_instance,)

    middleware_b = MiddlewareWithInstanceDep(middleware_a)

    resolved = _resolve_middleware_dependencies([middleware_b])

    # Should have both middleware
    assert len(resolved) == 2

    # The custom instance should be used
    a_instance = next(m for m in resolved if isinstance(m, MiddlewareA))
    assert a_instance is middleware_a
    assert a_instance.value == "custom_instance"


def test_mixed_class_and_instance_dependencies():
    """Test that middleware can depend on both classes and instances."""
    # Create a custom instance
    middleware_a_instance = MiddlewareA(value="instance_A")

    class MixedDependencyMiddleware(AgentMiddleware):
        """Middleware with mixed dependencies."""

        def __init__(self, instance_dep: AgentMiddleware) -> None:
            super().__init__()
            self.tools = []
            # Depend on both a class and an instance
            self.depends_on = (MiddlewareB, instance_dep)

    middleware_c = MixedDependencyMiddleware(middleware_a_instance)

    resolved = _resolve_middleware_dependencies([middleware_c])

    # Should have A (instance), B (auto-instantiated), and C
    assert len(resolved) == 3

    # Check that the custom instance was used
    a_instance = next(m for m in resolved if isinstance(m, MiddlewareA))
    assert a_instance is middleware_a_instance
    assert a_instance.value == "instance_A"

    # Check that B was auto-instantiated
    b_instance = next(m for m in resolved if isinstance(m, MiddlewareB))
    assert b_instance.value == "B"  # Default value
