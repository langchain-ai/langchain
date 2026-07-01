"""Tests for deterministic middleware orchestration MVP."""
# ruff: noqa: ARG002, PGH003

import asyncio
from typing import Any

import pytest
from langgraph.runtime import Runtime

from langchain.agents import create_agent
from langchain.agents.middleware.stack import MiddlewareSpec, MiddlewareStack
from langchain.agents.middleware.types import AgentMiddleware, AgentState
from tests.unit_tests.agents.model import FakeToolCallingModel


class DummyMiddleware(AgentMiddleware[AgentState[Any], Any, Any]):
    def __init__(self, name: str, overrides: list[str] | None = None) -> None:
        self._name = name
        self.overrides = overrides or []
        self.log: list[str] = []

    @property
    def name(self) -> str:
        return self._name

    def before_agent(self, state: AgentState[Any], runtime: Runtime[Any]) -> dict[str, Any] | None:
        self.log.append(f"{self.name}.before_agent")
        return {"messages": [{"role": "user", "content": self.name}]}


class DummyDelayMiddleware(AgentMiddleware[AgentState[Any], Any, Any]):
    def __init__(self, name: str, delay: float, key: str, val: str) -> None:
        self._name = name
        self.delay = delay
        self.key = key
        self.val = val
        self.log: list[str] = []

    @property
    def name(self) -> str:
        return self._name

    async def abefore_agent(
        self, state: AgentState[Any], runtime: Runtime[Any]
    ) -> dict[str, Any] | None:
        await asyncio.sleep(self.delay)
        self.log.append(f"{self.name}.abefore_agent")
        return {self.key: self.val}


class DummyWrapMiddleware(AgentMiddleware[AgentState[Any], Any, Any]):
    @property
    def name(self) -> str:
        return "wrap_middleware"

    def wrap_model_call(self, request: Any, handler: Any) -> Any:
        return handler(request)


def test_resolution_and_validation() -> None:
    # 1. Orders by priority desc, then name asc
    s1 = MiddlewareSpec(middleware=DummyMiddleware("b"), priority=10)
    s2 = MiddlewareSpec(middleware=DummyMiddleware("a"), priority=10)
    s3 = MiddlewareSpec(middleware=DummyMiddleware("c"), priority=100)
    stack = MiddlewareStack([s1, s2, s3])
    assert stack.execution_plan() == [("c", "sequential"), ("a", "sequential"), ("b", "sequential")]

    # 2. depends_on is enforced
    s4 = MiddlewareSpec(middleware=DummyMiddleware("d"), priority=1000, depends_on=["e"])
    s5 = MiddlewareSpec(middleware=DummyMiddleware("e"), priority=10)
    stack2 = MiddlewareStack([s4, s5])
    assert stack2.execution_plan() == [("e", "sequential"), ("d", "sequential")]

    # 3. Cycle raises ValueError
    s6 = MiddlewareSpec(middleware=DummyMiddleware("f"), depends_on=["g"])
    s7 = MiddlewareSpec(middleware=DummyMiddleware("g"), depends_on=["f"])
    with pytest.raises(ValueError, match="Dependency cycle"):
        MiddlewareStack([s6, s7])

    # 4. Unknown depends_on
    s8 = MiddlewareSpec(middleware=DummyMiddleware("h"), depends_on=["unknown"])
    with pytest.raises(ValueError, match="unknown target"):
        MiddlewareStack([s8])

    # 5. Duplicate names
    with pytest.raises(ValueError, match="Duplicate middleware name"):
        MiddlewareStack(
            [
                MiddlewareSpec(middleware=DummyMiddleware("i")),
                MiddlewareSpec(middleware=DummyMiddleware("i")),
            ]
        )

    # 6. mode="batched"
    with pytest.raises(NotImplementedError, match="batched"):
        MiddlewareSpec(middleware=DummyMiddleware("j"), mode="batched")

    # 7. Unsafe concurrent candidate
    with pytest.raises(ValueError, match="mutating hook"):
        MiddlewareStack(
            [MiddlewareSpec(middleware=DummyWrapMiddleware(), mode="parallel_readonly")]
        )

    # 8. Adjacent parallel_readonly collapse
    p1 = MiddlewareSpec(middleware=DummyMiddleware("p1"), mode="parallel_readonly")
    p2 = MiddlewareSpec(middleware=DummyMiddleware("p2"), mode="parallel_readonly")
    stack3 = MiddlewareStack([p1, p2])
    plan = stack3.resolve()
    assert len(plan) == 1
    assert "ParallelGroup" in plan[0].name

    # 9. execution_plan() returns (name, mode)
    assert stack3.execution_plan() == [("p1", "parallel_readonly"), ("p2", "parallel_readonly")]


@pytest.mark.asyncio
async def test_deterministic_merge() -> None:
    # 10. Disjoint keys merge
    m1 = DummyDelayMiddleware("m1", 0.05, "key1", "val1")
    m2 = DummyDelayMiddleware("m2", 0.01, "key2", "val2")
    stack = MiddlewareStack(
        [
            MiddlewareSpec(middleware=m1, mode="parallel_readonly", priority=10),
            MiddlewareSpec(middleware=m2, mode="parallel_readonly", priority=5),
        ]
    )

    # 11. Out of order completion still deterministic
    # m2 finishes first because delay 0.01 < 0.05
    # The output should still be deterministically merged
    group = stack.resolve()[0]
    # We test the composite's hook
    res = await group.__class__.abefore_agent(group, {}, None)  # type: ignore

    assert res == {"key1": "val1", "key2": "val2"}
    assert m2.log == ["m2.abefore_agent"]
    assert m1.log == ["m1.abefore_agent"]

    # 12. Same key conflict
    m3 = DummyDelayMiddleware("m3", 0.01, "key", "val3")
    m4 = DummyDelayMiddleware("m4", 0.01, "key", "val4")
    stack_conflict = MiddlewareStack(
        [
            MiddlewareSpec(middleware=m3, mode="parallel_readonly"),
            MiddlewareSpec(middleware=m4, mode="parallel_readonly"),
        ]
    )

    group_conflict = stack_conflict.resolve()[0]
    with pytest.raises(ValueError, match="Conflict during deterministic merge"):
        await group_conflict.__class__.abefore_agent(group_conflict, {}, None)  # type: ignore


def test_integration() -> None:
    # 14. Build a stack and pass to create_agent
    class TransformMiddleware(AgentMiddleware[AgentState[Any], Any, Any]):
        @property
        def name(self) -> str:
            return "transform"

        def before_model(
            self, state: AgentState[Any], runtime: Runtime[Any]
        ) -> dict[str, Any] | None:
            return {"messages": [{"role": "system", "content": "transformed"}]}

    class LoggingMiddleware(AgentMiddleware[AgentState[Any], Any, Any]):
        def __init__(self) -> None:
            self.called = False

        @property
        def name(self) -> str:
            return "logging"

        def after_model(
            self, state: AgentState[Any], runtime: Runtime[Any]
        ) -> dict[str, Any] | None:
            self.called = True
            return None

    class MetricsMiddleware(AgentMiddleware[AgentState[Any], Any, Any]):
        def __init__(self) -> None:
            self.called = False

        @property
        def name(self) -> str:
            return "metrics"

        def after_model(
            self, state: AgentState[Any], runtime: Runtime[Any]
        ) -> dict[str, Any] | None:
            self.called = True
            return None

    logger = LoggingMiddleware()
    metrics = MetricsMiddleware()
    stack = MiddlewareStack(
        [
            MiddlewareSpec(middleware=TransformMiddleware(), priority=100),
            MiddlewareSpec(middleware=logger, mode="parallel_readonly"),
            MiddlewareSpec(middleware=metrics, mode="parallel_readonly"),
        ]
    )

    model = FakeToolCallingModel()
    agent = create_agent(model=model, middleware=stack)

    res = agent.invoke({"messages": [{"role": "user", "content": "hello"}]})

    assert logger.called
    assert metrics.called
    assert "transformed" in str(res["messages"][1])

    assert stack.execution_plan() == [
        ("transform", "sequential"),
        ("logging", "parallel_readonly"),
        ("metrics", "parallel_readonly"),
    ]
