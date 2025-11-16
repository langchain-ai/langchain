from __future__ import annotations

from functools import partial
from typing import Sequence

import pytest

from langchain.agents.factory import _resolve_middleware, create_agent
from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    MiddlewareOrderCycleError,
    MiddlewareSpec,
    OrderingConstraints,
)
from langchain_core.language_models import FakeListChatModel
from langchain_core.messages import HumanMessage


class BareMiddleware(AgentMiddleware):
    """Minimal middleware used for unit tests."""

    def __init__(
        self,
        mid: str,
        *,
        priority: float | None = None,
        tags: Sequence[str] | None = None,
    ) -> None:
        self.id = mid
        self.priority = priority
        self.tags = tuple(tags or ())
        self.tools = []

    @property
    def name(self) -> str:  # pragma: no cover - exercised indirectly
        return self.id or super().name


class SharedMiddleware(BareMiddleware):
    def __init__(self, marker: str) -> None:
        super().__init__("shared")
        self.marker = marker


class ParentWithShared(BareMiddleware):
    def __init__(self, mid: str, marker: str, *, strategy: str = "first_wins") -> None:
        super().__init__(mid)
        self._marker = marker
        self._strategy = strategy

    def requires(self) -> Sequence[MiddlewareSpec[AgentState, object]]:
        return [
            MiddlewareSpec(
                factory=lambda: SharedMiddleware(self._marker),
                merge_strategy=self._strategy,
            )
        ]


class DependencyMiddleware(BareMiddleware):
    def __init__(self) -> None:
        super().__init__("dependency", tags=("helper",))


class ControllerMiddleware(BareMiddleware):
    def __init__(self) -> None:
        super().__init__("controller")

    def requires(self) -> Sequence[MiddlewareSpec[AgentState, object]]:
        return [MiddlewareSpec(factory=DependencyMiddleware)]


class OrderedControllerMiddleware(ControllerMiddleware):
    def requires(self) -> Sequence[MiddlewareSpec[AgentState, object]]:
        return [
            MiddlewareSpec(
                factory=DependencyMiddleware,
                ordering=OrderingConstraints(
                    after=("start",),
                    before=("tag:exit",),
                ),
            )
        ]


class PriorityDependency(BareMiddleware):
    def __init__(self, mid: str, priority: float) -> None:
        super().__init__(mid, priority=priority)


class PriorityParent(BareMiddleware):
    def __init__(self) -> None:
        super().__init__("priority-parent")

    def requires(self) -> Sequence[MiddlewareSpec[AgentState, object]]:
        return [
            MiddlewareSpec(factory=lambda: PriorityDependency("low", 1)),
            MiddlewareSpec(factory=lambda: PriorityDependency("high", 10)),
        ]


class CycleA(BareMiddleware):
    def __init__(self) -> None:
        super().__init__("cycle-a")

    def requires(self) -> Sequence[MiddlewareSpec[AgentState, object]]:
        return [MiddlewareSpec(factory=CycleB)]


class CycleB(BareMiddleware):
    def __init__(self) -> None:
        super().__init__("cycle-b")

    def requires(self) -> Sequence[MiddlewareSpec[AgentState, object]]:
        return [MiddlewareSpec(factory=CycleA)]


class RecordingMiddleware(AgentMiddleware):
    """Middleware that records hook execution order."""

    def __init__(
        self,
        mid: str,
        log: list[str],
        *,
        priority: float | None = None,
        tags: Sequence[str] | None = None,
    ) -> None:
        self.id = mid
        self._name = mid
        self.priority = priority
        self.tags = tuple(tags or ())
        self._log = log
        self.tools = []

    @property
    def name(self) -> str:
        return self._name

    def before_model(self, state, runtime) -> None:  # noqa: ARG002
        self._log.append(f"{self.id}:before_model")

    def after_model(self, state, runtime) -> None:  # noqa: ARG002
        self._log.append(f"{self.id}:after_model")

    def wrap_tool_call(self, request, handler):  # noqa: ARG002
        return handler(request)

    async def awrap_tool_call(self, request, handler):  # noqa: ARG002
        return await handler(request)


class DeepMiddleware(RecordingMiddleware):
    def __init__(self, log: list[str]) -> None:
        super().__init__("deep", log)


class InnerMiddleware(RecordingMiddleware):
    def __init__(self, log: list[str]) -> None:
        super().__init__("inner", log)

    def requires(self) -> Sequence[MiddlewareSpec[AgentState, object]]:
        return [MiddlewareSpec(factory=partial(DeepMiddleware, self._log))]


class OuterMiddleware(RecordingMiddleware):
    def __init__(self, log: list[str]) -> None:
        super().__init__("outer", log)

    def requires(self) -> Sequence[MiddlewareSpec[AgentState, object]]:
        return [MiddlewareSpec(factory=partial(InnerMiddleware, self._log))]


class SharedLoggingMiddleware(RecordingMiddleware):
    def __init__(self, log: list[str], marker: str) -> None:
        super().__init__("shared", log, tags=("shared",))
        self.marker = marker

    def before_model(self, state, runtime) -> None:  # noqa: ARG002
        self._log.append(f"shared:{self.marker}:before_model")

    def after_model(self, state, runtime) -> None:  # noqa: ARG002
        self._log.append(f"shared:{self.marker}:after_model")


class ParentOne(RecordingMiddleware):
    def __init__(self, log: list[str]) -> None:
        super().__init__("parent-one", log)

    def requires(self) -> Sequence[MiddlewareSpec[AgentState, object]]:
        return [
            MiddlewareSpec(
                factory=partial(SharedLoggingMiddleware, self._log, "first"),
                merge_strategy="first_wins",
            )
        ]


class ParentTwo(RecordingMiddleware):
    def __init__(self, log: list[str]) -> None:
        super().__init__("parent-two", log)

    def requires(self) -> Sequence[MiddlewareSpec[AgentState, object]]:
        return [
            MiddlewareSpec(
                factory=partial(SharedLoggingMiddleware, self._log, "second"),
                merge_strategy="first_wins",
            )
        ]


class AnonymousMiddleware(AgentMiddleware):
    def __init__(self, marker: str) -> None:
        self.marker = marker
        self.tools: list = []


def test_requires_flattens_dependencies() -> None:
    resolved = _resolve_middleware([ControllerMiddleware()])
    ids = [mw.id for mw in resolved]
    assert ids == ["dependency", "controller"]


def test_first_wins_deduplication_keeps_initial_instance() -> None:
    resolved = _resolve_middleware(
        [ParentWithShared("one", "first"), ParentWithShared("two", "second")]
    )
    shared = [mw for mw in resolved if mw.id == "shared"]
    assert len(shared) == 1
    assert shared[0].marker == "first"
    assert [mw.id for mw in resolved] == ["shared", "one", "two"]


def test_last_wins_replaces_existing_instance() -> None:
    resolved = _resolve_middleware(
        [
            ParentWithShared("one", "first", strategy="last_wins"),
            ParentWithShared("two", "second", strategy="last_wins"),
        ]
    )
    shared = [mw for mw in resolved if mw.id == "shared"]
    assert len(shared) == 1
    assert shared[0].marker == "second"


def test_duplicate_without_merge_strategy_errors() -> None:
    with pytest.raises(ValueError, match="Duplicate middleware id 'shared'"):
        _resolve_middleware(
            [
                ParentWithShared("one", "first", strategy="error"),
                ParentWithShared("two", "second", strategy="error"),
            ]
        )


def test_anonymous_instances_preserve_order_without_conflict() -> None:
    first = AnonymousMiddleware("first")
    second = AnonymousMiddleware("second")

    resolved = _resolve_middleware([first, second])

    assert resolved[:2] == [first, second]
    assert [mw.marker for mw in resolved] == ["first", "second"]
    assert resolved[0].id == "AnonymousMiddleware"
    assert resolved[1].id.startswith(
        f"{AnonymousMiddleware.__module__}.{AnonymousMiddleware.__qualname__}#"
    )


def test_ordering_constraints_with_ids_and_tags() -> None:
    start = BareMiddleware("start", tags=("entry",))
    terminus = BareMiddleware("terminus", tags=("exit",))
    controller = OrderedControllerMiddleware()

    resolved = _resolve_middleware([start, terminus, controller])
    assert [mw.id for mw in resolved] == ["start", "dependency", "terminus", "controller"]


def test_ordering_constraint_unknown_tag_raises() -> None:
    class TagController(BareMiddleware):
        def __init__(self) -> None:
            super().__init__("tag-controller")

        def requires(self) -> Sequence[MiddlewareSpec[AgentState, object]]:
            return [
                MiddlewareSpec(
                    factory=DependencyMiddleware,
                    ordering=OrderingConstraints(after=("tag:missing",)),
                )
            ]

    with pytest.raises(ValueError, match="unknown tag 'missing'"):
        _resolve_middleware([TagController()])


def test_priority_breaks_ties_when_order_equal() -> None:
    resolved = _resolve_middleware([PriorityParent()])
    ids = [mw.id for mw in resolved]
    assert ids == ["high", "low", "priority-parent"]


def test_cycle_detection_raises_error() -> None:
    with pytest.raises(MiddlewareOrderCycleError, match="cycle-a"):
        _resolve_middleware([CycleA()])


def test_nested_dependencies_execute_in_order() -> None:
    log: list[str] = []
    agent = create_agent(
        model=FakeListChatModel(responses=["done"]),
        middleware=[OuterMiddleware(log)],
    )

    agent.invoke({"messages": [HumanMessage(content="hi")]})

    assert log[:3] == [
        "deep:before_model",
        "inner:before_model",
        "outer:before_model",
    ]
    assert log[3:] == [
        "outer:after_model",
        "inner:after_model",
        "deep:after_model",
    ]


def test_shared_dependency_runs_once_with_first_wins() -> None:
    log: list[str] = []
    agent = create_agent(
        model=FakeListChatModel(responses=["done"]),
        middleware=[ParentOne(log), ParentTwo(log)],
    )

    agent.invoke({"messages": [HumanMessage(content="go")]})

    before_entries = [entry for entry in log if entry.endswith("before_model")]
    after_entries = [entry for entry in log if entry.endswith("after_model")]

    assert before_entries == [
        "shared:first:before_model",
        "parent-one:before_model",
        "parent-two:before_model",
    ]
    assert after_entries == [
        "parent-two:after_model",
        "parent-one:after_model",
        "shared:first:after_model",
    ]
