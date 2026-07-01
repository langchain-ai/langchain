"""Orchestrator for middleware stacks."""
# ruff: noqa: FBT001, FBT002

from __future__ import annotations

import asyncio
import concurrent.futures
import typing
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from graphlib import CycleError, TopologicalSorter
from typing import TYPE_CHECKING, Any, Generic, Literal

if TYPE_CHECKING:
    from langgraph.runtime import Runtime

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ContextT,
    ResponseT,
    StateT,
)


@dataclass
class MiddlewareSpec(Generic[StateT, ContextT, ResponseT]):
    """Specification for middleware orchestration."""

    middleware: AgentMiddleware[StateT, ContextT, ResponseT]
    name: str = ""
    priority: int = 0
    depends_on: Sequence[str] = ()
    mode: Literal["sequential", "parallel_readonly", "batched"] = "sequential"
    condition: Callable[[Any], bool] | None = None

    def __post_init__(self) -> None:
        """Validate middleware specification fields."""
        if not self.name:
            self.name = getattr(self.middleware, "name", self.middleware.__class__.__name__)
        if not self.name:
            msg = "Middleware name cannot be empty."
            raise ValueError(msg)
        if self.mode == "batched":
            msg = "batched mode is not yet implemented."
            raise NotImplementedError(msg)
        if self.mode not in ("sequential", "parallel_readonly"):
            msg = f"Invalid mode '{self.mode}'"
            raise ValueError(msg)


class _ParallelReadOnlyMiddleware(AgentMiddleware[StateT, ContextT, ResponseT]):
    """Composite middleware that runs read-only members concurrently and merges their results."""

    def __init__(
        self,
        members: list[MiddlewareSpec[StateT, ContextT, ResponseT]],
        deterministic_merge: bool = True,
    ) -> None:
        self.members = members
        self.deterministic_merge = deterministic_merge
        self._name = f"ParallelGroup({','.join(m.name for m in members)})"

        # Merge tools and transformers from members
        self.tools = [t for m in self.members for t in getattr(m.middleware, "tools", [])]
        self.transformers = [
            t for m in self.members for t in getattr(m.middleware, "transformers", [])
        ]

    @property
    def name(self) -> str:
        return self._name

    def _merge_results(
        self, results: list[tuple[str, dict[str, Any] | None]]
    ) -> dict[str, Any] | None:
        """Deterministically merge state updates, checking for disjoint keys."""
        merged: dict[str, Any] = {}
        for member_name, result in results:
            if not result:
                continue
            for k, v in result.items():
                if k == "messages":
                    if "messages" not in merged:
                        merged["messages"] = []
                    merged["messages"].extend(v if isinstance(v, list) else [v])
                else:
                    if k in merged and merged[k] != v:
                        msg = (
                            f"Conflict during deterministic merge: '{k}' modified "
                            f"by '{member_name}' but was already modified differently. "
                            "parallel_readonly middleware must be disjoint."
                        )
                        raise ValueError(msg)
                    merged[k] = v
        return merged or None

    # Sync Hooks
    def _run_sync(
        self, hook_name: str, state: StateT, runtime: Runtime[ContextT]
    ) -> dict[str, Any] | None:
        results: list[tuple[str, dict[str, Any] | None]] = []
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, len(self.members))
        ) as executor:
            futures: list[tuple[str, concurrent.futures.Future[Any] | None]] = []
            for spec in self.members:
                method = getattr(
                    spec.middleware.__class__, hook_name, getattr(AgentMiddleware, hook_name)
                )
                if method is not getattr(AgentMiddleware, hook_name):
                    futures.append(
                        (spec.name, executor.submit(method, spec.middleware, state, runtime))
                    )
                else:
                    futures.append((spec.name, None))

            for name, future in futures:
                if future is not None:
                    results.append((name, future.result()))
                else:
                    results.append((name, None))

        return self._merge_results(results)

    def before_agent(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None:
        return self._run_sync("before_agent", state, runtime)

    def before_model(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None:
        return self._run_sync("before_model", state, runtime)

    def after_model(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None:
        return self._run_sync("after_model", state, runtime)

    def after_agent(self, state: StateT, runtime: Runtime[ContextT]) -> dict[str, Any] | None:
        return self._run_sync("after_agent", state, runtime)

    # Async Hooks
    async def _run_async(
        self, hook_name: str, state: StateT, runtime: Runtime[ContextT]
    ) -> dict[str, Any] | None:
        coroutines = []
        names = []
        for spec in self.members:
            names.append(spec.name)
            method = getattr(
                spec.middleware.__class__, hook_name, getattr(AgentMiddleware, hook_name)
            )
            if method is not getattr(AgentMiddleware, hook_name):
                coroutines.append(method(spec.middleware, state, runtime))
            else:

                async def _empty() -> None:
                    return None

                coroutines.append(_empty())

        results = await asyncio.gather(*coroutines)
        named_results = list(zip(names, results, strict=False))
        return self._merge_results(named_results)

    async def abefore_agent(
        self, state: StateT, runtime: Runtime[ContextT]
    ) -> dict[str, Any] | None:
        return await self._run_async("abefore_agent", state, runtime)

    async def abefore_model(
        self, state: StateT, runtime: Runtime[ContextT]
    ) -> dict[str, Any] | None:
        return await self._run_async("abefore_model", state, runtime)

    async def aafter_model(
        self, state: StateT, runtime: Runtime[ContextT]
    ) -> dict[str, Any] | None:
        return await self._run_async("aafter_model", state, runtime)

    async def aafter_agent(
        self, state: StateT, runtime: Runtime[ContextT]
    ) -> dict[str, Any] | None:
        return await self._run_async("aafter_agent", state, runtime)


class MiddlewareStack(Sequence[AgentMiddleware[StateT, ContextT, ResponseT]]):
    """An ordered, validated collection of specs that resolves to an execution plan."""

    def __init__(
        self,
        specs: Iterable[MiddlewareSpec[StateT, ContextT, ResponseT]],
        *,
        deterministic_merge: bool = True,
    ) -> None:
        """Initialize a new middleware stack with the given specifications."""
        self._specs = list(specs)
        self.deterministic_merge = deterministic_merge
        self._validate()
        self._resolved_plan = self._resolve()

    def _validate(self) -> None:
        names = set()
        for spec in self._specs:
            if spec.name in names:
                msg = f"Duplicate middleware name found: '{spec.name}'"
                raise ValueError(msg)
            names.add(spec.name)

            for dep in spec.depends_on:
                if dep not in [s.name for s in self._specs]:
                    msg = f"Middleware '{spec.name}' depends on unknown target '{dep}'."
                    raise ValueError(msg)

            if spec.mode == "parallel_readonly":
                # Check for unsafe overrides
                m_class = spec.middleware.__class__
                mutating_hooks = [
                    "wrap_model_call",
                    "awrap_model_call",
                    "wrap_tool_call",
                    "awrap_tool_call",
                ]
                for hook in mutating_hooks:
                    if getattr(m_class, hook) is not getattr(AgentMiddleware, hook):
                        msg = (
                            f"Middleware '{spec.name}' is mode='parallel_readonly' but overrides "
                            f"mutating hook '{hook}'. "
                            "parallel_readonly middleware must be read-only."
                        )
                        raise ValueError(msg)

    def _resolve(self) -> list[AgentMiddleware[StateT, ContextT, ResponseT]]:
        # graphlib.TopologicalSorter handles DAG sorting.
        # Add nodes with dependencies.
        graph: dict[str, list[str]] = {spec.name: list(spec.depends_on) for spec in self._specs}
        sorter = TopologicalSorter(graph)
        try:
            sorter.prepare()
        except CycleError as e:
            # e.args[1] usually contains the cycle sequence string
            msg = f"Dependency cycle detected in middleware stack: {e.args[1]}"
            raise ValueError(msg) from e

        spec_by_name = {spec.name: spec for spec in self._specs}

        ordered_specs: list[MiddlewareSpec[StateT, ContextT, ResponseT]] = []
        ready_queue = list(sorter.get_ready())

        while ready_queue:
            # Sort to pick highest priority first. Ties broken alphabetically by name.
            ready_queue.sort(key=lambda n: (-spec_by_name[n].priority, n))

            node = ready_queue.pop(0)
            ordered_specs.append(spec_by_name[node])

            sorter.done(node)
            ready_queue.extend(sorter.get_ready())

        # Collapse contiguous parallel_readonly specs
        execution_plan: list[AgentMiddleware[StateT, ContextT, ResponseT]] = []
        current_parallel_group: list[MiddlewareSpec[StateT, ContextT, ResponseT]] = []

        for spec in ordered_specs:
            if spec.mode == "parallel_readonly":
                current_parallel_group.append(spec)
            else:
                if current_parallel_group:
                    if len(current_parallel_group) == 1:
                        execution_plan.append(current_parallel_group[0].middleware)
                    else:
                        execution_plan.append(
                            _ParallelReadOnlyMiddleware(
                                current_parallel_group, self.deterministic_merge
                            )
                        )
                    current_parallel_group = []
                execution_plan.append(spec.middleware)

        if current_parallel_group:
            if len(current_parallel_group) == 1:
                execution_plan.append(current_parallel_group[0].middleware)
            else:
                execution_plan.append(
                    _ParallelReadOnlyMiddleware(current_parallel_group, self.deterministic_merge)
                )

        # Store for introspection
        self._static_plan = [(s.name, str(s.mode)) for s in ordered_specs]
        return execution_plan

    def execution_plan(self) -> list[tuple[str, str]]:
        """Return the resolved (name, mode) order for debugging."""
        return self._static_plan

    def resolve(self) -> list[AgentMiddleware[StateT, ContextT, ResponseT]]:
        """Return the ordered list of middleware composites."""
        return self._resolved_plan

    def __iter__(self) -> typing.Iterator[AgentMiddleware[StateT, ContextT, ResponseT]]:
        """Return an iterator over the resolved execution plan."""
        return iter(self._resolved_plan)

    def __len__(self) -> int:
        """Return the number of middleware in the resolved execution plan."""
        return len(self._resolved_plan)

    def __getitem__(self, index: int | slice) -> Any:
        """Return the middleware at the given index from the resolved execution plan."""
        return self._resolved_plan[index]
