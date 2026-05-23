"""Trajectory-based assertion framework for real-model middleware evals.

Shape mirrors the deepagents eval framework so anyone moving between the two
repos sees the same API. Two-tier scoring:

- ``.success(...)`` — correctness assertions that hard-fail the test
- ``.expect(...)`` — efficiency assertions that are logged but never fail

Use ``run_agent(...)`` to invoke a `create_agent` graph against a query and
score the resulting trajectory.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from langgraph.graph.state import CompiledStateGraph


# ---------------------------------------------------------------------------
# Trajectory model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AgentStep:
    """One step of the agent: an AIMessage and the tool results it triggered."""

    index: int
    action: AIMessage
    observations: list[ToolMessage]


@dataclass(frozen=True)
class AgentTrajectory:
    """The sequence of agent steps for a single invocation."""

    steps: list[AgentStep]

    @property
    def final_text(self) -> str:
        """Text content of the last AIMessage (the loop-terminating message)."""
        if not self.steps:
            return ""
        content = self.steps[-1].action.content
        if isinstance(content, str):
            return content
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text") or "")
        return "".join(parts)

    @property
    def tool_call_count(self) -> int:
        return sum(len(s.action.tool_calls or []) for s in self.steps)

    def pretty(self) -> str:
        """Human-readable trajectory dump for failure messages."""
        lines: list[str] = []
        for step in self.steps:
            lines.append(f"step {step.index}:")
            lines.extend(
                f"  - tool_call: {tc.get('name')} {tc.get('args')}"
                for tc in step.action.tool_calls or []
            )
            text = (
                step.action.content
                if isinstance(step.action.content, str)
                else self.final_text
                if step is self.steps[-1]
                else ""
            )
            if text and text.strip():
                preview = text.strip().replace("\n", " ")
                if len(preview) > 200:
                    preview = preview[:197] + "..."
                lines.append(f"  text: {preview}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Assertion base classes
# ---------------------------------------------------------------------------


class SuccessAssertion:
    """Hard-fail correctness assertion."""

    def check(self, trajectory: AgentTrajectory) -> bool:
        raise NotImplementedError

    def describe_failure(self, trajectory: AgentTrajectory) -> str:
        raise NotImplementedError


class EfficiencyAssertion:
    """Logged-but-never-fail trajectory-shape assertion."""

    def check(self, trajectory: AgentTrajectory) -> bool:
        raise NotImplementedError

    def describe_failure(self, trajectory: AgentTrajectory) -> str:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Concrete success assertions
# ---------------------------------------------------------------------------


_ZERO_WIDTH = {
    ord("\u200b"): None,  # zero-width space
    ord("\u200c"): None,  # zero-width non-joiner
    ord("\u200d"): None,  # zero-width joiner
    ord("\ufeff"): None,  # zero-width no-break space / BOM
}


def _normalize(text: str) -> str:
    """Strip zero-width characters; some models insert them for rendering."""
    return text.translate(_ZERO_WIDTH)


@dataclass(frozen=True)
class FinalTextContains(SuccessAssertion):
    """Assert that the final AIMessage text contains a substring."""

    text: str
    case_insensitive: bool = False

    def check(self, trajectory: AgentTrajectory) -> bool:
        haystack = _normalize(trajectory.final_text)
        needle = _normalize(self.text)
        if self.case_insensitive:
            haystack = haystack.lower()
            needle = needle.lower()
        return needle in haystack

    def describe_failure(self, trajectory: AgentTrajectory) -> str:
        return (
            f"Expected final text to contain {self.text!r} "
            f"(case_insensitive={self.case_insensitive}); got: "
            f"{_normalize(trajectory.final_text)!r}"
        )


@dataclass(frozen=True)
class FinalTextExcludes(SuccessAssertion):
    """Assert that the final AIMessage text does NOT contain a substring."""

    text: str
    case_insensitive: bool = False

    def check(self, trajectory: AgentTrajectory) -> bool:
        haystack = _normalize(trajectory.final_text)
        needle = _normalize(self.text)
        if self.case_insensitive:
            haystack = haystack.lower()
            needle = needle.lower()
        return needle not in haystack

    def describe_failure(self, trajectory: AgentTrajectory) -> str:
        return (
            f"Expected final text NOT to contain {self.text!r}; "
            f"got: {_normalize(trajectory.final_text)!r}"
        )


@dataclass(frozen=True)
class FinalTextContainsAny(SuccessAssertion):
    """Assert that the final AIMessage contains AT LEAST ONE of these substrings.

    Useful when a behavior can be expressed in several equivalent phrasings
    (e.g., "unknown" / "no data" / "n/a" / "unable" for "value is missing").
    The any-of group still proves the behavior — a model that hallucinates
    instead of acknowledging the gap wouldn't include any of these phrases.
    """

    texts: tuple[str, ...]
    case_insensitive: bool = False

    def check(self, trajectory: AgentTrajectory) -> bool:
        haystack = _normalize(trajectory.final_text)
        if self.case_insensitive:
            haystack = haystack.lower()
        for t in self.texts:
            needle = _normalize(t)
            if self.case_insensitive:
                needle = needle.lower()
            if needle in haystack:
                return True
        return False

    def describe_failure(self, trajectory: AgentTrajectory) -> str:
        return (
            f"Expected final text to contain at least one of {list(self.texts)!r} "
            f"(case_insensitive={self.case_insensitive}); got: "
            f"{_normalize(trajectory.final_text)!r}"
        )


@dataclass(frozen=True)
class FinalTextMinLength(SuccessAssertion):
    """Assert that the final AIMessage text is at least ``n`` chars after strip."""

    n: int

    def check(self, trajectory: AgentTrajectory) -> bool:
        return len(trajectory.final_text.strip()) >= self.n

    def describe_failure(self, trajectory: AgentTrajectory) -> str:
        return (
            f"Expected final text length >= {self.n}, got "
            f"{len(trajectory.final_text.strip())}: "
            f"{_normalize(trajectory.final_text)!r}"
        )


# ---------------------------------------------------------------------------
# Concrete efficiency assertions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AgentSteps(EfficiencyAssertion):
    """Expected total number of agent steps in the trajectory."""

    n: int

    def check(self, trajectory: AgentTrajectory) -> bool:
        return len(trajectory.steps) == self.n

    def describe_failure(self, trajectory: AgentTrajectory) -> str:
        return f"Expected {self.n} agent steps, got {len(trajectory.steps)}"


@dataclass(frozen=True)
class MaxAgentSteps(EfficiencyAssertion):
    """Expected upper bound on the number of agent steps."""

    n: int

    def check(self, trajectory: AgentTrajectory) -> bool:
        return len(trajectory.steps) <= self.n

    def describe_failure(self, trajectory: AgentTrajectory) -> str:
        return f"Expected at most {self.n} agent steps, got {len(trajectory.steps)}"


@dataclass(frozen=True)
class ToolCallRequests(EfficiencyAssertion):
    """Expected total number of tool calls across the trajectory."""

    n: int

    def check(self, trajectory: AgentTrajectory) -> bool:
        return trajectory.tool_call_count == self.n

    def describe_failure(self, trajectory: AgentTrajectory) -> str:
        return f"Expected {self.n} tool calls, got {trajectory.tool_call_count}"


@dataclass(frozen=True)
class MaxToolCallRequests(EfficiencyAssertion):
    """Expected upper bound on total tool calls."""

    n: int

    def check(self, trajectory: AgentTrajectory) -> bool:
        return trajectory.tool_call_count <= self.n

    def describe_failure(self, trajectory: AgentTrajectory) -> str:
        return f"Expected at most {self.n} tool calls, got {trajectory.tool_call_count}"


@dataclass(frozen=True)
class ToolCall(EfficiencyAssertion):
    """Assert that a specific tool call appeared in the trajectory.

    When ``step`` is given (1-indexed), only that step is searched.
    """

    name: str
    step: int | None = None

    def check(self, trajectory: AgentTrajectory) -> bool:
        if self.step is not None:
            if self.step < 1 or self.step > len(trajectory.steps):
                return False
            steps = [trajectory.steps[self.step - 1]]
        else:
            steps = trajectory.steps
        for s in steps:
            for tc in s.action.tool_calls or []:
                if tc.get("name") == self.name:
                    return True
        return False

    def describe_failure(self, trajectory: AgentTrajectory) -> str:  # noqa: ARG002
        where = f" in step {self.step}" if self.step is not None else ""
        return f"Missing expected tool call{where}: name={self.name!r}"


# ---------------------------------------------------------------------------
# Factory functions (the public API used in test files)
# ---------------------------------------------------------------------------


def final_text_contains(text: str, *, case_insensitive: bool = False) -> FinalTextContains:
    return FinalTextContains(text=text, case_insensitive=case_insensitive)


def final_text_excludes(text: str, *, case_insensitive: bool = False) -> FinalTextExcludes:
    return FinalTextExcludes(text=text, case_insensitive=case_insensitive)


def final_text_contains_any(
    *texts: str, case_insensitive: bool = False
) -> FinalTextContainsAny:
    """Assert the final text contains at least one of ``texts``."""
    return FinalTextContainsAny(texts=tuple(texts), case_insensitive=case_insensitive)


def final_text_min_length(n: int) -> FinalTextMinLength:
    return FinalTextMinLength(n=n)


def agent_steps(n: int) -> AgentSteps:
    return AgentSteps(n=n)


def max_agent_steps(n: int) -> MaxAgentSteps:
    return MaxAgentSteps(n=n)


def tool_call_requests(n: int) -> ToolCallRequests:
    return ToolCallRequests(n=n)


def max_tool_call_requests(n: int) -> MaxToolCallRequests:
    return MaxToolCallRequests(n=n)


def tool_call(name: str, *, step: int | None = None) -> ToolCall:
    return ToolCall(name=name, step=step)


# ---------------------------------------------------------------------------
# TrajectoryScorer
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrajectoryScorer:
    """Builder for a two-tier scoring spec.

    Use ``.success(*assertions)`` to add hard-fail checks and
    ``.expect(...)`` to add logged-but-non-fatal efficiency checks.
    """

    _success: tuple[SuccessAssertion, ...] = ()
    _expectations: tuple[EfficiencyAssertion, ...] = ()

    def success(self, *assertions: SuccessAssertion) -> TrajectoryScorer:
        return TrajectoryScorer(
            _success=(*self._success, *assertions),
            _expectations=self._expectations,
        )

    def expect(
        self,
        *,
        agent_steps: int | None = None,
        tool_call_requests: int | None = None,
        tool_calls: list[ToolCall] | None = None,
    ) -> TrajectoryScorer:
        new: list[EfficiencyAssertion] = []
        if agent_steps is not None:
            new.append(AgentSteps(n=agent_steps))
        if tool_call_requests is not None:
            new.append(ToolCallRequests(n=tool_call_requests))
        if tool_calls is not None:
            new.extend(tool_calls)
        return TrajectoryScorer(
            _success=self._success,
            _expectations=(*self._expectations, *new),
        )


# ---------------------------------------------------------------------------
# Trajectory construction + run_agent
# ---------------------------------------------------------------------------


def _trajectory_from_messages(messages: list[AnyMessage]) -> AgentTrajectory:
    """Build a trajectory from the message list returned by ``agent.invoke``.

    The first message (HumanMessage) is skipped; each subsequent AIMessage
    starts a new step and the ToolMessages that follow it become its
    observations.
    """
    steps: list[AgentStep] = []
    current: AgentStep | None = None
    for msg in messages[1:]:
        if isinstance(msg, AIMessage):
            if current is not None:
                steps.append(current)
            current = AgentStep(index=len(steps) + 1, action=msg, observations=[])
        elif isinstance(msg, ToolMessage) and current is not None:
            current.observations.append(msg)
    if current is not None:
        steps.append(current)
    return AgentTrajectory(steps=steps)


def _assert_expectations(trajectory: AgentTrajectory, scorer: TrajectoryScorer) -> None:
    """Run the scorer's checks; hard-fail on success, log on efficiency."""
    # Efficiency assertions are reported but never fail the test.
    for exp in scorer._expectations:
        if not exp.check(trajectory):
            # Pytest-friendly print so the failure shows up in -v output.
            print(  # noqa: T201 - eval signal is intended user-facing output
                f"[eval expectation miss] {exp.describe_failure(trajectory)}"
            )

    # Success assertions fail the test if violated.
    for s in scorer._success:
        if not s.check(trajectory):
            detail = s.describe_failure(trajectory)
            message = f"success check failed: {detail}\n\ntrajectory:\n{trajectory.pretty()}"
            raise AssertionError(message)


def run_agent(
    agent: CompiledStateGraph[Any, Any],
    *,
    query: str,
    model: BaseChatModel,  # noqa: ARG001 - kept for parity with deepagents API
    scorer: TrajectoryScorer | None = None,
) -> AgentTrajectory:
    """Invoke an agent against a query and optionally score the trajectory."""
    config: dict[str, Any] = {"configurable": {"thread_id": str(uuid.uuid4())}}
    result = agent.invoke(  # type: ignore[call-overload]
        {"messages": [HumanMessage(content=query)]},
        config,
    )
    if not isinstance(result, Mapping):
        msg = f"Expected agent.invoke to return a Mapping, got {type(result)}"
        raise TypeError(msg)
    messages = result.get("messages")
    if not isinstance(messages, list):
        msg = f"Expected result['messages'] to be a list, got {type(messages)}"
        raise TypeError(msg)
    trajectory = _trajectory_from_messages(messages)
    if scorer is not None:
        _assert_expectations(trajectory, scorer)
    return trajectory
