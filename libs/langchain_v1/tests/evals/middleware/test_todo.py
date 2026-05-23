r"""Real-model evals for `TodoListMiddleware`.

These tests verify that `WRITE_TODOS_SYSTEM_PROMPT` produces the intended
agent-loop behavior: the substantive final answer lands in the
loop-terminating message (the AIMessage with no tool calls), not crammed
into the same turn as the final `write_todos(completed)` call.

Each test runs `create_agent` with `TodoListMiddleware` against a query that
historically triggered the "wasted post-tool turn" pattern under prior
versions of the prompt. Pass criteria check the final AIMessage for the
expected substantive content; efficiency expectations log trajectory shape
without failing the test.

Run:
    cd libs/langchain_v1
    uv run --group test pytest tests/evals/middleware/test_todo.py \\
        -v --model claude-sonnet-4-6
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from langchain_core.tools import tool

from langchain.agents import create_agent
from langchain.agents.middleware import TodoListMiddleware
from tests.evals.utils import (
    AgentTrajectory,
    MaxToolCallRequests,
    SuccessAssertion,
    TrajectoryScorer,
    final_text_contains,
    final_text_contains_any,
    final_text_min_length,
    run_agent,
    tool_call,
)

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

pytestmark = [pytest.mark.eval_category("middleware/todo")]


# ---------------------------------------------------------------------------
# Fixtures for tool-bearing tasks
# ---------------------------------------------------------------------------


@tool
def lookup_population(city: str) -> str:
    """Return the population of a city as a string."""
    data = {
        "tokyo": "13,960,000",
        "delhi": "32,900,000",
        "shanghai": "29,200,000",
        "cairo": "21,800,000",
    }
    return data.get(city.lower(), "unknown")


@tool
def lookup_area_km2(city: str) -> str:
    """Return the area of a city in square kilometers as a string."""
    data = {
        "tokyo": "2,194",
        "delhi": "1,484",
        "shanghai": "6,341",
        "cairo": "606",
    }
    return data.get(city.lower(), "unknown")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.eval_tier("baseline")
@pytest.mark.langsmith
def test_density_rank_lands_in_final_message(model: BaseChatModel) -> None:
    """Substantive ranked answer must land in the last AIMessage.

    The default prompt prior to the loop-contract fix would put the ranked
    output in the same turn as the final `write_todos(completed)` call, and
    the loop-terminating message would be a 20-30ch wrap-up that omits the
    cities by name. This test fails in that regime.
    """
    agent = create_agent(
        model=model,
        tools=[lookup_population, lookup_area_km2],
        middleware=[TodoListMiddleware()],
    )
    run_agent(
        agent,
        model=model,
        query=(
            "Rank Tokyo, Delhi, and Shanghai by population density (people per "
            "km²) from highest to lowest. Look up the population and area for "
            "each city, compute density for each, and present the ranking. Use "
            "a todo list to plan and track your work."
        ),
        scorer=TrajectoryScorer()
        .expect(
            agent_steps=6,
            tool_call_requests=8,
            tool_calls=[tool_call(name="write_todos")],
        )
        .success(
            final_text_contains("tokyo", case_insensitive=True),
            final_text_contains("delhi", case_insensitive=True),
            final_text_contains("shanghai", case_insensitive=True),
            final_text_min_length(80),
        ),
    )


@pytest.mark.eval_tier("baseline")
@pytest.mark.langsmith
def test_population_compare_lands_in_final_message(model: BaseChatModel) -> None:
    """Binary comparison answer must land in the last AIMessage."""
    agent = create_agent(
        model=model,
        tools=[lookup_population],
        middleware=[TodoListMiddleware()],
    )
    run_agent(
        agent,
        model=model,
        query=(
            "Which has more people: Tokyo or Delhi? Use a todo list to plan, "
            "look up the population for each city, and tell me which has more "
            "and by how much."
        ),
        scorer=TrajectoryScorer()
        .expect(
            agent_steps=4,
            tool_call_requests=4,
            tool_calls=[tool_call(name="write_todos")],
        )
        .success(
            final_text_contains("delhi", case_insensitive=True),
            final_text_contains("18,940,000"),
            final_text_min_length(50),
        ),
    )


@pytest.mark.eval_tier("baseline")
@pytest.mark.langsmith
def test_trivial_arithmetic_skips_write_todos(model: BaseChatModel) -> None:
    """One-shot arithmetic must NOT invoke `write_todos`.

    This is the cross-model baseline check for the "skip for simple" guidance
    in `WRITE_TODOS_SYSTEM_PROMPT`. The query is pure single-step arithmetic
    with no list-shape or planning bait, so every model with a working
    skip-for-simple disposition should answer directly. If a future prompt
    change accidentally removes the "skip for simple" line, every model
    starts cargo-culting `write_todos` on every query and this test catches
    it.
    """
    agent = create_agent(
        model=model,
        tools=[],
        middleware=[TodoListMiddleware()],
    )
    run_agent(
        agent,
        model=model,
        query="What is 47 + 18?",
        scorer=TrajectoryScorer()
        .expect(agent_steps=1, tool_call_requests=0)
        .success(
            final_text_contains("65"),
            _max_tool_calls_success(0),
        ),
    )


@pytest.mark.eval_tier("hillclimb")
@pytest.mark.langsmith
def test_trivial_plan_skips_write_todos(model: BaseChatModel) -> None:
    """Trivial request should NOT invoke `write_todos`.

    The system prompt's "skip for simple tasks" guidance is what prevents
    the model from cargo-culting `write_todos` on a one-shot conversational
    request. Even though the word "Plan" appears in the query, the answer is
    knowledge-only and the agent should respond in a single turn with no
    tool calls.

    Hillclimb tier (not baseline): some models (Llama, Gemini in our
    observations) treat the "Plan" + 3-explicit-items shape of this query
    literally enough to invoke `write_todos` despite the "skip for simple"
    guidance. Claude/GPT-5/DeepSeek pass it. Useful progress signal but not
    a hard regression gate. See `test_trivial_arithmetic_skips_write_todos`
    for the cross-model baseline check.
    """
    agent = create_agent(
        model=model,
        tools=[],
        middleware=[TodoListMiddleware()],
    )
    run_agent(
        agent,
        model=model,
        query="Plan a simple 3-course dinner menu (appetizer, main, dessert).",
        scorer=TrajectoryScorer()
        .expect(agent_steps=1, tool_call_requests=0)
        .success(
            final_text_contains("appetizer", case_insensitive=True),
            final_text_contains("main", case_insensitive=True),
            final_text_contains("dessert", case_insensitive=True),
            # Tool-budget check is a success assertion (hard-fail). Trivial
            # tasks must not trigger any tool call.
            _max_tool_calls_success(0),
        ),
    )


@pytest.mark.eval_tier("baseline")
@pytest.mark.langsmith
def test_rank_with_unknown_lookup_lands_in_final_message(model: BaseChatModel) -> None:
    """Substantive answer must land in the last AIMessage when a lookup fails.

    Atlantis is intentionally absent from the lookup data — `lookup_population`
    and `lookup_area_km2` both return ``"unknown"`` for it. The agent has to
    revise its plan, present a partial ranking for the cities it could look up,
    and surface the missing data. Without the loop-contract fix, the model
    sometimes cramps the substantive partial ranking into the same turn as the
    final `write_todos(completed)` and the loop-terminating message is empty
    or a useless recap — that's the failure mode this test catches.
    """
    agent = create_agent(
        model=model,
        tools=[lookup_population, lookup_area_km2],
        middleware=[TodoListMiddleware()],
    )
    run_agent(
        agent,
        model=model,
        query=(
            "Rank Tokyo, Atlantis, and Cairo by population density (people "
            "per km²) from highest to lowest. Look up the population and area "
            "for each city, compute density for each, and present the "
            "ranking. Use a todo list to plan and track your work."
        ),
        scorer=TrajectoryScorer()
        .expect(
            agent_steps=6,
            tool_call_requests=8,
            tool_calls=[tool_call(name="write_todos")],
        )
        .success(
            final_text_contains("tokyo", case_insensitive=True),
            final_text_contains("cairo", case_insensitive=True),
            final_text_contains("atlantis", case_insensitive=True),
            # Acknowledge that Atlantis has no real data — any of these
            # phrasings counts. A hallucinating model that confidently
            # invents stats for Atlantis would not include any of them,
            # so the check still catches the failure mode (model invents
            # data instead of surfacing the gap).
            final_text_contains_any(
                "unknown",
                "no data",
                "n/a",
                "unable",
                "cannot be ranked",
                "not available",
                "no information",
                "missing",
                "mythical",
                "fictional",
                "legendary",
                case_insensitive=True,
            ),
            # 150 ch is "two city stats + missing-data ack" in compact form.
            # Pure recap-only outputs cannot squeeze the substantive content
            # of three city comparisons into that budget. The earlier 200
            # threshold was a Sonnet-calibrated guess that rejected
            # DeepSeek's correct-but-tersely-worded answers.
            final_text_min_length(150),
        ),
    )


@pytest.mark.eval_tier("hillclimb")
@pytest.mark.langsmith
def test_design_api_lands_in_final_message(model: BaseChatModel) -> None:
    """Design/synthesis answer must land in the last AIMessage.

    No external tools; the agent has only `write_todos`. The substantive
    output is a small API design that must mention endpoints, authentication,
    and at least one HTTP method. Hillclimb tier: the bug fires less
    deterministically here than on density-rank, but the task gives diverse
    non-lookup coverage.
    """
    agent = create_agent(
        model=model,
        tools=[],
        middleware=[TodoListMiddleware()],
    )
    run_agent(
        agent,
        model=model,
        query=(
            "Design a small REST API for a todo-list application. Use a todo "
            "list to plan and track your work. Cover at minimum: endpoints "
            "with their HTTP methods, request/response shape, and "
            "authentication approach."
        ),
        scorer=TrajectoryScorer()
        .expect(
            agent_steps=5,
            tool_call_requests=4,
            tool_calls=[tool_call(name="write_todos")],
        )
        .success(
            final_text_contains("endpoint", case_insensitive=True),
            final_text_contains("authentication", case_insensitive=True),
            final_text_contains("POST", case_insensitive=True),
            final_text_min_length(200),
        ),
    )


@pytest.mark.eval_tier("hillclimb")
@pytest.mark.langsmith
def test_density_cairo_lands_in_final_message(model: BaseChatModel) -> None:
    """Single-density answer must land in the last AIMessage.

    Hillclimb tier: this task is partially passable under the buggy default
    (the model's recap-summary often happens to include the computed number),
    so it's a softer signal than density-rank / population-compare. Useful as
    a directional metric, not a regression gate.
    """
    agent = create_agent(
        model=model,
        tools=[lookup_population, lookup_area_km2],
        middleware=[TodoListMiddleware()],
    )
    run_agent(
        agent,
        model=model,
        query=(
            "What is the approximate population density (people per km²) of "
            "Cairo? Look up the population, look up the area, then compute "
            "and report the density. Use a todo list to plan and track your "
            "work."
        ),
        scorer=TrajectoryScorer()
        .expect(agent_steps=5, tool_call_requests=5)
        .success(
            final_text_contains("21,800,000"),
            final_text_contains("606"),
            final_text_min_length(80),
        ),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _max_tool_calls_success(n: int) -> SuccessAssertion:
    """Wrap `MaxToolCallRequests` as a hard-fail SuccessAssertion.

    `MaxToolCallRequests` is defined as an EfficiencyAssertion (logged but
    not failing) so it doesn't gate normal evals. For trivial-skip tests we
    want it to be a hard failure when violated — over-using the tool is the
    bug the test is designed to catch.
    """
    eff = MaxToolCallRequests(n=n)

    class _AsSuccess(SuccessAssertion):
        def check(self, trajectory: AgentTrajectory) -> bool:
            return eff.check(trajectory)

        def describe_failure(self, trajectory: AgentTrajectory) -> str:
            return eff.describe_failure(trajectory)

    return _AsSuccess()
