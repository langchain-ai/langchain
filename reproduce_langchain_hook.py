"""Reproduction script: Circuit Breaker Hooks for AgentExecutor.

Demonstrates that callbacks can now interrupt agent execution by returning
a ``CallbackDecision`` from ``on_agent_action``.

Before this patch, ``on_agent_action`` was purely read-only – callbacks
could observe actions but never influence control flow.  After the patch,
a handler returning ``CallbackDecision(stop_execution=True, ...)`` causes
the agent loop to terminate immediately with the provided response.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from langchain_core.agents import AgentAction, CallbackDecision
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.tools import Tool
from typing_extensions import override

from langchain_classic.agents import AgentExecutor, AgentType, initialize_agent


# -- Fake LLM that cycles through scripted responses ----------------------

class FakeLoopingLLM(LLM):
    """LLM that simulates an agent stuck in a loop."""

    responses: list[str]
    i: int = -1

    @override
    def _call(
        self,
        prompt: str,
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> str:
        self.i += 1
        return self.responses[self.i % len(self.responses)]

    @property
    def _llm_type(self) -> str:
        return "fake_looping"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {}


# -- Circuit Breaker handler ------------------------------------------------

class LoopDetectorHandler(BaseCallbackHandler):
    """Callback handler that detects repeated actions and stops the agent.

    This is the circuit-breaker pattern: after seeing the same tool called
    more than ``max_repeats`` times in ``intermediate_steps``, the handler
    returns a ``CallbackDecision`` telling the executor to halt.
    """

    def __init__(self, *, max_repeats: int = 2) -> None:
        self.max_repeats = max_repeats

    def on_agent_action(
        self,
        action: AgentAction,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> CallbackDecision | None:
        intermediate_steps: list[tuple[AgentAction, str]] = kwargs.get(
            "intermediate_steps", []
        )
        # Count how many times the *same* tool+input pair has appeared.
        same_action_count = sum(
            1
            for prev_action, _ in intermediate_steps
            if prev_action.tool == action.tool
            and prev_action.tool_input == action.tool_input
        )
        if same_action_count >= self.max_repeats:
            print(
                f"[CircuitBreaker] Loop detected! "
                f"'{action.tool}' called {same_action_count + 1} times "
                f"with input '{action.tool_input}'. Stopping."
            )
            return CallbackDecision(
                stop_execution=True,
                stop_response=(
                    f"Execution halted: loop detected after {same_action_count + 1} "
                    f"repeated calls to '{action.tool}'."
                ),
                metadata={"handler": "LoopDetectorHandler"},
            )
        return None


# -- Main ------------------------------------------------------------------

def main() -> None:
    # The LLM will keep requesting the same action forever.
    looping_responses = [
        "I need to search.\nAction: Search\nAction Input: langchain",
    ]
    llm = FakeLoopingLLM(responses=looping_responses, cache=False)

    tools = [
        Tool(
            name="Search",
            func=lambda x: f"Result for '{x}'",
            description="Search the web",
        ),
    ]

    agent: AgentExecutor = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        max_iterations=10,  # safety net
    )

    handler = LoopDetectorHandler(max_repeats=2)

    print("=" * 60)
    print("Running agent WITH circuit breaker (max_repeats=2)...")
    print("=" * 60)
    output = agent.run("Tell me about langchain", callbacks=[handler])
    print(f"\nFinal output: {output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
