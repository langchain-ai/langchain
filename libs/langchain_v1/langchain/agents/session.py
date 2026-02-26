"""Session management for agents with interrupt/resume and time-travel support."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Any

from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.types import Command

_UNSET = object()

if TYPE_CHECKING:
    from collections.abc import Iterator

    from langgraph.graph.state import CompiledStateGraph
    from langgraph.types import StateSnapshot


class AgentSession:
    """A convenience wrapper around a compiled agent graph for session management.

    Provides a high-level API for running agents with interrupt/resume support,
    appending messages during interrupts, and time-travel via checkpoint branching.

    Requires the agent to have been created with a ``checkpointer``.

    !!! warning "Experimental"

        This feature is experimental and may change in future releases.

    Example:
        ```python
        from langgraph.checkpoint.memory import InMemorySaver
        from langchain.agents import create_agent
        from langchain.agents.session import AgentSession

        agent = create_agent(
            model="openai:gpt-4o",
            tools=[my_tool],
            checkpointer=InMemorySaver(),
            interrupt_before=["tools"],
        )

        session = AgentSession(agent)

        # Run until interrupt
        result = session.run("What's the weather?")

        # Inspect pending actions
        state = session.get_state()
        print(state.next)  # ('tools',)

        # Resume with approval
        result = session.resume()

        # Branch from a previous checkpoint
        branched = session.branch(checkpoint_id=state.config["configurable"]["checkpoint_id"])
        result = branched.run("Try a different approach")
        ```
    """

    def __init__(
        self,
        agent: CompiledStateGraph,
        *,
        thread_id: str | None = None,
    ) -> None:
        """Initialize an agent session.

        Args:
            agent: A compiled agent graph with a checkpointer attached.
            thread_id: An optional thread identifier. If not provided, a random
                UUID is generated.
        """
        self._agent = agent
        self._thread_id = thread_id or uuid.uuid4().hex

    @property
    def thread_id(self) -> str:
        """The thread identifier for this session."""
        return self._thread_id

    @property
    def agent(self) -> CompiledStateGraph:
        """The underlying compiled agent graph."""
        return self._agent

    @property
    def config(self) -> dict[str, Any]:
        """The LangGraph config dict for this session thread."""
        return {"configurable": {"thread_id": self._thread_id}}

    def run(
        self,
        agent_input: str | dict[str, Any] | list[AnyMessage],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Run the agent with the given input.

        Args:
            agent_input: The input to the agent. Can be a string (converted to a
                `HumanMessage`), a list of messages, or a full state dict.
            **kwargs: Additional keyword arguments passed to `agent.invoke`.

        Returns:
            The agent output state dict.
        """
        return self._agent.invoke(
            _normalize_input(agent_input),
            config=self.config,
            **kwargs,
        )

    async def arun(
        self,
        agent_input: str | dict[str, Any] | list[AnyMessage],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Async version of :meth:`run`.

        Args:
            agent_input: The input to the agent.
            **kwargs: Additional keyword arguments passed to `agent.ainvoke`.

        Returns:
            The agent output state dict.
        """
        return await self._agent.ainvoke(
            _normalize_input(agent_input),
            config=self.config,
            **kwargs,
        )

    def resume(
        self,
        *,
        value: Any = _UNSET,
        update: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Resume the agent after an interrupt.

        Use this after the agent has been interrupted (via ``interrupt_before``
        or ``interrupt_after``) to continue execution. Optionally provide
        new data to inject into the graph state.

        Args:
            value: A resume value to pass to the interrupted node. This is
                received by ``interrupt()`` calls in middleware or custom nodes.
                When omitted, the agent simply continues from where it stopped.
            update: An optional state update dict to apply before resuming.
                Use this to append messages or modify state fields.
            **kwargs: Additional keyword arguments passed to `agent.invoke`.

        Returns:
            The agent output state dict.

        Example:
            ```python
            # Resume after interrupt_before (simple continuation)
            result = session.resume()

            # Resume with a human-in-the-loop approval
            result = session.resume(value={"decisions": [{"type": "approve"}]})

            # Resume and inject additional context
            result = session.resume(
                value=True,
                update={"messages": [HumanMessage("Also check the forecast")]},
            )
            ```
        """
        if update is not None:
            self._agent.update_state(self.config, update)

        invoke_input: Any = None if value is _UNSET else Command(resume=value)
        return self._agent.invoke(
            invoke_input,
            config=self.config,
            **kwargs,
        )

    async def aresume(
        self,
        *,
        value: Any = _UNSET,
        update: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Async version of :meth:`resume`.

        Args:
            value: A resume value to pass to the interrupted node.
            update: An optional state update dict to apply before resuming.
            **kwargs: Additional keyword arguments passed to `agent.ainvoke`.

        Returns:
            The agent output state dict.
        """
        if update is not None:
            await self._agent.aupdate_state(self.config, update)

        invoke_input: Any = None if value is _UNSET else Command(resume=value)
        return await self._agent.ainvoke(
            invoke_input,
            config=self.config,
            **kwargs,
        )

    def get_state(self) -> StateSnapshot:
        """Get the current state snapshot for this session.

        Returns:
            A ``StateSnapshot`` containing the current values, pending nodes,
            config (with ``checkpoint_id``), metadata, and any pending interrupts.
        """
        return self._agent.get_state(self.config)

    async def aget_state(self) -> StateSnapshot:
        """Async version of :meth:`get_state`.

        Returns:
            A ``StateSnapshot`` for the current session state.
        """
        return await self._agent.aget_state(self.config)

    def get_history(
        self,
        *,
        limit: int | None = None,
    ) -> Iterator[StateSnapshot]:
        """Iterate over the checkpoint history for this session.

        Yields snapshots from newest to oldest. Each snapshot includes a config
        with a ``checkpoint_id`` that can be used for branching.

        Args:
            limit: Maximum number of snapshots to return.

        Yields:
            ``StateSnapshot`` objects ordered from newest to oldest.
        """
        yield from self._agent.get_state_history(
            self.config,
            limit=limit,
        )

    def branch(
        self,
        *,
        checkpoint_id: str,
        new_thread_id: str | None = None,
    ) -> AgentSession:
        """Create a new session branching from a specific checkpoint.

        This implements **time-travel**: you pick a point in the session history
        and fork a new conversation from that point. The original session is
        left untouched.

        Args:
            checkpoint_id: The checkpoint to branch from. Obtain this from
                ``snapshot.config["configurable"]["checkpoint_id"]`` in the
                history.
            new_thread_id: An optional thread ID for the new branch. If not
                provided, a random UUID is generated.

        Returns:
            A new ``AgentSession`` whose state starts from the specified
            checkpoint.

        Example:
            ```python
            # List history and branch from 2 steps ago
            history = list(session.get_history(limit=3))
            old_snapshot = history[-1]
            checkpoint_id = old_snapshot.config["configurable"]["checkpoint_id"]

            branch = session.branch(checkpoint_id=checkpoint_id)
            result = branch.run("Let's try a completely different approach.")
            ```
        """
        branch_thread_id = new_thread_id or uuid.uuid4().hex
        source_state = self._agent.get_state(
            {
                "configurable": {
                    "thread_id": self._thread_id,
                    "checkpoint_id": checkpoint_id,
                }
            }
        )

        new_session = AgentSession(
            self._agent,
            thread_id=branch_thread_id,
        )
        self._agent.update_state(
            new_session.config,
            source_state.values,
        )
        return new_session


def _normalize_input(
    agent_input: str | dict[str, Any] | list[AnyMessage],
) -> dict[str, Any]:
    """Convert convenience input formats to the standard agent state dict."""
    if isinstance(agent_input, str):
        return {"messages": [HumanMessage(content=agent_input)]}
    if isinstance(agent_input, list):
        return {"messages": agent_input}
    return agent_input
