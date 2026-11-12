"""Client integrations for calling remote A2A agents from LangChain and LangGraph."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Annotated, Any, Literal, NoReturn, TypedDict, cast

import httpx  # noqa: TC002
from langchain_core.messages import AIMessageChunk
from langchain_core.runnables.config import merge_configs
from langchain_core.runnables.graph import Graph
from langchain_core.tools import ArgsSchema, BaseTool
from langgraph.pregel.protocol import PregelProtocol
from langgraph.types import GraphOutput
from pydantic import BaseModel, ConfigDict, Field, SkipValidation

from langchain.a2a._conversion import (
    A2AStreamAccumulator,
    a2a_result_to_langchain_output,
    async_iter_to_sync,
    derive_agent_name_from_url,
    langchain_input_to_a2a_message,
    resolve_agent_card,
    resolve_agent_card_sync,
    run_async_safely,
    slugify_identifier,
    stream_a2a_events,
    validate_a2a_url,
)

try:
    from a2a.client import ClientConfig  # noqa: TC002
    from a2a.types import AgentCard  # noqa: TC002
except ImportError as exc:
    msg = (
        "The 'a2a-sdk' package is required to use A2A clients. "
        "Install it with `pip install 'langchain[a2a]'` or "
        "`uv add 'langchain[a2a]'`."
    )
    raise ImportError(msg) from exc

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator

    from langchain_core.callbacks import (
        AsyncCallbackManagerForToolRun,
        CallbackManagerForToolRun,
    )
    from langchain_core.runnables import RunnableConfig
    from langgraph.types import All, Durability, StateSnapshot, StreamMode
    from typing_extensions import Self

    from langchain.a2a._conversion import A2ACollectedResult


class _A2ARemoteState(TypedDict, total=False):
    """Flexible `TypedDict` state marker satisfying `PregelProtocol` bounds."""


def _build_tool_description(card: AgentCard) -> str:
    """Build an informative tool description from an `AgentCard` and its skills."""
    parts: list[str] = []
    if card.description:
        parts.append(card.description.strip())

    skill_lines: list[str] = []
    for skill in card.skills:
        skill_name = skill.name or skill.id
        skill_desc = (skill.description or "").strip()
        if skill_name and skill_desc:
            skill_lines.append(f"- {skill_name}: {skill_desc}")
        elif skill_name:
            skill_lines.append(f"- {skill_name}")
        elif skill_desc:
            skill_lines.append(f"- {skill_desc}")

    if skill_lines:
        parts.append("Skills:\n" + "\n".join(skill_lines))

    if not parts:
        return f"Invoke remote A2A agent '{card.name}'."
    return "\n\n".join(parts)


class A2AToolInput(BaseModel):
    """Input schema for invoking a remote A2A agent tool."""

    query: str = Field(description="Instruction or question to send to the remote A2A agent.")
    context_id: str | None = Field(
        default=None,
        description=("Optional A2A context ID to continue an ongoing multi-turn conversation."),
    )


class A2ATool(BaseTool):
    """LangChain tool wrapping a remote A2A agent discovered via its Agent Card.

    Example:
        ```python
        from langchain.a2a import A2ATool
        from langchain.agents import create_agent

        billing = A2ATool.from_agent_card("https://billing.example.com/.well-known/agent-card.json")
        supervisor = create_agent("openai:gpt-4o", tools=[billing])
        ```
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    description: str
    args_schema: Annotated[ArgsSchema | None, SkipValidation()] = A2AToolInput
    agent_card: AgentCard
    client_config: ClientConfig | None = Field(default=None, exclude=True)
    httpx_client: httpx.AsyncClient | None = Field(default=None, exclude=True)
    headers: dict[str, str] | None = Field(default=None, exclude=True)
    timeout: float | None = None
    default_context_id: str | None = None
    handle_input_required: Literal["return", "error"] = "return"

    @classmethod
    def from_agent_card(
        cls,
        agent: str | AgentCard,
        *,
        name: str | None = None,
        description: str | None = None,
        client_config: ClientConfig | None = None,
        httpx_client: httpx.Client | None = None,
        async_httpx_client: httpx.AsyncClient | None = None,
        headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        context_id: str | None = None,
        response_format: Literal["content", "content_and_artifact"] = "content",
        handle_input_required: Literal["return", "error"] = "return",
    ) -> A2ATool:
        """Create an `A2ATool` synchronously from an Agent Card URL or `AgentCard`.

        Args:
            agent: Base A2A URL, full agent-card JSON URL, or pre-built `AgentCard`.
            name: Optional override for the LangChain tool name. Defaults to a
                sanitized version of `agent_card.name`.
            description: Optional override for the tool description.
            client_config: Optional `a2a.client.ClientConfig`.
            httpx_client: Optional synchronous `httpx.Client` used to fetch the card.
            async_httpx_client: Optional asynchronous `httpx.AsyncClient` reused when
                calling the agent via `_arun`.
            headers: Optional HTTP headers sent when fetching the card and calling the
                agent.
            timeout: Optional HTTP request timeout in seconds.
            context_id: Optional default A2A `context_id` sent on each call unless
                overridden in tool arguments.
            response_format: `"content"` to return text only, or
                `"content_and_artifact"` to also attach structured A2A metadata.
            handle_input_required: Behavior when the remote agent returns
                `TASK_STATE_INPUT_REQUIRED`: `"return"` returns the prompt text to the
                calling agent, while `"error"` raises `A2AExecutionError`.

        Returns:
            Configured `A2ATool` instance.
        """
        card = resolve_agent_card_sync(
            agent,
            httpx_client=httpx_client,
            headers=headers,
            timeout=timeout,
        )
        return cls._from_resolved_card(
            card,
            name=name,
            description=description,
            client_config=client_config,
            async_httpx_client=async_httpx_client,
            headers=headers,
            timeout=timeout,
            context_id=context_id,
            response_format=response_format,
            handle_input_required=handle_input_required,
        )

    @classmethod
    async def afrom_agent_card(
        cls,
        agent: str | AgentCard,
        *,
        name: str | None = None,
        description: str | None = None,
        client_config: ClientConfig | None = None,
        httpx_client: httpx.AsyncClient | None = None,
        headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        context_id: str | None = None,
        response_format: Literal["content", "content_and_artifact"] = "content",
        handle_input_required: Literal["return", "error"] = "return",
    ) -> A2ATool:
        """Create an `A2ATool` asynchronously from an Agent Card URL or `AgentCard`.

        Args:
            agent: Base A2A URL, full agent-card JSON URL, or pre-built `AgentCard`.
            name: Optional override for the LangChain tool name.
            description: Optional override for the tool description.
            client_config: Optional `a2a.client.ClientConfig`.
            httpx_client: Optional shared `httpx.AsyncClient` used for both card
                resolution and subsequent A2A tool calls.
            headers: Optional HTTP headers.
            timeout: Optional HTTP request timeout in seconds.
            context_id: Optional default A2A `context_id`.
            response_format: `"content"` or `"content_and_artifact"`.
            handle_input_required: Behavior when the remote task requires input.

        Returns:
            Configured `A2ATool` instance.
        """
        card = await resolve_agent_card(
            agent,
            httpx_client=httpx_client,
            headers=headers,
            timeout=timeout,
        )
        return cls._from_resolved_card(
            card,
            name=name,
            description=description,
            client_config=client_config,
            async_httpx_client=httpx_client,
            headers=headers,
            timeout=timeout,
            context_id=context_id,
            response_format=response_format,
            handle_input_required=handle_input_required,
        )

    @classmethod
    def _from_resolved_card(
        cls,
        card: AgentCard,
        *,
        name: str | None,
        description: str | None,
        client_config: ClientConfig | None,
        async_httpx_client: httpx.AsyncClient | None,
        headers: Mapping[str, str] | None,
        timeout: float | None,
        context_id: str | None,
        response_format: Literal["content", "content_and_artifact"],
        handle_input_required: Literal["return", "error"],
    ) -> A2ATool:
        """Construct an `A2ATool` from an already-resolved `AgentCard`."""
        resolved_name = slugify_identifier(name or card.name or "a2a_agent")
        resolved_description = description or _build_tool_description(card)
        return cls(
            name=resolved_name,
            description=resolved_description,
            agent_card=card,
            client_config=client_config,
            httpx_client=async_httpx_client,
            headers=dict(headers) if headers else None,
            timeout=timeout,
            default_context_id=context_id,
            response_format=response_format,
            handle_input_required=handle_input_required,
        )

    def _format_collected_result(
        self,
        collected: A2ACollectedResult,
    ) -> str | tuple[str, dict[str, Any]]:
        """Format an `A2ACollectedResult` for LangChain tool output."""
        text = collected.text
        if not text and collected.data:
            text = json.dumps(
                collected.data[-1] if len(collected.data) == 1 else collected.data,
                ensure_ascii=False,
            )

        if self.response_format == "content_and_artifact":
            artifact_payload = {
                "task_id": collected.task_id,
                "context_id": collected.context_id,
                "state": int(collected.state) if collected.state is not None else None,
                "is_input_required": collected.is_input_required,
                "data": collected.data,
                "artifacts": collected.artifacts,
            }
            return text, artifact_payload

        return text

    async def _call_remote(
        self,
        query: str,
        *,
        context_id: str | None = None,
    ) -> str | tuple[str, dict[str, Any]]:
        """Send a query to the target A2A agent and collect its response."""
        effective_context = context_id if context_id is not None else self.default_context_id
        outbound = langchain_input_to_a2a_message(
            query,
            context_id=effective_context,
        )
        accumulator = A2AStreamAccumulator()
        async for event in stream_a2a_events(
            self.agent_card,
            outbound,
            client_config=self.client_config,
            httpx_client=self.httpx_client,
            headers=self.headers,
            timeout=self.timeout,
        ):
            accumulator.consume(event)

        collected = accumulator.finalize(
            raise_on_failure=True,
            raise_on_input_required=self.handle_input_required == "error",
        )
        return self._format_collected_result(collected)

    def _run(
        self,
        query: str,
        context_id: str | None = None,
        run_manager: CallbackManagerForToolRun | None = None,  # noqa: ARG002
    ) -> str | tuple[str, dict[str, Any]]:
        """Synchronously execute the remote A2A agent tool."""
        return run_async_safely(lambda: self._call_remote(query, context_id=context_id))

    async def _arun(
        self,
        query: str,
        context_id: str | None = None,
        run_manager: AsyncCallbackManagerForToolRun | None = None,  # noqa: ARG002
    ) -> str | tuple[str, dict[str, Any]]:
        """Asynchronously execute the remote A2A agent tool."""
        return await self._call_remote(query, context_id=context_id)


def _extract_a2a_ids_from_config(
    config: RunnableConfig | None,
) -> tuple[str | None, str | None]:
    """Extract `(context_id, task_id)` from LangGraph `RunnableConfig`."""
    if not config:
        return None, None
    configurable = config.get("configurable")
    if not isinstance(configurable, Mapping):
        return None, None
    raw_context = configurable.get("a2a_context_id", configurable.get("thread_id"))
    raw_task = configurable.get("a2a_task_id")
    context_id = str(raw_context) if raw_context is not None else None
    task_id = str(raw_task) if raw_task is not None else None
    return context_id, task_id


class A2ARemoteGraph(PregelProtocol[_A2ARemoteState, None, Any, Any]):
    """`PregelProtocol` runnable representing a remote A2A agent in a `StateGraph`.

    Allows mounting an external A2A agent directly as a graph node:

    ```python
    from langgraph.graph import MessagesState, StateGraph
    from langchain.a2a import A2ARemoteGraph

    builder = StateGraph(MessagesState)
    builder.add_node("billing", A2ARemoteGraph("https://billing.example.com"))
    ```
    """

    def __init__(
        self,
        agent: str | AgentCard,
        *,
        name: str | None = None,
        config: RunnableConfig | None = None,
        client_config: ClientConfig | None = None,
        httpx_client: httpx.AsyncClient | None = None,
        headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        output_key: str | None = None,
    ) -> None:
        """Initialize a lazy reference to a remote A2A agent.

        No network call is made during `__init__` unless an `AgentCard` is supplied
        directly; agent card discovery occurs lazily on first invocation.

        Args:
            agent: Base URL, full agent-card JSON URL, or pre-built `AgentCard`.
            name: Optional LangGraph node name for this runnable.
            config: Optional default `RunnableConfig`.
            client_config: Optional `a2a.client.ClientConfig`.
            httpx_client: Optional shared `httpx.AsyncClient` (useful for
                `httpx.ASGITransport` testing).
            headers: Optional HTTP headers forwarded on discovery and RPC requests.
            timeout: Optional HTTP timeout in seconds.
            output_key: Optional explicit state key to store the remote text output.
        """
        if isinstance(agent, str):
            self._agent: str | AgentCard = validate_a2a_url(agent)
            self._cached_card: AgentCard | None = None
        else:
            self._agent = agent
            self._cached_card = agent

        self.name = name
        self.config = config
        self.client_config = client_config
        self.httpx_client = httpx_client
        self.headers = dict(headers) if headers else None
        self.timeout = timeout
        self.output_key = output_key

    def get_name(self, suffix: str | None = None, *, name: str | None = None) -> str:
        """Return a stable node/runnable name without triggering network calls."""
        base = name or self.name
        if not base:
            if self._cached_card is not None and self._cached_card.name:
                base = slugify_identifier(self._cached_card.name)
            elif isinstance(self._agent, str):
                base = derive_agent_name_from_url(self._agent)
            else:
                base = "a2a_remote_graph"
        return f"{base}_{suffix}" if suffix else base

    async def _get_card(self) -> AgentCard:
        """Resolve and cache the target `AgentCard` lazily."""
        if self._cached_card is None:
            self._cached_card = await resolve_agent_card(
                self._agent,
                httpx_client=self.httpx_client,
                headers=self.headers,
                timeout=self.timeout,
            )
        return self._cached_card

    def with_config(
        self,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Self:
        """Return a shallow copy of this runnable with merged configuration."""
        extra = cast("RunnableConfig", kwargs) if kwargs else None
        merged = merge_configs(self.config, config, extra)
        cloned = self.__class__(
            self._agent,
            name=self.name,
            config=merged,
            client_config=self.client_config,
            httpx_client=self.httpx_client,
            headers=self.headers,
            timeout=self.timeout,
            output_key=self.output_key,
        )
        cloned._cached_card = self._cached_card  # noqa: SLF001
        return cloned

    def get_graph(
        self,
        config: RunnableConfig | None = None,  # noqa: ARG002
        *,
        xray: int | bool = False,  # noqa: ARG002
    ) -> Graph:
        """Return a single-node `Graph` representation of this remote A2A agent."""
        graph = Graph()
        graph.add_node(self, id=self.get_name())
        return graph

    async def aget_graph(
        self,
        config: RunnableConfig | None = None,
        *,
        xray: int | bool = False,
    ) -> Graph:
        """Asynchronously return a single-node `Graph` representation."""
        return self.get_graph(config, xray=xray)

    def _raise_state_unsupported(self, method_name: str) -> NoReturn:
        msg = (
            f"A2ARemoteGraph.{method_name} is not supported because the A2A protocol "
            "does not expose LangGraph checkpoint snapshots."
        )
        raise NotImplementedError(msg)

    def get_state(
        self,
        config: RunnableConfig,  # noqa: ARG002
        *,
        subgraphs: bool = False,  # noqa: ARG002
    ) -> StateSnapshot:
        """Not supported by the A2A protocol."""
        return self._raise_state_unsupported("get_state")

    async def aget_state(
        self,
        config: RunnableConfig,  # noqa: ARG002
        *,
        subgraphs: bool = False,  # noqa: ARG002
    ) -> StateSnapshot:
        """Not supported by the A2A protocol."""
        return self._raise_state_unsupported("aget_state")

    def get_state_history(
        self,
        config: RunnableConfig,  # noqa: ARG002
        *,
        filter: dict[str, Any] | None = None,  # noqa: A002, ARG002
        before: RunnableConfig | None = None,  # noqa: ARG002
        limit: int | None = None,  # noqa: ARG002
    ) -> Iterator[StateSnapshot]:
        """Not supported by the A2A protocol."""
        return self._raise_state_unsupported("get_state_history")

    def aget_state_history(
        self,
        config: RunnableConfig,  # noqa: ARG002
        *,
        filter: dict[str, Any] | None = None,  # noqa: A002, ARG002
        before: RunnableConfig | None = None,  # noqa: ARG002
        limit: int | None = None,  # noqa: ARG002
    ) -> AsyncIterator[StateSnapshot]:
        """Not supported by the A2A protocol."""
        return self._raise_state_unsupported("aget_state_history")

    def bulk_update_state(
        self,
        config: RunnableConfig,  # noqa: ARG002
        supersteps: Sequence[Any],  # noqa: ARG002
    ) -> RunnableConfig:
        """Not supported by the A2A protocol."""
        return self._raise_state_unsupported("bulk_update_state")

    async def abulk_update_state(
        self,
        config: RunnableConfig,  # noqa: ARG002
        supersteps: Sequence[Any],  # noqa: ARG002
    ) -> RunnableConfig:
        """Not supported by the A2A protocol."""
        return self._raise_state_unsupported("abulk_update_state")

    def update_state(
        self,
        config: RunnableConfig,  # noqa: ARG002
        values: dict[str, Any] | Any | None,  # noqa: ARG002
        as_node: str | None = None,  # noqa: ARG002
    ) -> RunnableConfig:
        """Not supported by the A2A protocol."""
        return self._raise_state_unsupported("update_state")

    async def aupdate_state(
        self,
        config: RunnableConfig,  # noqa: ARG002
        values: dict[str, Any] | Any | None,  # noqa: ARG002
        as_node: str | None = None,  # noqa: ARG002
    ) -> RunnableConfig:
        """Not supported by the A2A protocol."""
        return self._raise_state_unsupported("aupdate_state")

    @staticmethod
    def _format_stream_payload(
        mode: str,
        payload: Any,
        *,
        version: Literal["v1", "v2"],
        is_multi_mode: bool,
    ) -> Any:
        """Wrap stream output according to `stream_mode` and `version`."""
        if version == "v2":
            return {"type": mode, "ns": (), "data": payload}
        if is_multi_mode:
            return (mode, payload)
        return payload

    async def ainvoke(
        self,
        input: Any,  # noqa: A002
        config: RunnableConfig | None = None,
        *,
        stream_mode: StreamMode = "values",  # noqa: ARG002
        print_mode: StreamMode | Sequence[StreamMode] = (),  # noqa: ARG002
        output_keys: str | Sequence[str] | None = None,  # noqa: ARG002
        interrupt_before: All | Sequence[str] | None = None,  # noqa: ARG002
        interrupt_after: All | Sequence[str] | None = None,  # noqa: ARG002
        durability: Durability | None = None,  # noqa: ARG002
        version: Literal["v1", "v2"] = "v1",
        **kwargs: Any,  # noqa: ARG002
    ) -> Any:
        """Asynchronously execute the remote A2A agent and return converted output."""
        merged_config = merge_configs(self.config, config)
        context_id, task_id = _extract_a2a_ids_from_config(merged_config)
        card = await self._get_card()

        outbound = langchain_input_to_a2a_message(
            input,
            context_id=context_id,
            task_id=task_id,
        )
        accumulator = A2AStreamAccumulator()
        async for event in stream_a2a_events(
            card,
            outbound,
            client_config=self.client_config,
            httpx_client=self.httpx_client,
            headers=self.headers,
            timeout=self.timeout,
        ):
            accumulator.consume(event)

        collected = accumulator.finalize(
            raise_on_failure=True,
            raise_on_input_required=True,
        )
        converted = a2a_result_to_langchain_output(
            collected,
            input,
            output_key=self.output_key,
        )
        if version == "v2":
            return GraphOutput(value=converted)
        return converted

    def invoke(
        self,
        input: Any,  # noqa: A002
        config: RunnableConfig | None = None,
        *,
        stream_mode: StreamMode = "values",
        print_mode: StreamMode | Sequence[StreamMode] = (),
        output_keys: str | Sequence[str] | None = None,
        interrupt_before: All | Sequence[str] | None = None,
        interrupt_after: All | Sequence[str] | None = None,
        durability: Durability | None = None,
        version: Literal["v1", "v2"] = "v1",
        **kwargs: Any,
    ) -> Any:
        """Synchronously execute the remote A2A agent and return converted output."""
        return run_async_safely(
            lambda: self.ainvoke(
                input,
                config=config,
                stream_mode=stream_mode,
                print_mode=print_mode,
                output_keys=output_keys,
                interrupt_before=interrupt_before,
                interrupt_after=interrupt_after,
                durability=durability,
                version=version,
                **kwargs,
            )
        )

    async def astream(  # type: ignore[override]
        self,
        input: Any,  # noqa: A002
        config: RunnableConfig | None = None,
        *,
        stream_mode: StreamMode | Sequence[StreamMode] | None = None,
        print_mode: StreamMode | Sequence[StreamMode] = (),  # noqa: ARG002
        output_keys: str | Sequence[str] | None = None,  # noqa: ARG002
        interrupt_before: All | Sequence[str] | None = None,  # noqa: ARG002
        interrupt_after: All | Sequence[str] | None = None,  # noqa: ARG002
        durability: Durability | None = None,  # noqa: ARG002
        subgraphs: bool = False,  # noqa: ARG002
        version: Literal["v1", "v2"] = "v1",
        **kwargs: Any,  # noqa: ARG002
    ) -> AsyncIterator[Any]:
        """Asynchronously stream outputs from the remote A2A agent."""
        merged_config = merge_configs(self.config, config)
        context_id, task_id = _extract_a2a_ids_from_config(merged_config)
        card = await self._get_card()

        is_multi_mode = isinstance(stream_mode, Sequence) and not isinstance(stream_mode, str)
        modes: list[StreamMode]
        if isinstance(stream_mode, str):
            modes = [stream_mode]
        elif stream_mode:
            modes = list(stream_mode)
        else:
            modes = ["values"]

        outbound = langchain_input_to_a2a_message(
            input,
            context_id=context_id,
            task_id=task_id,
        )
        accumulator = A2AStreamAccumulator()
        node_name = self.get_name()

        async for event in stream_a2a_events(
            card,
            outbound,
            client_config=self.client_config,
            httpx_client=self.httpx_client,
            headers=self.headers,
            timeout=self.timeout,
        ):
            delta = accumulator.consume(event)
            if "messages" in modes and delta:
                msg_tuple = (
                    AIMessageChunk(content=delta),
                    {
                        "langgraph_node": node_name,
                        "a2a_task_id": accumulator.task_id,
                        "a2a_context_id": accumulator.context_id,
                    },
                )
                yield self._format_stream_payload(
                    "messages",
                    msg_tuple,
                    version=version,
                    is_multi_mode=is_multi_mode,
                )

        collected = accumulator.finalize(
            raise_on_failure=True,
            raise_on_input_required=True,
        )
        converted = a2a_result_to_langchain_output(
            collected,
            input,
            output_key=self.output_key,
        )

        for mode in modes:
            if mode == "updates":
                yield self._format_stream_payload(
                    "updates",
                    {node_name: converted},
                    version=version,
                    is_multi_mode=is_multi_mode,
                )
            elif mode == "values":
                yield self._format_stream_payload(
                    "values",
                    converted,
                    version=version,
                    is_multi_mode=is_multi_mode,
                )

    def stream(  # type: ignore[override]
        self,
        input: Any,  # noqa: A002
        config: RunnableConfig | None = None,
        *,
        stream_mode: StreamMode | Sequence[StreamMode] | None = None,
        print_mode: StreamMode | Sequence[StreamMode] = (),
        output_keys: str | Sequence[str] | None = None,
        interrupt_before: All | Sequence[str] | None = None,
        interrupt_after: All | Sequence[str] | None = None,
        durability: Durability | None = None,
        subgraphs: bool = False,
        version: Literal["v1", "v2"] = "v1",
        **kwargs: Any,
    ) -> Iterator[Any]:
        """Synchronously stream outputs from the remote A2A agent."""
        return async_iter_to_sync(
            lambda: self.astream(
                input,
                config=config,
                stream_mode=stream_mode,
                print_mode=print_mode,
                output_keys=output_keys,
                interrupt_before=interrupt_before,
                interrupt_after=interrupt_after,
                durability=durability,
                subgraphs=subgraphs,
                version=version,
                **kwargs,
            )
        )
