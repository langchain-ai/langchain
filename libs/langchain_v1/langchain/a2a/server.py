"""Server utilities for exposing LangChain and LangGraph agents over A2A."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from langchain.a2a._conversion import (
    a2a_message_to_graph_input,
    graph_result_to_a2a_parts,
    join_url,
    slugify_identifier,
    validate_a2a_url,
)

try:
    from a2a.helpers import new_text_part
    from a2a.server.agent_execution import AgentExecutor, RequestContext
    from a2a.server.request_handlers import DefaultRequestHandler
    from a2a.server.routes import (
        create_agent_card_routes,
        create_jsonrpc_routes,
        create_rest_routes,
    )
    from a2a.server.tasks import InMemoryTaskStore, TaskStore, TaskUpdater
    from a2a.types import (
        AgentCapabilities,
        AgentCard,
        AgentInterface,
        AgentSkill,
        Message,
        Part,
        Task,
        TaskState,
        TaskStatus,
    )
    from a2a.utils.constants import (
        AGENT_CARD_WELL_KNOWN_PATH,
        PROTOCOL_VERSION_CURRENT,
        TransportProtocol,
    )
    from starlette.applications import Starlette
except ImportError as exc:
    msg = (
        "The 'a2a-sdk[http-server]' package is required to serve A2A endpoints. "
        "Install it with `pip install 'langchain[a2a]'` or "
        "`uv add 'langchain[a2a]'`."
    )
    raise ImportError(msg) from exc

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Sequence

    from a2a.server.events import EventQueue
    from langchain_core.runnables import Runnable, RunnableConfig
    from starlette.routing import BaseRoute


class LangGraphAgentExecutor(AgentExecutor):
    """A2A `AgentExecutor` backed by a LangGraph graph or LangChain runnable.

    Maps incoming A2A requests onto `graph.ainvoke(...)`, binding the A2A
    `context_id` to LangGraph's `configurable.thread_id` so checkpointers
    preserve multi-turn conversations across A2A requests, and translating
    LangGraph `interrupt()` pauses into `TASK_STATE_INPUT_REQUIRED`.
    """

    def __init__(
        self,
        graph: Runnable[Any, Any],
        *,
        input_key: str | None = None,
        artifact_name: str = "response",
        state_mapper: Callable[[Message, RequestContext], Any] | None = None,
        output_mapper: Callable[[Any], list[Part]] | None = None,
    ) -> None:
        """Initialize the A2A executor wrapper.

        Args:
            graph: Compiled LangGraph graph (`create_agent(...)` / `StateGraph`)
                or any LangChain `Runnable`.
            input_key: Optional explicit state dictionary key used when adapting
                text-only A2A messages.
            artifact_name: Name assigned to emitted A2A response artifacts.
            state_mapper: Optional callable overriding default A2A-to-graph input
                conversion.
            output_mapper: Optional callable overriding default graph-to-A2A `Part`
                conversion.
        """
        self.graph = graph
        self.input_key = input_key
        self.artifact_name = artifact_name
        self.state_mapper = state_mapper
        self.output_mapper = output_mapper
        self._running_tasks: dict[str, asyncio.Task[Any]] = {}
        self._canceled_tasks: set[str] = set()

    def _build_runnable_config(
        self,
        *,
        task_id: str,
        context_id: str,
    ) -> RunnableConfig:
        """Construct a LangGraph `RunnableConfig` bound to A2A thread metadata."""
        return {
            "configurable": {"thread_id": context_id},
            "metadata": {
                "a2a_task_id": task_id,
                "a2a_context_id": context_id,
            },
        }

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """Execute an A2A task by invoking the underlying LangGraph runnable.

        Args:
            context: A2A request context containing the user message and task state.
            event_queue: Event queue receiving A2A `Task`, status, and artifact updates.
        """
        if context.message is None:
            msg = "A2A request context is missing a user message."
            raise ValueError(msg)

        task_id = context.task_id or context.message.task_id or str(uuid4())
        context_id = context.context_id or context.message.context_id or str(uuid4())

        if context.current_task is None:
            initial_task = Task(
                id=task_id,
                context_id=context_id,
                status=TaskStatus(state=TaskState.TASK_STATE_SUBMITTED),
                history=[context.message],
            )
            await event_queue.enqueue_event(initial_task)

        updater = TaskUpdater(
            event_queue,
            task_id=task_id,
            context_id=context_id,
        )
        await updater.start_work()

        is_resuming = (
            context.current_task is not None
            and context.current_task.HasField("status")
            and context.current_task.status.state == TaskState.TASK_STATE_INPUT_REQUIRED
        )

        if self.state_mapper is not None:
            graph_input = self.state_mapper(context.message, context)
        else:
            graph_input = a2a_message_to_graph_input(
                context.message,
                self.graph,
                input_key=self.input_key,
                resume_mode=is_resuming,
            )

        config = self._build_runnable_config(
            task_id=task_id,
            context_id=context_id,
        )

        exec_task = asyncio.create_task(self.graph.ainvoke(graph_input, config=config))
        self._running_tasks[task_id] = exec_task
        try:
            result = await exec_task
        except asyncio.CancelledError:
            if task_id not in self._canceled_tasks:
                await updater.cancel()
            return
        except Exception as exc:
            error_message = updater.new_agent_message(
                [new_text_part(f"Agent execution failed: {exc}")]
            )
            await updater.failed(message=error_message)
            return
        finally:
            self._running_tasks.pop(task_id, None)
            self._canceled_tasks.discard(task_id)

        if self.output_mapper is not None:
            parts = self.output_mapper(result)
            is_interrupt = False
        else:
            parts, is_interrupt = graph_result_to_a2a_parts(result)

        if is_interrupt:
            await updater.requires_input(message=updater.new_agent_message(parts))
            return

        if parts:
            await updater.add_artifact(
                parts,
                name=self.artifact_name,
                last_chunk=True,
            )
        await updater.complete()

    async def cancel(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """Cancel an in-flight A2A task if running and publish cancellation status.

        Args:
            context: A2A request context identifying the task to cancel.
            event_queue: Event queue receiving the terminal `TASK_STATE_CANCELED` update.
        """
        task_id = context.task_id or str(uuid4())
        context_id = context.context_id or str(uuid4())

        running = self._running_tasks.get(task_id)
        if running is not None and not running.done():
            self._canceled_tasks.add(task_id)
            running.cancel()

        updater = TaskUpdater(
            event_queue,
            task_id=task_id,
            context_id=context_id,
        )
        await updater.cancel()


def _resolve_graph_name(graph: Runnable[Any, Any]) -> str:
    """Resolve a sensible default agent name from a LangGraph/Runnable instance."""
    explicit_name = getattr(graph, "name", None)
    if isinstance(explicit_name, str) and explicit_name.strip():
        return explicit_name.strip()
    try:
        candidate = graph.get_name()
    except Exception:
        return "langchain_agent"
    return candidate.strip() or "langchain_agent"


def build_agent_card(
    graph: Runnable[Any, Any],
    *,
    name: str | None = None,
    description: str | None = None,
    version: str = "1.0.0",
    url: str = "http://localhost:8000",
    skills: Sequence[AgentSkill] | None = None,
    capabilities: AgentCapabilities | None = None,
    default_input_modes: Sequence[str] = ("text/plain", "application/json"),
    default_output_modes: Sequence[str] = ("text/plain", "application/json"),
    rpc_url: str = "/",
    enable_rest: bool = True,
) -> AgentCard:
    """Construct an A2A v1.0 `AgentCard` describing a LangChain/LangGraph agent.

    Args:
        graph: Target LangGraph graph or LangChain runnable.
        name: Public agent name. Defaults to the graph's configured name.
        description: Human-readable agent description.
        version: Semantic agent version string.
        url: Base public URL where this A2A service is reachable.
        skills: Optional A2A `AgentSkill` definitions advertised on the card.
        capabilities: Optional `AgentCapabilities` overrides.
        default_input_modes: Supported input MIME types.
        default_output_modes: Supported output MIME types.
        rpc_url: Relative mount path for JSON-RPC requests.
        enable_rest: Whether to also advertise the HTTP+JSON REST interface.

    Returns:
        Configured `AgentCard` instance.
    """
    normalized_url = validate_a2a_url(url).rstrip("/")
    resolved_name = name.strip() if name and name.strip() else _resolve_graph_name(graph)
    resolved_description = (
        description.strip()
        if description and description.strip()
        else f"A2A agent backed by {resolved_name}."
    )
    resolved_skills = (
        list(skills)
        if skills is not None
        else [
            AgentSkill(
                id=slugify_identifier(resolved_name),
                name=resolved_name,
                description=resolved_description,
                tags=["langchain", "a2a"],
            )
        ]
    )
    resolved_capabilities = capabilities or AgentCapabilities(
        streaming=True,
        push_notifications=False,
    )

    rpc_endpoint = normalized_url if rpc_url in {"", "/"} else join_url(normalized_url, rpc_url)
    interfaces = [
        AgentInterface(
            url=rpc_endpoint,
            protocol_binding=TransportProtocol.JSONRPC,
            protocol_version=PROTOCOL_VERSION_CURRENT,
        )
    ]
    if enable_rest:
        interfaces.append(
            AgentInterface(
                url=normalized_url,
                protocol_binding=TransportProtocol.HTTP_JSON,
                protocol_version=PROTOCOL_VERSION_CURRENT,
            )
        )

    return AgentCard(
        name=resolved_name,
        description=resolved_description,
        version=version,
        supported_interfaces=interfaces,
        capabilities=resolved_capabilities,
        default_input_modes=list(default_input_modes),
        default_output_modes=list(default_output_modes),
        skills=resolved_skills,
    )


def to_a2a(
    graph: Runnable[Any, Any],
    *,
    name: str | None = None,
    description: str | None = None,
    version: str = "1.0.0",
    url: str = "http://localhost:8000",
    skills: Sequence[AgentSkill] | None = None,
    capabilities: AgentCapabilities | None = None,
    agent_card: AgentCard | None = None,
    default_input_modes: Sequence[str] = ("text/plain", "application/json"),
    default_output_modes: Sequence[str] = ("text/plain", "application/json"),
    rpc_url: str = "/",
    card_url: str = AGENT_CARD_WELL_KNOWN_PATH,
    enable_rest: bool = True,
    enable_v0_3_compat: bool = False,
    task_store: TaskStore | None = None,
    input_key: str | None = None,
    artifact_name: str = "response",
    state_mapper: Callable[[Message, RequestContext], Any] | None = None,
    output_mapper: Callable[[Any], list[Part]] | None = None,
) -> Starlette:
    """Expose a LangGraph graph or LangChain agent as an A2A-compliant ASGI app.

    The returned Starlette application serves:

    - The discovery Agent Card at `/.well-known/agent-card.json` (or `card_url`).
    - A2A JSON-RPC v1.0 endpoints (`POST /` by default).
    - A2A HTTP+JSON REST endpoints (when `enable_rest=True`).

    Args:
        graph: Compiled LangGraph graph (for example from `create_agent`) or runnable.
        name: Public agent name advertised in the Agent Card.
        description: Human-readable agent description advertised in the Agent Card.
        version: Agent version string.
        url: Base public URL where this server will be hosted.
        skills: Optional explicit sequence of `AgentSkill` definitions.
        capabilities: Optional `AgentCapabilities` overrides.
        agent_card: Optional pre-built `AgentCard` bypassing auto-generation.
        default_input_modes: Supported input media types.
        default_output_modes: Supported output media types.
        rpc_url: Path where JSON-RPC requests are handled.
        card_url: Discovery path where the Agent Card JSON is served.
        enable_rest: Whether to mount HTTP+JSON REST A2A routes alongside JSON-RPC.
        enable_v0_3_compat: Whether to accept legacy A2A v0.3 request shapes.
        task_store: Optional custom A2A `TaskStore` (defaults to `InMemoryTaskStore`).
        input_key: Optional explicit state key for text-only inbound messages.
        artifact_name: Name assigned to emitted response artifacts.
        state_mapper: Optional custom inbound message-to-graph-input mapper.
        output_mapper: Optional custom graph-result-to-`Part` mapper.

    Returns:
        Configured `Starlette` ASGI application ready to run under Uvicorn or ASGI
        transports.
    """
    resolved_card = agent_card or build_agent_card(
        graph,
        name=name,
        description=description,
        version=version,
        url=url,
        skills=skills,
        capabilities=capabilities,
        default_input_modes=default_input_modes,
        default_output_modes=default_output_modes,
        rpc_url=rpc_url,
        enable_rest=enable_rest,
    )

    executor = LangGraphAgentExecutor(
        graph,
        input_key=input_key,
        artifact_name=artifact_name,
        state_mapper=state_mapper,
        output_mapper=output_mapper,
    )
    request_handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=task_store or InMemoryTaskStore(),
        agent_card=resolved_card,
    )

    routes: list[BaseRoute] = []
    routes.extend(
        create_agent_card_routes(
            agent_card=resolved_card,
            card_url=card_url,
        )
    )
    routes.extend(
        create_jsonrpc_routes(
            request_handler=request_handler,
            rpc_url=rpc_url,
            enable_v0_3_compat=enable_v0_3_compat,
        )
    )
    if enable_rest:
        routes.extend(
            create_rest_routes(
                request_handler=request_handler,
                enable_v0_3_compat=enable_v0_3_compat,
            )
        )

    @asynccontextmanager
    async def _lifespan(_app: Starlette) -> AsyncIterator[None]:
        try:
            yield
        finally:
            await request_handler.aclose()

    return Starlette(routes=routes, lifespan=_lifespan)
