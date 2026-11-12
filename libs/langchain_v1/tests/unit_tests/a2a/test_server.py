"""Unit tests for `langchain.a2a.server` (`to_a2a`, `LangGraphAgentExecutor`)."""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any, TypedDict

import httpx
from a2a.client import ClientConfig, create_client
from a2a.helpers import new_data_part, new_message, new_text_part
from a2a.server.agent_execution import RequestContext
from a2a.server.context import ServerCallContext
from a2a.server.events import Event, EventQueue
from a2a.types import (
    Role,
    SendMessageRequest,
    TaskState,
    TaskStatusUpdateEvent,
)
from a2a.utils.constants import TransportProtocol
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import interrupt

from langchain.a2a import LangGraphAgentExecutor, to_a2a
from langchain.a2a._conversion import A2AStreamAccumulator
from langchain.agents import create_agent


def _build_billing_agent(*, with_memory: bool = False) -> Any:
    model = GenericFakeChatModel(
        messages=iter(
            [
                AIMessage(content="Invoice #101 is $250."),
                AIMessage(content="Yes, it was due yesterday."),
            ]
        )
    )
    return create_agent(
        model=model,
        name="billing",
        checkpointer=InMemorySaver() if with_memory else None,
    )


async def test_to_a2a_agent_card_and_jsonrpc_and_rest() -> None:
    """Serve an agent over A2A discovery, JSON-RPC, and HTTP+JSON REST."""
    agent = _build_billing_agent()
    app = to_a2a(
        agent,
        name="billing",
        description="Handles invoices and billing questions",
        url="http://testserver",
    )
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as http_client:
        card_resp = await http_client.get("/.well-known/agent-card.json")
        assert card_resp.status_code == 200
        card_json = card_resp.json()
        assert card_json["name"] == "billing"
        assert card_json["description"] == "Handles invoices and billing questions"
        bindings = {iface["protocolBinding"] for iface in card_json["supportedInterfaces"]}
        assert TransportProtocol.JSONRPC in bindings
        assert TransportProtocol.HTTP_JSON in bindings

        # Test over default JSON-RPC
        rpc_client = await create_client(
            "http://testserver",
            client_config=ClientConfig(httpx_client=http_client),
        )
        try:
            acc = A2AStreamAccumulator()
            req = SendMessageRequest(
                message=new_message(
                    [new_text_part("Check invoice #101")],
                    role=Role.ROLE_USER,
                )
            )
            async for event in rpc_client.send_message(req):
                acc.consume(event)
            result = acc.finalize()
            assert result.text == "Invoice #101 is $250."
            assert result.state == TaskState.TASK_STATE_COMPLETED
        finally:
            await rpc_client.close()

        # Test over HTTP+JSON REST transport
        rest_client = await create_client(
            "http://testserver",
            client_config=ClientConfig(
                httpx_client=http_client,
                supported_protocol_bindings=[TransportProtocol.HTTP_JSON],
            ),
        )
        try:
            rest_acc = A2AStreamAccumulator()
            rest_req = SendMessageRequest(
                message=new_message(
                    [new_text_part("When was it due?")],
                    role=Role.ROLE_USER,
                )
            )
            async for event in rest_client.send_message(rest_req):
                rest_acc.consume(event)
            rest_result = rest_acc.finalize()
            assert rest_result.text == "Yes, it was due yesterday."
        finally:
            await rest_client.close()


async def test_to_a2a_multiturn_context_persists_thread_state() -> None:
    """Persist LangGraph conversation history across turns sharing an A2A context_id."""

    def count_turns(state: MessagesState) -> dict[str, Any]:
        human_count = len(state["messages"])
        return {"messages": [AIMessage(content=f"Seen {human_count} messages")]}

    builder = StateGraph(MessagesState)
    builder.add_node("counter", count_turns)
    builder.add_edge(START, "counter")
    builder.add_edge("counter", END)
    graph = builder.compile(checkpointer=InMemorySaver())

    app = to_a2a(graph, name="counter", url="http://testserver")
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as http_client:
        client = await create_client(
            "http://testserver",
            client_config=ClientConfig(httpx_client=http_client),
        )
        try:
            acc1 = A2AStreamAccumulator()
            async for event in client.send_message(
                SendMessageRequest(
                    message=new_message(
                        [new_text_part("turn 1")],
                        context_id="thread-abc",
                        role=Role.ROLE_USER,
                    )
                )
            ):
                acc1.consume(event)
            assert acc1.finalize().text == "Seen 1 messages"

            acc2 = A2AStreamAccumulator()
            async for event in client.send_message(
                SendMessageRequest(
                    message=new_message(
                        [new_text_part("turn 2")],
                        context_id="thread-abc",
                        role=Role.ROLE_USER,
                    )
                )
            ):
                acc2.consume(event)
            assert acc2.finalize().text == "Seen 3 messages"
        finally:
            await client.close()


async def test_to_a2a_interrupt_and_resume() -> None:
    """Translate LangGraph `interrupt()` into `TASK_STATE_INPUT_REQUIRED` and resume."""

    class ApprovalState(TypedDict):
        prompt: str
        result: str

    def approval_node(state: ApprovalState) -> dict[str, Any]:
        decision = interrupt(f"Approve: {state['prompt']}?")
        return {"result": f"Decision={decision}"}

    builder = StateGraph(ApprovalState)
    builder.add_node("approve", approval_node)
    builder.add_edge(START, "approve")
    builder.add_edge("approve", END)
    graph = builder.compile(checkpointer=InMemorySaver())

    app = to_a2a(graph, name="approver", url="http://testserver")
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as http_client:
        client = await create_client(
            "http://testserver",
            client_config=ClientConfig(httpx_client=http_client),
        )
        try:
            acc1 = A2AStreamAccumulator()
            async for event in client.send_message(
                SendMessageRequest(
                    message=new_message(
                        [new_data_part({"prompt": "Wire $500"})],
                        context_id="approval-ctx",
                        role=Role.ROLE_USER,
                    )
                )
            ):
                acc1.consume(event)

            paused = acc1.finalize()
            assert paused.state == TaskState.TASK_STATE_INPUT_REQUIRED
            assert paused.is_input_required is True
            assert paused.text == "Approve: Wire $500?"
            assert paused.task_id is not None

            acc2 = A2AStreamAccumulator()
            async for event in client.send_message(
                SendMessageRequest(
                    message=new_message(
                        [new_text_part("yes")],
                        task_id=paused.task_id,
                        context_id="approval-ctx",
                        role=Role.ROLE_USER,
                    )
                )
            ):
                acc2.consume(event)

            resumed = acc2.finalize()
            assert resumed.state == TaskState.TASK_STATE_COMPLETED
            assert resumed.text == "Decision=yes"
        finally:
            await client.close()


async def test_to_a2a_failure_and_cancellation() -> None:
    """Propagate graph exceptions as FAILED and support canceling running tasks."""
    started = asyncio.Event()

    async def slow_or_failing(state: MessagesState) -> dict[str, Any]:
        last = str(state["messages"][-1].content)
        if last == "fail":
            msg = "Database unavailable"
            raise RuntimeError(msg)
        started.set()
        await asyncio.sleep(5.0)
        return {"messages": [AIMessage(content="Never reached")]}

    builder = StateGraph(MessagesState)
    builder.add_node("worker", slow_or_failing)
    builder.add_edge(START, "worker")
    builder.add_edge("worker", END)
    graph = builder.compile()

    app = to_a2a(graph, name="worker", url="http://testserver")
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as http_client:
        client = await create_client(
            "http://testserver",
            client_config=ClientConfig(httpx_client=http_client),
        )
        try:
            fail_acc = A2AStreamAccumulator()
            async for event in client.send_message(
                SendMessageRequest(
                    message=new_message([new_text_part("fail")], role=Role.ROLE_USER)
                )
            ):
                fail_acc.consume(event)
            failed = fail_acc.finalize(raise_on_failure=False)
            assert failed.state == TaskState.TASK_STATE_FAILED
            assert "Database unavailable" in failed.text

            # Non-streaming start + cancel
            poll_client = await create_client(
                "http://testserver",
                client_config=ClientConfig(
                    streaming=False,
                    httpx_client=http_client,
                ),
            )
            try:

                async def _send_slow() -> None:
                    async for _ev in poll_client.send_message(
                        SendMessageRequest(
                            message=new_message(
                                [new_text_part("sleep")],
                                role=Role.ROLE_USER,
                            )
                        )
                    ):
                        pass

                send_task = asyncio.create_task(_send_slow())
                await asyncio.wait_for(started.wait(), timeout=2.0)
                send_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await send_task
            finally:
                await poll_client.close()
        finally:
            await client.close()


class _RecordingEventQueue(EventQueue):
    """Minimal in-memory `EventQueue` capturing executor events synchronously."""

    def __init__(self) -> None:
        self.events: list[Event] = []

    async def enqueue_event(self, event: Event) -> None:
        self.events.append(event)


async def test_executor_cancel_method_emits_canceled_status() -> None:
    """Verify `LangGraphAgentExecutor.cancel` stops in-flight graph execution."""
    started = asyncio.Event()

    async def slow_node(state: MessagesState) -> dict[str, Any]:
        started.set()
        await asyncio.sleep(5.0)
        return dict(state)

    builder = StateGraph(MessagesState)
    builder.add_node("slow", slow_node)
    builder.add_edge(START, "slow")
    builder.add_edge("slow", END)

    executor = LangGraphAgentExecutor(builder.compile())
    event_queue = _RecordingEventQueue()
    context = RequestContext(
        call_context=ServerCallContext(),
        request=SendMessageRequest(
            message=new_message([new_text_part("wait")], role=Role.ROLE_USER)
        ),
        task_id="task-cancel-1",
        context_id="ctx-cancel-1",
    )

    exec_task = asyncio.create_task(executor.execute(context, event_queue))
    await asyncio.wait_for(started.wait(), timeout=2.0)

    await executor.cancel(context, event_queue)
    await asyncio.wait_for(exec_task, timeout=2.0)

    canceled_events = [
        ev
        for ev in event_queue.events
        if isinstance(ev, TaskStatusUpdateEvent)
        and ev.status.state == TaskState.TASK_STATE_CANCELED
    ]
    assert len(canceled_events) == 1
    assert canceled_events[0].task_id == "task-cancel-1"
