"""Unit tests for `langchain.a2a.client` (`A2ATool` and `A2ARemoteGraph`)."""

from __future__ import annotations

import asyncio
import json
from typing import Any, TypedDict

import httpx
import pytest
from a2a.types import AgentCapabilities, AgentCard, AgentInterface, AgentSkill
from a2a.utils.constants import PROTOCOL_VERSION_CURRENT, TransportProtocol
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import Command, GraphOutput, interrupt
from typing_extensions import Self

from langchain.a2a import A2AExecutionError, A2ARemoteGraph, A2ATool, to_a2a
from langchain.agents import create_agent


class _ToolCallingFakeModel(GenericFakeChatModel):
    """`GenericFakeChatModel` subclass supporting `.bind_tools()` for `create_agent`."""

    def bind_tools(self, tools: Any, **kwargs: Any) -> Self:  # noqa: ARG002
        return self


def _build_billing_app() -> Any:
    def reply_billing(state: MessagesState) -> dict[str, Any]:
        last_text = str(state["messages"][-1].content)
        return {
            "messages": [AIMessage(content=f"Billing handled: {last_text}")],
        }

    builder = StateGraph(MessagesState)
    builder.add_node("billing", reply_billing)
    builder.add_edge(START, "billing")
    builder.add_edge("billing", END)
    return to_a2a(
        builder.compile(),
        name="billing",
        description="Handles invoices and payments",
        url="http://billing.example.com",
        skills=[
            AgentSkill(
                id="invoices",
                name="Manage Invoices",
                description="Look up or charge customer invoices",
                tags=["billing"],
            )
        ],
    )


def test_a2a_tool_from_agent_card_sync_mock_transport() -> None:
    """Resolve Agent Card synchronously from a `.well-known/agent-card.json` URL."""
    card_payload = {
        "name": "Billing Service (v1)",
        "description": "Processes invoices",
        "version": "1.0.0",
        "supportedInterfaces": [
            {
                "url": "https://billing.example.com",
                "protocolBinding": TransportProtocol.JSONRPC,
                "protocolVersion": PROTOCOL_VERSION_CURRENT,
            }
        ],
        "capabilities": {"streaming": True, "pushNotifications": False},
        "defaultInputModes": ["text/plain"],
        "defaultOutputModes": ["text/plain"],
        "skills": [
            {
                "id": "invoice_check",
                "name": "Invoice Check",
                "description": "Inspect invoice status",
                "tags": ["billing"],
            }
        ],
    }

    seen_urls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_urls.append(str(request.url))
        return httpx.Response(200, json=card_payload)

    with httpx.Client(transport=httpx.MockTransport(handler)) as sync_client:
        tool = A2ATool.from_agent_card(
            "https://billing.example.com/.well-known/agent-card.json",
            httpx_client=sync_client,
        )

    assert seen_urls == ["https://billing.example.com/.well-known/agent-card.json"]
    assert tool.name == "Billing_Service_v1"
    assert "Processes invoices" in tool.description
    assert "Invoice Check: Inspect invoice status" in tool.description


async def test_a2a_tool_with_create_agent_and_artifacts() -> None:
    """Call an A2A server via `A2ATool` inside `create_agent`."""
    billing_app = _build_billing_app()
    transport = httpx.ASGITransport(app=billing_app)

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://billing.example.com",
    ) as http_client:
        billing_tool = await A2ATool.afrom_agent_card(
            "http://billing.example.com/.well-known/agent-card.json",
            httpx_client=http_client,
        )
        assert billing_tool.name == "billing"
        assert "Manage Invoices" in billing_tool.description

        # Direct tool invocation with string or dict input
        direct_out = await billing_tool.ainvoke("Status of #42?")
        assert direct_out == "Billing handled: Status of #42?"

        # Use tool inside `create_agent`
        model = _ToolCallingFakeModel(
            messages=iter(
                [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "id": "call_1",
                                "name": "billing",
                                "args": {"query": "Pay invoice #99"},
                            }
                        ],
                    ),
                    AIMessage(content="Done! Billing confirmed invoice #99."),
                ]
            )
        )
        supervisor = create_agent(model=model, tools=[billing_tool])
        result = await supervisor.ainvoke(
            {"messages": [HumanMessage(content="Please pay invoice #99")]}
        )
        assert result["messages"][-1].content == ("Done! Billing confirmed invoice #99.")

        # Artifact mode
        artifact_tool = await A2ATool.afrom_agent_card(
            "http://billing.example.com",
            httpx_client=http_client,
            response_format="content_and_artifact",
        )
        tool_msg = await artifact_tool.ainvoke(
            {
                "name": "billing",
                "id": "tc-1",
                "type": "tool_call",
                "args": {"query": "Check #7"},
            }
        )
        assert tool_msg.content == "Billing handled: Check #7"
        assert tool_msg.artifact["task_id"] is not None
        assert tool_msg.artifact["context_id"] is not None


async def test_a2a_tool_input_required_modes() -> None:
    """Support both returning prompts and raising errors on INPUT_REQUIRED."""

    def needs_more_info(state: MessagesState) -> dict[str, Any]:
        _ = state
        answer = interrupt("Provide PO number")
        return {"messages": [AIMessage(content=f"PO={answer}")]}

    builder = StateGraph(MessagesState)
    builder.add_node("ask", needs_more_info)
    builder.add_edge(START, "ask")
    builder.add_edge("ask", END)
    app = to_a2a(
        builder.compile(checkpointer=InMemorySaver()),
        name="po_agent",
        url="http://po.example.com",
    )
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://po.example.com",
    ) as http_client:
        returning_tool = await A2ATool.afrom_agent_card(
            "http://po.example.com",
            httpx_client=http_client,
            handle_input_required="return",
        )
        prompt = await returning_tool.ainvoke("Process order")
        assert prompt == "Provide PO number"

        raising_tool = await A2ATool.afrom_agent_card(
            "http://po.example.com",
            httpx_client=http_client,
            handle_input_required="error",
        )
        with pytest.raises(A2AExecutionError, match="Provide PO number"):
            await raising_tool.ainvoke("Process order")


async def test_a2a_remote_graph_in_state_graph_and_streaming() -> None:
    """Mount `A2ARemoteGraph` inside a `StateGraph` and stream responses."""
    billing_app = _build_billing_app()
    transport = httpx.ASGITransport(app=billing_app)

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://billing.example.com",
    ) as http_client:
        remote_node = A2ARemoteGraph(
            "http://billing.example.com",
            httpx_client=http_client,
        )
        assert remote_node.get_name() == "billing_example_com"

        builder = StateGraph(MessagesState)
        builder.add_node("billing", remote_node)
        builder.add_edge(START, "billing")
        builder.add_edge("billing", END)
        parent = builder.compile()

        out = await parent.ainvoke(
            {"messages": [HumanMessage(content="Charge $19.99")]},
            config={"configurable": {"thread_id": "parent-thread-1"}},
        )
        assert out["messages"][-1].content == "Billing handled: Charge $19.99"
        # After lazy resolution, card name is cached
        assert remote_node.get_name() == "billing"

        # Standalone v2 invoke
        v2_out = await remote_node.ainvoke("Ping", version="v2")
        assert isinstance(v2_out, GraphOutput)
        assert v2_out.value == "Billing handled: Ping"

        # Streaming modes
        stream_chunks = [
            chunk
            async for chunk in remote_node.astream(
                {"messages": [HumanMessage(content="Stream test")]},
                stream_mode=["messages", "updates", "values"],
            )
        ]
        modes_seen = [mode for mode, _ in stream_chunks]
        assert "messages" in modes_seen
        assert "updates" in modes_seen
        assert "values" in modes_seen

        msg_chunks = [
            payload[0]
            for mode, payload in stream_chunks
            if mode == "messages" and isinstance(payload[0], AIMessageChunk)
        ]
        assert msg_chunks
        assert msg_chunks[0].content == "Billing handled: Stream test"


async def test_a2a_remote_graph_custom_dict_and_resume() -> None:
    """Support custom TypedDict state roundtrips and resume via `Command`."""

    class OrderState(TypedDict):
        order_id: str
        total: int

    def process_order(state: OrderState) -> dict[str, Any]:
        return {"order_id": state["order_id"], "total": state["total"] + 10}

    builder = StateGraph(OrderState)
    builder.add_node("orders", process_order)
    builder.add_edge(START, "orders")
    builder.add_edge("orders", END)
    app = to_a2a(builder.compile(), name="orders", url="http://orders.example.com")

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://orders.example.com",
    ) as http_client:
        remote = A2ARemoteGraph(
            "http://orders.example.com",
            httpx_client=http_client,
        )
        res = await remote.ainvoke({"order_id": "ord-7", "total": 50})
        assert res["order_id"] == "ord-7"
        assert res["total"] == 60

        # Also test resuming an interrupted remote task using A2AExecutionError metadata
        class ReviewState(TypedDict):
            item: str
            approved: bool

        def review(state: ReviewState) -> dict[str, Any]:
            decision = interrupt(f"Approve {state['item']}?")
            return {"item": state["item"], "approved": bool(decision)}

        rev_builder = StateGraph(ReviewState)
        rev_builder.add_node("review", review)
        rev_builder.add_edge(START, "review")
        rev_builder.add_edge("review", END)
        rev_app = to_a2a(
            rev_builder.compile(checkpointer=InMemorySaver()),
            name="reviewer",
            url="http://review.example.com",
        )

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=rev_app),
            base_url="http://review.example.com",
        ) as rev_client:
            rev_remote = A2ARemoteGraph(
                "http://review.example.com",
                httpx_client=rev_client,
            )
            with pytest.raises(A2AExecutionError) as err_info:
                await rev_remote.ainvoke(
                    {"item": "laptop", "approved": False},
                    config={"configurable": {"thread_id": "rev-1"}},
                )
            assert err_info.value.state == "TASK_STATE_INPUT_REQUIRED"
            assert err_info.value.task_id is not None

            resumed = await rev_remote.ainvoke(
                Command(resume=True),
                config={
                    "configurable": {
                        "a2a_context_id": err_info.value.context_id,
                        "a2a_task_id": err_info.value.task_id,
                    }
                },
            )
            assert json.loads(json.dumps(resumed)) == {
                "item": "laptop",
                "approved": True,
            }


def test_a2a_remote_graph_sync_and_metadata_methods() -> None:
    """Verify sync `invoke`/`stream`, `with_config`, `get_graph`, and `NotImplementedError`."""
    billing_app = _build_billing_app()

    # Pass a pre-built AgentCard directly + non-async mock or test card metadata methods
    card = AgentCard(
        name="DirectCard",
        description="Direct card test",
        version="1.0.0",
        supported_interfaces=[
            AgentInterface(
                url="http://billing.example.com",
                protocol_binding=TransportProtocol.JSONRPC,
                protocol_version=PROTOCOL_VERSION_CURRENT,
            )
        ],
        capabilities=AgentCapabilities(streaming=True, push_notifications=False),
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        skills=[AgentSkill(id="bill", name="Bill", description="Bill", tags=["b"])],
    )
    remote = A2ARemoteGraph(card)
    configured = remote.with_config({"configurable": {"thread_id": "t-sync"}})
    assert configured.config is not None
    assert configured.get_name() == "DirectCard"
    assert configured.get_graph().nodes

    with pytest.raises(NotImplementedError):
        remote.get_state({"configurable": {"thread_id": "t-sync"}})
    with pytest.raises(NotImplementedError):
        remote.update_state({"configurable": {"thread_id": "t-sync"}}, {})

    async_http = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=billing_app),
        base_url="http://billing.example.com",
    )
    try:
        # Sync execution using ASGITransport inside fresh sync `run_async_safely` loop
        sync_remote = A2ARemoteGraph(card, httpx_client=async_http)
        assert sync_remote.invoke("Sync check") == "Billing handled: Sync check"
        streamed = list(sync_remote.stream("Sync stream", stream_mode="values"))
        assert streamed == ["Billing handled: Sync stream"]
    finally:
        asyncio.run(async_http.aclose())
