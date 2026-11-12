"""Unit tests for `langchain.a2a._conversion` helpers."""

from __future__ import annotations

from typing import Any

import pytest
from a2a.helpers import new_data_part, new_message, new_text_part
from a2a.types import (
    Artifact,
    Message,
    Role,
    StreamResponse,
    Task,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
)
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableLambda
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import Command, Interrupt
from pydantic import BaseModel

from langchain.a2a import A2AExecutionError
from langchain.a2a._conversion import (
    A2AStreamAccumulator,
    a2a_message_to_graph_input,
    a2a_result_to_langchain_output,
    derive_agent_name_from_url,
    graph_expects_messages,
    graph_result_to_a2a_parts,
    langchain_input_to_a2a_message,
    slugify_identifier,
    split_agent_card_url,
    validate_a2a_url,
)


def test_validate_a2a_url() -> None:
    """Validate valid and invalid A2A endpoint URLs."""
    assert validate_a2a_url(" https://billing.example.com ") == ("https://billing.example.com")
    assert validate_a2a_url("http://localhost:8000/a2a") == "http://localhost:8000/a2a"

    invalid_urls = [
        "",
        "   ",
        "ftp://example.com",
        "file:///etc/passwd",
        "not a url",
        "https://ex ample.com",
    ]
    for invalid in invalid_urls:
        with pytest.raises(ValueError, match="A2A URL"):
            validate_a2a_url(invalid)


def test_split_agent_card_url() -> None:
    """Split base URLs and agent-card JSON URLs accurately."""
    assert split_agent_card_url("https://billing.example.com") == (
        "https://billing.example.com",
        None,
    )
    assert split_agent_card_url("https://billing.example.com/") == (
        "https://billing.example.com",
        None,
    )
    assert split_agent_card_url("https://billing.example.com/.well-known/agent-card.json") == (
        "https://billing.example.com",
        ".well-known/agent-card.json",
    )
    assert split_agent_card_url("https://billing.example.com/team/.well-known/agent.json") == (
        "https://billing.example.com/team",
        ".well-known/agent.json",
    )
    assert split_agent_card_url("https://billing.example.com/v2/custom-card.json?env=prod") == (
        "https://billing.example.com/v2",
        "custom-card.json?env=prod",
    )


def test_slugify_and_derive_name() -> None:
    """Sanitize agent identifiers for LangChain tool and node names."""
    assert slugify_identifier("Billing Agent (v1.0)!") == "Billing_Agent_v1_0"
    assert slugify_identifier("___") == "a2a_agent"
    assert derive_agent_name_from_url("https://billing.example.com:8443/v1") == (
        "billing_example_com_v1"
    )


def test_a2a_message_to_graph_input_adaptation() -> None:
    """Adapt A2A messages to LangGraph messages, custom dicts, and resumes."""
    msg_builder = StateGraph(MessagesState)
    msg_builder.add_node("echo", lambda s: s)
    msg_builder.add_edge(START, "echo")
    msg_builder.add_edge("echo", END)
    msg_graph = msg_builder.compile()

    assert graph_expects_messages(msg_graph) is True
    converted_msg = a2a_message_to_graph_input(
        new_message([new_text_part("Pay invoice")], role=Role.ROLE_USER),
        msg_graph,
    )
    assert isinstance(converted_msg, dict)
    assert len(converted_msg["messages"]) == 1
    assert isinstance(converted_msg["messages"][0], HumanMessage)
    assert converted_msg["messages"][0].content == "Pay invoice"

    plain_runnable: RunnableLambda[Any, str] = RunnableLambda(lambda text: f"ok:{text}")
    assert (
        a2a_message_to_graph_input(
            new_message([new_text_part("hello")], role=Role.ROLE_USER),
            plain_runnable,
        )
        == "hello"
    )
    assert a2a_message_to_graph_input(
        new_message([new_text_part("hello")], role=Role.ROLE_USER),
        plain_runnable,
        input_key="query",
    ) == {"query": "hello"}

    resumed = a2a_message_to_graph_input(
        new_message(
            [new_data_part({"__langgraph_resume__": {"approved": True}})],
            role=Role.ROLE_USER,
        ),
        msg_graph,
        resume_mode=True,
    )
    assert isinstance(resumed, Command)
    assert resumed.resume == {"approved": True}


def test_graph_result_to_a2a_parts() -> None:
    """Convert graph outputs, models, and interrupts into A2A parts."""

    class InvoiceSummary(BaseModel):
        total: int

    parts, is_interrupt = graph_result_to_a2a_parts(
        {
            "messages": [HumanMessage(content="hi"), AIMessage(content="Invoice paid")],
            "structured_response": InvoiceSummary(total=42),
        }
    )
    assert is_interrupt is False
    assert parts[0].text == "Invoice paid"
    assert len(parts) == 2

    int_parts, is_interrupt = graph_result_to_a2a_parts(
        {"__interrupt__": (Interrupt(value="Approve charge?", id="i1"),)}
    )
    assert is_interrupt is True
    assert int_parts[0].text == "Approve charge?"


def test_langchain_input_roundtrip_and_stream_accumulator() -> None:
    """Roundtrip LangChain inputs and accumulate streaming updates cleanly."""
    outbound = langchain_input_to_a2a_message(
        {
            "messages": [
                SystemMessage(content="Be concise"),
                HumanMessage(content="Total due?"),
            ],
            "tenant": "acme",
        },
        context_id="ctx-1",
    )
    assert outbound.context_id == "ctx-1"
    assert outbound.parts[0].text == "Total due?"

    acc = A2AStreamAccumulator()
    d1 = acc.consume(
        StreamResponse(
            task=Task(
                id="t1",
                context_id="ctx-1",
                status=TaskStatus(state=TaskState.TASK_STATE_WORKING),
            )
        )
    )
    assert d1 is None

    d2 = acc.consume(
        StreamResponse(
            artifact_update=TaskArtifactUpdateEvent(
                task_id="t1",
                context_id="ctx-1",
                artifact=Artifact(
                    artifact_id="a1",
                    name="response",
                    parts=[new_text_part("Hello ")],
                ),
                append=False,
            )
        )
    )
    assert d2 == "Hello "

    d3 = acc.consume(
        StreamResponse(
            artifact_update=TaskArtifactUpdateEvent(
                task_id="t1",
                context_id="ctx-1",
                artifact=Artifact(
                    artifact_id="a1",
                    name="response",
                    parts=[new_text_part("world"), new_data_part({"amount": 100})],
                ),
                append=True,
                last_chunk=True,
            )
        )
    )
    assert d3 == "world"

    acc.consume(
        StreamResponse(
            status_update=TaskStatusUpdateEvent(
                task_id="t1",
                context_id="ctx-1",
                status=TaskStatus(state=TaskState.TASK_STATE_COMPLETED),
            )
        )
    )

    collected = acc.finalize()
    assert collected.text == "Hello world"
    assert collected.data == [{"amount": 100}]
    assert collected.task_id == "t1"
    assert collected.context_id == "ctx-1"

    out_state = a2a_result_to_langchain_output(
        collected,
        {"messages": [HumanMessage(content="Total due?")]},
    )
    assert out_state["amount"] == 100
    assert isinstance(out_state["messages"][0], AIMessage)
    assert out_state["messages"][0].content == "Hello world"


def test_stream_accumulator_failure_and_direct_message() -> None:
    """Handle direct agent messages and failed task states properly."""
    msg_acc = A2AStreamAccumulator()
    msg_acc.consume(
        StreamResponse(
            message=Message(
                message_id="m1",
                role=Role.ROLE_AGENT,
                parts=[new_text_part("Direct reply")],
            )
        )
    )
    assert msg_acc.finalize().text == "Direct reply"

    fail_acc = A2AStreamAccumulator()
    fail_acc.consume(
        StreamResponse(
            status_update=TaskStatusUpdateEvent(
                task_id="t-err",
                context_id="ctx-err",
                status=TaskStatus(
                    state=TaskState.TASK_STATE_FAILED,
                    message=new_message([new_text_part("Boom")]),
                ),
            )
        )
    )
    with pytest.raises(A2AExecutionError, match="Boom") as exc_info:
        fail_acc.finalize()
    assert exc_info.value.task_id == "t-err"
    assert exc_info.value.state == "TASK_STATE_FAILED"
