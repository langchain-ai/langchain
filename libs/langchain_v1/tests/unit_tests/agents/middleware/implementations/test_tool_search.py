"""Unit tests for client-side tool search middleware."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool, tool
from langgraph.checkpoint.memory import InMemorySaver

from langchain.agents import create_agent
from langchain.agents.middleware.human_in_the_loop import HumanInTheLoopMiddleware
from langchain.agents.middleware.tool_search import (
    BM25ToolRetriever,
    EmbeddingToolRetriever,
    ToolSearchMiddleware,
)
from langchain.agents.middleware.types import AgentMiddleware, ModelRequest, ModelResponse
from tests.unit_tests.agents.model import FakeToolCallingModel

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


@tool
def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f"Sunny in {city}"


@tool(extras={"defer_loading": True})
def refund_order(order_id: str) -> str:
    """Refund a customer's order."""
    return f"Refunded {order_id}"


@tool(extras={"defer_loading": True})
def get_compensation(employee_id: str) -> str:
    """Return an employee's annual pay."""
    return f"Pay for {employee_id}"


@tool
def list_invoices(customer_id: str) -> str:
    """List invoices for a customer."""
    return f"Invoices for {customer_id}"


class ToolsSeen(AgentMiddleware):
    """Records the tool names each model call is sent, innermost in the stack."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[list[str]] = []

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        self.calls.append([t.name for t in request.tools])  # type: ignore[union-attr]
        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        self.calls.append([t.name for t in request.tools])  # type: ignore[union-attr]
        return await handler(request)


def _stack(*middleware: AgentMiddleware[Any, Any, Any]) -> list[AgentMiddleware[Any, Any, Any]]:
    return list(middleware)


def _call(name: str, call_id: str, **args: Any) -> dict[str, Any]:
    return {"name": name, "args": args, "id": call_id}


def _agent(tool_calls: list[list[dict[str, Any]]], **search: Any) -> tuple[Any, ToolsSeen]:
    seen = ToolsSeen()
    agent = create_agent(
        FakeToolCallingModel(tool_calls=[*tool_calls, []]),
        tools=[get_weather, refund_order, get_compensation, list_invoices],
        middleware=_stack(ToolSearchMiddleware(retriever=BM25ToolRetriever(), **search), seen),
    )
    return agent, seen


def test_searchable_tools_stay_hidden_until_found() -> None:
    agent, seen = _agent(
        [
            [_call("tool_search", "s1", query="refund")],
            [_call("refund_order", "c1", order_id="42")],
            [_call("get_weather", "c2", city="Paris")],
        ]
    )
    result = agent.invoke({"messages": [HumanMessage("refund 42")]})

    first, after_search, later, last = seen.calls
    assert "refund_order" not in first
    assert "get_compensation" not in first
    assert "tool_search" in first
    assert "refund_order" in after_search
    assert "get_compensation" not in after_search
    # A turn that finds nothing new sends the same tool list as the turn before.
    assert later == after_search == last
    results = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert any(m.content == "Refunded 42" for m in results)


def test_loaded_tools_are_sent_without_defer_loading() -> None:
    sent: list[dict[str, Any]] = []

    class Extras(AgentMiddleware):
        def wrap_model_call(
            self, request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
        ) -> ModelResponse:
            sent.extend(
                {"name": t.name, "extras": t.extras}  # type: ignore[union-attr]
                for t in request.tools
                if t.name == "refund_order"  # type: ignore[union-attr]
            )
            return handler(request)

    agent = create_agent(
        FakeToolCallingModel(tool_calls=[[_call("tool_search", "s1", query="refund")], []]),
        tools=[refund_order],
        middleware=_stack(ToolSearchMiddleware(retriever=BM25ToolRetriever()), Extras()),
    )
    agent.invoke({"messages": [HumanMessage("refund")]})

    assert sent == [{"name": "refund_order", "extras": {}}]


def test_parallel_searches_both_load() -> None:
    agent, seen = _agent(
        [
            [
                _call("tool_search", "s1", query="refund order"),
                _call("tool_search", "s2", query="annual pay"),
            ],
        ]
    )
    agent.invoke({"messages": [HumanMessage("two things")]})

    assert {"refund_order", "get_compensation"} <= set(seen.calls[1])


def test_searchable_tools_by_name() -> None:
    agent, seen = _agent([[]], searchable_tools=["list_invoices"])
    agent.invoke({"messages": [HumanMessage("hi")]})

    assert "list_invoices" not in seen.calls[0]
    assert "get_weather" in seen.calls[0]


def test_no_match_says_so() -> None:
    agent, _ = _agent([[_call("tool_search", "s1", query="zzzz")]])
    result = agent.invoke({"messages": [HumanMessage("x")]})

    search = next(m for m in result["messages"] if isinstance(m, ToolMessage))
    assert str(search.content).startswith("No matching tools")


async def test_async_search_loads_tools() -> None:
    agent, seen = _agent([[_call("tool_search", "s1", query="refund an order")]])
    await agent.ainvoke({"messages": [HumanMessage("refund")]})

    assert "refund_order" in seen.calls[1]


def test_tools_found_in_copied_history_are_loaded() -> None:
    agent, seen = _agent([[]])
    history = [
        HumanMessage("refund 42"),
        AIMessage("", tool_calls=[_call("tool_search", "s1", query="refund")]),
        ToolMessage(
            "Found tools", tool_call_id="s1", name="tool_search", artifact=["refund_order"]
        ),
        HumanMessage("go on"),
    ]
    agent.invoke({"messages": history})

    assert "refund_order" in seen.calls[0]
    assert "get_compensation" not in seen.calls[0]


def test_loaded_tools_are_appended_in_the_order_found() -> None:
    agent, seen = _agent(
        [
            [_call("tool_search", "s1", query="annual pay")],
            [_call("tool_search", "s2", query="refund_order")],
        ]
    )
    agent.invoke({"messages": [HumanMessage("pay, then refund")]})

    always_on = seen.calls[0]
    assert seen.calls[1] == [*always_on, "get_compensation"]
    assert seen.calls[2][: len(seen.calls[1])] == seen.calls[1]
    assert "refund_order" in seen.calls[2][len(seen.calls[1]) :]


def test_a_tool_named_like_the_search_tool_is_rejected() -> None:
    @tool("tool_search")
    def shadow(query: str) -> str:
        """A user tool that reuses the reserved name."""
        return query

    agent = create_agent(
        FakeToolCallingModel(tool_calls=[[]]),
        tools=[get_weather, shadow],
        middleware=[ToolSearchMiddleware(retriever=BM25ToolRetriever())],
    )
    with pytest.raises(ValueError, match="reserves the tool name"):
        agent.invoke({"messages": [HumanMessage("hi")]})


def test_unknown_searchable_tool_name_is_rejected() -> None:
    agent, _ = _agent([[]], searchable_tools=["no_such_tool"])
    with pytest.raises(ValueError, match="no_such_tool"):
        agent.invoke({"messages": [HumanMessage("hi")]})


def test_bm25_reindexes_when_a_description_changes() -> None:
    retriever = BM25ToolRetriever()

    def make(description: str) -> Any:
        return StructuredTool.from_function(lambda: "ok", name="helper", description=description)

    assert retriever.retrieve("refund", [make("Refund an order.")], None)  # type: ignore[arg-type]
    assert not retriever.retrieve("refund", [make("Weather forecast.")], None)  # type: ignore[arg-type]


def test_bm25_ranks_exact_name_then_best_match() -> None:
    catalog = [refund_order, get_compensation, list_invoices, get_weather]
    retriever = BM25ToolRetriever()
    runtime: Any = None

    assert retriever.retrieve("list_invoices", catalog, runtime)[0] is list_invoices
    # Split names: "compensation" matches get_compensation by name.
    assert retriever.retrieve("compensation", catalog, runtime)[0] is get_compensation
    # Description words find a tool whose name doesn't contain them.
    assert retriever.retrieve("annual pay", catalog, runtime) == [get_compensation]
    assert retriever.retrieve("nothing relevant here", catalog, runtime) == []


def test_bm25_name_match_on_a_filler_word_does_not_outrank_the_description() -> None:
    about = StructuredTool.from_function(lambda: "v1", name="about", description="API version.")
    refund = StructuredTool.from_function(
        lambda: "ok", name="issue_refund", description="Return a payment for a purchase."
    )

    hits = BM25ToolRetriever().retrieve(
        "tell me about getting my payment back for a purchase",
        [about, refund],
        None,  # type: ignore[arg-type]
    )
    assert hits[0] is refund


def test_bm25_indexes_mcp_server_name() -> None:
    @tool
    def get_insights(period: str) -> str:
        """Get insights for a period."""
        return period

    @tool
    def get_insights_ads(period: str) -> str:
        """Get insights for a period."""
        return period

    get_insights.metadata = {"mcp": {"server": {"name": "hr-payroll"}}}
    get_insights_ads.metadata = {"mcp": {"server": {"name": "ads-manager"}}}

    hits = BM25ToolRetriever().retrieve("payroll insights", [get_insights_ads, get_insights], None)  # type: ignore[arg-type]
    assert hits[0] is get_insights


class ConceptEmbeddings(Embeddings):
    """Embeds text as counts of a few concepts, so paraphrases land near each other."""

    CONCEPTS = (
        ("pay", "salary", "earn", "compensation"),
        ("refund", "money back", "return"),
        ("weather", "rain", "sunny"),
        ("invoice", "bill"),
    )

    def __init__(self) -> None:
        self.embedded: list[str] = []

    def _vector(self, text: str) -> list[float]:
        lowered = text.lower()
        return [float(sum(word in lowered for word in words)) + 0.01 for words in self.CONCEPTS]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.embedded.extend(texts)
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


def test_embedding_retriever_matches_paraphrases() -> None:
    retriever = EmbeddingToolRetriever(ConceptEmbeddings(), k=1)
    catalog = [refund_order, get_compensation, list_invoices, get_weather]

    assert retriever.retrieve("how much does Jane earn", catalog, None) == [get_compensation]  # type: ignore[arg-type]
    assert retriever.retrieve("I want my money back", catalog, None) == [refund_order]  # type: ignore[arg-type]
    hits = retriever.retrieve("pay this bill", catalog, None)  # type: ignore[arg-type]
    assert hits[0].name in {"list_invoices", "get_compensation"}


def test_embedding_retriever_embeds_each_tool_once() -> None:
    embeddings = ConceptEmbeddings()
    retriever = EmbeddingToolRetriever(embeddings)
    catalog = [refund_order, get_compensation]
    retriever.retrieve("refund", catalog, None)  # type: ignore[arg-type]
    retriever.retrieve("pay", catalog, None)  # type: ignore[arg-type]
    retriever.retrieve("pay", [*catalog, get_weather], None)  # type: ignore[arg-type]

    assert len(embeddings.embedded) == 3


def test_embedding_retriever_ignores_tools_outside_the_catalog() -> None:
    retriever = EmbeddingToolRetriever(ConceptEmbeddings())
    retriever.retrieve("pay", [get_compensation, refund_order], None)  # type: ignore[arg-type]

    assert retriever.retrieve("pay", [refund_order], None) == [refund_order]  # type: ignore[arg-type]


def test_agent_searches_with_embeddings() -> None:
    seen = ToolsSeen()
    agent = create_agent(
        FakeToolCallingModel(
            tool_calls=[[_call("tool_search", "s1", query="what does Jane earn")], []]
        ),
        tools=[get_weather, refund_order, get_compensation],
        middleware=_stack(
            ToolSearchMiddleware(retriever=EmbeddingToolRetriever(ConceptEmbeddings(), k=1)),
            seen,
        ),
    )
    agent.invoke({"messages": [HumanMessage("Jane's pay")]})

    assert "get_compensation" in seen.calls[1]
    assert "refund_order" not in seen.calls[1]


class MessagesSeen(AgentMiddleware):
    """Records the tools and tool-call names each model call is sent, innermost."""

    def __init__(self) -> None:
        super().__init__()
        self.tool_names: list[list[str]] = []
        self.calls: list[list[str]] = []
        self.results: list[list[str | None]] = []

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        self.tool_names.append([t.name for t in request.tools])  # type: ignore[union-attr]
        self.calls.append(
            [c["name"] for m in request.messages if isinstance(m, AIMessage) for c in m.tool_calls]
        )
        self.results.append([m.name for m in request.messages if isinstance(m, ToolMessage)])
        return handler(request)


def _dispatch(name: str, call_id: str, **arguments: Any) -> dict[str, Any]:
    return {"name": "call_tool", "args": {"name": name, "arguments": arguments}, "id": call_id}


def _dispatch_agent(
    tool_calls: list[list[dict[str, Any]]], *extra: AgentMiddleware, **kwargs: Any
) -> tuple[Any, MessagesSeen]:
    seen = MessagesSeen()
    search = ToolSearchMiddleware(retriever=BM25ToolRetriever(), loading="dispatch")
    agent = create_agent(
        FakeToolCallingModel(tool_calls=[*tool_calls, []]),
        tools=[get_weather, refund_order, get_compensation],
        middleware=_stack(*extra, search, seen),
        **kwargs,
    )
    return agent, seen


def test_dispatch_runs_a_found_tool_with_a_fixed_tool_list() -> None:
    agent, seen = _dispatch_agent(
        [
            [_call("tool_search", "s1", query="refund")],
            [_dispatch("refund_order", "c1", order_id="42")],
        ]
    )
    result = agent.invoke({"messages": [HumanMessage("refund 42")]})

    assert seen.tool_names[0] == seen.tool_names[1] == seen.tool_names[2]
    assert "refund_order" not in seen.tool_names[2]
    assert "call_tool" in seen.tool_names[0]
    # State records the call under the tool's own name.
    refund = next(
        m for m in result["messages"] if isinstance(m, ToolMessage) and m.name != "tool_search"
    )
    assert (refund.name, refund.content) == ("refund_order", "Refunded 42")
    state_calls = [
        c["name"] for m in result["messages"] if isinstance(m, AIMessage) for c in m.tool_calls
    ]
    assert state_calls == ["tool_search", "refund_order"]
    # The model is shown its own call as it made it.
    assert seen.calls[2] == ["tool_search", "call_tool"]
    assert seen.results[2] == ["tool_search", "call_tool"]


def test_dispatch_search_result_carries_schemas() -> None:
    agent, _ = _dispatch_agent([[_call("tool_search", "s1", query="refund")]])
    result = agent.invoke({"messages": [HumanMessage("refund")]})

    search = next(m for m in result["messages"] if isinstance(m, ToolMessage))
    assert '"order_id"' in search.content
    assert "call_tool(name, arguments)" in search.content


def test_dispatch_rejects_a_tool_that_was_not_found() -> None:
    agent, _ = _dispatch_agent([[_dispatch("refund_order", "c1", order_id="42")]])
    result = agent.invoke({"messages": [HumanMessage("refund 42")]})

    reply = next(m for m in result["messages"] if isinstance(m, ToolMessage))
    assert reply.name == "call_tool"
    assert "has not been found yet" in reply.content


def test_dispatch_rejects_an_unknown_tool() -> None:
    agent, _ = _dispatch_agent([[_dispatch("delete_everything", "c1")]])
    result = agent.invoke({"messages": [HumanMessage("x")]})

    reply = next(m for m in result["messages"] if isinstance(m, ToolMessage))
    assert "No tool named 'delete_everything'" in reply.content


def test_dispatch_checks_json_schema_tools() -> None:
    lookup = StructuredTool.from_function(
        func=lambda **kwargs: f"got {kwargs}",
        name="lookup_ticket",
        description="Look up a support ticket.",
        args_schema={
            "type": "object",
            "properties": {"ticket_id": {"type": "string"}},
            "required": ["ticket_id"],
            "additionalProperties": False,
        },
    )
    lookup.extras = {"defer_loading": True}
    agent = create_agent(
        FakeToolCallingModel(
            tool_calls=[
                [_call("tool_search", "s1", query="support ticket")],
                [_dispatch("lookup_ticket", "c1", id="7")],
                [_dispatch("lookup_ticket", "c2", ticket_id="7")],
                [],
            ]
        ),
        tools=[lookup],
        middleware=[ToolSearchMiddleware(retriever=BM25ToolRetriever(), loading="dispatch")],
    )
    result = agent.invoke({"messages": [HumanMessage("ticket 7")]})

    replies = [m.content for m in result["messages"] if isinstance(m, ToolMessage)][1:]
    assert "Missing required argument(s) for lookup_ticket: ticket_id" in replies[0]
    assert replies[1] == "got {'ticket_id': '7'}"


def test_dispatch_reports_bad_arguments_from_typed_tools() -> None:
    agent, _ = _dispatch_agent(
        [
            [_call("tool_search", "s1", query="refund")],
            [_dispatch("refund_order", "c1", wrong="42")],
        ]
    )
    result = agent.invoke({"messages": [HumanMessage("refund 42")]})

    reply = [m for m in result["messages"] if isinstance(m, ToolMessage)][1]
    assert reply.status == "error"
    assert "order_id" in reply.content


def test_dispatch_call_triggers_approval_for_the_real_tool() -> None:
    agent, _ = _dispatch_agent(
        [
            [_call("tool_search", "s1", query="refund")],
            [_dispatch("refund_order", "c1", order_id="42")],
        ],
        HumanInTheLoopMiddleware(interrupt_on={"refund_order": True}),
        checkpointer=InMemorySaver(),
    )
    result = agent.invoke(
        {"messages": [HumanMessage("refund 42")]}, {"configurable": {"thread_id": "t"}}
    )

    request = result["__interrupt__"][0].value["action_requests"][0]
    assert request["name"] == "refund_order"
    assert request["args"] == {"order_id": "42"}
    assert not [
        m for m in result["messages"] if isinstance(m, ToolMessage) and m.name == "refund_order"
    ]
