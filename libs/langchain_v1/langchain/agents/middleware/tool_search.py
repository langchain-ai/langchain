"""Client-side tool search: the model searches a catalog and loads what it finds."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import TYPE_CHECKING, Annotated, Any, NotRequired, Protocol, TypeGuard

from langchain_core.messages import AIMessage, AnyMessage, ToolMessage
from langchain_core.runnables.config import run_in_executor
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.types import Command

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ContextT,
    ModelRequest,
    ModelResponse,
    PrivateStateAttr,
    ResponseT,
)
from langchain.tools import ToolRuntime  # noqa: TC001  # resolved at runtime for tool injection

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping, Sequence

    from langgraph.runtime import Runtime

DEFER_LOADING = "defer_loading"
"""Tool `extras` key that marks a tool as searchable instead of bound up front."""

SEARCH_TOOL_NAME = "tool_search"
_OWN_TOOL = "langchain_tool_search"
"""Metadata key on the tools this middleware adds."""

_SEARCH_AGAIN = (
    "\n\nIf none of these does what you need, search again with different words "
    "instead of using a near match."
)

_WORD = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|[0-9]+")


class ToolRetriever(Protocol):
    """Ranks searchable tools for a query.

    Thresholds, top-k and reranking belong here, not in the middleware.
    """

    def retrieve(
        self, query: str, tools: Sequence[BaseTool], runtime: ToolRuntime
    ) -> Sequence[BaseTool]:
        """Return the tools that best match `query`, best first."""
        ...


def _keep_first(current: dict[str, str] | None, update: dict[str, str] | None) -> dict[str, str]:
    """Merge loaded-tool records so a tool keeps the search that first loaded it."""
    merged = dict(current or {})
    for name, anchor in (update or {}).items():
        merged.setdefault(name, anchor)
    return merged


class ToolSearchState(AgentState[ResponseT]):
    """State for `ToolSearchMiddleware`."""

    tool_search_loaded: NotRequired[Annotated[dict[str, str], PrivateStateAttr, _keep_first]]
    """Loaded tool name to the `tool_call_id` of the search that found it, in load order."""


def _tokens(text: str) -> list[str]:
    return [word.casefold() for word in _WORD.findall(text)]


def _index_text(tool: BaseTool) -> list[str]:
    args = " ".join(tool.args)
    return [*_tokens(tool.description), *_tokens(args), *_tokens(_server_name(tool))]


def _server_name(tool: BaseTool) -> str:
    """Return the MCP server a tool came from, or an empty string."""
    server = ((tool.metadata or {}).get("mcp") or {}).get("server")
    name = server.get("name") if isinstance(server, dict) else server
    return name if isinstance(name, str) else ""


class _Bm25:
    """Okapi BM25 over pre-tokenized documents."""

    def __init__(
        self, documents: Sequence[Sequence[str]], k1: float = 1.2, b: float = 0.75
    ) -> None:
        self._k1 = k1
        self._b = b
        self._counts = [Counter(document) for document in documents]
        average = sum(map(len, documents)) / len(documents) if documents else 0
        self._lengths = [len(document) / (average or 1) for document in documents]
        self._having = Counter(term for count in self._counts for term in count)

    def scores(self, query: Sequence[str]) -> list[float]:
        total = len(self._counts)
        scores = [0.0] * total
        for term in set(query):
            having = self._having[term]
            if not having:
                continue
            idf = math.log(1 + (total - having + 0.5) / (having + 0.5))
            for index, count in enumerate(self._counts):
                frequency = count[term]
                if frequency:
                    norm = self._k1 * (1 - self._b + self._b * self._lengths[index])
                    scores[index] += idf * frequency * (self._k1 + 1) / (frequency + norm)
        return scores


class BM25ToolRetriever:
    """Keyword search over tool names, descriptions, argument names and MCP server names.

    All of a tool's text is scored as one document. An exact name match ranks first.
    """

    def __init__(self, *, k: int = 5) -> None:
        """Initialize the retriever.

        Args:
            k: Maximum number of tools to return. Every returned tool is loaded
                for the rest of the thread, so keep it small.
        """
        self.k = k
        self._cache: tuple[tuple[object, ...], _Bm25] | None = None

    def _index(self, tools: Sequence[BaseTool]) -> _Bm25:
        key = tuple((tool.name, tool.description, *tool.args) for tool in tools)
        cache = self._cache
        if cache is None or cache[0] != key:
            index = _Bm25([[*_tokens(tool.name), *_index_text(tool)] for tool in tools])
            cache = self._cache = (key, index)
        return cache[1]

    def retrieve(
        self,
        query: str,
        tools: Sequence[BaseTool],
        runtime: ToolRuntime,  # noqa: ARG002  # protocol signature
    ) -> Sequence[BaseTool]:
        """Return up to `k` tools matching `query`, best first."""
        if not tools:
            return []
        scores = self._index(tools).scores(_tokens(query))
        exact = query.strip()
        ranked = sorted(
            (tool.name != exact, -score, tool.name, index)
            for index, (tool, score) in enumerate(zip(tools, scores, strict=True))
            if score > 0 or tool.name == exact
        )
        return [tools[entry[-1]] for entry in ranked[: self.k]]


class ToolSearchMiddleware(AgentMiddleware[ToolSearchState[ResponseT], ContextT, ResponseT]):
    """Let the model search a tool catalog and load only the tools it finds.

    Searchable tools are hidden from the model until a `tool_search` call returns
    them. Loaded tools are appended to the tool list and never unloaded.

    Place this middleware after any middleware that swaps the model, and before
    prompt caching, so it shapes the final request.

    Search covers every searchable tool given to the agent, including tools an
    outer middleware removes from a request. Hiding a tool does not block it: the
    agent still runs a hidden tool if the model calls it by name.

    !!! warning

        Experimental. The retriever signature may change.
    """

    state_schema = ToolSearchState  # type: ignore[assignment]

    def __init__(
        self,
        *,
        retriever: ToolRetriever,
        searchable_tools: Sequence[str | BaseTool] | None = None,
    ) -> None:
        """Initialize tool search.

        Args:
            retriever: Ranks searchable tools for a query.
            searchable_tools: Tools or tool names to make searchable, in addition to
                tools whose `extras` set `defer_loading`.
        """
        super().__init__()
        self.retriever = retriever
        self.searchable_names = {
            t if isinstance(t, str) else t.name for t in searchable_tools or ()
        }
        self.tools = [self._search_tool()]

    def _is_searchable(self, tool: BaseTool | dict[str, Any]) -> TypeGuard[BaseTool]:
        if not isinstance(tool, BaseTool) or tool.name == SEARCH_TOOL_NAME:
            return False
        return (tool.extras or {}).get(DEFER_LOADING) is True or tool.name in self.searchable_names

    def _catalog(self, runtime: ToolRuntime) -> list[BaseTool]:
        return [tool for tool in runtime.tools if self._is_searchable(tool)]

    def _found(self, hits: Sequence[BaseTool], runtime: ToolRuntime) -> Command[Any]:
        if hits:
            lines = [f"- {tool.name}: {tool.description}" for tool in hits]
            content = "Found tools, now available to call:\n" + "\n".join(lines) + _SEARCH_AGAIN
        else:
            content = "No matching tools. Try other words."
        return Command(
            update={
                "tool_search_loaded": {tool.name: runtime.tool_call_id for tool in hits},
                "messages": [
                    ToolMessage(
                        content,
                        tool_call_id=runtime.tool_call_id,
                        name=SEARCH_TOOL_NAME,
                        artifact=[tool.name for tool in hits],
                    )
                ],
            }
        )

    def _search_tool(self) -> BaseTool:
        def search(query: str, runtime: ToolRuntime) -> Command[Any]:
            return self._found(
                self.retriever.retrieve(query, self._catalog(runtime), runtime), runtime
            )

        async def asearch(query: str, runtime: ToolRuntime) -> Command[Any]:
            catalog = self._catalog(runtime)
            hits = await run_in_executor(None, self.retriever.retrieve, query, catalog, runtime)
            return self._found(hits, runtime)

        return StructuredTool.from_function(
            func=search,
            coroutine=asearch,
            name=SEARCH_TOOL_NAME,
            metadata={_OWN_TOOL: True},
            description=(
                "Search for more tools by describing the capability you need. "
                "Tools it finds become available on your next turn."
            ),
        )

    def _validate(self, tools: Sequence[BaseTool | dict[str, Any]]) -> None:
        """Raise on a tool that shadows one of ours, or an unknown searchable name."""
        present = {tool.name: tool for tool in tools if isinstance(tool, BaseTool)}
        shadowed = sorted(
            own.name
            for own in self.tools
            if own.name in present and not (present[own.name].metadata or {}).get(_OWN_TOOL)
        )
        if shadowed:
            msg = f"ToolSearchMiddleware reserves the tool name(s): {', '.join(shadowed)}"
            raise ValueError(msg)
        unknown = sorted(self.searchable_names - present.keys())
        if unknown:
            msg = (
                "ToolSearchMiddleware: searchable_tools references tool(s) not bound to "
                f"the model: {', '.join(unknown)}"
            )
            raise ValueError(msg)

    def _prepare(self, request: ModelRequest[ContextT]) -> ModelRequest[ContextT]:
        self._validate(request.tools)
        loaded = _loaded(request.state)
        found = {tool.name: tool for tool in request.tools if self._is_searchable(tool)}
        kept = [tool for tool in request.tools if not self._is_searchable(tool)]
        kept.extend(_without_defer(found[name]) for name in loaded if name in found)
        return request.override(tools=kept)

    def before_agent(
        self,
        state: ToolSearchState[ResponseT],
        runtime: Runtime[ContextT],  # noqa: ARG002  # hook signature
    ) -> dict[str, Any] | None:
        """Reload tools that earlier searches in the message history found.

        A run that starts from copied messages, such as a forked subagent, has the
        search results but not the private record of what they loaded.
        """
        loaded = _loaded(state)
        missing = {
            name: anchor
            for name, anchor in _found_in(state["messages"]).items()
            if name not in loaded
        }
        return {"tool_search_loaded": missing} if missing else None

    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]],
    ) -> ModelResponse[ResponseT] | AIMessage:
        """Hide unloaded searchable tools and show loaded ones."""
        return handler(self._prepare(request))

    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]],
    ) -> ModelResponse[ResponseT] | AIMessage:
        """Hide unloaded searchable tools and show loaded ones."""
        return await handler(self._prepare(request))


def _found_in(messages: Sequence[AnyMessage]) -> dict[str, str]:
    """Return each tool the search results in `messages` found, with the first search that did."""
    found: dict[str, str] = {}
    for message in messages:
        if isinstance(message, ToolMessage) and message.name == SEARCH_TOOL_NAME:
            for name in message.artifact if isinstance(message.artifact, list) else ():
                found.setdefault(name, message.tool_call_id)
    return found


def _loaded(state: Mapping[str, Any]) -> dict[str, str]:
    """Return the loaded-tool record: tool name to the search that found it, in found order."""
    return state.get("tool_search_loaded") or {}


def _without_defer(tool: BaseTool) -> BaseTool:
    extras = tool.extras or {}
    if DEFER_LOADING not in extras:
        return tool
    rest = {key: value for key, value in extras.items() if key != DEFER_LOADING}
    return tool.model_copy(update={"extras": rest})
