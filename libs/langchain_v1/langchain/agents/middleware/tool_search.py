"""Client-side tool search: the model searches a catalog and loads what it finds."""

from __future__ import annotations

import importlib.util
import json
import math
import re
from collections import Counter
from dataclasses import replace
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Literal,
    NotRequired,
    Protocol,
    TypeGuard,
    cast,
)

from langchain_core.documents import Document
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    ToolCall,
    ToolMessage,
)
from langchain_core.runnables.config import run_in_executor
from langchain_core.tools import BaseTool, StructuredTool, ToolException
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.vectorstores import InMemoryVectorStore
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

    from langchain_core.embeddings import Embeddings
    from langgraph.runtime import Runtime

DEFER_LOADING = "defer_loading"
"""Tool `extras` key that marks a tool as searchable instead of bound up front."""

SEARCH_TOOL_NAME = "tool_search"
_OWN_TOOL = "langchain_tool_search"
"""Metadata key on the tools this middleware adds."""
DISPATCH_TOOL_NAME = "call_tool"

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


class EmbeddingToolRetriever:
    """Semantic search over tool names and descriptions.

    Finds tools whose wording differs from the query, such as "annual pay" for
    `get_compensation`. Each tool is embedded once, and again only if its
    description changes.
    """

    def __init__(self, embeddings: Embeddings, *, k: int = 5) -> None:
        """Initialize the retriever.

        Args:
            embeddings: Model that embeds tool text and queries.
            k: Maximum number of tools to return. Every returned tool is loaded
                for the rest of the thread, so keep it small.
        """
        if importlib.util.find_spec("numpy") is None:
            msg = "EmbeddingToolRetriever needs numpy. Install it with `pip install numpy`."
            raise ImportError(msg)
        self.k = k
        self._store = InMemoryVectorStore(embeddings)
        self._indexed: dict[str, str] = {}

    def _pending(self, tools: Sequence[BaseTool]) -> list[Document]:
        return [
            Document(text, id=tool.name, metadata={"name": tool.name})
            for tool in tools
            if self._indexed.get(tool.name) != (text := _embedding_text(tool))
        ]

    def _mark_indexed(self, documents: Sequence[Document]) -> None:
        self._indexed.update({d.metadata["name"]: d.page_content for d in documents})

    def _ranked(self, documents: Sequence[Document], tools: Sequence[BaseTool]) -> list[BaseTool]:
        by_name = {tool.name: tool for tool in tools}
        names = [document.metadata["name"] for document in documents]
        return [by_name[name] for name in names if name in by_name][: self.k]

    def _fetch(self, tools: Sequence[BaseTool]) -> int:
        # Over-fetch, so tools indexed earlier but absent from this catalog don't crowd the top k.
        return self.k + max(0, len(self._indexed) - len(tools))

    def retrieve(
        self,
        query: str,
        tools: Sequence[BaseTool],
        runtime: ToolRuntime,  # noqa: ARG002  # protocol signature
    ) -> Sequence[BaseTool]:
        """Return up to `k` tools closest in meaning to `query`, best first."""
        if not tools:
            return []
        if pending := self._pending(tools):
            self._store.add_documents(pending)
            self._mark_indexed(pending)
        return self._ranked(self._store.similarity_search(query, k=self._fetch(tools)), tools)


def _embedding_text(tool: BaseTool) -> str:
    server = _server_name(tool)
    prefix = f"[{server}] " if server else ""
    return f"{prefix}{' '.join(_tokens(tool.name))}: {tool.description}"


class ToolSearchMiddleware(AgentMiddleware[ToolSearchState[ResponseT], ContextT, ResponseT]):
    """Let the model search a tool catalog and load only the tools it finds.

    Searchable tools are hidden from the model until a `tool_search` call returns
    them. Loaded tools are never unloaded.

    With `loading="bind"`, loaded tools become ordinary tools, appended to the
    tool list.

    With `loading="dispatch"`, the tool list never changes on any model. The search
    result carries each found tool's schema and the model runs one through
    `call_tool(name, arguments)`. The call is recorded in state under the tool's
    own name, so approval, limits and tracing treat it as a direct call.

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
        loading: Literal["bind", "dispatch"] = "bind",
    ) -> None:
        """Initialize tool search.

        Args:
            retriever: Ranks searchable tools for a query.
            searchable_tools: Tools or tool names to make searchable, in addition to
                tools whose `extras` set `defer_loading`.
            loading: How the model reaches a found tool. `"bind"` makes it an
                ordinary tool. `"dispatch"` keeps the tool list fixed and runs it
                through `call_tool`.
        """
        super().__init__()
        self.retriever = retriever
        self.loading = loading
        self.searchable_names = {
            t if isinstance(t, str) else t.name for t in searchable_tools or ()
        }
        self.tools = [self._search_tool()]
        if loading == "dispatch":
            self.tools.append(self._dispatch_tool())

    def _is_searchable(self, tool: BaseTool | dict[str, Any]) -> TypeGuard[BaseTool]:
        if not isinstance(tool, BaseTool) or tool.name == SEARCH_TOOL_NAME:
            return False
        return (tool.extras or {}).get(DEFER_LOADING) is True or tool.name in self.searchable_names

    def _catalog(self, runtime: ToolRuntime) -> list[BaseTool]:
        return [tool for tool in runtime.tools if self._is_searchable(tool)]

    def _found(self, hits: Sequence[BaseTool], runtime: ToolRuntime) -> Command[Any]:
        if hits and self.loading == "dispatch":
            specs = [convert_to_openai_tool(tool)["function"] for tool in hits]
            header = f"Found tools. Run one with {DISPATCH_TOOL_NAME}(name, arguments):\n"
            content = header + json.dumps(specs, indent=1) + _SEARCH_AGAIN
        elif hits:
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

    def _dispatch_problem(
        self, name: object, arguments: object, catalog: dict[str, BaseTool], loaded: dict[str, str]
    ) -> str | None:
        """Return why `call_tool(name, arguments)` can't run, or `None` if it can."""
        if not isinstance(name, str) or name not in catalog:
            return f"No tool named {name!r}. Use {SEARCH_TOOL_NAME} to find tools."
        if name not in loaded:
            return f"Tool {name!r} has not been found yet. Call {SEARCH_TOOL_NAME} first."
        if not isinstance(arguments, dict):
            return "`arguments` must be an object of the tool's arguments."
        return _schema_problem(catalog[name], arguments)

    def _dispatch_tool(self) -> BaseTool:
        def call_tool(
            name: str, runtime: ToolRuntime, arguments: dict[str, Any] | None = None
        ) -> str:
            catalog = {tool.name: tool for tool in self._catalog(runtime)}
            loaded = _loaded(runtime.state)
            problem = self._dispatch_problem(name, _as_object(arguments or {}), catalog, loaded)
            msg = problem or "The call could not be dispatched."
            raise ToolException(msg)

        return StructuredTool.from_function(
            func=call_tool,
            name=DISPATCH_TOOL_NAME,
            metadata={_OWN_TOOL: True},
            handle_tool_error=True,
            description=(
                f"Run a tool found by {SEARCH_TOOL_NAME}. Pass the tool's name and its "
                "arguments as an object."
            ),
        )

    def _to_direct_calls(
        self, response: ModelResponse[ResponseT] | AIMessage, request: ModelRequest[ContextT]
    ) -> ModelResponse[ResponseT] | AIMessage:
        """Record valid `call_tool` calls under the name of the tool they run."""
        if self.loading != "dispatch":
            return response
        catalog = {t.name: t for t in request.tools if self._is_searchable(t)}
        loaded = _loaded(request.state)

        def direct(call: ToolCall) -> ToolCall:
            name = call["args"].get("name")
            arguments = _as_object(call["args"].get("arguments") or {})
            if call["name"] != DISPATCH_TOOL_NAME or self._dispatch_problem(
                name, arguments, catalog, loaded
            ):
                return call
            args = cast("dict[str, Any]", arguments)
            return ToolCall(name=str(name), args=args, id=call["id"], type="tool_call")

        if isinstance(response, AIMessage):
            return _with_calls(response, direct)
        result = [
            _with_calls(m, direct) if isinstance(m, AIMessage) else m for m in response.result
        ]
        return replace(response, result=result)

    def _to_dispatch_calls(self, request: ModelRequest[ContextT]) -> list[AnyMessage]:
        """Show the model its past calls to searchable tools as `call_tool` calls."""
        names = {t.name for t in request.tools if self._is_searchable(t)}

        def dispatch(call: ToolCall) -> ToolCall:
            if call["name"] not in names:
                return call
            args = {"name": call["name"], "arguments": call["args"]}
            return ToolCall(name=DISPATCH_TOOL_NAME, args=args, id=call["id"], type="tool_call")

        messages: list[AnyMessage] = []
        dispatched: set[str | None] = set()
        for message in request.messages:
            if isinstance(message, AIMessage):
                dispatched.update(c["id"] for c in message.tool_calls if c["name"] in names)
                messages.append(_with_calls(message, dispatch))
            elif isinstance(message, ToolMessage) and message.tool_call_id in dispatched:
                messages.append(message.model_copy(update={"name": DISPATCH_TOOL_NAME}))
            else:
                messages.append(message)
        return messages

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
        if self.loading == "dispatch":
            kept = [tool for tool in request.tools if not self._is_searchable(tool)]
            return request.override(tools=kept, messages=self._to_dispatch_calls(request))
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
        return self._to_direct_calls(handler(self._prepare(request)), request)

    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]],
    ) -> ModelResponse[ResponseT] | AIMessage:
        """Hide unloaded searchable tools and show loaded ones."""
        return self._to_direct_calls(await handler(self._prepare(request)), request)


def _as_object(arguments: object) -> object:
    """Parse `arguments` when a model sent the object as a JSON string."""
    if isinstance(arguments, str):
        try:
            return json.loads(arguments)
        except ValueError:
            return arguments
    return arguments


def _schema_problem(tool: BaseTool, arguments: dict[str, Any]) -> str | None:
    """Check `arguments` against a JSON-schema tool, which the tool node doesn't validate."""
    schema = tool.args_schema
    if not isinstance(schema, dict):
        return None
    missing = [key for key in schema.get("required", ()) if key not in arguments]
    if missing:
        return f"Missing required argument(s) for {tool.name}: {', '.join(missing)}."
    properties = schema.get("properties", {})
    unknown = [key for key in arguments if key not in properties]
    if unknown and schema.get("additionalProperties") is False:
        return f"Unknown argument(s) for {tool.name}: {', '.join(unknown)}."
    return None


def _with_calls(message: AIMessage, change: Callable[[ToolCall], ToolCall]) -> AIMessage:
    """Return `message` with `change` applied to its tool calls and their content blocks."""
    if not message.tool_calls:
        return message
    calls = [change(call) for call in message.tool_calls]
    changed: dict[str | None, ToolCall] = {
        c["id"]: c
        for c, old in zip(calls, message.tool_calls, strict=True)
        if c is not old and c["id"]
    }
    if not changed:
        return message
    content = message.content
    if isinstance(content, list):
        content = [_with_call_block(block, changed) for block in content]
    return message.model_copy(update={"tool_calls": calls, "content": content})


def _with_call_block(block: Any, changed: dict[str | None, ToolCall]) -> Any:
    """Mirror a changed tool call into a provider or standard tool-call content block."""
    if not isinstance(block, dict) or "name" not in block:
        return block
    call = changed.get(block.get("call_id")) or changed.get(block.get("id"))
    if call is None:
        return block
    updated = {**block, "name": call["name"]}
    for key in ("input", "args"):
        if key in block:
            updated[key] = call["args"]
    if "arguments" in block:
        updated["arguments"] = json.dumps(call["args"])
    return updated


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
