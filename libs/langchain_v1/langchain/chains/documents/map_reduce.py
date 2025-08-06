"""Map-Reduce Extraction Implementation using LangGraph Send API."""

from __future__ import annotations

import operator
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Callable,
    Generic,
    Literal,
    Optional,
    Union,
    cast,
)

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from typing_extensions import NotRequired, TypedDict

from langchain._internal._documents import format_document_xml
from langchain._internal._prompts import aresolve_prompt, resolve_prompt
from langchain._internal._typing import ContextT, StateNode
from langchain._internal._utils import RunnableCallable
from langchain.chat_models import init_chat_model

if TYPE_CHECKING:
    from langchain_core.documents import Document
    from langchain_core.language_models.chat_models import BaseChatModel

    # Pycharm is unable to identify that AIMessage is used in the cast below
    from langchain_core.messages import (
        AIMessage,
        MessageLikeRepresentation,
    )
    from langchain_core.runnables import RunnableConfig
    from langgraph.runtime import Runtime
    from pydantic import BaseModel


class ExtractionResult(TypedDict):
    """Result from processing a document or group of documents."""

    indexes: list[int]
    """Document indexes that contributed to this result."""
    result: Any
    """Extracted result from the document(s)."""


class MapReduceState(TypedDict):
    """State for map-reduce extraction chain.

    This state tracks the map-reduce process where documents are processed
    in parallel during the map phase, then combined in the reduce phase.
    """

    documents: list[Document]
    """List of documents to process."""
    map_results: Annotated[list[ExtractionResult], operator.add]
    """Individual results from the map phase."""
    result: NotRequired[Any]
    """Final combined result from the reduce phase if applicable."""


# The payload for the map phase is a list of documents and their indexes.
# The current implementation only supports a single document per map operation,
# but the structure allows for future expansion to process a group of documents.
# A user would provide an input split function that returns groups of documents
# to process together, if desired.
class MapState(TypedDict):
    """State for individual map operations."""

    documents: list[Document]
    """List of documents to process in map phase."""
    indexes: list[int]
    """List of indexes of the documents in the original list."""


class InputSchema(TypedDict):
    """Input schema for the map-reduce extraction chain.

    Defines the expected input format when invoking the extraction chain.
    """

    documents: list[Document]
    """List of documents to process."""


class OutputSchema(TypedDict):
    """Output schema for the map-reduce extraction chain.

    Defines the format of the final result returned by the chain.
    """

    map_results: list[ExtractionResult]
    """List of individual extraction results from the map phase."""

    result: Any
    """Final combined result from all documents."""


class MapReduceNodeUpdate(TypedDict):
    """Update returned by map-reduce nodes."""

    map_results: NotRequired[list[ExtractionResult]]
    """Updated results after map phase."""
    result: NotRequired[Any]
    """Final result after reduce phase."""


class _MapReduceExtractor(Generic[ContextT]):
    """Map-reduce extraction implementation using LangGraph Send API.

    This implementation uses a language model to process documents through up
    to two phases:

    1. **Map Phase**: Each document is processed independently by the LLM using
       the configured map_prompt to generate individual extraction results.
    2. **Reduce Phase (Optional)**: Individual results can optionally be
       combined using either:
       - The default LLM-based reducer with the configured reduce_prompt
       - A custom reducer function (which can be non-LLM based)
       - Skipped entirely by setting reduce=None

    The map phase processes documents in parallel for efficiency, making this approach
    well-suited for large document collections. The reduce phase is flexible and can be
    customized or omitted based on your specific requirements.
    """

    def __init__(
        self,
        model: Union[BaseChatModel, str],
        *,
        map_prompt: Union[
            str,
            None,
            Callable[
                [MapState, Runtime[ContextT]],
                list[MessageLikeRepresentation],
            ],
        ] = None,
        reduce_prompt: Union[
            str,
            None,
            Callable[
                [MapReduceState, Runtime[ContextT]],
                list[MessageLikeRepresentation],
            ],
        ] = None,
        reduce: Union[
            Literal["default_reducer"],
            None,
            StateNode,
        ] = "default_reducer",
        context_schema: type[ContextT] | None = None,
        response_format: Optional[type[BaseModel]] = None,
    ) -> None:
        """Initialize the MapReduceExtractor.

        Args:
            model: The language model either a chat model instance
                  (e.g., `ChatAnthropic()`) or string identifier
                  (e.g., `"anthropic:claude-sonnet-4-20250514"`)
            map_prompt: Prompt for individual document processing. Can be:
                - str: A system message string
                - None: Use default system message
                - Callable: A function that takes (state, runtime) and returns messages
            reduce_prompt: Prompt for combining results. Can be:
                - str: A system message string
                - None: Use default system message
                - Callable: A function that takes (state, runtime) and returns messages
            reduce: Controls the reduce behavior. Can be:
                - "default_reducer": Use the default LLM-based reduce step
                - None: Skip the reduce step entirely
                - Callable: Custom reduce function (sync or async)
            context_schema: Optional context schema for the LangGraph runtime.
            response_format: Optional pydantic BaseModel for structured output.
        """
        if (reduce is None or callable(reduce)) and reduce_prompt is not None:
            msg = (
                "reduce_prompt must be None when reduce is None or a custom "
                "callable. Custom reduce functions handle their own logic and "
                "should not use reduce_prompt."
            )
            raise ValueError(msg)

        self.response_format = response_format

        if isinstance(model, str):
            model = init_chat_model(model)

        self.model = (
            model.with_structured_output(response_format) if response_format else model
        )
        self.map_prompt = map_prompt
        self.reduce_prompt = reduce_prompt
        self.reduce = reduce
        self.context_schema = context_schema

    def _get_map_prompt(
        self, state: MapState, runtime: Runtime[ContextT]
    ) -> list[MessageLikeRepresentation]:
        """Generate the LLM prompt for processing documents."""
        documents = state["documents"]
        user_content = "\n\n".join(format_document_xml(doc) for doc in documents)
        default_system = (
            "You are a helpful assistant that processes documents. "
            "Please process the following documents and provide a result."
        )

        return resolve_prompt(
            self.map_prompt,
            state,
            runtime,
            user_content,
            default_system,
        )

    async def _aget_map_prompt(
        self, state: MapState, runtime: Runtime[ContextT]
    ) -> list[MessageLikeRepresentation]:
        """Generate the LLM prompt for processing documents in the map phase.

        Async version.
        """
        documents = state["documents"]
        user_content = "\n\n".join(format_document_xml(doc) for doc in documents)
        default_system = (
            "You are a helpful assistant that processes documents. "
            "Please process the following documents and provide a result."
        )

        return await aresolve_prompt(
            self.map_prompt,
            state,
            runtime,
            user_content,
            default_system,
        )

    def _get_reduce_prompt(
        self, state: MapReduceState, runtime: Runtime[ContextT]
    ) -> list[MessageLikeRepresentation]:
        """Generate the LLM prompt for combining individual results.

        Combines map results in the reduce phase.
        """
        map_results = state.get("map_results", [])
        if not map_results:
            msg = (
                "Internal programming error: Results must exist when reducing. "
                "This indicates that the reduce node was reached without "
                "first processing the map nodes, which violates "
                "the expected graph execution order."
            )
            raise AssertionError(msg)

        results_text = "\n\n".join(
            f"Result {i + 1} (from documents "
            f"{', '.join(map(str, result['indexes']))}):\n{result['result']}"
            for i, result in enumerate(map_results)
        )
        user_content = (
            f"Please combine the following results into a single, "
            f"comprehensive result:\n\n{results_text}"
        )
        default_system = (
            "You are a helpful assistant that combines multiple results. "
            "Given several individual results, create a single comprehensive "
            "result that captures the key information from all inputs while "
            "maintaining conciseness and coherence."
        )

        return resolve_prompt(
            self.reduce_prompt,
            state,
            runtime,
            user_content,
            default_system,
        )

    async def _aget_reduce_prompt(
        self, state: MapReduceState, runtime: Runtime[ContextT]
    ) -> list[MessageLikeRepresentation]:
        """Generate the LLM prompt for combining individual results.

        Async version of reduce phase.
        """
        map_results = state.get("map_results", [])
        if not map_results:
            msg = (
                "Internal programming error: Results must exist when reducing. "
                "This indicates that the reduce node was reached without "
                "first processing the map nodes, which violates "
                "the expected graph execution order."
            )
            raise AssertionError(msg)

        results_text = "\n\n".join(
            f"Result {i + 1} (from documents "
            f"{', '.join(map(str, result['indexes']))}):\n{result['result']}"
            for i, result in enumerate(map_results)
        )
        user_content = (
            f"Please combine the following results into a single, "
            f"comprehensive result:\n\n{results_text}"
        )
        default_system = (
            "You are a helpful assistant that combines multiple results. "
            "Given several individual results, create a single comprehensive "
            "result that captures the key information from all inputs while "
            "maintaining conciseness and coherence."
        )

        return await aresolve_prompt(
            self.reduce_prompt,
            state,
            runtime,
            user_content,
            default_system,
        )

    def create_map_node(self) -> RunnableCallable:
        """Create a LangGraph node that processes individual documents using the LLM."""

        def _map_node(
            state: MapState, runtime: Runtime[ContextT], config: RunnableConfig
        ) -> dict[str, list[ExtractionResult]]:
            prompt = self._get_map_prompt(state, runtime)
            response = cast("AIMessage", self.model.invoke(prompt, config=config))
            result = response if self.response_format else response.text()
            extraction_result: ExtractionResult = {
                "indexes": state["indexes"],
                "result": result,
            }
            return {"map_results": [extraction_result]}

        async def _amap_node(
            state: MapState,
            runtime: Runtime[ContextT],
            config: RunnableConfig,
        ) -> dict[str, list[ExtractionResult]]:
            prompt = await self._aget_map_prompt(state, runtime)
            response = cast(
                "AIMessage", await self.model.ainvoke(prompt, config=config)
            )
            result = response if self.response_format else response.text()
            extraction_result: ExtractionResult = {
                "indexes": state["indexes"],
                "result": result,
            }
            return {"map_results": [extraction_result]}

        return RunnableCallable(
            _map_node,
            _amap_node,
            trace=False,
        )

    def create_reduce_node(self) -> RunnableCallable:
        """Create a LangGraph node that combines individual results using the LLM."""

        def _reduce_node(
            state: MapReduceState, runtime: Runtime[ContextT], config: RunnableConfig
        ) -> MapReduceNodeUpdate:
            prompt = self._get_reduce_prompt(state, runtime)
            response = cast("AIMessage", self.model.invoke(prompt, config=config))
            result = response if self.response_format else response.text()
            return {"result": result}

        async def _areduce_node(
            state: MapReduceState,
            runtime: Runtime[ContextT],
            config: RunnableConfig,
        ) -> MapReduceNodeUpdate:
            prompt = await self._aget_reduce_prompt(state, runtime)
            response = cast(
                "AIMessage", await self.model.ainvoke(prompt, config=config)
            )
            result = response if self.response_format else response.text()
            return {"result": result}

        return RunnableCallable(
            _reduce_node,
            _areduce_node,
            trace=False,
        )

    def continue_to_map(self, state: MapReduceState) -> list[Send]:
        """Generate Send objects for parallel map operations."""
        return [
            Send("map_process", {"documents": [doc], "indexes": [i]})
            for i, doc in enumerate(state["documents"])
        ]

    def build(
        self,
    ) -> StateGraph[MapReduceState, ContextT, InputSchema, OutputSchema]:
        """Build and compile the LangGraph for map-reduce summarization."""
        builder = StateGraph(
            MapReduceState,
            context_schema=self.context_schema,
            input_schema=InputSchema,
            output_schema=OutputSchema,
        )

        builder.add_node("map_process", self.create_map_node())

        builder.add_edge(START, "continue_to_map")
        # Add-conditional edges doesn't explicitly type Send
        builder.add_conditional_edges(
            "continue_to_map",
            self.continue_to_map,  # type: ignore[arg-type]
            ["map_process"],
        )

        if self.reduce is None:
            builder.add_edge("map_process", END)
        elif self.reduce == "default_reducer":
            builder.add_node("reduce_process", self.create_reduce_node())
            builder.add_edge("map_process", "reduce_process")
            builder.add_edge("reduce_process", END)
        else:
            reduce_node = cast("StateNode", self.reduce)
            # The type is ignored here. Requires parameterizing with generics.
            builder.add_node("reduce_process", reduce_node)  # type: ignore[arg-type]
            builder.add_edge("map_process", "reduce_process")
            builder.add_edge("reduce_process", END)

        return builder


def create_map_reduce_chain(
    model: Union[BaseChatModel, str],
    *,
    map_prompt: Union[
        str,
        None,
        Callable[[MapState, Runtime[ContextT]], list[MessageLikeRepresentation]],
    ] = None,
    reduce_prompt: Union[
        str,
        None,
        Callable[[MapReduceState, Runtime[ContextT]], list[MessageLikeRepresentation]],
    ] = None,
    reduce: Union[
        Literal["default_reducer"],
        None,
        StateNode,
    ] = "default_reducer",
    context_schema: type[ContextT] | None = None,
    response_format: Optional[type[BaseModel]] = None,
) -> StateGraph[MapReduceState, ContextT, InputSchema, OutputSchema]:
    """Create a map-reduce document extraction chain.

    This implementation uses a language model to extract information from documents
    through a flexible approach that efficiently handles large document collections
    by processing documents in parallel.

    **Processing Flow:**
    1. **Map Phase**: Each document is independently processed by the LLM
       using the map_prompt to extract relevant information and generate
       individual results.
    2. **Reduce Phase (Optional)**: Individual extraction results can
       optionally be combined using:
       - The default LLM-based reducer with reduce_prompt (default behavior)
       - A custom reducer function (can be non-LLM based)
       - Skipped entirely by setting reduce=None
    3. **Output**: Returns the individual map results and optionally the final
       combined result.

    Example:
        >>> from langchain_anthropic import ChatAnthropic
        >>> from langchain_core.documents import Document
        >>>
        >>> model = ChatAnthropic(
        ...     model="claude-sonnet-4-20250514",
        ...     temperature=0,
        ...     max_tokens=62_000,
        ...     timeout=None,
        ...     max_retries=2,
        ... )
        >>> builder = create_map_reduce_chain(model)
        >>> chain = builder.compile()
        >>> docs = [
        ...     Document(page_content="First document content..."),
        ...     Document(page_content="Second document content..."),
        ...     Document(page_content="Third document content..."),
        ... ]
        >>> result = chain.invoke({"documents": docs})
        >>> print(result["result"])

    Example with string model:
        >>> builder = create_map_reduce_chain("anthropic:claude-sonnet-4-20250514")
        >>> chain = builder.compile()
        >>> result = chain.invoke({"documents": docs})
        >>> print(result["result"])

    Example with structured output:
        ```python
        from pydantic import BaseModel

        class ExtractionModel(BaseModel):
            title: str
            key_points: list[str]
            conclusion: str

        builder = create_map_reduce_chain(
            model,
            response_format=ExtractionModel
        )
        chain = builder.compile()
        result = chain.invoke({"documents": docs})
        print(result["result"].title)  # Access structured fields
        ```

    Example skipping the reduce phase:
        ```python
        # Only perform map phase, skip combining results
        builder = create_map_reduce_chain(model, reduce=None)
        chain = builder.compile()
        result = chain.invoke({"documents": docs})
        # result["result"] will be None, only map_results are available
        for map_result in result["map_results"]:
            print(f"Document {map_result['indexes'][0]}: {map_result['result']}")
        ```

    Example with custom reducer:
        ```python
        def custom_aggregator(state, runtime):
            # Custom non-LLM based reduction logic
            map_results = state["map_results"]
            combined_text = " | ".join(r["result"] for r in map_results)
            word_count = len(combined_text.split())
            return {
                "result": f"Combined {len(map_results)} results with "
                          f"{word_count} total words"
            }

        builder = create_map_reduce_chain(model, reduce=custom_aggregator)
        chain = builder.compile()
        result = chain.invoke({"documents": docs})
        print(result["result"])  # Custom aggregated result
        ```

    Args:
        model: The language model either a chat model instance
              (e.g., `ChatAnthropic()`) or string identifier
              (e.g., `"anthropic:claude-sonnet-4-20250514"`)
        map_prompt: Prompt for individual document processing. Can be:
            - str: A system message string
            - None: Use default system message
            - Callable: A function that takes (state, runtime) and returns messages
        reduce_prompt: Prompt for combining results. Can be:
            - str: A system message string
            - None: Use default system message
            - Callable: A function that takes (state, runtime) and returns messages
        reduce: Controls the reduce behavior. Can be:
            - "default_reducer": Use the default LLM-based reduce step
            - None: Skip the reduce step entirely
            - Callable: Custom reduce function (sync or async)
        context_schema: Optional context schema for the LangGraph runtime.
        response_format: Optional pydantic BaseModel for structured output.

    Returns:
        A LangGraph that can be invoked with documents to get map-reduce
        extraction results.

    Note:
        This implementation is well-suited for large document collections as it
        processes documents in parallel during the map phase. The Send API enables
        efficient parallelization while maintaining clean state management.
    """
    extractor = _MapReduceExtractor(
        model,
        map_prompt=map_prompt,
        reduce_prompt=reduce_prompt,
        reduce=reduce,
        context_schema=context_schema,
        response_format=response_format,
    )
    return extractor.build()


__all__ = ["create_map_reduce_chain"]
