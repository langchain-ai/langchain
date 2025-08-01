"""Map-Reduce Extraction Implementation using LangGraph Send API."""

from __future__ import annotations

import operator
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Callable,
    Generic,
    NotRequired,
    Optional,
    Type,
    Union,
    cast,
)

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from langgraph.typing import ContextT
from typing_extensions import TypedDict

from langchain._internal.documents import format_document_xml
from langchain._internal.prompts import aresolve_prompt, resolve_prompt
from langchain._internal.utils import RunnableCallable
from langchain.chat_models import init_chat_model

# Needs to be in global scope as the type annotation is used at runtime
from langchain_core.documents import Document as Document

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage, MessageLikeRepresentation
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
    results: Annotated[list[ExtractionResult], operator.add]
    """Individual results from the map phase."""
    result: NotRequired[Any]
    """Final combined result from the reduce phase."""


class MapState(TypedDict):
    """State for individual map operations."""

    document: Document
    """Single document to process in map phase."""
    document_index: int
    """Index of the document in the original list."""


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
    results: list[ExtractionResult]
    """List of individual extraction results from the map phase."""

    result: Any
    """Final combined result from all documents."""


class MapReduceNodeUpdate(TypedDict):
    """Update returned by map-reduce nodes."""

    results: NotRequired[list[ExtractionResult]]
    """Updated results after map phase."""
    result: NotRequired[Any]
    """Final result after reduce phase."""


class _MapReduceExtractor(Generic[ContextT]):
    """Map-reduce extraction implementation using LangGraph Send API.

    This implementation processes documents in parallel during the map phase,
    generating individual results, then combines them in a reduce phase.
    This approach is efficient for large document collections and leverages
    parallelization capabilities.
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
        context_schema: ContextT = None,
        response_format: Optional[Type[BaseModel]] = None,
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
            context_schema: Optional context schema for the LangGraph runtime.
            response_format: Optional pydantic BaseModel for structured output.
        """
        self.response_format = response_format

        if isinstance(model, str):
            model = cast("BaseChatModel", init_chat_model(model))

        self.model = (
            model.with_structured_output(response_format) if response_format else model
        )
        self.map_prompt = map_prompt
        self.reduce_prompt = reduce_prompt
        self.context_schema = context_schema or {}

    def _get_map_prompt(
        self, state: MapState, runtime: Runtime
    ) -> list[MessageLikeRepresentation]:
        """Generate the map phase prompt for individual document processing."""
        document = state["document"]
        user_content = format_document_xml(document)
        default_system = (
            "You are a helpful assistant that processes documents. "
            "Please process the following document and provide a result."
        )

        return resolve_prompt(
            self.map_prompt,
            state,
            runtime,
            user_content,
            default_system,
        )

    async def _aget_map_prompt(
        self, state: MapState, runtime: Runtime
    ) -> list[MessageLikeRepresentation]:
        """Generate the map phase prompt (async version)."""
        document = state["document"]
        user_content = format_document_xml(document)
        default_system = (
            "You are a helpful assistant that processes documents. "
            "Please process the following document and provide a result."
        )

        return await aresolve_prompt(
            self.map_prompt,
            state,
            runtime,
            user_content,
            default_system,
        )

    def _get_reduce_prompt(
        self, state: MapReduceState, runtime: Runtime
    ) -> list[MessageLikeRepresentation]:
        """Generate the reduce phase prompt for combining results."""
        results = state.get("results", [])
        if not results:
            msg = (
                "Internal programming error: Results must exist when reducing. "
                "This indicates that the reduce node was reached without "
                "first processing the map nodes, which violates "
                "the expected graph execution order."
            )
            raise AssertionError(msg)

        results_text = "\n\n".join(
            f"Result {i + 1} (from documents {', '.join(map(str, result['indexes']))}):\n{result['result']}" 
            for i, result in enumerate(results)
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
        self, state: MapReduceState, runtime: Runtime
    ) -> list[MessageLikeRepresentation]:
        """Generate the reduce phase prompt (async version)."""
        results = state.get("results", [])
        if not results:
            msg = (
                "Internal programming error: Results must exist when reducing. "
                "This indicates that the reduce node was reached without "
                "first processing the map nodes, which violates "
                "the expected graph execution order."
            )
            raise AssertionError(msg)

        results_text = "\n\n".join(
            f"Result {i + 1} (from documents {', '.join(map(str, result['indexes']))}):\n{result['result']}" 
            for i, result in enumerate(results)
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
        """Create a node that generates individual document results."""

        def _map_node(
            state: MapState, runtime: Runtime, config: RunnableConfig
        ) -> dict[str, list[ExtractionResult]]:
            prompt = self._get_map_prompt(state, runtime)
            response = cast("AIMessage", self.model.invoke(prompt, config=config))
            result = response if self.response_format else response.text()
            extraction_result: ExtractionResult = {
                "indexes": [state["document_index"]],
                "result": result
            }
            return {"results": [extraction_result]}

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
                "indexes": [state["document_index"]],
                "result": result
            }
            return {"results": [extraction_result]}

        return RunnableCallable(
            _map_node,
            _amap_node,
            trace=False,
        )

    def create_reduce_node(self) -> RunnableCallable:
        """Create a node that combines individual results."""

        def _reduce_node(
            state: MapReduceState, runtime: Runtime, config: RunnableConfig
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
            Send("map_process", {"document": doc, "document_index": i})
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

        builder.add_node("continue_to_map", lambda state: {})
        builder.add_node("map_process", self.create_map_node())
        builder.add_node("reduce_process", self.create_reduce_node())

        builder.add_edge(START, "continue_to_map")
        builder.add_conditional_edges(
            "continue_to_map", self.continue_to_map, ["map_process"]
        )
        builder.add_edge("map_process", "reduce_process")
        builder.add_edge("reduce_process", END)

        return builder


def create_map_reduce_extractor(
    model: Union[BaseChatModel, str],
    *,
    map_prompt: Union[
        str,
        None,
        Callable[[MapState, Runtime], list[MessageLikeRepresentation]],
    ] = None,
    reduce_prompt: Union[
        str,
        None,
        Callable[[MapReduceState, Runtime], list[MessageLikeRepresentation]],
    ] = None,
    context_schema: ContextT = None,
    response_format: Optional[Type[BaseModel]] = None,
) -> StateGraph[MapReduceState, ContextT, InputSchema, OutputSchema]:
    """Create a map-reduce document extraction chain.

    This implementation processes documents in parallel during the map phase,
    generating individual results, then combines them in a reduce phase.
    This approach efficiently handles large document collections by leveraging
    parallelization.

    The process works as follows:
    1. Map phase: Process each document in parallel to generate individual results
    2. Reduce phase: Combine all individual results into a final comprehensive result
    3. Return the final combined result

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
        >>> builder = create_map_reduce_extractor(model)
        >>> chain = builder.compile()
        >>> docs = [
        ...     Document(page_content="First document content..."),
        ...     Document(page_content="Second document content..."),
        ...     Document(page_content="Third document content..."),
        ... ]
        >>> result = chain.invoke({"documents": docs})
        >>> print(result["result"])

    Example with string model:
        >>> builder = create_map_reduce_extractor("anthropic:claude-sonnet-4-20250514")
        >>> chain = builder.compile()
        >>> result = chain.invoke({"documents": docs})
        >>> print(result["result"])

    Example with structured output:
        >>> from pydantic import BaseModel
        >>>
        >>> class ExtractionModel(BaseModel):
        ...     title: str
        ...     key_points: list[str]
        ...     conclusion: str
        >>>
        >>> builder = create_map_reduce_extractor(
        ...     model,
        ...     response_format=ExtractionModel
        ... )
        >>> chain = builder.compile()
        >>> result = chain.invoke({"documents": docs})
        >>> print(result["result"].title)  # Access structured fields

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
        context_schema: Optional context schema for the LangGraph runtime.
        response_format: Optional pydantic BaseModel for structured output.

    Returns:
        A LangGraph that can be invoked with documents to get map-reduce extraction results.

    Note:
        This implementation is well-suited for large document collections as it
        processes documents in parallel during the map phase. The Send API enables
        efficient parallelization while maintaining clean state management.
    """
    extractor = _MapReduceExtractor(
        model,
        map_prompt=map_prompt,
        reduce_prompt=reduce_prompt,
        context_schema=context_schema,
        response_format=response_format,
    )
    return extractor.build()


__all__ = ["create_map_reduce_extractor"]
