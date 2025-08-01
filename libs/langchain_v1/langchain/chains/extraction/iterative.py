"""General-purpose document extractor with iterative refinement."""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    Literal,
    NotRequired,
    Optional,
    Type,
    Union,
    cast,
)

from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph
from langgraph.typing import ContextT
from typing_extensions import TypedDict

from langchain._internal.documents import format_document_xml
from langchain._internal.prompts import aresolve_prompt, resolve_prompt
from langchain._internal.utils import RunnableCallable
from langchain.chat_models import init_chat_model

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import MessageLikeRepresentation
    from langchain_core.runnables import RunnableConfig
    from langgraph.runtime import Runtime
    from pydantic import BaseModel


# Default system prompts
DEFAULT_INIT_SEQUENTIAL_PROMPT = (
    "You are a helpful assistant that summarizes text. "
    "Please provide a concise summary of the following document."
)

DEFAULT_INIT_BATCH_PROMPT = (
    "You are a helpful assistant that summarizes text. "
    "Please provide a concise summary of the documents "
    "provided by the user."
)

DEFAULT_REFINE_PROMPT = (
    "You are a helpful assistant that refines summaries. "
    "Given an existing summary and new context, produce a refined summary "
    "that incorporates the new information while maintaining conciseness."
)


class ExtractionState(TypedDict):
    """State for extraction chain.

    This state tracks the extraction process where documents
    are processed sequentially, with the result being refined at each step.
    """

    documents: list[Document]
    """List of documents to process."""
    index: NotRequired[int]
    """Current document index being processed."""
    result: NotRequired[Any]
    """Current result, refined with each document."""


class InputSchema(TypedDict):
    """Input schema for the extraction chain.

    Defines the expected input format when invoking the extraction chain.
    """

    documents: list[Document]
    """List of documents to process."""
    result: NotRequired[Any]
    """Existing result to refine (optional)."""


class OutputSchema(TypedDict):
    """Output schema for the extraction chain.

    Defines the format of the final result returned by the chain.
    """

    result: Any
    """Result from processing the documents."""


class ExtractionNodeUpdate(TypedDict):
    """Update returned by processing nodes."""

    result: NotRequired[Any]
    """Updated result after processing a document."""
    index: NotRequired[int]
    """Updated document index."""


class _Extractor(Generic[ContextT]):
    """General-purpose document extractor supporting multiple strategies.

    This implementation supports both sequential (iterative) and batch (inline)
    processing strategies for extracting information from documents. It can extract
    text summaries or structured data, and can refine existing results when provided.
    Sequential mode processes documents one by one, while batch mode processes
    all documents together in a single request.
    """

    def __init__(
        self,
        model: Union[BaseChatModel, str],
        *,
        strategy: Literal["sequential", "batch"] = "sequential",
        initial_prompt: Union[
            str,
            None,
            Callable[
                [ExtractionState, Runtime[ContextT]],
                list[MessageLikeRepresentation],
            ],
        ] = None,
        refine_prompt: Union[
            str,
            None,
            Callable[
                [ExtractionState, Runtime[ContextT]],
                list[MessageLikeRepresentation],
            ],
        ] = None,
        context_schema: ContextT = None,
        response_format: Optional[Type[BaseModel]] = None,
    ) -> None:
        """Initialize the Extractor.

        Args:
            model: The language model either a chat model instance
                  (e.g., `ChatAnthropic()`) or string identifier
                  (e.g., `"anthropic:claude-sonnet-4-20250514"`)
            strategy: Processing strategy - "sequential" for iterative processing,
                "batch" for processing all documents at once.
            initial_prompt: Prompt for initial processing. Can be:
                - str: A system message string
                - None: Use default system message
                - Callable: A function that takes (state, runtime) and returns messages
            refine_prompt: Prompt for refinement steps. Can be:
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
        if strategy not in ["sequential", "batch"]:
            msg = "Invalid strategy. Must be one of: 'sequential', 'batch'."
            raise ValueError(msg)
        self.strategy = strategy
        self.initial_prompt = initial_prompt
        self.refine_prompt = refine_prompt
        self.context_schema = context_schema or {}

    def _format_documents_content(
        self, state: ExtractionState, index: int | None = None
    ) -> str:
        """Format documents into content string based on strategy.

        Args:
            state: Current state containing documents.
            index: Document index for sequential strategy. If None, uses all documents.

        Returns:
            Formatted document content string.
        """

        if self.strategy == "batch":
            return "\n\n".join(
                format_document_xml(doc) for doc in state["documents"]
            )
        # Sequential strategy - use specific document by index
        if index is None:
            index = 0  # Default to first document for initial prompt
        return format_document_xml(state["documents"][index])

    def _get_initial_prompt(
        self, state: ExtractionState, runtime: Runtime[ContextT]
    ) -> list[MessageLikeRepresentation]:
        """Generate the initial extraction prompt."""
        user_content = self._format_documents_content(state)

        if self.strategy == "batch":
            default_system_content = DEFAULT_INIT_BATCH_PROMPT
        else:
            default_system_content = DEFAULT_INIT_SEQUENTIAL_PROMPT

        return resolve_prompt(
            self.initial_prompt,
            state,
            runtime,
            user_content,
            default_system_content,
        )

    async def _aget_initial_prompt(
        self, state: ExtractionState, runtime: Runtime[ContextT]
    ) -> list[MessageLikeRepresentation]:
        """Generate the initial extraction prompt (async version)."""
        user_content = self._format_documents_content(state)

        if self.strategy == "batch":
            default_system_content = DEFAULT_INIT_BATCH_PROMPT
        else:
            default_system_content = DEFAULT_INIT_SEQUENTIAL_PROMPT

        return await aresolve_prompt(
            self.initial_prompt,
            state,
            runtime,
            user_content,
            default_system_content,
        )

    def _get_refine_prompt(
        self, state: ExtractionState, runtime: Runtime
    ) -> list[MessageLikeRepresentation]:
        """Generate the refinement prompt."""
        # Result should be guaranteed to exist at refinement stage
        if "result" not in state or state["result"] == "":
            msg = (
                "Internal programming error: Result must exist when refining. "
                "This indicates that the refinement node was reached without "
                "first processing the initial result node, which violates "
                "the expected graph execution order."
            )
            raise AssertionError(msg)

        # Get document content based on strategy
        if self.strategy == "batch":
            new_context = self._format_documents_content(state)
        else:
            index = state.get("index", 1)
            new_context = self._format_documents_content(state, index)

        user_content = (
            f"Previous result:\n{state['result']}\n\n"
            f"New context:\n{new_context}\n\n"
            f"Please provide a refined result."
        )

        return resolve_prompt(
            self.refine_prompt,
            state,
            runtime,
            user_content,
            DEFAULT_REFINE_PROMPT,
        )

    async def _aget_refine_prompt(
        self, state: ExtractionState, runtime: Runtime
    ) -> list[MessageLikeRepresentation]:
        """Generate the refinement prompt (async version)."""
        # Result should be guaranteed to exist at refinement stage
        if "result" not in state or state["result"] == "":
            msg = (
                "Internal programming error: Result must exist when refining. "
                "This indicates that the refinement node was reached without "
                "first processing the initial result node, which violates "
                "the expected graph execution order."
            )
            raise AssertionError(msg)

        # Get document content based on strategy
        if self.strategy == "batch":
            new_context = self._format_documents_content(state)
        else:
            index = state.get("index", 1)
            new_context = self._format_documents_content(state, index)

        user_content = (
            f"Previous result:\n{state['result']}\n\n"
            f"New context:\n{new_context}\n\n"
            f"Please provide a refined result."
        )

        return await aresolve_prompt(
            self.refine_prompt,
            state,
            runtime,
            user_content,
            DEFAULT_REFINE_PROMPT,
        )

    def init_node(self, state: ExtractionState) -> ExtractionNodeUpdate:
        """Initialize the process by setting index based on strategy."""
        if self.strategy == "batch":
            return {"index": -1}
        return {"index": 0}

    def create_process_node(self) -> RunnableCallable:
        """Create a unified node that handles both initial extraction and refinement."""

        def _process_node(
            state: ExtractionState, runtime: Runtime, config: RunnableConfig
        ) -> ExtractionNodeUpdate:
            # Handle empty document list
            if not state["documents"]:
                return {"index": 0}

            # Determine if this is initial processing or refinement
            if "result" not in state or state["result"] == "":
                # Initial processing
                prompt = self._get_initial_prompt(state, runtime)
                response = cast("AIMessage", self.model.invoke(prompt, config=config))
                # For batch strategy, we process all documents at once
                final_index = len(state["documents"]) if self.strategy == "batch" else 1
                result = response if self.response_format else response.text()
                return {"result": result, "index": final_index}
            # Refinement
            prompt = self._get_refine_prompt(state, runtime)
            response = cast("AIMessage", self.model.invoke(prompt, config=config))
            result = response if self.response_format else response.text()
            if self.strategy == "batch":
                # For batch, refinement processes all documents at once
                return {"result": result, "index": len(state["documents"])}
            # For sequential, increment index
            index = state.get("index", 1)
            return {"result": result, "index": index + 1}

        async def _aprocess_node(
            state: ExtractionState,
            runtime: Runtime[ContextT],
            config: RunnableConfig,
        ) -> ExtractionNodeUpdate:
            # Handle empty document list
            if not state["documents"]:
                return {"index": 0}

            # Determine if this is initial processing or refinement
            if "result" not in state or state["result"] == "":
                # Initial processing
                prompt = await self._aget_initial_prompt(state, runtime)
                response = cast(
                    "AIMessage", await self.model.ainvoke(prompt, config=config)
                )
                # For batch strategy, we process all documents at once
                final_index = len(state["documents"]) if self.strategy == "batch" else 1
                if self.response_format:
                    result = response
                else:
                    result = response.text()
                return {"result": result, "index": final_index}
            # Refinement
            prompt = await self._aget_refine_prompt(state, runtime)
            response = cast(
                "AIMessage", await self.model.ainvoke(prompt, config=config)
            )
            if self.response_format:
                result = response
            else:
                result = response.text()
            if self.strategy == "batch":
                # For batch, refinement processes all documents at once
                return {"result": result, "index": len(state["documents"])}
            # For sequential, increment index
            index = state.get("index", 1)
            return {"result": result, "index": index + 1}

        return RunnableCallable(
            _process_node,
            _aprocess_node,
            trace=False,
        )

    def should_continue(self, state: ExtractionState) -> Literal["process", "__end__"]:
        """Determine whether to continue processing documents or end the process."""
        documents = state["documents"]

        # If no documents, end immediately
        if not documents:
            return END

        index = state.get("index", 0)

        # Continue if we haven't processed all documents
        if index < len(documents):
            return "process"

        return END

    def build(
        self,
    ) -> StateGraph[ExtractionState, ContextT, InputSchema, OutputSchema]:
        """Build and compile the LangGraph for iterative refinement extraction."""
        builder = StateGraph(
            ExtractionState,
            context_schema=self.context_schema,
            input_schema=InputSchema,
            output_schema=OutputSchema,
        )

        builder.add_node("init", self.init_node)
        builder.add_node("process", self.create_process_node())

        builder.add_edge(START, "init")
        builder.add_edge("init", "process")
        builder.add_conditional_edges("process", self.should_continue)

        return builder


def create_iterative_extractor(
    model: Union[BaseChatModel, str],
    *,
    strategy: Literal["sequential", "batch"] = "batch",
    initial_prompt: Union[
        str,
        None,
        Callable[[ExtractionState, Runtime], list[MessageLikeRepresentation]],
    ] = None,
    refine_prompt: Union[
        str,
        None,
        Callable[[ExtractionState, Runtime], list[MessageLikeRepresentation]],
    ] = None,
    context_schema: ContextT = None,
    response_format: Optional[Type[BaseModel]] = None,
) -> StateGraph[ExtractionState, ContextT, InputSchema, OutputSchema]:
    """Create a general-purpose document extractor with multiple strategies.

    This implementation supports both sequential (iterative) and batch (inline)
    processing strategies for extracting information from documents. It can extract
    text summaries or structured data, and can refine existing results when provided.
    The default prompts are optimized for summarization tasks, but can be customized
    for other extraction tasks via the prompt parameters or response_format.

    Sequential strategy:
    1. Generate an initial result from the first document
    2. For each remaining document, refine the existing result by incorporating
       the new document's content
    3. Return the final refined result

    Batch strategy:
    1. Process all documents together in a single request
    2. If an existing result is provided, refine it with all documents at once
    3. Return the result

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
        >>> builder = create_extractor(model, strategy="sequential")
        >>> chain = builder.compile()
        >>> docs = [
        ...     Document(page_content="First document content..."),
        ...     Document(page_content="Second document content..."),
        ...     Document(page_content="Third document content..."),
        ... ]
        >>> result = chain.invoke({"documents": docs})
        >>> print(result["result"])
        >>>
        >>> # Batch processing
        >>> builder = create_extractor(model, strategy="batch")
        >>> chain = builder.compile()
        >>> result = chain.invoke({"documents": docs})
        >>> print(result["result"])
        >>>
        >>> # String model example
        >>> builder = create_extractor("anthropic:claude-sonnet-4-20250514", strategy="sequential")
        >>> chain = builder.compile()
        >>> result = chain.invoke({"documents": docs})
        >>> print(result["result"])
        >>>
        >>> # Structured output with Pydantic
        >>> from pydantic import BaseModel
        >>>
        >>> class Summary(BaseModel):
        ...     title: str
        ...     key_points: list[str]
        >>>
        >>> builder = create_extractor(model, response_format=Summary)
        >>> chain = builder.compile()
        >>> result = chain.invoke({"documents": docs})
        >>> print(result["result"].title)  # Access structured fields

    Args:
        model: The language model for document processing.
        strategy: Processing strategy - "sequential" for iterative processing,
            "batch" for processing all documents at once.
        initial_prompt: Prompt for initial processing. Can be:
            - str: A system message string
            - None: Use default system message (optimized for summarization)
            - Callable: A function that takes (state, runtime) and returns messages
        refine_prompt: Prompt for refinement steps. Can be:
            - str: A system message string
            - None: Use default system message (optimized for summarization)
            - Callable: A function that takes (state, runtime) and returns messages
        context_schema: Optional context schema for the LangGraph runtime.
        response_format: Optional pydantic BaseModel for structured output.

    Returns:
        A LangGraph that can be invoked with documents to extract information.

    Note:
        Sequential strategy is more memory-efficient for large document collections,
        while batch strategy can be faster for smaller collections that fit within
        token limits. Both strategies support refining existing results. Default prompts
        are optimized for summarization but can be customized for other tasks.
    """
    extractor = _Extractor(
        model,
        strategy=strategy,
        initial_prompt=initial_prompt,
        refine_prompt=refine_prompt,
        context_schema=context_schema,
        response_format=response_format,
    )
    return extractor.build()


__all__ = ["create_iterative_extractor"]
