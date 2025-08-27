"""Stuff documents chain for processing documents by putting them all in context."""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    Optional,
    Union,
    cast,
)

# Used not only for type checking, but is fetched at runtime by Pydantic.
from langchain_core.documents import Document as Document  # noqa: TC002
from langgraph.graph import START, StateGraph
from typing_extensions import NotRequired, TypedDict

from langchain._internal._documents import format_document_xml
from langchain._internal._prompts import aresolve_prompt, resolve_prompt
from langchain._internal._typing import ContextT
from langchain._internal._utils import RunnableCallable
from langchain.chat_models import init_chat_model

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel

    # Used for type checking, but IDEs may not recognize it inside the cast.
    from langchain_core.messages import AIMessage as AIMessage
    from langchain_core.messages import MessageLikeRepresentation
    from langchain_core.runnables import RunnableConfig
    from langgraph.runtime import Runtime
    from pydantic import BaseModel


# Default system prompts
DEFAULT_INIT_PROMPT = (
    "You are a helpful assistant that summarizes text. "
    "Please provide a concise summary of the documents "
    "provided by the user."
)

DEFAULT_STRUCTURED_INIT_PROMPT = (
    "You are a helpful assistant that extracts structured information from documents. "
    "Use the provided content and optional question to generate your output, formatted "
    "according to the predefined schema."
)

DEFAULT_REFINE_PROMPT = (
    "You are a helpful assistant that refines summaries. "
    "Given an existing summary and new context, produce a refined summary "
    "that incorporates the new information while maintaining conciseness."
)

DEFAULT_STRUCTURED_REFINE_PROMPT = (
    "You are a helpful assistant refining structured information extracted "
    "from documents. "
    "You are given a previous result and new document context. "
    "Update the output to reflect the new context, staying consistent with "
    "the expected schema."
)


def _format_documents_content(documents: list[Document]) -> str:
    """Format documents into content string.

    Args:
        documents: List of documents to format.

    Returns:
        Formatted document content string.
    """
    return "\n\n".join(format_document_xml(doc) for doc in documents)


class ExtractionState(TypedDict):
    """State for extraction chain.

    This state tracks the extraction process where documents
    are processed in batch, with the result being refined if needed.
    """

    documents: list[Document]
    """List of documents to process."""
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


class _Extractor(Generic[ContextT]):
    """Stuff documents chain implementation.

    This chain works by putting all the documents in the batch into the context
    window of the language model. It processes all documents together in a single
    request for extracting information or summaries. Can refine existing results
    when provided.

    Important: This chain does not attempt to control for the size of the context
    window of the LLM. Ensure your documents fit within the model's context limits.
    """

    def __init__(
        self,
        model: Union[BaseChatModel, str],
        *,
        prompt: Union[
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
        context_schema: type[ContextT] | None = None,
        response_format: Optional[type[BaseModel]] = None,
    ) -> None:
        """Initialize the Extractor.

        Args:
            model: The language model either a chat model instance
                  (e.g., `ChatAnthropic()`) or string identifier
                  (e.g., `"anthropic:claude-sonnet-4-20250514"`)
            prompt: Prompt for initial processing. Can be:
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
            model = init_chat_model(model)

        self.model = (
            model.with_structured_output(response_format) if response_format else model
        )
        self.initial_prompt = prompt
        self.refine_prompt = refine_prompt
        self.context_schema = context_schema

    def _get_initial_prompt(
        self, state: ExtractionState, runtime: Runtime[ContextT]
    ) -> list[MessageLikeRepresentation]:
        """Generate the initial extraction prompt."""
        user_content = _format_documents_content(state["documents"])

        # Choose default prompt based on structured output format
        default_prompt = (
            DEFAULT_STRUCTURED_INIT_PROMPT
            if self.response_format
            else DEFAULT_INIT_PROMPT
        )

        return resolve_prompt(
            self.initial_prompt,
            state,
            runtime,
            user_content,
            default_prompt,
        )

    async def _aget_initial_prompt(
        self, state: ExtractionState, runtime: Runtime[ContextT]
    ) -> list[MessageLikeRepresentation]:
        """Generate the initial extraction prompt (async version)."""
        user_content = _format_documents_content(state["documents"])

        # Choose default prompt based on structured output format
        default_prompt = (
            DEFAULT_STRUCTURED_INIT_PROMPT
            if self.response_format
            else DEFAULT_INIT_PROMPT
        )

        return await aresolve_prompt(
            self.initial_prompt,
            state,
            runtime,
            user_content,
            default_prompt,
        )

    def _get_refine_prompt(
        self, state: ExtractionState, runtime: Runtime[ContextT]
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

        new_context = _format_documents_content(state["documents"])

        user_content = (
            f"Previous result:\n{state['result']}\n\n"
            f"New context:\n{new_context}\n\n"
            f"Please provide a refined result."
        )

        # Choose default prompt based on structured output format
        default_prompt = (
            DEFAULT_STRUCTURED_REFINE_PROMPT
            if self.response_format
            else DEFAULT_REFINE_PROMPT
        )

        return resolve_prompt(
            self.refine_prompt,
            state,
            runtime,
            user_content,
            default_prompt,
        )

    async def _aget_refine_prompt(
        self, state: ExtractionState, runtime: Runtime[ContextT]
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

        new_context = _format_documents_content(state["documents"])

        user_content = (
            f"Previous result:\n{state['result']}\n\n"
            f"New context:\n{new_context}\n\n"
            f"Please provide a refined result."
        )

        # Choose default prompt based on structured output format
        default_prompt = (
            DEFAULT_STRUCTURED_REFINE_PROMPT
            if self.response_format
            else DEFAULT_REFINE_PROMPT
        )

        return await aresolve_prompt(
            self.refine_prompt,
            state,
            runtime,
            user_content,
            default_prompt,
        )

    def create_document_processor_node(self) -> RunnableCallable:
        """Create the main document processing node.

        The node handles both initial processing and refinement of results.

        Refinement is done by providing the existing result and new context.

        If the workflow is run with a checkpointer enabled, the result will be
        persisted and available for a given thread id.
        """

        def _process_node(
            state: ExtractionState, runtime: Runtime[ContextT], config: RunnableConfig
        ) -> ExtractionNodeUpdate:
            # Handle empty document list
            if not state["documents"]:
                return {}

            # Determine if this is initial processing or refinement
            if "result" not in state or state["result"] == "":
                # Initial processing
                prompt = self._get_initial_prompt(state, runtime)
                response = cast("AIMessage", self.model.invoke(prompt, config=config))
                result = response if self.response_format else response.text
                return {"result": result}
            # Refinement
            prompt = self._get_refine_prompt(state, runtime)
            response = cast("AIMessage", self.model.invoke(prompt, config=config))
            result = response if self.response_format else response.text
            return {"result": result}

        async def _aprocess_node(
            state: ExtractionState,
            runtime: Runtime[ContextT],
            config: RunnableConfig,
        ) -> ExtractionNodeUpdate:
            # Handle empty document list
            if not state["documents"]:
                return {}

            # Determine if this is initial processing or refinement
            if "result" not in state or state["result"] == "":
                # Initial processing
                prompt = await self._aget_initial_prompt(state, runtime)
                response = cast(
                    "AIMessage", await self.model.ainvoke(prompt, config=config)
                )
                result = response if self.response_format else response.text
                return {"result": result}
            # Refinement
            prompt = await self._aget_refine_prompt(state, runtime)
            response = cast(
                "AIMessage", await self.model.ainvoke(prompt, config=config)
            )
            result = response if self.response_format else response.text
            return {"result": result}

        return RunnableCallable(
            _process_node,
            _aprocess_node,
            trace=False,
        )

    def build(
        self,
    ) -> StateGraph[ExtractionState, ContextT, InputSchema, OutputSchema]:
        """Build and compile the LangGraph for batch document extraction."""
        builder = StateGraph(
            ExtractionState,
            context_schema=self.context_schema,
            input_schema=InputSchema,
            output_schema=OutputSchema,
        )
        builder.add_edge(START, "process")
        builder.add_node("process", self.create_document_processor_node())
        return builder


def create_stuff_documents_chain(
    model: Union[BaseChatModel, str],
    *,
    prompt: Union[
        str,
        None,
        Callable[[ExtractionState, Runtime[ContextT]], list[MessageLikeRepresentation]],
    ] = None,
    refine_prompt: Union[
        str,
        None,
        Callable[[ExtractionState, Runtime[ContextT]], list[MessageLikeRepresentation]],
    ] = None,
    context_schema: type[ContextT] | None = None,
    response_format: Optional[type[BaseModel]] = None,
) -> StateGraph[ExtractionState, ContextT, InputSchema, OutputSchema]:
    """Create a stuff documents chain for processing documents.

    This chain works by putting all the documents in the batch into the context
    window of the language model. It processes all documents together in a single
    request for extracting information or summaries. Can refine existing results
    when provided. The default prompts are optimized for summarization tasks, but
    can be customized for other extraction tasks via the prompt parameters or
    response_format.

    Strategy:
    1. Put all documents into the context window
    2. Process all documents together in a single request
    3. If an existing result is provided, refine it with all documents at once
    4. Return the result

    Important:
        This chain does not attempt to control for the size of the context
        window of the LLM. Ensure your documents fit within the model's context limits.

    Example:
        ```python
        from langchain.chat_models import init_chat_model
        from langchain_core.documents import Document

        model = init_chat_model("anthropic:claude-sonnet-4-20250514")
        builder = create_stuff_documents_chain(model)
        chain = builder.compile()
        docs = [
            Document(page_content="First document content..."),
            Document(page_content="Second document content..."),
            Document(page_content="Third document content..."),
        ]
        result = chain.invoke({"documents": docs})
        print(result["result"])

        # Structured summary/extraction by passing a schema
        from pydantic import BaseModel

        class Summary(BaseModel):
            title: str
            key_points: list[str]

        builder = create_stuff_documents_chain(model, response_format=Summary)
        chain = builder.compile()
        result = chain.invoke({"documents": docs})
        print(result["result"].title)  # Access structured fields
        ```

    Args:
        model: The language model for document processing.
        prompt: Prompt for initial processing. Can be:
            - str: A system message string
            - None: Use default system message
            - Callable: A function that takes (state, runtime) and returns messages
        refine_prompt: Prompt for refinement steps. Can be:
            - str: A system message string
            - None: Use default system message
            - Callable: A function that takes (state, runtime) and returns messages
        context_schema: Optional context schema for the LangGraph runtime.
        response_format: Optional pydantic BaseModel for structured output.

    Returns:
        A LangGraph that can be invoked with documents to extract information.

    .. note::
        This is a "stuff" documents chain that puts all documents into the context
        window and processes them together. It supports refining existing results.
        Default prompts are optimized for summarization but can be customized for
        other tasks. Important: Does not control for context window size.
    """
    extractor = _Extractor(
        model,
        prompt=prompt,
        refine_prompt=refine_prompt,
        context_schema=context_schema,
        response_format=response_format,
    )
    return extractor.build()


__all__ = ["create_stuff_documents_chain"]
