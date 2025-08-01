"""Recursive Summarization Implementation using LangGraph."""

from __future__ import annotations

import operator
from typing import (
    TYPE_CHECKING,
    Annotated,
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

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from langgraph.typing import ContextT
from typing_extensions import TypedDict

from langchain._internal.prompts import aresolve_prompt, resolve_prompt
from langchain._internal.utils import RunnableCallable
from langchain.chat_models import init_chat_model

if TYPE_CHECKING:
    from langchain_core.documents import Document
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage, MessageLikeRepresentation
    from langchain_core.runnables import RunnableConfig
    from langgraph.runtime import Runtime
    from pydantic import BaseModel


# Default system prompts
DEFAULT_MAP_PROMPT = (
    "You are a helpful assistant that summarizes text. "
    "Please provide a concise summary of the following document."
)

DEFAULT_REDUCE_PROMPT = (
    "You are a helpful assistant that combines multiple summaries. "
    "Given several individual summaries, create a single comprehensive "
    "summary that captures the key information from all inputs while "
    "maintaining conciseness and coherence."
)


class RecursiveState(TypedDict):
    """State for recursive summarization chain.

    This state tracks the recursive process where documents are progressively
    compressed through multiple levels of map-reduce operations until reaching
    a stopping criterion.
    """

    documents: list[Document]
    """List of documents to summarize."""
    summaries: NotRequired[Annotated[list[str], operator.add]]
    """Current level summaries being processed."""
    level: NotRequired[int]
    """Current recursion level (0 = original documents)."""
    max_depth: NotRequired[int]
    """Maximum recursion depth allowed."""
    min_docs_per_level: NotRequired[int]
    """Minimum documents per level to continue recursion."""
    final_summary: NotRequired[str]
    """Final summary after all recursion levels."""


class MapState(TypedDict):
    """State for individual map operations."""

    content: str
    """Content to process in map phase (document or summary text)."""
    level: int
    """Current recursion level."""


class InputSchema(TypedDict):
    """Input schema for the recursive summarization chain.

    Defines the expected input format when invoking the summarization chain.
    """

    documents: list[Document]
    """List of documents to summarize."""
    max_depth: NotRequired[int]
    """Maximum recursion depth (default: 10)."""
    min_docs_per_level: NotRequired[int]
    """Minimum documents per level to continue (default: 2)."""


class OutputSchema(TypedDict):
    """Output schema for the recursive summarization chain.

    Defines the format of the final result returned by the chain.
    """

    final_summary: str
    """Final recursive summary of all documents."""
    levels_processed: int
    """Number of recursion levels processed."""


class RecursiveNodeUpdate(TypedDict):
    """Update returned by recursive nodes."""

    summaries: NotRequired[list[str]]
    """Updated summaries after map phase."""
    final_summary: NotRequired[str]
    """Final summary after all levels."""
    level: NotRequired[int]
    """Updated recursion level."""
    levels_processed: NotRequired[int]
    """Total levels processed."""


class _RecursiveSummarizer(Generic[ContextT]):
    """Recursive summarization implementation using progressive map-reduce.

    This implementation applies a recursive strategy to documents by:
    1. Level 0: Summarize each document individually (map phase)
    2. Level 1: Combine summaries in pairs/groups (reduce phase)
    3. Level N: Continue until reaching stopping criteria

    The process achieves progressive compression while maintaining key information.
    """

    def __init__(
        self,
        model: BaseChatModel,
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
                [RecursiveState, Runtime[ContextT]],
                list[MessageLikeRepresentation],
            ],
        ] = None,
        max_depth: int = 10,
        min_docs_per_level: int = 2,
        context_schema: ContextT = None,
        output_parser: Callable[[AIMessage], Any] = None,
    ) -> None:
        """Initialize the RecursiveSummarizer.

        Args:
            model: The chat model to use for summarization.
            map_prompt: Prompt for individual document/summary processing. Can be:
                - str: A system message string
                - None: Use default system message
                - Callable: A function that takes (state, runtime) and returns messages
            reduce_prompt: Prompt for combining summaries. Can be:
                - str: A system message string
                - None: Use default system message
                - Callable: A function that takes (state, runtime) and returns messages
            max_depth: Maximum recursion depth allowed.
            min_docs_per_level: Minimum documents per level to continue recursion.
            context_schema: Optional context schema for the LangGraph runtime.
            output_parser: Optional function to parse model output. Defaults to .text()
        """
        self.model = model
        self.map_prompt = map_prompt
        self.reduce_prompt = reduce_prompt
        self.max_depth = max_depth
        self.min_docs_per_level = min_docs_per_level
        self.context_schema = context_schema or {}
        self.output_parser = output_parser or (lambda msg: msg.text())

    def _get_map_prompt(
        self, state: MapState, runtime: Runtime
    ) -> list[MessageLikeRepresentation]:
        """Generate the map phase prompt for content summarization."""
        content = state["content"]
        level = state["level"]

        if level == 0:
            # Summarizing original documents
            user_content = content
        else:
            # Summarizing existing summaries
            user_content = (
                f"Previous summary:\n{content}\n\nPlease create a more concise summary."
            )

        return resolve_prompt(
            self.map_prompt,
            state,
            runtime,
            user_content,
            DEFAULT_MAP_PROMPT,
        )

    async def _aget_map_prompt(
        self, state: MapState, runtime: Runtime
    ) -> list[MessageLikeRepresentation]:
        """Generate the map phase prompt (async version)."""
        content = state["content"]
        level = state["level"]

        if level == 0:
            # Summarizing original documents
            user_content = content
        else:
            # Summarizing existing summaries
            user_content = (
                f"Previous summary:\n{content}\n\nPlease create a more concise summary."
            )

        return await aresolve_prompt(
            self.map_prompt,
            state,
            runtime,
            user_content,
            DEFAULT_MAP_PROMPT,
        )

    def _get_reduce_prompt(
        self, state: RecursiveState, runtime: Runtime
    ) -> list[MessageLikeRepresentation]:
        """Generate the reduce phase prompt for combining summaries."""
        summaries = state.get("summaries", [])
        if not summaries:
            msg = (
                "Internal programming error: Summaries must exist when reducing. "
                "This indicates that the reduce node was reached without "
                "first processing the map nodes, which violates "
                "the expected graph execution order."
            )
            raise AssertionError(msg)

        # Group summaries for reduction (pairs or small groups)
        group_size = min(4, len(summaries))  # Process up to 4 summaries at once
        summary_groups = [
            summaries[i : i + group_size] for i in range(0, len(summaries), group_size)
        ]

        # For now, just combine the first group (we'll handle multiple groups in iterations)
        current_summaries = summary_groups[0]
        summaries_text = "\n\n".join(
            f"Summary {i + 1}:\n{summary}"
            for i, summary in enumerate(current_summaries)
        )

        user_content = (
            f"Please combine the following summaries into a single, "
            f"comprehensive summary:\n\n{summaries_text}"
        )

        return resolve_prompt(
            self.reduce_prompt,
            state,
            runtime,
            user_content,
            DEFAULT_REDUCE_PROMPT,
        )

    async def _aget_reduce_prompt(
        self, state: RecursiveState, runtime: Runtime
    ) -> list[MessageLikeRepresentation]:
        """Generate the reduce phase prompt (async version)."""
        summaries = state.get("summaries", [])
        if not summaries:
            msg = (
                "Internal programming error: Summaries must exist when reducing. "
                "This indicates that the reduce node was reached without "
                "first processing the map nodes, which violates "
                "the expected graph execution order."
            )
            raise AssertionError(msg)

        # Group summaries for reduction (pairs or small groups)
        group_size = min(4, len(summaries))  # Process up to 4 summaries at once
        summary_groups = [
            summaries[i : i + group_size] for i in range(0, len(summaries), group_size)
        ]

        # For now, just combine the first group (we'll handle multiple groups in iterations)
        current_summaries = summary_groups[0]
        summaries_text = "\n\n".join(
            f"Summary {i + 1}:\n{summary}"
            for i, summary in enumerate(current_summaries)
        )

        user_content = (
            f"Please combine the following summaries into a single, "
            f"comprehensive summary:\n\n{summaries_text}"
        )

        return await aresolve_prompt(
            self.reduce_prompt,
            state,
            runtime,
            user_content,
            DEFAULT_REDUCE_PROMPT,
        )

    def create_map_node(self) -> RunnableCallable:
        """Create a node that processes individual documents or summaries."""

        def _map_node(
            state: MapState, runtime: Runtime, config: RunnableConfig
        ) -> dict[str, list[str]]:
            prompt = self._get_map_prompt(state, runtime)
            response = cast("AIMessage", self.model.invoke(prompt, config=config))
            parsed_output = self.output_parser(response)
            return {"summaries": [str(parsed_output)]}

        async def _amap_node(
            state: MapState,
            runtime: Runtime[ContextT],
            config: RunnableConfig,
        ) -> dict[str, list[str]]:
            prompt = await self._aget_map_prompt(state, runtime)
            response = cast(
                "AIMessage", await self.model.ainvoke(prompt, config=config)
            )
            parsed_output = self.output_parser(response)
            return {"summaries": [str(parsed_output)]}

        return RunnableCallable(
            _map_node,
            _amap_node,
            trace=False,
        )

    def create_reduce_node(self) -> RunnableCallable:
        """Create a node that combines summaries and handles recursion."""

        def _reduce_node(
            state: RecursiveState, runtime: Runtime, config: RunnableConfig
        ) -> RecursiveNodeUpdate:
            summaries = state.get("summaries", [])
            level = state.get("level", 0)

            if len(summaries) <= 1:
                # Single summary left - we're done
                final_summary = summaries[0] if summaries else ""
                return {"final_summary": final_summary, "levels_processed": level}

            # Group summaries for reduction
            group_size = min(4, len(summaries))
            reduced_summaries = []

            for i in range(0, len(summaries), group_size):
                group = summaries[i : i + group_size]
                if len(group) == 1:
                    # Single summary, just add it
                    reduced_summaries.append(group[0])
                else:
                    # Multiple summaries, combine them
                    group_state = {**state, "summaries": group}
                    prompt = self._get_reduce_prompt(group_state, runtime)
                    response = cast(
                        "AIMessage", self.model.invoke(prompt, config=config)
                    )
                    parsed_output = self.output_parser(response)
                    reduced_summaries.append(str(parsed_output))

            return {"summaries": reduced_summaries, "level": level + 1}

        async def _areduce_node(
            state: RecursiveState,
            runtime: Runtime[ContextT],
            config: RunnableConfig,
        ) -> RecursiveNodeUpdate:
            summaries = state.get("summaries", [])
            level = state.get("level", 0)

            if len(summaries) <= 1:
                # Single summary left - we're done
                final_summary = summaries[0] if summaries else ""
                return {"final_summary": final_summary, "levels_processed": level}

            # Group summaries for reduction
            group_size = min(4, len(summaries))
            reduced_summaries = []

            for i in range(0, len(summaries), group_size):
                group = summaries[i : i + group_size]
                if len(group) == 1:
                    # Single summary, just add it
                    reduced_summaries.append(group[0])
                else:
                    # Multiple summaries, combine them
                    group_state = {**state, "summaries": group}
                    prompt = await self._aget_reduce_prompt(group_state, runtime)
                    response = cast(
                        "AIMessage", await self.model.ainvoke(prompt, config=config)
                    )
                    parsed_output = self.output_parser(response)
                    reduced_summaries.append(str(parsed_output))

            return {"summaries": reduced_summaries, "level": level + 1}

        return RunnableCallable(
            _reduce_node,
            _areduce_node,
            trace=False,
        )

    def continue_to_map(self, state: RecursiveState) -> list[Send]:
        """Generate Send objects for parallel map operations."""
        level = state.get("level", 0)

        if level == 0:
            # First level: process original documents
            return [
                Send("map_summarize", {"content": doc.page_content, "level": 0})
                for doc in state["documents"]
            ]
        # Subsequent levels: process existing summaries
        summaries = state.get("summaries", [])
        return [
            Send("map_summarize", {"content": summary, "level": level})
            for summary in summaries
        ]

    def should_continue(
        self, state: RecursiveState
    ) -> Literal["continue_to_map", "reduce_summaries", "__end__"]:
        """Determine the next step in the recursive process."""
        level = state.get("level", 0)
        max_depth = state.get("max_depth", self.max_depth)
        min_docs = state.get("min_docs_per_level", self.min_docs_per_level)
        summaries = state.get("summaries", [])

        # Check if we have a final summary
        if "final_summary" in state:
            return END

        # First iteration - start with documents
        if level == 0 and not summaries:
            return "continue_to_map"

        # Check stopping criteria
        if level >= max_depth:
            return "reduce_summaries"

        if len(summaries) < min_docs:
            return "reduce_summaries"

        # Continue recursion if we have multiple summaries
        if len(summaries) > 1:
            return "continue_to_map"

        # Single summary left - reduce to final
        return "reduce_summaries"

    def initialize_state(self, state: RecursiveState) -> RecursiveState:
        """Initialize state with default values."""
        return {
            **state,
            "level": 0,
            "max_depth": state.get("max_depth", self.max_depth),
            "min_docs_per_level": state.get(
                "min_docs_per_level", self.min_docs_per_level
            ),
        }

    def build(
        self,
    ) -> StateGraph[RecursiveState, ContextT, InputSchema, OutputSchema]:
        """Build and compile the LangGraph for recursive summarization."""
        builder = StateGraph(
            RecursiveState,
            context_schema=self.context_schema,
            input_schema=InputSchema,
            output_schema=OutputSchema,
        )

        builder.add_node("initialize", lambda state: self.initialize_state(state))
        builder.add_node("continue_to_map", lambda state: {})
        builder.add_node("map_summarize", self.create_map_node())
        builder.add_node("reduce_summaries", self.create_reduce_node())

        builder.add_edge(START, "initialize")
        builder.add_conditional_edges("initialize", self.should_continue)
        builder.add_conditional_edges(
            "continue_to_map", self.continue_to_map, ["map_summarize"]
        )
        builder.add_conditional_edges("map_summarize", self.should_continue)
        builder.add_conditional_edges("reduce_summaries", self.should_continue)

        return builder


def create_recursive_extractor(
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
        Callable[[RecursiveState, Runtime], list[MessageLikeRepresentation]],
    ] = None,
    max_depth: int = 10,
    min_docs_per_level: int = 2,
    context_schema: ContextT = None,
    response_format: Optional[Type[BaseModel]] = None,
) -> StateGraph[RecursiveState, ContextT, InputSchema, OutputSchema]:
    """Create a recursive document extraction chain.

    This implementation applies a recursive strategy to documents by progressively
    compressing them through multiple levels of map-reduce operations. The process
    continues until reaching a stopping criterion (max depth, min documents, etc.).

    The recursive process works as follows:
    1. Level 0: Summarize each original document individually (map phase)
    2. Level 1: Group and combine summaries from level 0 (reduce phase)
    3. Level N: Continue grouping and combining until stopping criteria met
    4. Final: Return the ultimate compressed summary

    How It Works:

    Level 0: [Doc1] [Doc2] [Doc3] [Doc4] [Doc5] [Doc6] [Doc7] [Doc8]
               ↓      ↓      ↓      ↓      ↓      ↓      ↓      ↓
    Level 0: [Sum1] [Sum2] [Sum3] [Sum4] [Sum5] [Sum6] [Sum7] [Sum8]
                   ↓                   ↓                  ↓
    Level 1:    [CombSum1]        [CombSum2]        [CombSum3]
                         ↓                       ↓
    Level 2:         [MegaSum1]              [MegaSum2]
                                     ↓
    Level 3:                  [FINAL_SUMMARY]

    This approach achieves better compression ratios than simple map-reduce while
    maintaining key information through the hierarchical structure.

    Example:
        >>> from langchain_openai import ChatOpenAI
        >>> from langchain_core.documents import Document
        >>>
        >>> model = ChatOpenAI(model="gpt-3.5-turbo")
        >>> builder = create_recursive_summarizer(
        ...     model,
        ...     max_depth=5,
        ...     min_docs_per_level=3
        ... )
        >>> chain = builder.compile()
        >>> docs = [
        ...     Document(page_content="Document 1 content..."),
        ...     Document(page_content="Document 2 content..."),
        ...     Document(page_content="Document 3 content..."),
        ...     Document(page_content="Document 4 content..."),
        ...     Document(page_content="Document 5 content..."),
        ... ]
        >>> result = chain.invoke({
        ...     "documents": docs,
        ...     "max_depth": 3,
        ...     "min_docs_per_level": 2
        ... })
        >>> print(result["final_summary"])
        >>> print(f"Processed {result['levels_processed']} levels")

    Example with custom prompts:
        >>> map_prompt = "Summarize this technical document, focusing on key findings:"
        >>> reduce_prompt = "Combine these summaries into a comprehensive overview:"
        >>> builder = create_recursive_summarizer(
        ...     model,
        ...     map_prompt=map_prompt,
        ...     reduce_prompt=reduce_prompt
        ... )

    Args:
        model: The chat model to use for summarization.
        map_prompt: Prompt for individual document/summary processing. Can be:
            - str: A system message string
            - None: Use default system message
            - Callable: A function that takes (state, runtime) and returns messages
        reduce_prompt: Prompt for combining summaries. Can be:
            - str: A system message string
            - None: Use default system message
            - Callable: A function that takes (state, runtime) and returns messages
        max_depth: Maximum recursion depth allowed (default: 10).
        min_docs_per_level: Minimum documents per level to continue recursion (default: 2).
        context_schema: Optional context schema for the LangGraph runtime.
        output_parser: Optional function to parse model output. Defaults to .text()

    Returns:
        A LangGraph that can be invoked with documents to get recursive summaries.

    Note:
        This implementation is well-suited for large document collections that need
        significant compression. The recursive approach often produces more coherent
        final summaries than single-level map-reduce by maintaining hierarchical
        structure throughout the compression process.

        Stopping criteria can be customized via max_depth and min_docs_per_level
        parameters to balance between compression ratio and processing time.
    """
    if isinstance(model, str):
        model = cast("BaseChatModel", init_chat_model(model))
    
    if response_format:
        model = model.with_structured_output(response_format)
        output_parser = lambda msg: msg  # Return structured object directly
    else:
        output_parser = lambda msg: msg.text()
    
    summarizer = _RecursiveSummarizer(
        model,
        map_prompt=map_prompt,
        reduce_prompt=reduce_prompt,
        max_depth=max_depth,
        min_docs_per_level=min_docs_per_level,
        context_schema=context_schema,
        output_parser=output_parser,
    )
    return summarizer.build()


__all__ = ["create_recursive_extractor"]
