"""Base callback handler for LangChain."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional, Union

from typing_extensions import Self

if TYPE_CHECKING:
    from collections.abc import Sequence
    from uuid import UUID

    from tenacity import RetryCallState

    from langchain_core.agents import AgentAction, AgentFinish
    from langchain_core.documents import Document
    from langchain_core.messages import BaseMessage
    from langchain_core.outputs import ChatGenerationChunk, GenerationChunk, LLMResult

_LOGGER = logging.getLogger(__name__)


class RetrieverManagerMixin:
    """Mixin for Retriever callbacks."""

    def on_retriever_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when Retriever errors.

        Args:
            error (BaseException): The error that occurred.
            run_id (UUID): The run ID. This is the ID of the current run.
            parent_run_id (UUID): The parent run ID. This is the ID of the parent run.
            kwargs (Any): Additional keyword arguments.
        """

    def on_retriever_end(
        self,
        documents: Sequence[Document],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when Retriever ends running.

        Args:
            documents (Sequence[Document]): The documents retrieved.
            run_id (UUID): The run ID. This is the ID of the current run.
            parent_run_id (UUID): The parent run ID. This is the ID of the parent run.
            kwargs (Any): Additional keyword arguments.
        """


class LLMManagerMixin:
    """Mixin for LLM callbacks."""

    def on_llm_new_token(
        self,
        token: str,
        *,
        chunk: Optional[Union[GenerationChunk, ChatGenerationChunk]] = None,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Run on new LLM token. Only available when streaming is enabled.

        Args:
            token (str): The new token.
            chunk (GenerationChunk | ChatGenerationChunk): The new generated chunk,
              containing content and other information.
            run_id (UUID): The run ID. This is the ID of the current run.
            parent_run_id (UUID): The parent run ID. This is the ID of the parent run.
            kwargs (Any): Additional keyword arguments.
        """

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when LLM ends running.

        Args:
            response (LLMResult): The response which was generated.
            run_id (UUID): The run ID. This is the ID of the current run.
            parent_run_id (UUID): The parent run ID. This is the ID of the parent run.
            kwargs (Any): Additional keyword arguments.
        """

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when LLM errors.

        Args:
            error (BaseException): The error that occurred.
            run_id (UUID): The run ID. This is the ID of the current run.
            parent_run_id (UUID): The parent run ID. This is the ID of the parent run.
            kwargs (Any): Additional keyword arguments.
        """


class ChainManagerMixin:
    """Mixin for chain callbacks."""

    def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when chain ends running.

        Args:
            outputs (dict[str, Any]): The outputs of the chain.
            run_id (UUID): The run ID. This is the ID of the current run.
            parent_run_id (UUID): The parent run ID. This is the ID of the parent run.
            kwargs (Any): Additional keyword arguments.
        """

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when chain errors.

        Args:
            error (BaseException): The error that occurred.
            run_id (UUID): The run ID. This is the ID of the current run.
            parent_run_id (UUID): The parent run ID. This is the ID of the parent run.
            kwargs (Any): Additional keyword arguments.
        """

    def on_agent_action(
        self,
        action: AgentAction,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Run on agent action.

        Args:
            action (AgentAction): The agent action.
            run_id (UUID): The run ID. This is the ID of the current run.
            parent_run_id (UUID): The parent run ID. This is the ID of the parent run.
            kwargs (Any): Additional keyword arguments.
        """

    def on_agent_finish(
        self,
        finish: AgentFinish,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Run on the agent end.

        Args:
            finish (AgentFinish): The agent finish.
            run_id (UUID): The run ID. This is the ID of the current run.
            parent_run_id (UUID): The parent run ID. This is the ID of the parent run.
            kwargs (Any): Additional keyword arguments.
        """


class ToolManagerMixin:
    """Mixin for tool callbacks."""

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when the tool ends running.

        Args:
            output (Any): The output of the tool.
            run_id (UUID): The run ID. This is the ID of the current run.
            parent_run_id (UUID): The parent run ID. This is the ID of the parent run.
            kwargs (Any): Additional keyword arguments.
        """

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when tool errors.

        Args:
            error (BaseException): The error that occurred.
            run_id (UUID): The run ID. This is the ID of the current run.
            parent_run_id (UUID): The parent run ID. This is the ID of the parent run.
            kwargs (Any): Additional keyword arguments.
        """


class CallbackManagerMixin:
    """Mixin for callback manager."""

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when LLM starts running.

        **ATTENTION**: This method is called for non-chat models (regular LLMs). If
            you're implementing a handler for a chat model,
            you should use on_chat_model_start instead.

        Args:
            serialized (dict[str, Any]): The serialized LLM.
            prompts (list[str]): The prompts.
            run_id (UUID): The run ID. This is the ID of the current run.
            parent_run_id (UUID): The parent run ID. This is the ID of the parent run.
            tags (Optional[list[str]]): The tags.
            metadata (Optional[dict[str, Any]]): The metadata.
            kwargs (Any): Additional keyword arguments.
        """

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when a chat model starts running.

        **ATTENTION**: This method is called for chat models. If you're implementing
            a handler for a non-chat model, you should use on_llm_start instead.

        Args:
            serialized (dict[str, Any]): The serialized chat model.
            messages (list[list[BaseMessage]]): The messages.
            run_id (UUID): The run ID. This is the ID of the current run.
            parent_run_id (UUID): The parent run ID. This is the ID of the parent run.
            tags (Optional[list[str]]): The tags.
            metadata (Optional[dict[str, Any]]): The metadata.
            kwargs (Any): Additional keyword arguments.
        """
        # NotImplementedError is thrown intentionally
        # Callback handler will fall back to on_llm_start if this is exception is thrown
        msg = f"{self.__class__.__name__} does not implement `on_chat_model_start`"
        raise NotImplementedError(msg)

    def on_retriever_start(
        self,
        serialized: dict[str, Any],
        query: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when the Retriever starts running.

        Args:
            serialized (dict[str, Any]): The serialized Retriever.
            query (str): The query.
            run_id (UUID): The run ID. This is the ID of the current run.
            parent_run_id (UUID): The parent run ID. This is the ID of the parent run.
            tags (Optional[list[str]]): The tags.
            metadata (Optional[dict[str, Any]]): The metadata.
            kwargs (Any): Additional keyword arguments.
        """

    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when a chain starts running.

        Args:
            serialized (dict[str, Any]): The serialized chain.
            inputs (dict[str, Any]): The inputs.
            run_id (UUID): The run ID. This is the ID of the current run.
            parent_run_id (UUID): The parent run ID. This is the ID of the parent run.
            tags (Optional[list[str]]): The tags.
            metadata (Optional[dict[str, Any]]): The metadata.
            kwargs (Any): Additional keyword arguments.
        """

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        inputs: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when the tool starts running.

        Args:
            serialized (dict[str, Any]): The serialized tool.
            input_str (str): The input string.
            run_id (UUID): The run ID. This is the ID of the current run.
            parent_run_id (UUID): The parent run ID. This is the ID of the parent run.
            tags (Optional[list[str]]): The tags.
            metadata (Optional[dict[str, Any]]): The metadata.
            inputs (Optional[dict[str, Any]]): The inputs.
            kwargs (Any): Additional keyword arguments.
        """


class RunManagerMixin:
    """Mixin for run manager."""

    def on_text(
        self,
        text: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Run on an arbitrary text.

        Args:
            text (str): The text.
            run_id (UUID): The run ID. This is the ID of the current run.
            parent_run_id (UUID): The parent run ID. This is the ID of the parent run.
            kwargs (Any): Additional keyword arguments.
        """

    def on_retry(
        self,
        retry_state: RetryCallState,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Run on a retry event.

        Args:
            retry_state (RetryCallState): The retry state.
            run_id (UUID): The run ID. This is the ID of the current run.
            parent_run_id (UUID): The parent run ID. This is the ID of the parent run.
            kwargs (Any): Additional keyword arguments.
        """

    def on_custom_event(
        self,
        name: str,
        data: Any,
        *,
        run_id: UUID,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Override to define a handler for a custom event.

        Args:
            name: The name of the custom event.
            data: The data for the custom event. Format will match
                  the format specified by the user.
            run_id: The ID of the run.
            tags: The tags associated with the custom event
                (includes inherited tags).
            metadata: The metadata associated with the custom event
                (includes inherited metadata).

        .. versionadded:: 0.2.15
        """


class BaseCallbackHandler(
    LLMManagerMixin,
    ChainManagerMixin,
    ToolManagerMixin,
    RetrieverManagerMixin,
    CallbackManagerMixin,
    RunManagerMixin,
):
    """Base callback handler for LangChain."""

    raise_error: bool = False
    """Whether to raise an error if an exception occurs."""

    run_inline: bool = False
    """Whether to run the callback inline."""

    @property
    def ignore_llm(self) -> bool:
        """Whether to ignore LLM callbacks."""
        return False

    @property
    def ignore_retry(self) -> bool:
        """Whether to ignore retry callbacks."""
        return False

    @property
    def ignore_chain(self) -> bool:
        """Whether to ignore chain callbacks."""
        return False

    @property
    def ignore_agent(self) -> bool:
        """Whether to ignore agent callbacks."""
        return False

    @property
    def ignore_retriever(self) -> bool:
        """Whether to ignore retriever callbacks."""
        return False

    @property
    def ignore_chat_model(self) -> bool:
        """Whether to ignore chat model callbacks."""
        return False

    @property
    def ignore_custom_event(self) -> bool:
        """Ignore custom event."""
        return False


class AsyncCallbackHandler(BaseCallbackHandler):
    """Async callback handler for LangChain."""

    async def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Run when LLM starts running.

        **ATTENTION**: This method is called for non-chat models (regular LLMs). If
            you're implementing a handler for a chat model,
            you should use on_chat_model_start instead.

        Args:
            serialized (dict[str, Any]): The serialized LLM.
            prompts (list[str]): The prompts.
            run_id (UUID): The run ID. This is the ID of the current run.
            parent_run_id (UUID): The parent run ID. This is the ID of the parent run.
            tags (Optional[list[str]]): The tags.
            metadata (Optional[dict[str, Any]]): The metadata.
            kwargs (Any): Additional keyword arguments.
        """

    async def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when a chat model starts running.

        **ATTENTION**: This method is called for chat models. If you're implementing
            a handler for a non-chat model, you should use on_llm_start instead.

        Args:
            serialized (dict[str, Any]): The serialized chat model.
            messages (list[list[BaseMessage]]): The messages.
            run_id (UUID): The run ID. This is the ID of the current run.
            parent_run_id (UUID): The parent run ID. This is the ID of the parent run.
            tags (Optional[list[str]]): The tags.
            metadata (Optional[dict[str, Any]]): The metadata.
            kwargs (Any): Additional keyword arguments.
        """
        # NotImplementedError is thrown intentionally
        # Callback handler will fall back to on_llm_start if this is exception is thrown
        msg = f"{self.__class__.__name__} does not implement `on_chat_model_start`"
        raise NotImplementedError(msg)

    async def on_llm_new_token(
        self,
        token: str,
        *,
        chunk: Optional[Union[GenerationChunk, ChatGenerationChunk]] = None,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Run on new LLM token. Only available when streaming is enabled.

        Args:
            token (str): The new token.
            chunk (GenerationChunk | ChatGenerationChunk): The new generated chunk,
              containing content and other information.
            run_id (UUID): The run ID. This is the ID of the current run.
            parent_run_id (UUID): The parent run ID. This is the ID of the parent run.
            tags (Optional[list[str]]): The tags.
            kwargs (Any): Additional keyword arguments.
        """

    async def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Run when LLM ends running.

        Args:
            response (LLMResult): The response which was generated.
            run_id (UUID): The run ID. This is the ID of the current run.
            parent_run_id (UUID): The parent run ID. This is the ID of the parent run.
            tags (Optional[list[str]]): The tags.
            kwargs (Any): Additional keyword arguments.
        """

    async def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Run when LLM errors.

        Args:
            error: The error that occurred.
            run_id: The run ID. This is the ID of the current run.
            parent_run_id: The parent run ID. This is the ID of the parent run.
            tags: The tags.
            kwargs (Any): Additional keyword arguments.
                - response (LLMResult): The response which was generated before
                    the error occurred.
        """

    async def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Run when a chain starts running.

        Args:
            serialized (dict[str, Any]): The serialized chain.
            inputs (dict[str, Any]): The inputs.
            run_id (UUID): The run ID. This is the ID of the current run.
            parent_run_id (UUID): The parent run ID. This is the ID of the parent run.
            tags (Optional[list[str]]): The tags.
            metadata (Optional[dict[str, Any]]): The metadata.
            kwargs (Any): Additional keyword arguments.
        """

    async def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Run when a chain ends running.

        Args:
            outputs (dict[str, Any]): The outputs of the chain.
            run_id (UUID): The run ID. This is the ID of the current run.
            parent_run_id (UUID): The parent run ID. This is the ID of the parent run.
            tags (Optional[list[str]]): The tags.
            kwargs (Any): Additional keyword arguments.
        """

    async def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Run when chain errors.

        Args:
            error (BaseException): The error that occurred.
            run_id (UUID): The run ID. This is the ID of the current run.
            parent_run_id (UUID): The parent run ID. This is the ID of the parent run.
            tags (Optional[list[str]]): The tags.
            kwargs (Any): Additional keyword arguments.
        """

    async def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        inputs: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Run when the tool starts running.

        Args:
            serialized (dict[str, Any]): The serialized tool.
            input_str (str): The input string.
            run_id (UUID): The run ID. This is the ID of the current run.
            parent_run_id (UUID): The parent run ID. This is the ID of the parent run.
            tags (Optional[list[str]]): The tags.
            metadata (Optional[dict[str, Any]]): The metadata.
            inputs (Optional[dict[str, Any]]): The inputs.
            kwargs (Any): Additional keyword arguments.
        """

    async def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Run when the tool ends running.

        Args:
            output (Any): The output of the tool.
            run_id (UUID): The run ID. This is the ID of the current run.
            parent_run_id (UUID): The parent run ID. This is the ID of the parent run.
            tags (Optional[list[str]]): The tags.
            kwargs (Any): Additional keyword arguments.
        """

    async def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Run when tool errors.

        Args:
            error (BaseException): The error that occurred.
            run_id (UUID): The run ID. This is the ID of the current run.
            parent_run_id (UUID): The parent run ID. This is the ID of the parent run.
            tags (Optional[list[str]]): The tags.
            kwargs (Any): Additional keyword arguments.
        """

    async def on_text(
        self,
        text: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Run on an arbitrary text.

        Args:
            text (str): The text.
            run_id (UUID): The run ID. This is the ID of the current run.
            parent_run_id (UUID): The parent run ID. This is the ID of the parent run.
            tags (Optional[list[str]]): The tags.
            kwargs (Any): Additional keyword arguments.
        """

    async def on_retry(
        self,
        retry_state: RetryCallState,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Run on a retry event.

        Args:
            retry_state (RetryCallState): The retry state.
            run_id (UUID): The run ID. This is the ID of the current run.
            parent_run_id (UUID): The parent run ID. This is the ID of the parent run.
            kwargs (Any): Additional keyword arguments.
        """

    async def on_agent_action(
        self,
        action: AgentAction,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Run on agent action.

        Args:
            action (AgentAction): The agent action.
            run_id (UUID): The run ID. This is the ID of the current run.
            parent_run_id (UUID): The parent run ID. This is the ID of the parent run.
            tags (Optional[list[str]]): The tags.
            kwargs (Any): Additional keyword arguments.
        """

    async def on_agent_finish(
        self,
        finish: AgentFinish,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Run on the agent end.

        Args:
            finish (AgentFinish): The agent finish.
            run_id (UUID): The run ID. This is the ID of the current run.
            parent_run_id (UUID): The parent run ID. This is the ID of the parent run.
            tags (Optional[list[str]]): The tags.
            kwargs (Any): Additional keyword arguments.
        """

    async def on_retriever_start(
        self,
        serialized: dict[str, Any],
        query: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Run on the retriever start.

        Args:
            serialized (dict[str, Any]): The serialized retriever.
            query (str): The query.
            run_id (UUID): The run ID. This is the ID of the current run.
            parent_run_id (UUID): The parent run ID. This is the ID of the parent run.
            tags (Optional[list[str]]): The tags.
            metadata (Optional[dict[str, Any]]): The metadata.
            kwargs (Any): Additional keyword arguments.
        """

    async def on_retriever_end(
        self,
        documents: Sequence[Document],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Run on the retriever end.

        Args:
            documents (Sequence[Document]): The documents retrieved.
            run_id (UUID): The run ID. This is the ID of the current run.
            parent_run_id (UUID): The parent run ID. This is the ID of the parent run.
            tags (Optional[list[str]]): The tags.
            kwargs (Any): Additional keyword arguments.
        """

    async def on_retriever_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> None:
        """Run on retriever error.

        Args:
            error (BaseException): The error that occurred.
            run_id (UUID): The run ID. This is the ID of the current run.
            parent_run_id (UUID): The parent run ID. This is the ID of the parent run.
            tags (Optional[list[str]]): The tags.
            kwargs (Any): Additional keyword arguments.
        """

    async def on_custom_event(
        self,
        name: str,
        data: Any,
        *,
        run_id: UUID,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Override to define a handler for a custom event.

        Args:
            name: The name of the custom event.
            data: The data for the custom event. Format will match
                  the format specified by the user.
            run_id: The ID of the run.
            tags: The tags associated with the custom event
                (includes inherited tags).
            metadata: The metadata associated with the custom event
                (includes inherited metadata).

        .. versionadded:: 0.2.15
        """


class BaseCallbackManager(CallbackManagerMixin):
    """Base callback manager for LangChain."""

    def __init__(
        self,
        handlers: list[BaseCallbackHandler],
        inheritable_handlers: Optional[list[BaseCallbackHandler]] = None,
        parent_run_id: Optional[UUID] = None,
        *,
        tags: Optional[list[str]] = None,
        inheritable_tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        inheritable_metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Initialize callback manager.

        Args:
            handlers (list[BaseCallbackHandler]): The handlers.
            inheritable_handlers (Optional[list[BaseCallbackHandler]]):
              The inheritable handlers. Default is None.
            parent_run_id (Optional[UUID]): The parent run ID. Default is None.
            tags (Optional[list[str]]): The tags. Default is None.
            inheritable_tags (Optional[list[str]]): The inheritable tags.
                Default is None.
            metadata (Optional[dict[str, Any]]): The metadata. Default is None.
            inheritable_metadata (Optional[dict[str, Any]]): The inheritable metadata.
                Default is None.
        """
        self.handlers: list[BaseCallbackHandler] = handlers
        self.inheritable_handlers: list[BaseCallbackHandler] = (
            inheritable_handlers or []
        )
        self.parent_run_id: Optional[UUID] = parent_run_id
        self.tags = tags or []
        self.inheritable_tags = inheritable_tags or []
        self.metadata = metadata or {}
        self.inheritable_metadata = inheritable_metadata or {}

    def copy(self) -> Self:
        """Copy the callback manager."""
        return self.__class__(
            handlers=self.handlers.copy(),
            inheritable_handlers=self.inheritable_handlers.copy(),
            parent_run_id=self.parent_run_id,
            tags=self.tags.copy(),
            inheritable_tags=self.inheritable_tags.copy(),
            metadata=self.metadata.copy(),
            inheritable_metadata=self.inheritable_metadata.copy(),
        )

    def merge(self, other: BaseCallbackManager) -> Self:
        """Merge the callback manager with another callback manager.

        May be overwritten in subclasses. Primarily used internally
        within merge_configs.

        Returns:
            BaseCallbackManager: The merged callback manager of the same type
                as the current object.

        Example: Merging two callback managers.

            .. code-block:: python

                from langchain_core.callbacks.manager import CallbackManager, trace_as_chain_group
                from langchain_core.callbacks.stdout import StdOutCallbackHandler

                manager = CallbackManager(handlers=[StdOutCallbackHandler()], tags=["tag2"])
                with trace_as_chain_group("My Group Name", tags=["tag1"]) as group_manager:
                    merged_manager = group_manager.merge(manager)
                    print(merged_manager.handlers)
                    # [
                    #    <langchain_core.callbacks.stdout.StdOutCallbackHandler object at ...>,
                    #    <langchain_core.callbacks.streaming_stdout.StreamingStdOutCallbackHandler object at ...>,
                    # ]

                    print(merged_manager.tags)
                    #    ['tag2', 'tag1']

        """  # noqa: E501
        manager = self.__class__(
            parent_run_id=self.parent_run_id or other.parent_run_id,
            handlers=[],
            inheritable_handlers=[],
            tags=list(set(self.tags + other.tags)),
            inheritable_tags=list(set(self.inheritable_tags + other.inheritable_tags)),
            metadata={
                **self.metadata,
                **other.metadata,
            },
        )

        handlers = self.handlers + other.handlers
        inheritable_handlers = self.inheritable_handlers + other.inheritable_handlers

        for handler in handlers:
            manager.add_handler(handler)

        for handler in inheritable_handlers:
            manager.add_handler(handler, inherit=True)
        return manager

    @property
    def is_async(self) -> bool:
        """Whether the callback manager is async."""
        return False

    def add_handler(
        self,
        handler: BaseCallbackHandler,
        inherit: bool = True,  # noqa: FBT001,FBT002
    ) -> None:
        """Add a handler to the callback manager.

        Args:
            handler (BaseCallbackHandler): The handler to add.
            inherit (bool): Whether to inherit the handler. Default is True.
        """
        if handler not in self.handlers:
            self.handlers.append(handler)
        if inherit and handler not in self.inheritable_handlers:
            self.inheritable_handlers.append(handler)

    def remove_handler(self, handler: BaseCallbackHandler) -> None:
        """Remove a handler from the callback manager.

        Args:
            handler (BaseCallbackHandler): The handler to remove.
        """
        if handler in self.handlers:
            self.handlers.remove(handler)
        if handler in self.inheritable_handlers:
            self.inheritable_handlers.remove(handler)

    def set_handlers(
        self,
        handlers: list[BaseCallbackHandler],
        inherit: bool = True,  # noqa: FBT001,FBT002
    ) -> None:
        """Set handlers as the only handlers on the callback manager.

        Args:
            handlers (list[BaseCallbackHandler]): The handlers to set.
            inherit (bool): Whether to inherit the handlers. Default is True.
        """
        self.handlers = []
        self.inheritable_handlers = []
        for handler in handlers:
            self.add_handler(handler, inherit=inherit)

    def set_handler(
        self,
        handler: BaseCallbackHandler,
        inherit: bool = True,  # noqa: FBT001,FBT002
    ) -> None:
        """Set handler as the only handler on the callback manager.

        Args:
            handler (BaseCallbackHandler): The handler to set.
            inherit (bool): Whether to inherit the handler. Default is True.
        """
        self.set_handlers([handler], inherit=inherit)

    def add_tags(
        self,
        tags: list[str],
        inherit: bool = True,  # noqa: FBT001,FBT002
    ) -> None:
        """Add tags to the callback manager.

        Args:
            tags (list[str]): The tags to add.
            inherit (bool): Whether to inherit the tags. Default is True.
        """
        for tag in tags:
            if tag in self.tags:
                self.remove_tags([tag])
        self.tags.extend(tags)
        if inherit:
            self.inheritable_tags.extend(tags)

    def remove_tags(self, tags: list[str]) -> None:
        """Remove tags from the callback manager.

        Args:
            tags (list[str]): The tags to remove.
        """
        for tag in tags:
            self.tags.remove(tag)
            self.inheritable_tags.remove(tag)

    def add_metadata(
        self,
        metadata: dict[str, Any],
        inherit: bool = True,  # noqa: FBT001,FBT002
    ) -> None:
        """Add metadata to the callback manager.

        Args:
            metadata (dict[str, Any]): The metadata to add.
            inherit (bool): Whether to inherit the metadata. Default is True.
        """
        self.metadata.update(metadata)
        if inherit:
            self.inheritable_metadata.update(metadata)

    def remove_metadata(self, keys: list[str]) -> None:
        """Remove metadata from the callback manager.

        Args:
            keys (list[str]): The keys to remove.
        """
        for key in keys:
            self.metadata.pop(key)
            self.inheritable_metadata.pop(key)


Callbacks = Optional[Union[list[BaseCallbackHandler], BaseCallbackManager]]
