"""Base callback handler for LangChain."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence
    from uuid import UUID

    from tenacity import RetryCallState
    from typing_extensions import Self

    from langchain_core.agents import AgentAction, AgentFinish
    from langchain_core.documents import Document
    from langchain_core.messages import BaseMessage
    from langchain_core.outputs import ChatGenerationChunk, GenerationChunk, LLMResult

_LOGGER = logging.getLogger(__name__)


class RetrieverManagerMixin:
    """Mixin for `Retriever` callbacks."""

    def on_retriever_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        """Run when `Retriever` errors.

        Args:
            error: The error that occurred.
            run_id: The ID of the current run.
            parent_run_id: The ID of the parent run.
            **kwargs: Additional keyword arguments.
        """

    def on_retriever_end(
        self,
        documents: Sequence[Document],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        """Run when `Retriever` ends running.

        Args:
            documents: The documents retrieved.
            run_id: The ID of the current run.
            parent_run_id: The ID of the parent run.
            **kwargs: Additional keyword arguments.
        """


class LLMManagerMixin:
    """Mixin for LLM callbacks."""

    def on_llm_new_token(
        self,
        token: str,
        *,
        chunk: GenerationChunk | ChatGenerationChunk | None = None,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Run on new output token.

        Only available when streaming is enabled.

        For both chat models and non-chat models (legacy text completion LLMs).

        Args:
            token: The new token.
            chunk: The new generated chunk, containing content and other information.
            run_id: The ID of the current run.
            parent_run_id: The ID of the parent run.
            tags: The tags.
            **kwargs: Additional keyword arguments.
        """

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Run when LLM ends running.

        Args:
            response: The response which was generated.
            run_id: The ID of the current run.
            parent_run_id: The ID of the parent run.
            tags: The tags.
            **kwargs: Additional keyword arguments.
        """

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Run when LLM errors.

        Args:
            error: The error that occurred.
            run_id: The ID of the current run.
            parent_run_id: The ID of the parent run.
            tags: The tags.
            **kwargs: Additional keyword arguments.
        """


class ChainManagerMixin:
    """Mixin for chain callbacks."""

    def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        """Run when chain ends running.

        Args:
            outputs: The outputs of the chain.
            run_id: The ID of the current run.
            parent_run_id: The ID of the parent run.
            **kwargs: Additional keyword arguments.
        """

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        """Run when chain errors.

        Args:
            error: The error that occurred.
            run_id: The ID of the current run.
            parent_run_id: The ID of the parent run.
            **kwargs: Additional keyword arguments.
        """

    def on_agent_action(
        self,
        action: AgentAction,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        """Run on agent action.

        Args:
            action: The agent action.
            run_id: The ID of the current run.
            parent_run_id: The ID of the parent run.
            **kwargs: Additional keyword arguments.
        """

    def on_agent_finish(
        self,
        finish: AgentFinish,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        """Run on the agent end.

        Args:
            finish: The agent finish.
            run_id: The ID of the current run.
            parent_run_id: The ID of the parent run.
            **kwargs: Additional keyword arguments.
        """


class ToolManagerMixin:
    """Mixin for tool callbacks."""

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        """Run when the tool ends running.

        Args:
            output: The output of the tool.
            run_id: The ID of the current run.
            parent_run_id: The ID of the parent run.
            **kwargs: Additional keyword arguments.
        """

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        """Run when tool errors.

        Args:
            error: The error that occurred.
            run_id: The ID of the current run.
            parent_run_id: The ID of the parent run.
            **kwargs: Additional keyword arguments.
        """


class CallbackManagerMixin:
    """Mixin for callback manager."""

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Run when LLM starts running.

        !!! warning

            This method is called for non-chat models (regular text completion LLMs). If
            you're implementing a handler for a chat model, you should use
            `on_chat_model_start` instead.

        Args:
            serialized: The serialized LLM.
            prompts: The prompts.
            run_id: The ID of the current run.
            parent_run_id: The ID of the parent run.
            tags: The tags.
            metadata: The metadata.
            **kwargs: Additional keyword arguments.
        """

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Run when a chat model starts running.

        !!! warning

            This method is called for chat models. If you're implementing a handler for
            a non-chat model, you should use `on_llm_start` instead.

        Args:
            serialized: The serialized chat model.
            messages: The messages.
            run_id: The ID of the current run.
            parent_run_id: The ID of the parent run.
            tags: The tags.
            metadata: The metadata.
            **kwargs: Additional keyword arguments.
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
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Run when the `Retriever` starts running.

        Args:
            serialized: The serialized `Retriever`.
            query: The query.
            run_id: The ID of the current run.
            parent_run_id: The ID of the parent run.
            tags: The tags.
            metadata: The metadata.
            **kwargs: Additional keyword arguments.
        """

    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Run when a chain starts running.

        Args:
            serialized: The serialized chain.
            inputs: The inputs.
            run_id: The ID of the current run.
            parent_run_id: The ID of the parent run.
            tags: The tags.
            metadata: The metadata.
            **kwargs: Additional keyword arguments.
        """

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Run when the tool starts running.

        Args:
            serialized: The serialized chain.
            input_str: The input string.
            run_id: The ID of the current run.
            parent_run_id: The ID of the parent run.
            tags: The tags.
            metadata: The metadata.
            inputs: The inputs.
            **kwargs: Additional keyword arguments.
        """


class RunManagerMixin:
    """Mixin for run manager."""

    def on_text(
        self,
        text: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        """Run on an arbitrary text.

        Args:
            text: The text.
            run_id: The ID of the current run.
            parent_run_id: The ID of the parent run.
            **kwargs: Additional keyword arguments.
        """

    def on_retry(
        self,
        retry_state: RetryCallState,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        """Run on a retry event.

        Args:
            retry_state: The retry state.
            run_id: The ID of the current run.
            parent_run_id: The ID of the parent run.
            **kwargs: Additional keyword arguments.
        """

    def on_custom_event(
        self,
        name: str,
        data: Any,
        *,
        run_id: UUID,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Override to define a handler for a custom event.

        Args:
            name: The name of the custom event.
            data: The data for the custom event.

                Format will match the format specified by the user.
            run_id: The ID of the run.
            tags: The tags associated with the custom event (includes inherited tags).
            metadata: The metadata associated with the custom event (includes inherited
                metadata).
        """


class BaseCallbackHandler(
    LLMManagerMixin,
    ChainManagerMixin,
    ToolManagerMixin,
    RetrieverManagerMixin,
    CallbackManagerMixin,
    RunManagerMixin,
):
    """Base callback handler."""

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
    """Base async callback handler."""

    async def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Run when the model starts running.

        !!! warning

            This method is called for non-chat models (regular text completion LLMs). If
            you're implementing a handler for a chat model, you should use
            `on_chat_model_start` instead.

        Args:
            serialized: The serialized LLM.
            prompts: The prompts.
            run_id: The ID of the current run.
            parent_run_id: The ID of the parent run.
            tags: The tags.
            metadata: The metadata.
            **kwargs: Additional keyword arguments.
        """

    async def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Run when a chat model starts running.

        !!! warning

            This method is called for chat models. If you're implementing a handler for
            a non-chat model, you should use `on_llm_start` instead.

        Args:
            serialized: The serialized chat model.
            messages: The messages.
            run_id: The ID of the current run.
            parent_run_id: The ID of the parent run.
            tags: The tags.
            metadata: The metadata.
            **kwargs: Additional keyword arguments.
        """
        # NotImplementedError is thrown intentionally
        # Callback handler will fall back to on_llm_start if this is exception is thrown
        msg = f"{self.__class__.__name__} does not implement `on_chat_model_start`"
        raise NotImplementedError(msg)

    async def on_llm_new_token(
        self,
        token: str,
        *,
        chunk: GenerationChunk | ChatGenerationChunk | None = None,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Run on new output token. Only available when streaming is enabled.

        For both chat models and non-chat models (legacy text completion LLMs).

        Args:
            token: The new token.
            chunk: The new generated chunk, containing content and other information.
            run_id: The ID of the current run.
            parent_run_id: The ID of the parent run.
            tags: The tags.
            **kwargs: Additional keyword arguments.
        """

    async def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Run when the model ends running.

        Args:
            response: The response which was generated.
            run_id: The ID of the current run.
            parent_run_id: The ID of the parent run.
            tags: The tags.
            **kwargs: Additional keyword arguments.
        """

    async def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Run when LLM errors.

        Args:
            error: The error that occurred.
            run_id: The ID of the current run.
            parent_run_id: The ID of the parent run.
            tags: The tags.
            **kwargs: Additional keyword arguments.

                - response (LLMResult): The response which was generated before
                    the error occurred.
        """

    async def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Run when a chain starts running.

        Args:
            serialized: The serialized chain.
            inputs: The inputs.
            run_id: The ID of the current run.
            parent_run_id: The ID of the parent run.
            tags: The tags.
            metadata: The metadata.
            **kwargs: Additional keyword arguments.
        """

    async def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Run when a chain ends running.

        Args:
            outputs: The outputs of the chain.
            run_id: The ID of the current run.
            parent_run_id: The ID of the parent run.
            tags: The tags.
            **kwargs: Additional keyword arguments.
        """

    async def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Run when chain errors.

        Args:
            error: The error that occurred.
            run_id: The ID of the current run.
            parent_run_id: The ID of the parent run.
            tags: The tags.
            **kwargs: Additional keyword arguments.
        """

    async def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Run when the tool starts running.

        Args:
            serialized: The serialized tool.
            input_str: The input string.
            run_id: The ID of the current run.
            parent_run_id: The ID of the parent run.
            tags: The tags.
            metadata: The metadata.
            inputs: The inputs.
            **kwargs: Additional keyword arguments.
        """

    async def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Run when the tool ends running.

        Args:
            output: The output of the tool.
            run_id: The ID of the current run.
            parent_run_id: The ID of the parent run.
            tags: The tags.
            **kwargs: Additional keyword arguments.
        """

    async def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Run when tool errors.

        Args:
            error: The error that occurred.
            run_id: The ID of the current run.
            parent_run_id: The ID of the parent run.
            tags: The tags.
            **kwargs: Additional keyword arguments.
        """

    async def on_text(
        self,
        text: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Run on an arbitrary text.

        Args:
            text: The text.
            run_id: The ID of the current run.
            parent_run_id: The ID of the parent run.
            tags: The tags.
            **kwargs: Additional keyword arguments.
        """

    async def on_retry(
        self,
        retry_state: RetryCallState,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> Any:
        """Run on a retry event.

        Args:
            retry_state: The retry state.
            run_id: The ID of the current run.
            parent_run_id: The ID of the parent run.
            **kwargs: Additional keyword arguments.
        """

    async def on_agent_action(
        self,
        action: AgentAction,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Run on agent action.

        Args:
            action: The agent action.
            run_id: The ID of the current run.
            parent_run_id: The ID of the parent run.
            tags: The tags.
            **kwargs: Additional keyword arguments.
        """

    async def on_agent_finish(
        self,
        finish: AgentFinish,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Run on the agent end.

        Args:
            finish: The agent finish.
            run_id: The ID of the current run.
            parent_run_id: The ID of the parent run.
            tags: The tags.
            **kwargs: Additional keyword arguments.
        """

    async def on_retriever_start(
        self,
        serialized: dict[str, Any],
        query: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Run on the retriever start.

        Args:
            serialized: The serialized retriever.
            query: The query.
            run_id: The ID of the current run.
            parent_run_id: The ID of the parent run.
            tags: The tags.
            metadata: The metadata.
            **kwargs: Additional keyword arguments.
        """

    async def on_retriever_end(
        self,
        documents: Sequence[Document],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Run on the retriever end.

        Args:
            documents: The documents retrieved.
            run_id: The ID of the current run.
            parent_run_id: The ID of the parent run.
            tags: The tags.
            **kwargs: Additional keyword arguments.
        """

    async def on_retriever_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        """Run on retriever error.

        Args:
            error: The error that occurred.
            run_id: The ID of the current run.
            parent_run_id: The ID of the parent run.
            tags: The tags.
            **kwargs: Additional keyword arguments.
        """

    async def on_custom_event(
        self,
        name: str,
        data: Any,
        *,
        run_id: UUID,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Override to define a handler for custom events.

        Args:
            name: The name of the custom event.
            data: The data for the custom event.

                Format will match the format specified by the user.
            run_id: The ID of the run.
            tags: The tags associated with the custom event (includes inherited tags).
            metadata: The metadata associated with the custom event (includes inherited
                metadata).
        """


class BaseCallbackManager(CallbackManagerMixin):
    """Base callback manager."""

    def __init__(
        self,
        handlers: list[BaseCallbackHandler],
        inheritable_handlers: list[BaseCallbackHandler] | None = None,
        parent_run_id: UUID | None = None,
        *,
        tags: list[str] | None = None,
        inheritable_tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        inheritable_metadata: dict[str, Any] | None = None,
    ) -> None:
        """Initialize callback manager.

        Args:
            handlers: The handlers.
            inheritable_handlers: The inheritable handlers.
            parent_run_id: The parent run ID.
            tags: The tags.
            inheritable_tags: The inheritable tags.
            metadata: The metadata.
            inheritable_metadata: The inheritable metadata.
        """
        self.handlers: list[BaseCallbackHandler] = handlers
        self.inheritable_handlers: list[BaseCallbackHandler] = (
            inheritable_handlers or []
        )
        self.parent_run_id: UUID | None = parent_run_id
        self.tags = tags or []
        self.inheritable_tags = inheritable_tags or []
        self.metadata = metadata or {}
        self.inheritable_metadata = inheritable_metadata or {}

    def copy(self) -> Self:
        """Return a copy of the callback manager."""
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

        May be overwritten in subclasses.

        Primarily used internally within `merge_configs`.

        Returns:
            The merged callback manager of the same type as the current object.

        Example:
            ```python
            # Merging two callback managers`
            from langchain_core.callbacks.manager import (
                CallbackManager,
                trace_as_chain_group,
            )
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
            ```
        """  # noqa: E501
        # Combine handlers and inheritable_handlers separately, using sets
        # to deduplicate (order not preserved)
        combined_handlers = list(set(self.handlers) | set(other.handlers))
        combined_inheritable = list(
            set(self.inheritable_handlers) | set(other.inheritable_handlers)
        )

        return self.__class__(
            parent_run_id=self.parent_run_id or other.parent_run_id,
            handlers=combined_handlers,
            inheritable_handlers=combined_inheritable,
            tags=list(set(self.tags + other.tags)),
            inheritable_tags=list(set(self.inheritable_tags + other.inheritable_tags)),
            metadata={
                **self.metadata,
                **other.metadata,
            },
            inheritable_metadata={
                **self.inheritable_metadata,
                **other.inheritable_metadata,
            },
        )

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
            handler: The handler to add.
            inherit: Whether to inherit the handler.
        """
        if handler not in self.handlers:
            self.handlers.append(handler)
        if inherit and handler not in self.inheritable_handlers:
            self.inheritable_handlers.append(handler)

    def remove_handler(self, handler: BaseCallbackHandler) -> None:
        """Remove a handler from the callback manager.

        Args:
            handler: The handler to remove.
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
            handlers: The handlers to set.
            inherit: Whether to inherit the handlers.
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
            handler: The handler to set.
            inherit: Whether to inherit the handler.
        """
        self.set_handlers([handler], inherit=inherit)

    def add_tags(
        self,
        tags: list[str],
        inherit: bool = True,  # noqa: FBT001,FBT002
    ) -> None:
        """Add tags to the callback manager.

        Args:
            tags: The tags to add.
            inherit: Whether to inherit the tags.
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
            tags: The tags to remove.
        """
        for tag in tags:
            if tag in self.tags:
                self.tags.remove(tag)
            if tag in self.inheritable_tags:
                self.inheritable_tags.remove(tag)

    def add_metadata(
        self,
        metadata: dict[str, Any],
        inherit: bool = True,  # noqa: FBT001,FBT002
    ) -> None:
        """Add metadata to the callback manager.

        Args:
            metadata: The metadata to add.
            inherit: Whether to inherit the metadata.
        """
        self.metadata.update(metadata)
        if inherit:
            self.inheritable_metadata.update(metadata)

    def remove_metadata(self, keys: list[str]) -> None:
        """Remove metadata from the callback manager.

        Args:
            keys: The keys to remove.
        """
        for key in keys:
            self.metadata.pop(key, None)
            self.inheritable_metadata.pop(key, None)


Callbacks = list[BaseCallbackHandler] | BaseCallbackManager | None
