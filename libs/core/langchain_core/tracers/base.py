"""Base interfaces for tracing runs."""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import (
    TYPE_CHECKING,
    Any,
    Optional,
    Union,
)

from typing_extensions import override

from langchain_core.callbacks.base import AsyncCallbackHandler, BaseCallbackHandler
from langchain_core.exceptions import TracerException  # noqa: F401
from langchain_core.tracers.core import _TracerCore

if TYPE_CHECKING:
    from collections.abc import Sequence
    from uuid import UUID

    from tenacity import RetryCallState

    from langchain_core.documents import Document
    from langchain_core.messages import BaseMessage
    from langchain_core.outputs import ChatGenerationChunk, GenerationChunk, LLMResult
    from langchain_core.tracers.schemas import Run

logger = logging.getLogger(__name__)


class BaseTracer(_TracerCore, BaseCallbackHandler, ABC):
    """Base interface for tracers."""

    @abstractmethod
    def _persist_run(self, run: Run) -> None:
        """Persist a run."""

    def _start_trace(self, run: Run) -> None:
        """Start a trace for a run."""
        super()._start_trace(run)
        self._on_run_create(run)

    def _end_trace(self, run: Run) -> None:
        """End a trace for a run."""
        if not run.parent_run_id:
            self._persist_run(run)
        self.run_map.pop(str(run.id))
        self._on_run_update(run)

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        tags: Optional[list[str]] = None,
        parent_run_id: Optional[UUID] = None,
        metadata: Optional[dict[str, Any]] = None,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> Run:
        """Start a trace for an LLM run.

        Args:
            serialized: The serialized model.
            messages: The messages to start the chat with.
            run_id: The run ID.
            tags: The tags for the run. Defaults to None.
            parent_run_id: The parent run ID. Defaults to None.
            metadata: The metadata for the run. Defaults to None.
            name: The name of the run.
            kwargs: Additional arguments.

        Returns:
            The run.
        """
        chat_model_run = self._create_chat_model_run(
            serialized=serialized,
            messages=messages,
            run_id=run_id,
            parent_run_id=parent_run_id,
            tags=tags,
            metadata=metadata,
            name=name,
            **kwargs,
        )
        self._start_trace(chat_model_run)
        self._on_chat_model_start(chat_model_run)
        return chat_model_run

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        tags: Optional[list[str]] = None,
        parent_run_id: Optional[UUID] = None,
        metadata: Optional[dict[str, Any]] = None,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> Run:
        """Start a trace for an LLM run.

        Args:
            serialized: The serialized model.
            prompts: The prompts to start the LLM with.
            run_id: The run ID.
            tags: The tags for the run. Defaults to None.
            parent_run_id: The parent run ID. Defaults to None.
            metadata: The metadata for the run. Defaults to None.
            name: The name of the run.
            kwargs: Additional arguments.

        Returns:
            The run.
        """
        llm_run = self._create_llm_run(
            serialized=serialized,
            prompts=prompts,
            run_id=run_id,
            parent_run_id=parent_run_id,
            tags=tags,
            metadata=metadata,
            name=name,
            **kwargs,
        )
        self._start_trace(llm_run)
        self._on_llm_start(llm_run)
        return llm_run

    @override
    def on_llm_new_token(
        self,
        token: str,
        *,
        chunk: Optional[Union[GenerationChunk, ChatGenerationChunk]] = None,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Run:
        """Run on new LLM token. Only available when streaming is enabled.

        Args:
            token: The token.
            chunk: The chunk. Defaults to None.
            run_id: The run ID.
            parent_run_id: The parent run ID. Defaults to None.
            kwargs: Additional arguments.

        Returns:
            The run.
        """
        # "chat_model" is only used for the experimental new streaming_events format.
        # This change should not affect any existing tracers.
        llm_run = self._llm_run_with_token_event(
            token=token,
            run_id=run_id,
            chunk=chunk,
            parent_run_id=parent_run_id,
        )
        self._on_llm_new_token(llm_run, token, chunk)
        return llm_run

    @override
    def on_retry(
        self,
        retry_state: RetryCallState,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> Run:
        """Run on retry.

        Args:
            retry_state: The retry state.
            run_id: The run ID.
            kwargs: Additional arguments.

        Returns:
            The run.
        """
        return self._llm_run_with_retry_event(
            retry_state=retry_state,
            run_id=run_id,
        )

    @override
    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> Run:
        """End a trace for an LLM run.

        Args:
            response: The response.
            run_id: The run ID.
            kwargs: Additional arguments.

        Returns:
            The run.
        """
        # "chat_model" is only used for the experimental new streaming_events format.
        # This change should not affect any existing tracers.
        llm_run = self._complete_llm_run(
            response=response,
            run_id=run_id,
        )
        self._end_trace(llm_run)
        self._on_llm_end(llm_run)
        return llm_run

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> Run:
        """Handle an error for an LLM run.

        Args:
            error: The error.
            run_id: The run ID.
            kwargs: Additional arguments.

        Returns:
            The run.
        """
        # "chat_model" is only used for the experimental new streaming_events format.
        # This change should not affect any existing tracers.
        llm_run = self._errored_llm_run(
            error=error, run_id=run_id, response=kwargs.pop("response", None)
        )
        self._end_trace(llm_run)
        self._on_llm_error(llm_run)
        return llm_run

    @override
    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        tags: Optional[list[str]] = None,
        parent_run_id: Optional[UUID] = None,
        metadata: Optional[dict[str, Any]] = None,
        run_type: Optional[str] = None,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> Run:
        """Start a trace for a chain run.

        Args:
            serialized: The serialized chain.
            inputs: The inputs for the chain.
            run_id: The run ID.
            tags: The tags for the run. Defaults to None.
            parent_run_id: The parent run ID. Defaults to None.
            metadata: The metadata for the run. Defaults to None.
            run_type: The type of the run. Defaults to None.
            name: The name of the run.
            kwargs: Additional arguments.

        Returns:
            The run.
        """
        chain_run = self._create_chain_run(
            serialized=serialized,
            inputs=inputs,
            run_id=run_id,
            tags=tags,
            parent_run_id=parent_run_id,
            metadata=metadata,
            run_type=run_type,
            name=name,
            **kwargs,
        )
        self._start_trace(chain_run)
        self._on_chain_start(chain_run)
        return chain_run

    @override
    def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        inputs: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Run:
        """End a trace for a chain run.

        Args:
            outputs: The outputs for the chain.
            run_id: The run ID.
            inputs: The inputs for the chain. Defaults to None.
            kwargs: Additional arguments.

        Returns:
            The run.
        """
        chain_run = self._complete_chain_run(
            outputs=outputs,
            run_id=run_id,
            inputs=inputs,
        )
        self._end_trace(chain_run)
        self._on_chain_end(chain_run)
        return chain_run

    @override
    def on_chain_error(
        self,
        error: BaseException,
        *,
        inputs: Optional[dict[str, Any]] = None,
        run_id: UUID,
        **kwargs: Any,
    ) -> Run:
        """Handle an error for a chain run.

        Args:
            error: The error.
            inputs: The inputs for the chain. Defaults to None.
            run_id: The run ID.
            kwargs: Additional arguments.

        Returns:
            The run.
        """
        chain_run = self._errored_chain_run(
            error=error,
            run_id=run_id,
            inputs=inputs,
        )
        self._end_trace(chain_run)
        self._on_chain_error(chain_run)
        return chain_run

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        tags: Optional[list[str]] = None,
        parent_run_id: Optional[UUID] = None,
        metadata: Optional[dict[str, Any]] = None,
        name: Optional[str] = None,
        inputs: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Run:
        """Start a trace for a tool run.

        Args:
            serialized: The serialized tool.
            input_str: The input string.
            run_id: The run ID.
            tags: The tags for the run. Defaults to None.
            parent_run_id: The parent run ID. Defaults to None.
            metadata: The metadata for the run. Defaults to None.
            name: The name of the run.
            inputs: The inputs for the tool.
            kwargs: Additional arguments.

        Returns:
            The run.
        """
        tool_run = self._create_tool_run(
            serialized=serialized,
            input_str=input_str,
            run_id=run_id,
            tags=tags,
            parent_run_id=parent_run_id,
            metadata=metadata,
            name=name,
            inputs=inputs,
            **kwargs,
        )
        self._start_trace(tool_run)
        self._on_tool_start(tool_run)
        return tool_run

    @override
    def on_tool_end(self, output: Any, *, run_id: UUID, **kwargs: Any) -> Run:
        """End a trace for a tool run.

        Args:
            output: The output for the tool.
            run_id: The run ID.
            kwargs: Additional arguments.

        Returns:
            The run.
        """
        tool_run = self._complete_tool_run(
            output=output,
            run_id=run_id,
        )
        self._end_trace(tool_run)
        self._on_tool_end(tool_run)
        return tool_run

    @override
    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> Run:
        """Handle an error for a tool run.

        Args:
            error: The error.
            run_id: The run ID.
            kwargs: Additional arguments.

        Returns:
            The run.
        """
        tool_run = self._errored_tool_run(
            error=error,
            run_id=run_id,
        )
        self._end_trace(tool_run)
        self._on_tool_error(tool_run)
        return tool_run

    def on_retriever_start(
        self,
        serialized: dict[str, Any],
        query: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> Run:
        """Run when the Retriever starts running.

        Args:
            serialized: The serialized retriever.
            query: The query.
            run_id: The run ID.
            parent_run_id: The parent run ID. Defaults to None.
            tags: The tags for the run. Defaults to None.
            metadata: The metadata for the run. Defaults to None.
            name: The name of the run.
            kwargs: Additional arguments.

        Returns:
            The run.
        """
        retrieval_run = self._create_retrieval_run(
            serialized=serialized,
            query=query,
            run_id=run_id,
            parent_run_id=parent_run_id,
            tags=tags,
            metadata=metadata,
            name=name,
            **kwargs,
        )
        self._start_trace(retrieval_run)
        self._on_retriever_start(retrieval_run)
        return retrieval_run

    @override
    def on_retriever_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> Run:
        """Run when Retriever errors.

        Args:
            error: The error.
            run_id: The run ID.
            kwargs: Additional arguments.

        Returns:
            The run.
        """
        retrieval_run = self._errored_retrieval_run(
            error=error,
            run_id=run_id,
        )
        self._end_trace(retrieval_run)
        self._on_retriever_error(retrieval_run)
        return retrieval_run

    @override
    def on_retriever_end(
        self, documents: Sequence[Document], *, run_id: UUID, **kwargs: Any
    ) -> Run:
        """Run when the Retriever ends running.

        Args:
            documents: The documents.
            run_id: The run ID.
            kwargs: Additional arguments.

        Returns:
            The run.
        """
        retrieval_run = self._complete_retrieval_run(
            documents=documents,
            run_id=run_id,
        )
        self._end_trace(retrieval_run)
        self._on_retriever_end(retrieval_run)
        return retrieval_run

    def __deepcopy__(self, memo: dict) -> BaseTracer:
        """Return self."""
        return self

    def __copy__(self) -> BaseTracer:
        """Return self."""
        return self


class AsyncBaseTracer(_TracerCore, AsyncCallbackHandler, ABC):
    """Async Base interface for tracers."""

    @abstractmethod
    @override
    async def _persist_run(self, run: Run) -> None:
        """Persist a run."""

    @override
    async def _start_trace(self, run: Run) -> None:
        """Start a trace for a run.

        Starting a trace will run concurrently with each _on_[run_type]_start method.
        No _on_[run_type]_start callback should depend on operations in _start_trace.
        """
        super()._start_trace(run)
        await self._on_run_create(run)

    @override
    async def _end_trace(self, run: Run) -> None:
        """End a trace for a run.

        Ending a trace will run concurrently with each _on_[run_type]_end method.
        No _on_[run_type]_end callback should depend on operations in _end_trace.
        """
        if not run.parent_run_id:
            await self._persist_run(run)
        self.run_map.pop(str(run.id))
        await self._on_run_update(run)

    @override
    async def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        chat_model_run = self._create_chat_model_run(
            serialized=serialized,
            messages=messages,
            run_id=run_id,
            parent_run_id=parent_run_id,
            tags=tags,
            metadata=metadata,
            name=name,
            **kwargs,
        )
        tasks = [
            self._start_trace(chat_model_run),
            self._on_chat_model_start(chat_model_run),
        ]
        await asyncio.gather(*tasks)
        return chat_model_run

    @override
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
        llm_run = self._create_llm_run(
            serialized=serialized,
            prompts=prompts,
            run_id=run_id,
            parent_run_id=parent_run_id,
            tags=tags,
            metadata=metadata,
            **kwargs,
        )
        tasks = [self._start_trace(llm_run), self._on_llm_start(llm_run)]
        await asyncio.gather(*tasks)

    @override
    async def on_llm_new_token(
        self,
        token: str,
        *,
        chunk: Optional[Union[GenerationChunk, ChatGenerationChunk]] = None,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        llm_run = self._llm_run_with_token_event(
            token=token,
            run_id=run_id,
            chunk=chunk,
            parent_run_id=parent_run_id,
        )
        await self._on_llm_new_token(llm_run, token, chunk)

    @override
    async def on_retry(
        self,
        retry_state: RetryCallState,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        self._llm_run_with_retry_event(
            retry_state=retry_state,
            run_id=run_id,
        )

    @override
    async def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> None:
        llm_run = self._complete_llm_run(
            response=response,
            run_id=run_id,
        )
        tasks = [self._on_llm_end(llm_run), self._end_trace(llm_run)]
        await asyncio.gather(*tasks)

    @override
    async def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> None:
        llm_run = self._errored_llm_run(
            error=error,
            run_id=run_id,
        )
        tasks = [self._on_llm_error(llm_run), self._end_trace(llm_run)]
        await asyncio.gather(*tasks)

    @override
    async def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        tags: Optional[list[str]] = None,
        parent_run_id: Optional[UUID] = None,
        metadata: Optional[dict[str, Any]] = None,
        run_type: Optional[str] = None,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        chain_run = self._create_chain_run(
            serialized=serialized,
            inputs=inputs,
            run_id=run_id,
            tags=tags,
            parent_run_id=parent_run_id,
            metadata=metadata,
            run_type=run_type,
            name=name,
            **kwargs,
        )
        tasks = [self._start_trace(chain_run), self._on_chain_start(chain_run)]
        await asyncio.gather(*tasks)

    @override
    async def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        inputs: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        chain_run = self._complete_chain_run(
            outputs=outputs,
            run_id=run_id,
            inputs=inputs,
        )
        tasks = [self._end_trace(chain_run), self._on_chain_end(chain_run)]
        await asyncio.gather(*tasks)

    @override
    async def on_chain_error(
        self,
        error: BaseException,
        *,
        inputs: Optional[dict[str, Any]] = None,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        chain_run = self._errored_chain_run(
            error=error,
            inputs=inputs,
            run_id=run_id,
        )
        tasks = [self._end_trace(chain_run), self._on_chain_error(chain_run)]
        await asyncio.gather(*tasks)

    @override
    async def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        tags: Optional[list[str]] = None,
        parent_run_id: Optional[UUID] = None,
        metadata: Optional[dict[str, Any]] = None,
        name: Optional[str] = None,
        inputs: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        tool_run = self._create_tool_run(
            serialized=serialized,
            input_str=input_str,
            run_id=run_id,
            tags=tags,
            parent_run_id=parent_run_id,
            metadata=metadata,
            inputs=inputs,
            **kwargs,
        )
        tasks = [self._start_trace(tool_run), self._on_tool_start(tool_run)]
        await asyncio.gather(*tasks)

    @override
    async def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        tool_run = self._complete_tool_run(
            output=output,
            run_id=run_id,
        )
        tasks = [self._end_trace(tool_run), self._on_tool_end(tool_run)]
        await asyncio.gather(*tasks)

    @override
    async def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> None:
        tool_run = self._errored_tool_run(
            error=error,
            run_id=run_id,
        )
        tasks = [self._end_trace(tool_run), self._on_tool_error(tool_run)]
        await asyncio.gather(*tasks)

    @override
    async def on_retriever_start(
        self,
        serialized: dict[str, Any],
        query: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        retriever_run = self._create_retrieval_run(
            serialized=serialized,
            query=query,
            run_id=run_id,
            parent_run_id=parent_run_id,
            tags=tags,
            metadata=metadata,
            name=name,
        )
        tasks = [
            self._start_trace(retriever_run),
            self._on_retriever_start(retriever_run),
        ]
        await asyncio.gather(*tasks)

    @override
    async def on_retriever_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> None:
        retrieval_run = self._errored_retrieval_run(
            error=error,
            run_id=run_id,
        )
        tasks = [
            self._end_trace(retrieval_run),
            self._on_retriever_error(retrieval_run),
        ]
        await asyncio.gather(*tasks)

    @override
    async def on_retriever_end(
        self,
        documents: Sequence[Document],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> None:
        retrieval_run = self._complete_retrieval_run(
            documents=documents,
            run_id=run_id,
        )
        tasks = [self._end_trace(retrieval_run), self._on_retriever_end(retrieval_run)]
        await asyncio.gather(*tasks)

    async def _on_run_create(self, run: Run) -> None:
        """Process a run upon creation."""

    async def _on_run_update(self, run: Run) -> None:
        """Process a run upon update."""

    async def _on_llm_start(self, run: Run) -> None:
        """Process the LLM Run upon start."""

    async def _on_llm_end(self, run: Run) -> None:
        """Process the LLM Run."""

    async def _on_llm_error(self, run: Run) -> None:
        """Process the LLM Run upon error."""

    async def _on_llm_new_token(
        self,
        run: Run,
        token: str,
        chunk: Optional[Union[GenerationChunk, ChatGenerationChunk]],
    ) -> None:
        """Process new LLM token."""

    async def _on_chain_start(self, run: Run) -> None:
        """Process the Chain Run upon start."""

    async def _on_chain_end(self, run: Run) -> None:
        """Process the Chain Run."""

    async def _on_chain_error(self, run: Run) -> None:
        """Process the Chain Run upon error."""

    async def _on_tool_start(self, run: Run) -> None:
        """Process the Tool Run upon start."""

    async def _on_tool_end(self, run: Run) -> None:
        """Process the Tool Run."""

    async def _on_tool_error(self, run: Run) -> None:
        """Process the Tool Run upon error."""

    async def _on_chat_model_start(self, run: Run) -> None:
        """Process the Chat Model Run upon start."""

    async def _on_retriever_start(self, run: Run) -> None:
        """Process the Retriever Run upon start."""

    async def _on_retriever_end(self, run: Run) -> None:
        """Process the Retriever Run."""

    async def _on_retriever_error(self, run: Run) -> None:
        """Process the Retriever Run upon error."""
