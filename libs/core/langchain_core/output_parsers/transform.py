"""Base classes for output parsers that can handle streaming input."""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Optional,
    Union,
)

from typing_extensions import override

from langchain_core.messages import BaseMessage, BaseMessageChunk
from langchain_core.output_parsers.base import BaseOutputParser, T
from langchain_core.outputs import (
    ChatGeneration,
    ChatGenerationChunk,
    Generation,
    GenerationChunk,
)
from langchain_core.runnables.config import run_in_executor
from langchain_core.v1.messages import AIMessage, AIMessageChunk

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator

    from langchain_core.runnables import RunnableConfig


class BaseTransformOutputParser(BaseOutputParser[T]):
    """Base class for an output parser that can handle streaming input."""

    def _transform(
        self,
        input: Iterator[Union[str, BaseMessage, AIMessage]],
    ) -> Iterator[T]:
        for chunk in input:
            if isinstance(chunk, BaseMessage):
                yield self.parse_result([ChatGeneration(message=chunk)])
            elif isinstance(chunk, AIMessage):
                yield self.parse_result(chunk)
            else:
                yield self.parse_result([Generation(text=chunk)])

    async def _atransform(
        self,
        input: AsyncIterator[Union[str, BaseMessage, AIMessage]],
    ) -> AsyncIterator[T]:
        async for chunk in input:
            if isinstance(chunk, BaseMessage):
                yield await run_in_executor(
                    None, self.parse_result, [ChatGeneration(message=chunk)]
                )
            elif isinstance(chunk, AIMessage):
                yield await run_in_executor(None, self.parse_result, chunk)
            else:
                yield await run_in_executor(
                    None, self.parse_result, [Generation(text=chunk)]
                )

    @override
    def transform(
        self,
        input: Iterator[Union[str, BaseMessage, AIMessage]],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Iterator[T]:
        """Transform the input into the output format.

        Args:
            input: The input to transform.
            config: The configuration to use for the transformation.
            kwargs: Additional keyword arguments.

        Yields:
            The transformed output.
        """
        yield from self._transform_stream_with_config(
            input, self._transform, config, run_type="parser"
        )

    @override
    async def atransform(
        self,
        input: AsyncIterator[Union[str, BaseMessage, AIMessage]],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> AsyncIterator[T]:
        """Async transform the input into the output format.

        Args:
            input: The input to transform.
            config: The configuration to use for the transformation.
            kwargs: Additional keyword arguments.

        Yields:
            The transformed output.
        """
        async for chunk in self._atransform_stream_with_config(
            input, self._atransform, config, run_type="parser"
        ):
            yield chunk


class BaseCumulativeTransformOutputParser(BaseTransformOutputParser[T]):
    """Base class for an output parser that can handle streaming input."""

    diff: bool = False
    """In streaming mode, whether to yield diffs between the previous and current
    parsed output, or just the current parsed output.
    """

    def _diff(
        self,
        prev: Optional[T],
        next: T,  # noqa: A002
    ) -> T:
        """Convert parsed outputs into a diff format.

        The semantics of this are up to the output parser.

        Args:
            prev: The previous parsed output.
            next: The current parsed output.

        Returns:
            The diff between the previous and current parsed output.
        """
        raise NotImplementedError

    @override
    def _transform(
        self, input: Iterator[Union[str, BaseMessage, AIMessage]]
    ) -> Iterator[Any]:
        prev_parsed = None
        acc_gen: Union[GenerationChunk, ChatGenerationChunk, AIMessageChunk, None] = (
            None
        )
        for chunk in input:
            chunk_gen: Union[GenerationChunk, ChatGenerationChunk, AIMessageChunk]
            if isinstance(chunk, BaseMessageChunk):
                chunk_gen = ChatGenerationChunk(message=chunk)
            elif isinstance(chunk, BaseMessage):
                chunk_gen = ChatGenerationChunk(
                    message=BaseMessageChunk(**chunk.model_dump())
                )
            elif isinstance(chunk, AIMessageChunk):
                chunk_gen = chunk
            elif isinstance(chunk, AIMessage):
                chunk_gen = AIMessageChunk(
                    content=chunk.content,
                    id=chunk.id,
                    name=chunk.name,
                    lc_version=chunk.lc_version,
                    response_metadata=chunk.response_metadata,
                    usage_metadata=chunk.usage_metadata,
                    parsed=chunk.parsed,
                )
            else:
                chunk_gen = GenerationChunk(text=chunk)

            acc_gen = chunk_gen if acc_gen is None else acc_gen + chunk_gen  # type: ignore[operator]

            if isinstance(acc_gen, AIMessageChunk):
                parsed = self.parse_result(acc_gen, partial=True)
            else:
                parsed = self.parse_result([acc_gen], partial=True)
            if parsed is not None and parsed != prev_parsed:
                if self.diff:
                    yield self._diff(prev_parsed, parsed)
                else:
                    yield parsed
                prev_parsed = parsed

    @override
    async def _atransform(
        self, input: AsyncIterator[Union[str, BaseMessage, AIMessage]]
    ) -> AsyncIterator[T]:
        prev_parsed = None
        acc_gen: Union[GenerationChunk, ChatGenerationChunk, AIMessageChunk, None] = (
            None
        )
        async for chunk in input:
            chunk_gen: Union[GenerationChunk, ChatGenerationChunk, AIMessageChunk]
            if isinstance(chunk, BaseMessageChunk):
                chunk_gen = ChatGenerationChunk(message=chunk)
            elif isinstance(chunk, BaseMessage):
                chunk_gen = ChatGenerationChunk(
                    message=BaseMessageChunk(**chunk.model_dump())
                )
            elif isinstance(chunk, AIMessageChunk):
                chunk_gen = chunk
            elif isinstance(chunk, AIMessage):
                chunk_gen = AIMessageChunk(
                    content=chunk.content,
                    id=chunk.id,
                    name=chunk.name,
                    lc_version=chunk.lc_version,
                    response_metadata=chunk.response_metadata,
                    usage_metadata=chunk.usage_metadata,
                    parsed=chunk.parsed,
                )
            else:
                chunk_gen = GenerationChunk(text=chunk)

            acc_gen = chunk_gen if acc_gen is None else acc_gen + chunk_gen  # type: ignore[operator]

            if isinstance(acc_gen, AIMessageChunk):
                parsed = await self.aparse_result(acc_gen, partial=True)
            else:
                parsed = await self.aparse_result([acc_gen], partial=True)
            if parsed is not None and parsed != prev_parsed:
                if self.diff:
                    yield await run_in_executor(None, self._diff, prev_parsed, parsed)
                else:
                    yield parsed
                prev_parsed = parsed
