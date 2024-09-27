from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import (
    TYPE_CHECKING,
    Any,
    Optional,
    Union,
)

from langchain_core.messages import BaseMessage, BaseMessageChunk
from langchain_core.output_parsers.base import BaseOutputParser, T
from langchain_core.outputs import (
    ChatGeneration,
    ChatGenerationChunk,
    Generation,
    GenerationChunk,
)
from langchain_core.runnables.config import run_in_executor

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig


class BaseTransformOutputParser(BaseOutputParser[T]):
    """Base class for an output parser that can handle streaming input."""

    def _transform(self, input: Iterator[Union[str, BaseMessage]]) -> Iterator[T]:
        for chunk in input:
            if isinstance(chunk, BaseMessage):
                yield self.parse_result([ChatGeneration(message=chunk)])
            else:
                yield self.parse_result([Generation(text=chunk)])

    async def _atransform(
        self, input: AsyncIterator[Union[str, BaseMessage]]
    ) -> AsyncIterator[T]:
        async for chunk in input:
            if isinstance(chunk, BaseMessage):
                yield await run_in_executor(
                    None, self.parse_result, [ChatGeneration(message=chunk)]
                )
            else:
                yield await run_in_executor(
                    None, self.parse_result, [Generation(text=chunk)]
                )

    def transform(
        self,
        input: Iterator[Union[str, BaseMessage]],
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

    async def atransform(
        self,
        input: AsyncIterator[Union[str, BaseMessage]],
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

    def _diff(self, prev: Optional[T], next: T) -> T:
        """Convert parsed outputs into a diff format. The semantics of this are
        up to the output parser.

        Args:
            prev: The previous parsed output.
            next: The current parsed output.

        Returns:
            The diff between the previous and current parsed output.
        """
        raise NotImplementedError()

    def _transform(self, input: Iterator[Union[str, BaseMessage]]) -> Iterator[Any]:
        prev_parsed = None
        acc_gen: Union[GenerationChunk, ChatGenerationChunk, None] = None
        for chunk in input:
            chunk_gen: Union[GenerationChunk, ChatGenerationChunk]
            if isinstance(chunk, BaseMessageChunk):
                chunk_gen = ChatGenerationChunk(message=chunk)
            elif isinstance(chunk, BaseMessage):
                chunk_gen = ChatGenerationChunk(
                    message=BaseMessageChunk(**chunk.dict())
                )
            else:
                chunk_gen = GenerationChunk(text=chunk)

            acc_gen = chunk_gen if acc_gen is None else acc_gen + chunk_gen  # type: ignore[operator]

            parsed = self.parse_result([acc_gen], partial=True)
            if parsed is not None and parsed != prev_parsed:
                if self.diff:
                    yield self._diff(prev_parsed, parsed)
                else:
                    yield parsed
                prev_parsed = parsed

    async def _atransform(
        self, input: AsyncIterator[Union[str, BaseMessage]]
    ) -> AsyncIterator[T]:
        prev_parsed = None
        acc_gen: Union[GenerationChunk, ChatGenerationChunk, None] = None
        async for chunk in input:
            chunk_gen: Union[GenerationChunk, ChatGenerationChunk]
            if isinstance(chunk, BaseMessageChunk):
                chunk_gen = ChatGenerationChunk(message=chunk)
            elif isinstance(chunk, BaseMessage):
                chunk_gen = ChatGenerationChunk(
                    message=BaseMessageChunk(**chunk.dict())
                )
            else:
                chunk_gen = GenerationChunk(text=chunk)

            acc_gen = chunk_gen if acc_gen is None else acc_gen + chunk_gen  # type: ignore[operator]

            parsed = await self.aparse_result([acc_gen], partial=True)
            if parsed is not None and parsed != prev_parsed:
                if self.diff:
                    yield await run_in_executor(None, self._diff, prev_parsed, parsed)
                else:
                    yield parsed
                prev_parsed = parsed
