from __future__ import annotations

from typing import Any, AsyncIterator, Iterator, List, Optional

from langchain.load.serializable import Serializable
from langchain.schema.runnable.base import Input, Runnable
from langchain.schema.runnable.config import RunnableConfig


def identity(x: Input) -> Input:
    return x


async def aidentity(x: Input) -> Input:
    return x


class RunnablePassthrough(Serializable, Runnable[Input, Input]):
    """
    A runnable that passes through the input.
    """

    @property
    def lc_serializable(self) -> bool:
        return True

    @property
    def lc_namespace(self) -> List[str]:
        return self.__class__.__module__.split(".")[:-1]

    def invoke(self, input: Input, config: Optional[RunnableConfig] = None) -> Input:
        return self._call_with_config(identity, input, config)

    async def ainvoke(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> Input:
        return await self._acall_with_config(aidentity, input, config)

    def transform(
        self,
        input: Iterator[Input],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Iterator[Input]:
        return self._transform_stream_with_config(input, identity, config)

    async def atransform(
        self,
        input: AsyncIterator[Input],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> AsyncIterator[Input]:
        async for chunk in self._atransform_stream_with_config(input, identity, config):
            yield chunk
