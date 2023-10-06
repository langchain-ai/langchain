from __future__ import annotations

import asyncio
import threading
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Type,
    Union,
    cast,
)

from langchain.pydantic_v1 import BaseModel, create_model
from langchain.schema.runnable.base import (
    Input,
    Runnable,
    RunnableMap,
    RunnableSerializable,
)
from langchain.schema.runnable.config import RunnableConfig, get_executor_for_config
from langchain.schema.runnable.utils import AddableDict, ConfigurableFieldSpec
from langchain.utils.aiter import atee, py_anext
from langchain.utils.iter import safetee


def identity(x: Input) -> Input:
    return x


async def aidentity(x: Input) -> Input:
    return x


class RunnablePassthrough(RunnableSerializable[Input, Input]):
    """
    A runnable that passes through the input.
    """

    input_type: Optional[Type[Input]] = None

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return True

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        return cls.__module__.split(".")[:-1]

    @property
    def InputType(self) -> Any:
        return self.input_type or Any

    @property
    def OutputType(self) -> Any:
        return self.input_type or Any

    @classmethod
    def assign(
        cls,
        **kwargs: Union[
            Runnable[Dict[str, Any], Any],
            Callable[[Dict[str, Any]], Any],
            Mapping[
                str,
                Union[Runnable[Dict[str, Any], Any], Callable[[Dict[str, Any]], Any]],
            ],
        ],
    ) -> RunnableAssign:
        """
        Merge the Dict input with the output produced by the mapping argument.

        Args:
            mapping: A mapping from keys to runnables or callables.

        Returns:
            A runnable that merges the Dict input with the output produced by the
            mapping argument.
        """
        return RunnableAssign(RunnableMap(kwargs))

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


class RunnableAssign(RunnableSerializable[Dict[str, Any], Dict[str, Any]]):
    """
    A runnable that assigns key-value pairs to Dict[str, Any] inputs.
    """

    mapper: RunnableMap[Dict[str, Any]]

    def __init__(self, mapper: RunnableMap[Dict[str, Any]], **kwargs: Any) -> None:
        super().__init__(mapper=mapper, **kwargs)

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return True

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        return cls.__module__.split(".")[:-1]

    @property
    def input_schema(self) -> Type[BaseModel]:
        map_input_schema = self.mapper.input_schema
        if not map_input_schema.__custom_root_type__:
            # ie. it's a dict
            return map_input_schema

        return super().input_schema

    @property
    def output_schema(self) -> Type[BaseModel]:
        map_input_schema = self.mapper.input_schema
        map_output_schema = self.mapper.output_schema
        if (
            not map_input_schema.__custom_root_type__
            and not map_output_schema.__custom_root_type__
        ):
            # ie. both are dicts
            return create_model(  # type: ignore[call-overload]
                "RunnableAssignOutput",
                **{
                    k: (v.type_, v.default)
                    for s in (map_input_schema, map_output_schema)
                    for k, v in s.__fields__.items()
                },
            )

        return super().output_schema

    @property
    def config_specs(self) -> Sequence[ConfigurableFieldSpec]:
        return self.mapper.config_specs

    def invoke(
        self,
        input: Dict[str, Any],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        assert isinstance(
            input, dict
        ), "The input to RunnablePassthrough.assign() must be a dict."
        return {
            **input,
            **self.mapper.invoke(input, config, **kwargs),
        }

    async def ainvoke(
        self,
        input: Dict[str, Any],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        assert isinstance(
            input, dict
        ), "The input to RunnablePassthrough.assign() must be a dict."
        return {
            **input,
            **await self.mapper.ainvoke(input, config, **kwargs),
        }

    def transform(
        self,
        input: Iterator[Dict[str, Any]],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Iterator[Dict[str, Any]]:
        # collect mapper keys
        mapper_keys = set(self.mapper.steps.keys())
        # create two streams, one for the map and one for the passthrough
        for_passthrough, for_map = safetee(input, 2, lock=threading.Lock())
        # create map output stream
        map_output = self.mapper.transform(for_map, config, **kwargs)
        # get executor to start map output stream in background
        with get_executor_for_config(config or {}) as executor:
            # start map output stream
            first_map_chunk_future = executor.submit(
                next,
                map_output,  # type: ignore
                None,
            )
            # consume passthrough stream
            for chunk in for_passthrough:
                assert isinstance(
                    chunk, dict
                ), "The input to RunnablePassthrough.assign() must be a dict."
                # remove mapper keys from passthrough chunk, to be overwritten by map
                filtered = AddableDict(
                    {k: v for k, v in chunk.items() if k not in mapper_keys}
                )
                if filtered:
                    yield filtered
            # yield map output
            yield cast(Dict[str, Any], first_map_chunk_future.result())
            for chunk in map_output:
                yield chunk

    async def atransform(
        self,
        input: AsyncIterator[Dict[str, Any]],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> AsyncIterator[Dict[str, Any]]:
        # collect mapper keys
        mapper_keys = set(self.mapper.steps.keys())
        # create two streams, one for the map and one for the passthrough
        for_passthrough, for_map = atee(input, 2, lock=asyncio.Lock())
        # create map output stream
        map_output = self.mapper.atransform(for_map, config, **kwargs)
        # start map output stream
        first_map_chunk_task: asyncio.Task = asyncio.create_task(
            py_anext(map_output, None),  # type: ignore[arg-type]
        )
        # consume passthrough stream
        async for chunk in for_passthrough:
            assert isinstance(
                chunk, dict
            ), "The input to RunnablePassthrough.assign() must be a dict."
            # remove mapper keys from passthrough chunk, to be overwritten by map output
            filtered = AddableDict(
                {k: v for k, v in chunk.items() if k not in mapper_keys}
            )
            if filtered:
                yield filtered
        # yield map output
        yield await first_map_chunk_task
        async for chunk in map_output:
            yield chunk

    def stream(
        self,
        input: Dict[str, Any],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Iterator[Dict[str, Any]]:
        return self.transform(iter([input]), config, **kwargs)

    async def astream(
        self,
        input: Dict[str, Any],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> AsyncIterator[Dict[str, Any]]:
        async def input_aiter() -> AsyncIterator[Dict[str, Any]]:
            yield input

        async for chunk in self.atransform(input_aiter(), config, **kwargs):
            yield chunk
