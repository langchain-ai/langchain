"""Implementation of the RunnablePassthrough."""
from __future__ import annotations

import asyncio
import inspect
import threading
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Type,
    Union,
    cast,
)

from langchain_core.pydantic_v1 import BaseModel, create_model
from langchain_core.runnables.base import (
    Other,
    Runnable,
    RunnableParallel,
    RunnableSerializable,
)
from langchain_core.runnables.config import (
    RunnableConfig,
    acall_func_with_variable_args,
    call_func_with_variable_args,
    ensure_config,
    get_executor_for_config,
    patch_config,
)
from langchain_core.runnables.graph import Graph
from langchain_core.runnables.utils import AddableDict, ConfigurableFieldSpec
from langchain_core.utils.aiter import atee, py_anext
from langchain_core.utils.iter import safetee

if TYPE_CHECKING:
    from langchain_core.callbacks.manager import (
        AsyncCallbackManagerForChainRun,
        CallbackManagerForChainRun,
    )


def identity(x: Other) -> Other:
    """An identity function"""
    return x


async def aidentity(x: Other) -> Other:
    """An async identity function"""
    return x


class RunnablePassthrough(RunnableSerializable[Other, Other]):
    """A runnable to passthrough inputs unchanged or with additional keys.

    This runnable behaves almost like the identity function, except that it
    can be configured to add additional keys to the output, if the input is a
    dict.

    The examples below demonstrate this runnable works using a few simple
    chains. The chains rely on simple lambdas to make the examples easy to execute
    and experiment with.

    Examples:

        .. code-block:: python

            from langchain_core.runnables import (
                RunnableLambda,
                RunnableParallel,
                RunnablePassthrough,
            )

            runnable = RunnableParallel(
                origin=RunnablePassthrough(),
                modified=lambda x: x+1
            )

            runnable.invoke(1) # {'origin': 1, 'modified': 2}


            def fake_llm(prompt: str) -> str: # Fake LLM for the example
                return "completion"

            chain = RunnableLambda(fake_llm) | {
                'original': RunnablePassthrough(), # Original LLM output
                'parsed': lambda text: text[::-1] # Parsing logic
            }

            chain.invoke('hello') # {'original': 'completion', 'parsed': 'noitelpmoc'}

    In some cases, it may be useful to pass the input through while adding some
    keys to the output. In this case, you can use the `assign` method:

        .. code-block:: python

            from langchain_core.runnables import RunnablePassthrough, RunnableParallel

            def fake_llm(prompt: str) -> str: # Fake LLM for the example
                return "completion"

            runnable = {
                'llm1':  fake_llm,
                'llm2':  fake_llm,
            }
            | RunnablePassthrough.assign(
                total_chars=lambda inputs: len(inputs['llm1'] + inputs['llm2'])
              )

            runnable.invoke('hello')
            # {'llm1': 'completion', 'llm2': 'completion', 'total_chars': 20}
    """

    input_type: Optional[Type[Other]] = None

    func: Optional[
        Union[Callable[[Other], None], Callable[[Other, RunnableConfig], None]]
    ] = None

    afunc: Optional[
        Union[
            Callable[[Other], Awaitable[None]],
            Callable[[Other, RunnableConfig], Awaitable[None]],
        ]
    ] = None

    def __repr_args__(self) -> Any:
        # Without this repr(self) raises a RecursionError
        # See https://github.com/pydantic/pydantic/issues/7327
        return []

    def __init__(
        self,
        func: Optional[
            Union[
                Union[Callable[[Other], None], Callable[[Other, RunnableConfig], None]],
                Union[
                    Callable[[Other], Awaitable[None]],
                    Callable[[Other, RunnableConfig], Awaitable[None]],
                ],
            ]
        ] = None,
        afunc: Optional[
            Union[
                Callable[[Other], Awaitable[None]],
                Callable[[Other, RunnableConfig], Awaitable[None]],
            ]
        ] = None,
        *,
        input_type: Optional[Type[Other]] = None,
        **kwargs: Any,
    ) -> None:
        if inspect.iscoroutinefunction(func):
            afunc = func
            func = None

        super().__init__(func=func, afunc=afunc, input_type=input_type, **kwargs)

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return True

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "schema", "runnable"]

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
    ) -> "RunnableAssign":
        """Merge the Dict input with the output produced by the mapping argument.

        Args:
            mapping: A mapping from keys to runnables or callables.

        Returns:
            A runnable that merges the Dict input with the output produced by the
            mapping argument.
        """
        return RunnableAssign(RunnableParallel(kwargs))

    def invoke(
        self, input: Other, config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> Other:
        if self.func is not None:
            call_func_with_variable_args(
                self.func, input, ensure_config(config), **kwargs
            )
        return self._call_with_config(identity, input, config)

    async def ainvoke(
        self,
        input: Other,
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> Other:
        if self.afunc is not None:
            await acall_func_with_variable_args(
                self.afunc, input, ensure_config(config), **kwargs
            )
        elif self.func is not None:
            call_func_with_variable_args(
                self.func, input, ensure_config(config), **kwargs
            )
        return await self._acall_with_config(aidentity, input, config)

    def transform(
        self,
        input: Iterator[Other],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Iterator[Other]:
        if self.func is None:
            for chunk in self._transform_stream_with_config(input, identity, config):
                yield chunk
        else:
            final = None

            for chunk in self._transform_stream_with_config(input, identity, config):
                yield chunk
                if final is None:
                    final = chunk
                else:
                    final = final + chunk

            if final is not None:
                call_func_with_variable_args(
                    self.func, final, ensure_config(config), **kwargs
                )

    async def atransform(
        self,
        input: AsyncIterator[Other],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> AsyncIterator[Other]:
        if self.afunc is None and self.func is None:
            async for chunk in self._atransform_stream_with_config(
                input, identity, config
            ):
                yield chunk
        else:
            final = None

            async for chunk in self._atransform_stream_with_config(
                input, identity, config
            ):
                yield chunk
                if final is None:
                    final = chunk
                else:
                    final = final + chunk

            if final is not None:
                config = ensure_config(config)
                if self.afunc is not None:
                    await acall_func_with_variable_args(
                        self.afunc, final, config, **kwargs
                    )
                elif self.func is not None:
                    call_func_with_variable_args(self.func, final, config, **kwargs)

    def stream(
        self,
        input: Other,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Iterator[Other]:
        return self.transform(iter([input]), config, **kwargs)

    async def astream(
        self,
        input: Other,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> AsyncIterator[Other]:
        async def input_aiter() -> AsyncIterator[Other]:
            yield input

        async for chunk in self.atransform(input_aiter(), config, **kwargs):
            yield chunk


_graph_passthrough: RunnablePassthrough = RunnablePassthrough()


class RunnableAssign(RunnableSerializable[Dict[str, Any], Dict[str, Any]]):
    """
    A runnable that assigns key-value pairs to Dict[str, Any] inputs.
    """

    mapper: RunnableParallel[Dict[str, Any]]

    def __init__(self, mapper: RunnableParallel[Dict[str, Any]], **kwargs: Any) -> None:
        super().__init__(mapper=mapper, **kwargs)

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return True

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "schema", "runnable"]

    def get_name(
        self, suffix: Optional[str] = None, *, name: Optional[str] = None
    ) -> str:
        name = (
            name or self.name or f"RunnableAssign<{','.join(self.mapper.steps.keys())}>"
        )
        return super().get_name(suffix, name=name)

    def get_input_schema(
        self, config: Optional[RunnableConfig] = None
    ) -> Type[BaseModel]:
        map_input_schema = self.mapper.get_input_schema(config)
        if not map_input_schema.__custom_root_type__:
            # ie. it's a dict
            return map_input_schema

        return super().get_input_schema(config)

    def get_output_schema(
        self, config: Optional[RunnableConfig] = None
    ) -> Type[BaseModel]:
        map_input_schema = self.mapper.get_input_schema(config)
        map_output_schema = self.mapper.get_output_schema(config)
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
        elif not map_output_schema.__custom_root_type__:
            # ie. only map output is a dict
            # ie. input type is either unknown or inferred incorrectly
            return map_output_schema

        return super().get_output_schema(config)

    @property
    def config_specs(self) -> List[ConfigurableFieldSpec]:
        return self.mapper.config_specs

    def get_graph(self, config: RunnableConfig | None = None) -> Graph:
        # get graph from mapper
        graph = self.mapper.get_graph(config)
        # add passthrough node and edges
        input_node = graph.first_node()
        output_node = graph.last_node()
        if input_node is not None and output_node is not None:
            passthrough_node = graph.add_node(_graph_passthrough)
            graph.add_edge(input_node, passthrough_node)
            graph.add_edge(passthrough_node, output_node)
        return graph

    def _invoke(
        self,
        input: Dict[str, Any],
        run_manager: CallbackManagerForChainRun,
        config: RunnableConfig,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        assert isinstance(
            input, dict
        ), "The input to RunnablePassthrough.assign() must be a dict."

        return {
            **input,
            **self.mapper.invoke(
                input,
                patch_config(config, callbacks=run_manager.get_child()),
                **kwargs,
            ),
        }

    def invoke(
        self,
        input: Dict[str, Any],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return self._call_with_config(self._invoke, input, config, **kwargs)

    async def _ainvoke(
        self,
        input: Dict[str, Any],
        run_manager: AsyncCallbackManagerForChainRun,
        config: RunnableConfig,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        assert isinstance(
            input, dict
        ), "The input to RunnablePassthrough.assign() must be a dict."

        return {
            **input,
            **await self.mapper.ainvoke(
                input,
                patch_config(config, callbacks=run_manager.get_child()),
                **kwargs,
            ),
        }

    async def ainvoke(
        self,
        input: Dict[str, Any],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return await self._acall_with_config(self._ainvoke, input, config, **kwargs)

    def _transform(
        self,
        input: Iterator[Dict[str, Any]],
        run_manager: CallbackManagerForChainRun,
        config: RunnableConfig,
        **kwargs: Any,
    ) -> Iterator[Dict[str, Any]]:
        # collect mapper keys
        mapper_keys = set(self.mapper.steps.keys())
        # create two streams, one for the map and one for the passthrough
        for_passthrough, for_map = safetee(input, 2, lock=threading.Lock())

        # create map output stream
        map_output = self.mapper.transform(
            for_map,
            patch_config(
                config,
                callbacks=run_manager.get_child(),
            ),
            **kwargs,
        )

        # get executor to start map output stream in background
        with get_executor_for_config(config) as executor:
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

    def transform(
        self,
        input: Iterator[Dict[str, Any]],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any | None,
    ) -> Iterator[Dict[str, Any]]:
        yield from self._transform_stream_with_config(
            input, self._transform, config, **kwargs
        )

    async def _atransform(
        self,
        input: AsyncIterator[Dict[str, Any]],
        run_manager: AsyncCallbackManagerForChainRun,
        config: RunnableConfig,
        **kwargs: Any,
    ) -> AsyncIterator[Dict[str, Any]]:
        # collect mapper keys
        mapper_keys = set(self.mapper.steps.keys())
        # create two streams, one for the map and one for the passthrough
        for_passthrough, for_map = atee(input, 2, lock=asyncio.Lock())
        # create map output stream
        map_output = self.mapper.atransform(
            for_map,
            patch_config(
                config,
                callbacks=run_manager.get_child(),
            ),
            **kwargs,
        )
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

    async def atransform(
        self,
        input: AsyncIterator[Dict[str, Any]],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> AsyncIterator[Dict[str, Any]]:
        async for chunk in self._atransform_stream_with_config(
            input, self._atransform, config, **kwargs
        ):
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


class RunnablePick(RunnableSerializable[Dict[str, Any], Dict[str, Any]]):
    """
    A runnable that picks keys from Dict[str, Any] inputs.
    """

    keys: Union[str, List[str]]

    def __init__(self, keys: Union[str, List[str]], **kwargs: Any) -> None:
        super().__init__(keys=keys, **kwargs)

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return True

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "schema", "runnable"]

    def get_name(
        self, suffix: Optional[str] = None, *, name: Optional[str] = None
    ) -> str:
        name = (
            name
            or self.name
            or f"RunnablePick<{','.join([self.keys] if isinstance(self.keys, str) else self.keys)}>"  # noqa: E501
        )
        return super().get_name(suffix, name=name)

    def _pick(self, input: Dict[str, Any]) -> Any:
        assert isinstance(
            input, dict
        ), "The input to RunnablePassthrough.assign() must be a dict."

        if isinstance(self.keys, str):
            return input.get(self.keys)
        else:
            picked = {k: input.get(k) for k in self.keys if k in input}
            if picked:
                return AddableDict(picked)
            else:
                return None

    def _invoke(
        self,
        input: Dict[str, Any],
    ) -> Dict[str, Any]:
        return self._pick(input)

    def invoke(
        self,
        input: Dict[str, Any],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return self._call_with_config(self._invoke, input, config, **kwargs)

    async def _ainvoke(
        self,
        input: Dict[str, Any],
    ) -> Dict[str, Any]:
        return self._pick(input)

    async def ainvoke(
        self,
        input: Dict[str, Any],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return await self._acall_with_config(self._ainvoke, input, config, **kwargs)

    def _transform(
        self,
        input: Iterator[Dict[str, Any]],
    ) -> Iterator[Dict[str, Any]]:
        for chunk in input:
            picked = self._pick(chunk)
            if picked is not None:
                yield picked

    def transform(
        self,
        input: Iterator[Dict[str, Any]],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Iterator[Dict[str, Any]]:
        yield from self._transform_stream_with_config(
            input, self._transform, config, **kwargs
        )

    async def _atransform(
        self,
        input: AsyncIterator[Dict[str, Any]],
    ) -> AsyncIterator[Dict[str, Any]]:
        async for chunk in input:
            picked = self._pick(chunk)
            if picked is not None:
                yield picked

    async def atransform(
        self,
        input: AsyncIterator[Dict[str, Any]],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> AsyncIterator[Dict[str, Any]]:
        async for chunk in self._atransform_stream_with_config(
            input, self._atransform, config, **kwargs
        ):
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
