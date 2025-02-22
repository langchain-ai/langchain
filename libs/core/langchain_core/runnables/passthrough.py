"""Implementation of the RunnablePassthrough."""

from __future__ import annotations

import asyncio
import inspect
import threading
from collections.abc import AsyncIterator, Awaitable, Iterator, Mapping
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Optional,
    Union,
    cast,
)

from pydantic import BaseModel, RootModel
from typing_extensions import override

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
from langchain_core.runnables.utils import (
    AddableDict,
    ConfigurableFieldSpec,
)
from langchain_core.utils.aiter import atee, py_anext
from langchain_core.utils.iter import safetee
from langchain_core.utils.pydantic import create_model_v2

if TYPE_CHECKING:
    from langchain_core.callbacks.manager import (
        AsyncCallbackManagerForChainRun,
        CallbackManagerForChainRun,
    )


def identity(x: Other) -> Other:
    """Identity function.

    Args:
        x (Other): input.

    Returns:
        Other: output.
    """
    return x


async def aidentity(x: Other) -> Other:
    """Async identity function.

    Args:
        x (Other): input.

    Returns:
        Other: output.
    """
    return x


class RunnablePassthrough(RunnableSerializable[Other, Other]):
    """Runnable to passthrough inputs unchanged or with additional keys.

    This Runnable behaves almost like the identity function, except that it
    can be configured to add additional keys to the output, if the input is a
    dict.

    The examples below demonstrate this Runnable works using a few simple
    chains. The chains rely on simple lambdas to make the examples easy to execute
    and experiment with.

    Parameters:
        func (Callable[[Other], None], optional): Function to be called with the input.
        afunc (Callable[[Other], Awaitable[None]], optional): Async function to
            be called with the input.
        input_type (Optional[Type[Other]], optional): Type of the input.
        **kwargs (Any): Additional keyword arguments.

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

            from langchain_core.runnables import RunnablePassthrough

            def fake_llm(prompt: str) -> str: # Fake LLM for the example
                return "completion"

            runnable = {
                'llm1':  fake_llm,
                'llm2':  fake_llm,
            } | RunnablePassthrough.assign(
                total_chars=lambda inputs: len(inputs['llm1'] + inputs['llm2'])
            )

            runnable.invoke('hello')
            # {'llm1': 'completion', 'llm2': 'completion', 'total_chars': 20}
    """

    input_type: Optional[type[Other]] = None

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
        input_type: Optional[type[Other]] = None,
        **kwargs: Any,
    ) -> None:
        if inspect.iscoroutinefunction(func):
            afunc = func
            func = None

        super().__init__(func=func, afunc=afunc, input_type=input_type, **kwargs)  # type: ignore[call-arg]

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return True

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "schema", "runnable"]

    @property
    @override
    def InputType(self) -> Any:
        return self.input_type or Any

    @property
    @override
    def OutputType(self) -> Any:
        return self.input_type or Any

    @classmethod
    def assign(
        cls,
        **kwargs: Union[
            Runnable[dict[str, Any], Any],
            Callable[[dict[str, Any]], Any],
            Mapping[
                str,
                Union[Runnable[dict[str, Any], Any], Callable[[dict[str, Any]], Any]],
            ],
        ],
    ) -> RunnableAssign:
        """Merge the Dict input with the output produced by the mapping argument.

        Args:
            **kwargs: Runnable, Callable or a Mapping from keys to Runnables
                or Callables.

        Returns:
            A Runnable that merges the Dict input with the output produced by the
            mapping argument.
        """
        return RunnableAssign(RunnableParallel[dict[str, Any]](kwargs))

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
            final: Other
            got_first_chunk = False

            for chunk in self._transform_stream_with_config(input, identity, config):
                yield chunk

                if not got_first_chunk:
                    final = chunk
                    got_first_chunk = True
                else:
                    try:
                        final = final + chunk  # type: ignore[operator]
                    except TypeError:
                        final = chunk

            if got_first_chunk:
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
            got_first_chunk = False

            async for chunk in self._atransform_stream_with_config(
                input, identity, config
            ):
                yield chunk

                # By definitions, a function will operate on the aggregated
                # input. So we'll aggregate the input until we get to the last
                # chunk.
                # If the input is not addable, then we'll assume that we can
                # only operate on the last chunk.
                if not got_first_chunk:
                    final = chunk
                    got_first_chunk = True
                else:
                    try:
                        final = final + chunk  # type: ignore[operator]
                    except TypeError:
                        final = chunk

            if got_first_chunk:
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


class RunnableAssign(RunnableSerializable[dict[str, Any], dict[str, Any]]):
    """Runnable that assigns key-value pairs to Dict[str, Any] inputs.

    The `RunnableAssign` class takes input dictionaries and, through a
    `RunnableParallel` instance, applies transformations, then combines
    these with the original data, introducing new key-value pairs based
    on the mapper's logic.

    Parameters:
        mapper (RunnableParallel[Dict[str, Any]]): A `RunnableParallel` instance
            that will be used to transform the input dictionary.

    Examples:
        .. code-block:: python

            # This is a RunnableAssign
            from typing import Dict
            from langchain_core.runnables.passthrough import (
                RunnableAssign,
                RunnableParallel,
            )
            from langchain_core.runnables.base import RunnableLambda

            def add_ten(x: Dict[str, int]) -> Dict[str, int]:
                return {"added": x["input"] + 10}

            mapper = RunnableParallel(
                {"add_step": RunnableLambda(add_ten),}
            )

            runnable_assign = RunnableAssign(mapper)

            # Synchronous example
            runnable_assign.invoke({"input": 5})
            # returns {'input': 5, 'add_step': {'added': 15}}

            # Asynchronous example
            await runnable_assign.ainvoke({"input": 5})
            # returns {'input': 5, 'add_step': {'added': 15}}
    """

    mapper: RunnableParallel

    def __init__(self, mapper: RunnableParallel[dict[str, Any]], **kwargs: Any) -> None:
        super().__init__(mapper=mapper, **kwargs)  # type: ignore[call-arg]

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return True

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "schema", "runnable"]

    def get_name(
        self, suffix: Optional[str] = None, *, name: Optional[str] = None
    ) -> str:
        name = (
            name
            or self.name
            or f"RunnableAssign<{','.join(self.mapper.steps__.keys())}>"
        )
        return super().get_name(suffix, name=name)

    def get_input_schema(
        self, config: Optional[RunnableConfig] = None
    ) -> type[BaseModel]:
        map_input_schema = self.mapper.get_input_schema(config)
        if not issubclass(map_input_schema, RootModel):
            # ie. it's a dict
            return map_input_schema

        return super().get_input_schema(config)

    def get_output_schema(
        self, config: Optional[RunnableConfig] = None
    ) -> type[BaseModel]:
        map_input_schema = self.mapper.get_input_schema(config)
        map_output_schema = self.mapper.get_output_schema(config)
        if not issubclass(map_input_schema, RootModel) and not issubclass(
            map_output_schema, RootModel
        ):
            fields = {}

            for name, field_info in map_input_schema.model_fields.items():
                fields[name] = (field_info.annotation, field_info.default)

            for name, field_info in map_output_schema.model_fields.items():
                fields[name] = (field_info.annotation, field_info.default)

            return create_model_v2(  # type: ignore[call-overload]
                "RunnableAssignOutput", field_definitions=fields
            )
        elif not issubclass(map_output_schema, RootModel):
            # ie. only map output is a dict
            # ie. input type is either unknown or inferred incorrectly
            return map_output_schema

        return super().get_output_schema(config)

    @property
    def config_specs(self) -> list[ConfigurableFieldSpec]:
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
        input: dict[str, Any],
        run_manager: CallbackManagerForChainRun,
        config: RunnableConfig,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if not isinstance(input, dict):
            msg = "The input to RunnablePassthrough.assign() must be a dict."
            raise ValueError(msg)  # noqa: TRY004

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
        input: dict[str, Any],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return self._call_with_config(self._invoke, input, config, **kwargs)

    async def _ainvoke(
        self,
        input: dict[str, Any],
        run_manager: AsyncCallbackManagerForChainRun,
        config: RunnableConfig,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if not isinstance(input, dict):
            msg = "The input to RunnablePassthrough.assign() must be a dict."
            raise ValueError(msg)  # noqa: TRY004

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
        input: dict[str, Any],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return await self._acall_with_config(self._ainvoke, input, config, **kwargs)

    def _transform(
        self,
        input: Iterator[dict[str, Any]],
        run_manager: CallbackManagerForChainRun,
        config: RunnableConfig,
        **kwargs: Any,
    ) -> Iterator[dict[str, Any]]:
        # collect mapper keys
        mapper_keys = set(self.mapper.steps__.keys())
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
                if not isinstance(chunk, dict):
                    msg = "The input to RunnablePassthrough.assign() must be a dict."
                    raise ValueError(msg)  # noqa: TRY004
                # remove mapper keys from passthrough chunk, to be overwritten by map
                filtered = AddableDict(
                    {k: v for k, v in chunk.items() if k not in mapper_keys}
                )
                if filtered:
                    yield filtered
            # yield map output
            yield cast(dict[str, Any], first_map_chunk_future.result())
            for chunk in map_output:
                yield chunk

    def transform(
        self,
        input: Iterator[dict[str, Any]],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any | None,
    ) -> Iterator[dict[str, Any]]:
        yield from self._transform_stream_with_config(
            input, self._transform, config, **kwargs
        )

    async def _atransform(
        self,
        input: AsyncIterator[dict[str, Any]],
        run_manager: AsyncCallbackManagerForChainRun,
        config: RunnableConfig,
        **kwargs: Any,
    ) -> AsyncIterator[dict[str, Any]]:
        # collect mapper keys
        mapper_keys = set(self.mapper.steps__.keys())
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
            if not isinstance(chunk, dict):
                msg = "The input to RunnablePassthrough.assign() must be a dict."
                raise ValueError(msg)  # noqa: TRY004

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
        input: AsyncIterator[dict[str, Any]],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> AsyncIterator[dict[str, Any]]:
        async for chunk in self._atransform_stream_with_config(
            input, self._atransform, config, **kwargs
        ):
            yield chunk

    def stream(
        self,
        input: dict[str, Any],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Iterator[dict[str, Any]]:
        return self.transform(iter([input]), config, **kwargs)

    async def astream(
        self,
        input: dict[str, Any],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> AsyncIterator[dict[str, Any]]:
        async def input_aiter() -> AsyncIterator[dict[str, Any]]:
            yield input

        async for chunk in self.atransform(input_aiter(), config, **kwargs):
            yield chunk


class RunnablePick(RunnableSerializable[dict[str, Any], dict[str, Any]]):
    """Runnable that picks keys from Dict[str, Any] inputs.

    RunnablePick class represents a Runnable that selectively picks keys from a
    dictionary input. It allows you to specify one or more keys to extract
    from the input dictionary. It returns a new dictionary containing only
    the selected keys.

    Parameters:
        keys (Union[str, List[str]]): A single key or a list of keys to pick from
            the input dictionary.

    Example :
        .. code-block:: python

            from langchain_core.runnables.passthrough import RunnablePick

            input_data = {
                'name': 'John',
                'age': 30,
                'city': 'New York',
                'country': 'USA'
            }

            runnable = RunnablePick(keys=['name', 'age'])

            output_data = runnable.invoke(input_data)

            print(output_data)  # Output: {'name': 'John', 'age': 30}
    """

    keys: Union[str, list[str]]

    def __init__(self, keys: Union[str, list[str]], **kwargs: Any) -> None:
        super().__init__(keys=keys, **kwargs)  # type: ignore[call-arg]

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return True

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
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

    def _pick(self, input: dict[str, Any]) -> Any:
        if not isinstance(input, dict):
            msg = "The input to RunnablePassthrough.assign() must be a dict."
            raise ValueError(msg)  # noqa: TRY004

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
        input: dict[str, Any],
    ) -> dict[str, Any]:
        return self._pick(input)

    def invoke(
        self,
        input: dict[str, Any],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return self._call_with_config(self._invoke, input, config, **kwargs)

    async def _ainvoke(
        self,
        input: dict[str, Any],
    ) -> dict[str, Any]:
        return self._pick(input)

    async def ainvoke(
        self,
        input: dict[str, Any],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return await self._acall_with_config(self._ainvoke, input, config, **kwargs)

    def _transform(
        self,
        input: Iterator[dict[str, Any]],
    ) -> Iterator[dict[str, Any]]:
        for chunk in input:
            picked = self._pick(chunk)
            if picked is not None:
                yield picked

    def transform(
        self,
        input: Iterator[dict[str, Any]],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Iterator[dict[str, Any]]:
        yield from self._transform_stream_with_config(
            input, self._transform, config, **kwargs
        )

    async def _atransform(
        self,
        input: AsyncIterator[dict[str, Any]],
    ) -> AsyncIterator[dict[str, Any]]:
        async for chunk in input:
            picked = self._pick(chunk)
            if picked is not None:
                yield picked

    async def atransform(
        self,
        input: AsyncIterator[dict[str, Any]],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> AsyncIterator[dict[str, Any]]:
        async for chunk in self._atransform_stream_with_config(
            input, self._atransform, config, **kwargs
        ):
            yield chunk

    def stream(
        self,
        input: dict[str, Any],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Iterator[dict[str, Any]]:
        return self.transform(iter([input]), config, **kwargs)

    async def astream(
        self,
        input: dict[str, Any],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> AsyncIterator[dict[str, Any]]:
        async def input_aiter() -> AsyncIterator[dict[str, Any]]:
            yield input

        async for chunk in self.atransform(input_aiter(), config, **kwargs):
            yield chunk
