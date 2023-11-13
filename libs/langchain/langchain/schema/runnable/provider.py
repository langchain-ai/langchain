import asyncio
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Generic,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)

from langchain.load.dump import dumpd
from langchain.pydantic_v1 import BaseModel, PrivateAttr
from langchain.schema.runnable.base import (
    Other,
    Runnable,
    RunnableLike,
    RunnableParallel,
    RunnableSerializable,
    coerce_to_runnable,
)
from langchain.schema.runnable.config import (
    RunnableConfig,
    ensure_config,
    get_async_callback_manager_for_config,
    get_callback_manager_for_config,
    get_executor_for_config,
    patch_config,
)
from langchain.schema.runnable.utils import (
    ConfigurableFieldSpec,
    Input,
    Output,
    get_unique_config_specs,
)


class KeyValueContext(BaseModel):
    key_value_map: Dict[str, Any] = {}


class RunnableContextGetter(RunnableSerializable[Any, Output]):
    key: str
    context: KeyValueContext

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return True

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        return cls.__module__.split(".")[:-1]

    def invoke(self, input: Input, config: Optional[RunnableConfig] = None) -> Input:
        if self.key not in self.context.key_value_map:
            raise RuntimeError(f"Cannot find key {self.key} in context.")
        return self.context.key_value_map[self.key]


class RunnableContextSetter(RunnableSerializable[Other, Other]):
    key: str
    context: KeyValueContext

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return True

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        return cls.__module__.split(".")[:-1]

    def invoke(self, input: Other, config: Optional[RunnableConfig] = None) -> Other:
        self.context.key_value_map[self.key] = input
        return input


ContextProviderFactory = Callable[
    [Callable[[str], RunnableContextGetter], Callable[[str], RunnableContextSetter]],
    Runnable[Input, Output],
]


class RunnableContextProvider(RunnableSerializable[Input, Output]):
    chain_factory: ContextProviderFactory

    def __init__(self, chain_factory: ContextProviderFactory):
        super().__init__(chain_factory=chain_factory)

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return True

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        return cls.__module__.split(".")[:-1]

    @property
    def chain(self):
        context = KeyValueContext()

        def getter(key: str) -> RunnableContextGetter:
            return RunnableContextGetter(key=key, context=context)

        def setter(key: str) -> RunnableContextSetter:
            return RunnableContextSetter(key=key, context=context)

        return self.chain_factory(getter, setter)

    def invoke(self, input: Input, config: Optional[RunnableConfig] = None) -> Output:
        config = ensure_config(config)
        callback_manager = get_callback_manager_for_config(config)
        run_manager = callback_manager.on_chain_start(
            dumpd(self),
            input,
            name=config.get("run_name"),
        )

        try:
            result = self.chain.invoke(
                input,
                config=patch_config(
                    config,
                    callbacks=run_manager.get_child(),
                ),
            )
        except BaseException as e:
            run_manager.on_chain_error(e)
            raise e

        run_manager.on_chain_end(result)
        return result

    async def ainvoke(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Output:
        config = ensure_config(config)
        callback_manager = get_async_callback_manager_for_config(config)

        run_manager = await callback_manager.on_chain_start(
            dumpd(self), input, name=config.get("run_name")
        )

        try:
            result = await self.chain.ainvoke(
                input,
                config=patch_config(
                    config,
                    callbacks=run_manager.get_child(),
                ),
                **kwargs,
            )
        except BaseException as e:
            await run_manager.on_chain_error(e)
            raise e

        await run_manager.on_chain_end(result)
        return result

    def batch(
        self,
        inputs: List[Input],
        config: Optional[Union[RunnableConfig, List[RunnableConfig]]] = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Optional[Any],
    ) -> List[Output]:
        config = ensure_config(config)
        callback_manager = get_callback_manager_for_config(config)
        run_manager = callback_manager.on_chain_start(
            dumpd(self),
            input,
            name=config.get("run_name"),
        )

        chains = [self.chain for _ in inputs]

        try:
            with get_executor_for_config(config) as executor:
                futures = [
                    executor.submit(
                        chain.invoke,
                        inputs[idx],
                        config=patch_config(
                            config,
                            callbacks=run_manager.get_child(tag=f"chain:{idx + 1}"),
                        ),
                        **kwargs,
                    )
                    for idx, chain in enumerate(chains)
                ]

                results = [future.result() for future in futures]
        except BaseException as e:
            run_manager.on_chain_error(e)
            raise e

        run_manager.on_chain_end(results)
        return results

    async def abatch(
        self,
        inputs: List[Input],
        config: Optional[Union[RunnableConfig, List[RunnableConfig]]] = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Optional[Any],
    ) -> List[Output]:
        config = ensure_config(config)
        callback_manager = get_async_callback_manager_for_config(config)
        run_manager = await callback_manager.on_chain_start(
            dumpd(self), input, name=config.get("run_name")
        )

        chains = [self.chain for _ in inputs]

        try:
            results = await asyncio.gather(
                *(
                    chain.ainvoke(
                        inputs[idx],
                        config=patch_config(
                            config,
                            callbacks=run_manager.get_child(tag=f"chain:{idx + 1}"),
                        ),
                        **kwargs,
                    )
                    for idx, chain in enumerate(chains)
                )
            )
        except BaseException as e:
            await run_manager.on_chain_error(e)
            raise e

        await run_manager.on_chain_end(results)
        return list(results)


def context_provider(
    chain_factory: ContextProviderFactory,
) -> RunnableContextProvider:
    return RunnableContextProvider(chain_factory=chain_factory)
