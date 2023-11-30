import asyncio
import threading
from collections import defaultdict
from functools import partial
from itertools import groupby
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Type,
    TypeVar,
    Union,
)

from langchain_core.runnables.base import (
    Runnable,
    RunnableSerializable,
    coerce_to_runnable,
)
from langchain_core.runnables.config import RunnableConfig, patch_config
from langchain_core.runnables.utils import ConfigurableFieldSpec, Input, Output

T = TypeVar("T")
Values = Dict[asyncio.Event, Any]
CONTEXT_CONFIG_PREFIX = "__context__/"
CONTEXT_CONFIG_SUFFIX_GET = "/get"
CONTEXT_CONFIG_SUFFIX_SET = "/set"


async def _asetter(done: asyncio.Event, values: Values, value: T) -> T:
    values[done] = value
    done.set()
    return value


async def _agetter(done: asyncio.Event, values: Values) -> T:
    await done.wait()
    return values[done]


def _setter(done: threading.Event, values: Values, value: T) -> T:
    values[done] = value
    done.set()
    return value


def _getter(done: threading.Event, values: Values) -> T:
    done.wait()
    return values[done]


def _config_with_context(
    config: RunnableConfig,
    specs: List[ConfigurableFieldSpec],
    setter: Callable,
    getter: Callable,
    event: Union[Type[threading.Event], Type[asyncio.Event]],
) -> RunnableConfig:
    if any(k.startswith(CONTEXT_CONFIG_PREFIX) for k in config.get("configurable", {})):
        return config

    context_specs = [
        spec for spec in specs if spec.id.startswith(CONTEXT_CONFIG_PREFIX)
    ]
    grouped_by_key = groupby(
        sorted(context_specs, key=lambda s: s.id), key=lambda s: s.id.split("/")[1]
    )

    values: Values = {}
    events = defaultdict(event)
    context_funcs: Dict[str, Callable[[], Any]] = {}
    for key, group in grouped_by_key:
        group = list(group)
        getters = [s for s in group if s.id.endswith(CONTEXT_CONFIG_SUFFIX_GET)]
        setters = [s for s in group if s.id.endswith(CONTEXT_CONFIG_SUFFIX_SET)]

        if len(getters) < 1:
            raise KeyError(f"Expected at least one getter for context key {key}")
        if len(setters) != 1:
            raise KeyError(f"Expected exactly one setter for context key {key}")

        context_funcs[getters[0].id] = partial(getter, events[key], values)
        context_funcs[setters[0].id] = partial(setter, events[key], values)

    return patch_config(config, configurable=context_funcs)


def aconfig_with_context(
    config: RunnableConfig, specs: List[ConfigurableFieldSpec]
) -> RunnableConfig:
    return _config_with_context(config, specs, _asetter, _agetter, asyncio.Event)


def config_with_context(
    config: RunnableConfig, specs: List[ConfigurableFieldSpec]
) -> RunnableConfig:
    return _config_with_context(config, specs, _setter, _getter, threading.Event)


class ContextGet(RunnableSerializable):
    key: Union[str, List[str]]

    def __init__(self, key: Union[str, List[str]]):
        super().__init__(key=key)

    @property
    def ids(self) -> List[str]:
        keys = self.key if isinstance(self.key, list) else [self.key]
        return [f"{CONTEXT_CONFIG_PREFIX}{k}{CONTEXT_CONFIG_SUFFIX_GET}" for k in keys]

    @property
    def config_specs(self) -> List[ConfigurableFieldSpec]:
        return super().config_specs + [
            ConfigurableFieldSpec(
                id=id_,
                annotation=Callable[[], Any],
            )
            for id_ in self.ids
        ]

    def invoke(self, input: Any, config: Optional[RunnableConfig] = None) -> Any:
        config = config or {}
        configurable = config.get("configurable", {})
        if isinstance(self.key, list):
            return {
                key: configurable.get(id_)() for key, id_ in zip(self.key, self.ids)
            }
        else:
            return configurable.get(self.ids[0])()

    async def ainvoke(
        self, input: Any, config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> Any:
        config = config or {}
        configurable = config.get("configurable", {})
        if isinstance(self.key, list):
            values = await asyncio.gather(
                *(configurable.get(id_)() for id_ in self.ids)
            )
            return {key: value for key, value in zip(self.key, values)}
        else:
            return await config.get("configurable", {}).get(self.ids[0])()


SetValue = Union[
    Runnable[Input, Output],
    Callable[[Input], Output],
    Callable[[Input], Awaitable[Output]],
    Any,
]


def _coerce_set_value(value: SetValue) -> Runnable[Input, Output]:
    if not isinstance(value, Runnable) and not callable(value):
        return coerce_to_runnable(lambda _: value)
    return coerce_to_runnable(value)


class ContextSet(RunnableSerializable):
    keys: Mapping[str, Optional[Runnable]]

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        key: Optional[str] = None,
        value: Optional[SetValue] = None,
        **kwargs: SetValue,
    ):
        if key is not None:
            kwargs[key] = value
        super().__init__(
            keys={
                k: _coerce_set_value(v) if v is not None else None
                for k, v in kwargs.items()
            }
        )

    @property
    def ids(self) -> List[str]:
        return [
            f"{CONTEXT_CONFIG_PREFIX}{key}{CONTEXT_CONFIG_SUFFIX_SET}"
            for key in self.keys
        ]

    @property
    def config_specs(self) -> List[ConfigurableFieldSpec]:
        mapper_config_specs = [
            s
            for mapper in self.keys.values()
            if mapper is not None
            for s in mapper.config_specs
        ]
        for spec in mapper_config_specs:
            if spec.id.endswith(CONTEXT_CONFIG_SUFFIX_GET):
                getter_key = spec.id.split("/")[1]
                if getter_key in self.keys:
                    raise ValueError(
                        f"Circular reference in context setter for key {getter_key}"
                    )
        return super().config_specs + [
            ConfigurableFieldSpec(
                id=id_,
                annotation=Callable[[], Any],
            )
            for id_ in self.ids
        ]

    def invoke(self, input: Any, config: Optional[RunnableConfig] = None) -> Any:
        config = config or {}
        configurable = config.get("configurable", {})
        for id_, mapper in zip(self.ids, self.keys.values()):
            if mapper is not None:
                configurable[id_](mapper.invoke(input, config))
            else:
                configurable[id_](input)
        return input

    async def ainvoke(
        self, input: Any, config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> Any:
        config = config or {}
        configurable = config.get("configurable", {})
        for id_, mapper in zip(self.ids, self.keys.values()):
            if mapper is not None:
                await configurable[id_](await mapper.ainvoke(input, config))
            else:
                await configurable[id_](input)
        return input
