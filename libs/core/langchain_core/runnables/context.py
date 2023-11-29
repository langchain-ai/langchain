import asyncio
import threading
from collections import defaultdict
from functools import partial
from itertools import groupby
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union

from langchain_core.runnables.base import RunnableSerializable
from langchain_core.runnables.config import RunnableConfig, patch_config
from langchain_core.runnables.utils import ConfigurableFieldSpec

T = TypeVar("T")
Values = Dict[asyncio.Event, Any]
CONTEXT_CONFIG_PREFIX = "__context__/"


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
        getters = [s for s in group if s.id.endswith("/get")]
        setters = [s for s in group if s.id.endswith("/set")]

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
    key: str

    def __init__(self, key: str):
        super().__init__(key=key)

    @property
    def id(self) -> str:
        return f"{CONTEXT_CONFIG_PREFIX}{self.key}/get"

    @property
    def config_specs(self) -> List[ConfigurableFieldSpec]:
        return super().config_specs + [
            ConfigurableFieldSpec(
                id=self.id,
                annotation=Callable[[], Any],
            )
        ]

    def invoke(self, input: Any, config: Optional[RunnableConfig] = None) -> Any:
        config = config or {}
        return config.get("configurable", {}).get(self.id)()

    async def ainvoke(
        self, input: Any, config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> Any:
        config = config or {}
        return await config.get("configurable", {}).get(self.id)()


class ContextSet(RunnableSerializable):
    key: str

    def __init__(self, key: str):
        super().__init__(key=key)

    @property
    def id(self) -> str:
        return f"{CONTEXT_CONFIG_PREFIX}{self.key}/set"

    @property
    def config_specs(self) -> List[ConfigurableFieldSpec]:
        return super().config_specs + [
            ConfigurableFieldSpec(
                id=self.id,
                annotation=Callable[[], Any],
            )
        ]

    def invoke(self, input: Any, config: Optional[RunnableConfig] = None) -> Any:
        config = config or {}
        return config.get("configurable", {}).get(self.id)(input)

    async def ainvoke(
        self, input: Any, config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> Any:
        config = config or {}
        return await config.get("configurable", {}).get(self.id)(input)
