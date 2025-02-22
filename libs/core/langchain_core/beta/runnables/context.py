import asyncio
import threading
from collections import defaultdict
from collections.abc import Awaitable, Mapping, Sequence
from functools import partial
from itertools import groupby
from typing import (
    Any,
    Callable,
    Optional,
    TypeVar,
    Union,
)

from pydantic import ConfigDict

from langchain_core._api.beta_decorator import beta
from langchain_core.runnables.base import (
    Runnable,
    RunnableSerializable,
    coerce_to_runnable,
)
from langchain_core.runnables.config import RunnableConfig, ensure_config, patch_config
from langchain_core.runnables.utils import ConfigurableFieldSpec, Input, Output

T = TypeVar("T")
Values = dict[Union[asyncio.Event, threading.Event], Any]
CONTEXT_CONFIG_PREFIX = "__context__/"
CONTEXT_CONFIG_SUFFIX_GET = "/get"
CONTEXT_CONFIG_SUFFIX_SET = "/set"


async def _asetter(done: asyncio.Event, values: Values, value: T) -> T:
    values[done] = value
    done.set()
    return value


async def _agetter(done: asyncio.Event, values: Values) -> Any:
    await done.wait()
    return values[done]


def _setter(done: threading.Event, values: Values, value: T) -> T:
    values[done] = value
    done.set()
    return value


def _getter(done: threading.Event, values: Values) -> Any:
    done.wait()
    return values[done]


def _key_from_id(id_: str) -> str:
    wout_prefix = id_.split(CONTEXT_CONFIG_PREFIX, maxsplit=1)[1]
    if wout_prefix.endswith(CONTEXT_CONFIG_SUFFIX_GET):
        return wout_prefix[: -len(CONTEXT_CONFIG_SUFFIX_GET)]
    elif wout_prefix.endswith(CONTEXT_CONFIG_SUFFIX_SET):
        return wout_prefix[: -len(CONTEXT_CONFIG_SUFFIX_SET)]
    else:
        msg = f"Invalid context config id {id_}"
        raise ValueError(msg)


def _config_with_context(
    config: RunnableConfig,
    steps: list[Runnable],
    setter: Callable,
    getter: Callable,
    event_cls: Union[type[threading.Event], type[asyncio.Event]],
) -> RunnableConfig:
    if any(k.startswith(CONTEXT_CONFIG_PREFIX) for k in config.get("configurable", {})):
        return config

    context_specs = [
        (spec, i)
        for i, step in enumerate(steps)
        for spec in step.config_specs
        if spec.id.startswith(CONTEXT_CONFIG_PREFIX)
    ]
    grouped_by_key = {
        key: list(group)
        for key, group in groupby(
            sorted(context_specs, key=lambda s: s[0].id),
            key=lambda s: _key_from_id(s[0].id),
        )
    }
    deps_by_key = {
        key: {
            _key_from_id(dep) for spec in group for dep in (spec[0].dependencies or [])
        }
        for key, group in grouped_by_key.items()
    }

    values: Values = {}
    events: defaultdict[str, Union[asyncio.Event, threading.Event]] = defaultdict(
        event_cls
    )
    context_funcs: dict[str, Callable[[], Any]] = {}
    for key, group in grouped_by_key.items():
        getters = [s for s in group if s[0].id.endswith(CONTEXT_CONFIG_SUFFIX_GET)]
        setters = [s for s in group if s[0].id.endswith(CONTEXT_CONFIG_SUFFIX_SET)]

        for dep in deps_by_key[key]:
            if key in deps_by_key[dep]:
                msg = f"Deadlock detected between context keys {key} and {dep}"
                raise ValueError(msg)
        if len(setters) != 1:
            msg = f"Expected exactly one setter for context key {key}"
            raise ValueError(msg)
        setter_idx = setters[0][1]
        if any(getter_idx < setter_idx for _, getter_idx in getters):
            msg = f"Context setter for key {key} must be defined after all getters."
            raise ValueError(msg)

        if getters:
            context_funcs[getters[0][0].id] = partial(getter, events[key], values)
        context_funcs[setters[0][0].id] = partial(setter, events[key], values)

    return patch_config(config, configurable=context_funcs)


def aconfig_with_context(
    config: RunnableConfig,
    steps: list[Runnable],
) -> RunnableConfig:
    """Asynchronously patch a runnable config with context getters and setters.

    Args:
        config: The runnable config.
        steps: The runnable steps.

    Returns:
        The patched runnable config.
    """
    return _config_with_context(config, steps, _asetter, _agetter, asyncio.Event)


def config_with_context(
    config: RunnableConfig,
    steps: list[Runnable],
) -> RunnableConfig:
    """Patch a runnable config with context getters and setters.

    Args:
        config: The runnable config.
        steps: The runnable steps.

    Returns:
        The patched runnable config.
    """
    return _config_with_context(config, steps, _setter, _getter, threading.Event)


@beta()
class ContextGet(RunnableSerializable):
    """Get a context value."""

    prefix: str = ""

    key: Union[str, list[str]]

    def __str__(self) -> str:
        return f"ContextGet({_print_keys(self.key)})"

    @property
    def ids(self) -> list[str]:
        prefix = self.prefix + "/" if self.prefix else ""
        keys = self.key if isinstance(self.key, list) else [self.key]
        return [
            f"{CONTEXT_CONFIG_PREFIX}{prefix}{k}{CONTEXT_CONFIG_SUFFIX_GET}"
            for k in keys
        ]

    @property
    def config_specs(self) -> list[ConfigurableFieldSpec]:
        return super().config_specs + [
            ConfigurableFieldSpec(
                id=id_,
                annotation=Callable[[], Any],
            )
            for id_ in self.ids
        ]

    def invoke(
        self, input: Any, config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> Any:
        config = ensure_config(config)
        configurable = config.get("configurable", {})
        if isinstance(self.key, list):
            return {key: configurable[id_]() for key, id_ in zip(self.key, self.ids)}
        else:
            return configurable[self.ids[0]]()

    async def ainvoke(
        self, input: Any, config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> Any:
        config = ensure_config(config)
        configurable = config.get("configurable", {})
        if isinstance(self.key, list):
            values = await asyncio.gather(*(configurable[id_]() for id_ in self.ids))
            return dict(zip(self.key, values))
        else:
            return await configurable[self.ids[0]]()


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


@beta()
class ContextSet(RunnableSerializable):
    """Set a context value."""

    prefix: str = ""

    keys: Mapping[str, Optional[Runnable]]

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    def __init__(
        self,
        key: Optional[str] = None,
        value: Optional[SetValue] = None,
        prefix: str = "",
        **kwargs: SetValue,
    ):
        if key is not None:
            kwargs[key] = value
        super().__init__(  # type: ignore[call-arg]
            keys={
                k: _coerce_set_value(v) if v is not None else None
                for k, v in kwargs.items()
            },
            prefix=prefix,
        )

    def __str__(self) -> str:
        return f"ContextSet({_print_keys(list(self.keys.keys()))})"

    @property
    def ids(self) -> list[str]:
        prefix = self.prefix + "/" if self.prefix else ""
        return [
            f"{CONTEXT_CONFIG_PREFIX}{prefix}{key}{CONTEXT_CONFIG_SUFFIX_SET}"
            for key in self.keys
        ]

    @property
    def config_specs(self) -> list[ConfigurableFieldSpec]:
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
                    msg = f"Circular reference in context setter for key {getter_key}"
                    raise ValueError(msg)
        return super().config_specs + [
            ConfigurableFieldSpec(
                id=id_,
                annotation=Callable[[], Any],
            )
            for id_ in self.ids
        ]

    def invoke(
        self, input: Any, config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> Any:
        config = ensure_config(config)
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
        config = ensure_config(config)
        configurable = config.get("configurable", {})
        for id_, mapper in zip(self.ids, self.keys.values()):
            if mapper is not None:
                await configurable[id_](await mapper.ainvoke(input, config))
            else:
                await configurable[id_](input)
        return input


class Context:
    """Context for a runnable.

    The `Context` class provides methods for creating context scopes,
    getters, and setters within a runnable. It allows for managing
    and accessing contextual information throughout the execution
    of a program.

    Example:
        .. code-block:: python

            from langchain_core.beta.runnables.context import Context
            from langchain_core.runnables.passthrough import RunnablePassthrough
            from langchain_core.prompts.prompt import PromptTemplate
            from langchain_core.output_parsers.string import StrOutputParser
            from tests.unit_tests.fake.llm import FakeListLLM

            chain = (
                Context.setter("input")
                | {
                    "context": RunnablePassthrough()
                            | Context.setter("context"),
                    "question": RunnablePassthrough(),
                }
                | PromptTemplate.from_template("{context} {question}")
                | FakeListLLM(responses=["hello"])
                | StrOutputParser()
                | {
                    "result": RunnablePassthrough(),
                    "context": Context.getter("context"),
                    "input": Context.getter("input"),
                }
            )

            # Use the chain
            output = chain.invoke("What's your name?")
            print(output["result"])  # Output: "hello"
            print(output["context"])  # Output: "What's your name?"
            print(output["input"])  # Output: "What's your name?
    """

    @staticmethod
    def create_scope(scope: str, /) -> "PrefixContext":
        """Create a context scope.

        Args:
            scope: The scope.

        Returns:
            The context scope.
        """
        return PrefixContext(prefix=scope)

    @staticmethod
    def getter(key: Union[str, list[str]], /) -> ContextGet:
        return ContextGet(key=key)

    @staticmethod
    def setter(
        _key: Optional[str] = None,
        _value: Optional[SetValue] = None,
        /,
        **kwargs: SetValue,
    ) -> ContextSet:
        return ContextSet(_key, _value, prefix="", **kwargs)


class PrefixContext:
    """Context for a runnable with a prefix."""

    prefix: str = ""

    def __init__(self, prefix: str = ""):
        self.prefix = prefix

    def getter(self, key: Union[str, list[str]], /) -> ContextGet:
        return ContextGet(key=key, prefix=self.prefix)

    def setter(
        self,
        _key: Optional[str] = None,
        _value: Optional[SetValue] = None,
        /,
        **kwargs: SetValue,
    ) -> ContextSet:
        return ContextSet(_key, _value, prefix=self.prefix, **kwargs)


def _print_keys(keys: Union[str, Sequence[str]]) -> str:
    if isinstance(keys, str):
        return f"'{keys}'"
    else:
        return ", ".join(f"'{k}'" for k in keys)
