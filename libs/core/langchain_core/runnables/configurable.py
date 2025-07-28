"""Runnables that can be dynamically configured."""

from __future__ import annotations

import enum
import threading
from abc import abstractmethod
from collections.abc import (
    AsyncIterator,
    Iterator,
    Sequence,
)
from functools import wraps
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Optional,
    Union,
    cast,
)
from weakref import WeakValueDictionary

from pydantic import BaseModel, ConfigDict
from typing_extensions import override

from langchain_core.runnables.base import Runnable, RunnableSerializable
from langchain_core.runnables.config import (
    RunnableConfig,
    ensure_config,
    get_config_list,
    get_executor_for_config,
    merge_configs,
)
from langchain_core.runnables.utils import (
    AnyConfigurableField,
    ConfigurableField,
    ConfigurableFieldMultiOption,
    ConfigurableFieldSingleOption,
    ConfigurableFieldSpec,
    Input,
    Output,
    gather_with_concurrency,
    get_unique_config_specs,
)

if TYPE_CHECKING:
    from langchain_core.runnables.graph import Graph


class DynamicRunnable(RunnableSerializable[Input, Output]):
    """Serializable Runnable that can be dynamically configured.

    A DynamicRunnable should be initiated using the `configurable_fields` or
    `configurable_alternatives` method of a Runnable.

    Parameters:
        default: The default Runnable to use.
        config: The configuration to use.
    """

    default: RunnableSerializable[Input, Output]

    config: Optional[RunnableConfig] = None

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    @classmethod
    @override
    def is_lc_serializable(cls) -> bool:
        return True

    @classmethod
    @override
    def get_lc_namespace(cls) -> list[str]:
        return ["langchain", "schema", "runnable"]

    @property
    @override
    def InputType(self) -> type[Input]:
        return self.default.InputType

    @property
    @override
    def OutputType(self) -> type[Output]:
        return self.default.OutputType

    @override
    def get_input_schema(
        self, config: Optional[RunnableConfig] = None
    ) -> type[BaseModel]:
        runnable, config = self.prepare(config)
        return runnable.get_input_schema(config)

    @override
    def get_output_schema(
        self, config: Optional[RunnableConfig] = None
    ) -> type[BaseModel]:
        runnable, config = self.prepare(config)
        return runnable.get_output_schema(config)

    @override
    def get_graph(self, config: Optional[RunnableConfig] = None) -> Graph:
        runnable, config = self.prepare(config)
        return runnable.get_graph(config)

    @override
    def with_config(
        self,
        config: Optional[RunnableConfig] = None,
        # Sadly Unpack is not well supported by mypy so this will have to be untyped
        **kwargs: Any,
    ) -> Runnable[Input, Output]:
        return self.__class__(
            **{**self.__dict__, "config": ensure_config(merge_configs(config, kwargs))}  # type: ignore[arg-type]
        )

    def prepare(
        self, config: Optional[RunnableConfig] = None
    ) -> tuple[Runnable[Input, Output], RunnableConfig]:
        """Prepare the Runnable for invocation.

        Args:
            config: The configuration to use. Defaults to None.

        Returns:
            tuple[Runnable[Input, Output], RunnableConfig]: The prepared Runnable and
            configuration.
        """
        runnable: Runnable[Input, Output] = self
        while isinstance(runnable, DynamicRunnable):
            runnable, config = runnable._prepare(merge_configs(runnable.config, config))  # noqa: SLF001
        return runnable, cast("RunnableConfig", config)

    @abstractmethod
    def _prepare(
        self, config: Optional[RunnableConfig] = None
    ) -> tuple[Runnable[Input, Output], RunnableConfig]: ...

    @override
    def invoke(
        self, input: Input, config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> Output:
        runnable, config = self.prepare(config)
        return runnable.invoke(input, config, **kwargs)

    @override
    async def ainvoke(
        self, input: Input, config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> Output:
        runnable, config = self.prepare(config)
        return await runnable.ainvoke(input, config, **kwargs)

    @override
    def batch(
        self,
        inputs: list[Input],
        config: Optional[Union[RunnableConfig, list[RunnableConfig]]] = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Optional[Any],
    ) -> list[Output]:
        configs = get_config_list(config, len(inputs))
        prepared = [self.prepare(c) for c in configs]

        if all(p is self.default for p, _ in prepared):
            return self.default.batch(
                inputs,
                [c for _, c in prepared],
                return_exceptions=return_exceptions,
                **kwargs,
            )

        if not inputs:
            return []

        def invoke(
            prepared: tuple[Runnable[Input, Output], RunnableConfig],
            input_: Input,
        ) -> Union[Output, Exception]:
            bound, config = prepared
            if return_exceptions:
                try:
                    return bound.invoke(input_, config, **kwargs)
                except Exception as e:
                    return e
            else:
                return bound.invoke(input_, config, **kwargs)

        # If there's only one input, don't bother with the executor
        if len(inputs) == 1:
            return cast("list[Output]", [invoke(prepared[0], inputs[0])])

        with get_executor_for_config(configs[0]) as executor:
            return cast("list[Output]", list(executor.map(invoke, prepared, inputs)))

    @override
    async def abatch(
        self,
        inputs: list[Input],
        config: Optional[Union[RunnableConfig, list[RunnableConfig]]] = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Optional[Any],
    ) -> list[Output]:
        configs = get_config_list(config, len(inputs))
        prepared = [self.prepare(c) for c in configs]

        if all(p is self.default for p, _ in prepared):
            return await self.default.abatch(
                inputs,
                [c for _, c in prepared],
                return_exceptions=return_exceptions,
                **kwargs,
            )

        if not inputs:
            return []

        async def ainvoke(
            prepared: tuple[Runnable[Input, Output], RunnableConfig],
            input_: Input,
        ) -> Union[Output, Exception]:
            bound, config = prepared
            if return_exceptions:
                try:
                    return await bound.ainvoke(input_, config, **kwargs)
                except Exception as e:
                    return e
            else:
                return await bound.ainvoke(input_, config, **kwargs)

        coros = map(ainvoke, prepared, inputs)
        return await gather_with_concurrency(configs[0].get("max_concurrency"), *coros)

    @override
    def stream(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> Iterator[Output]:
        runnable, config = self.prepare(config)
        return runnable.stream(input, config, **kwargs)

    @override
    async def astream(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> AsyncIterator[Output]:
        runnable, config = self.prepare(config)
        async for chunk in runnable.astream(input, config, **kwargs):
            yield chunk

    @override
    def transform(
        self,
        input: Iterator[Input],
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> Iterator[Output]:
        runnable, config = self.prepare(config)
        return runnable.transform(input, config, **kwargs)

    @override
    async def atransform(
        self,
        input: AsyncIterator[Input],
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> AsyncIterator[Output]:
        runnable, config = self.prepare(config)
        async for chunk in runnable.atransform(input, config, **kwargs):
            yield chunk

    @override
    def __getattr__(self, name: str) -> Any:  # type: ignore[misc]
        attr = getattr(self.default, name)
        if callable(attr):

            @wraps(attr)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                for key, arg in kwargs.items():
                    if key == "config" and (
                        isinstance(arg, dict)
                        and "configurable" in arg
                        and isinstance(arg["configurable"], dict)
                    ):
                        runnable, config = self.prepare(cast("RunnableConfig", arg))
                        kwargs = {**kwargs, "config": config}
                        return getattr(runnable, name)(*args, **kwargs)

                for idx, arg in enumerate(args):
                    if (
                        isinstance(arg, dict)
                        and "configurable" in arg
                        and isinstance(arg["configurable"], dict)
                    ):
                        runnable, config = self.prepare(cast("RunnableConfig", arg))
                        argsl = list(args)
                        argsl[idx] = config
                        return getattr(runnable, name)(*argsl, **kwargs)

                if self.config:
                    runnable, config = self.prepare()
                    return getattr(runnable, name)(*args, **kwargs)

                return attr(*args, **kwargs)

            return wrapper

        return attr


class RunnableConfigurableFields(DynamicRunnable[Input, Output]):
    """Runnable that can be dynamically configured.

    A RunnableConfigurableFields should be initiated using the
    `configurable_fields` method of a Runnable.

    Parameters:
        fields: The configurable fields to use.

    Here is an example of using a RunnableConfigurableFields with LLMs:

        .. code-block:: python

            from langchain_core.prompts import PromptTemplate
            from langchain_core.runnables import ConfigurableField
            from langchain_openai import ChatOpenAI

            model = ChatOpenAI(temperature=0).configurable_fields(
                temperature=ConfigurableField(
                    id="temperature",
                    name="LLM Temperature",
                    description="The temperature of the LLM",
                )
            )
            # This creates a RunnableConfigurableFields for a chat model.

            # When invoking the created RunnableSequence, you can pass in the
            # value for your ConfigurableField's id which in this case
            # will be change in temperature

            prompt = PromptTemplate.from_template("Pick a random number above {x}")
            chain = prompt | model

            chain.invoke({"x": 0})
            chain.invoke({"x": 0}, config={"configurable": {"temperature": 0.9}})


    Here is an example of using a RunnableConfigurableFields with HubRunnables:

        .. code-block:: python

            from langchain_core.prompts import PromptTemplate
            from langchain_core.runnables import ConfigurableField
            from langchain_openai import ChatOpenAI
            from langchain.runnables.hub import HubRunnable

            prompt = HubRunnable("rlm/rag-prompt").configurable_fields(
                owner_repo_commit=ConfigurableField(
                    id="hub_commit",
                    name="Hub Commit",
                    description="The Hub commit to pull from",
                )
            )

            prompt.invoke({"question": "foo", "context": "bar"})

            # Invoking prompt with `with_config` method

            prompt.invoke(
                {"question": "foo", "context": "bar"},
                config={"configurable": {"hub_commit": "rlm/rag-prompt-llama"}},
            )

    """

    fields: dict[str, AnyConfigurableField]

    @property
    def config_specs(self) -> list[ConfigurableFieldSpec]:
        """Get the configuration specs for the RunnableConfigurableFields.

        Returns:
            list[ConfigurableFieldSpec]: The configuration specs.
        """
        config_specs = []

        default_fields = type(self.default).model_fields
        for field_name, spec in self.fields.items():
            if isinstance(spec, ConfigurableField):
                config_specs.append(
                    ConfigurableFieldSpec(
                        id=spec.id,
                        name=spec.name,
                        description=spec.description
                        or default_fields[field_name].description,
                        annotation=spec.annotation
                        or default_fields[field_name].annotation,
                        default=getattr(self.default, field_name),
                        is_shared=spec.is_shared,
                    )
                )
            else:
                config_specs.append(
                    make_options_spec(spec, default_fields[field_name].description)
                )

        config_specs.extend(self.default.config_specs)

        return get_unique_config_specs(config_specs)

    @override
    def configurable_fields(
        self, **kwargs: AnyConfigurableField
    ) -> RunnableSerializable[Input, Output]:
        return self.default.configurable_fields(**{**self.fields, **kwargs})

    def _prepare(
        self, config: Optional[RunnableConfig] = None
    ) -> tuple[Runnable[Input, Output], RunnableConfig]:
        config = ensure_config(config)
        specs_by_id = {spec.id: (key, spec) for key, spec in self.fields.items()}
        configurable_fields = {
            specs_by_id[k][0]: v
            for k, v in config.get("configurable", {}).items()
            if k in specs_by_id and isinstance(specs_by_id[k][1], ConfigurableField)
        }
        configurable_single_options = {
            k: v.options[(config.get("configurable", {}).get(v.id) or v.default)]
            for k, v in self.fields.items()
            if isinstance(v, ConfigurableFieldSingleOption)
        }
        configurable_multi_options = {
            k: [
                v.options[o]
                for o in config.get("configurable", {}).get(v.id, v.default)
            ]
            for k, v in self.fields.items()
            if isinstance(v, ConfigurableFieldMultiOption)
        }
        configurable = {
            **configurable_fields,
            **configurable_single_options,
            **configurable_multi_options,
        }

        if configurable:
            init_params = {
                k: v
                for k, v in self.default.__dict__.items()
                if k in type(self.default).model_fields
            }
            return (
                self.default.__class__(**{**init_params, **configurable}),
                config,
            )
        return (self.default, config)


# Before Python 3.11 native StrEnum is not available
class StrEnum(str, enum.Enum):
    """String enum."""


_enums_for_spec: WeakValueDictionary[
    Union[
        ConfigurableFieldSingleOption, ConfigurableFieldMultiOption, ConfigurableField
    ],
    type[StrEnum],
] = WeakValueDictionary()

_enums_for_spec_lock = threading.Lock()


class RunnableConfigurableAlternatives(DynamicRunnable[Input, Output]):
    """Runnable that can be dynamically configured.

    A RunnableConfigurableAlternatives should be initiated using the
    `configurable_alternatives` method of a Runnable or can be
    initiated directly as well.

    Here is an example of using a RunnableConfigurableAlternatives that uses
    alternative prompts to illustrate its functionality:

        .. code-block:: python

            from langchain_core.runnables import ConfigurableField
            from langchain_openai import ChatOpenAI

            # This creates a RunnableConfigurableAlternatives for Prompt Runnable
            # with two alternatives.
            prompt = PromptTemplate.from_template(
                "Tell me a joke about {topic}"
            ).configurable_alternatives(
                ConfigurableField(id="prompt"),
                default_key="joke",
                poem=PromptTemplate.from_template("Write a short poem about {topic}")
            )

            # When invoking the created RunnableSequence, you can pass in the
            # value for your ConfigurableField's id which in this case will either be
            # `joke` or `poem`.
            chain = prompt | ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)

            # The `with_config` method brings in the desired Prompt Runnable in your
            # Runnable Sequence.
            chain.with_config(configurable={"prompt": "poem"}).invoke({"topic": "bears"})


    Equivalently, you can initialize RunnableConfigurableAlternatives directly
    and use in LCEL in the same way:

        .. code-block:: python

            from langchain_core.runnables import ConfigurableField
            from langchain_core.runnables.configurable import RunnableConfigurableAlternatives
            from langchain_openai import ChatOpenAI

            prompt = RunnableConfigurableAlternatives(
                which=ConfigurableField(id='prompt'),
                default=PromptTemplate.from_template("Tell me a joke about {topic}"),
                default_key='joke',
                prefix_keys=False,
                alternatives={"poem":PromptTemplate.from_template("Write a short poem about {topic}")}
            )
            chain = prompt | ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
            chain.with_config(configurable={"prompt": "poem"}).invoke({"topic": "bears"})

    """  # noqa: E501

    which: ConfigurableField
    """The ConfigurableField to use to choose between alternatives."""

    alternatives: dict[
        str,
        Union[Runnable[Input, Output], Callable[[], Runnable[Input, Output]]],
    ]
    """The alternatives to choose from."""

    default_key: str = "default"
    """The enum value to use for the default option. Defaults to ``'default'``."""

    prefix_keys: bool
    """Whether to prefix configurable fields of each alternative with a namespace
    of the form <which.id>==<alternative_key>, eg. a key named "temperature" used by
    the alternative named "gpt3" becomes "model==gpt3/temperature"."""

    @property
    @override
    def config_specs(self) -> list[ConfigurableFieldSpec]:
        with _enums_for_spec_lock:
            if which_enum := _enums_for_spec.get(self.which):
                pass
            else:
                which_enum = StrEnum(  # type: ignore[call-overload]
                    self.which.name or self.which.id,
                    (
                        (v, v)
                        for v in [*list(self.alternatives.keys()), self.default_key]
                    ),
                )
                _enums_for_spec[self.which] = cast("type[StrEnum]", which_enum)
        return get_unique_config_specs(
            # which alternative
            [
                ConfigurableFieldSpec(
                    id=self.which.id,
                    name=self.which.name,
                    description=self.which.description,
                    annotation=which_enum,
                    default=self.default_key,
                    is_shared=self.which.is_shared,
                ),
            ]
            # config specs of the default option
            + (
                [
                    prefix_config_spec(s, f"{self.which.id}=={self.default_key}")
                    for s in self.default.config_specs
                ]
                if self.prefix_keys
                else self.default.config_specs
            )
            # config specs of the alternatives
            + [
                (
                    prefix_config_spec(s, f"{self.which.id}=={alt_key}")
                    if self.prefix_keys
                    else s
                )
                for alt_key, alt in self.alternatives.items()
                if isinstance(alt, RunnableSerializable)
                for s in alt.config_specs
            ]
        )

    @override
    def configurable_fields(
        self, **kwargs: AnyConfigurableField
    ) -> RunnableSerializable[Input, Output]:
        return self.__class__(
            which=self.which,
            default=self.default.configurable_fields(**kwargs),
            alternatives=self.alternatives,
            default_key=self.default_key,
            prefix_keys=self.prefix_keys,
        )

    def _prepare(
        self, config: Optional[RunnableConfig] = None
    ) -> tuple[Runnable[Input, Output], RunnableConfig]:
        config = ensure_config(config)
        which = config.get("configurable", {}).get(self.which.id, self.default_key)
        # remap configurable keys for the chosen alternative
        if self.prefix_keys:
            config = cast(
                "RunnableConfig",
                {
                    **config,
                    "configurable": {
                        _strremoveprefix(k, f"{self.which.id}=={which}/"): v
                        for k, v in config.get("configurable", {}).items()
                    },
                },
            )
        # return the chosen alternative
        if which == self.default_key:
            return (self.default, config)
        if which in self.alternatives:
            alt = self.alternatives[which]
            if isinstance(alt, Runnable):
                return (alt, config)
            return (alt(), config)
        msg = f"Unknown alternative: {which}"
        raise ValueError(msg)


def _strremoveprefix(s: str, prefix: str) -> str:
    """str.removeprefix() is only available in Python 3.9+."""
    return s.replace(prefix, "", 1) if s.startswith(prefix) else s


def prefix_config_spec(
    spec: ConfigurableFieldSpec, prefix: str
) -> ConfigurableFieldSpec:
    """Prefix the id of a ConfigurableFieldSpec.

    This is useful when a RunnableConfigurableAlternatives is used as a
    ConfigurableField of another RunnableConfigurableAlternatives.

    Args:
        spec: The ConfigurableFieldSpec to prefix.
        prefix: The prefix to add.

    Returns:
        ConfigurableFieldSpec: The prefixed ConfigurableFieldSpec.
    """
    return (
        ConfigurableFieldSpec(
            id=f"{prefix}/{spec.id}",
            name=spec.name,
            description=spec.description,
            annotation=spec.annotation,
            default=spec.default,
            is_shared=spec.is_shared,
        )
        if not spec.is_shared
        else spec
    )


def make_options_spec(
    spec: Union[ConfigurableFieldSingleOption, ConfigurableFieldMultiOption],
    description: Optional[str],
) -> ConfigurableFieldSpec:
    """Make a ConfigurableFieldSpec for a ConfigurableFieldSingleOption or ConfigurableFieldMultiOption.

    Args:
        spec: The ConfigurableFieldSingleOption or ConfigurableFieldMultiOption.
        description: The description to use if the spec does not have one.

    Returns:
        The ConfigurableFieldSpec.
    """  # noqa: E501
    with _enums_for_spec_lock:
        if enum := _enums_for_spec.get(spec):
            pass
        else:
            enum = StrEnum(  # type: ignore[call-overload]
                spec.name or spec.id,
                ((v, v) for v in list(spec.options.keys())),
            )
            _enums_for_spec[spec] = cast("type[StrEnum]", enum)
    if isinstance(spec, ConfigurableFieldSingleOption):
        return ConfigurableFieldSpec(
            id=spec.id,
            name=spec.name,
            description=spec.description or description,
            annotation=enum,
            default=spec.default,
            is_shared=spec.is_shared,
        )
    return ConfigurableFieldSpec(
        id=spec.id,
        name=spec.name,
        description=spec.description or description,
        annotation=Sequence[enum],  # type: ignore[valid-type]
        default=spec.default,
        is_shared=spec.is_shared,
    )
