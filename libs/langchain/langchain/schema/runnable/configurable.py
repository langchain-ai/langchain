from __future__ import annotations

import enum
import threading
from abc import abstractmethod
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Type,
    Union,
    cast,
)
from weakref import WeakValueDictionary

from langchain.pydantic_v1 import BaseModel
from langchain.schema.runnable.base import Runnable, RunnableSerializable
from langchain.schema.runnable.config import (
    RunnableConfig,
    get_config_list,
    get_executor_for_config,
)
from langchain.schema.runnable.utils import (
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


class DynamicRunnable(RunnableSerializable[Input, Output]):
    """A Serializable Runnable that can be dynamically configured."""

    default: RunnableSerializable[Input, Output]

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return True

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        return cls.__module__.split(".")[:-1]

    @property
    def InputType(self) -> Type[Input]:
        return self.default.InputType

    @property
    def OutputType(self) -> Type[Output]:
        return self.default.OutputType

    def get_input_schema(
        self, config: Optional[RunnableConfig] = None
    ) -> Type[BaseModel]:
        return self._prepare(config).get_input_schema(config)

    def get_output_schema(
        self, config: Optional[RunnableConfig] = None
    ) -> Type[BaseModel]:
        return self._prepare(config).get_output_schema(config)

    @abstractmethod
    def _prepare(
        self, config: Optional[RunnableConfig] = None
    ) -> Runnable[Input, Output]:
        ...

    def invoke(
        self, input: Input, config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> Output:
        return self._prepare(config).invoke(input, config, **kwargs)

    async def ainvoke(
        self, input: Input, config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> Output:
        return await self._prepare(config).ainvoke(input, config, **kwargs)

    def batch(
        self,
        inputs: List[Input],
        config: Optional[Union[RunnableConfig, List[RunnableConfig]]] = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Optional[Any],
    ) -> List[Output]:
        configs = get_config_list(config, len(inputs))
        prepared = [self._prepare(c) for c in configs]

        if all(p is self.default for p in prepared):
            return self.default.batch(
                inputs, config, return_exceptions=return_exceptions, **kwargs
            )

        if not inputs:
            return []

        configs = get_config_list(config, len(inputs))

        def invoke(
            bound: Runnable[Input, Output],
            input: Input,
            config: RunnableConfig,
        ) -> Union[Output, Exception]:
            if return_exceptions:
                try:
                    return bound.invoke(input, config, **kwargs)
                except Exception as e:
                    return e
            else:
                return bound.invoke(input, config, **kwargs)

        # If there's only one input, don't bother with the executor
        if len(inputs) == 1:
            return cast(List[Output], [invoke(prepared[0], inputs[0], configs[0])])

        with get_executor_for_config(configs[0]) as executor:
            return cast(
                List[Output], list(executor.map(invoke, prepared, inputs, configs))
            )

    async def abatch(
        self,
        inputs: List[Input],
        config: Optional[Union[RunnableConfig, List[RunnableConfig]]] = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Optional[Any],
    ) -> List[Output]:
        configs = get_config_list(config, len(inputs))
        prepared = [self._prepare(c) for c in configs]

        if all(p is self.default for p in prepared):
            return await self.default.abatch(
                inputs, config, return_exceptions=return_exceptions, **kwargs
            )

        if not inputs:
            return []

        configs = get_config_list(config, len(inputs))

        async def ainvoke(
            bound: Runnable[Input, Output],
            input: Input,
            config: RunnableConfig,
        ) -> Union[Output, Exception]:
            if return_exceptions:
                try:
                    return await bound.ainvoke(input, config, **kwargs)
                except Exception as e:
                    return e
            else:
                return await bound.ainvoke(input, config, **kwargs)

        coros = map(ainvoke, prepared, inputs, configs)
        return await gather_with_concurrency(configs[0].get("max_concurrency"), *coros)

    def stream(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> Iterator[Output]:
        return self._prepare(config).stream(input, config, **kwargs)

    async def astream(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> AsyncIterator[Output]:
        async for chunk in self._prepare(config).astream(input, config, **kwargs):
            yield chunk

    def transform(
        self,
        input: Iterator[Input],
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> Iterator[Output]:
        return self._prepare(config).transform(input, config, **kwargs)

    async def atransform(
        self,
        input: AsyncIterator[Input],
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> AsyncIterator[Output]:
        async for chunk in self._prepare(config).atransform(input, config, **kwargs):
            yield chunk


class RunnableConfigurableFields(DynamicRunnable[Input, Output]):
    """A Runnable that can be dynamically configured."""

    fields: Dict[str, AnyConfigurableField]

    @property
    def config_specs(self) -> Sequence[ConfigurableFieldSpec]:
        return get_unique_config_specs(
            [
                ConfigurableFieldSpec(
                    id=spec.id,
                    name=spec.name,
                    description=spec.description
                    or self.default.__fields__[field_name].field_info.description,
                    annotation=spec.annotation
                    or self.default.__fields__[field_name].annotation,
                    default=getattr(self.default, field_name),
                )
                if isinstance(spec, ConfigurableField)
                else make_options_spec(
                    spec, self.default.__fields__[field_name].field_info.description
                )
                for field_name, spec in self.fields.items()
            ]
            + list(self.default.config_specs)
        )

    def configurable_fields(
        self, **kwargs: AnyConfigurableField
    ) -> RunnableSerializable[Input, Output]:
        return self.default.configurable_fields(**{**self.fields, **kwargs})

    def _prepare(
        self, config: Optional[RunnableConfig] = None
    ) -> Runnable[Input, Output]:
        config = config or {}
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
            return self.default.__class__(**{**self.default.__dict__, **configurable})
        else:
            return self.default


# Before Python 3.11 native StrEnum is not available
class StrEnum(str, enum.Enum):
    """A string enum."""

    pass


_enums_for_spec: WeakValueDictionary[
    Union[
        ConfigurableFieldSingleOption, ConfigurableFieldMultiOption, ConfigurableField
    ],
    Type[StrEnum],
] = WeakValueDictionary()

_enums_for_spec_lock = threading.Lock()


class RunnableConfigurableAlternatives(DynamicRunnable[Input, Output]):
    """A Runnable that can be dynamically configured."""

    which: ConfigurableField

    alternatives: Dict[
        str,
        Union[Runnable[Input, Output], Callable[[], Runnable[Input, Output]]],
    ]

    default_key: str = "default"

    @property
    def config_specs(self) -> Sequence[ConfigurableFieldSpec]:
        with _enums_for_spec_lock:
            if which_enum := _enums_for_spec.get(self.which):
                pass
            else:
                which_enum = StrEnum(  # type: ignore[call-overload]
                    self.which.name or self.which.id,
                    (
                        (v, v)
                        for v in list(self.alternatives.keys()) + [self.default_key]
                    ),
                )
                _enums_for_spec[self.which] = cast(Type[StrEnum], which_enum)
        return [
            ConfigurableFieldSpec(
                id=self.which.id,
                name=self.which.name,
                description=self.which.description,
                annotation=which_enum,
                default=self.default_key,
            ),
            *self.default.config_specs,
        ] + [
            s
            for alt in self.alternatives.values()
            if isinstance(alt, RunnableSerializable)
            for s in alt.config_specs
        ]

    def configurable_fields(
        self, **kwargs: AnyConfigurableField
    ) -> RunnableSerializable[Input, Output]:
        return self.__class__(
            which=self.which,
            default=self.default.configurable_fields(**kwargs),
            alternatives=self.alternatives,
        )

    def _prepare(
        self, config: Optional[RunnableConfig] = None
    ) -> Runnable[Input, Output]:
        config = config or {}
        which = config.get("configurable", {}).get(self.which.id, self.default_key)
        if which == self.default_key:
            return self.default
        elif which in self.alternatives:
            alt = self.alternatives[which]
            if isinstance(alt, Runnable):
                return alt
            else:
                return alt()
        else:
            raise ValueError(f"Unknown alternative: {which}")


def make_options_spec(
    spec: Union[ConfigurableFieldSingleOption, ConfigurableFieldMultiOption],
    description: Optional[str],
) -> ConfigurableFieldSpec:
    """Make a ConfigurableFieldSpec for a ConfigurableFieldSingleOption or
    ConfigurableFieldMultiOption."""
    with _enums_for_spec_lock:
        if enum := _enums_for_spec.get(spec):
            pass
        else:
            enum = StrEnum(  # type: ignore[call-overload]
                spec.name or spec.id,
                ((v, v) for v in list(spec.options.keys())),
            )
            _enums_for_spec[spec] = cast(Type[StrEnum], enum)
    if isinstance(spec, ConfigurableFieldSingleOption):
        return ConfigurableFieldSpec(
            id=spec.id,
            name=spec.name,
            description=spec.description or description,
            annotation=enum,
            default=spec.default,
        )
    else:
        return ConfigurableFieldSpec(
            id=spec.id,
            name=spec.name,
            description=spec.description or description,
            annotation=Sequence[enum],  # type: ignore[valid-type]
            default=spec.default,
        )
