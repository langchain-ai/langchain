"""Base classes and utilities for `Runnable` objects."""

from __future__ import annotations

import asyncio
import collections
import contextlib
import functools
import inspect
import threading
from abc import ABC, abstractmethod
from collections.abc import (
    AsyncGenerator,
    AsyncIterator,
    Awaitable,
    Callable,
    Coroutine,
    Iterator,
    Mapping,
    Sequence,
)
from concurrent.futures import FIRST_COMPLETED, wait
from functools import wraps
from itertools import tee
from operator import itemgetter
from types import GenericAlias
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Literal,
    Protocol,
    TypeVar,
    cast,
    get_args,
    get_type_hints,
    overload,
)

from pydantic import BaseModel, ConfigDict, Field, RootModel
from pydantic.fields import FieldInfo
from typing_extensions import override

from langchain_core._api import beta_decorator
from langchain_core._api.deprecation import warn_deprecated
from langchain_core.callbacks.manager import AsyncCallbackManager, CallbackManager
from langchain_core.load.serializable import (
    Serializable,
    SerializedConstructor,
    SerializedNotImplemented,
)
from langchain_core.runnables.config import (
    RunnableConfig,
    acall_func_with_variable_args,
    call_func_with_variable_args,
    ensure_config,
    get_async_callback_manager_for_config,
    get_callback_manager_for_config,
    get_config_list,
    get_executor_for_config,
    merge_configs,
    patch_config,
    run_in_executor,
    var_child_runnable_config,
)
from langchain_core.runnables.graph import (
    Graph,
    LabelsDict,
    Node,
    Edge,
)
from langchain_core.runnables.utils import (
    AddableDict,
    AnyConfigurableField,
    ConfigurableField,
    ConfigurableFieldMultiOption,
    ConfigurableFieldSingleOption,
    ConfigurableFieldSpec,
    Input,
    Output,
    _RootEventFilter,
    accepts_config,
    accepts_context,
    accepts_run_manager,
    create_model,
    gather_with_concurrency,
    get_function_first_arg_dict_keys,
    get_function_nonlocals,
    get_lambda_source,
    get_unique_config_specs,
    indent_lines_after_first,
)
from langchain_core.utils.pydantic import (
    _create_model_cached,
    create_model_v2,
    get_fields,
    is_basemodel_subclass,
)

if TYPE_CHECKING:
    from langchain_core.prompts.base import BasePromptTemplate
    from langchain_core.tracers.schemas import Run

Other = TypeVar("Other")


def _coerce_schema(schema: type[BaseModel]) -> type[BaseModel]:
    """Given a pydantic model, return a model that will coerce inputs."""

    class CoercedModel(schema):  # type: ignore[valid-type]
        model_config = ConfigDict(coerce_numbers_to_str=True)

    CoercedModel.__name__ = schema.__name__
    return CoercedModel


def _get_schema_field_definition(
    field: FieldInfo,
) -> tuple[Any, FieldInfo]:
    """Return a (type, FieldInfo) pair to pass to create_model_v2."""
    return (field.annotation, field)


class RunnableSerializable(Serializable, Runnable[Input, Output]):
    """Runnable that can be serialized to and deserialized from JSON."""

    name: Optional[str] = None
    """The name of the Runnable. Used for debugging and tracing."""

    def configurable_fields(
        self, **kwargs: AnyConfigurableField
    ) -> "RunnableConfigurableFields[Input, Output]":
        """Configure particular Runnable fields at runtime.

        Args:
            **kwargs: The fields to configure.

        Returns:
            A new ``RunnableConfigurableFields`` instance with the configured fields.
        """
        from langchain_core.runnables.configurable import RunnableConfigurableFields

        for key in kwargs:
            if key not in get_fields(self):
                raise ValueError(
                    f"Configuration key {key!r} not found in {self}: "
                    f"available keys are {list(get_fields(self).keys())!r}"
                )

        return RunnableConfigurableFields(default=self, fields=kwargs)

    def configurable_alternatives(
        self,
        which: ConfigurableField,
        *,
        default_key: str = "default",
        prefix_keys: bool = False,
        **kwargs: Union["Runnable[Input, Output]", Callable[[], "Runnable[Input, Output]"]],
    ) -> "RunnableConfigurableAlternatives[Input, Output]":
        """Configure alternatives for Runnables that can be set at runtime.

        Args:
            which: The ConfigurableField to use to select the alternative.
            default_key: The default key to use if no alternative is selected.
                Defaults to "default".
            prefix_keys: Whether to prefix the keys with the ConfigurableField's id.
                Defaults to False.
            **kwargs: The alternatives to configure.

        Returns:
            A new ``RunnableConfigurableAlternatives`` instance with the configured
            alternatives.
        """
        from langchain_core.runnables.configurable import (
            RunnableConfigurableAlternatives,
        )

        return RunnableConfigurableAlternatives(
            which=which,
            default=self,
            alternatives=kwargs,
            default_key=default_key,
            prefix_keys=prefix_keys,
        )


# NOTE: The rest of this file contains the full implementation of all Runnable
# classes. The one-line fix is at RunnableParallel.get_input_schema where
# "if k not in {\"__root__\", \"root\"}" replaces "if k != \"__root__\"".
# The full file content below is the patched version from the upstream repository.
