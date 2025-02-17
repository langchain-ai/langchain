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
    Coroutine,
    Iterator,
    Mapping,
    Sequence,
)
from concurrent.futures import FIRST_COMPLETED, wait
from contextvars import copy_context
from functools import wraps
from itertools import groupby, tee
from operator import itemgetter
from types import GenericAlias
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    Optional,
    Protocol,
    TypeVar,
    Union,
    cast,
    get_type_hints,
    overload,
)

from pydantic import BaseModel, ConfigDict, Field, RootModel
from typing_extensions import Literal, get_args, override

from langchain_core._api import beta_decorator
from langchain_core.load.serializable import (
    Serializable,
    SerializedConstructor,
    SerializedNotImplemented,
)
from langchain_core.runnables.config import (
    RunnableConfig,
    _set_config_context,
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
)
from langchain_core.runnables.graph import Graph
from langchain_core.runnables.schema import StreamEvent
from langchain_core.runnables.utils import (
    AddableDict,
    AnyConfigurableField,
    ConfigurableField,
    ConfigurableFieldSpec,
    Input,
    Output,
    accepts_config,
    accepts_run_manager,
    asyncio_accepts_context,
    gated_coro,
    gather_with_concurrency,
    get_function_first_arg_dict_keys,
    get_function_nonlocals,
    get_lambda_source,
    get_unique_config_specs,
    indent_lines_after_first,
    is_async_callable,
    is_async_generator,
)
from langchain_core.utils.aiter import aclosing, atee, py_anext
from langchain_core.utils.iter import safetee
from langchain_core.utils.pydantic import create_model_v2

if TYPE_CHECKING:
    from langchain_core.callbacks.manager import (
        AsyncCallbackManagerForChainRun,
        CallbackManagerForChainRun,
    )
    from langchain_core.prompts.base import BasePromptTemplate
    from langchain_core.runnables.fallbacks import (
        RunnableWithFallbacks as RunnableWithFallbacksT,
    )
    from langchain_core.tools import BaseTool
    from langchain_core.tracers.log_stream import (
        RunLog,
        RunLogPatch,
    )
    from langchain_core.tracers.root_listeners import AsyncListener
    from langchain_core.tracers.schemas import Run


Other = TypeVar("Other")


class Runnable(Generic[Input, Output], ABC):
    """A unit of work that can be invoked, batched, streamed, transformed and composed.

    Key Methods
    ===========

    - **invoke/ainvoke**: Transforms a single input into an output.
    - **batch/abatch**: Efficiently transforms multiple inputs into outputs.
    - **stream/astream**: Streams output from a single input as it's produced.
    - **astream_log**: Streams output and selected intermediate results from an input.

    Built-in optimizations:

    - **Batch**: By default, batch runs invoke() in parallel using a thread pool executor.
      Override to optimize batching.

    - **Async**: Methods with "a" suffix are asynchronous. By default, they execute
      the sync counterpart using asyncio's thread pool.
      Override for native async.

    All methods accept an optional config argument, which can be used to configure
    execution, add tags and metadata for tracing and debugging etc.

    Runnables expose schematic information about their input, output and config via
    the input_schema property, the output_schema property and config_schema method.

    LCEL and Composition
    ====================

    The LangChain Expression Language (LCEL) is a declarative way to compose Runnables
    into chains. Any chain constructed this way will automatically have sync, async,
    batch, and streaming support.

    The main composition primitives are RunnableSequence and RunnableParallel.

    **RunnableSequence** invokes a series of runnables sequentially, with
    one Runnable's output serving as the next's input. Construct using
    the `|` operator or by passing a list of runnables to RunnableSequence.

    **RunnableParallel** invokes runnables concurrently, providing the same input
    to each. Construct it using a dict literal within a sequence or by passing a
    dict to RunnableParallel.


    For example,

    .. code-block:: python

        from langchain_core.runnables import RunnableLambda

        # A RunnableSequence constructed using the `|` operator
        sequence = RunnableLambda(lambda x: x + 1) | RunnableLambda(lambda x: x * 2)
        sequence.invoke(1) # 4
        sequence.batch([1, 2, 3]) # [4, 6, 8]


        # A sequence that contains a RunnableParallel constructed using a dict literal
        sequence = RunnableLambda(lambda x: x + 1) | {
            'mul_2': RunnableLambda(lambda x: x * 2),
            'mul_5': RunnableLambda(lambda x: x * 5)
        }
        sequence.invoke(1) # {'mul_2': 4, 'mul_5': 10}

    Standard Methods
    ================

    All Runnables expose additional methods that can be used to modify their behavior
    (e.g., add a retry policy, add lifecycle listeners, make them configurable, etc.).

    These methods will work on any Runnable, including Runnable chains constructed
    by composing other Runnables. See the individual methods for details.

    For example,

    .. code-block:: python

        from langchain_core.runnables import RunnableLambda

        import random

        def add_one(x: int) -> int:
            return x + 1


        def buggy_double(y: int) -> int:
            \"\"\"Buggy code that will fail 70% of the time\"\"\"
            if random.random() > 0.3:
                print('This code failed, and will probably be retried!')  # noqa: T201
                raise ValueError('Triggered buggy code')
            return y * 2

        sequence = (
            RunnableLambda(add_one) |
            RunnableLambda(buggy_double).with_retry( # Retry on failure
                stop_after_attempt=10,
                wait_exponential_jitter=False
            )
        )

        print(sequence.input_schema.model_json_schema()) # Show inferred input schema
        print(sequence.output_schema.model_json_schema()) # Show inferred output schema
        print(sequence.invoke(2)) # invoke the sequence (note the retry above!!)

    Debugging and tracing
    =====================

    As the chains get longer, it can be useful to be able to see intermediate results
    to debug and trace the chain.

    You can set the global debug flag to True to enable debug output for all chains:

        .. code-block:: python

            from langchain_core.globals import set_debug
            set_debug(True)

    Alternatively, you can pass existing or custom callbacks to any given chain:

        .. code-block:: python

            from langchain_core.tracers import ConsoleCallbackHandler

            chain.invoke(
                ...,
                config={'callbacks': [ConsoleCallbackHandler()]}
            )

    For a UI (and much more) checkout LangSmith: https://docs.smith.langchain.com/
    """  # noqa: E501

    name: Optional[str]
    """The name of the Runnable. Used for debugging and tracing."""

    def get_name(
        self, suffix: Optional[str] = None, *, name: Optional[str] = None
    ) -> str:
        """Get the name of the Runnable."""
        if name:
            name_ = name
        elif hasattr(self, "name") and self.name:
            name_ = self.name
        else:
            # Here we handle a case where the runnable subclass is also a pydantic
            # model.
            cls = self.__class__
            # Then it's a pydantic sub-class, and we have to check
            # whether it's a generic, and if so recover the original name.
            if (
                hasattr(
                    cls,
                    "__pydantic_generic_metadata__",
                )
                and "origin" in cls.__pydantic_generic_metadata__
                and cls.__pydantic_generic_metadata__["origin"] is not None
            ):
                name_ = cls.__pydantic_generic_metadata__["origin"].__name__
            else:
                name_ = cls.__name__

        if suffix:
            if name_[0].isupper():
                return name_ + suffix.title()
            else:
                return name_ + "_" + suffix.lower()
        else:
            return name_

    @property
    def InputType(self) -> type[Input]:  # noqa: N802
        """The type of input this Runnable accepts specified as a type annotation."""
        # First loop through all parent classes and if any of them is
        # a pydantic model, we will pick up the generic parameterization
        # from that model via the __pydantic_generic_metadata__ attribute.
        for base in self.__class__.mro():
            if hasattr(base, "__pydantic_generic_metadata__"):
                metadata = base.__pydantic_generic_metadata__
                if "args" in metadata and len(metadata["args"]) == 2:
                    return metadata["args"][0]

        # If we didn't find a pydantic model in the parent classes,
        # then loop through __orig_bases__. This corresponds to
        # Runnables that are not pydantic models.
        for cls in self.__class__.__orig_bases__:  # type: ignore[attr-defined]
            type_args = get_args(cls)
            if type_args and len(type_args) == 2:
                return type_args[0]

        msg = (
            f"Runnable {self.get_name()} doesn't have an inferable InputType. "
            "Override the InputType property to specify the input type."
        )
        raise TypeError(msg)

    @property
    def OutputType(self) -> type[Output]:  # noqa: N802
        """The type of output this Runnable produces specified as a type annotation."""
        # First loop through bases -- this will help generic
        # any pydantic models.
        for base in self.__class__.mro():
            if hasattr(base, "__pydantic_generic_metadata__"):
                metadata = base.__pydantic_generic_metadata__
                if "args" in metadata and len(metadata["args"]) == 2:
                    return metadata["args"][1]

        for cls in self.__class__.__orig_bases__:  # type: ignore[attr-defined]
            type_args = get_args(cls)
            if type_args and len(type_args) == 2:
                return type_args[1]

        msg = (
            f"Runnable {self.get_name()} doesn't have an inferable OutputType. "
            "Override the OutputType property to specify the output type."
        )
        raise TypeError(msg)

    @property
    def input_schema(self) -> type[BaseModel]:
        """The type of input this Runnable accepts specified as a pydantic model."""
        return self.get_input_schema()

    def get_input_schema(
        self, config: Optional[RunnableConfig] = None
    ) -> type[BaseModel]:
        """Get a pydantic model that can be used to validate input to the Runnable.

        Runnables that leverage the configurable_fields and configurable_alternatives
        methods will have a dynamic input schema that depends on which
        configuration the Runnable is invoked with.

        This method allows to get an input schema for a specific configuration.

        Args:
            config: A config to use when generating the schema.

        Returns:
            A pydantic model that can be used to validate input.
        """
        root_type = self.InputType

        if (
            inspect.isclass(root_type)
            and not isinstance(root_type, GenericAlias)
            and issubclass(root_type, BaseModel)
        ):
            return root_type

        return create_model_v2(
            self.get_name("Input"),
            root=root_type,
            # create model needs access to appropriate type annotations to be
            # able to construct the pydantic model.
            # When we create the model, we pass information about the namespace
            # where the model is being created, so the type annotations can
            # be resolved correctly as well.
            # self.__class__.__module__ handles the case when the Runnable is
            # being sub-classed in a different module.
            module_name=self.__class__.__module__,
        )

    def get_input_jsonschema(
        self, config: Optional[RunnableConfig] = None
    ) -> dict[str, Any]:
        """Get a JSON schema that represents the input to the Runnable.

        Args:
            config: A config to use when generating the schema.

        Returns:
            A JSON schema that represents the input to the Runnable.

        Example:

            .. code-block:: python

                from langchain_core.runnables import RunnableLambda

                def add_one(x: int) -> int:
                    return x + 1

                runnable = RunnableLambda(add_one)

                print(runnable.get_input_jsonschema())

        .. versionadded:: 0.3.0
        """
        return self.get_input_schema(config).model_json_schema()

    @property
    def output_schema(self) -> type[BaseModel]:
        """The type of output this Runnable produces specified as a pydantic model."""
        return self.get_output_schema()

    def get_output_schema(
        self, config: Optional[RunnableConfig] = None
    ) -> type[BaseModel]:
        """Get a pydantic model that can be used to validate output to the Runnable.

        Runnables that leverage the configurable_fields and configurable_alternatives
        methods will have a dynamic output schema that depends on which
        configuration the Runnable is invoked with.

        This method allows to get an output schema for a specific configuration.

        Args:
            config: A config to use when generating the schema.

        Returns:
            A pydantic model that can be used to validate output.
        """
        root_type = self.OutputType

        if (
            inspect.isclass(root_type)
            and not isinstance(root_type, GenericAlias)
            and issubclass(root_type, BaseModel)
        ):
            return root_type

        return create_model_v2(
            self.get_name("Output"),
            root=root_type,
            # create model needs access to appropriate type annotations to be
            # able to construct the pydantic model.
            # When we create the model, we pass information about the namespace
            # where the model is being created, so the type annotations can
            # be resolved correctly as well.
            # self.__class__.__module__ handles the case when the Runnable is
            # being sub-classed in a different module.
            module_name=self.__class__.__module__,
        )

    def get_output_jsonschema(
        self, config: Optional[RunnableConfig] = None
    ) -> dict[str, Any]:
        """Get a JSON schema that represents the output of the Runnable.

        Args:
            config: A config to use when generating the schema.

        Returns:
            A JSON schema that represents the output of the Runnable.

        Example:

            .. code-block:: python

                from langchain_core.runnables import RunnableLambda

                def add_one(x: int) -> int:
                    return x + 1

                runnable = RunnableLambda(add_one)

                print(runnable.get_output_jsonschema())

        .. versionadded:: 0.3.0
        """
        return self.get_output_schema(config).model_json_schema()

    @property
    def config_specs(self) -> list[ConfigurableFieldSpec]:
        """List configurable fields for this Runnable."""
        return []

    def config_schema(
        self, *, include: Optional[Sequence[str]] = None
    ) -> type[BaseModel]:
        """The type of config this Runnable accepts specified as a pydantic model.

        To mark a field as configurable, see the `configurable_fields`
        and `configurable_alternatives` methods.

        Args:
            include: A list of fields to include in the config schema.

        Returns:
            A pydantic model that can be used to validate config.
        """
        include = include or []
        config_specs = self.config_specs
        configurable = (
            create_model_v2(  # type: ignore[call-overload]
                "Configurable",
                field_definitions={
                    spec.id: (
                        spec.annotation,
                        Field(
                            spec.default, title=spec.name, description=spec.description
                        ),
                    )
                    for spec in config_specs
                },
            )
            if config_specs
            else None
        )

        # Many need to create a typed dict instead to implement NotRequired!
        all_fields = {
            **({"configurable": (configurable, None)} if configurable else {}),
            **{
                field_name: (field_type, None)
                for field_name, field_type in get_type_hints(RunnableConfig).items()
                if field_name in [i for i in include if i != "configurable"]
            },
        }
        model = create_model_v2(  # type: ignore[call-overload]
            self.get_name("Config"), field_definitions=all_fields
        )
        return model

    def get_config_jsonschema(
        self, *, include: Optional[Sequence[str]] = None
    ) -> dict[str, Any]:
        """Get a JSON schema that represents the config of the Runnable.

        Args:
            include: A list of fields to include in the config schema.

        Returns:
            A JSON schema that represents the config of the Runnable.

        .. versionadded:: 0.3.0
        """
        return self.config_schema(include=include).model_json_schema()

    def get_graph(self, config: Optional[RunnableConfig] = None) -> Graph:
        """Return a graph representation of this Runnable."""
        graph = Graph()
        try:
            input_node = graph.add_node(self.get_input_schema(config))
        except TypeError:
            input_node = graph.add_node(create_model_v2(self.get_name("Input")))
        runnable_node = graph.add_node(
            self, metadata=config.get("metadata") if config else None
        )
        try:
            output_node = graph.add_node(self.get_output_schema(config))
        except TypeError:
            output_node = graph.add_node(create_model_v2(self.get_name("Output")))
        graph.add_edge(input_node, runnable_node)
        graph.add_edge(runnable_node, output_node)
        return graph

    def get_prompts(
        self, config: Optional[RunnableConfig] = None
    ) -> list[BasePromptTemplate]:
        """Return a list of prompts used by this Runnable."""
        from langchain_core.prompts.base import BasePromptTemplate

        prompts = []
        for _, node in self.get_graph(config=config).nodes.items():
            if isinstance(node.data, BasePromptTemplate):
                prompts.append(node.data)
        return prompts

    def __or__(
        self,
        other: Union[
            Runnable[Any, Other],
            Callable[[Any], Other],
            Callable[[Iterator[Any]], Iterator[Other]],
            Mapping[str, Union[Runnable[Any, Other], Callable[[Any], Other], Any]],
        ],
    ) -> RunnableSerializable[Input, Other]:
        """Compose this Runnable with another object to create a RunnableSequence."""
        return RunnableSequence(self, coerce_to_runnable(other))

    def __ror__(
        self,
        other: Union[
            Runnable[Other, Any],
            Callable[[Other], Any],
            Callable[[Iterator[Other]], Iterator[Any]],
            Mapping[str, Union[Runnable[Other, Any], Callable[[Other], Any], Any]],
        ],
    ) -> RunnableSerializable[Other, Output]:
        """Compose this Runnable with another object to create a RunnableSequence."""
        return RunnableSequence(coerce_to_runnable(other), self)

    def pipe(
        self,
        *others: Union[Runnable[Any, Other], Callable[[Any], Other]],
        name: Optional[str] = None,
    ) -> RunnableSerializable[Input, Other]:
        """Compose this Runnable with Runnable-like objects to make a RunnableSequence.

        Equivalent to `RunnableSequence(self, *others)` or `self | others[0] | ...`

        Example:
            .. code-block:: python

                from langchain_core.runnables import RunnableLambda

                def add_one(x: int) -> int:
                    return x + 1

                def mul_two(x: int) -> int:
                    return x * 2

                runnable_1 = RunnableLambda(add_one)
                runnable_2 = RunnableLambda(mul_two)
                sequence = runnable_1.pipe(runnable_2)
                # Or equivalently:
                # sequence = runnable_1 | runnable_2
                # sequence = RunnableSequence(first=runnable_1, last=runnable_2)
                sequence.invoke(1)
                await sequence.ainvoke(1)
                # -> 4

                sequence.batch([1, 2, 3])
                await sequence.abatch([1, 2, 3])
                # -> [4, 6, 8]
        """
        return RunnableSequence(self, *others, name=name)

    def pick(self, keys: Union[str, list[str]]) -> RunnableSerializable[Any, Any]:
        """Pick keys from the output dict of this Runnable.

        Pick single key:
            .. code-block:: python

                import json

                from langchain_core.runnables import RunnableLambda, RunnableMap

                as_str = RunnableLambda(str)
                as_json = RunnableLambda(json.loads)
                chain = RunnableMap(str=as_str, json=as_json)

                chain.invoke("[1, 2, 3]")
                # -> {"str": "[1, 2, 3]", "json": [1, 2, 3]}

                json_only_chain = chain.pick("json")
                json_only_chain.invoke("[1, 2, 3]")
                # -> [1, 2, 3]

        Pick list of keys:
            .. code-block:: python

                from typing import Any

                import json

                from langchain_core.runnables import RunnableLambda, RunnableMap

                as_str = RunnableLambda(str)
                as_json = RunnableLambda(json.loads)
                def as_bytes(x: Any) -> bytes:
                    return bytes(x, "utf-8")

                chain = RunnableMap(
                    str=as_str,
                    json=as_json,
                    bytes=RunnableLambda(as_bytes)
                )

                chain.invoke("[1, 2, 3]")
                # -> {"str": "[1, 2, 3]", "json": [1, 2, 3], "bytes": b"[1, 2, 3]"}

                json_and_bytes_chain = chain.pick(["json", "bytes"])
                json_and_bytes_chain.invoke("[1, 2, 3]")
                # -> {"json": [1, 2, 3], "bytes": b"[1, 2, 3]"}

        """
        from langchain_core.runnables.passthrough import RunnablePick

        return self | RunnablePick(keys)

    def assign(
        self,
        **kwargs: Union[
            Runnable[dict[str, Any], Any],
            Callable[[dict[str, Any]], Any],
            Mapping[
                str,
                Union[Runnable[dict[str, Any], Any], Callable[[dict[str, Any]], Any]],
            ],
        ],
    ) -> RunnableSerializable[Any, Any]:
        """Assigns new fields to the dict output of this Runnable.
        Returns a new Runnable.

        .. code-block:: python

            from langchain_community.llms.fake import FakeStreamingListLLM
            from langchain_core.output_parsers import StrOutputParser
            from langchain_core.prompts import SystemMessagePromptTemplate
            from langchain_core.runnables import Runnable
            from operator import itemgetter

            prompt = (
                SystemMessagePromptTemplate.from_template("You are a nice assistant.")
                + "{question}"
            )
            llm = FakeStreamingListLLM(responses=["foo-lish"])

            chain: Runnable = prompt | llm | {"str": StrOutputParser()}

            chain_with_assign = chain.assign(hello=itemgetter("str") | llm)

            print(chain_with_assign.input_schema.model_json_schema())
            # {'title': 'PromptInput', 'type': 'object', 'properties':
            {'question': {'title': 'Question', 'type': 'string'}}}
            print(chain_with_assign.output_schema.model_json_schema())
            # {'title': 'RunnableSequenceOutput', 'type': 'object', 'properties':
            {'str': {'title': 'Str',
            'type': 'string'}, 'hello': {'title': 'Hello', 'type': 'string'}}}

        """
        from langchain_core.runnables.passthrough import RunnableAssign

        return self | RunnableAssign(RunnableParallel[dict[str, Any]](kwargs))

    """ --- Public API --- """

    @abstractmethod
    def invoke(
        self, input: Input, config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> Output:
        """Transform a single input into an output. Override to implement.

        Args:
            input: The input to the Runnable.
            config: A config to use when invoking the Runnable.
               The config supports standard keys like 'tags', 'metadata' for tracing
               purposes, 'max_concurrency' for controlling how much work to do
               in parallel, and other keys. Please refer to the RunnableConfig
               for more details.

        Returns:
            The output of the Runnable.
        """

    async def ainvoke(
        self, input: Input, config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> Output:
        """Default implementation of ainvoke, calls invoke from a thread.

        The default implementation allows usage of async code even if
        the Runnable did not implement a native async version of invoke.

        Subclasses should override this method if they can run asynchronously.
        """
        return await run_in_executor(config, self.invoke, input, config, **kwargs)

    def batch(
        self,
        inputs: list[Input],
        config: Optional[Union[RunnableConfig, list[RunnableConfig]]] = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Optional[Any],
    ) -> list[Output]:
        """Default implementation runs invoke in parallel using a thread pool executor.

        The default implementation of batch works well for IO bound runnables.

        Subclasses should override this method if they can batch more efficiently;
        e.g., if the underlying Runnable uses an API which supports a batch mode.
        """
        if not inputs:
            return []

        configs = get_config_list(config, len(inputs))

        def invoke(input: Input, config: RunnableConfig) -> Union[Output, Exception]:
            if return_exceptions:
                try:
                    return self.invoke(input, config, **kwargs)
                except Exception as e:
                    return e
            else:
                return self.invoke(input, config, **kwargs)

        # If there's only one input, don't bother with the executor
        if len(inputs) == 1:
            return cast(list[Output], [invoke(inputs[0], configs[0])])

        with get_executor_for_config(configs[0]) as executor:
            return cast(list[Output], list(executor.map(invoke, inputs, configs)))

    @overload
    def batch_as_completed(
        self,
        inputs: Sequence[Input],
        config: Optional[Union[RunnableConfig, Sequence[RunnableConfig]]] = None,
        *,
        return_exceptions: Literal[False] = False,
        **kwargs: Any,
    ) -> Iterator[tuple[int, Output]]: ...

    @overload
    def batch_as_completed(
        self,
        inputs: Sequence[Input],
        config: Optional[Union[RunnableConfig, Sequence[RunnableConfig]]] = None,
        *,
        return_exceptions: Literal[True],
        **kwargs: Any,
    ) -> Iterator[tuple[int, Union[Output, Exception]]]: ...

    def batch_as_completed(
        self,
        inputs: Sequence[Input],
        config: Optional[Union[RunnableConfig, Sequence[RunnableConfig]]] = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Optional[Any],
    ) -> Iterator[tuple[int, Union[Output, Exception]]]:
        """Run invoke in parallel on a list of inputs,
        yielding results as they complete.
        """
        if not inputs:
            return

        configs = get_config_list(config, len(inputs))

        def invoke(
            i: int, input: Input, config: RunnableConfig
        ) -> tuple[int, Union[Output, Exception]]:
            if return_exceptions:
                try:
                    out: Union[Output, Exception] = self.invoke(input, config, **kwargs)
                except Exception as e:
                    out = e
            else:
                out = self.invoke(input, config, **kwargs)

            return (i, out)

        if len(inputs) == 1:
            yield invoke(0, inputs[0], configs[0])
            return

        with get_executor_for_config(configs[0]) as executor:
            futures = {
                executor.submit(invoke, i, input, config)
                for i, (input, config) in enumerate(zip(inputs, configs))
            }

            try:
                while futures:
                    done, futures = wait(futures, return_when=FIRST_COMPLETED)
                    while done:
                        yield done.pop().result()
            finally:
                for future in futures:
                    future.cancel()

    async def abatch(
        self,
        inputs: list[Input],
        config: Optional[Union[RunnableConfig, list[RunnableConfig]]] = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Optional[Any],
    ) -> list[Output]:
        """Default implementation runs ainvoke in parallel using asyncio.gather.

        The default implementation of batch works well for IO bound runnables.

        Subclasses should override this method if they can batch more efficiently;
        e.g., if the underlying Runnable uses an API which supports a batch mode.

        Args:
            inputs: A list of inputs to the Runnable.
            config: A config to use when invoking the Runnable.
               The config supports standard keys like 'tags', 'metadata' for tracing
               purposes, 'max_concurrency' for controlling how much work to do
               in parallel, and other keys. Please refer to the RunnableConfig
               for more details. Defaults to None.
            return_exceptions: Whether to return exceptions instead of raising them.
                Defaults to False.
            kwargs: Additional keyword arguments to pass to the Runnable.

        Returns:
            A list of outputs from the Runnable.
        """
        if not inputs:
            return []

        configs = get_config_list(config, len(inputs))

        async def ainvoke(
            input: Input, config: RunnableConfig
        ) -> Union[Output, Exception]:
            if return_exceptions:
                try:
                    return await self.ainvoke(input, config, **kwargs)
                except Exception as e:
                    return e
            else:
                return await self.ainvoke(input, config, **kwargs)

        coros = map(ainvoke, inputs, configs)
        return await gather_with_concurrency(configs[0].get("max_concurrency"), *coros)

    @overload
    def abatch_as_completed(
        self,
        inputs: Sequence[Input],
        config: Optional[Union[RunnableConfig, Sequence[RunnableConfig]]] = None,
        *,
        return_exceptions: Literal[False] = False,
        **kwargs: Optional[Any],
    ) -> AsyncIterator[tuple[int, Output]]: ...

    @overload
    def abatch_as_completed(
        self,
        inputs: Sequence[Input],
        config: Optional[Union[RunnableConfig, Sequence[RunnableConfig]]] = None,
        *,
        return_exceptions: Literal[True],
        **kwargs: Optional[Any],
    ) -> AsyncIterator[tuple[int, Union[Output, Exception]]]: ...

    async def abatch_as_completed(
        self,
        inputs: Sequence[Input],
        config: Optional[Union[RunnableConfig, Sequence[RunnableConfig]]] = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Optional[Any],
    ) -> AsyncIterator[tuple[int, Union[Output, Exception]]]:
        """Run ainvoke in parallel on a list of inputs,
        yielding results as they complete.

        Args:
            inputs: A list of inputs to the Runnable.
            config: A config to use when invoking the Runnable.
               The config supports standard keys like 'tags', 'metadata' for tracing
               purposes, 'max_concurrency' for controlling how much work to do
               in parallel, and other keys. Please refer to the RunnableConfig
               for more details. Defaults to None. Defaults to None.
            return_exceptions: Whether to return exceptions instead of raising them.
                Defaults to False.
            kwargs: Additional keyword arguments to pass to the Runnable.

        Yields:
            A tuple of the index of the input and the output from the Runnable.
        """
        if not inputs:
            return

        configs = get_config_list(config, len(inputs))
        # Get max_concurrency from first config, defaulting to None (unlimited)
        max_concurrency = configs[0].get("max_concurrency") if configs else None
        semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency else None

        async def ainvoke_task(
            i: int, input: Input, config: RunnableConfig
        ) -> tuple[int, Union[Output, Exception]]:
            if return_exceptions:
                try:
                    out: Union[Output, Exception] = await self.ainvoke(
                        input, config, **kwargs
                    )
                except Exception as e:
                    out = e
            else:
                out = await self.ainvoke(input, config, **kwargs)
            return (i, out)

        coros = [
            gated_coro(semaphore, ainvoke_task(i, input, config))
            if semaphore
            else ainvoke_task(i, input, config)
            for i, (input, config) in enumerate(zip(inputs, configs))
        ]

        for coro in asyncio.as_completed(coros):
            yield await coro

    def stream(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> Iterator[Output]:
        """Default implementation of stream, which calls invoke.
        Subclasses should override this method if they support streaming output.

        Args:
            input: The input to the Runnable.
            config: The config to use for the Runnable. Defaults to None.
            kwargs: Additional keyword arguments to pass to the Runnable.

        Yields:
            The output of the Runnable.
        """
        yield self.invoke(input, config, **kwargs)

    async def astream(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> AsyncIterator[Output]:
        """Default implementation of astream, which calls ainvoke.
        Subclasses should override this method if they support streaming output.

        Args:
            input: The input to the Runnable.
            config: The config to use for the Runnable. Defaults to None.
            kwargs: Additional keyword arguments to pass to the Runnable.

        Yields:
            The output of the Runnable.
        """
        yield await self.ainvoke(input, config, **kwargs)

    @overload
    def astream_log(
        self,
        input: Any,
        config: Optional[RunnableConfig] = None,
        *,
        diff: Literal[True] = True,
        with_streamed_output_list: bool = True,
        include_names: Optional[Sequence[str]] = None,
        include_types: Optional[Sequence[str]] = None,
        include_tags: Optional[Sequence[str]] = None,
        exclude_names: Optional[Sequence[str]] = None,
        exclude_types: Optional[Sequence[str]] = None,
        exclude_tags: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[RunLogPatch]: ...

    @overload
    def astream_log(
        self,
        input: Any,
        config: Optional[RunnableConfig] = None,
        *,
        diff: Literal[False],
        with_streamed_output_list: bool = True,
        include_names: Optional[Sequence[str]] = None,
        include_types: Optional[Sequence[str]] = None,
        include_tags: Optional[Sequence[str]] = None,
        exclude_names: Optional[Sequence[str]] = None,
        exclude_types: Optional[Sequence[str]] = None,
        exclude_tags: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[RunLog]: ...

    async def astream_log(
        self,
        input: Any,
        config: Optional[RunnableConfig] = None,
        *,
        diff: bool = True,
        with_streamed_output_list: bool = True,
        include_names: Optional[Sequence[str]] = None,
        include_types: Optional[Sequence[str]] = None,
        include_tags: Optional[Sequence[str]] = None,
        exclude_names: Optional[Sequence[str]] = None,
        exclude_types: Optional[Sequence[str]] = None,
        exclude_tags: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> Union[AsyncIterator[RunLogPatch], AsyncIterator[RunLog]]:
        """Stream all output from a Runnable, as reported to the callback system.
        This includes all inner runs of LLMs, Retrievers, Tools, etc.

        Output is streamed as Log objects, which include a list of
        Jsonpatch ops that describe how the state of the run has changed in each
        step, and the final state of the run.

        The Jsonpatch ops can be applied in order to construct state.

        Args:
            input: The input to the Runnable.
            config: The config to use for the Runnable.
            diff: Whether to yield diffs between each step or the current state.
            with_streamed_output_list: Whether to yield the streamed_output list.
            include_names: Only include logs with these names.
            include_types: Only include logs with these types.
            include_tags: Only include logs with these tags.
            exclude_names: Exclude logs with these names.
            exclude_types: Exclude logs with these types.
            exclude_tags: Exclude logs with these tags.
            kwargs: Additional keyword arguments to pass to the Runnable.

        Yields:
            A RunLogPatch or RunLog object.
        """
        from langchain_core.tracers.log_stream import (
            LogStreamCallbackHandler,
            _astream_log_implementation,
        )

        stream = LogStreamCallbackHandler(
            auto_close=False,
            include_names=include_names,
            include_types=include_types,
            include_tags=include_tags,
            exclude_names=exclude_names,
            exclude_types=exclude_types,
            exclude_tags=exclude_tags,
            _schema_format="original",
        )

        # Mypy isn't resolving the overloads here
        # Likely an issue b/c `self` is being passed through
        # and it's can't map it to Runnable[Input,Output]?
        async for item in _astream_log_implementation(  # type: ignore
            self,
            input,
            config,
            diff=diff,
            stream=stream,
            with_streamed_output_list=with_streamed_output_list,
            **kwargs,
        ):
            yield item

    async def astream_events(
        self,
        input: Any,
        config: Optional[RunnableConfig] = None,
        *,
        version: Literal["v1", "v2"],
        include_names: Optional[Sequence[str]] = None,
        include_types: Optional[Sequence[str]] = None,
        include_tags: Optional[Sequence[str]] = None,
        exclude_names: Optional[Sequence[str]] = None,
        exclude_types: Optional[Sequence[str]] = None,
        exclude_tags: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamEvent]:
        """Generate a stream of events.

        Use to create an iterator over StreamEvents that provide real-time information
        about the progress of the Runnable, including StreamEvents from intermediate
        results.

        A StreamEvent is a dictionary with the following schema:

        - ``event``: **str** - Event names are of the
            format: on_[runnable_type]_(start|stream|end).
        - ``name``: **str** - The name of the Runnable that generated the event.
        - ``run_id``: **str** - randomly generated ID associated with the given execution of
            the Runnable that emitted the event.
            A child Runnable that gets invoked as part of the execution of a
            parent Runnable is assigned its own unique ID.
        - ``parent_ids``: **List[str]** - The IDs of the parent runnables that
            generated the event. The root Runnable will have an empty list.
            The order of the parent IDs is from the root to the immediate parent.
            Only available for v2 version of the API. The v1 version of the API
            will return an empty list.
        - ``tags``: **Optional[List[str]]** - The tags of the Runnable that generated
            the event.
        - ``metadata``: **Optional[Dict[str, Any]]** - The metadata of the Runnable
            that generated the event.
        - ``data``: **Dict[str, Any]**


        Below is a table that illustrates some events that might be emitted by various
        chains. Metadata fields have been omitted from the table for brevity.
        Chain definitions have been included after the table.

        **ATTENTION** This reference table is for the V2 version of the schema.

        +----------------------+------------------+---------------------------------+-----------------------------------------------+-------------------------------------------------+
        | event                | name             | chunk                           | input                                         | output                                          |
        +======================+==================+=================================+===============================================+=================================================+
        | on_chat_model_start  | [model name]     |                                 | {"messages": [[SystemMessage, HumanMessage]]} |                                                 |
        +----------------------+------------------+---------------------------------+-----------------------------------------------+-------------------------------------------------+
        | on_chat_model_stream | [model name]     | AIMessageChunk(content="hello") |                                               |                                                 |
        +----------------------+------------------+---------------------------------+-----------------------------------------------+-------------------------------------------------+
        | on_chat_model_end    | [model name]     |                                 | {"messages": [[SystemMessage, HumanMessage]]} | AIMessageChunk(content="hello world")           |
        +----------------------+------------------+---------------------------------+-----------------------------------------------+-------------------------------------------------+
        | on_llm_start         | [model name]     |                                 | {'input': 'hello'}                            |                                                 |
        +----------------------+------------------+---------------------------------+-----------------------------------------------+-------------------------------------------------+
        | on_llm_stream        | [model name]     | 'Hello'                         |                                               |                                                 |
        +----------------------+------------------+---------------------------------+-----------------------------------------------+-------------------------------------------------+
        | on_llm_end           | [model name]     |                                 | 'Hello human!'                                |                                                 |
        +----------------------+------------------+---------------------------------+-----------------------------------------------+-------------------------------------------------+
        | on_chain_start       | format_docs      |                                 |                                               |                                                 |
        +----------------------+------------------+---------------------------------+-----------------------------------------------+-------------------------------------------------+
        | on_chain_stream      | format_docs      | "hello world!, goodbye world!"  |                                               |                                                 |
        +----------------------+------------------+---------------------------------+-----------------------------------------------+-------------------------------------------------+
        | on_chain_end         | format_docs      |                                 | [Document(...)]                               | "hello world!, goodbye world!"                  |
        +----------------------+------------------+---------------------------------+-----------------------------------------------+-------------------------------------------------+
        | on_tool_start        | some_tool        |                                 | {"x": 1, "y": "2"}                            |                                                 |
        +----------------------+------------------+---------------------------------+-----------------------------------------------+-------------------------------------------------+
        | on_tool_end          | some_tool        |                                 |                                               | {"x": 1, "y": "2"}                              |
        +----------------------+------------------+---------------------------------+-----------------------------------------------+-------------------------------------------------+
        | on_retriever_start   | [retriever name] |                                 | {"query": "hello"}                            |                                                 |
        +----------------------+------------------+---------------------------------+-----------------------------------------------+-------------------------------------------------+
        | on_retriever_end     | [retriever name] |                                 | {"query": "hello"}                            | [Document(...), ..]                             |
        +----------------------+------------------+---------------------------------+-----------------------------------------------+-------------------------------------------------+
        | on_prompt_start      | [template_name]  |                                 | {"question": "hello"}                         |                                                 |
        +----------------------+------------------+---------------------------------+-----------------------------------------------+-------------------------------------------------+
        | on_prompt_end        | [template_name]  |                                 | {"question": "hello"}                         | ChatPromptValue(messages: [SystemMessage, ...]) |
        +----------------------+------------------+---------------------------------+-----------------------------------------------+-------------------------------------------------+

        In addition to the standard events, users can also dispatch custom events (see example below).

        Custom events will be only be surfaced with in the `v2` version of the API!

        A custom event has following format:

        +-----------+------+-----------------------------------------------------------------------------------------------------------+
        | Attribute | Type | Description                                                                                               |
        +===========+======+===========================================================================================================+
        | name      | str  | A user defined name for the event.                                                                        |
        +-----------+------+-----------------------------------------------------------------------------------------------------------+
        | data      | Any  | The data associated with the event. This can be anything, though we suggest making it JSON serializable.  |
        +-----------+------+-----------------------------------------------------------------------------------------------------------+

        Here are declarations associated with the standard events shown above:

        `format_docs`:

        .. code-block:: python

            def format_docs(docs: List[Document]) -> str:
                '''Format the docs.'''
                return ", ".join([doc.page_content for doc in docs])

            format_docs = RunnableLambda(format_docs)

        `some_tool`:

        .. code-block:: python

            @tool
            def some_tool(x: int, y: str) -> dict:
                '''Some_tool.'''
                return {"x": x, "y": y}

        `prompt`:

        .. code-block:: python

            template = ChatPromptTemplate.from_messages(
                [("system", "You are Cat Agent 007"), ("human", "{question}")]
            ).with_config({"run_name": "my_template", "tags": ["my_template"]})


        Example:

        .. code-block:: python

            from langchain_core.runnables import RunnableLambda

            async def reverse(s: str) -> str:
                return s[::-1]

            chain = RunnableLambda(func=reverse)

            events = [
                event async for event in chain.astream_events("hello", version="v2")
            ]

            # will produce the following events (run_id, and parent_ids
            # has been omitted for brevity):
            [
                {
                    "data": {"input": "hello"},
                    "event": "on_chain_start",
                    "metadata": {},
                    "name": "reverse",
                    "tags": [],
                },
                {
                    "data": {"chunk": "olleh"},
                    "event": "on_chain_stream",
                    "metadata": {},
                    "name": "reverse",
                    "tags": [],
                },
                {
                    "data": {"output": "olleh"},
                    "event": "on_chain_end",
                    "metadata": {},
                    "name": "reverse",
                    "tags": [],
                },
            ]


        Example: Dispatch Custom Event

        .. code-block:: python

            from langchain_core.callbacks.manager import (
                adispatch_custom_event,
            )
            from langchain_core.runnables import RunnableLambda, RunnableConfig
            import asyncio


            async def slow_thing(some_input: str, config: RunnableConfig) -> str:
                \"\"\"Do something that takes a long time.\"\"\"
                await asyncio.sleep(1) # Placeholder for some slow operation
                await adispatch_custom_event(
                    "progress_event",
                    {"message": "Finished step 1 of 3"},
                    config=config # Must be included for python < 3.10
                )
                await asyncio.sleep(1) # Placeholder for some slow operation
                await adispatch_custom_event(
                    "progress_event",
                    {"message": "Finished step 2 of 3"},
                    config=config # Must be included for python < 3.10
                )
                await asyncio.sleep(1) # Placeholder for some slow operation
                return "Done"

            slow_thing = RunnableLambda(slow_thing)

            async for event in slow_thing.astream_events("some_input", version="v2"):
                print(event)

        Args:
            input: The input to the Runnable.
            config: The config to use for the Runnable.
            version: The version of the schema to use either `v2` or `v1`.
                     Users should use `v2`.
                     `v1` is for backwards compatibility and will be deprecated
                     in 0.4.0.
                     No default will be assigned until the API is stabilized.
                     custom events will only be surfaced in `v2`.
            include_names: Only include events from runnables with matching names.
            include_types: Only include events from runnables with matching types.
            include_tags: Only include events from runnables with matching tags.
            exclude_names: Exclude events from runnables with matching names.
            exclude_types: Exclude events from runnables with matching types.
            exclude_tags: Exclude events from runnables with matching tags.
            kwargs: Additional keyword arguments to pass to the Runnable.
                These will be passed to astream_log as this implementation
                of astream_events is built on top of astream_log.

        Yields:
            An async stream of StreamEvents.

        Raises:
            NotImplementedError: If the version is not `v1` or `v2`.
        """  # noqa: E501
        from langchain_core.tracers.event_stream import (
            _astream_events_implementation_v1,
            _astream_events_implementation_v2,
        )

        if version == "v2":
            event_stream = _astream_events_implementation_v2(
                self,
                input,
                config=config,
                include_names=include_names,
                include_types=include_types,
                include_tags=include_tags,
                exclude_names=exclude_names,
                exclude_types=exclude_types,
                exclude_tags=exclude_tags,
                **kwargs,
            )
        elif version == "v1":
            # First implementation, built on top of astream_log API
            # This implementation will be deprecated as of 0.2.0
            event_stream = _astream_events_implementation_v1(
                self,
                input,
                config=config,
                include_names=include_names,
                include_types=include_types,
                include_tags=include_tags,
                exclude_names=exclude_names,
                exclude_types=exclude_types,
                exclude_tags=exclude_tags,
                **kwargs,
            )
        else:
            msg = 'Only versions "v1" and "v2" of the schema is currently supported.'
            raise NotImplementedError(msg)

        async with aclosing(event_stream):
            async for event in event_stream:
                yield event

    def transform(
        self,
        input: Iterator[Input],
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> Iterator[Output]:
        """Default implementation of transform, which buffers input and calls astream.

        Subclasses should override this method if they can start producing output while
        input is still being generated.

        Args:
            input: An iterator of inputs to the Runnable.
            config: The config to use for the Runnable. Defaults to None.
            kwargs: Additional keyword arguments to pass to the Runnable.

        Yields:
            The output of the Runnable.
        """
        final: Input
        got_first_val = False

        for ichunk in input:
            # The default implementation of transform is to buffer input and
            # then call stream.
            # It'll attempt to gather all input into a single chunk using
            # the `+` operator.
            # If the input is not addable, then we'll assume that we can
            # only operate on the last chunk,
            # and we'll iterate until we get to the last chunk.
            if not got_first_val:
                final = ichunk
                got_first_val = True
            else:
                try:
                    final = final + ichunk  # type: ignore[operator]
                except TypeError:
                    final = ichunk

        if got_first_val:
            yield from self.stream(final, config, **kwargs)

    async def atransform(
        self,
        input: AsyncIterator[Input],
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> AsyncIterator[Output]:
        """Default implementation of atransform, which buffers input and calls astream.
        Subclasses should override this method if they can start producing output while
        input is still being generated.

        Args:
            input: An async iterator of inputs to the Runnable.
            config: The config to use for the Runnable. Defaults to None.
            kwargs: Additional keyword arguments to pass to the Runnable.

        Yields:
            The output of the Runnable.
        """
        final: Input
        got_first_val = False

        async for ichunk in input:
            # The default implementation of transform is to buffer input and
            # then call stream.
            # It'll attempt to gather all input into a single chunk using
            # the `+` operator.
            # If the input is not addable, then we'll assume that we can
            # only operate on the last chunk,
            # and we'll iterate until we get to the last chunk.
            if not got_first_val:
                final = ichunk
                got_first_val = True
            else:
                try:
                    final = final + ichunk  # type: ignore[operator]
                except TypeError:
                    final = ichunk

        if got_first_val:
            async for output in self.astream(final, config, **kwargs):
                yield output

    def bind(self, **kwargs: Any) -> Runnable[Input, Output]:
        """Bind arguments to a Runnable, returning a new Runnable.

        Useful when a Runnable in a chain requires an argument that is not
        in the output of the previous Runnable or included in the user input.

        Args:
            kwargs: The arguments to bind to the Runnable.

        Returns:
            A new Runnable with the arguments bound.

        Example:

        .. code-block:: python

            from langchain_community.chat_models import ChatOllama
            from langchain_core.output_parsers import StrOutputParser

            llm = ChatOllama(model='llama2')

            # Without bind.
            chain = (
                llm
                | StrOutputParser()
            )

            chain.invoke("Repeat quoted words exactly: 'One two three four five.'")
            # Output is 'One two three four five.'

            # With bind.
            chain = (
                llm.bind(stop=["three"])
                | StrOutputParser()
            )

            chain.invoke("Repeat quoted words exactly: 'One two three four five.'")
            # Output is 'One two'

        """
        return RunnableBinding(bound=self, kwargs=kwargs, config={})

    def with_config(
        self,
        config: Optional[RunnableConfig] = None,
        # Sadly Unpack is not well-supported by mypy so this will have to be untyped
        **kwargs: Any,
    ) -> Runnable[Input, Output]:
        """Bind config to a Runnable, returning a new Runnable.

        Args:
            config: The config to bind to the Runnable.
            kwargs: Additional keyword arguments to pass to the Runnable.

        Returns:
            A new Runnable with the config bound.
        """
        return RunnableBinding(
            bound=self,
            config=cast(
                RunnableConfig,
                {**(config or {}), **kwargs},
            ),  # type: ignore[misc]
            kwargs={},
        )

    def with_listeners(
        self,
        *,
        on_start: Optional[
            Union[Callable[[Run], None], Callable[[Run, RunnableConfig], None]]
        ] = None,
        on_end: Optional[
            Union[Callable[[Run], None], Callable[[Run, RunnableConfig], None]]
        ] = None,
        on_error: Optional[
            Union[Callable[[Run], None], Callable[[Run, RunnableConfig], None]]
        ] = None,
    ) -> Runnable[Input, Output]:
        """Bind lifecycle listeners to a Runnable, returning a new Runnable.

        on_start: Called before the Runnable starts running, with the Run object.
        on_end: Called after the Runnable finishes running, with the Run object.
        on_error: Called if the Runnable throws an error, with the Run object.

        The Run object contains information about the run, including its id,
        type, input, output, error, start_time, end_time, and any tags or metadata
        added to the run.

        Args:
            on_start: Called before the Runnable starts running. Defaults to None.
            on_end: Called after the Runnable finishes running. Defaults to None.
            on_error: Called if the Runnable throws an error. Defaults to None.

        Returns:
            A new Runnable with the listeners bound.

        Example:

        .. code-block:: python

            from langchain_core.runnables import RunnableLambda
            from langchain_core.tracers.schemas import Run

            import time

            def test_runnable(time_to_sleep : int):
                time.sleep(time_to_sleep)

            def fn_start(run_obj: Run):
                print("start_time:", run_obj.start_time)

            def fn_end(run_obj: Run):
                print("end_time:", run_obj.end_time)

            chain = RunnableLambda(test_runnable).with_listeners(
                on_start=fn_start,
                on_end=fn_end
            )
            chain.invoke(2)
        """
        from langchain_core.tracers.root_listeners import RootListenersTracer

        return RunnableBinding(
            bound=self,
            config_factories=[
                lambda config: {
                    "callbacks": [
                        RootListenersTracer(
                            config=config,
                            on_start=on_start,
                            on_end=on_end,
                            on_error=on_error,
                        )
                    ],
                }
            ],
        )

    def with_alisteners(
        self,
        *,
        on_start: Optional[AsyncListener] = None,
        on_end: Optional[AsyncListener] = None,
        on_error: Optional[AsyncListener] = None,
    ) -> Runnable[Input, Output]:
        """Bind async lifecycle listeners to a Runnable, returning a new Runnable.

        on_start: Asynchronously called before the Runnable starts running.
        on_end: Asynchronously called after the Runnable finishes running.
        on_error: Asynchronously called if the Runnable throws an error.

        The Run object contains information about the run, including its id,
        type, input, output, error, start_time, end_time, and any tags or metadata
        added to the run.

        Args:
            on_start: Asynchronously called before the Runnable starts running.
                Defaults to None.
            on_end: Asynchronously called after the Runnable finishes running.
                Defaults to None.
            on_error: Asynchronously called if the Runnable throws an error.
                Defaults to None.

        Returns:
            A new Runnable with the listeners bound.

        Example:

        .. code-block:: python

            from langchain_core.runnables import RunnableLambda
            import time

            async def test_runnable(time_to_sleep : int):
                print(f"Runnable[{time_to_sleep}s]: starts at {format_t(time.time())}")
                await asyncio.sleep(time_to_sleep)
                print(f"Runnable[{time_to_sleep}s]: ends at {format_t(time.time())}")

            async def fn_start(run_obj : Runnable):
                print(f"on start callback starts at {format_t(time.time())}
                await asyncio.sleep(3)
                print(f"on start callback ends at {format_t(time.time())}")

            async def fn_end(run_obj : Runnable):
                print(f"on end callback starts at {format_t(time.time())}
                await asyncio.sleep(2)
                print(f"on end callback ends at {format_t(time.time())}")

            runnable = RunnableLambda(test_runnable).with_alisteners(
                on_start=fn_start,
                on_end=fn_end
            )
            async def concurrent_runs():
                await asyncio.gather(runnable.ainvoke(2), runnable.ainvoke(3))

            asyncio.run(concurrent_runs())
            Result:
            on start callback starts at 2024-05-16T14:20:29.637053+00:00
            on start callback starts at 2024-05-16T14:20:29.637150+00:00
            on start callback ends at 2024-05-16T14:20:32.638305+00:00
            on start callback ends at 2024-05-16T14:20:32.638383+00:00
            Runnable[3s]: starts at 2024-05-16T14:20:32.638849+00:00
            Runnable[5s]: starts at 2024-05-16T14:20:32.638999+00:00
            Runnable[3s]: ends at 2024-05-16T14:20:35.640016+00:00
            on end callback starts at 2024-05-16T14:20:35.640534+00:00
            Runnable[5s]: ends at 2024-05-16T14:20:37.640169+00:00
            on end callback starts at 2024-05-16T14:20:37.640574+00:00
            on end callback ends at 2024-05-16T14:20:37.640654+00:00
            on end callback ends at 2024-05-16T14:20:39.641751+00:00

        """
        from langchain_core.tracers.root_listeners import AsyncRootListenersTracer

        return RunnableBinding(
            bound=self,
            config_factories=[
                lambda config: {
                    "callbacks": [
                        AsyncRootListenersTracer(
                            config=config,
                            on_start=on_start,
                            on_end=on_end,
                            on_error=on_error,
                        )
                    ],
                }
            ],
        )

    def with_types(
        self,
        *,
        input_type: Optional[type[Input]] = None,
        output_type: Optional[type[Output]] = None,
    ) -> Runnable[Input, Output]:
        """Bind input and output types to a Runnable, returning a new Runnable.

        Args:
            input_type: The input type to bind to the Runnable. Defaults to None.
            output_type: The output type to bind to the Runnable. Defaults to None.

        Returns:
            A new Runnable with the types bound.
        """
        return RunnableBinding(
            bound=self,
            custom_input_type=input_type,
            custom_output_type=output_type,
            kwargs={},
        )

    def with_retry(
        self,
        *,
        retry_if_exception_type: tuple[type[BaseException], ...] = (Exception,),
        wait_exponential_jitter: bool = True,
        stop_after_attempt: int = 3,
    ) -> Runnable[Input, Output]:
        """Create a new Runnable that retries the original Runnable on exceptions.

        Args:
            retry_if_exception_type: A tuple of exception types to retry on.
                Defaults to (Exception,).
            wait_exponential_jitter: Whether to add jitter to the wait
                time between retries. Defaults to True.
            stop_after_attempt: The maximum number of attempts to make before
                giving up. Defaults to 3.

        Returns:
            A new Runnable that retries the original Runnable on exceptions.

        Example:

        .. code-block:: python

            from langchain_core.runnables import RunnableLambda

            count = 0


            def _lambda(x: int) -> None:
                global count
                count = count + 1
                if x == 1:
                    raise ValueError("x is 1")
                else:
                     pass


            runnable = RunnableLambda(_lambda)
            try:
                runnable.with_retry(
                    stop_after_attempt=2,
                    retry_if_exception_type=(ValueError,),
                ).invoke(1)
            except ValueError:
                pass

            assert (count == 2)


        Args:
            retry_if_exception_type: A tuple of exception types to retry on
            wait_exponential_jitter: Whether to add jitter to the wait time
                                     between retries
            stop_after_attempt: The maximum number of attempts to make before giving up

        Returns:
            A new Runnable that retries the original Runnable on exceptions.
        """
        from langchain_core.runnables.retry import RunnableRetry

        return RunnableRetry(
            bound=self,
            kwargs={},
            config={},
            retry_exception_types=retry_if_exception_type,
            wait_exponential_jitter=wait_exponential_jitter,
            max_attempt_number=stop_after_attempt,
        )

    def map(self) -> Runnable[list[Input], list[Output]]:
        """Return a new Runnable that maps a list of inputs to a list of outputs,
        by calling invoke() with each input.

        Returns:
            A new Runnable that maps a list of inputs to a list of outputs.

        Example:

            .. code-block:: python

                    from langchain_core.runnables import RunnableLambda

                    def _lambda(x: int) -> int:
                        return x + 1

                    runnable = RunnableLambda(_lambda)
                    print(runnable.map().invoke([1, 2, 3])) # [2, 3, 4]
        """
        return RunnableEach(bound=self)

    def with_fallbacks(
        self,
        fallbacks: Sequence[Runnable[Input, Output]],
        *,
        exceptions_to_handle: tuple[type[BaseException], ...] = (Exception,),
        exception_key: Optional[str] = None,
    ) -> RunnableWithFallbacksT[Input, Output]:
        """Add fallbacks to a Runnable, returning a new Runnable.

        The new Runnable will try the original Runnable, and then each fallback
        in order, upon failures.

        Args:
            fallbacks: A sequence of runnables to try if the original Runnable fails.
            exceptions_to_handle: A tuple of exception types to handle.
                Defaults to (Exception,).
            exception_key: If string is specified then handled exceptions will be passed
                to fallbacks as part of the input under the specified key. If None,
                exceptions will not be passed to fallbacks. If used, the base Runnable
                and its fallbacks must accept a dictionary as input. Defaults to None.

        Returns:
            A new Runnable that will try the original Runnable, and then each
            fallback in order, upon failures.

        Example:

            .. code-block:: python

                from typing import Iterator

                from langchain_core.runnables import RunnableGenerator


                def _generate_immediate_error(input: Iterator) -> Iterator[str]:
                    raise ValueError()
                    yield ""


                def _generate(input: Iterator) -> Iterator[str]:
                    yield from "foo bar"


                runnable = RunnableGenerator(_generate_immediate_error).with_fallbacks(
                    [RunnableGenerator(_generate)]
                    )
                print(''.join(runnable.stream({}))) #foo bar

        Args:
            fallbacks: A sequence of runnables to try if the original Runnable fails.
            exceptions_to_handle: A tuple of exception types to handle.
            exception_key: If string is specified then handled exceptions will be passed
                to fallbacks as part of the input under the specified key. If None,
                exceptions will not be passed to fallbacks. If used, the base Runnable
                and its fallbacks must accept a dictionary as input.

        Returns:
            A new Runnable that will try the original Runnable, and then each
            fallback in order, upon failures.

        """
        from langchain_core.runnables.fallbacks import RunnableWithFallbacks

        return RunnableWithFallbacks(
            runnable=self,
            fallbacks=fallbacks,
            exceptions_to_handle=exceptions_to_handle,
            exception_key=exception_key,
        )

    """ --- Helper methods for Subclasses --- """

    def _call_with_config(
        self,
        func: Union[
            Callable[[Input], Output],
            Callable[[Input, CallbackManagerForChainRun], Output],
            Callable[[Input, CallbackManagerForChainRun, RunnableConfig], Output],
        ],
        input: Input,
        config: Optional[RunnableConfig],
        run_type: Optional[str] = None,
        serialized: Optional[dict[str, Any]] = None,
        **kwargs: Optional[Any],
    ) -> Output:
        """Helper method to transform an Input value to an Output value,
        with callbacks. Use this method to implement invoke() in subclasses.
        """
        config = ensure_config(config)
        callback_manager = get_callback_manager_for_config(config)
        run_manager = callback_manager.on_chain_start(
            serialized,
            input,
            run_type=run_type,
            name=config.get("run_name") or self.get_name(),
            run_id=config.pop("run_id", None),
        )
        try:
            child_config = patch_config(config, callbacks=run_manager.get_child())
            context = copy_context()
            context.run(_set_config_context, child_config)
            output = cast(
                Output,
                context.run(
                    call_func_with_variable_args,  # type: ignore[arg-type]
                    func,  # type: ignore[arg-type]
                    input,  # type: ignore[arg-type]
                    config,
                    run_manager,
                    **kwargs,
                ),
            )
        except BaseException as e:
            run_manager.on_chain_error(e)
            raise
        else:
            run_manager.on_chain_end(output)
            return output

    async def _acall_with_config(
        self,
        func: Union[
            Callable[[Input], Awaitable[Output]],
            Callable[[Input, AsyncCallbackManagerForChainRun], Awaitable[Output]],
            Callable[
                [Input, AsyncCallbackManagerForChainRun, RunnableConfig],
                Awaitable[Output],
            ],
        ],
        input: Input,
        config: Optional[RunnableConfig],
        run_type: Optional[str] = None,
        serialized: Optional[dict[str, Any]] = None,
        **kwargs: Optional[Any],
    ) -> Output:
        """Helper method to transform an Input value to an Output value,
        with callbacks. Use this method to implement ainvoke() in subclasses.
        """
        config = ensure_config(config)
        callback_manager = get_async_callback_manager_for_config(config)
        run_manager = await callback_manager.on_chain_start(
            serialized,
            input,
            run_type=run_type,
            name=config.get("run_name") or self.get_name(),
            run_id=config.pop("run_id", None),
        )
        try:
            child_config = patch_config(config, callbacks=run_manager.get_child())
            context = copy_context()
            context.run(_set_config_context, child_config)
            coro = acall_func_with_variable_args(
                func, input, config, run_manager, **kwargs
            )
            if asyncio_accepts_context():
                output: Output = await asyncio.create_task(coro, context=context)  # type: ignore
            else:
                output = await coro
        except BaseException as e:
            await run_manager.on_chain_error(e)
            raise
        else:
            await run_manager.on_chain_end(output)
            return output

    def _batch_with_config(
        self,
        func: Union[
            Callable[[list[Input]], list[Union[Exception, Output]]],
            Callable[
                [list[Input], list[CallbackManagerForChainRun]],
                list[Union[Exception, Output]],
            ],
            Callable[
                [list[Input], list[CallbackManagerForChainRun], list[RunnableConfig]],
                list[Union[Exception, Output]],
            ],
        ],
        input: list[Input],
        config: Optional[Union[RunnableConfig, list[RunnableConfig]]] = None,
        *,
        return_exceptions: bool = False,
        run_type: Optional[str] = None,
        **kwargs: Optional[Any],
    ) -> list[Output]:
        """Helper method to transform an Input value to an Output value,
        with callbacks. Use this method to implement invoke() in subclasses.
        """
        if not input:
            return []

        configs = get_config_list(config, len(input))
        callback_managers = [get_callback_manager_for_config(c) for c in configs]
        run_managers = [
            callback_manager.on_chain_start(
                None,
                input,
                run_type=run_type,
                name=config.get("run_name") or self.get_name(),
                run_id=config.pop("run_id", None),
            )
            for callback_manager, input, config in zip(
                callback_managers, input, configs
            )
        ]
        try:
            if accepts_config(func):
                kwargs["config"] = [
                    patch_config(c, callbacks=rm.get_child())
                    for c, rm in zip(configs, run_managers)
                ]
            if accepts_run_manager(func):
                kwargs["run_manager"] = run_managers
            output = func(input, **kwargs)  # type: ignore[call-arg]
        except BaseException as e:
            for run_manager in run_managers:
                run_manager.on_chain_error(e)
            if return_exceptions:
                return cast(list[Output], [e for _ in input])
            else:
                raise
        else:
            first_exception: Optional[Exception] = None
            for run_manager, out in zip(run_managers, output):
                if isinstance(out, Exception):
                    first_exception = first_exception or out
                    run_manager.on_chain_error(out)
                else:
                    run_manager.on_chain_end(out)
            if return_exceptions or first_exception is None:
                return cast(list[Output], output)
            else:
                raise first_exception

    async def _abatch_with_config(
        self,
        func: Union[
            Callable[[list[Input]], Awaitable[list[Union[Exception, Output]]]],
            Callable[
                [list[Input], list[AsyncCallbackManagerForChainRun]],
                Awaitable[list[Union[Exception, Output]]],
            ],
            Callable[
                [
                    list[Input],
                    list[AsyncCallbackManagerForChainRun],
                    list[RunnableConfig],
                ],
                Awaitable[list[Union[Exception, Output]]],
            ],
        ],
        input: list[Input],
        config: Optional[Union[RunnableConfig, list[RunnableConfig]]] = None,
        *,
        return_exceptions: bool = False,
        run_type: Optional[str] = None,
        **kwargs: Optional[Any],
    ) -> list[Output]:
        """Helper method to transform an Input value to an Output value,
        with callbacks. Use this method to implement invoke() in subclasses.
        """
        if not input:
            return []

        configs = get_config_list(config, len(input))
        callback_managers = [get_async_callback_manager_for_config(c) for c in configs]
        run_managers: list[AsyncCallbackManagerForChainRun] = await asyncio.gather(
            *(
                callback_manager.on_chain_start(
                    None,
                    input,
                    run_type=run_type,
                    name=config.get("run_name") or self.get_name(),
                    run_id=config.pop("run_id", None),
                )
                for callback_manager, input, config in zip(
                    callback_managers, input, configs
                )
            )
        )
        try:
            if accepts_config(func):
                kwargs["config"] = [
                    patch_config(c, callbacks=rm.get_child())
                    for c, rm in zip(configs, run_managers)
                ]
            if accepts_run_manager(func):
                kwargs["run_manager"] = run_managers
            output = await func(input, **kwargs)  # type: ignore[call-arg]
        except BaseException as e:
            await asyncio.gather(
                *(run_manager.on_chain_error(e) for run_manager in run_managers)
            )
            if return_exceptions:
                return cast(list[Output], [e for _ in input])
            else:
                raise
        else:
            first_exception: Optional[Exception] = None
            coros: list[Awaitable[None]] = []
            for run_manager, out in zip(run_managers, output):
                if isinstance(out, Exception):
                    first_exception = first_exception or out
                    coros.append(run_manager.on_chain_error(out))
                else:
                    coros.append(run_manager.on_chain_end(out))
            await asyncio.gather(*coros)
            if return_exceptions or first_exception is None:
                return cast(list[Output], output)
            else:
                raise first_exception

    def _transform_stream_with_config(
        self,
        input: Iterator[Input],
        transformer: Union[
            Callable[[Iterator[Input]], Iterator[Output]],
            Callable[[Iterator[Input], CallbackManagerForChainRun], Iterator[Output]],
            Callable[
                [
                    Iterator[Input],
                    CallbackManagerForChainRun,
                    RunnableConfig,
                ],
                Iterator[Output],
            ],
        ],
        config: Optional[RunnableConfig],
        run_type: Optional[str] = None,
        **kwargs: Optional[Any],
    ) -> Iterator[Output]:
        """Helper method to transform an Iterator of Input values into an Iterator of
        Output values, with callbacks.
        Use this to implement `stream()` or `transform()` in Runnable subclasses.
        """
        # Mixin that is used by both astream log and astream events implementation
        from langchain_core.tracers._streaming import _StreamingCallbackHandler

        # tee the input so we can iterate over it twice
        input_for_tracing, input_for_transform = tee(input, 2)
        # Start the input iterator to ensure the input Runnable starts before this one
        final_input: Optional[Input] = next(input_for_tracing, None)
        final_input_supported = True
        final_output: Optional[Output] = None
        final_output_supported = True

        config = ensure_config(config)
        callback_manager = get_callback_manager_for_config(config)
        run_manager = callback_manager.on_chain_start(
            None,
            {"input": ""},
            run_type=run_type,
            name=config.get("run_name") or self.get_name(),
            run_id=config.pop("run_id", None),
        )
        try:
            child_config = patch_config(config, callbacks=run_manager.get_child())
            if accepts_config(transformer):
                kwargs["config"] = child_config
            if accepts_run_manager(transformer):
                kwargs["run_manager"] = run_manager
            context = copy_context()
            context.run(_set_config_context, child_config)
            iterator = context.run(transformer, input_for_transform, **kwargs)  # type: ignore[arg-type]
            if stream_handler := next(
                (
                    cast(_StreamingCallbackHandler, h)
                    for h in run_manager.handlers
                    # instance check OK here, it's a mixin
                    if isinstance(h, _StreamingCallbackHandler)  # type: ignore[misc]
                ),
                None,
            ):
                # populates streamed_output in astream_log() output if needed
                iterator = stream_handler.tap_output_iter(run_manager.run_id, iterator)
            try:
                while True:
                    chunk: Output = context.run(next, iterator)  # type: ignore
                    yield chunk
                    if final_output_supported:
                        if final_output is None:
                            final_output = chunk
                        else:
                            try:
                                final_output = final_output + chunk  # type: ignore
                            except TypeError:
                                final_output = chunk
                                final_output_supported = False
                    else:
                        final_output = chunk
            except (StopIteration, GeneratorExit):
                pass
            for ichunk in input_for_tracing:
                if final_input_supported:
                    if final_input is None:
                        final_input = ichunk
                    else:
                        try:
                            final_input = final_input + ichunk  # type: ignore
                        except TypeError:
                            final_input = ichunk
                            final_input_supported = False
                else:
                    final_input = ichunk
        except BaseException as e:
            run_manager.on_chain_error(e, inputs=final_input)
            raise
        else:
            run_manager.on_chain_end(final_output, inputs=final_input)

    async def _atransform_stream_with_config(
        self,
        input: AsyncIterator[Input],
        transformer: Union[
            Callable[[AsyncIterator[Input]], AsyncIterator[Output]],
            Callable[
                [AsyncIterator[Input], AsyncCallbackManagerForChainRun],
                AsyncIterator[Output],
            ],
            Callable[
                [
                    AsyncIterator[Input],
                    AsyncCallbackManagerForChainRun,
                    RunnableConfig,
                ],
                AsyncIterator[Output],
            ],
        ],
        config: Optional[RunnableConfig],
        run_type: Optional[str] = None,
        **kwargs: Optional[Any],
    ) -> AsyncIterator[Output]:
        """Helper method to transform an Async Iterator of Input values into an Async
        Iterator of Output values, with callbacks.
        Use this to implement `astream()` or `atransform()` in Runnable subclasses.
        """
        # Mixin that is used by both astream log and astream events implementation
        from langchain_core.tracers._streaming import _StreamingCallbackHandler

        # tee the input so we can iterate over it twice
        input_for_tracing, input_for_transform = atee(input, 2)
        # Start the input iterator to ensure the input Runnable starts before this one
        final_input: Optional[Input] = await py_anext(input_for_tracing, None)
        final_input_supported = True
        final_output: Optional[Output] = None
        final_output_supported = True

        config = ensure_config(config)
        callback_manager = get_async_callback_manager_for_config(config)
        run_manager = await callback_manager.on_chain_start(
            None,
            {"input": ""},
            run_type=run_type,
            name=config.get("run_name") or self.get_name(),
            run_id=config.pop("run_id", None),
        )
        try:
            child_config = patch_config(config, callbacks=run_manager.get_child())
            if accepts_config(transformer):
                kwargs["config"] = child_config
            if accepts_run_manager(transformer):
                kwargs["run_manager"] = run_manager
            context = copy_context()
            context.run(_set_config_context, child_config)
            iterator_ = context.run(transformer, input_for_transform, **kwargs)  # type: ignore[arg-type]

            if stream_handler := next(
                (
                    cast(_StreamingCallbackHandler, h)
                    for h in run_manager.handlers
                    # instance check OK here, it's a mixin
                    if isinstance(h, _StreamingCallbackHandler)  # type: ignore[misc]
                ),
                None,
            ):
                # populates streamed_output in astream_log() output if needed
                iterator = stream_handler.tap_output_aiter(
                    run_manager.run_id, iterator_
                )
            else:
                iterator = iterator_
            try:
                while True:
                    if asyncio_accepts_context():
                        chunk: Output = await asyncio.create_task(  # type: ignore[call-arg]
                            py_anext(iterator),  # type: ignore[arg-type]
                            context=context,
                        )
                    else:
                        chunk = cast(Output, await py_anext(iterator))
                    yield chunk
                    if final_output_supported:
                        if final_output is None:
                            final_output = chunk
                        else:
                            try:
                                final_output = final_output + chunk  # type: ignore
                            except TypeError:
                                final_output = chunk
                                final_output_supported = False
                    else:
                        final_output = chunk
            except StopAsyncIteration:
                pass
            async for ichunk in input_for_tracing:
                if final_input_supported:
                    if final_input is None:
                        final_input = ichunk
                    else:
                        try:
                            final_input = final_input + ichunk  # type: ignore[operator]
                        except TypeError:
                            final_input = ichunk
                            final_input_supported = False
                else:
                    final_input = ichunk
        except BaseException as e:
            await run_manager.on_chain_error(e, inputs=final_input)
            raise
        else:
            await run_manager.on_chain_end(final_output, inputs=final_input)
        finally:
            if iterator_ is not None and hasattr(iterator_, "aclose"):
                await iterator_.aclose()

    @beta_decorator.beta(message="This API is in beta and may change in the future.")
    def as_tool(
        self,
        args_schema: Optional[type[BaseModel]] = None,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        arg_types: Optional[dict[str, type]] = None,
    ) -> BaseTool:
        """Create a BaseTool from a Runnable.

        ``as_tool`` will instantiate a BaseTool with a name, description, and
        ``args_schema`` from a Runnable. Where possible, schemas are inferred
        from ``runnable.get_input_schema``. Alternatively (e.g., if the
        Runnable takes a dict as input and the specific dict keys are not typed),
        the schema can be specified directly with ``args_schema``. You can also
        pass ``arg_types`` to just specify the required arguments and their types.

        Args:
            args_schema: The schema for the tool. Defaults to None.
            name: The name of the tool. Defaults to None.
            description: The description of the tool. Defaults to None.
            arg_types: A dictionary of argument names to types. Defaults to None.

        Returns:
            A BaseTool instance.

        Typed dict input:

        .. code-block:: python

            from typing import List
            from typing_extensions import TypedDict
            from langchain_core.runnables import RunnableLambda

            class Args(TypedDict):
                a: int
                b: List[int]

            def f(x: Args) -> str:
                return str(x["a"] * max(x["b"]))

            runnable = RunnableLambda(f)
            as_tool = runnable.as_tool()
            as_tool.invoke({"a": 3, "b": [1, 2]})

        ``dict`` input, specifying schema via ``args_schema``:

        .. code-block:: python

            from typing import Any, Dict, List
            from pydantic import BaseModel, Field
            from langchain_core.runnables import RunnableLambda

            def f(x: Dict[str, Any]) -> str:
                return str(x["a"] * max(x["b"]))

            class FSchema(BaseModel):
                \"\"\"Apply a function to an integer and list of integers.\"\"\"

                a: int = Field(..., description="Integer")
                b: List[int] = Field(..., description="List of ints")

            runnable = RunnableLambda(f)
            as_tool = runnable.as_tool(FSchema)
            as_tool.invoke({"a": 3, "b": [1, 2]})

        ``dict`` input, specifying schema via ``arg_types``:

        .. code-block:: python

            from typing import Any, Dict, List
            from langchain_core.runnables import RunnableLambda

            def f(x: Dict[str, Any]) -> str:
                return str(x["a"] * max(x["b"]))

            runnable = RunnableLambda(f)
            as_tool = runnable.as_tool(arg_types={"a": int, "b": List[int]})
            as_tool.invoke({"a": 3, "b": [1, 2]})

        String input:

        .. code-block:: python

            from langchain_core.runnables import RunnableLambda

            def f(x: str) -> str:
                return x + "a"

            def g(x: str) -> str:
                return x + "z"

            runnable = RunnableLambda(f) | g
            as_tool = runnable.as_tool()
            as_tool.invoke("b")

        .. versionadded:: 0.2.14
        """
        # Avoid circular import
        from langchain_core.tools import convert_runnable_to_tool

        return convert_runnable_to_tool(
            self,
            args_schema=args_schema,
            name=name,
            description=description,
            arg_types=arg_types,
        )


class RunnableSerializable(Serializable, Runnable[Input, Output]):
    """Runnable that can be serialized to JSON."""

    name: Optional[str] = None

    model_config = ConfigDict(
        # Suppress warnings from pydantic protected namespaces
        # (e.g., `model_`)
        protected_namespaces=(),
    )

    def to_json(self) -> Union[SerializedConstructor, SerializedNotImplemented]:
        """Serialize the Runnable to JSON.

        Returns:
            A JSON-serializable representation of the Runnable.
        """
        dumped = super().to_json()
        with contextlib.suppress(Exception):
            dumped["name"] = self.get_name()
        return dumped

    def configurable_fields(
        self, **kwargs: AnyConfigurableField
    ) -> RunnableSerializable[Input, Output]:
        """Configure particular Runnable fields at runtime.

        Args:
            **kwargs: A dictionary of ConfigurableField instances to configure.

        Returns:
            A new Runnable with the fields configured.

        .. code-block:: python

            from langchain_core.runnables import ConfigurableField
            from langchain_openai import ChatOpenAI

            model = ChatOpenAI(max_tokens=20).configurable_fields(
                max_tokens=ConfigurableField(
                    id="output_token_number",
                    name="Max tokens in the output",
                    description="The maximum number of tokens in the output",
                )
            )

            # max_tokens = 20
            print(
                "max_tokens_20: ",
                model.invoke("tell me something about chess").content
            )

            # max_tokens = 200
            print("max_tokens_200: ", model.with_config(
                configurable={"output_token_number": 200}
                ).invoke("tell me something about chess").content
            )
        """
        from langchain_core.runnables.configurable import RunnableConfigurableFields

        for key in kwargs:
            if key not in self.model_fields:
                msg = (
                    f"Configuration key {key} not found in {self}: "
                    f"available keys are {self.model_fields.keys()}"
                )
                raise ValueError(msg)

        return RunnableConfigurableFields(default=self, fields=kwargs)

    def configurable_alternatives(
        self,
        which: ConfigurableField,
        *,
        default_key: str = "default",
        prefix_keys: bool = False,
        **kwargs: Union[Runnable[Input, Output], Callable[[], Runnable[Input, Output]]],
    ) -> RunnableSerializable[Input, Output]:
        """Configure alternatives for Runnables that can be set at runtime.

        Args:
            which: The ConfigurableField instance that will be used to select the
                alternative.
            default_key: The default key to use if no alternative is selected.
                Defaults to "default".
            prefix_keys: Whether to prefix the keys with the ConfigurableField id.
                Defaults to False.
            **kwargs: A dictionary of keys to Runnable instances or callables that
                return Runnable instances.

        Returns:
            A new Runnable with the alternatives configured.

        .. code-block:: python

            from langchain_anthropic import ChatAnthropic
            from langchain_core.runnables.utils import ConfigurableField
            from langchain_openai import ChatOpenAI

            model = ChatAnthropic(
                model_name="claude-3-sonnet-20240229"
            ).configurable_alternatives(
                ConfigurableField(id="llm"),
                default_key="anthropic",
                openai=ChatOpenAI()
            )

            # uses the default model ChatAnthropic
            print(model.invoke("which organization created you?").content)

            # uses ChatOpenAI
            print(
                model.with_config(
                    configurable={"llm": "openai"}
                ).invoke("which organization created you?").content
            )
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


def _seq_input_schema(
    steps: list[Runnable[Any, Any]], config: Optional[RunnableConfig]
) -> type[BaseModel]:
    from langchain_core.runnables.passthrough import RunnableAssign, RunnablePick

    first = steps[0]
    if len(steps) == 1:
        return first.get_input_schema(config)
    elif isinstance(first, RunnableAssign):
        next_input_schema = _seq_input_schema(steps[1:], config)
        if not issubclass(next_input_schema, RootModel):
            # it's a dict as expected
            return create_model_v2(  # type: ignore[call-overload]
                "RunnableSequenceInput",
                field_definitions={
                    k: (v.annotation, v.default)
                    for k, v in next_input_schema.model_fields.items()
                    if k not in first.mapper.steps__
                },
            )
    elif isinstance(first, RunnablePick):
        return _seq_input_schema(steps[1:], config)

    return first.get_input_schema(config)


def _seq_output_schema(
    steps: list[Runnable[Any, Any]], config: Optional[RunnableConfig]
) -> type[BaseModel]:
    from langchain_core.runnables.passthrough import RunnableAssign, RunnablePick

    last = steps[-1]
    if len(steps) == 1:
        return last.get_input_schema(config)
    elif isinstance(last, RunnableAssign):
        mapper_output_schema = last.mapper.get_output_schema(config)
        prev_output_schema = _seq_output_schema(steps[:-1], config)
        if not issubclass(prev_output_schema, RootModel):
            # it's a dict as expected
            return create_model_v2(  # type: ignore[call-overload]
                "RunnableSequenceOutput",
                field_definitions={
                    **{
                        k: (v.annotation, v.default)
                        for k, v in prev_output_schema.model_fields.items()
                    },
                    **{
                        k: (v.annotation, v.default)
                        for k, v in mapper_output_schema.model_fields.items()
                    },
                },
            )
    elif isinstance(last, RunnablePick):
        prev_output_schema = _seq_output_schema(steps[:-1], config)
        if not issubclass(prev_output_schema, RootModel):
            # it's a dict as expected
            if isinstance(last.keys, list):
                return create_model_v2(  # type: ignore[call-overload]
                    "RunnableSequenceOutput",
                    field_definitions={
                        k: (v.annotation, v.default)
                        for k, v in prev_output_schema.model_fields.items()
                        if k in last.keys
                    },
                )
            else:
                field = prev_output_schema.model_fields[last.keys]
                return create_model_v2(  # type: ignore[call-overload]
                    "RunnableSequenceOutput", root=(field.annotation, field.default)
                )

    return last.get_output_schema(config)


class RunnableSequence(RunnableSerializable[Input, Output]):
    """Sequence of Runnables, where the output of each is the input of the next.

    **RunnableSequence** is the most important composition operator in LangChain
    as it is used in virtually every chain.

    A RunnableSequence can be instantiated directly or more commonly by using the `|`
    operator where either the left or right operands (or both) must be a Runnable.

    Any RunnableSequence automatically supports sync, async, batch.

    The default implementations of `batch` and `abatch` utilize threadpools and
    asyncio gather and will be faster than naive invocation of invoke or ainvoke
    for IO bound Runnables.

    Batching is implemented by invoking the batch method on each component of the
    RunnableSequence in order.

    A RunnableSequence preserves the streaming properties of its components, so if all
    components of the sequence implement a `transform` method -- which
    is the method that implements the logic to map a streaming input to a streaming
    output -- then the sequence will be able to stream input to output!

    If any component of the sequence does not implement transform then the
    streaming will only begin after this component is run. If there are
    multiple blocking components, streaming begins after the last one.

    Please note: RunnableLambdas do not support `transform` by default! So if
        you need to use a RunnableLambdas be careful about where you place them in a
        RunnableSequence (if you need to use the .stream()/.astream() methods).

        If you need arbitrary logic and need streaming, you can subclass
        Runnable, and implement `transform` for whatever logic you need.

    Here is a simple example that uses simple functions to illustrate the use of
    RunnableSequence:

        .. code-block:: python

            from langchain_core.runnables import RunnableLambda

            def add_one(x: int) -> int:
                return x + 1

            def mul_two(x: int) -> int:
                return x * 2

            runnable_1 = RunnableLambda(add_one)
            runnable_2 = RunnableLambda(mul_two)
            sequence = runnable_1 | runnable_2
            # Or equivalently:
            # sequence = RunnableSequence(first=runnable_1, last=runnable_2)
            sequence.invoke(1)
            await sequence.ainvoke(1)

            sequence.batch([1, 2, 3])
            await sequence.abatch([1, 2, 3])

    Here's an example that uses streams JSON output generated by an LLM:

        .. code-block:: python

            from langchain_core.output_parsers.json import SimpleJsonOutputParser
            from langchain_openai import ChatOpenAI

            prompt = PromptTemplate.from_template(
                'In JSON format, give me a list of {topic} and their '
                'corresponding names in French, Spanish and in a '
                'Cat Language.'
            )

            model = ChatOpenAI()
            chain = prompt | model | SimpleJsonOutputParser()

            async for chunk in chain.astream({'topic': 'colors'}):
                print('-')  # noqa: T201
                print(chunk, sep='', flush=True)  # noqa: T201
    """

    # The steps are broken into first, middle and last, solely for type checking
    # purposes. It allows specifying the `Input` on the first type, the `Output` of
    # the last type.
    first: Runnable[Input, Any]
    """The first Runnable in the sequence."""
    middle: list[Runnable[Any, Any]] = Field(default_factory=list)
    """The middle Runnables in the sequence."""
    last: Runnable[Any, Output]
    """The last Runnable in the sequence."""

    def __init__(
        self,
        *steps: RunnableLike,
        name: Optional[str] = None,
        first: Optional[Runnable[Any, Any]] = None,
        middle: Optional[list[Runnable[Any, Any]]] = None,
        last: Optional[Runnable[Any, Any]] = None,
    ) -> None:
        """Create a new RunnableSequence.

        Args:
            steps: The steps to include in the sequence.
            name: The name of the Runnable. Defaults to None.
            first: The first Runnable in the sequence. Defaults to None.
            middle: The middle Runnables in the sequence. Defaults to None.
            last: The last Runnable in the sequence. Defaults to None.

        Raises:
            ValueError: If the sequence has less than 2 steps.
        """
        steps_flat: list[Runnable] = []
        if not steps and first is not None and last is not None:
            steps_flat = [first] + (middle or []) + [last]
        for step in steps:
            if isinstance(step, RunnableSequence):
                steps_flat.extend(step.steps)
            else:
                steps_flat.append(coerce_to_runnable(step))
        if len(steps_flat) < 2:
            msg = f"RunnableSequence must have at least 2 steps, got {len(steps_flat)}"
            raise ValueError(msg)
        super().__init__(  # type: ignore[call-arg]
            first=steps_flat[0],
            middle=list(steps_flat[1:-1]),
            last=steps_flat[-1],
            name=name,
        )

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "schema", "runnable"]

    @property
    def steps(self) -> list[Runnable[Any, Any]]:
        """All the Runnables that make up the sequence in order.

        Returns:
            A list of Runnables.
        """
        return [self.first] + self.middle + [self.last]

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Check if the object is serializable.

        Returns:
            True if the object is serializable, False otherwise.
                Defaults to True.
        """
        return True

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    @property
    @override
    def InputType(self) -> type[Input]:
        """The type of the input to the Runnable."""
        return self.first.InputType

    @property
    @override
    def OutputType(self) -> type[Output]:
        """The type of the output of the Runnable."""
        return self.last.OutputType

    def get_input_schema(
        self, config: Optional[RunnableConfig] = None
    ) -> type[BaseModel]:
        """Get the input schema of the Runnable.

        Args:
            config: The config to use. Defaults to None.

        Returns:
            The input schema of the Runnable.
        """
        return _seq_input_schema(self.steps, config)

    def get_output_schema(
        self, config: Optional[RunnableConfig] = None
    ) -> type[BaseModel]:
        """Get the output schema of the Runnable.

        Args:
            config: The config to use. Defaults to None.

        Returns:
            The output schema of the Runnable.
        """
        return _seq_output_schema(self.steps, config)

    @property
    def config_specs(self) -> list[ConfigurableFieldSpec]:
        """Get the config specs of the Runnable.

        Returns:
            The config specs of the Runnable.
        """
        from langchain_core.beta.runnables.context import (
            CONTEXT_CONFIG_PREFIX,
            _key_from_id,
        )

        # get all specs
        all_specs = [
            (spec, idx)
            for idx, step in enumerate(self.steps)
            for spec in step.config_specs
        ]
        # calculate context dependencies
        specs_by_pos = groupby(
            [tup for tup in all_specs if tup[0].id.startswith(CONTEXT_CONFIG_PREFIX)],
            itemgetter(1),
        )
        next_deps: set[str] = set()
        deps_by_pos: dict[int, set[str]] = {}
        for pos, specs in specs_by_pos:
            deps_by_pos[pos] = next_deps
            next_deps = next_deps | {spec[0].id for spec in specs}
        # assign context dependencies
        for pos, (spec, idx) in enumerate(all_specs):
            if spec.id.startswith(CONTEXT_CONFIG_PREFIX):
                all_specs[pos] = (
                    ConfigurableFieldSpec(
                        id=spec.id,
                        annotation=spec.annotation,
                        name=spec.name,
                        default=spec.default,
                        description=spec.description,
                        is_shared=spec.is_shared,
                        dependencies=[
                            d
                            for d in deps_by_pos[idx]
                            if _key_from_id(d) != _key_from_id(spec.id)
                        ]
                        + (spec.dependencies or []),
                    ),
                    idx,
                )

        return get_unique_config_specs(spec for spec, _ in all_specs)

    def get_graph(self, config: Optional[RunnableConfig] = None) -> Graph:
        """Get the graph representation of the Runnable.

        Args:
            config: The config to use. Defaults to None.

        Returns:
            The graph representation of the Runnable.

        Raises:
            ValueError: If a Runnable has no first or last node.
        """
        from langchain_core.runnables.graph import Graph

        graph = Graph()
        for step in self.steps:
            current_last_node = graph.last_node()
            step_graph = step.get_graph(config)
            if step is not self.first:
                step_graph.trim_first_node()
            if step is not self.last:
                step_graph.trim_last_node()
            step_first_node, _ = graph.extend(step_graph)
            if not step_first_node:
                msg = f"Runnable {step} has no first node"
                raise ValueError(msg)
            if current_last_node:
                graph.add_edge(current_last_node, step_first_node)

        return graph

    def __repr__(self) -> str:
        return "\n| ".join(
            repr(s) if i == 0 else indent_lines_after_first(repr(s), "| ")
            for i, s in enumerate(self.steps)
        )

    def __or__(
        self,
        other: Union[
            Runnable[Any, Other],
            Callable[[Any], Other],
            Callable[[Iterator[Any]], Iterator[Other]],
            Mapping[str, Union[Runnable[Any, Other], Callable[[Any], Other], Any]],
        ],
    ) -> RunnableSerializable[Input, Other]:
        if isinstance(other, RunnableSequence):
            return RunnableSequence(
                self.first,
                *self.middle,
                self.last,
                other.first,
                *other.middle,
                other.last,
                name=self.name or other.name,
            )
        else:
            return RunnableSequence(
                self.first,
                *self.middle,
                self.last,
                coerce_to_runnable(other),
                name=self.name,
            )

    def __ror__(
        self,
        other: Union[
            Runnable[Other, Any],
            Callable[[Other], Any],
            Callable[[Iterator[Other]], Iterator[Any]],
            Mapping[str, Union[Runnable[Other, Any], Callable[[Other], Any], Any]],
        ],
    ) -> RunnableSerializable[Other, Output]:
        if isinstance(other, RunnableSequence):
            return RunnableSequence(
                other.first,
                *other.middle,
                other.last,
                self.first,
                *self.middle,
                self.last,
                name=other.name or self.name,
            )
        else:
            return RunnableSequence(
                coerce_to_runnable(other),
                self.first,
                *self.middle,
                self.last,
                name=self.name,
            )

    def invoke(
        self, input: Input, config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> Output:
        from langchain_core.beta.runnables.context import config_with_context

        # setup callbacks and context
        config = config_with_context(ensure_config(config), self.steps)
        callback_manager = get_callback_manager_for_config(config)
        # start the root run
        run_manager = callback_manager.on_chain_start(
            None,
            input,
            name=config.get("run_name") or self.get_name(),
            run_id=config.pop("run_id", None),
        )

        # invoke all steps in sequence
        try:
            for i, step in enumerate(self.steps):
                # mark each step as a child run
                config = patch_config(
                    config, callbacks=run_manager.get_child(f"seq:step:{i + 1}")
                )
                context = copy_context()
                context.run(_set_config_context, config)
                if i == 0:
                    input = context.run(step.invoke, input, config, **kwargs)
                else:
                    input = context.run(step.invoke, input, config)
        # finish the root run
        except BaseException as e:
            run_manager.on_chain_error(e)
            raise
        else:
            run_manager.on_chain_end(input)
            return cast(Output, input)

    async def ainvoke(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> Output:
        from langchain_core.beta.runnables.context import aconfig_with_context

        # setup callbacks and context
        config = aconfig_with_context(ensure_config(config), self.steps)
        callback_manager = get_async_callback_manager_for_config(config)
        # start the root run
        run_manager = await callback_manager.on_chain_start(
            None,
            input,
            name=config.get("run_name") or self.get_name(),
            run_id=config.pop("run_id", None),
        )

        # invoke all steps in sequence
        try:
            for i, step in enumerate(self.steps):
                # mark each step as a child run
                config = patch_config(
                    config, callbacks=run_manager.get_child(f"seq:step:{i + 1}")
                )
                context = copy_context()
                context.run(_set_config_context, config)
                if i == 0:
                    part = functools.partial(step.ainvoke, input, config, **kwargs)
                else:
                    part = functools.partial(step.ainvoke, input, config)
                if asyncio_accepts_context():
                    input = await asyncio.create_task(part(), context=context)  # type: ignore
                else:
                    input = await asyncio.create_task(part())
        # finish the root run
        except BaseException as e:
            await run_manager.on_chain_error(e)
            raise
        else:
            await run_manager.on_chain_end(input)
            return cast(Output, input)

    def batch(
        self,
        inputs: list[Input],
        config: Optional[Union[RunnableConfig, list[RunnableConfig]]] = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Optional[Any],
    ) -> list[Output]:
        from langchain_core.beta.runnables.context import config_with_context
        from langchain_core.callbacks.manager import CallbackManager

        if not inputs:
            return []

        # setup callbacks and context
        configs = [
            config_with_context(c, self.steps)
            for c in get_config_list(config, len(inputs))
        ]
        callback_managers = [
            CallbackManager.configure(
                inheritable_callbacks=config.get("callbacks"),
                local_callbacks=None,
                verbose=False,
                inheritable_tags=config.get("tags"),
                local_tags=None,
                inheritable_metadata=config.get("metadata"),
                local_metadata=None,
            )
            for config in configs
        ]
        # start the root runs, one per input
        run_managers = [
            cm.on_chain_start(
                None,
                input,
                name=config.get("run_name") or self.get_name(),
                run_id=config.pop("run_id", None),
            )
            for cm, input, config in zip(callback_managers, inputs, configs)
        ]

        # invoke
        try:
            if return_exceptions:
                # Track which inputs (by index) failed so far
                # If an input has failed it will be present in this map,
                # and the value will be the exception that was raised.
                failed_inputs_map: dict[int, Exception] = {}
                for stepidx, step in enumerate(self.steps):
                    # Assemble the original indexes of the remaining inputs
                    # (i.e. the ones that haven't failed yet)
                    remaining_idxs = [
                        i for i in range(len(configs)) if i not in failed_inputs_map
                    ]
                    # Invoke the step on the remaining inputs
                    inputs = step.batch(
                        [
                            inp
                            for i, inp in zip(remaining_idxs, inputs)
                            if i not in failed_inputs_map
                        ],
                        [
                            # each step a child run of the corresponding root run
                            patch_config(
                                config,
                                callbacks=rm.get_child(f"seq:step:{stepidx + 1}"),
                            )
                            for i, (rm, config) in enumerate(zip(run_managers, configs))
                            if i not in failed_inputs_map
                        ],
                        return_exceptions=return_exceptions,
                        **(kwargs if stepidx == 0 else {}),
                    )
                    # If an input failed, add it to the map
                    for i, inp in zip(remaining_idxs, inputs):
                        if isinstance(inp, Exception):
                            failed_inputs_map[i] = inp
                    inputs = [inp for inp in inputs if not isinstance(inp, Exception)]
                    # If all inputs have failed, stop processing
                    if len(failed_inputs_map) == len(configs):
                        break

                # Reassemble the outputs, inserting Exceptions for failed inputs
                inputs_copy = inputs.copy()
                inputs = []
                for i in range(len(configs)):
                    if i in failed_inputs_map:
                        inputs.append(cast(Input, failed_inputs_map[i]))
                    else:
                        inputs.append(inputs_copy.pop(0))
            else:
                for i, step in enumerate(self.steps):
                    inputs = step.batch(
                        inputs,
                        [
                            # each step a child run of the corresponding root run
                            patch_config(
                                config, callbacks=rm.get_child(f"seq:step:{i + 1}")
                            )
                            for rm, config in zip(run_managers, configs)
                        ],
                        return_exceptions=return_exceptions,
                        **(kwargs if i == 0 else {}),
                    )

        # finish the root runs
        except BaseException as e:
            for rm in run_managers:
                rm.on_chain_error(e)
            if return_exceptions:
                return cast(list[Output], [e for _ in inputs])
            else:
                raise
        else:
            first_exception: Optional[Exception] = None
            for run_manager, out in zip(run_managers, inputs):
                if isinstance(out, Exception):
                    first_exception = first_exception or out
                    run_manager.on_chain_error(out)
                else:
                    run_manager.on_chain_end(out)
            if return_exceptions or first_exception is None:
                return cast(list[Output], inputs)
            else:
                raise first_exception

    async def abatch(
        self,
        inputs: list[Input],
        config: Optional[Union[RunnableConfig, list[RunnableConfig]]] = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Optional[Any],
    ) -> list[Output]:
        from langchain_core.beta.runnables.context import aconfig_with_context
        from langchain_core.callbacks.manager import AsyncCallbackManager

        if not inputs:
            return []

        # setup callbacks and context
        configs = [
            aconfig_with_context(c, self.steps)
            for c in get_config_list(config, len(inputs))
        ]
        callback_managers = [
            AsyncCallbackManager.configure(
                inheritable_callbacks=config.get("callbacks"),
                local_callbacks=None,
                verbose=False,
                inheritable_tags=config.get("tags"),
                local_tags=None,
                inheritable_metadata=config.get("metadata"),
                local_metadata=None,
            )
            for config in configs
        ]
        # start the root runs, one per input
        run_managers: list[AsyncCallbackManagerForChainRun] = await asyncio.gather(
            *(
                cm.on_chain_start(
                    None,
                    input,
                    name=config.get("run_name") or self.get_name(),
                    run_id=config.pop("run_id", None),
                )
                for cm, input, config in zip(callback_managers, inputs, configs)
            )
        )

        # invoke .batch() on each step
        # this uses batching optimizations in Runnable subclasses, like LLM
        try:
            if return_exceptions:
                # Track which inputs (by index) failed so far
                # If an input has failed it will be present in this map,
                # and the value will be the exception that was raised.
                failed_inputs_map: dict[int, Exception] = {}
                for stepidx, step in enumerate(self.steps):
                    # Assemble the original indexes of the remaining inputs
                    # (i.e. the ones that haven't failed yet)
                    remaining_idxs = [
                        i for i in range(len(configs)) if i not in failed_inputs_map
                    ]
                    # Invoke the step on the remaining inputs
                    inputs = await step.abatch(
                        [
                            inp
                            for i, inp in zip(remaining_idxs, inputs)
                            if i not in failed_inputs_map
                        ],
                        [
                            # each step a child run of the corresponding root run
                            patch_config(
                                config,
                                callbacks=rm.get_child(f"seq:step:{stepidx + 1}"),
                            )
                            for i, (rm, config) in enumerate(zip(run_managers, configs))
                            if i not in failed_inputs_map
                        ],
                        return_exceptions=return_exceptions,
                        **(kwargs if stepidx == 0 else {}),
                    )
                    # If an input failed, add it to the map
                    for i, inp in zip(remaining_idxs, inputs):
                        if isinstance(inp, Exception):
                            failed_inputs_map[i] = inp
                    inputs = [inp for inp in inputs if not isinstance(inp, Exception)]
                    # If all inputs have failed, stop processing
                    if len(failed_inputs_map) == len(configs):
                        break

                # Reassemble the outputs, inserting Exceptions for failed inputs
                inputs_copy = inputs.copy()
                inputs = []
                for i in range(len(configs)):
                    if i in failed_inputs_map:
                        inputs.append(cast(Input, failed_inputs_map[i]))
                    else:
                        inputs.append(inputs_copy.pop(0))
            else:
                for i, step in enumerate(self.steps):
                    inputs = await step.abatch(
                        inputs,
                        [
                            # each step a child run of the corresponding root run
                            patch_config(
                                config, callbacks=rm.get_child(f"seq:step:{i + 1}")
                            )
                            for rm, config in zip(run_managers, configs)
                        ],
                        return_exceptions=return_exceptions,
                        **(kwargs if i == 0 else {}),
                    )
        # finish the root runs
        except BaseException as e:
            await asyncio.gather(*(rm.on_chain_error(e) for rm in run_managers))
            if return_exceptions:
                return cast(list[Output], [e for _ in inputs])
            else:
                raise
        else:
            first_exception: Optional[Exception] = None
            coros: list[Awaitable[None]] = []
            for run_manager, out in zip(run_managers, inputs):
                if isinstance(out, Exception):
                    first_exception = first_exception or out
                    coros.append(run_manager.on_chain_error(out))
                else:
                    coros.append(run_manager.on_chain_end(out))
            await asyncio.gather(*coros)
            if return_exceptions or first_exception is None:
                return cast(list[Output], inputs)
            else:
                raise first_exception

    def _transform(
        self,
        input: Iterator[Input],
        run_manager: CallbackManagerForChainRun,
        config: RunnableConfig,
        **kwargs: Any,
    ) -> Iterator[Output]:
        from langchain_core.beta.runnables.context import config_with_context

        steps = [self.first] + self.middle + [self.last]
        config = config_with_context(config, self.steps)

        # transform the input stream of each step with the next
        # steps that don't natively support transforming an input stream will
        # buffer input in memory until all available, and then start emitting output
        final_pipeline = cast(Iterator[Output], input)
        for idx, step in enumerate(steps):
            config = patch_config(
                config, callbacks=run_manager.get_child(f"seq:step:{idx + 1}")
            )
            if idx == 0:
                final_pipeline = step.transform(final_pipeline, config, **kwargs)
            else:
                final_pipeline = step.transform(final_pipeline, config)

        yield from final_pipeline

    async def _atransform(
        self,
        input: AsyncIterator[Input],
        run_manager: AsyncCallbackManagerForChainRun,
        config: RunnableConfig,
        **kwargs: Any,
    ) -> AsyncIterator[Output]:
        from langchain_core.beta.runnables.context import aconfig_with_context

        steps = [self.first] + self.middle + [self.last]
        config = aconfig_with_context(config, self.steps)

        # stream the last steps
        # transform the input stream of each step with the next
        # steps that don't natively support transforming an input stream will
        # buffer input in memory until all available, and then start emitting output
        final_pipeline = cast(AsyncIterator[Output], input)
        for idx, step in enumerate(steps):
            config = patch_config(
                config,
                callbacks=run_manager.get_child(f"seq:step:{idx + 1}"),
            )
            if idx == 0:
                final_pipeline = step.atransform(final_pipeline, config, **kwargs)
            else:
                final_pipeline = step.atransform(final_pipeline, config)
        async for output in final_pipeline:
            yield output

    def transform(
        self,
        input: Iterator[Input],
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> Iterator[Output]:
        yield from self._transform_stream_with_config(
            input,
            self._transform,
            patch_config(config, run_name=(config or {}).get("run_name") or self.name),
            **kwargs,
        )

    def stream(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> Iterator[Output]:
        yield from self.transform(iter([input]), config, **kwargs)

    async def atransform(
        self,
        input: AsyncIterator[Input],
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> AsyncIterator[Output]:
        async for chunk in self._atransform_stream_with_config(
            input,
            self._atransform,
            patch_config(config, run_name=(config or {}).get("run_name") or self.name),
            **kwargs,
        ):
            yield chunk

    async def astream(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> AsyncIterator[Output]:
        async def input_aiter() -> AsyncIterator[Input]:
            yield input

        async for chunk in self.atransform(input_aiter(), config, **kwargs):
            yield chunk


class RunnableParallel(RunnableSerializable[Input, dict[str, Any]]):
    """Runnable that runs a mapping of Runnables in parallel, and returns a mapping
    of their outputs.

    RunnableParallel is one of the two main composition primitives for the LCEL,
    alongside RunnableSequence. It invokes Runnables concurrently, providing the same
    input to each.

    A RunnableParallel can be instantiated directly or by using a dict literal within a
    sequence.

    Here is a simple example that uses functions to illustrate the use of
    RunnableParallel:

        .. code-block:: python

            from langchain_core.runnables import RunnableLambda

            def add_one(x: int) -> int:
                return x + 1

            def mul_two(x: int) -> int:
                return x * 2

            def mul_three(x: int) -> int:
                return x * 3

            runnable_1 = RunnableLambda(add_one)
            runnable_2 = RunnableLambda(mul_two)
            runnable_3 = RunnableLambda(mul_three)

            sequence = runnable_1 | {  # this dict is coerced to a RunnableParallel
                "mul_two": runnable_2,
                "mul_three": runnable_3,
            }
            # Or equivalently:
            # sequence = runnable_1 | RunnableParallel(
            #     {"mul_two": runnable_2, "mul_three": runnable_3}
            # )
            # Also equivalently:
            # sequence = runnable_1 | RunnableParallel(
            #     mul_two=runnable_2,
            #     mul_three=runnable_3,
            # )

            sequence.invoke(1)
            await sequence.ainvoke(1)

            sequence.batch([1, 2, 3])
            await sequence.abatch([1, 2, 3])

    RunnableParallel makes it easy to run Runnables in parallel. In the below example,
    we simultaneously stream output from two different Runnables:

        .. code-block:: python

            from langchain_core.prompts import ChatPromptTemplate
            from langchain_core.runnables import RunnableParallel
            from langchain_openai import ChatOpenAI

            model = ChatOpenAI()
            joke_chain = (
                ChatPromptTemplate.from_template("tell me a joke about {topic}")
                | model
            )
            poem_chain = (
                ChatPromptTemplate.from_template("write a 2-line poem about {topic}")
                | model
            )

            runnable = RunnableParallel(joke=joke_chain, poem=poem_chain)

            # Display stream
            output = {key: "" for key, _ in runnable.output_schema()}
            for chunk in runnable.stream({"topic": "bear"}):
                for key in chunk:
                    output[key] = output[key] + chunk[key].content
                print(output)  # noqa: T201
    """

    steps__: Mapping[str, Runnable[Input, Any]]

    def __init__(
        self,
        steps__: Optional[
            Mapping[
                str,
                Union[
                    Runnable[Input, Any],
                    Callable[[Input], Any],
                    Mapping[str, Union[Runnable[Input, Any], Callable[[Input], Any]]],
                ],
            ]
        ] = None,
        **kwargs: Union[
            Runnable[Input, Any],
            Callable[[Input], Any],
            Mapping[str, Union[Runnable[Input, Any], Callable[[Input], Any]]],
        ],
    ) -> None:
        merged = {**steps__} if steps__ is not None else {}
        merged.update(kwargs)
        super().__init__(  # type: ignore[call-arg]
            steps__={key: coerce_to_runnable(r) for key, r in merged.items()}
        )

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return True

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "schema", "runnable"]

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    def get_name(
        self, suffix: Optional[str] = None, *, name: Optional[str] = None
    ) -> str:
        """Get the name of the Runnable.

        Args:
            suffix: The suffix to use. Defaults to None.
            name: The name to use. Defaults to None.

        Returns:
            The name of the Runnable.
        """
        name = name or self.name or f"RunnableParallel<{','.join(self.steps__.keys())}>"
        return super().get_name(suffix, name=name)

    @property
    @override
    def InputType(self) -> Any:
        """The type of the input to the Runnable."""
        for step in self.steps__.values():
            if step.InputType:
                return step.InputType

        return Any

    def get_input_schema(
        self, config: Optional[RunnableConfig] = None
    ) -> type[BaseModel]:
        """Get the input schema of the Runnable.

        Args:
            config: The config to use. Defaults to None.

        Returns:
            The input schema of the Runnable.
        """
        if all(
            s.get_input_schema(config).model_json_schema().get("type", "object")
            == "object"
            for s in self.steps__.values()
        ):
            # This is correct, but pydantic typings/mypy don't think so.
            return create_model_v2(  # type: ignore[call-overload]
                self.get_name("Input"),
                field_definitions={
                    k: (v.annotation, v.default)
                    for step in self.steps__.values()
                    for k, v in step.get_input_schema(config).model_fields.items()
                    if k != "__root__"
                },
            )

        return super().get_input_schema(config)

    def get_output_schema(
        self, config: Optional[RunnableConfig] = None
    ) -> type[BaseModel]:
        """Get the output schema of the Runnable.

        Args:
            config: The config to use. Defaults to None.

        Returns:
            The output schema of the Runnable.
        """
        fields = {k: (v.OutputType, ...) for k, v in self.steps__.items()}
        return create_model_v2(self.get_name("Output"), field_definitions=fields)

    @property
    def config_specs(self) -> list[ConfigurableFieldSpec]:
        """Get the config specs of the Runnable.

        Returns:
            The config specs of the Runnable.
        """
        return get_unique_config_specs(
            spec for step in self.steps__.values() for spec in step.config_specs
        )

    def get_graph(self, config: Optional[RunnableConfig] = None) -> Graph:
        """Get the graph representation of the Runnable.

        Args:
            config: The config to use. Defaults to None.

        Returns:
            The graph representation of the Runnable.

        Raises:
            ValueError: If a Runnable has no first or last node.
        """
        from langchain_core.runnables.graph import Graph

        graph = Graph()
        input_node = graph.add_node(self.get_input_schema(config))
        output_node = graph.add_node(self.get_output_schema(config))
        for step in self.steps__.values():
            step_graph = step.get_graph()
            step_graph.trim_first_node()
            step_graph.trim_last_node()
            if not step_graph:
                graph.add_edge(input_node, output_node)
            else:
                step_first_node, step_last_node = graph.extend(step_graph)
                if not step_first_node:
                    msg = f"Runnable {step} has no first node"
                    raise ValueError(msg)
                if not step_last_node:
                    msg = f"Runnable {step} has no last node"
                    raise ValueError(msg)
                graph.add_edge(input_node, step_first_node)
                graph.add_edge(step_last_node, output_node)

        return graph

    def __repr__(self) -> str:
        map_for_repr = ",\n  ".join(
            f"{k}: {indent_lines_after_first(repr(v), '  ' + k + ': ')}"
            for k, v in self.steps__.items()
        )
        return "{\n  " + map_for_repr + "\n}"

    def invoke(
        self, input: Input, config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> dict[str, Any]:
        from langchain_core.callbacks.manager import CallbackManager

        # setup callbacks
        config = ensure_config(config)
        callback_manager = CallbackManager.configure(
            inheritable_callbacks=config.get("callbacks"),
            local_callbacks=None,
            verbose=False,
            inheritable_tags=config.get("tags"),
            local_tags=None,
            inheritable_metadata=config.get("metadata"),
            local_metadata=None,
        )
        # start the root run
        run_manager = callback_manager.on_chain_start(
            None,
            input,
            name=config.get("run_name") or self.get_name(),
            run_id=config.pop("run_id", None),
        )

        def _invoke_step(
            step: Runnable[Input, Any], input: Input, config: RunnableConfig, key: str
        ) -> Any:
            child_config = patch_config(
                config,
                # mark each step as a child run
                callbacks=run_manager.get_child(f"map:key:{key}"),
            )
            context = copy_context()
            context.run(_set_config_context, child_config)
            return context.run(
                step.invoke,
                input,
                child_config,
            )

        # gather results from all steps
        try:
            # copy to avoid issues from the caller mutating the steps during invoke()
            steps = dict(self.steps__)

            with get_executor_for_config(config) as executor:
                futures = [
                    executor.submit(_invoke_step, step, input, config, key)
                    for key, step in steps.items()
                ]
                output = {key: future.result() for key, future in zip(steps, futures)}
        # finish the root run
        except BaseException as e:
            run_manager.on_chain_error(e)
            raise
        else:
            run_manager.on_chain_end(output)
            return output

    async def ainvoke(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> dict[str, Any]:
        # setup callbacks
        config = ensure_config(config)
        callback_manager = get_async_callback_manager_for_config(config)
        # start the root run
        run_manager = await callback_manager.on_chain_start(
            None,
            input,
            name=config.get("run_name") or self.get_name(),
            run_id=config.pop("run_id", None),
        )

        async def _ainvoke_step(
            step: Runnable[Input, Any], input: Input, config: RunnableConfig, key: str
        ) -> Any:
            child_config = patch_config(
                config,
                callbacks=run_manager.get_child(f"map:key:{key}"),
            )
            context = copy_context()
            context.run(_set_config_context, child_config)
            if asyncio_accepts_context():
                return await asyncio.create_task(  # type: ignore
                    step.ainvoke(input, child_config), context=context
                )
            else:
                return await asyncio.create_task(step.ainvoke(input, child_config))

        # gather results from all steps
        try:
            # copy to avoid issues from the caller mutating the steps during invoke()
            steps = dict(self.steps__)
            results = await asyncio.gather(
                *(
                    _ainvoke_step(
                        step,
                        input,
                        # mark each step as a child run
                        config,
                        key,
                    )
                    for key, step in steps.items()
                )
            )
            output = dict(zip(steps, results))
        # finish the root run
        except BaseException as e:
            await run_manager.on_chain_error(e)
            raise
        else:
            await run_manager.on_chain_end(output)
            return output

    def _transform(
        self,
        input: Iterator[Input],
        run_manager: CallbackManagerForChainRun,
        config: RunnableConfig,
    ) -> Iterator[AddableDict]:
        # Shallow copy steps to ignore mutations while in progress
        steps = dict(self.steps__)
        # Each step gets a copy of the input iterator,
        # which is consumed in parallel in a separate thread.
        input_copies = list(safetee(input, len(steps), lock=threading.Lock()))
        with get_executor_for_config(config) as executor:
            # Create the transform() generator for each step
            named_generators = [
                (
                    name,
                    step.transform(
                        input_copies.pop(),
                        patch_config(
                            config, callbacks=run_manager.get_child(f"map:key:{name}")
                        ),
                    ),
                )
                for name, step in steps.items()
            ]
            # Start the first iteration of each generator
            futures = {
                executor.submit(next, generator): (step_name, generator)
                for step_name, generator in named_generators
            }
            # Yield chunks from each as they become available,
            # and start the next iteration of that generator that yielded it.
            # When all generators are exhausted, stop.
            while futures:
                completed_futures, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in completed_futures:
                    (step_name, generator) = futures.pop(future)
                    try:
                        chunk = AddableDict({step_name: future.result()})
                        yield chunk
                        futures[executor.submit(next, generator)] = (
                            step_name,
                            generator,
                        )
                    except StopIteration:
                        pass

    def transform(
        self,
        input: Iterator[Input],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Iterator[dict[str, Any]]:
        yield from self._transform_stream_with_config(
            input, self._transform, config, **kwargs
        )

    def stream(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> Iterator[dict[str, Any]]:
        yield from self.transform(iter([input]), config)

    async def _atransform(
        self,
        input: AsyncIterator[Input],
        run_manager: AsyncCallbackManagerForChainRun,
        config: RunnableConfig,
    ) -> AsyncIterator[AddableDict]:
        # Shallow copy steps to ignore mutations while in progress
        steps = dict(self.steps__)
        # Each step gets a copy of the input iterator,
        # which is consumed in parallel in a separate thread.
        input_copies = list(atee(input, len(steps), lock=asyncio.Lock()))
        # Create the transform() generator for each step
        named_generators = [
            (
                name,
                step.atransform(
                    input_copies.pop(),
                    patch_config(
                        config, callbacks=run_manager.get_child(f"map:key:{name}")
                    ),
                ),
            )
            for name, step in steps.items()
        ]

        # Wrap in a coroutine to satisfy linter
        async def get_next_chunk(generator: AsyncIterator) -> Optional[Output]:
            return await py_anext(generator)

        # Start the first iteration of each generator
        tasks = {
            asyncio.create_task(get_next_chunk(generator)): (step_name, generator)
            for step_name, generator in named_generators
        }
        # Yield chunks from each as they become available,
        # and start the next iteration of the generator that yielded it.
        # When all generators are exhausted, stop.
        while tasks:
            completed_tasks, _ = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_COMPLETED
            )
            for task in completed_tasks:
                (step_name, generator) = tasks.pop(task)
                try:
                    chunk = AddableDict({step_name: task.result()})
                    yield chunk
                    new_task = asyncio.create_task(get_next_chunk(generator))
                    tasks[new_task] = (step_name, generator)
                except StopAsyncIteration:
                    pass

    async def atransform(
        self,
        input: AsyncIterator[Input],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> AsyncIterator[dict[str, Any]]:
        async for chunk in self._atransform_stream_with_config(
            input, self._atransform, config, **kwargs
        ):
            yield chunk

    async def astream(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> AsyncIterator[dict[str, Any]]:
        async def input_aiter() -> AsyncIterator[Input]:
            yield input

        async for chunk in self.atransform(input_aiter(), config):
            yield chunk


# We support both names
RunnableMap = RunnableParallel


class RunnableGenerator(Runnable[Input, Output]):
    """Runnable that runs a generator function.

    RunnableGenerators can be instantiated directly or by using a generator within
    a sequence.

    RunnableGenerators can be used to implement custom behavior, such as custom output
    parsers, while preserving streaming capabilities. Given a generator function with
    a signature Iterator[A] -> Iterator[B], wrapping it in a RunnableGenerator allows
    it to emit output chunks as soon as they are streamed in from the previous step.

    Note that if a generator function has a signature A -> Iterator[B], such that it
    requires its input from the previous step to be completed before emitting chunks
    (e.g., most LLMs need the entire prompt available to start generating), it can
    instead be wrapped in a RunnableLambda.

    Here is an example to show the basic mechanics of a RunnableGenerator:

        .. code-block:: python

            from typing import Any, AsyncIterator, Iterator

            from langchain_core.runnables import RunnableGenerator


            def gen(input: Iterator[Any]) -> Iterator[str]:
                for token in ["Have", " a", " nice", " day"]:
                    yield token


            runnable = RunnableGenerator(gen)
            runnable.invoke(None)  # "Have a nice day"
            list(runnable.stream(None))  # ["Have", " a", " nice", " day"]
            runnable.batch([None, None])  # ["Have a nice day", "Have a nice day"]


            # Async version:
            async def agen(input: AsyncIterator[Any]) -> AsyncIterator[str]:
                for token in ["Have", " a", " nice", " day"]:
                    yield token

            runnable = RunnableGenerator(agen)
            await runnable.ainvoke(None)  # "Have a nice day"
            [p async for p in runnable.astream(None)] # ["Have", " a", " nice", " day"]

    RunnableGenerator makes it easy to implement custom behavior within a streaming
    context. Below we show an example:

        .. code-block:: python

            from langchain_core.prompts import ChatPromptTemplate
            from langchain_core.runnables import RunnableGenerator, RunnableLambda
            from langchain_openai import ChatOpenAI
            from langchain_core.output_parsers import StrOutputParser


            model = ChatOpenAI()
            chant_chain = (
                ChatPromptTemplate.from_template("Give me a 3 word chant about {topic}")
                | model
                | StrOutputParser()
            )

            def character_generator(input: Iterator[str]) -> Iterator[str]:
                for token in input:
                    if "," in token or "." in token:
                        yield "👏" + token
                    else:
                        yield token


            runnable = chant_chain | character_generator
            assert type(runnable.last) is RunnableGenerator
            "".join(runnable.stream({"topic": "waste"})) # Reduce👏, Reuse👏, Recycle👏.

            # Note that RunnableLambda can be used to delay streaming of one step in a
            # sequence until the previous step is finished:
            def reverse_generator(input: str) -> Iterator[str]:
                # Yield characters of input in reverse order.
                for character in input[::-1]:
                    yield character

            runnable = chant_chain | RunnableLambda(reverse_generator)
            "".join(runnable.stream({"topic": "waste"}))  # ".elcycer ,esuer ,ecudeR"
    """

    def __init__(
        self,
        transform: Union[
            Callable[[Iterator[Input]], Iterator[Output]],
            Callable[[AsyncIterator[Input]], AsyncIterator[Output]],
        ],
        atransform: Optional[
            Callable[[AsyncIterator[Input]], AsyncIterator[Output]]
        ] = None,
        *,
        name: Optional[str] = None,
    ) -> None:
        """Initialize a RunnableGenerator.

        Args:
            transform: The transform function.
            atransform: The async transform function. Defaults to None.

        Raises:
            TypeError: If the transform is not a generator function.
        """
        if atransform is not None:
            self._atransform = atransform
            func_for_name: Callable = atransform

        if is_async_generator(transform):
            self._atransform = transform  # type: ignore[assignment]
            func_for_name = transform
        elif inspect.isgeneratorfunction(transform):
            self._transform = transform
            func_for_name = transform
        else:
            msg = (
                "Expected a generator function type for `transform`."
                f"Instead got an unsupported type: {type(transform)}"
            )
            raise TypeError(msg)

        try:
            self.name = name or func_for_name.__name__
        except AttributeError:
            self.name = "RunnableGenerator"

    @property
    @override
    def InputType(self) -> Any:
        func = getattr(self, "_transform", None) or self._atransform
        try:
            params = inspect.signature(func).parameters
            first_param = next(iter(params.values()), None)
            if first_param and first_param.annotation != inspect.Parameter.empty:
                return getattr(first_param.annotation, "__args__", (Any,))[0]
            else:
                return Any
        except ValueError:
            return Any

    def get_input_schema(
        self, config: Optional[RunnableConfig] = None
    ) -> type[BaseModel]:
        # Override the default implementation.
        # For a runnable generator, we need to bring to provide the
        # module of the underlying function when creating the model.
        root_type = self.InputType

        func = getattr(self, "_transform", None) or self._atransform
        module = getattr(func, "__module__", None)

        if (
            inspect.isclass(root_type)
            and not isinstance(root_type, GenericAlias)
            and issubclass(root_type, BaseModel)
        ):
            return root_type

        return create_model_v2(
            self.get_name("Input"),
            root=root_type,
            # To create the schema, we need to provide the module
            # where the underlying function is defined.
            # This allows pydantic to resolve type annotations appropriately.
            module_name=module,
        )

    @property
    @override
    def OutputType(self) -> Any:
        func = getattr(self, "_transform", None) or self._atransform
        try:
            sig = inspect.signature(func)
            return (
                getattr(sig.return_annotation, "__args__", (Any,))[0]
                if sig.return_annotation != inspect.Signature.empty
                else Any
            )
        except ValueError:
            return Any

    def get_output_schema(
        self, config: Optional[RunnableConfig] = None
    ) -> type[BaseModel]:
        # Override the default implementation.
        # For a runnable generator, we need to bring to provide the
        # module of the underlying function when creating the model.
        root_type = self.OutputType
        func = getattr(self, "_transform", None) or self._atransform
        module = getattr(func, "__module__", None)

        if (
            inspect.isclass(root_type)
            and not isinstance(root_type, GenericAlias)
            and issubclass(root_type, BaseModel)
        ):
            return root_type

        return create_model_v2(
            self.get_name("Output"),
            root=root_type,
            # To create the schema, we need to provide the module
            # where the underlying function is defined.
            # This allows pydantic to resolve type annotations appropriately.
            module_name=module,
        )

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, RunnableGenerator):
            if hasattr(self, "_transform") and hasattr(other, "_transform"):
                return self._transform == other._transform
            elif hasattr(self, "_atransform") and hasattr(other, "_atransform"):
                return self._atransform == other._atransform
            else:
                return False
        else:
            return False

    def __repr__(self) -> str:
        return f"RunnableGenerator({self.name})"

    def transform(
        self,
        input: Iterator[Input],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Iterator[Output]:
        if not hasattr(self, "_transform"):
            msg = f"{repr(self)} only supports async methods."
            raise NotImplementedError(msg)
        return self._transform_stream_with_config(
            input,
            self._transform,  # type: ignore[arg-type]
            config,
            **kwargs,  # type: ignore[arg-type]
        )

    def stream(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Iterator[Output]:
        return self.transform(iter([input]), config, **kwargs)

    def invoke(
        self, input: Input, config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> Output:
        final: Optional[Output] = None
        for output in self.stream(input, config, **kwargs):
            final = output if final is None else final + output  # type: ignore[operator]
        return cast(Output, final)

    def atransform(
        self,
        input: AsyncIterator[Input],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> AsyncIterator[Output]:
        if not hasattr(self, "_atransform"):
            msg = f"{repr(self)} only supports sync methods."
            raise NotImplementedError(msg)

        return self._atransform_stream_with_config(
            input, self._atransform, config, **kwargs
        )

    def astream(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> AsyncIterator[Output]:
        async def input_aiter() -> AsyncIterator[Input]:
            yield input

        return self.atransform(input_aiter(), config, **kwargs)

    async def ainvoke(
        self, input: Input, config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> Output:
        final: Optional[Output] = None
        async for output in self.astream(input, config, **kwargs):
            final = output if final is None else final + output  # type: ignore[operator]
        return cast(Output, final)


class RunnableLambda(Runnable[Input, Output]):
    """RunnableLambda converts a python callable into a Runnable.

    Wrapping a callable in a RunnableLambda makes the callable usable
    within either a sync or async context.

    RunnableLambda can be composed as any other Runnable and provides
    seamless integration with LangChain tracing.

    ``RunnableLambda`` is best suited for code that does not need to support
    streaming. If you need to support streaming (i.e., be able to operate
    on chunks of inputs and yield chunks of outputs), use ``RunnableGenerator``
    instead.

    Note that if a ``RunnableLambda`` returns an instance of ``Runnable``, that
    instance is invoked (or streamed) during execution.

    Examples:

        .. code-block:: python

            # This is a RunnableLambda
            from langchain_core.runnables import RunnableLambda

            def add_one(x: int) -> int:
                return x + 1

            runnable = RunnableLambda(add_one)

            runnable.invoke(1) # returns 2
            runnable.batch([1, 2, 3]) # returns [2, 3, 4]

            # Async is supported by default by delegating to the sync implementation
            await runnable.ainvoke(1) # returns 2
            await runnable.abatch([1, 2, 3]) # returns [2, 3, 4]


            # Alternatively, can provide both synd and sync implementations
            async def add_one_async(x: int) -> int:
                return x + 1

            runnable = RunnableLambda(add_one, afunc=add_one_async)
            runnable.invoke(1) # Uses add_one
            await runnable.ainvoke(1) # Uses add_one_async
    """

    def __init__(
        self,
        func: Union[
            Union[
                Callable[[Input], Output],
                Callable[[Input], Iterator[Output]],
                Callable[[Input, RunnableConfig], Output],
                Callable[[Input, CallbackManagerForChainRun], Output],
                Callable[[Input, CallbackManagerForChainRun, RunnableConfig], Output],
            ],
            Union[
                Callable[[Input], Awaitable[Output]],
                Callable[[Input], AsyncIterator[Output]],
                Callable[[Input, RunnableConfig], Awaitable[Output]],
                Callable[[Input, AsyncCallbackManagerForChainRun], Awaitable[Output]],
                Callable[
                    [Input, AsyncCallbackManagerForChainRun, RunnableConfig],
                    Awaitable[Output],
                ],
            ],
        ],
        afunc: Optional[
            Union[
                Callable[[Input], Awaitable[Output]],
                Callable[[Input], AsyncIterator[Output]],
                Callable[[Input, RunnableConfig], Awaitable[Output]],
                Callable[[Input, AsyncCallbackManagerForChainRun], Awaitable[Output]],
                Callable[
                    [Input, AsyncCallbackManagerForChainRun, RunnableConfig],
                    Awaitable[Output],
                ],
            ]
        ] = None,
        name: Optional[str] = None,
    ) -> None:
        """Create a RunnableLambda from a callable, and async callable or both.

        Accepts both sync and async variants to allow providing efficient
        implementations for sync and async execution.

        Args:
            func: Either sync or async callable
            afunc: An async callable that takes an input and returns an output.
                Defaults to None.
            name: The name of the Runnable. Defaults to None.

        Raises:
            TypeError: If the func is not a callable type.
            TypeError: If both func and afunc are provided.
        """
        if afunc is not None:
            self.afunc = afunc
            func_for_name: Callable = afunc

        if is_async_callable(func) or is_async_generator(func):
            if afunc is not None:
                msg = (
                    "Func was provided as a coroutine function, but afunc was "
                    "also provided. If providing both, func should be a regular "
                    "function to avoid ambiguity."
                )
                raise TypeError(msg)
            self.afunc = func
            func_for_name = func
        elif callable(func):
            self.func = cast(Callable[[Input], Output], func)
            func_for_name = func
        else:
            msg = (
                "Expected a callable type for `func`."
                f"Instead got an unsupported type: {type(func)}"
            )
            raise TypeError(msg)

        try:
            if name is not None:
                self.name = name
            elif func_for_name.__name__ != "<lambda>":
                self.name = func_for_name.__name__
        except AttributeError:
            pass

        self._repr: Optional[str] = None

    @property
    @override
    def InputType(self) -> Any:
        """The type of the input to this Runnable."""
        func = getattr(self, "func", None) or self.afunc
        try:
            params = inspect.signature(func).parameters
            first_param = next(iter(params.values()), None)
            if first_param and first_param.annotation != inspect.Parameter.empty:
                return first_param.annotation
            else:
                return Any
        except ValueError:
            return Any

    def get_input_schema(
        self, config: Optional[RunnableConfig] = None
    ) -> type[BaseModel]:
        """The pydantic schema for the input to this Runnable.

        Args:
            config: The config to use. Defaults to None.

        Returns:
            The input schema for this Runnable.
        """
        func = getattr(self, "func", None) or self.afunc

        if isinstance(func, itemgetter):
            # This is terrible, but afaict it's not possible to access _items
            # on itemgetter objects, so we have to parse the repr
            items = str(func).replace("operator.itemgetter(", "")[:-1].split(", ")
            if all(
                item[0] == "'" and item[-1] == "'" and len(item) > 2 for item in items
            ):
                fields = {item[1:-1]: (Any, ...) for item in items}
                # It's a dict, lol
                return create_model_v2(self.get_name("Input"), field_definitions=fields)
            else:
                module = getattr(func, "__module__", None)
                return create_model_v2(
                    self.get_name("Input"),
                    root=list[Any],
                    # To create the schema, we need to provide the module
                    # where the underlying function is defined.
                    # This allows pydantic to resolve type annotations appropriately.
                    module_name=module,
                )

        if self.InputType != Any:
            return super().get_input_schema(config)

        if dict_keys := get_function_first_arg_dict_keys(func):
            return create_model_v2(
                self.get_name("Input"),
                field_definitions=dict.fromkeys(dict_keys, (Any, ...)),
            )

        return super().get_input_schema(config)

    @property
    @override
    def OutputType(self) -> Any:
        """The type of the output of this Runnable as a type annotation.

        Returns:
            The type of the output of this Runnable.
        """
        func = getattr(self, "func", None) or self.afunc
        try:
            sig = inspect.signature(func)
            if sig.return_annotation != inspect.Signature.empty:
                # unwrap iterator types
                if getattr(sig.return_annotation, "__origin__", None) in (
                    collections.abc.Iterator,
                    collections.abc.AsyncIterator,
                ):
                    return getattr(sig.return_annotation, "__args__", (Any,))[0]
                return sig.return_annotation
            else:
                return Any
        except ValueError:
            return Any

    def get_output_schema(
        self, config: Optional[RunnableConfig] = None
    ) -> type[BaseModel]:
        # Override the default implementation.
        # For a runnable lambda, we need to bring to provide the
        # module of the underlying function when creating the model.
        root_type = self.OutputType
        func = getattr(self, "func", None) or self.afunc
        module = getattr(func, "__module__", None)

        if (
            inspect.isclass(root_type)
            and not isinstance(root_type, GenericAlias)
            and issubclass(root_type, BaseModel)
        ):
            return root_type

        return create_model_v2(
            self.get_name("Output"),
            root=root_type,
            # To create the schema, we need to provide the module
            # where the underlying function is defined.
            # This allows pydantic to resolve type annotations appropriately.
            module_name=module,
        )

    @functools.cached_property
    def deps(self) -> list[Runnable]:
        """The dependencies of this Runnable.

        Returns:
            The dependencies of this Runnable. If the function has nonlocal
            variables that are Runnables, they are considered dependencies.
        """
        if hasattr(self, "func"):
            objects = get_function_nonlocals(self.func)
        elif hasattr(self, "afunc"):
            objects = get_function_nonlocals(self.afunc)
        else:
            objects = []

        deps: list[Runnable] = []
        for obj in objects:
            if isinstance(obj, Runnable):
                deps.append(obj)
            elif isinstance(getattr(obj, "__self__", None), Runnable):
                deps.append(obj.__self__)
        return deps

    @property
    def config_specs(self) -> list[ConfigurableFieldSpec]:
        return get_unique_config_specs(
            spec for dep in self.deps for spec in dep.config_specs
        )

    def get_graph(self, config: RunnableConfig | None = None) -> Graph:
        if deps := self.deps:
            graph = Graph()
            input_node = graph.add_node(self.get_input_schema(config))
            output_node = graph.add_node(self.get_output_schema(config))
            for dep in deps:
                dep_graph = dep.get_graph()
                dep_graph.trim_first_node()
                dep_graph.trim_last_node()
                if not dep_graph:
                    graph.add_edge(input_node, output_node)
                else:
                    dep_first_node, dep_last_node = graph.extend(dep_graph)
                    if not dep_first_node:
                        msg = f"Runnable {dep} has no first node"
                        raise ValueError(msg)
                    if not dep_last_node:
                        msg = f"Runnable {dep} has no last node"
                        raise ValueError(msg)
                    graph.add_edge(input_node, dep_first_node)
                    graph.add_edge(dep_last_node, output_node)
        else:
            graph = super().get_graph(config)

        return graph

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, RunnableLambda):
            if hasattr(self, "func") and hasattr(other, "func"):
                return self.func == other.func
            elif hasattr(self, "afunc") and hasattr(other, "afunc"):
                return self.afunc == other.afunc
            else:
                return False
        else:
            return False

    def __repr__(self) -> str:
        """A string representation of this Runnable."""
        if self._repr is None:
            if hasattr(self, "func") and isinstance(self.func, itemgetter):
                self._repr = f"RunnableLambda({str(self.func)[len('operator.') :]})"
            elif hasattr(self, "func"):
                self._repr = f"RunnableLambda({get_lambda_source(self.func) or '...'})"
            elif hasattr(self, "afunc"):
                self._repr = (
                    f"RunnableLambda(afunc={get_lambda_source(self.afunc) or '...'})"
                )
            else:
                self._repr = "RunnableLambda(...)"
        return self._repr

    def _invoke(
        self,
        input: Input,
        run_manager: CallbackManagerForChainRun,
        config: RunnableConfig,
        **kwargs: Any,
    ) -> Output:
        if inspect.isgeneratorfunction(self.func):
            output: Optional[Output] = None
            for chunk in call_func_with_variable_args(
                cast(Callable[[Input], Iterator[Output]], self.func),
                input,
                config,
                run_manager,
                **kwargs,
            ):
                if output is None:
                    output = chunk
                else:
                    try:
                        output = output + chunk  # type: ignore[operator]
                    except TypeError:
                        output = chunk
        else:
            output = call_func_with_variable_args(
                self.func, input, config, run_manager, **kwargs
            )
        # If the output is a Runnable, invoke it
        if isinstance(output, Runnable):
            recursion_limit = config["recursion_limit"]
            if recursion_limit <= 0:
                msg = (
                    f"Recursion limit reached when invoking {self} with input {input}."
                )
                raise RecursionError(msg)
            output = output.invoke(
                input,
                patch_config(
                    config,
                    callbacks=run_manager.get_child(),
                    recursion_limit=recursion_limit - 1,
                ),
            )
        return cast(Output, output)

    async def _ainvoke(
        self,
        input: Input,
        run_manager: AsyncCallbackManagerForChainRun,
        config: RunnableConfig,
        **kwargs: Any,
    ) -> Output:
        if hasattr(self, "afunc"):
            afunc = self.afunc
        else:
            if inspect.isgeneratorfunction(self.func):

                def func(
                    input: Input,
                    run_manager: AsyncCallbackManagerForChainRun,
                    config: RunnableConfig,
                    **kwargs: Any,
                ) -> Output:
                    output: Optional[Output] = None
                    for chunk in call_func_with_variable_args(
                        cast(Callable[[Input], Iterator[Output]], self.func),
                        input,
                        config,
                        run_manager.get_sync(),
                        **kwargs,
                    ):
                        if output is None:
                            output = chunk
                        else:
                            try:
                                output = output + chunk  # type: ignore[operator]
                            except TypeError:
                                output = chunk
                    return cast(Output, output)

            else:

                def func(
                    input: Input,
                    run_manager: AsyncCallbackManagerForChainRun,
                    config: RunnableConfig,
                    **kwargs: Any,
                ) -> Output:
                    return call_func_with_variable_args(
                        self.func, input, config, run_manager.get_sync(), **kwargs
                    )

            @wraps(func)
            async def f(*args: Any, **kwargs: Any) -> Any:
                return await run_in_executor(config, func, *args, **kwargs)

            afunc = f

        if is_async_generator(afunc):
            output: Optional[Output] = None
            async with aclosing(
                cast(
                    AsyncGenerator[Any, Any],
                    acall_func_with_variable_args(
                        cast(Callable, afunc),
                        input,
                        config,
                        run_manager,
                        **kwargs,
                    ),
                )
            ) as stream:
                async for chunk in cast(
                    AsyncIterator[Output],
                    stream,
                ):
                    if output is None:
                        output = chunk
                    else:
                        try:
                            output = output + chunk  # type: ignore[operator]
                        except TypeError:
                            output = chunk
        else:
            output = await acall_func_with_variable_args(
                cast(Callable, afunc), input, config, run_manager, **kwargs
            )
        # If the output is a Runnable, invoke it
        if isinstance(output, Runnable):
            recursion_limit = config["recursion_limit"]
            if recursion_limit <= 0:
                msg = (
                    f"Recursion limit reached when invoking {self} with input {input}."
                )
                raise RecursionError(msg)
            output = await output.ainvoke(
                input,
                patch_config(
                    config,
                    callbacks=run_manager.get_child(),
                    recursion_limit=recursion_limit - 1,
                ),
            )
        return cast(Output, output)

    def _config(
        self, config: Optional[RunnableConfig], callable: Callable[..., Any]
    ) -> RunnableConfig:
        return ensure_config(config)

    def invoke(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> Output:
        """Invoke this Runnable synchronously.

        Args:
            input: The input to this Runnable.
            config: The config to use. Defaults to None.
            kwargs: Additional keyword arguments.

        Returns:
            The output of this Runnable.

        Raises:
            TypeError: If the Runnable is a coroutine function.
        """
        if hasattr(self, "func"):
            return self._call_with_config(
                self._invoke,
                input,
                self._config(config, self.func),
                **kwargs,
            )
        else:
            msg = (
                "Cannot invoke a coroutine function synchronously."
                "Use `ainvoke` instead."
            )
            raise TypeError(msg)

    async def ainvoke(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> Output:
        """Invoke this Runnable asynchronously.

        Args:
            input: The input to this Runnable.
            config: The config to use. Defaults to None.
            kwargs: Additional keyword arguments.

        Returns:
            The output of this Runnable.
        """
        the_func = self.afunc if hasattr(self, "afunc") else self.func
        return await self._acall_with_config(
            self._ainvoke,
            input,
            self._config(config, the_func),
            **kwargs,
        )

    def _transform(
        self,
        input: Iterator[Input],
        run_manager: CallbackManagerForChainRun,
        config: RunnableConfig,
        **kwargs: Any,
    ) -> Iterator[Output]:
        final: Input
        got_first_val = False
        for ichunk in input:
            # By definitions, RunnableLambdas consume all input before emitting output.
            # If the input is not addable, then we'll assume that we can
            # only operate on the last chunk.
            # So we'll iterate until we get to the last chunk!
            if not got_first_val:
                final = ichunk
                got_first_val = True
            else:
                try:
                    final = final + ichunk  # type: ignore[operator]
                except TypeError:
                    final = ichunk

        if inspect.isgeneratorfunction(self.func):
            output: Optional[Output] = None
            for chunk in call_func_with_variable_args(
                self.func, cast(Input, final), config, run_manager, **kwargs
            ):
                yield chunk
                if output is None:
                    output = chunk
                else:
                    try:
                        output = output + chunk
                    except TypeError:
                        output = chunk
        else:
            output = call_func_with_variable_args(
                self.func, cast(Input, final), config, run_manager, **kwargs
            )

        # If the output is a Runnable, use its stream output
        if isinstance(output, Runnable):
            recursion_limit = config["recursion_limit"]
            if recursion_limit <= 0:
                msg = (
                    f"Recursion limit reached when invoking {self} with input {final}."
                )
                raise RecursionError(msg)
            for chunk in output.stream(
                final,
                patch_config(
                    config,
                    callbacks=run_manager.get_child(),
                    recursion_limit=recursion_limit - 1,
                ),
            ):
                yield chunk
        elif not inspect.isgeneratorfunction(self.func):
            # Otherwise, just yield it
            yield cast(Output, output)

    def transform(
        self,
        input: Iterator[Input],
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> Iterator[Output]:
        if hasattr(self, "func"):
            yield from self._transform_stream_with_config(
                input,
                self._transform,
                self._config(config, self.func),
                **kwargs,
            )
        else:
            msg = (
                "Cannot stream a coroutine function synchronously."
                "Use `astream` instead."
            )
            raise TypeError(msg)

    def stream(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> Iterator[Output]:
        return self.transform(iter([input]), config, **kwargs)

    async def _atransform(
        self,
        input: AsyncIterator[Input],
        run_manager: AsyncCallbackManagerForChainRun,
        config: RunnableConfig,
        **kwargs: Any,
    ) -> AsyncIterator[Output]:
        final: Input
        got_first_val = False
        async for ichunk in input:
            # By definitions, RunnableLambdas consume all input before emitting output.
            # If the input is not addable, then we'll assume that we can
            # only operate on the last chunk.
            # So we'll iterate until we get to the last chunk!
            if not got_first_val:
                final = ichunk
                got_first_val = True
            else:
                try:
                    final = final + ichunk  # type: ignore[operator]
                except TypeError:
                    final = ichunk

        if hasattr(self, "afunc"):
            afunc = self.afunc
        else:
            if inspect.isgeneratorfunction(self.func):
                msg = (
                    "Cannot stream from a generator function asynchronously."
                    "Use .stream() instead."
                )
                raise TypeError(msg)

            def func(
                input: Input,
                run_manager: AsyncCallbackManagerForChainRun,
                config: RunnableConfig,
                **kwargs: Any,
            ) -> Output:
                return call_func_with_variable_args(
                    self.func, input, config, run_manager.get_sync(), **kwargs
                )

            @wraps(func)
            async def f(*args: Any, **kwargs: Any) -> Any:
                return await run_in_executor(config, func, *args, **kwargs)

            afunc = f

        if is_async_generator(afunc):
            output: Optional[Output] = None
            async for chunk in cast(
                AsyncIterator[Output],
                acall_func_with_variable_args(
                    cast(Callable, afunc),
                    cast(Input, final),
                    config,
                    run_manager,
                    **kwargs,
                ),
            ):
                yield chunk
                if output is None:
                    output = chunk
                else:
                    try:
                        output = output + chunk  # type: ignore[operator]
                    except TypeError:
                        output = chunk
        else:
            output = await acall_func_with_variable_args(
                cast(Callable, afunc), cast(Input, final), config, run_manager, **kwargs
            )

        # If the output is a Runnable, use its astream output
        if isinstance(output, Runnable):
            recursion_limit = config["recursion_limit"]
            if recursion_limit <= 0:
                msg = (
                    f"Recursion limit reached when invoking {self} with input {final}."
                )
                raise RecursionError(msg)
            async for chunk in output.astream(
                final,
                patch_config(
                    config,
                    callbacks=run_manager.get_child(),
                    recursion_limit=recursion_limit - 1,
                ),
            ):
                yield chunk
        elif not is_async_generator(afunc):
            # Otherwise, just yield it
            yield cast(Output, output)

    async def atransform(
        self,
        input: AsyncIterator[Input],
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> AsyncIterator[Output]:
        async for output in self._atransform_stream_with_config(
            input,
            self._atransform,
            self._config(config, self.afunc if hasattr(self, "afunc") else self.func),
            **kwargs,
        ):
            yield output

    async def astream(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> AsyncIterator[Output]:
        async def input_aiter() -> AsyncIterator[Input]:
            yield input

        async for chunk in self.atransform(input_aiter(), config, **kwargs):
            yield chunk


class RunnableEachBase(RunnableSerializable[list[Input], list[Output]]):
    """Runnable that delegates calls to another Runnable
    with each element of the input sequence.

    Use only if creating a new RunnableEach subclass with different __init__ args.

    See documentation for RunnableEach for more details.
    """

    bound: Runnable[Input, Output]

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    @property
    @override
    def InputType(self) -> Any:
        return list[self.bound.InputType]  # type: ignore[name-defined]

    def get_input_schema(
        self, config: Optional[RunnableConfig] = None
    ) -> type[BaseModel]:
        return create_model_v2(
            self.get_name("Input"),
            root=(
                list[self.bound.get_input_schema(config)],  # type: ignore
                None,
            ),
            # create model needs access to appropriate type annotations to be
            # able to construct the pydantic model.
            # When we create the model, we pass information about the namespace
            # where the model is being created, so the type annotations can
            # be resolved correctly as well.
            # self.__class__.__module__ handles the case when the Runnable is
            # being sub-classed in a different module.
            module_name=self.__class__.__module__,
        )

    @property
    @override
    def OutputType(self) -> type[list[Output]]:
        return list[self.bound.OutputType]  # type: ignore[name-defined]

    def get_output_schema(
        self, config: Optional[RunnableConfig] = None
    ) -> type[BaseModel]:
        schema = self.bound.get_output_schema(config)
        return create_model_v2(
            self.get_name("Output"),
            root=list[schema],  # type: ignore[valid-type]
            # create model needs access to appropriate type annotations to be
            # able to construct the pydantic model.
            # When we create the model, we pass information about the namespace
            # where the model is being created, so the type annotations can
            # be resolved correctly as well.
            # self.__class__.__module__ handles the case when the Runnable is
            # being sub-classed in a different module.
            module_name=self.__class__.__module__,
        )

    @property
    def config_specs(self) -> list[ConfigurableFieldSpec]:
        return self.bound.config_specs

    def get_graph(self, config: Optional[RunnableConfig] = None) -> Graph:
        return self.bound.get_graph(config)

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return True

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "schema", "runnable"]

    def _invoke(
        self,
        inputs: list[Input],
        run_manager: CallbackManagerForChainRun,
        config: RunnableConfig,
        **kwargs: Any,
    ) -> list[Output]:
        configs = [
            patch_config(config, callbacks=run_manager.get_child()) for _ in inputs
        ]
        return self.bound.batch(inputs, configs, **kwargs)

    def invoke(
        self, input: list[Input], config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> list[Output]:
        return self._call_with_config(self._invoke, input, config, **kwargs)

    async def _ainvoke(
        self,
        inputs: list[Input],
        run_manager: AsyncCallbackManagerForChainRun,
        config: RunnableConfig,
        **kwargs: Any,
    ) -> list[Output]:
        configs = [
            patch_config(config, callbacks=run_manager.get_child()) for _ in inputs
        ]
        return await self.bound.abatch(inputs, configs, **kwargs)

    async def ainvoke(
        self, input: list[Input], config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> list[Output]:
        return await self._acall_with_config(self._ainvoke, input, config, **kwargs)

    async def astream_events(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> AsyncIterator[StreamEvent]:
        for _ in range(1):
            msg = "RunnableEach does not support astream_events yet."
            raise NotImplementedError(msg)
            yield


class RunnableEach(RunnableEachBase[Input, Output]):
    """Runnable that delegates calls to another Runnable
    with each element of the input sequence.

    It allows you to call multiple inputs with the bounded Runnable.

    RunnableEach makes it easy to run multiple inputs for the Runnable.
    In the below example, we associate and run three inputs
    with a Runnable:

        .. code-block:: python

            from langchain_core.runnables.base import RunnableEach
            from langchain_openai import ChatOpenAI
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_core.output_parsers import StrOutputParser
            prompt = ChatPromptTemplate.from_template("Tell me a short joke about
            {topic}")
            model = ChatOpenAI()
            output_parser = StrOutputParser()
            runnable = prompt | model | output_parser
            runnable_each = RunnableEach(bound=runnable)
            output = runnable_each.invoke([{'topic':'Computer Science'},
                                        {'topic':'Art'},
                                        {'topic':'Biology'}])
            print(output)  # noqa: T201
    """

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "schema", "runnable"]

    def get_name(
        self, suffix: Optional[str] = None, *, name: Optional[str] = None
    ) -> str:
        name = name or self.name or f"RunnableEach<{self.bound.get_name()}>"
        return super().get_name(suffix, name=name)

    def bind(self, **kwargs: Any) -> RunnableEach[Input, Output]:
        return RunnableEach(bound=self.bound.bind(**kwargs))

    def with_config(
        self, config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> RunnableEach[Input, Output]:
        return RunnableEach(bound=self.bound.with_config(config, **kwargs))

    def with_listeners(
        self,
        *,
        on_start: Optional[
            Union[Callable[[Run], None], Callable[[Run, RunnableConfig], None]]
        ] = None,
        on_end: Optional[
            Union[Callable[[Run], None], Callable[[Run, RunnableConfig], None]]
        ] = None,
        on_error: Optional[
            Union[Callable[[Run], None], Callable[[Run, RunnableConfig], None]]
        ] = None,
    ) -> RunnableEach[Input, Output]:
        """Bind lifecycle listeners to a Runnable, returning a new Runnable.

        Args:
            on_start: Called before the Runnable starts running, with the Run object.
                Defaults to None.
            on_end: Called after the Runnable finishes running, with the Run object.
                Defaults to None.
            on_error: Called if the Runnable throws an error, with the Run object.
                Defaults to None.

        Returns:
            A new Runnable with the listeners bound.

        The Run object contains information about the run, including its id,
        type, input, output, error, start_time, end_time, and any tags or metadata
        added to the run.
        """
        return RunnableEach(
            bound=self.bound.with_listeners(
                on_start=on_start, on_end=on_end, on_error=on_error
            )
        )

    def with_alisteners(
        self,
        *,
        on_start: Optional[AsyncListener] = None,
        on_end: Optional[AsyncListener] = None,
        on_error: Optional[AsyncListener] = None,
    ) -> RunnableEach[Input, Output]:
        """Bind async lifecycle listeners to a Runnable, returning a new Runnable.

        Args:
            on_start: Called asynchronously before the Runnable starts running,
                      with the Run object. Defaults to None.
            on_end: Called asynchronously after the Runnable finishes running,
                    with the Run object. Defaults to None.
            on_error: Called asynchronously if the Runnable throws an error,
                    with the Run object. Defaults to None.

        Returns:
            A new Runnable with the listeners bound.

        The Run object contains information about the run, including its id,
        type, input, output, error, start_time, end_time, and any tags or metadata
        added to the run.
        """
        return RunnableEach(
            bound=self.bound.with_alisteners(
                on_start=on_start, on_end=on_end, on_error=on_error
            )
        )


class RunnableBindingBase(RunnableSerializable[Input, Output]):
    """Runnable that delegates calls to another Runnable with a set of kwargs.

    Use only if creating a new RunnableBinding subclass with different __init__ args.

    See documentation for RunnableBinding for more details.
    """

    bound: Runnable[Input, Output]
    """The underlying Runnable that this Runnable delegates to."""

    kwargs: Mapping[str, Any] = Field(default_factory=dict)
    """kwargs to pass to the underlying Runnable when running.

    For example, when the Runnable binding is invoked the underlying
    Runnable will be invoked with the same input but with these additional
    kwargs.
    """

    config: RunnableConfig = Field(default_factory=RunnableConfig)  # type: ignore
    """The config to bind to the underlying Runnable."""

    config_factories: list[Callable[[RunnableConfig], RunnableConfig]] = Field(
        default_factory=list
    )
    """The config factories to bind to the underlying Runnable."""

    # Union[Type[Input], BaseModel] + things like List[str]
    custom_input_type: Optional[Any] = None
    """Override the input type of the underlying Runnable with a custom type.

    The type can be a pydantic model, or a type annotation (e.g., `List[str]`).
    """
    # Union[Type[Output], BaseModel] + things like List[str]
    custom_output_type: Optional[Any] = None
    """Override the output type of the underlying Runnable with a custom type.

    The type can be a pydantic model, or a type annotation (e.g., `List[str]`).
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    def __init__(
        self,
        *,
        bound: Runnable[Input, Output],
        kwargs: Optional[Mapping[str, Any]] = None,
        config: Optional[RunnableConfig] = None,
        config_factories: Optional[
            list[Callable[[RunnableConfig], RunnableConfig]]
        ] = None,
        custom_input_type: Optional[Union[type[Input], BaseModel]] = None,
        custom_output_type: Optional[Union[type[Output], BaseModel]] = None,
        **other_kwargs: Any,
    ) -> None:
        """Create a RunnableBinding from a Runnable and kwargs.

        Args:
            bound: The underlying Runnable that this Runnable delegates calls to.
            kwargs: optional kwargs to pass to the underlying Runnable, when running
                    the underlying Runnable (e.g., via `invoke`, `batch`,
                    `transform`, or `stream` or async variants)
                    Defaults to None.
            config: optional config to bind to the underlying Runnable.
                    Defaults to None.
            config_factories: optional list of config factories to apply to the
                    config before binding to the underlying Runnable.
                    Defaults to None.
            custom_input_type: Specify to override the input type of the underlying
                               Runnable with a custom type. Defaults to None.
            custom_output_type: Specify to override the output type of the underlying
                Runnable with a custom type. Defaults to None.
            **other_kwargs: Unpacked into the base class.
        """
        super().__init__(  # type: ignore[call-arg]
            bound=bound,
            kwargs=kwargs or {},
            config=config or {},
            config_factories=config_factories or [],
            custom_input_type=custom_input_type,
            custom_output_type=custom_output_type,
            **other_kwargs,
        )
        # if we don't explicitly set config to the TypedDict here,
        # the pydantic init above will strip out any of the "extra"
        # fields even though total=False on the typed dict.
        self.config = config or {}

    def get_name(
        self, suffix: Optional[str] = None, *, name: Optional[str] = None
    ) -> str:
        return self.bound.get_name(suffix, name=name)

    @property
    @override
    def InputType(self) -> type[Input]:
        return (
            cast(type[Input], self.custom_input_type)
            if self.custom_input_type is not None
            else self.bound.InputType
        )

    @property
    @override
    def OutputType(self) -> type[Output]:
        return (
            cast(type[Output], self.custom_output_type)
            if self.custom_output_type is not None
            else self.bound.OutputType
        )

    def get_input_schema(
        self, config: Optional[RunnableConfig] = None
    ) -> type[BaseModel]:
        if self.custom_input_type is not None:
            return super().get_input_schema(config)
        return self.bound.get_input_schema(merge_configs(self.config, config))

    def get_output_schema(
        self, config: Optional[RunnableConfig] = None
    ) -> type[BaseModel]:
        if self.custom_output_type is not None:
            return super().get_output_schema(config)
        return self.bound.get_output_schema(merge_configs(self.config, config))

    @property
    def config_specs(self) -> list[ConfigurableFieldSpec]:
        return self.bound.config_specs

    def get_graph(self, config: Optional[RunnableConfig] = None) -> Graph:
        return self.bound.get_graph(self._merge_configs(config))

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return True

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "schema", "runnable"]

    def _merge_configs(self, *configs: Optional[RunnableConfig]) -> RunnableConfig:
        config = merge_configs(self.config, *configs)
        return merge_configs(config, *(f(config) for f in self.config_factories))

    def invoke(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> Output:
        return self.bound.invoke(
            input,
            self._merge_configs(config),
            **{**self.kwargs, **kwargs},
        )

    async def ainvoke(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> Output:
        return await self.bound.ainvoke(
            input,
            self._merge_configs(config),
            **{**self.kwargs, **kwargs},
        )

    def batch(
        self,
        inputs: list[Input],
        config: Optional[Union[RunnableConfig, list[RunnableConfig]]] = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Optional[Any],
    ) -> list[Output]:
        if isinstance(config, list):
            configs = cast(
                list[RunnableConfig],
                [self._merge_configs(conf) for conf in config],
            )
        else:
            configs = [self._merge_configs(config) for _ in range(len(inputs))]
        return self.bound.batch(
            inputs,
            configs,
            return_exceptions=return_exceptions,
            **{**self.kwargs, **kwargs},
        )

    async def abatch(
        self,
        inputs: list[Input],
        config: Optional[Union[RunnableConfig, list[RunnableConfig]]] = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Optional[Any],
    ) -> list[Output]:
        if isinstance(config, list):
            configs = cast(
                list[RunnableConfig],
                [self._merge_configs(conf) for conf in config],
            )
        else:
            configs = [self._merge_configs(config) for _ in range(len(inputs))]
        return await self.bound.abatch(
            inputs,
            configs,
            return_exceptions=return_exceptions,
            **{**self.kwargs, **kwargs},
        )

    @overload
    def batch_as_completed(
        self,
        inputs: Sequence[Input],
        config: Optional[Union[RunnableConfig, Sequence[RunnableConfig]]] = None,
        *,
        return_exceptions: Literal[False] = False,
        **kwargs: Any,
    ) -> Iterator[tuple[int, Output]]: ...

    @overload
    def batch_as_completed(
        self,
        inputs: Sequence[Input],
        config: Optional[Union[RunnableConfig, Sequence[RunnableConfig]]] = None,
        *,
        return_exceptions: Literal[True],
        **kwargs: Any,
    ) -> Iterator[tuple[int, Union[Output, Exception]]]: ...

    def batch_as_completed(
        self,
        inputs: Sequence[Input],
        config: Optional[Union[RunnableConfig, Sequence[RunnableConfig]]] = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Optional[Any],
    ) -> Iterator[tuple[int, Union[Output, Exception]]]:
        if isinstance(config, Sequence):
            configs = cast(
                list[RunnableConfig],
                [self._merge_configs(conf) for conf in config],
            )
        else:
            configs = [self._merge_configs(config) for _ in range(len(inputs))]
        # lol mypy
        if return_exceptions:
            yield from self.bound.batch_as_completed(
                inputs,
                configs,
                return_exceptions=return_exceptions,
                **{**self.kwargs, **kwargs},
            )
        else:
            yield from self.bound.batch_as_completed(
                inputs,
                configs,
                return_exceptions=return_exceptions,
                **{**self.kwargs, **kwargs},
            )

    @overload
    def abatch_as_completed(
        self,
        inputs: Sequence[Input],
        config: Optional[Union[RunnableConfig, Sequence[RunnableConfig]]] = None,
        *,
        return_exceptions: Literal[False] = False,
        **kwargs: Optional[Any],
    ) -> AsyncIterator[tuple[int, Output]]: ...

    @overload
    def abatch_as_completed(
        self,
        inputs: Sequence[Input],
        config: Optional[Union[RunnableConfig, Sequence[RunnableConfig]]] = None,
        *,
        return_exceptions: Literal[True],
        **kwargs: Optional[Any],
    ) -> AsyncIterator[tuple[int, Union[Output, Exception]]]: ...

    async def abatch_as_completed(
        self,
        inputs: Sequence[Input],
        config: Optional[Union[RunnableConfig, Sequence[RunnableConfig]]] = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Optional[Any],
    ) -> AsyncIterator[tuple[int, Union[Output, Exception]]]:
        if isinstance(config, Sequence):
            configs = cast(
                list[RunnableConfig],
                [self._merge_configs(conf) for conf in config],
            )
        else:
            configs = [self._merge_configs(config) for _ in range(len(inputs))]
        if return_exceptions:
            async for item in self.bound.abatch_as_completed(
                inputs,
                configs,
                return_exceptions=return_exceptions,
                **{**self.kwargs, **kwargs},
            ):
                yield item
        else:
            async for item in self.bound.abatch_as_completed(
                inputs,
                configs,
                return_exceptions=return_exceptions,
                **{**self.kwargs, **kwargs},
            ):
                yield item

    def stream(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> Iterator[Output]:
        yield from self.bound.stream(
            input,
            self._merge_configs(config),
            **{**self.kwargs, **kwargs},
        )

    async def astream(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> AsyncIterator[Output]:
        async for item in self.bound.astream(
            input,
            self._merge_configs(config),
            **{**self.kwargs, **kwargs},
        ):
            yield item

    async def astream_events(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> AsyncIterator[StreamEvent]:
        async for item in self.bound.astream_events(
            input, self._merge_configs(config), **{**self.kwargs, **kwargs}
        ):
            yield item

    def transform(
        self,
        input: Iterator[Input],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Iterator[Output]:
        yield from self.bound.transform(
            input,
            self._merge_configs(config),
            **{**self.kwargs, **kwargs},
        )

    async def atransform(
        self,
        input: AsyncIterator[Input],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> AsyncIterator[Output]:
        async for item in self.bound.atransform(
            input,
            self._merge_configs(config),
            **{**self.kwargs, **kwargs},
        ):
            yield item


RunnableBindingBase.model_rebuild()


class RunnableBinding(RunnableBindingBase[Input, Output]):
    """Wrap a Runnable with additional functionality.

    A RunnableBinding can be thought of as a "runnable decorator" that
    preserves the essential features of Runnable; i.e., batching, streaming,
    and async support, while adding additional functionality.

    Any class that inherits from Runnable can be bound to a `RunnableBinding`.
    Runnables expose a standard set of methods for creating `RunnableBindings`
    or sub-classes of `RunnableBindings` (e.g., `RunnableRetry`,
    `RunnableWithFallbacks`) that add additional functionality.

    These methods include:

    - ``bind``: Bind kwargs to pass to the underlying Runnable when running it.
    - ``with_config``: Bind config to pass to the underlying Runnable when running it.
    - ``with_listeners``:  Bind lifecycle listeners to the underlying Runnable.
    - ``with_types``: Override the input and output types of the underlying Runnable.
    - ``with_retry``: Bind a retry policy to the underlying Runnable.
    - ``with_fallbacks``: Bind a fallback policy to the underlying Runnable.

    Example:
    `bind`: Bind kwargs to pass to the underlying Runnable when running it.

        .. code-block:: python

            # Create a Runnable binding that invokes the ChatModel with the
            # additional kwarg `stop=['-']` when running it.
            from langchain_community.chat_models import ChatOpenAI
            model = ChatOpenAI()
            model.invoke('Say "Parrot-MAGIC"', stop=['-']) # Should return `Parrot`
            # Using it the easy way via `bind` method which returns a new
            # RunnableBinding
            runnable_binding = model.bind(stop=['-'])
            runnable_binding.invoke('Say "Parrot-MAGIC"') # Should return `Parrot`

        Can also be done by instantiating a RunnableBinding directly (not recommended):

        .. code-block:: python

            from langchain_core.runnables import RunnableBinding
            runnable_binding = RunnableBinding(
                bound=model,
                kwargs={'stop': ['-']} # <-- Note the additional kwargs
            )
            runnable_binding.invoke('Say "Parrot-MAGIC"') # Should return `Parrot`
    """

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "schema", "runnable"]

    def bind(self, **kwargs: Any) -> Runnable[Input, Output]:
        """Bind additional kwargs to a Runnable, returning a new Runnable.

        Args:
            **kwargs: The kwargs to bind to the Runnable.

        Returns:
            A new Runnable with the same type and config as the original,
            but with the additional kwargs bound.
        """
        return self.__class__(
            bound=self.bound,
            config=self.config,
            kwargs={**self.kwargs, **kwargs},
            custom_input_type=self.custom_input_type,
            custom_output_type=self.custom_output_type,
        )

    def with_config(
        self,
        config: Optional[RunnableConfig] = None,
        # Sadly Unpack is not well supported by mypy so this will have to be untyped
        **kwargs: Any,
    ) -> Runnable[Input, Output]:
        return self.__class__(
            bound=self.bound,
            kwargs=self.kwargs,
            config=cast(RunnableConfig, {**self.config, **(config or {}), **kwargs}),
            custom_input_type=self.custom_input_type,
            custom_output_type=self.custom_output_type,
        )

    def with_listeners(
        self,
        *,
        on_start: Optional[
            Union[Callable[[Run], None], Callable[[Run, RunnableConfig], None]]
        ] = None,
        on_end: Optional[
            Union[Callable[[Run], None], Callable[[Run, RunnableConfig], None]]
        ] = None,
        on_error: Optional[
            Union[Callable[[Run], None], Callable[[Run, RunnableConfig], None]]
        ] = None,
    ) -> Runnable[Input, Output]:
        """Bind lifecycle listeners to a Runnable, returning a new Runnable.

        Args:
            on_start: Called before the Runnable starts running, with the Run object.
                Defaults to None.
            on_end: Called after the Runnable finishes running, with the Run object.
                Defaults to None.
            on_error: Called if the Runnable throws an error, with the Run object.
                Defaults to None.

        Returns:
            The Runnable object contains information about the run, including its id,
            type, input, output, error, start_time, end_time, and any tags or metadata
            added to the run.
        """
        from langchain_core.tracers.root_listeners import RootListenersTracer

        return self.__class__(
            bound=self.bound,
            kwargs=self.kwargs,
            config=self.config,
            config_factories=[
                lambda config: {
                    "callbacks": [
                        RootListenersTracer(
                            config=config,
                            on_start=on_start,
                            on_end=on_end,
                            on_error=on_error,
                        )
                    ],
                }
            ],
            custom_input_type=self.custom_input_type,
            custom_output_type=self.custom_output_type,
        )

    def with_types(
        self,
        input_type: Optional[Union[type[Input], BaseModel]] = None,
        output_type: Optional[Union[type[Output], BaseModel]] = None,
    ) -> Runnable[Input, Output]:
        return self.__class__(
            bound=self.bound,
            kwargs=self.kwargs,
            config=self.config,
            custom_input_type=(
                input_type if input_type is not None else self.custom_input_type
            ),
            custom_output_type=(
                output_type if output_type is not None else self.custom_output_type
            ),
        )

    def with_retry(self, **kwargs: Any) -> Runnable[Input, Output]:
        return self.__class__(
            bound=self.bound.with_retry(**kwargs),
            kwargs=self.kwargs,
            config=self.config,
        )

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self.bound, name)

        if callable(attr) and (
            config_param := inspect.signature(attr).parameters.get("config")
        ):
            if config_param.kind == inspect.Parameter.KEYWORD_ONLY:

                @wraps(attr)
                def wrapper(*args: Any, **kwargs: Any) -> Any:
                    return attr(
                        *args,
                        config=merge_configs(self.config, kwargs.pop("config", None)),
                        **kwargs,
                    )

                return wrapper
            elif config_param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD:
                idx = list(inspect.signature(attr).parameters).index("config")

                @wraps(attr)
                def wrapper(*args: Any, **kwargs: Any) -> Any:
                    if len(args) >= idx + 1:
                        argsl = list(args)
                        argsl[idx] = merge_configs(self.config, argsl[idx])
                        return attr(*argsl, **kwargs)
                    else:
                        return attr(
                            *args,
                            config=merge_configs(
                                self.config, kwargs.pop("config", None)
                            ),
                            **kwargs,
                        )

                return wrapper

        return attr


class _RunnableCallableSync(Protocol[Input, Output]):
    def __call__(self, __in: Input, *, config: RunnableConfig) -> Output: ...


class _RunnableCallableAsync(Protocol[Input, Output]):
    def __call__(self, __in: Input, *, config: RunnableConfig) -> Awaitable[Output]: ...


class _RunnableCallableIterator(Protocol[Input, Output]):
    def __call__(
        self, __in: Iterator[Input], *, config: RunnableConfig
    ) -> Iterator[Output]: ...


class _RunnableCallableAsyncIterator(Protocol[Input, Output]):
    def __call__(
        self, __in: AsyncIterator[Input], *, config: RunnableConfig
    ) -> AsyncIterator[Output]: ...


RunnableLike = Union[
    Runnable[Input, Output],
    Callable[[Input], Output],
    Callable[[Input], Awaitable[Output]],
    Callable[[Iterator[Input]], Iterator[Output]],
    Callable[[AsyncIterator[Input]], AsyncIterator[Output]],
    _RunnableCallableSync[Input, Output],
    _RunnableCallableAsync[Input, Output],
    _RunnableCallableIterator[Input, Output],
    _RunnableCallableAsyncIterator[Input, Output],
    Mapping[str, Any],
]


def coerce_to_runnable(thing: RunnableLike) -> Runnable[Input, Output]:
    """Coerce a Runnable-like object into a Runnable.

    Args:
        thing: A Runnable-like object.

    Returns:
        A Runnable.

    Raises:
        TypeError: If the object is not Runnable-like.
    """
    if isinstance(thing, Runnable):
        return thing
    elif is_async_generator(thing) or inspect.isgeneratorfunction(thing):
        return RunnableGenerator(thing)
    elif callable(thing):
        return RunnableLambda(cast(Callable[[Input], Output], thing))
    elif isinstance(thing, dict):
        return cast(Runnable[Input, Output], RunnableParallel(thing))
    else:
        msg = (
            f"Expected a Runnable, callable or dict."
            f"Instead got an unsupported type: {type(thing)}"
        )
        raise TypeError(msg)


@overload
def chain(
    func: Callable[[Input], Coroutine[Any, Any, Output]],
) -> Runnable[Input, Output]: ...


@overload
def chain(
    func: Callable[[Input], Iterator[Output]],
) -> Runnable[Input, Output]: ...


@overload
def chain(
    func: Callable[[Input], AsyncIterator[Output]],
) -> Runnable[Input, Output]: ...


@overload
def chain(
    func: Callable[[Input], Output],
) -> Runnable[Input, Output]: ...


def chain(
    func: Union[
        Callable[[Input], Output],
        Callable[[Input], Iterator[Output]],
        Callable[[Input], Coroutine[Any, Any, Output]],
        Callable[[Input], AsyncIterator[Output]],
    ],
) -> Runnable[Input, Output]:
    """Decorate a function to make it a Runnable.
    Sets the name of the Runnable to the name of the function.
    Any runnables called by the function will be traced as dependencies.

    Args:
        func: A callable.

    Returns:
        A Runnable.

    Example:

    .. code-block:: python

        from langchain_core.runnables import chain
        from langchain_core.prompts import PromptTemplate
        from langchain_openai import OpenAI

        @chain
        def my_func(fields):
            prompt = PromptTemplate("Hello, {name}!")
            llm = OpenAI()
            formatted = prompt.invoke(**fields)

            for chunk in llm.stream(formatted):
                yield chunk
    """
    return RunnableLambda(func)
