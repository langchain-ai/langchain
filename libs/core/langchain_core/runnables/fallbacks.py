"""Runnable that can fallback to other Runnables if it fails."""

import asyncio
import inspect
import typing
from collections.abc import AsyncIterator, Iterator, Sequence
from functools import wraps
from typing import (
    TYPE_CHECKING,
    Any,
    Optional,
    Union,
)

from pydantic import BaseModel, ConfigDict
from typing_extensions import override

from langchain_core.runnables.base import Runnable, RunnableSerializable
from langchain_core.runnables.config import (
    RunnableConfig,
    ensure_config,
    get_async_callback_manager_for_config,
    get_callback_manager_for_config,
    get_config_list,
    patch_config,
    set_config_context,
)
from langchain_core.runnables.utils import (
    ConfigurableFieldSpec,
    Input,
    Output,
    coro_with_context,
    get_unique_config_specs,
)
from langchain_core.utils.aiter import py_anext

if TYPE_CHECKING:
    from langchain_core.callbacks.manager import AsyncCallbackManagerForChainRun


class RunnableWithFallbacks(RunnableSerializable[Input, Output]):
    """Runnable that can fallback to other Runnables if it fails.

    External APIs (e.g., APIs for a language model) may at times experience
    degraded performance or even downtime.

    In these cases, it can be useful to have a fallback Runnable that can be
    used in place of the original Runnable (e.g., fallback to another LLM provider).

    Fallbacks can be defined at the level of a single Runnable, or at the level
    of a chain of Runnables. Fallbacks are tried in order until one succeeds or
    all fail.

    While you can instantiate a ``RunnableWithFallbacks`` directly, it is usually
    more convenient to use the ``with_fallbacks`` method on a Runnable.

    Example:

        .. code-block:: python

            from langchain_core.chat_models.openai import ChatOpenAI
            from langchain_core.chat_models.anthropic import ChatAnthropic

            model = ChatAnthropic(
                model="claude-3-haiku-20240307"
            ).with_fallbacks([ChatOpenAI(model="gpt-3.5-turbo-0125")])
            # Will usually use ChatAnthropic, but fallback to ChatOpenAI
            # if ChatAnthropic fails.
            model.invoke('hello')

            # And you can also use fallbacks at the level of a chain.
            # Here if both LLM providers fail, we'll fallback to a good hardcoded
            # response.

            from langchain_core.prompts import PromptTemplate
            from langchain_core.output_parser import StrOutputParser
            from langchain_core.runnables import RunnableLambda

            def when_all_is_lost(inputs):
                return ("Looks like our LLM providers are down. "
                        "Here's a nice 🦜️ emoji for you instead.")

            chain_with_fallback = (
                PromptTemplate.from_template('Tell me a joke about {topic}')
                | model
                | StrOutputParser()
            ).with_fallbacks([RunnableLambda(when_all_is_lost)])
    """

    runnable: Runnable[Input, Output]
    """The Runnable to run first."""
    fallbacks: Sequence[Runnable[Input, Output]]
    """A sequence of fallbacks to try."""
    exceptions_to_handle: tuple[type[BaseException], ...] = (Exception,)
    """The exceptions on which fallbacks should be tried.

    Any exception that is not a subclass of these exceptions will be raised immediately.
    """
    exception_key: Optional[str] = None
    """If string is specified then handled exceptions will be passed to fallbacks as
        part of the input under the specified key. If None, exceptions
        will not be passed to fallbacks. If used, the base Runnable and its fallbacks
        must accept a dictionary as input."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    @property
    @override
    def InputType(self) -> type[Input]:
        return self.runnable.InputType

    @property
    @override
    def OutputType(self) -> type[Output]:
        return self.runnable.OutputType

    @override
    def get_input_schema(
        self, config: Optional[RunnableConfig] = None
    ) -> type[BaseModel]:
        return self.runnable.get_input_schema(config)

    @override
    def get_output_schema(
        self, config: Optional[RunnableConfig] = None
    ) -> type[BaseModel]:
        return self.runnable.get_output_schema(config)

    @property
    @override
    def config_specs(self) -> list[ConfigurableFieldSpec]:
        return get_unique_config_specs(
            spec
            for step in [self.runnable, *self.fallbacks]
            for spec in step.config_specs
        )

    @classmethod
    @override
    def is_lc_serializable(cls) -> bool:
        return True

    @classmethod
    @override
    def get_lc_namespace(cls) -> list[str]:
        """Get the namespace of the langchain object.

        Defaults to ["langchain", "schema", "runnable"].
        """
        return ["langchain", "schema", "runnable"]

    @property
    def runnables(self) -> Iterator[Runnable[Input, Output]]:
        """Iterator over the Runnable and its fallbacks."""
        yield self.runnable
        yield from self.fallbacks

    @override
    def invoke(
        self, input: Input, config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> Output:
        if self.exception_key is not None and not isinstance(input, dict):
            msg = (
                "If 'exception_key' is specified then input must be a dictionary."
                f"However found a type of {type(input)} for input"
            )
            raise ValueError(msg)
        # setup callbacks
        config = ensure_config(config)
        callback_manager = get_callback_manager_for_config(config)
        # start the root run
        run_manager = callback_manager.on_chain_start(
            None,
            input,
            name=config.get("run_name") or self.get_name(),
            run_id=config.pop("run_id", None),
        )
        first_error = None
        last_error = None
        for runnable in self.runnables:
            try:
                if self.exception_key and last_error is not None:
                    input[self.exception_key] = last_error  # type: ignore[index]
                child_config = patch_config(config, callbacks=run_manager.get_child())
                with set_config_context(child_config) as context:
                    output = context.run(
                        runnable.invoke,
                        input,
                        config,
                        **kwargs,
                    )
            except self.exceptions_to_handle as e:
                if first_error is None:
                    first_error = e
                last_error = e
            except BaseException as e:
                run_manager.on_chain_error(e)
                raise
            else:
                run_manager.on_chain_end(output)
                return output
        if first_error is None:
            msg = "No error stored at end of fallbacks."
            raise ValueError(msg)
        run_manager.on_chain_error(first_error)
        raise first_error

    @override
    async def ainvoke(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> Output:
        if self.exception_key is not None and not isinstance(input, dict):
            msg = (
                "If 'exception_key' is specified then input must be a dictionary."
                f"However found a type of {type(input)} for input"
            )
            raise ValueError(msg)
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

        first_error = None
        last_error = None
        for runnable in self.runnables:
            try:
                if self.exception_key and last_error is not None:
                    input[self.exception_key] = last_error  # type: ignore[index]
                child_config = patch_config(config, callbacks=run_manager.get_child())
                with set_config_context(child_config) as context:
                    coro = context.run(runnable.ainvoke, input, config, **kwargs)
                    output = await coro_with_context(coro, context)
            except self.exceptions_to_handle as e:
                if first_error is None:
                    first_error = e
                last_error = e
            except BaseException as e:
                await run_manager.on_chain_error(e)
                raise
            else:
                await run_manager.on_chain_end(output)
                return output
        if first_error is None:
            msg = "No error stored at end of fallbacks."
            raise ValueError(msg)
        await run_manager.on_chain_error(first_error)
        raise first_error

    @override
    def batch(
        self,
        inputs: list[Input],
        config: Optional[Union[RunnableConfig, list[RunnableConfig]]] = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Optional[Any],
    ) -> list[Output]:
        from langchain_core.callbacks.manager import CallbackManager

        if self.exception_key is not None and not all(
            isinstance(input, dict) for input in inputs
        ):
            msg = (
                "If 'exception_key' is specified then inputs must be dictionaries."
                f"However found a type of {type(inputs[0])} for input"
            )
            raise ValueError(msg)

        if not inputs:
            return []

        # setup callbacks
        configs = get_config_list(config, len(inputs))
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
                input if isinstance(input, dict) else {"input": input},
                name=config.get("run_name") or self.get_name(),
                run_id=config.pop("run_id", None),
            )
            for cm, input, config in zip(callback_managers, inputs, configs)
        ]

        to_return: dict[int, Any] = {}
        run_again = dict(enumerate(inputs))
        handled_exceptions: dict[int, BaseException] = {}
        first_to_raise = None
        for runnable in self.runnables:
            outputs = runnable.batch(
                [input for _, input in sorted(run_again.items())],
                [
                    # each step a child run of the corresponding root run
                    patch_config(configs[i], callbacks=run_managers[i].get_child())
                    for i in sorted(run_again)
                ],
                return_exceptions=True,
                **kwargs,
            )
            for (i, input), output in zip(sorted(run_again.copy().items()), outputs):
                if isinstance(output, BaseException) and not isinstance(
                    output, self.exceptions_to_handle
                ):
                    if not return_exceptions:
                        first_to_raise = first_to_raise or output
                    else:
                        handled_exceptions[i] = output
                    run_again.pop(i)
                elif isinstance(output, self.exceptions_to_handle):
                    if self.exception_key:
                        input[self.exception_key] = output  # type: ignore[index]
                    handled_exceptions[i] = output
                else:
                    run_managers[i].on_chain_end(output)
                    to_return[i] = output
                    run_again.pop(i)
                    handled_exceptions.pop(i, None)
            if first_to_raise:
                raise first_to_raise
            if not run_again:
                break

        sorted_handled_exceptions = sorted(handled_exceptions.items())
        for i, error in sorted_handled_exceptions:
            run_managers[i].on_chain_error(error)
        if not return_exceptions and sorted_handled_exceptions:
            raise sorted_handled_exceptions[0][1]
        to_return.update(handled_exceptions)
        return [output for _, output in sorted(to_return.items())]

    @override
    async def abatch(
        self,
        inputs: list[Input],
        config: Optional[Union[RunnableConfig, list[RunnableConfig]]] = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Optional[Any],
    ) -> list[Output]:
        from langchain_core.callbacks.manager import AsyncCallbackManager

        if self.exception_key is not None and not all(
            isinstance(input, dict) for input in inputs
        ):
            msg = (
                "If 'exception_key' is specified then inputs must be dictionaries."
                f"However found a type of {type(inputs[0])} for input"
            )
            raise ValueError(msg)

        if not inputs:
            return []

        # setup callbacks
        configs = get_config_list(config, len(inputs))
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

        to_return = {}
        run_again = dict(enumerate(inputs))
        handled_exceptions: dict[int, BaseException] = {}
        first_to_raise = None
        for runnable in self.runnables:
            outputs = await runnable.abatch(
                [input for _, input in sorted(run_again.items())],
                [
                    # each step a child run of the corresponding root run
                    patch_config(configs[i], callbacks=run_managers[i].get_child())
                    for i in sorted(run_again)
                ],
                return_exceptions=True,
                **kwargs,
            )

            for (i, input), output in zip(sorted(run_again.copy().items()), outputs):
                if isinstance(output, BaseException) and not isinstance(
                    output, self.exceptions_to_handle
                ):
                    if not return_exceptions:
                        first_to_raise = first_to_raise or output
                    else:
                        handled_exceptions[i] = output
                    run_again.pop(i)
                elif isinstance(output, self.exceptions_to_handle):
                    if self.exception_key:
                        input[self.exception_key] = output  # type: ignore[index]
                    handled_exceptions[i] = output
                else:
                    to_return[i] = output
                    await run_managers[i].on_chain_end(output)
                    run_again.pop(i)
                    handled_exceptions.pop(i, None)

            if first_to_raise:
                raise first_to_raise
            if not run_again:
                break

        sorted_handled_exceptions = sorted(handled_exceptions.items())
        await asyncio.gather(
            *(
                run_managers[i].on_chain_error(error)
                for i, error in sorted_handled_exceptions
            )
        )
        if not return_exceptions and sorted_handled_exceptions:
            raise sorted_handled_exceptions[0][1]
        to_return.update(handled_exceptions)
        return [output for _, output in sorted(to_return.items())]  # type: ignore[misc]

    @override
    def stream(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> Iterator[Output]:
        if self.exception_key is not None and not isinstance(input, dict):
            msg = (
                "If 'exception_key' is specified then input must be a dictionary."
                f"However found a type of {type(input)} for input"
            )
            raise ValueError(msg)
        # setup callbacks
        config = ensure_config(config)
        callback_manager = get_callback_manager_for_config(config)
        # start the root run
        run_manager = callback_manager.on_chain_start(
            None,
            input,
            name=config.get("run_name") or self.get_name(),
            run_id=config.pop("run_id", None),
        )
        first_error = None
        last_error = None
        for runnable in self.runnables:
            try:
                if self.exception_key and last_error is not None:
                    input[self.exception_key] = last_error  # type: ignore[index]
                child_config = patch_config(config, callbacks=run_manager.get_child())
                with set_config_context(child_config) as context:
                    stream = context.run(
                        runnable.stream,
                        input,
                        **kwargs,
                    )
                    chunk: Output = context.run(next, stream)
            except self.exceptions_to_handle as e:
                first_error = e if first_error is None else first_error
                last_error = e
            except BaseException as e:
                run_manager.on_chain_error(e)
                raise
            else:
                first_error = None
                break
        if first_error:
            run_manager.on_chain_error(first_error)
            raise first_error

        yield chunk
        output: Optional[Output] = chunk
        try:
            for chunk in stream:
                yield chunk
                try:
                    output = output + chunk  # type: ignore[operator]
                except TypeError:
                    output = None
        except BaseException as e:
            run_manager.on_chain_error(e)
            raise
        run_manager.on_chain_end(output)

    @override
    async def astream(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> AsyncIterator[Output]:
        if self.exception_key is not None and not isinstance(input, dict):
            msg = (
                "If 'exception_key' is specified then input must be a dictionary."
                f"However found a type of {type(input)} for input"
            )
            raise ValueError(msg)
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
        first_error = None
        last_error = None
        for runnable in self.runnables:
            try:
                if self.exception_key and last_error is not None:
                    input[self.exception_key] = last_error  # type: ignore[index]
                child_config = patch_config(config, callbacks=run_manager.get_child())
                with set_config_context(child_config) as context:
                    stream = runnable.astream(
                        input,
                        child_config,
                        **kwargs,
                    )
                    chunk = await coro_with_context(py_anext(stream), context)
            except self.exceptions_to_handle as e:
                first_error = e if first_error is None else first_error
                last_error = e
            except BaseException as e:
                await run_manager.on_chain_error(e)
                raise
            else:
                first_error = None
                break
        if first_error:
            await run_manager.on_chain_error(first_error)
            raise first_error

        yield chunk
        output: Optional[Output] = chunk
        try:
            async for chunk in stream:
                yield chunk
                try:
                    output = output + chunk
                except TypeError:
                    output = None
        except BaseException as e:
            await run_manager.on_chain_error(e)
            raise
        await run_manager.on_chain_end(output)

    def __getattr__(self, name: str) -> Any:
        """Get an attribute from the wrapped Runnable and its fallbacks.

        Returns:
            If the attribute is anything other than a method that outputs a Runnable,
            returns getattr(self.runnable, name). If the attribute is a method that
            does return a new Runnable (e.g. llm.bind_tools([...]) outputs a new
            RunnableBinding) then self.runnable and each of the runnables in
            self.fallbacks is replaced with getattr(x, name).

        Example:
            .. code-block:: python

                from langchain_openai import ChatOpenAI
                from langchain_anthropic import ChatAnthropic

                gpt_4o = ChatOpenAI(model="gpt-4o")
                claude_3_sonnet = ChatAnthropic(model="claude-3-sonnet-20240229")
                llm = gpt_4o.with_fallbacks([claude_3_sonnet])

                llm.model_name
                # -> "gpt-4o"

                # .bind_tools() is called on both ChatOpenAI and ChatAnthropic
                # Equivalent to:
                # gpt_4o.bind_tools([...]).with_fallbacks([claude_3_sonnet.bind_tools([...])])
                llm.bind_tools([...])
                # -> RunnableWithFallbacks(
                    runnable=RunnableBinding(bound=ChatOpenAI(...), kwargs={"tools": [...]}),
                    fallbacks=[RunnableBinding(bound=ChatAnthropic(...), kwargs={"tools": [...]})],
                )

        """  # noqa: E501
        attr = getattr(self.runnable, name)
        if _returns_runnable(attr):

            @wraps(attr)
            def wrapped(*args: Any, **kwargs: Any) -> Any:
                new_runnable = attr(*args, **kwargs)
                new_fallbacks = []
                for fallback in self.fallbacks:
                    fallback_attr = getattr(fallback, name)
                    new_fallbacks.append(fallback_attr(*args, **kwargs))

                return self.__class__(
                    **{
                        **self.model_dump(),
                        "runnable": new_runnable,
                        "fallbacks": new_fallbacks,
                    }
                )

            return wrapped

        return attr


def _returns_runnable(attr: Any) -> bool:
    if not callable(attr):
        return False
    return_type = typing.get_type_hints(attr).get("return")
    return bool(return_type and _is_runnable_type(return_type))


def _is_runnable_type(type_: Any) -> bool:
    if inspect.isclass(type_):
        return issubclass(type_, Runnable)
    origin = getattr(type_, "__origin__", None)
    if inspect.isclass(origin):
        return issubclass(origin, Runnable)
    if origin is typing.Union:
        return all(_is_runnable_type(t) for t in type_.__args__)
    return False
