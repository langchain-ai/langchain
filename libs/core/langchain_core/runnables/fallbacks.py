"""`Runnable` that can fallback to other `Runnable` objects if it fails."""

import asyncio
import contextlib
import inspect
import typing
from collections.abc import AsyncIterator, Iterator, Sequence
from dataclasses import dataclass
from functools import wraps
from typing import TYPE_CHECKING, Any, cast

from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import override

from langchain_core.callbacks.manager import AsyncCallbackManager, CallbackManager
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

if TYPE_CHECKING:
    from langchain_core.callbacks.manager import AsyncCallbackManagerForChainRun


@dataclass
class FallbackLatch:
    """Mutable shared state used by [`RunnableWithFallbacks`][langchain_core.runnables.fallbacks.RunnableWithFallbacks] for circuit-breaking.

    The default fallback strategy retries the primary on every call.
    For failures that are unlikely to recover within the lifetime of the
    wrapper (a wrong API key, a sustained provider outage), this means
    every call pays a failing round-trip to the primary before falling
    back to the fallback. A `FallbackLatch` flips once on the first
    handled exception from the primary and stays flipped, so subsequent
    calls on the wrapper — and on any sibling wrapper sharing the same
    latch instance — skip the primary entirely and start at the first
    fallback.

    Pass the same `latch` instance to multiple `with_fallbacks(...)`
    calls (e.g., a bare model and the same model after `bind_tools(...)`)
    when you want them to act as one circuit. Reset with `reset()`.

    Example:
        ```python
        from langchain_core.runnables.fallbacks import FallbackLatch

        latch = FallbackLatch()
        chat = primary.with_fallbacks([fallback], latch=latch)

        chat.invoke("hi")  # primary fails -> latch trips, fallback returns
        chat.invoke("again")  # primary skipped entirely, straight to fallback

        latch.reset()
        chat.invoke("retry")  # primary attempted again
        ```

    !!! version-added "Added in `langchain-core` 1.2.0"
    """  # noqa: E501

    latched: bool = False
    """`True` once a handled exception has tripped the latch."""

    def reset(self) -> None:
        """Reset the latch so the primary is attempted on the next call."""
        self.latched = False


class RunnableWithFallbacks(RunnableSerializable[Input, Output]):
    """`Runnable` that can fallback to other `Runnable` objects if it fails.

    External APIs (e.g., APIs for a language model) may at times experience
    degraded performance or even downtime.

    In these cases, it can be useful to have a fallback `Runnable` that can be
    used in place of the original `Runnable` (e.g., fallback to another LLM provider).

    Fallbacks can be defined at the level of a single `Runnable`, or at the level
    of a chain of `Runnable`s. Fallbacks are tried in order until one succeeds or
    all fail.

    While you can instantiate a `RunnableWithFallbacks` directly, it is usually
    more convenient to use the `with_fallbacks` method on a `Runnable`.

    Example:
        ```python
        from langchain_core.chat_models.openai import ChatOpenAI
        from langchain_core.chat_models.anthropic import ChatAnthropic

        model = ChatAnthropic(model="claude-sonnet-4-6").with_fallbacks(
            [ChatOpenAI(model="gpt-5.4-mini")]
        )
        # Will usually use ChatAnthropic, but fallback to ChatOpenAI
        # if ChatAnthropic fails.
        model.invoke("hello")

        # And you can also use fallbacks at the level of a chain.
        # Here if both LLM providers fail, we'll fallback to a good hardcoded
        # response.

        from langchain_core.prompts import PromptTemplate
        from langchain_core.output_parser import StrOutputParser
        from langchain_core.runnables import RunnableLambda


        def when_all_is_lost(inputs):
            return (
                "Looks like our LLM providers are down. "
                "Here's a nice 🦜️ emoji for you instead."
            )


        chain_with_fallback = (
            PromptTemplate.from_template("Tell me a joke about {topic}")
            | model
            | StrOutputParser()
        ).with_fallbacks([RunnableLambda(when_all_is_lost)])
        ```
    """

    runnable: Runnable[Input, Output]
    """The `Runnable` to run first."""
    fallbacks: Sequence[Runnable[Input, Output]]
    """A sequence of fallbacks to try."""
    exceptions_to_handle: tuple[type[BaseException], ...] = (Exception,)
    """The exceptions on which fallbacks should be tried.

    Any exception that is not a subclass of these exceptions will be raised immediately.
    """
    exception_key: str | None = None
    """If `string` is specified then handled exceptions will be passed to fallbacks as
    part of the input under the specified key.

    If `None`, exceptions will not be passed to fallbacks.

    If used, the base `Runnable` and its fallbacks must accept a dictionary as input.
    """
    latch: FallbackLatch | None = Field(default=None, exclude=True)
    """Optional shared circuit-breaker that, once tripped, skips the primary.

    On the first handled exception from `runnable`, `latch.latched` flips
    to `True` and every subsequent call on this wrapper goes straight to
    `fallbacks`. Pass the same `FallbackLatch` instance to multiple
    `with_fallbacks(...)` calls (e.g., the bare model and the same model
    after `bind_tools(...)`) to make them act as one circuit. Reset with
    `latch.reset()` to re-enable primary attempts.

    !!! version-added "Added in `langchain-core` 1.2.0"
    """

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
    def get_input_schema(self, config: RunnableConfig | None = None) -> type[BaseModel]:
        return self.runnable.get_input_schema(config)

    @override
    def get_output_schema(
        self, config: RunnableConfig | None = None
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
        """Return `True` as this class is serializable."""
        return True

    @classmethod
    @override
    def get_lc_namespace(cls) -> list[str]:
        """Get the namespace of the LangChain object.

        Returns:
            `["langchain", "schema", "runnable"]`
        """
        return ["langchain", "schema", "runnable"]

    @property
    def runnables(self) -> Iterator[Runnable[Input, Output]]:
        """Iterator over the `Runnable` and its fallbacks.

        Yields:
            The `Runnable` then its fallbacks.

        !!! note
            Does not consult `latch`. Use `_run_targets` from the execution
            paths so a tripped latch can short-circuit past the primary;
            this property is for introspection / serialization.
        """
        yield self.runnable
        yield from self.fallbacks

    @property
    def _run_targets(self) -> Iterator[tuple[bool, Runnable[Input, Output]]]:
        """Yield (is_primary, runnable) pairs for execution paths.

        Skips the primary when `latch` is tripped so subsequent calls go
        straight to the fallbacks. The `is_primary` flag lets the caller
        trip the latch on the primary's first handled failure without
        re-scanning the list.
        """
        if self.latch is None or not self.latch.latched:
            yield True, self.runnable
        for fb in self.fallbacks:
            yield False, fb

    def _trip_latch_if_primary(self, is_primary: bool) -> None:  # noqa: FBT001
        """Flip `latch.latched` when the primary raises a handled exception."""
        if is_primary and self.latch is not None:
            self.latch.latched = True

    @override
    def invoke(
        self, input: Input, config: RunnableConfig | None = None, **kwargs: Any
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
        for is_primary, runnable in self._run_targets:
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
                self._trip_latch_if_primary(is_primary)
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
        config: RunnableConfig | None = None,
        **kwargs: Any | None,
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
        for is_primary, runnable in self._run_targets:
            try:
                if self.exception_key and last_error is not None:
                    input[self.exception_key] = last_error  # type: ignore[index]
                child_config = patch_config(config, callbacks=run_manager.get_child())
                with set_config_context(child_config) as context:
                    coro = context.run(runnable.ainvoke, input, config, **kwargs)
                    output = await coro_with_context(coro, context)
            except self.exceptions_to_handle as e:
                self._trip_latch_if_primary(is_primary)
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
        config: RunnableConfig | list[RunnableConfig] | None = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Any | None,
    ) -> list[Output]:
        if self.exception_key is not None and not all(
            isinstance(input_, dict) for input_ in inputs
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
                input_ if isinstance(input_, dict) else {"input": input_},
                name=config.get("run_name") or self.get_name(),
                run_id=config.pop("run_id", None),
            )
            for cm, input_, config in zip(
                callback_managers, inputs, configs, strict=False
            )
        ]

        to_return: dict[int, Any] = {}
        run_again = dict(enumerate(inputs))
        handled_exceptions: dict[int, BaseException] = {}
        first_to_raise = None
        for is_primary, runnable in self._run_targets:
            outputs = runnable.batch(
                [input_ for _, input_ in sorted(run_again.items())],
                [
                    # each step a child run of the corresponding root run
                    patch_config(configs[i], callbacks=run_managers[i].get_child())
                    for i in sorted(run_again)
                ],
                return_exceptions=True,
                **kwargs,
            )
            primary_had_handled_exception = False
            for (i, input_), output in zip(
                sorted(run_again.copy().items()), outputs, strict=False
            ):
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
                        input_[self.exception_key] = output  # type: ignore[index]
                    handled_exceptions[i] = output
                    primary_had_handled_exception = True
                else:
                    run_managers[i].on_chain_end(output)
                    to_return[i] = output
                    run_again.pop(i)
                    handled_exceptions.pop(i, None)
            # Any handled exception on the primary trips the latch — even
            # one bad input is enough to skip the primary for subsequent
            # batches sharing the same latch.
            if primary_had_handled_exception:
                self._trip_latch_if_primary(is_primary)
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
        config: RunnableConfig | list[RunnableConfig] | None = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Any | None,
    ) -> list[Output]:
        if self.exception_key is not None and not all(
            isinstance(input_, dict) for input_ in inputs
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
                    input_,
                    name=config.get("run_name") or self.get_name(),
                    run_id=config.pop("run_id", None),
                )
                for cm, input_, config in zip(
                    callback_managers, inputs, configs, strict=False
                )
            )
        )

        to_return: dict[int, Output | BaseException] = {}
        run_again = dict(enumerate(inputs))
        handled_exceptions: dict[int, BaseException] = {}
        first_to_raise = None
        for is_primary, runnable in self._run_targets:
            outputs = await runnable.abatch(
                [input_ for _, input_ in sorted(run_again.items())],
                [
                    # each step a child run of the corresponding root run
                    patch_config(configs[i], callbacks=run_managers[i].get_child())
                    for i in sorted(run_again)
                ],
                return_exceptions=True,
                **kwargs,
            )

            primary_had_handled_exception = False
            for (i, input_), output in zip(
                sorted(run_again.copy().items()), outputs, strict=False
            ):
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
                        input_[self.exception_key] = output  # type: ignore[index]
                    handled_exceptions[i] = output
                    primary_had_handled_exception = True
                else:
                    to_return[i] = output
                    await run_managers[i].on_chain_end(output)
                    run_again.pop(i)
                    handled_exceptions.pop(i, None)

            if primary_had_handled_exception:
                self._trip_latch_if_primary(is_primary)
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
        return [cast("Output", output) for _, output in sorted(to_return.items())]

    @override
    def stream(
        self,
        input: Input,
        config: RunnableConfig | None = None,
        **kwargs: Any | None,
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
        for is_primary, runnable in self._run_targets:
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
                self._trip_latch_if_primary(is_primary)
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
        output: Output | None = chunk
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
        config: RunnableConfig | None = None,
        **kwargs: Any | None,
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
        for is_primary, runnable in self._run_targets:
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
                    chunk = await coro_with_context(anext(stream), context)
            except self.exceptions_to_handle as e:
                self._trip_latch_if_primary(is_primary)
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
        output: Output | None = chunk
        try:
            async for chunk in stream:
                yield chunk
                try:
                    output = output + chunk  # type: ignore[operator]
                except TypeError:
                    output = None
        except BaseException as e:
            await run_manager.on_chain_error(e)
            raise
        await run_manager.on_chain_end(output)

    def __getattr__(self, name: str) -> Any:
        """Get an attribute from the wrapped `Runnable` and its fallbacks.

        Returns:
            If the attribute is anything other than a method that outputs a `Runnable`,
            returns `getattr(self.runnable, name)`. If the attribute is a method that
            does return a new `Runnable` (e.g. `model.bind_tools([...])` outputs a new
            `RunnableBinding`) then `self.runnable` and each of the runnables in
            `self.fallbacks` is replaced with `getattr(x, name)`.

        Example:
            ```python
            from langchain_openai import ChatOpenAI
            from langchain_anthropic import ChatAnthropic

            gpt_55 = ChatOpenAI(model="openai:gpt-5.5")
            claude_3_sonnet = ChatAnthropic(model="claude-sonnet-4-5-20250929")
            model = gpt_55.with_fallbacks([claude_3_sonnet])

            model.model_name
            # -> "gpt-5.5"

            # .bind_tools() is called on both ChatOpenAI and ChatAnthropic
            # Equivalent to:
            # gpt_55.bind_tools([...]).with_fallbacks([claude_3_sonnet.bind_tools([...])])
            model.bind_tools([...])
            # -> RunnableWithFallbacks(
                runnable=RunnableBinding(bound=ChatOpenAI(...), kwargs={"tools": [...]}),
                fallbacks=[RunnableBinding(bound=ChatAnthropic(...), kwargs={"tools": [...]})],
            )
            ```
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

                # `latch` is `exclude=True` so it's missing from `model_dump`;
                # forward the same instance explicitly so derived wrappers
                # (e.g. `model.bind_tools([...])`) share the original
                # circuit-breaker rather than each getting an orphan latch
                # that nothing can trip.
                return self.__class__(
                    **{
                        **self.model_dump(),
                        "runnable": new_runnable,
                        "fallbacks": new_fallbacks,
                        "latch": self.latch,
                    }
                )

            return wrapped

        return attr

    # -- Resource lifecycle ---------------------------------------------------

    def close(self) -> None:
        """Release sync resources held by the wrapped `Runnable` + `fallbacks`.

        Walks `self.runnables` and calls `close()` on each one that has it.
        Per-runnable failures are suppressed so one bad close doesn't
        prevent the others from running. Idempotent.

        Useful when the wrapped runnables own HTTP clients / connection
        pools that should be released when the wrapper is no longer in
        use (e.g., per-request chat models in a long-lived worker).
        """
        for r in self.runnables:
            close_fn = getattr(r, "close", None)
            if callable(close_fn):
                with contextlib.suppress(Exception):
                    close_fn()

    async def aclose(self) -> None:
        """Async sibling of `close()`. Walks runnables, awaits async closes."""
        for r in self.runnables:
            aclose_fn = getattr(r, "aclose", None)
            if aclose_fn is not None and inspect.iscoroutinefunction(aclose_fn):
                with contextlib.suppress(Exception):
                    await aclose_fn()
                continue
            close_fn = getattr(r, "close", None)
            if callable(close_fn):
                with contextlib.suppress(Exception):
                    close_fn()


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
