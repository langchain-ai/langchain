import asyncio
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Awaitable,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
    cast,
)

from langchain_core.load.dump import dumpd
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables.base import Runnable, RunnableSerializable
from langchain_core.runnables.config import (
    RunnableConfig,
    ensure_config,
    get_async_callback_manager_for_config,
    get_callback_manager_for_config,
    get_config_list,
    patch_config,
)
from langchain_core.runnables.utils import (
    ConfigurableFieldSpec,
    Input,
    Output,
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

            model = ChatAnthropic().with_fallbacks([ChatOpenAI()])
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
    """The runnable to run first."""
    fallbacks: Sequence[Runnable[Input, Output]]
    """A sequence of fallbacks to try."""
    exceptions_to_handle: Tuple[Type[BaseException], ...] = (Exception,)
    """The exceptions on which fallbacks should be tried.
    
    Any exception that is not a subclass of these exceptions will be raised immediately.
    """
    exception_key: Optional[str] = None
    """If string is specified then handled exceptions will be passed to fallbacks as 
        part of the input under the specified key. If None, exceptions
        will not be passed to fallbacks. If used, the base runnable and its fallbacks 
        must accept a dictionary as input."""

    class Config:
        arbitrary_types_allowed = True

    @property
    def InputType(self) -> Type[Input]:
        return self.runnable.InputType

    @property
    def OutputType(self) -> Type[Output]:
        return self.runnable.OutputType

    def get_input_schema(
        self, config: Optional[RunnableConfig] = None
    ) -> Type[BaseModel]:
        return self.runnable.get_input_schema(config)

    def get_output_schema(
        self, config: Optional[RunnableConfig] = None
    ) -> Type[BaseModel]:
        return self.runnable.get_output_schema(config)

    @property
    def config_specs(self) -> List[ConfigurableFieldSpec]:
        return get_unique_config_specs(
            spec
            for step in [self.runnable, *self.fallbacks]
            for spec in step.config_specs
        )

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return True

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "schema", "runnable"]

    @property
    def runnables(self) -> Iterator[Runnable[Input, Output]]:
        yield self.runnable
        yield from self.fallbacks

    def invoke(
        self, input: Input, config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> Output:
        if self.exception_key is not None and not isinstance(input, dict):
            raise ValueError(
                "If 'exception_key' is specified then input must be a dictionary."
                f"However found a type of {type(input)} for input"
            )
        # setup callbacks
        config = ensure_config(config)
        callback_manager = get_callback_manager_for_config(config)
        # start the root run
        run_manager = callback_manager.on_chain_start(
            dumpd(self),
            input,
            name=config.get("run_name"),
            run_id=config.pop("run_id", None),
        )
        first_error = None
        last_error = None
        for runnable in self.runnables:
            try:
                if self.exception_key and last_error is not None:
                    input[self.exception_key] = last_error
                output = runnable.invoke(
                    input,
                    patch_config(config, callbacks=run_manager.get_child()),
                    **kwargs,
                )
            except self.exceptions_to_handle as e:
                if first_error is None:
                    first_error = e
                last_error = e
            except BaseException as e:
                run_manager.on_chain_error(e)
                raise e
            else:
                run_manager.on_chain_end(output)
                return output
        if first_error is None:
            raise ValueError("No error stored at end of fallbacks.")
        run_manager.on_chain_error(first_error)
        raise first_error

    async def ainvoke(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> Output:
        if self.exception_key is not None and not isinstance(input, dict):
            raise ValueError(
                "If 'exception_key' is specified then input must be a dictionary."
                f"However found a type of {type(input)} for input"
            )
        # setup callbacks
        config = ensure_config(config)
        callback_manager = get_async_callback_manager_for_config(config)
        # start the root run
        run_manager = await callback_manager.on_chain_start(
            dumpd(self),
            input,
            name=config.get("run_name"),
            run_id=config.pop("run_id", None),
        )

        first_error = None
        last_error = None
        for runnable in self.runnables:
            try:
                if self.exception_key and last_error is not None:
                    input[self.exception_key] = last_error
                output = await runnable.ainvoke(
                    input,
                    patch_config(config, callbacks=run_manager.get_child()),
                    **kwargs,
                )
            except self.exceptions_to_handle as e:
                if first_error is None:
                    first_error = e
                last_error = e
            except BaseException as e:
                await run_manager.on_chain_error(e)
                raise e
            else:
                await run_manager.on_chain_end(output)
                return output
        if first_error is None:
            raise ValueError("No error stored at end of fallbacks.")
        await run_manager.on_chain_error(first_error)
        raise first_error

    def batch(
        self,
        inputs: List[Input],
        config: Optional[Union[RunnableConfig, List[RunnableConfig]]] = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Optional[Any],
    ) -> List[Output]:
        from langchain_core.callbacks.manager import CallbackManager

        if self.exception_key is not None and not all(
            isinstance(input, dict) for input in inputs
        ):
            raise ValueError(
                "If 'exception_key' is specified then inputs must be dictionaries."
                f"However found a type of {type(inputs[0])} for input"
            )

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
                dumpd(self),
                input if isinstance(input, dict) else {"input": input},
                name=config.get("run_name"),
                run_id=config.pop("run_id", None),
            )
            for cm, input, config in zip(callback_managers, inputs, configs)
        ]

        to_return: Dict[int, Any] = {}
        run_again = {i: input for i, input in enumerate(inputs)}
        handled_exceptions: Dict[int, BaseException] = {}
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
                        handled_exceptions[i] = cast(BaseException, output)
                    run_again.pop(i)
                elif isinstance(output, self.exceptions_to_handle):
                    if self.exception_key:
                        input[self.exception_key] = output  # type: ignore
                    handled_exceptions[i] = cast(BaseException, output)
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

    async def abatch(
        self,
        inputs: List[Input],
        config: Optional[Union[RunnableConfig, List[RunnableConfig]]] = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Optional[Any],
    ) -> List[Output]:
        from langchain_core.callbacks.manager import AsyncCallbackManager

        if self.exception_key is not None and not all(
            isinstance(input, dict) for input in inputs
        ):
            raise ValueError(
                "If 'exception_key' is specified then inputs must be dictionaries."
                f"However found a type of {type(inputs[0])} for input"
            )

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
        run_managers: List[AsyncCallbackManagerForChainRun] = await asyncio.gather(
            *(
                cm.on_chain_start(
                    dumpd(self),
                    input,
                    name=config.get("run_name"),
                    run_id=config.pop("run_id", None),
                )
                for cm, input, config in zip(callback_managers, inputs, configs)
            )
        )

        to_return = {}
        run_again = {i: input for i, input in enumerate(inputs)}
        handled_exceptions: Dict[int, BaseException] = {}
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
                        handled_exceptions[i] = cast(BaseException, output)
                    run_again.pop(i)
                elif isinstance(output, self.exceptions_to_handle):
                    if self.exception_key:
                        input[self.exception_key] = output  # type: ignore
                    handled_exceptions[i] = cast(BaseException, output)
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
        return [output for _, output in sorted(to_return.items())]  # type: ignore

    def stream(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> Iterator[Output]:
        """"""
        if self.exception_key is not None and not isinstance(input, dict):
            raise ValueError(
                "If 'exception_key' is specified then input must be a dictionary."
                f"However found a type of {type(input)} for input"
            )
        # setup callbacks
        config = ensure_config(config)
        callback_manager = get_callback_manager_for_config(config)
        # start the root run
        run_manager = callback_manager.on_chain_start(
            dumpd(self),
            input,
            name=config.get("run_name"),
            run_id=config.pop("run_id", None),
        )
        first_error = None
        last_error = None
        for runnable in self.runnables:
            try:
                if self.exception_key and last_error is not None:
                    input[self.exception_key] = last_error
                stream = runnable.stream(
                    input,
                    patch_config(config, callbacks=run_manager.get_child()),
                    **kwargs,
                )
                chunk = next(stream)
            except self.exceptions_to_handle as e:
                first_error = e if first_error is None else first_error
                last_error = e
            except BaseException as e:
                run_manager.on_chain_error(e)
                raise e
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
                    output = output + chunk  # type: ignore
                except TypeError:
                    output = None
        except BaseException as e:
            run_manager.on_chain_error(e)
            raise e
        run_manager.on_chain_end(output)

    async def astream(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> AsyncIterator[Output]:
        if self.exception_key is not None and not isinstance(input, dict):
            raise ValueError(
                "If 'exception_key' is specified then input must be a dictionary."
                f"However found a type of {type(input)} for input"
            )
        # setup callbacks
        config = ensure_config(config)
        callback_manager = get_async_callback_manager_for_config(config)
        # start the root run
        run_manager = await callback_manager.on_chain_start(
            dumpd(self),
            input,
            name=config.get("run_name"),
            run_id=config.pop("run_id", None),
        )
        first_error = None
        last_error = None
        for runnable in self.runnables:
            try:
                if self.exception_key and last_error is not None:
                    input[self.exception_key] = last_error
                stream = runnable.astream(
                    input,
                    patch_config(config, callbacks=run_manager.get_child()),
                    **kwargs,
                )
                chunk = await cast(Awaitable[Output], py_anext(stream))
            except self.exceptions_to_handle as e:
                first_error = e if first_error is None else first_error
                last_error = e
            except BaseException as e:
                await run_manager.on_chain_error(e)
                raise e
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
                    output = output + chunk  # type: ignore
                except TypeError:
                    output = None
        except BaseException as e:
            await run_manager.on_chain_error(e)
            raise e
        await run_manager.on_chain_end(output)
