import asyncio
from typing import (
    TYPE_CHECKING,
    Any,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

from langchain.load.dump import dumpd
from langchain.pydantic_v1 import BaseModel
from langchain.schema.runnable.base import Runnable, RunnableSerializable
from langchain.schema.runnable.config import (
    RunnableConfig,
    ensure_config,
    get_async_callback_manager_for_config,
    get_callback_manager_for_config,
    get_config_list,
    patch_config,
)
from langchain.schema.runnable.utils import (
    ConfigurableFieldSpec,
    Input,
    Output,
    get_unique_config_specs,
)

if TYPE_CHECKING:
    from langchain.callbacks.manager import AsyncCallbackManagerForChainRun


class RunnableWithFallbacks(RunnableSerializable[Input, Output]):
    """A Runnable that can fallback to other Runnables if it fails.

    External APIs (e.g., APIs for a language model) may at times experience
    degraded performance or even downtime.

    In these cases, it can be useful to have a fallback runnable that can be
    used in place of the original runnable (e.g., fallback to another LLM provider).

    Fallbacks can be defined at the level of a single runnable, or at the level
    of a chain of runnables. Fallbacks are tried in order until one succeeds or
    all fail.

    While you can instantiate a ``RunnableWithFallbacks`` directly, it is usually
    more convenient to use the ``with_fallbacks`` method on a runnable.

    Example:

        .. code-block:: python

            from langchain.chat_models.openai import ChatOpenAI
            from langchain.chat_models.anthropic import ChatAnthropic

            model = ChatAnthropic().with_fallbacks([ChatOpenAI()])
            # Will usually use ChatAnthropic, but fallback to ChatOpenAI
            # if ChatAnthropic fails.
            model.invoke('hello')

            # And you can also use fallbacks at the level of a chain.
            # Here if both LLM providers fail, we'll fallback to a good hardcoded
            # response.

            from langchain.prompts import PromptTemplate
            from langchain.schema.output_parser import StrOutputParser
            from langchain.schema.runnable import RunnableLambda

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
        return cls.__module__.split(".")[:-1]

    @property
    def runnables(self) -> Iterator[Runnable[Input, Output]]:
        yield self.runnable
        yield from self.fallbacks

    def invoke(
        self, input: Input, config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> Output:
        # setup callbacks
        config = ensure_config(config)
        callback_manager = get_callback_manager_for_config(config)
        # start the root run
        run_manager = callback_manager.on_chain_start(
            dumpd(self), input, name=config.get("run_name")
        )
        first_error = None
        for runnable in self.runnables:
            try:
                output = runnable.invoke(
                    input,
                    patch_config(config, callbacks=run_manager.get_child()),
                    **kwargs,
                )
            except self.exceptions_to_handle as e:
                if first_error is None:
                    first_error = e
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
        # setup callbacks
        config = ensure_config(config)
        callback_manager = get_async_callback_manager_for_config(config)
        # start the root run
        run_manager = await callback_manager.on_chain_start(
            dumpd(self), input, name=config.get("run_name")
        )

        first_error = None
        for runnable in self.runnables:
            try:
                output = await runnable.ainvoke(
                    input,
                    patch_config(config, callbacks=run_manager.get_child()),
                    **kwargs,
                )
            except self.exceptions_to_handle as e:
                if first_error is None:
                    first_error = e
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
        from langchain.callbacks.manager import CallbackManager

        if return_exceptions:
            raise NotImplementedError()

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
            )
            for cm, input, config in zip(callback_managers, inputs, configs)
        ]

        first_error = None
        for runnable in self.runnables:
            try:
                outputs = runnable.batch(
                    inputs,
                    [
                        # each step a child run of the corresponding root run
                        patch_config(config, callbacks=rm.get_child())
                        for rm, config in zip(run_managers, configs)
                    ],
                    return_exceptions=return_exceptions,
                    **kwargs,
                )
            except self.exceptions_to_handle as e:
                if first_error is None:
                    first_error = e
            except BaseException as e:
                for rm in run_managers:
                    rm.on_chain_error(e)
                raise e
            else:
                for rm, output in zip(run_managers, outputs):
                    rm.on_chain_end(output)
                return outputs
        if first_error is None:
            raise ValueError("No error stored at end of fallbacks.")
        for rm in run_managers:
            rm.on_chain_error(first_error)
        raise first_error

    async def abatch(
        self,
        inputs: List[Input],
        config: Optional[Union[RunnableConfig, List[RunnableConfig]]] = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Optional[Any],
    ) -> List[Output]:
        from langchain.callbacks.manager import AsyncCallbackManager

        if return_exceptions:
            raise NotImplementedError()

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
                )
                for cm, input, config in zip(callback_managers, inputs, configs)
            )
        )

        first_error = None
        for runnable in self.runnables:
            try:
                outputs = await runnable.abatch(
                    inputs,
                    [
                        # each step a child run of the corresponding root run
                        patch_config(config, callbacks=rm.get_child())
                        for rm, config in zip(run_managers, configs)
                    ],
                    return_exceptions=return_exceptions,
                    **kwargs,
                )
            except self.exceptions_to_handle as e:
                if first_error is None:
                    first_error = e
            except BaseException as e:
                await asyncio.gather(*(rm.on_chain_error(e) for rm in run_managers))
            else:
                await asyncio.gather(
                    *(
                        rm.on_chain_end(output)
                        for rm, output in zip(run_managers, outputs)
                    )
                )
                return outputs
        if first_error is None:
            raise ValueError("No error stored at end of fallbacks.")
        await asyncio.gather(*(rm.on_chain_error(first_error) for rm in run_managers))
        raise first_error
