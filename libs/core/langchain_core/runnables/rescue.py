"""Runnable that recovers from selected exceptions."""

from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from typing import Any

from pydantic import ConfigDict
from typing_extensions import override

from langchain_core.runnables.base import Runnable, RunnableSerializable
from langchain_core.runnables.config import (
    RunnableConfig,
    ensure_config,
    get_async_callback_manager_for_config,
    get_callback_manager_for_config,
    patch_config,
    set_config_context,
)
from langchain_core.runnables.utils import Input, Output, coro_with_context
from langchain_core.utils.pydantic import TypeBaseModel


class RunnableRescue(RunnableSerializable[Input, Output]):
    """Runnable that recovers from selected exceptions raised by another Runnable."""

    runnable: Runnable[Input, Output]

    handler: Callable[[Exception, Input], Output] | None = None
    async_handler: Callable[[Exception, Input], Awaitable[Output]] | None = None
    fallback: Any = None
    has_fallback: bool = False
    exceptions: tuple[type[BaseException], ...] = (Exception,)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    @override
    def InputType(self) -> type[Input]:
        return self.runnable.InputType

    @property
    @override
    def OutputType(self) -> type[Output]:
        return self.runnable.OutputType

    @override
    def get_input_schema(self, config: RunnableConfig | None = None) -> TypeBaseModel:
        return self.runnable.get_input_schema(config)

    @override
    def get_output_schema(self, config: RunnableConfig | None = None) -> TypeBaseModel:
        return self.runnable.get_output_schema(config)

    @classmethod
    @override
    def is_lc_serializable(cls) -> bool:
        return True

    @classmethod
    @override
    def get_lc_namespace(cls) -> list[str]:
        return ["langchain", "schema", "runnable"]

    def _recover(self, error: Exception, input: Input) -> Output:
        if self.handler is not None:
            return self.handler(error, input)
        if self.has_fallback:
            return self.fallback
        raise error

    async def _arecover(self, error: Exception, input: Input) -> Output:
        if self.async_handler is not None:
            return await self.async_handler(error, input)
        return self._recover(error, input)

    @override
    def invoke(
        self,
        input: Input,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Output:
        config = ensure_config(config)
        callback_manager = get_callback_manager_for_config(config)
        run_manager = callback_manager.on_chain_start(
            None,
            input,
            name=config.get("run_name") or self.get_name(),
            run_id=config.pop("run_id", None),
        )

        try:
            child_config = patch_config(config, callbacks=run_manager.get_child())
            with set_config_context(child_config) as context:
                output = context.run(
                    self.runnable.invoke,
                    input,
                    child_config,
                    **kwargs,
                )
        except self.exceptions as error:
            try:
                output = self._recover(error, input)
            except BaseException as recovery_error:
                run_manager.on_chain_error(recovery_error)
                raise
            run_manager.on_chain_end(output)
            return output
        except BaseException as error:
            run_manager.on_chain_error(error)
            raise

        run_manager.on_chain_end(output)
        return output

    @override
    async def ainvoke(
        self,
        input: Input,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Output:
        config = ensure_config(config)
        callback_manager = get_async_callback_manager_for_config(config)
        run_manager = await callback_manager.on_chain_start(
            None,
            input,
            name=config.get("run_name") or self.get_name(),
            run_id=config.pop("run_id", None),
        )

        try:
            child_config = patch_config(config, callbacks=run_manager.get_child())
            with set_config_context(child_config) as context:
                coro = context.run(self.runnable.ainvoke, input, child_config, **kwargs)
                output = await coro_with_context(coro, context)
        except self.exceptions as error:
            try:
                output = await self._arecover(error, input)
            except BaseException as recovery_error:
                await run_manager.on_chain_error(recovery_error)
                raise
            await run_manager.on_chain_end(output)
            return output
        except BaseException as error:
            await run_manager.on_chain_error(error)
            raise

        await run_manager.on_chain_end(output)
        return output

    @override
    def stream(
        self,
        input: Input,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> Iterator[Output]:
        config = ensure_config(config)
        callback_manager = get_callback_manager_for_config(config)
        run_manager = callback_manager.on_chain_start(
            None,
            input,
            name=config.get("run_name") or self.get_name(),
            run_id=config.pop("run_id", None),
        )

        child_config = patch_config(config, callbacks=run_manager.get_child())
        try:
            with set_config_context(child_config) as context:
                stream = context.run(
                    self.runnable.stream,
                    input,
                    child_config,
                    **kwargs,
                )
                iterator = iter(stream)
                try:
                    first_chunk = next(iterator)
                except StopIteration:
                    run_manager.on_chain_end(None)
                    return
        except self.exceptions as error:
            try:
                output = self._recover(error, input)
            except BaseException as recovery_error:
                run_manager.on_chain_error(recovery_error)
                raise
            run_manager.on_chain_end(output)
            yield output
            return
        except BaseException as error:
            run_manager.on_chain_error(error)
            raise

        output = first_chunk
        try:
            yield first_chunk
            for chunk in iterator:
                yield chunk
                try:
                    output = output + chunk
                except TypeError:
                    output = None
        except BaseException as error:
            run_manager.on_chain_error(error)
            raise
        else:
            run_manager.on_chain_end(output)

    @override
    async def astream(
        self,
        input: Input,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[Output]:
        config = ensure_config(config)
        callback_manager = get_async_callback_manager_for_config(config)
        run_manager = await callback_manager.on_chain_start(
            None,
            input,
            name=config.get("run_name") or self.get_name(),
            run_id=config.pop("run_id", None),
        )

        child_config = patch_config(config, callbacks=run_manager.get_child())
        try:
            with set_config_context(child_config) as context:
                stream = context.run(
                    self.runnable.astream,
                    input,
                    child_config,
                    **kwargs,
                )
                iterator = stream.__aiter__()
                try:
                    first_chunk = await iterator.__anext__()
                except StopAsyncIteration:
                    await run_manager.on_chain_end(None)
                    return
        except self.exceptions as error:
            try:
                output = await self._arecover(error, input)
            except BaseException as recovery_error:
                await run_manager.on_chain_error(recovery_error)
                raise
            await run_manager.on_chain_end(output)
            yield output
            return
        except BaseException as error:
            await run_manager.on_chain_error(error)
            raise

        output = first_chunk
        try:
            yield first_chunk
            async for chunk in iterator:
                yield chunk
                try:
                    output = output + chunk
                except TypeError:
                    output = None
        except BaseException as error:
            await run_manager.on_chain_error(error)
            raise
        else:
            await run_manager.on_chain_end(output)
