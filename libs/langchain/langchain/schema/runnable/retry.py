from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type, TypeVar, Union

from tenacity import AsyncRetrying, BaseRetrying, RetryCallState, Retrying

from langchain.schema.runnable.base import Input, Output, RunnableBinding
from langchain.schema.runnable.config import RunnableConfig, patch_config

if TYPE_CHECKING:
    from langchain.callbacks.manager import (
        AsyncCallbackManager as AsyncCallbackManagerT,
        CallbackManager as CallbackManagerT,
    )

    T = TypeVar("T", CallbackManagerT, AsyncCallbackManagerT)
else:
    T = TypeVar("T")


class RunnableRetry(RunnableBinding[Input, Output]):
    """Retry a Runnable if it fails."""

    retry: BaseRetrying

    def _kwargs_retrying(self) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {}
        if self.retry.sleep is not None:
            kwargs["sleep"] = self.retry.sleep
        if self.retry.stop is not None:
            kwargs["stop"] = self.retry.stop
        if self.retry.wait is not None:
            kwargs["wait"] = self.retry.wait
        if self.retry.retry is not None:
            kwargs["retry"] = self.retry.retry
        if self.retry.before is not None:
            kwargs["before"] = self.retry.before
        if self.retry.after is not None:
            kwargs["after"] = self.retry.after
        if self.retry.before_sleep is not None:
            kwargs["before_sleep"] = self.retry.before_sleep
        if self.retry.reraise is not None:
            kwargs["reraise"] = self.retry.reraise
        if self.retry.retry_error_cls is not None:
            kwargs["retry_error_cls"] = self.retry.retry_error_cls
        if self.retry.retry_error_callback is not None:
            kwargs["retry_error_callback"] = self.retry.retry_error_callback
        return kwargs

    def _sync_retrying(self) -> Retrying:
        return Retrying(**self._kwargs_retrying())

    def _async_retrying(self) -> AsyncRetrying:
        return AsyncRetrying(**self._kwargs_retrying())

    def _patch_config(
        self,
        config: Optional[RunnableConfig],
        retry_state: RetryCallState,
        cm_cls: Type[T],
    ) -> RunnableConfig:
        config = config or {}
        return (
            patch_config(
                config,
                callbacks=cm_cls.configure(
                    inheritable_callbacks=config.get("callbacks"),
                    local_tags=["retry:attempt:{}".format(retry_state.attempt_number)],
                ),
            )
            if retry_state.attempt_number > 1
            else config
        )

    def _patch_config_list(
        self,
        config: Optional[Union[RunnableConfig, List[RunnableConfig]]],
        retry_state: RetryCallState,
        cm_cls: Type[T],
    ) -> Union[RunnableConfig, List[RunnableConfig]]:
        if isinstance(config, list):
            return [self._patch_config(c, retry_state, cm_cls) for c in config]

        return self._patch_config(config, retry_state, cm_cls)

    def invoke(
        self, input: Input, config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> Output:
        from langchain.callbacks.manager import CallbackManager

        for attempt in self._sync_retrying():
            with attempt:
                result = super().invoke(
                    input,
                    self._patch_config(config, attempt.retry_state, CallbackManager),
                    **kwargs
                )
            if attempt.retry_state.outcome and not attempt.retry_state.outcome.failed:
                attempt.retry_state.set_result(result)
        return result

    async def ainvoke(
        self, input: Input, config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> Output:
        from langchain.callbacks.manager import AsyncCallbackManager

        async for attempt in self._async_retrying():
            with attempt:
                result = await super().ainvoke(
                    input,
                    self._patch_config(
                        config, attempt.retry_state, AsyncCallbackManager
                    ),
                    **kwargs
                )
            if attempt.retry_state.outcome and not attempt.retry_state.outcome.failed:
                attempt.retry_state.set_result(result)
        return result

    def batch(
        self,
        inputs: List[Input],
        config: Optional[Union[RunnableConfig, List[RunnableConfig]]] = None,
        **kwargs: Any
    ) -> List[Output]:
        from langchain.callbacks.manager import CallbackManager

        for attempt in self._sync_retrying():
            with attempt:
                result = super().batch(
                    inputs,
                    self._patch_config_list(
                        config, attempt.retry_state, CallbackManager
                    ),
                    **kwargs
                )
            if attempt.retry_state.outcome and not attempt.retry_state.outcome.failed:
                attempt.retry_state.set_result(result)
        return result

    async def abatch(
        self,
        inputs: List[Input],
        config: Optional[Union[RunnableConfig, List[RunnableConfig]]] = None,
        **kwargs: Any
    ) -> List[Output]:
        from langchain.callbacks.manager import AsyncCallbackManager

        async for attempt in self._async_retrying():
            with attempt:
                result = await super().abatch(
                    inputs,
                    self._patch_config_list(
                        config, attempt.retry_state, AsyncCallbackManager
                    ),
                    **kwargs
                )
            if attempt.retry_state.outcome and not attempt.retry_state.outcome.failed:
                attempt.retry_state.set_result(result)
        return result

    # stream() and transform() are not retried because retrying a stream
    # is not very intuitive.
