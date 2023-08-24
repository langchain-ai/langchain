from typing import Any, List, Optional, Union
from langchain.schema.runnable.base import Input, Output, Runnable, RunnableBinding
from langchain.schema.runnable.config import RunnableConfig, patch_config
from tenacity import AsyncRetrying, BaseRetrying, RetryCallState, Retrying


class RunnableRetry(RunnableBinding[Input, Output]):
    """Retry a Runnable if it fails."""

    retry: BaseRetrying

    def _sync_retrying(self) -> Retrying:
        return Retrying(
            sleep=self.retry.sleep,
            stop=self.retry.stop,
            wait=self.retry.wait,
            retry=self.retry.retry,
            before=self.retry.before,
            after=self.retry.after,
            before_sleep=self.retry.before_sleep,
            reraise=self.retry.reraise,
            retry_error_cls=self.retry.retry_error_cls,
            retry_error_callback=self.retry.retry_error_callback,
        )

    def _async_retrying(self) -> AsyncRetrying:
        return AsyncRetrying(
            sleep=self.retry.sleep,
            stop=self.retry.stop,
            wait=self.retry.wait,
            retry=self.retry.retry,
            before=self.retry.before,
            after=self.retry.after,
            before_sleep=self.retry.before_sleep,
            reraise=self.retry.reraise,
            retry_error_cls=self.retry.retry_error_cls,
            retry_error_callback=self.retry.retry_error_callback,
        )

    def _patch_config(
        self,
        config: Optional[Union[RunnableConfig, List[RunnableConfig]]],
        retry_state: RetryCallState,
    ) -> RunnableConfig:
        if isinstance(config, list):
            return [self._patch_config(c, retry_state) for c in config]

        config = config or {}
        original_tags = config.get("tags") or []
        return patch_config(
            config,
            tags=original_tags
            + ["retry:attempt:{}".format(retry_state.attempt_number)],
        )

    def invoke(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,
        **kwargs: Any | None
    ) -> Output:
        for attempt in self._sync_retrying():
            with attempt:
                result = super().invoke(
                    input, self._patch_config(config, attempt.retry_state), **kwargs
                )
            if not attempt.retry_state.outcome.failed:
                attempt.retry_state.set_result(result)
        return result

    async def ainvoke(
        self, input: Input, config: RunnableConfig | None = None, **kwargs: Any | None
    ) -> Output:
        async for attempt in self._async_retrying():
            with attempt:
                result = await super().ainvoke(
                    input, self._patch_config(config, attempt.retry_state), **kwargs
                )
            if not attempt.retry_state.outcome.failed:
                attempt.retry_state.set_result(result)
        return result

    def batch(
        self,
        inputs: List[Input],
        config: Optional[Union[RunnableConfig, List[RunnableConfig]]] = None,
        **kwargs: Any
    ) -> List[Output]:
        for attempt in self._sync_retrying():
            with attempt:
                result = super().batch(
                    inputs, self._patch_config(config, attempt.retry_state), **kwargs
                )
            if not attempt.retry_state.outcome.failed:
                attempt.retry_state.set_result(result)
        return result

    async def abatch(
        self,
        inputs: List[Input],
        config: Optional[Union[RunnableConfig, List[RunnableConfig]]] = None,
        **kwargs: Any
    ) -> List[Output]:
        async for attempt in self._async_retrying():
            with attempt:
                result = await super().abatch(
                    inputs, self._patch_config(config, attempt.retry_state), **kwargs
                )
            if not attempt.retry_state.outcome.failed:
                attempt.retry_state.set_result(result)
        return result
