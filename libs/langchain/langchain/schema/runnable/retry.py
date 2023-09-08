from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union, cast

from tenacity import (
    AsyncRetrying,
    RetryCallState,
    RetryError,
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain.schema.runnable.base import Input, Output, RunnableBinding
from langchain.schema.runnable.config import RunnableConfig, patch_config

T = TypeVar("T", CallbackManagerForChainRun, AsyncCallbackManagerForChainRun)
U = TypeVar("U")


class RunnableRetry(RunnableBinding[Input, Output]):
    """Retry a Runnable if it fails."""

    retry_exception_types: Tuple[Type[BaseException], ...] = (Exception,)

    wait_exponential_jitter: bool = True

    max_attempt_number: int = 3

    @property
    def _kwargs_retrying(self) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = dict()

        if self.max_attempt_number:
            kwargs["stop"] = stop_after_attempt(self.max_attempt_number)

        if self.wait_exponential_jitter:
            kwargs["wait"] = wait_exponential_jitter()

        if self.retry_exception_types:
            kwargs["retry"] = retry_if_exception_type(self.retry_exception_types)

        return kwargs

    def _sync_retrying(self, **kwargs: Any) -> Retrying:
        return Retrying(**self._kwargs_retrying, **kwargs)

    def _async_retrying(self, **kwargs: Any) -> AsyncRetrying:
        return AsyncRetrying(**self._kwargs_retrying, **kwargs)

    def _patch_config(
        self,
        config: RunnableConfig,
        run_manager: T,
        retry_state: RetryCallState,
    ) -> RunnableConfig:
        attempt = retry_state.attempt_number
        tag = "retry:attempt:{}".format(attempt) if attempt > 1 else None
        return patch_config(config, callbacks=run_manager.get_child(tag))

    def _patch_config_list(
        self,
        config: List[RunnableConfig],
        run_manager: List[T],
        retry_state: RetryCallState,
    ) -> List[RunnableConfig]:
        return [
            self._patch_config(c, rm, retry_state) for c, rm in zip(config, run_manager)
        ]

    def _invoke(
        self,
        input: Input,
        run_manager: CallbackManagerForChainRun,
        config: RunnableConfig,
    ) -> Output:
        for attempt in self._sync_retrying(reraise=True):
            with attempt:
                result = super().invoke(
                    input,
                    self._patch_config(config, run_manager, attempt.retry_state),
                )
            if attempt.retry_state.outcome and not attempt.retry_state.outcome.failed:
                attempt.retry_state.set_result(result)
        return result

    def invoke(
        self, input: Input, config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> Output:
        return self._call_with_config(self._invoke, input, config, **kwargs)

    async def _ainvoke(
        self,
        input: Input,
        run_manager: AsyncCallbackManagerForChainRun,
        config: RunnableConfig,
    ) -> Output:
        async for attempt in self._async_retrying(reraise=True):
            with attempt:
                result = await super().ainvoke(
                    input,
                    self._patch_config(config, run_manager, attempt.retry_state),
                )
            if attempt.retry_state.outcome and not attempt.retry_state.outcome.failed:
                attempt.retry_state.set_result(result)
        return result

    async def ainvoke(
        self, input: Input, config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> Output:
        return await self._acall_with_config(self._ainvoke, input, config, **kwargs)

    def _batch(
        self,
        inputs: List[Input],
        run_manager: List[CallbackManagerForChainRun],
        config: List[RunnableConfig],
    ) -> List[Union[Output, Exception]]:
        results_map: Dict[int, Output] = {}

        def pending(iterable: List[U]) -> List[U]:
            return [item for idx, item in enumerate(iterable) if idx not in results_map]

        try:
            for attempt in self._sync_retrying():
                with attempt:
                    # Get the results of the inputs that have not succeeded yet.
                    result = super().batch(
                        pending(inputs),
                        self._patch_config_list(
                            pending(config), pending(run_manager), attempt.retry_state
                        ),
                        return_exceptions=True,
                    )
                    # Register the results of the inputs that have succeeded.
                    first_exception = None
                    for i, r in enumerate(result):
                        if isinstance(r, Exception):
                            if not first_exception:
                                first_exception = r
                            continue
                        results_map[i] = r
                    # If any exception occurred, raise it, to retry the failed ones
                    if first_exception:
                        raise first_exception
                if (
                    attempt.retry_state.outcome
                    and not attempt.retry_state.outcome.failed
                ):
                    attempt.retry_state.set_result(result)
        except RetryError as e:
            try:
                result
            except UnboundLocalError:
                result = cast(List[Output], [e] * len(inputs))

        outputs: List[Union[Output, Exception]] = []
        for idx, _ in enumerate(inputs):
            if idx in results_map:
                outputs.append(results_map[idx])
            else:
                outputs.append(result.pop(0))
        return outputs

    def batch(
        self,
        inputs: List[Input],
        config: Optional[Union[RunnableConfig, List[RunnableConfig]]] = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Any
    ) -> List[Output]:
        return self._batch_with_config(
            self._batch, inputs, config, return_exceptions=return_exceptions, **kwargs
        )

    async def _abatch(
        self,
        inputs: List[Input],
        run_manager: List[AsyncCallbackManagerForChainRun],
        config: List[RunnableConfig],
    ) -> List[Union[Output, Exception]]:
        results_map: Dict[int, Output] = {}

        def pending(iterable: List[U]) -> List[U]:
            return [item for idx, item in enumerate(iterable) if idx not in results_map]

        try:
            async for attempt in self._async_retrying():
                with attempt:
                    # Get the results of the inputs that have not succeeded yet.
                    result = await super().abatch(
                        pending(inputs),
                        self._patch_config_list(
                            pending(config), pending(run_manager), attempt.retry_state
                        ),
                        return_exceptions=True,
                    )
                    # Register the results of the inputs that have succeeded.
                    first_exception = None
                    for i, r in enumerate(result):
                        if isinstance(r, Exception):
                            if not first_exception:
                                first_exception = r
                            continue
                        results_map[i] = r
                    # If any exception occurred, raise it, to retry the failed ones
                    if first_exception:
                        raise first_exception
                if (
                    attempt.retry_state.outcome
                    and not attempt.retry_state.outcome.failed
                ):
                    attempt.retry_state.set_result(result)
        except RetryError as e:
            try:
                result
            except UnboundLocalError:
                result = cast(List[Output], [e] * len(inputs))

        outputs: List[Union[Output, Exception]] = []
        for idx, _ in enumerate(inputs):
            if idx in results_map:
                outputs.append(results_map[idx])
            else:
                outputs.append(result.pop(0))
        return outputs

    async def abatch(
        self,
        inputs: List[Input],
        config: Optional[Union[RunnableConfig, List[RunnableConfig]]] = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Any
    ) -> List[Output]:
        return await self._abatch_with_config(
            self._abatch, inputs, config, return_exceptions=return_exceptions, **kwargs
        )

    # stream() and transform() are not retried because retrying a stream
    # is not very intuitive.
