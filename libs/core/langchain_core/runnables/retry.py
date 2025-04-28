"""Runnable that retries a Runnable if it fails."""

from typing import (
    TYPE_CHECKING,
    Any,
    Optional,
    TypeVar,
    Union,
    cast,
)

from tenacity import (
    AsyncRetrying,
    RetryCallState,
    RetryError,
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)
from typing_extensions import TypedDict, override

from langchain_core.runnables.base import RunnableBindingBase
from langchain_core.runnables.config import RunnableConfig, patch_config
from langchain_core.runnables.utils import Input, Output

if TYPE_CHECKING:
    from langchain_core.callbacks.manager import (
        AsyncCallbackManagerForChainRun,
        CallbackManagerForChainRun,
    )

    T = TypeVar("T", CallbackManagerForChainRun, AsyncCallbackManagerForChainRun)
U = TypeVar("U")


class ExponentialJitterParams(TypedDict, total=False):
    """Parameters for ``tenacity.wait_exponential_jitter``."""

    initial: float
    """Initial wait."""
    max: float
    """Maximum wait."""
    exp_base: float
    """Base for exponential backoff."""
    jitter: float
    """Random additional wait sampled from random.uniform(0, jitter)."""


class RunnableRetry(RunnableBindingBase[Input, Output]):
    """Retry a Runnable if it fails.

    RunnableRetry can be used to add retry logic to any object
    that subclasses the base Runnable.

    Such retries are especially useful for network calls that may fail
    due to transient errors.

    The RunnableRetry is implemented as a RunnableBinding. The easiest
    way to use it is through the `.with_retry()` method on all Runnables.

    Example:
    Here's an example that uses a RunnableLambda to raise an exception

        .. code-block:: python

            import time

            def foo(input) -> None:
                '''Fake function that raises an exception.'''
                raise ValueError(f"Invoking foo failed. At time {time.time()}")

            runnable = RunnableLambda(foo)

            runnable_with_retries = runnable.with_retry(
                retry_if_exception_type=(ValueError,), # Retry only on ValueError
                wait_exponential_jitter=True, # Add jitter to the exponential backoff
                stop_after_attempt=2, # Try twice
                exponential_jitter_params={"initial": 2},  # if desired, customize backoff
            )

            # The method invocation above is equivalent to the longer form below:

            runnable_with_retries = RunnableRetry(
                bound=runnable,
                retry_exception_types=(ValueError,),
                max_attempt_number=2,
                wait_exponential_jitter=True,
                exponential_jitter_params={"initial": 2},
            )

    This logic can be used to retry any Runnable, including a chain of Runnables,
    but in general it's best practice to keep the scope of the retry as small as
    possible. For example, if you have a chain of Runnables, you should only retry
    the Runnable that is likely to fail, not the entire chain.

    Example:

        .. code-block:: python

            from langchain_core.chat_models import ChatOpenAI
            from langchain_core.prompts import PromptTemplate

            template = PromptTemplate.from_template("tell me a joke about {topic}.")
            model = ChatOpenAI(temperature=0.5)

            # Good
            chain = template | model.with_retry()

            # Bad
            chain = template | model
            retryable_chain = chain.with_retry()
    """  # noqa: E501

    retry_exception_types: tuple[type[BaseException], ...] = (Exception,)
    """The exception types to retry on. By default all exceptions are retried.

    In general you should only retry on exceptions that are likely to be
    transient, such as network errors.

    Good exceptions to retry are all server errors (5xx) and selected client
    errors (4xx) such as 429 Too Many Requests.
    """

    wait_exponential_jitter: bool = True
    """Whether to add jitter to the exponential backoff."""

    exponential_jitter_params: Optional[ExponentialJitterParams] = None
    """Parameters for ``tenacity.wait_exponential_jitter``. Namely: ``initial``,
    ``max``, ``exp_base``, and ``jitter`` (all float values).
    """

    max_attempt_number: int = 3
    """The maximum number of attempts to retry the Runnable."""

    @property
    def _kwargs_retrying(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}

        if self.max_attempt_number:
            kwargs["stop"] = stop_after_attempt(self.max_attempt_number)

        if self.wait_exponential_jitter:
            kwargs["wait"] = wait_exponential_jitter(
                **(self.exponential_jitter_params or {})
            )

        if self.retry_exception_types:
            kwargs["retry"] = retry_if_exception_type(self.retry_exception_types)

        return kwargs

    def _sync_retrying(self, **kwargs: Any) -> Retrying:
        return Retrying(**self._kwargs_retrying, **kwargs)

    def _async_retrying(self, **kwargs: Any) -> AsyncRetrying:
        return AsyncRetrying(**self._kwargs_retrying, **kwargs)

    @staticmethod
    def _patch_config(
        config: RunnableConfig,
        run_manager: "T",
        retry_state: RetryCallState,
    ) -> RunnableConfig:
        attempt = retry_state.attempt_number
        tag = f"retry:attempt:{attempt}" if attempt > 1 else None
        return patch_config(config, callbacks=run_manager.get_child(tag))

    def _patch_config_list(
        self,
        config: list[RunnableConfig],
        run_manager: list["T"],
        retry_state: RetryCallState,
    ) -> list[RunnableConfig]:
        return [
            self._patch_config(c, rm, retry_state) for c, rm in zip(config, run_manager)
        ]

    def _invoke(
        self,
        input: Input,
        run_manager: "CallbackManagerForChainRun",
        config: RunnableConfig,
        **kwargs: Any,
    ) -> Output:
        for attempt in self._sync_retrying(reraise=True):
            with attempt:
                result = super().invoke(
                    input,
                    self._patch_config(config, run_manager, attempt.retry_state),
                    **kwargs,
                )
            if attempt.retry_state.outcome and not attempt.retry_state.outcome.failed:
                attempt.retry_state.set_result(result)
        return result

    @override
    def invoke(
        self, input: Input, config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> Output:
        return self._call_with_config(self._invoke, input, config, **kwargs)

    async def _ainvoke(
        self,
        input: Input,
        run_manager: "AsyncCallbackManagerForChainRun",
        config: RunnableConfig,
        **kwargs: Any,
    ) -> Output:
        async for attempt in self._async_retrying(reraise=True):
            with attempt:
                result = await super().ainvoke(
                    input,
                    self._patch_config(config, run_manager, attempt.retry_state),
                    **kwargs,
                )
            if attempt.retry_state.outcome and not attempt.retry_state.outcome.failed:
                attempt.retry_state.set_result(result)
        return result

    @override
    async def ainvoke(
        self, input: Input, config: Optional[RunnableConfig] = None, **kwargs: Any
    ) -> Output:
        return await self._acall_with_config(self._ainvoke, input, config, **kwargs)

    def _batch(
        self,
        inputs: list[Input],
        run_manager: list["CallbackManagerForChainRun"],
        config: list[RunnableConfig],
        **kwargs: Any,
    ) -> list[Union[Output, Exception]]:
        results_map: dict[int, Output] = {}

        def pending(iterable: list[U]) -> list[U]:
            return [item for idx, item in enumerate(iterable) if idx not in results_map]

        not_set: list[Output] = []
        result = not_set
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
                        **kwargs,
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
            if result is not_set:
                result = cast("list[Output]", [e] * len(inputs))

        outputs: list[Union[Output, Exception]] = []
        for idx in range(len(inputs)):
            if idx in results_map:
                outputs.append(results_map[idx])
            else:
                outputs.append(result.pop(0))
        return outputs

    @override
    def batch(
        self,
        inputs: list[Input],
        config: Optional[Union[RunnableConfig, list[RunnableConfig]]] = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Any,
    ) -> list[Output]:
        return self._batch_with_config(
            self._batch, inputs, config, return_exceptions=return_exceptions, **kwargs
        )

    async def _abatch(
        self,
        inputs: list[Input],
        run_manager: list["AsyncCallbackManagerForChainRun"],
        config: list[RunnableConfig],
        **kwargs: Any,
    ) -> list[Union[Output, Exception]]:
        results_map: dict[int, Output] = {}

        def pending(iterable: list[U]) -> list[U]:
            return [item for idx, item in enumerate(iterable) if idx not in results_map]

        not_set: list[Output] = []
        result = not_set
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
                        **kwargs,
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
            if result is not_set:
                result = cast("list[Output]", [e] * len(inputs))

        outputs: list[Union[Output, Exception]] = []
        for idx in range(len(inputs)):
            if idx in results_map:
                outputs.append(results_map[idx])
            else:
                outputs.append(result.pop(0))
        return outputs

    @override
    async def abatch(
        self,
        inputs: list[Input],
        config: Optional[Union[RunnableConfig, list[RunnableConfig]]] = None,
        *,
        return_exceptions: bool = False,
        **kwargs: Any,
    ) -> list[Output]:
        return await self._abatch_with_config(
            self._abatch, inputs, config, return_exceptions=return_exceptions, **kwargs
        )

    # stream() and transform() are not retried because retrying a stream
    # is not very intuitive.
