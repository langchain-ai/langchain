"""Reusable Runnable execution for typed decision models."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from langsmith.run_helpers import get_current_run_tree
from pydantic import ValidationError
from typing_extensions import override

from langchain_core._api import beta
from langchain_core.decision_models._validation import (
    DecisionResponseValidationError,
    normalize_request,
    validate_capabilities,
    validate_response,
)
from langchain_core.decision_models.types import (
    DecisionModelProfile,
    DecisionRequest,
    DecisionResponse,
    DecisionUsage,
)
from langchain_core.exceptions import ModelInvalidRequestError
from langchain_core.runnables import RunnableConfig, RunnableSerializable
from langchain_core.runnables.config import ensure_config, run_in_executor

if TYPE_CHECKING:
    from langchain_core.callbacks import (
        AsyncCallbackManagerForChainRun,
        CallbackManagerForChainRun,
    )

logger = logging.getLogger(__name__)


@beta()
class BaseDecisionModel(RunnableSerializable[DecisionRequest, DecisionResponse], ABC):
    """Evaluate typed questions with one shared request and result interface.

    !!! warning
        This interface is in beta. Implementations must translate provider-native
        responses and preserve refusals and confidence semantics.

    Implement `_decide` and, optionally, `_adecide`. The base validates requests,
    capabilities, and responses within a single callback lifecycle. Standard Runnable
    batching, composition, retries, and fallbacks remain available. Streaming yields
    one complete response. Refusals are results, so exception fallbacks do not fire
    automatically for them.

    SDK retries and client ownership remain with provider implementations. Configure
    Runnable retry exception types explicitly: `ModelError.is_retryable` does not
    automatically control `with_retry`. The default async implementation uses an
    executor and cannot cancel an already-running synchronous transport.

    Args:
        profile: Known capabilities. Missing information is unknown.
    """

    profile: DecisionModelProfile | None = None
    """Known supported question/state types and limits."""

    @abstractmethod
    def _decide(
        self,
        request: DecisionRequest,
        *,
        config: RunnableConfig,
        run_manager: CallbackManagerForChainRun,
    ) -> DecisionResponse:
        """Perform inference and translate its complete response."""

    async def _adecide(
        self,
        request: DecisionRequest,
        *,
        config: RunnableConfig,
        run_manager: AsyncCallbackManagerForChainRun,
    ) -> DecisionResponse:
        """Use the synchronous provider hook in a context-preserving executor."""
        return await run_in_executor(
            config,
            self._decide,
            request,
            config=config,
            run_manager=run_manager.get_sync(),
        )

    @property
    def _tracing_metadata(self) -> dict[str, Any]:
        """Use existing trace conventions without introducing a new run enum."""
        return {"ls_model_type": "llm", "decision_model": True}

    def _traced_config(self, config: RunnableConfig | None) -> RunnableConfig:
        config = ensure_config(config)
        config["metadata"] = {
            **(config.get("metadata") or {}),
            **self._tracing_metadata,
        }
        return config

    @override
    def invoke(
        self,
        input: DecisionRequest,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> DecisionResponse:
        """Evaluate and validate one request synchronously.

        Args:
            input: State and named typed questions, or equivalent dictionaries.
            config: Runnable callbacks, tags, metadata, and execution settings.
            **kwargs: Unsupported invocation options are rejected.

        Returns:
            Validated answers with reported provider metadata and usage.

        Raises:
            ModelInvalidRequestError: If the request or capabilities are invalid.
            DecisionResponseValidationError: If the response violates the contract.
        """
        return self._call_with_config(
            self._invoke, input, self._traced_config(config), run_type="llm", **kwargs
        )

    @override
    async def ainvoke(
        self,
        input: DecisionRequest,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> DecisionResponse:
        """Evaluate and validate one request asynchronously.

        Args:
            input: State and named typed questions, or equivalent dictionaries.
            config: Runnable callbacks, tags, metadata, and execution settings.
            **kwargs: Unsupported invocation options are rejected.

        Returns:
            Validated answers with reported provider metadata and usage.

        Raises:
            ModelInvalidRequestError: If the request or capabilities are invalid.
            DecisionResponseValidationError: If the response violates the contract.
        """
        return await self._acall_with_config(
            self._ainvoke, input, self._traced_config(config), run_type="llm", **kwargs
        )

    def _prepare(
        self, request: DecisionRequest, kwargs: dict[str, Any]
    ) -> DecisionRequest:
        if kwargs:
            msg = "Decision models do not support invocation keyword arguments."
            raise ModelInvalidRequestError(msg)
        request = normalize_request(request)
        validate_capabilities(request, self.profile)
        return request

    def _invoke(
        self,
        request: DecisionRequest,
        run_manager: CallbackManagerForChainRun,
        config: RunnableConfig,
        **kwargs: Any,
    ) -> DecisionResponse:
        request = self._prepare(request, kwargs)
        try:
            response = self._decide(request, config=config, run_manager=run_manager)
        except ValidationError:
            msg = "Invalid decision response fields."
            raise DecisionResponseValidationError(msg) from None
        self._record_usage(response)
        return validate_response(request, response)

    async def _ainvoke(
        self,
        request: DecisionRequest,
        run_manager: AsyncCallbackManagerForChainRun,
        config: RunnableConfig,
        **kwargs: Any,
    ) -> DecisionResponse:
        request = self._prepare(request, kwargs)
        try:
            response = await self._adecide(
                request, config=config, run_manager=run_manager
            )
        except ValidationError:
            msg = "Invalid decision response fields."
            raise DecisionResponseValidationError(msg) from None
        self._record_usage(response)
        return validate_response(request, response)

    @staticmethod
    def _record_usage(response: DecisionResponse) -> None:
        try:
            usage = DecisionUsage.model_validate(response.usage)
            counts = {
                key: value
                for key, value in (
                    ("input_tokens", usage.input_tokens),
                    ("output_tokens", usage.output_tokens),
                    ("total_tokens", usage.total_tokens),
                )
                if value is not None
            }
            run = get_current_run_tree()
            if run is not None and counts:
                run.extra.setdefault("metadata", {})["usage_metadata"] = counts
        except Exception:
            logger.debug("Could not attach decision usage.", exc_info=True)
