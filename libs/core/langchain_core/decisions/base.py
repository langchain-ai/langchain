"""Base class for decision models."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from langsmith.run_helpers import get_current_run_tree
from pydantic import Field
from typing_extensions import override

from langchain_core._api import beta
from langchain_core.decisions.types import DecisionRequest, DecisionResponse
from langchain_core.runnables import RunnableConfig, RunnableSerializable
from langchain_core.runnables.config import ensure_config, run_in_executor

logger = logging.getLogger(__name__)


@beta()
class BaseDecisionModel(RunnableSerializable[DecisionRequest, DecisionResponse], ABC):
    """Answer typed questions about text, messages, or structured state.

    A decision model evaluates shared input against one or more questions and
    returns typed answers: a probability for a `Predicate`, a selected value for a
    `Choice`, and an expected level for a `Score`. Answers keep their probability
    distributions and confidence so applications can choose their own thresholds.

    Subclasses implement `_decide`, and optionally `_adecide`, to call a provider.
    This class handles callbacks, tracing metadata, and token-usage reporting.
    """

    model: str = Field(min_length=1)
    """Provider model name used to answer questions."""

    @property
    @abstractmethod
    def _provider(self) -> str:
        """Provider name reported to LangSmith as `ls_provider`."""

    @abstractmethod
    def _decide(self, request: DecisionRequest) -> DecisionResponse:
        """Answer one request synchronously.

        Args:
            request: Input and questions to evaluate.

        Returns:
            Typed answers keyed by question name, with request metadata.
        """

    async def _adecide(self, request: DecisionRequest) -> DecisionResponse:
        """Answer one request asynchronously.

        The default implementation runs `_decide` in an executor. Override it to use
        a native asynchronous client.

        Args:
            request: Input and questions to evaluate.

        Returns:
            Typed answers keyed by question name, with request metadata.
        """
        return await run_in_executor(None, self._decide, request)

    @override
    def invoke(
        self,
        input: DecisionRequest,
        config: RunnableConfig | None = None,
        **_: Any,
    ) -> DecisionResponse:
        """Answer one request synchronously.

        Args:
            input: Input and questions to evaluate.
            config: Optional runnable configuration for callbacks, tags, metadata,
                and tracing.
            **_: Accepted for `Runnable` compatibility and otherwise ignored.

        Returns:
            Typed answers keyed by question name, with request metadata.
        """
        return self._call_with_config(
            self._decide_and_record,
            input,
            self._traced_config(config),
            run_type="llm",
        )

    @override
    async def ainvoke(
        self,
        input: DecisionRequest,
        config: RunnableConfig | None = None,
        **_: Any,
    ) -> DecisionResponse:
        """Answer one request asynchronously.

        Args:
            input: Input and questions to evaluate.
            config: Optional runnable configuration for callbacks, tags, metadata,
                and tracing.
            **_: Accepted for `Runnable` compatibility and otherwise ignored.

        Returns:
            Typed answers keyed by question name, with request metadata.
        """
        return await self._acall_with_config(
            self._adecide_and_record,
            input,
            self._traced_config(config),
            run_type="llm",
        )

    def _decide_and_record(self, request: DecisionRequest) -> DecisionResponse:
        return _record_usage(self._decide(request))

    async def _adecide_and_record(self, request: DecisionRequest) -> DecisionResponse:
        return _record_usage(await self._adecide(request))

    def _traced_config(self, config: RunnableConfig | None) -> RunnableConfig:
        """Set `ls_provider` and `ls_model_name` when the run is created."""
        config = ensure_config(config)
        config["metadata"] = {
            **(config.get("metadata") or {}),
            "ls_provider": self._provider,
            "ls_model_name": self.model,
            "ls_model_type": "chat",
        }
        return config


def _record_usage(response: DecisionResponse) -> DecisionResponse:
    """Attach token usage to the active run, if there is one.

    Nothing is written when tracing is disabled, and a tracing failure never fails
    an otherwise successful request.
    """
    input_tokens = response.usage.input_tokens or 0
    output_tokens = response.usage.output_tokens or 0
    try:
        run_tree = get_current_run_tree()
        if run_tree is not None:
            run_tree.extra.setdefault("metadata", {})["usage_metadata"] = {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            }
    except Exception:  # tracing must not break decisions
        logger.debug("Could not attach decision usage.", exc_info=True)
    return response
