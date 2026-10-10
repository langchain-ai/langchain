"""Deterministic decision model for offline examples and tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

from typing_extensions import override

from langchain_core.decision_models.base import BaseDecisionModel
from langchain_core.decision_models.types import DecisionResponse  # noqa: TC001

if TYPE_CHECKING:
    from langchain_core.callbacks import (
        AsyncCallbackManagerForChainRun,
        CallbackManagerForChainRun,
    )
    from langchain_core.decision_models.types import DecisionRequest
    from langchain_core.runnables import RunnableConfig


class FakeDecisionModel(BaseDecisionModel):
    """Return a configured response through the real validation and run lifecycle.

    Args:
        response: Complete response returned independently for every request.
        profile: Optional known capabilities.
    """

    response: DecisionResponse

    @override
    def _decide(
        self,
        request: DecisionRequest,
        *,
        config: RunnableConfig,
        run_manager: CallbackManagerForChainRun,
    ) -> DecisionResponse:
        return self.response.model_copy(deep=True)

    @override
    async def _adecide(
        self,
        request: DecisionRequest,
        *,
        config: RunnableConfig,
        run_manager: AsyncCallbackManagerForChainRun,
    ) -> DecisionResponse:
        return self.response.model_copy(deep=True)
