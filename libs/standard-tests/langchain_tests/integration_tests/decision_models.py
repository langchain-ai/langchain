"""Opt-in live checks of the beta decision contract, without accuracy benchmarks."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_tests.unit_tests.decision_models import DecisionModelTests

if TYPE_CHECKING:
    from langchain_core.decision_models import BaseDecisionModel, DecisionRequest


class DecisionModelIntegrationTests(DecisionModelTests):
    """Live inference tests requiring explicitly configured provider credentials.

    Override the constructor properties, as for the offline suite. Only supported
    mixed text questions are exercised; refusal and calibration are not forced.
    """

    def test_invoke(
        self, model: BaseDecisionModel, decision_request: DecisionRequest
    ) -> None:
        """Validate a complete live response."""
        self._assert_response(decision_request, model.invoke(decision_request))

    async def test_ainvoke(
        self, model: BaseDecisionModel, decision_request: DecisionRequest
    ) -> None:
        """Validate a complete live async response."""
        self._assert_response(decision_request, await model.ainvoke(decision_request))
