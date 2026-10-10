"""Offline conformance tests for the beta decision-model interface.

Import this module directly; existing standard-test imports remain usable with older
core releases. Provider unit suites must inject mock transports into model parameters.
"""

from __future__ import annotations

import math
from abc import abstractmethod
from typing import Any

import pytest
from langchain_core.decision_models import (
    BaseDecisionModel,
    Choice,
    ChoiceAnswer,
    DecisionLevel,
    DecisionOption,
    DecisionRequest,
    DecisionResponse,
    Noul,
    NoulAnswer,
    RefusalAnswer,
    Score,
    ScoreAnswer,
)
from langchain_core.exceptions import ModelInvalidRequestError

from langchain_tests.base import BaseStandardTests


class DecisionModelTests(BaseStandardTests):
    """Constructor and request fixtures shared by decision test suites."""

    @property
    @abstractmethod
    def decision_model_class(self) -> type[BaseDecisionModel]:
        """Return the decision model class to construct."""

    @property
    def decision_model_params(self) -> dict[str, Any]:
        """Return constructor parameters; unit tests must include mock transports."""
        return {}

    @pytest.fixture
    def model(self) -> BaseDecisionModel:
        """Construct the configured decision model."""
        return self.decision_model_class(**self.decision_model_params)

    @pytest.fixture
    def decision_request(self) -> DecisionRequest:
        """Ask all three primitives about a shared support ticket."""
        return {
            "state": "Payments have failed for three days. Please help!",
            "questions": {
                "urgent": Noul(instructions="Is this urgent?"),
                "team": Choice(
                    instructions="Which team?",
                    options=[
                        DecisionOption(value="billing"),
                        DecisionOption(value="technical"),
                    ],
                ),
                "severity": Score(
                    instructions="How severe?",
                    levels=[
                        DecisionLevel(label="calm"),
                        DecisionLevel(label="frustrated"),
                        DecisionLevel(label="angry"),
                    ],
                ),
            },
        }

    @staticmethod
    def _assert_response(request: DecisionRequest, response: DecisionResponse) -> None:
        assert isinstance(response, DecisionResponse)
        assert set(response.answers) == set(request["questions"])
        for name, question in request["questions"].items():
            answer = response.answers[name]
            if isinstance(answer, RefusalAnswer):
                continue
            assert question.type == answer.type
            if isinstance(answer, NoulAnswer):
                assert math.isfinite(answer.probability)
                assert 0 <= answer.probability <= 1
            elif isinstance(answer, ChoiceAnswer):
                assert isinstance(question, Choice)
                values = [(type(x.value), x.value) for x in answer.probabilities]
                assert values == [(type(x.value), x.value) for x in question.options]
                assert (type(answer.value), answer.value) in values
                assert all(
                    math.isfinite(x.probability) and 0 <= x.probability <= 1
                    for x in answer.probabilities
                )
                assert math.isclose(
                    sum(x.probability for x in answer.probabilities),
                    1,
                    rel_tol=0,
                    abs_tol=1e-3,
                )
            else:
                assert isinstance(question, Score)
                assert isinstance(answer, ScoreAnswer)
                assert answer.levels == question.levels
                assert len(answer.probabilities) == len(question.levels)
                assert all(
                    math.isfinite(x) and 0 <= x <= 1 for x in answer.probabilities
                )
                assert math.isclose(
                    sum(answer.probabilities), 1, rel_tol=0, abs_tol=1e-3
                )
                assert math.isclose(
                    answer.score,
                    sum(i * p for i, p in enumerate(answer.probabilities)),
                    rel_tol=0,
                    abs_tol=1e-3,
                )


class DecisionModelUnitTests(DecisionModelTests):
    """Reusable offline tests for provider adapters and local implementations.

    Override `decision_model_class` and `decision_model_params`. Adapters returned by
    `as_decision_model()` can be tested using their type and configured native instance.
    Mandatory tests must not be overridden. These checks assert the contract rather
    than model accuracy, calibration, or equivalent provider confidence thresholds.
    """

    def test_init(self) -> None:
        """Construct a model using the supplied parameters."""
        assert isinstance(
            self.decision_model_class(**self.decision_model_params), BaseDecisionModel
        )

    def test_invoke(
        self, model: BaseDecisionModel, decision_request: DecisionRequest
    ) -> None:
        """Return complete typed answers for mixed questions."""
        self._assert_response(decision_request, model.invoke(decision_request))

    async def test_ainvoke(
        self, model: BaseDecisionModel, decision_request: DecisionRequest
    ) -> None:
        """Return the same contract asynchronously."""
        self._assert_response(decision_request, await model.ainvoke(decision_request))

    def test_batch(
        self, model: BaseDecisionModel, decision_request: DecisionRequest
    ) -> None:
        """Retain output count and order for batching."""
        outputs = model.batch([decision_request, decision_request])
        assert len(outputs) == 2
        for output in outputs:
            self._assert_response(decision_request, output)

    async def test_abatch(
        self, model: BaseDecisionModel, decision_request: DecisionRequest
    ) -> None:
        """Return all async batch outputs."""
        outputs = await model.abatch([decision_request, decision_request])
        assert len(outputs) == 2
        for output in outputs:
            self._assert_response(decision_request, output)

    def test_stream(
        self, model: BaseDecisionModel, decision_request: DecisionRequest
    ) -> None:
        """Yield one complete response, without invented token chunks."""
        outputs = list(model.stream(decision_request))
        assert len(outputs) == 1
        self._assert_response(decision_request, outputs[0])

    async def test_astream(
        self, model: BaseDecisionModel, decision_request: DecisionRequest
    ) -> None:
        """Yield one complete response asynchronously."""
        outputs = [output async for output in model.astream(decision_request)]
        assert len(outputs) == 1
        self._assert_response(decision_request, outputs[0])

    def test_invalid_request(self, model: BaseDecisionModel) -> None:
        """Reject missing questions before any provider call."""
        with pytest.raises(ModelInvalidRequestError):
            model.invoke({"state": "hello", "questions": {}})

    async def test_invalid_request_async(self, model: BaseDecisionModel) -> None:
        """Reject missing questions asynchronously before any provider call."""
        with pytest.raises(ModelInvalidRequestError):
            await model.ainvoke({"state": "hello", "questions": {}})

    def test_batch_return_exceptions(
        self, model: BaseDecisionModel, decision_request: DecisionRequest
    ) -> None:
        """Keep valid results and failures in their input positions."""
        outputs = model.batch(
            [decision_request, {"state": "hello", "questions": {}}],
            return_exceptions=True,
        )
        assert isinstance(outputs[0], DecisionResponse)
        self._assert_response(decision_request, outputs[0])
        assert isinstance(outputs[1], ModelInvalidRequestError)
