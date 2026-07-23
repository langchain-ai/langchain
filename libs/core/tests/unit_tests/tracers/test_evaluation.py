"""Regression tests for EvaluationResult.feedback_config field preservation.

These tests guard against the bug where EvaluationResult silently strips unknown
or non-standard keys from the ``feedback_config`` dict.

Root cause: ``feedback_config`` is typed as ``Optional[Union[FeedbackConfig, dict]]``.
Pydantic attempts to resolve the value against the ``FeedbackConfig`` TypedDict branch
first; if that succeeds (even partially), it may discard keys not declared in
``FeedbackConfig``.  The ``make_evaluation_result`` factory works around this by
assigning ``feedback_config`` after ``__init__``, bypassing Pydantic's validator.

Covered scenarios
-----------------
- Direct ``EvaluationResult`` constructor — the originally reported bug path.
- ``make_evaluation_result`` factory — the provided workaround.
- Purely unknown keys (e.g. ``threshold``).
- Mixed known + unknown keys (e.g. ``type`` + ``threshold``).
- Only recognized ``FeedbackConfig`` keys (regression-safe baseline).
- Nested dict values.
- ``None`` / missing ``feedback_config``.
- Empty dict ``{}``.
- Multiple extra keys.
- Value identity: original dict object is not mutated by the constructor.
"""

from __future__ import annotations

import pytest
from langsmith.evaluation.evaluator import EvaluationResult

from langchain_core.tracers.evaluation import make_evaluation_result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_UNKNOWN_ONLY = {"threshold": 1.0}
_MIXED = {"type": "continuous", "threshold": 1.0}
_VALID_ONLY = {"type": "continuous", "min": 0.0, "max": 1.0}
_MULTIPLE_UNKNOWN = {"threshold": 1.0, "custom_field": "test", "extra": 42}
_NESTED = {"threshold": 1.0, "nested": {"a": 1, "b": [1, 2, 3]}}


# ===========================================================================
# Section 1: Direct EvaluationResult constructor (originally reported bug path)
# ===========================================================================


class TestEvaluationResultConstructorFeedbackConfig:
    """Guard the EvaluationResult constructor against silent key stripping."""

    def test_unknown_only_key_preserved(self) -> None:
        """``threshold`` is not a FeedbackConfig field; must not be dropped.

        This is the exact scenario from the original bug report:

            EvaluationResult(key="sentiment", value="positive",
                             feedback_config={"threshold": 1.0})
        """
        result = EvaluationResult(
            key="sentiment",
            value="positive",
            feedback_config={"threshold": 1.0},
        )
        assert result.feedback_config == {"threshold": 1.0}, (
            "EvaluationResult constructor silently dropped 'threshold' from "
            "feedback_config.  The Union[FeedbackConfig, dict] annotation is "
            "causing Pydantic to strip unknown keys — the original bug has regressed."
        )

    def test_mixed_known_and_unknown_keys_preserved(self) -> None:
        """All keys must survive when known and unknown keys are mixed."""
        result = EvaluationResult(
            key="score",
            score=0.9,
            feedback_config=_MIXED.copy(),
        )
        assert result.feedback_config == _MIXED, (
            f"Expected {_MIXED!r}, got {result.feedback_config!r}. "
            "Unknown key 'threshold' was silently stripped."
        )

    def test_valid_only_keys_preserved(self) -> None:
        """Recognised FeedbackConfig keys must still be present after construction."""
        result = EvaluationResult(
            key="score",
            score=0.8,
            feedback_config=_VALID_ONLY.copy(),
        )
        assert result.feedback_config == _VALID_ONLY

    def test_multiple_unknown_keys_all_preserved(self) -> None:
        """Every unknown key must survive, not just the first."""
        result = EvaluationResult(
            key="label",
            value="positive",
            feedback_config=_MULTIPLE_UNKNOWN.copy(),
        )
        assert result.feedback_config == _MULTIPLE_UNKNOWN, (
            f"Expected {_MULTIPLE_UNKNOWN!r}, got {result.feedback_config!r}."
        )

    def test_nested_dict_values_preserved(self) -> None:
        """Nested structures inside feedback_config must not be flattened or lost."""
        result = EvaluationResult(
            key="label",
            value="positive",
            feedback_config=_NESTED.copy(),
        )
        assert result.feedback_config == _NESTED

    def test_empty_dict_preserved(self) -> None:
        """An empty dict is a valid, distinct value from ``None``."""
        result = EvaluationResult(key="label", feedback_config={})
        assert result.feedback_config == {}
        assert result.feedback_config is not None

    def test_none_feedback_config(self) -> None:
        """Omitting feedback_config should leave it as ``None``."""
        result = EvaluationResult(key="label")
        assert result.feedback_config is None

    def test_explicit_none_feedback_config(self) -> None:
        """Explicitly passing None should leave it as ``None``."""
        result = EvaluationResult(key="label", feedback_config=None)
        assert result.feedback_config is None

    def test_original_dict_not_mutated(self) -> None:
        """The constructor must not mutate the caller's original dict."""
        original = {"threshold": 1.0, "extra": "keep_me"}
        EvaluationResult(key="label", feedback_config=original)
        assert original == {"threshold": 1.0, "extra": "keep_me"}, (
            "EvaluationResult constructor mutated the caller's dict."
        )

    def test_complete_key_value_equality(self) -> None:
        """Exact dict equality check: no keys added, none removed, no value mutation."""
        config = {"threshold": 1.0}
        result = EvaluationResult(
            key="sentiment",
            value="positive",
            feedback_config=config,
        )
        stored = result.feedback_config
        assert isinstance(stored, dict), f"Expected dict, got {type(stored)}"
        assert set(stored.keys()) == {"threshold"}, (
            f"Stored keys {set(stored.keys())} differ from input {{'threshold'}}."
        )
        assert stored["threshold"] == 1.0


# ===========================================================================
# Section 2: make_evaluation_result factory (existing workaround)
# ===========================================================================


class TestMakeEvaluationResultFeedbackConfig:
    """The factory must preserve all feedback_config keys regardless of type."""

    def test_unknown_only_key_preserved(self) -> None:
        """Unknown key must be preserved via the factory workaround."""
        result = make_evaluation_result(
            key="sentiment",
            value="positive",
            feedback_config={"threshold": 1.0},
        )
        assert result.feedback_config == {"threshold": 1.0}

    def test_multiple_unknown_keys_preserved(self) -> None:
        """All unknown keys must survive through the factory."""
        config = {"threshold": 1.0, "custom_field": "test"}
        result = make_evaluation_result(
            key="sentiment",
            value="positive",
            feedback_config=config,
        )
        assert result.key == "sentiment"
        assert result.value == "positive"
        assert result.feedback_config == {"threshold": 1.0, "custom_field": "test"}

    def test_mixed_known_and_unknown_keys_preserved(self) -> None:
        """Known and unknown keys mixed — all must survive."""
        result = make_evaluation_result(
            key="score",
            score=0.9,
            feedback_config=_MIXED.copy(),
        )
        assert result.feedback_config == _MIXED

    def test_valid_only_keys_preserved(self) -> None:
        """Recognised FeedbackConfig keys are passed through intact."""
        config = {"type": "continuous", "min": 0, "max": 1}
        result = make_evaluation_result(
            key="score",
            score=0.8,
            feedback_config=config,
        )
        assert result.score == 0.8
        assert result.feedback_config == {"type": "continuous", "min": 0, "max": 1}

    def test_none_feedback_config(self) -> None:
        """Omitting feedback_config leaves it as None."""
        result = make_evaluation_result(key="sentiment", value="positive")
        assert result.key == "sentiment"
        assert result.feedback_config is None

    def test_nested_dict_values_preserved(self) -> None:
        """Nested structures inside feedback_config must not be lost."""
        result = make_evaluation_result(
            key="label",
            value="positive",
            feedback_config=_NESTED.copy(),
        )
        assert result.feedback_config == _NESTED

    def test_empty_dict_preserved(self) -> None:
        """An empty dict is preserved as-is (not coerced to None)."""
        result = make_evaluation_result(key="label", feedback_config={})
        assert result.feedback_config == {}

    def test_score_and_key_unaffected(self) -> None:
        """Non-feedback_config fields must be unaffected by the factory workaround."""
        result = make_evaluation_result(
            key="precision",
            score=0.95,
            value="high",
            comment="looks good",
            feedback_config={"threshold": 0.5},
        )
        assert result.key == "precision"
        assert result.score == 0.95
        assert result.value == "high"
        assert result.comment == "looks good"
        assert result.feedback_config == {"threshold": 0.5}


# ===========================================================================
# Section 3: Consistency — constructor vs factory must agree
# ===========================================================================


class TestConstructorAndFactoryConsistency:
    """Both paths must produce the same feedback_config outcome."""

    @pytest.mark.parametrize(
        "config",
        [
            pytest.param({"threshold": 1.0}, id="unknown_only"),
            pytest.param({"type": "continuous", "threshold": 1.0}, id="mixed"),
            pytest.param({"type": "continuous", "min": 0.0, "max": 1.0}, id="valid_only"),
            pytest.param(
                {"threshold": 1.0, "custom_field": "test", "extra": 42},
                id="multiple_unknown",
            ),
            pytest.param(
                {"threshold": 1.0, "nested": {"a": 1}}, id="nested_dict"
            ),
            pytest.param({}, id="empty_dict"),
        ],
    )
    def test_constructor_and_factory_agree(self, config: dict) -> None:
        """EvaluationResult() and make_evaluation_result() must store the same dict."""
        via_constructor = EvaluationResult(key="k", feedback_config=config.copy())
        via_factory = make_evaluation_result(key="k", feedback_config=config.copy())

        assert via_constructor.feedback_config == config, (
            f"Constructor lost keys: got {via_constructor.feedback_config!r}, "
            f"expected {config!r}"
        )
        assert via_factory.feedback_config == config, (
            f"Factory lost keys: got {via_factory.feedback_config!r}, "
            f"expected {config!r}"
        )
        assert via_constructor.feedback_config == via_factory.feedback_config, (
            "Constructor and factory produced different feedback_config values — "
            "the factory workaround is no longer equivalent to the constructor."
        )
