"""Evaluators for parsing strings."""

import json
import logging
from collections.abc import Callable
from operator import eq
from typing import Any, cast

from langchain_core.utils.json import parse_json_markdown
from typing_extensions import override

from langchain_classic.evaluation.schema import StringEvaluator

_logger = logging.getLogger(__name__)


class JsonValidityEvaluator(StringEvaluator):
    """Evaluate whether the prediction is valid JSON.

    This evaluator checks if the prediction is a valid JSON string. It does not
        require any input or reference.

    Attributes:
        requires_input: Whether this evaluator requires an input
            string. Always False.
        requires_reference: Whether this evaluator requires a
            reference string. Always False.
        evaluation_name: The name of the evaluation metric.
            Always "json".

    Examples:
        >>> evaluator = JsonValidityEvaluator()
        >>> prediction = '{"name": "John", "age": 30, "city": "New York"}'
        >>> evaluator.evaluate(prediction)
        {'score': 1}

        >>> prediction = '{"name": "John", "age": 30, "city": "New York",}'
        >>> evaluator.evaluate(prediction)
        {'score': 0, 'reasoning': 'Expecting property name enclosed in double quotes'}
    """

    def __init__(self, **_: Any) -> None:
        """Initialize the JsonValidityEvaluator."""
        super().__init__()

    @property
    @override
    def requires_input(self) -> bool:
        return False

    @property
    @override
    def requires_reference(self) -> bool:
        return False

    @property
    @override
    def evaluation_name(self) -> str:
        return "json_validity"

    @override
    def _evaluate_strings(
        self,
        prediction: str,
        **kwargs: Any,
    ) -> dict:
        """Evaluate the prediction string.

        Args:
            prediction: The prediction string to evaluate.
            **kwargs: Additional keyword arguments (not used).

        Returns:
            `dict` containing the evaluation score. The score is `1` if
                the prediction is valid JSON, and `0` otherwise.

                If the prediction is not valid JSON, the dictionary also contains
                a `reasoning` field with the error message.

        """
        try:
            parse_json_markdown(prediction, parser=json.loads)
        except json.JSONDecodeError as e:
            return {"score": 0, "reasoning": str(e)}
        except Exception as e:
            _logger.exception("Passing JSON failed with unexpected error.")
            return {"score": 0, "reasoning": str(e)}
        return {"score": 1}


class JsonEqualityEvaluator(StringEvaluator):
    """Json Equality Evaluator.

    Evaluate whether the prediction is equal to the reference after
    parsing both as JSON.

    This evaluator checks if the prediction, after parsing as JSON, is equal
        to the reference,
    which is also parsed as JSON. It does not require an input string.

    Attributes:
        requires_input: Whether this evaluator requires an
            input string. Always False.
        requires_reference: Whether this evaluator requires
            a reference string. Always True.
        evaluation_name: The name of the evaluation metric.
            Always "parsed_equality".

    Examples:
        >>> evaluator = JsonEqualityEvaluator()
        >>> evaluator.evaluate_strings('{"a": 1}', reference='{"a": 1}')
        {'score': True}
        >>> evaluator.evaluate_strings('{"a": 1}', reference='{"a": 2}')
        {'score': False}

        >>> evaluator = JsonEqualityEvaluator(operator=lambda x, y: x["a"] == y["a"])
        >>> evaluator.evaluate_strings('{"a": 1}', reference='{"a": 1}')
        {'score': True}
        >>> evaluator.evaluate_strings('{"a": 1}', reference='{"a": 2}')
        {'score': False}

    """

    def __init__(self, operator: Callable | None = None, **_: Any) -> None:
        """Initialize the JsonEqualityEvaluator.

        Args:
            operator: A custom operator to compare the parsed JSON objects.
                Defaults to equality (`eq`).
        """
        super().__init__()
        self.operator = operator or eq

    @property
    @override
    def requires_input(self) -> bool:
        return False

    @property
    @override
    def requires_reference(self) -> bool:
        return True

    @property
    @override
    def evaluation_name(self) -> str:
        return "json_equality"

    def _parse_json(
        self,
        string: Any,
    ) -> dict | list | None | float | bool | int | str:
        if isinstance(string, str):
            return parse_json_markdown(string)
        return string

    @override
    def _evaluate_strings(
        self,
        prediction: str,
        reference: str | None = None,
        **kwargs: Any,
    ) -> dict:
        """Evaluate the prediction string.

        Args:
            prediction: The prediction string to evaluate.
            reference: The reference string to compare against.
            **kwargs: Additional keyword arguments (not used).

        Returns:
            `dict` containing the evaluation score.
        """
        parsed = self._parse_json(prediction)
        label = self._parse_json(cast("str", reference))
        if isinstance(label, list):
            if not isinstance(parsed, list):
                return {"score": 0}
            parsed = sorted(parsed, key=lambda x: str(x))
            label = sorted(label, key=lambda x: str(x))
        return {"score": self.operator(parsed, label)}
