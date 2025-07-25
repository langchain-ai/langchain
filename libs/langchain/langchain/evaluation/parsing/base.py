"""Evaluators for parsing strings."""

import json
from operator import eq
from typing import Any, Callable, Optional, Union, cast

from langchain_core.utils.json import parse_json_markdown
from typing_extensions import override

from langchain.evaluation.schema import StringEvaluator


class JsonValidityEvaluator(StringEvaluator):
    """Evaluate whether the prediction is valid JSON.

    This evaluator checks if the prediction is a valid JSON string. It does not
        require any input or reference.

    Attributes:
        requires_input (bool): Whether this evaluator requires an input
            string. Always False.
        requires_reference (bool): Whether this evaluator requires a
            reference string. Always False.
        evaluation_name (str): The name of the evaluation metric.
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
        input: Optional[str] = None,
        reference: Optional[str] = None,
        **kwargs: Any,
    ) -> dict:
        """Evaluate the prediction string.

        Args:
            prediction (str): The prediction string to evaluate.
            input (str, optional): Not used in this evaluator. Defaults to None.
            reference (str, optional): Not used in this evaluator. Defaults to None.

        Returns:
            dict: A dictionary containing the evaluation score. The score is 1 if
            the prediction is valid JSON, and 0 otherwise.
                If the prediction is not valid JSON, the dictionary also contains
                a "reasoning" field with the error message.

        """
        try:
            parse_json_markdown(prediction, parser=json.loads)
        except Exception as e:
            return {"score": 0, "reasoning": str(e)}
        return {"score": 1}


class JsonEqualityEvaluator(StringEvaluator):
    """Evaluate whether the prediction is equal to the reference after
        parsing both as JSON.

    This evaluator checks if the prediction, after parsing as JSON, is equal
        to the reference,
    which is also parsed as JSON. It does not require an input string.

    Attributes:
        requires_input (bool): Whether this evaluator requires an
            input string. Always False.
        requires_reference (bool): Whether this evaluator requires
            a reference string. Always True.
        evaluation_name (str): The name of the evaluation metric.
            Always "parsed_equality".

    Examples:
        >>> evaluator = JsonEqualityEvaluator()
        >>> evaluator.evaluate_strings('{"a": 1}', reference='{"a": 1}')
        {'score': True}
        >>> evaluator.evaluate_strings('{"a": 1}', reference='{"a": 2}')
        {'score': False}

        >>> evaluator = JsonEqualityEvaluator(operator=lambda x, y: x['a'] == y['a'])
        >>> evaluator.evaluate_strings('{"a": 1}', reference='{"a": 1}')
        {'score': True}
        >>> evaluator.evaluate_strings('{"a": 1}', reference='{"a": 2}')
        {'score': False}

    """

    def __init__(self, operator: Optional[Callable] = None, **_: Any) -> None:
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
    ) -> Union[dict, list, None, float, bool, int, str]:
        if isinstance(string, str):
            return parse_json_markdown(string)
        return string

    @override
    def _evaluate_strings(
        self,
        prediction: str,
        input: Optional[str] = None,
        reference: Optional[str] = None,
        **kwargs: Any,
    ) -> dict:
        """Evaluate the prediction string.

        Args:
            prediction (str): The prediction string to evaluate.
            input (str, optional): Not used in this evaluator.
            reference (str): The reference string to compare against.

        Returns:
            dict: A dictionary containing the evaluation score.
        """
        parsed = self._parse_json(prediction)
        label = self._parse_json(cast(str, reference))
        if isinstance(label, list):
            if not isinstance(parsed, list):
                return {"score": 0}
            parsed = sorted(parsed, key=lambda x: str(x))
            label = sorted(label, key=lambda x: str(x))
        return {"score": self.operator(parsed, label)}
