"""Evaluators for parsing strings."""
from operator import eq
from typing import Any, Callable, Optional, Union, cast

from langchain.evaluation.schema import StringEvaluator
from langchain.output_parsers.json import parse_json_markdown
import json


class JsonValidityEvaluator(StringEvaluator):
    """Evaluates whether the prediction is valid JSON.

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

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()

    @property
    def requires_input(self) -> bool:
        return False

    @property
    def requires_reference(self) -> bool:
        return False

    @property
    def evaluation_name(self) -> str:
        return "json_validity"

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
            parse_json_markdown(prediction)
            return {"score": 1}
        except Exception as e:
            return {"score": 0, "reasoning": str(e)}


class JsonEqualityEvaluator(StringEvaluator):
    """Evaluates whether the prediction is equal to the reference after
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

    def __init__(self, operator: Optional[Callable] = None, **kwargs: Any) -> None:
        super().__init__()
        self.operator = operator or eq

    @property
    def requires_input(self) -> bool:
        return False

    @property
    def requires_reference(self) -> bool:
        return True

    @property
    def evaluation_name(self) -> str:
        return "json_equality"

    def _parse_json(
        self, string: str
    ) -> Union[dict, list, None, float, bool, int, str]:
        return parse_json_markdown(string)

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


class JsonSchemaEvaluator(StringEvaluator):
    """Evaluates whether the prediction conforms to a given JSON schema.

    This evaluator checks if the prediction, when parsed as JSON, conforms to a
    specified JSON schema. It does not require an input string, but does require
    a reference string which should be the JSON schema.

    Attributes:
        requires_input (bool): Whether this evaluator requires an
            input string. Always False.
        requires_reference (bool): Whether this evaluator requires
            a reference string. Always True.
        evaluation_name (str): The name of the evaluation metric.
            Always "json_schema".

    Examples:
        >>> evaluator = JsonSchemaEvaluator()
        >>> schema = {
        ...     "type": "object",
        ...     "properties": {
        ...         "name": {"type": "string"},
        ...         "age": {"type": "integer"}
        ...     },
        ...     "required": ["name", "age"]
        ... }
        >>> evaluator.evaluate_strings('{"name": "John", "age": 30}', reference=schema)
        {'score': 1}
        >>> evaluator.evaluate_strings('{"name": "John", "age": "30"}', reference=schema)
        {'score': 0, 'reasoning': '30 is not of type \'integer\''}

    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()

    @property
    def requires_input(self) -> bool:
        return False

    @property
    def requires_reference(self) -> bool:
        return True

    @property
    def evaluation_name(self) -> str:
        return "json_schema"

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
            reference (str): The JSON schema to validate against.

        Returns:
            dict: A dictionary containing the evaluation score. The score is 1 if
            the prediction conforms to the schema, and 0 otherwise.
                If the prediction does not conform to the schema, the dictionary
                also contains a "reasoning" field with the error message.
        """
        try:
            import jsonschema
        except ImportError:
            raise ImportError(
                "The jsonschema package is required for the JsonSchemaEvaluator. "
                "You can install it with `pip install jsonschema`."
            )
        if isinstance(reference, str):
            schema_json = parse_json_markdown(reference)
        else:
            schema_json = reference
        try:
            prediction_json = parse_json_markdown(prediction)
            # Validate the prediction against the schema
            jsonschema.validate(instance=prediction_json, schema=schema_json)
            return {"score": 1}
        except jsonschema.exceptions.ValidationError as e:
            return {"score": 0, "reasoning": str(e)}
        except json.JSONDecodeError as e:
            return {"score": 0, "reasoning": f"JSON Decode Error: {str(e)}"}
