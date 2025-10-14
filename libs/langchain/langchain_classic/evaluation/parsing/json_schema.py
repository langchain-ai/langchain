from typing import Any

from langchain_core.utils.json import parse_json_markdown
from typing_extensions import override

from langchain_classic.evaluation.schema import StringEvaluator


class JsonSchemaEvaluator(StringEvaluator):
    """An evaluator that validates a JSON prediction against a JSON schema reference.

    This evaluator checks if a given JSON prediction conforms to the provided JSON schema.
    If the prediction is valid, the score is True (no errors). Otherwise, the score is False (error occurred).

    Attributes:
        requires_input: Whether the evaluator requires input.
        requires_reference: Whether the evaluator requires reference.
        evaluation_name: The name of the evaluation.

    Examples:
        evaluator = JsonSchemaEvaluator()
        result = evaluator.evaluate_strings(
            prediction='{"name": "John", "age": 30}',
            reference={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"}
                }
            }
        )
        assert result["score"] is not None

    """  # noqa: E501

    def __init__(self, **_: Any) -> None:
        """Initializes the JsonSchemaEvaluator.

        Raises:
            ImportError: If the jsonschema package is not installed.
        """
        super().__init__()
        try:
            import jsonschema  # noqa: F401
        except ImportError as e:
            msg = (
                "The JsonSchemaEvaluator requires the jsonschema package."
                " Please install it with `pip install jsonschema`."
            )
            raise ImportError(msg) from e

    @property
    def requires_input(self) -> bool:
        """Returns whether the evaluator requires input."""
        return False

    @property
    def requires_reference(self) -> bool:
        """Returns whether the evaluator requires reference."""
        return True

    @property
    def evaluation_name(self) -> str:
        """Returns the name of the evaluation."""
        return "json_schema_validation"

    def _parse_json(self, node: Any) -> dict | list | None | float | bool | int | str:
        if isinstance(node, str):
            return parse_json_markdown(node)
        if hasattr(node, "model_json_schema") and callable(node.model_json_schema):
            # Pydantic v2 model
            return node.model_json_schema()
        if hasattr(node, "schema") and callable(node.schema):
            # Pydantic v1 model
            return node.schema()
        return node

    def _validate(self, prediction: Any, schema: Any) -> dict:
        from jsonschema import ValidationError, validate

        try:
            validate(instance=prediction, schema=schema)
        except ValidationError as e:
            return {"score": False, "reasoning": repr(e)}
        return {"score": True}

    @override
    def _evaluate_strings(
        self,
        prediction: str | Any,
        input: str | Any = None,
        reference: str | Any = None,
        **kwargs: Any,
    ) -> dict:
        parsed_prediction = self._parse_json(prediction)
        schema = self._parse_json(reference)
        return self._validate(parsed_prediction, schema)
