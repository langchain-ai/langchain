from typing import Any, Optional, Union

from langchain.evaluation.schema import StringEvaluator
from langchain.output_parsers.json import parse_json_markdown


class JsonSchemaEvaluator(StringEvaluator):
    """
    An evaluator that validates a JSON prediction against a JSON schema reference.

    This evaluator checks if a given JSON prediction conforms to the provided JSON schema.
    If the prediction is valid, the score is 0 (no errors). Otherwise, the score is 1 (error occurred).

    Parameters
    ----------
    **kwargs : Any
        Additional keyword arguments.

    Attributes
    ----------

    _error_message : Optional[str]
        Stores the error message in case of a schema validation failure.

    Examples
    --------
    >>> evaluator = JsonSchemaEvaluator()
    >>> result = evaluator.evaluate_strings(
                        prediction='{"name": "John", "age": 30}',
                        reference={
                                    "type": "object",
                                    "properties": {
                                        "name": {"type": "string"},
                                        "age": {"type": "integer"}
                                    }
                        }
                )
    >>> assert result["score"] is not None

    """  # noqa: E501

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        try:
            import jsonschema  # noqa: F401
        except ImportError:
            raise ImportError(
                "The JsonSchemaEvaluator requires the jsonschema package."
                " Please install it with `pip install jsonschema`."
            )

    @property
    def requires_input(self) -> bool:
        return False

    @property
    def requires_reference(self) -> bool:
        return True

    @property
    def evaluation_name(self) -> str:
        return "json_schema_validation"

    def _parse_json(self, node: Any) -> Union[dict, list, None, float, bool, int, str]:
        if isinstance(node, str):
            return parse_json_markdown(node)
        elif hasattr(node, "schema") and callable(getattr(node, "schema")):
            # Pydantic model
            return getattr(node, "schema")()
        return node

    def _validate(self, prediction: Any, schema: Any) -> bool:
        """
        Validate the prediction against the provided JSON schema.

        Parameters
        ----------
        prediction : Any
            The parsed prediction JSON.
        schema : Any
            The parsed JSON schema.

        Returns
        -------
        bool
            True if the prediction adheres to the schema, False otherwise.
        """
        from jsonschema import ValidationError, validate  # noqa: F811

        try:
            validate(instance=prediction, schema=schema)
            return {
                "score": True,
            }
        except ValidationError as e:
            return {"score": False, "reasoning": repr(e)}

    def _evaluate_strings(
        self,
        prediction: str,
        input: Optional[str] = None,
        reference: Optional[str] = None,
        **kwargs: Any
    ) -> dict:
        parsed_prediction = self._parse_json(prediction)
        schema = self._parse_json(reference)
        return self._validate(parsed_prediction, schema)
