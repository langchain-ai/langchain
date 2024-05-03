import json
from typing import Any, Callable, Optional, Union

from langchain_core.utils.json import parse_json_markdown

from langchain.evaluation.schema import StringEvaluator


class JsonEditDistanceEvaluator(StringEvaluator):
    """
    An evaluator that calculates the edit distance between JSON strings.

    This evaluator computes a normalized Damerau-Levenshtein distance between two JSON strings
    after parsing them and converting them to a canonical format (i.e., whitespace and key order are normalized).
    It can be customized with alternative distance and canonicalization functions.

    Args:
        string_distance (Optional[Callable[[str, str], float]]): A callable that computes the distance between two strings.
            If not provided, a Damerau-Levenshtein distance from the `rapidfuzz` package will be used.
        canonicalize (Optional[Callable[[Any], Any]]): A callable that converts a parsed JSON object into its canonical string form.
            If not provided, the default behavior is to serialize the JSON with sorted keys and no extra whitespace.
        **kwargs (Any): Additional keyword arguments.

    Attributes:
        _string_distance (Callable[[str, str], float]): The internal distance computation function.
        _canonicalize (Callable[[Any], Any]): The internal canonicalization function.

    Examples:
        >>> evaluator = JsonEditDistanceEvaluator()
        >>> result = evaluator.evaluate_strings(prediction='{"a": 1, "b": 2}', reference='{"a": 1, "b": 3}')
        >>> assert result["score"] is not None

    Raises:
        ImportError: If `rapidfuzz` is not installed and no alternative `string_distance` function is provided.

    """  # noqa: E501

    def __init__(
        self,
        string_distance: Optional[Callable[[str, str], float]] = None,
        canonicalize: Optional[Callable[[Any], Any]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if string_distance is not None:
            self._string_distance = string_distance
        else:
            try:
                from rapidfuzz import distance as rfd
            except ImportError:
                raise ImportError(
                    "The default string_distance operator for the "
                    " JsonEditDistanceEvaluator requires installation of "
                    "the rapidfuzz package. "
                    "Please install it with `pip install rapidfuzz`."
                )
            self._string_distance = rfd.DamerauLevenshtein.normalized_distance
        if canonicalize is not None:
            self._canonicalize = canonicalize
        else:
            self._canonicalize = lambda x: json.dumps(
                x,
                separators=(",", ":"),
                sort_keys=True,  # eliminate whitespace
            )

    @property
    def requires_input(self) -> bool:
        return False

    @property
    def requires_reference(self) -> bool:
        return True

    @property
    def evaluation_name(self) -> str:
        return "json_edit_distance"

    def _parse_json(self, node: Any) -> Union[dict, list, None, float, bool, int, str]:
        if isinstance(node, str):
            return parse_json_markdown(node)
        return node

    def _evaluate_strings(
        self,
        prediction: str,
        input: Optional[str] = None,
        reference: Optional[str] = None,
        **kwargs: Any,
    ) -> dict:
        parsed = self._canonicalize(self._parse_json(prediction))
        label = self._canonicalize(self._parse_json(reference))
        distance = self._string_distance(parsed, label)
        return {"score": distance}
