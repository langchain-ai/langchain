import json
from typing import Any, Callable, Optional, Union

from langchain.evaluation.schema import StringEvaluator
from langchain.output_parsers.json import parse_json_markdown


class JsonEditDistanceEvaluator(StringEvaluator):
    def __init__(
        self,
        string_distance: Optional[Callable[[str, str], float]] = None,
        canonicalize: Optional[Callable[[Any], Any]] = None,
        **kwargs: Any
    ) -> None:
        super().__init__()
        if string_distance is not None:
            self._string_distance = string_distance
        else:
            try:
                from rapidfuzz import distance as rfd  # noqa: F401
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
                x, separators=(",", ":"), sort_keys=True
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

    def _distance(self, a: Any, b: Any):
        return self._string_distance(a, b)

    def _evaluate_strings(
        self,
        prediction: str,
        input: Optional[str] = None,
        reference: Optional[str] = None,
        **kwargs: Any
    ) -> dict:
        parsed = self._canonicalize(self._parse_json(prediction))
        label = self._canonicalize(self._parse_json(reference))
        distance = self._distance(parsed, label)
        return {"score": distance}
