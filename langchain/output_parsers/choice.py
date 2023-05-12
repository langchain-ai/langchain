import math
from typing import Any, List

from langchain.schema import BaseOutputParser, OutputParserException


def _import_levenshtein_distance() -> Any:
    """Import Levenshtein if available, otherwise raise error."""
    try:
        from Levenshtein import distance
    except ImportError:
        raise ValueError(
            "Could not import Levenshtein. Please install it with `pip install python-Levenshtein`."
        )
    return distance


class ChoiceOutputParser(BaseOutputParser[str]):
    """Parses one of a set of options."""

    options: List[str] = []
    max_distance: int = 0

    def get_format_instructions(self):
        return f"Select one of the following options: {', '.join(self.options)}"

    def parse(self, response):
        response = response.strip()
        min_distance, closest_option = self._get_min_dist_and_closest(response)
        if min_distance <= self.max_distance or self.max_distance is None:
            return closest_option
        else:
            # handle answers that add extra words
            for word in response.split():
                min_distance, closest_option = self._get_min_dist_and_closest(word)
                if min_distance <= self.max_distance:
                    return closest_option
        raise OutputParserException(
            f"Response '{response}' does not match any options within the min_distance {self.max_distance}"
        )

    def _get_min_dist_and_closest(self, word):
        # do Levenshtein distance matching
        closest_option, min_distance = None, math.inf
        for option in self.options:
            levenshtein_distance = _import_levenshtein_distance()
            distance = levenshtein_distance(word, option)
            if distance <= min_distance:
                min_distance = distance
                closest_option = option
        return min_distance, closest_option

    @property
    def _type(self) -> str:
        return "choice_output_parser"
