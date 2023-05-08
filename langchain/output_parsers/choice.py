import math
from typing import Any, List

import Levenshtein

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

        # do Levenshtein distance matching
        closest_option, min_distance = None, math.inf
        for option in self.options:
            levenshtein_distance = _import_levenshtein_distance()
            distance = levenshtein_distance(response, option)
            if distance <= min_distance:
                min_distance = distance
                closest_option = option

        print(f"min_distance: {min_distance}")
        print(f"closest_option: {closest_option}")
        if min_distance <= self.max_distance or self.max_distance is None:
            return closest_option
        else:
            raise OutputParserException(
                f"Response '{response}' does not match any options within the min_distance {self.max_distance}"
            )
