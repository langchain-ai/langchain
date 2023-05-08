from typing import List

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

        if self.max_distance is None:
            # do fuzzy matching
            closest_option, min_distance = None, float("inf")
            for option in self.options:
                if distance < min_distance:
                    min_distance = distance
                    closest_option = option

            if min_distance <= self.max_distance:
                return closest_option
            else:
                raise OutputParserException(
                    f"Response '{response}' does not match any options within the min_distance {self.max_distance}"
                )

        if response not in self.options:
            levenshtein_distance = _import_levenshtein_distance()
            distance = levenshtein_distance(response, option)
            raise OutputParserException(
                f"Response '{response}' not in options {self.options}"
            )
        return response
