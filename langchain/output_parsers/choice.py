from typing import List

import Levenshtein

from langchain.schema import BaseOutputParser, OutputParserException


class ChoiceOutputParser(BaseOutputParser[str]):
    """Parses one of a set of options."""

    options: List[str] = []
    min_distance: int = None

    def get_format_instructions(self):
        return f"Select one of the following options: {', '.join(self.options)}"

    def parse(self, response):
        response = response.strip()

        if self.min_distance is None:
            # do fuzzy matching
            closest_option, min_distance = None, float("inf")
            for option in self.options:
                distance = Levenshtein.distance(response, option)
                if distance < min_distance:
                    min_distance = distance
                    closest_option = option

            if min_distance <= self.min_distance:
                return closest_option
            else:
                raise OutputParserException(
                    f"Response '{response}' does not match any options within the min_distance {self.min_distance}"
                )

        if response not in self.options:
            raise OutputParserException(
                f"Response '{response}' not in options {self.options}"
            )
        return response
