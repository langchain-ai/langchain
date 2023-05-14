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
    try_each_word: bool = True
    case_insensitive: bool = True

    def get_format_instructions(self):
        return f"Select one of the following options: {', '.join(self.options)}"

    def parse(self, response) -> str:
        print(f"Parsing choice parser response: {response}")
        response = response.strip("\r\n\t ,.:;!?'\"[]()&*^%$#@+=-")
        min_distance, closest_option = self._get_min_dist_and_closest(response)
        print(f"Min distance: {min_distance}, closest option: {closest_option}")
        if min_distance <= self.max_distance or self.max_distance is None:
            return closest_option
        print("Trying each word", self.try_each_word)
        if self.try_each_word:
            print("Trying each word for real")
            # handle answers that add extra words
            for word in response.split():
                word = word.strip("\r\n\t ,.:;!?'\"[]()&*^%$#@+=-")
                min_distance, closest_option = self._get_min_dist_and_closest(word)
                print(
                    f"Word {word}, Min distance: {min_distance}, closest option: {closest_option}"
                )
                if min_distance <= self.max_distance or self.max_distance is None:
                    return closest_option
        raise OutputParserException(
            f"Response '{response}' does not match any options within the min_distance {self.max_distance}"
        )

    def _get_min_dist_and_closest(self, word):
        # do Levenshtein distance matching
        closest_option, min_distance = None, math.inf
        for option in self.options:
            levenshtein_distance = _import_levenshtein_distance()
            if self.case_insensitive:
                word = word.lower()
                option = option.lower()
            distance = levenshtein_distance(word, option)
            if distance <= min_distance:
                min_distance = distance
                closest_option = option
        return min_distance, closest_option

    @property
    def _type(self) -> str:
        return "choice_output_parser"
