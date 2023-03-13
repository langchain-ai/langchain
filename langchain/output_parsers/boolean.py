"""Class to parse output to boolean."""
import re
from typing import Dict, List

from pydantic import Field, root_validator

from langchain.output_parsers.base import BaseOutputParser


class BooleanOutputParser(BaseOutputParser):
    """Class to parse output to boolean."""

    true_values: List[str] = Field(default=["1"])
    false_values: List[str] = Field(default=["0"])

    @root_validator(pre=True)
    def validate_values(cls, values: Dict) -> Dict:
        """Validate that the false/true values are consistent."""
        true_values = values["true_values"]
        false_values = values["false_values"]
        if any([true_value in false_values for true_value in true_values]):
            raise ValueError(
                "The true values and false values lists contain the same value."
            )
        return values

    def parse(self, text: str) -> bool:
        """Output a boolean from a string.

        Allows a LLM's response to be parsed into a boolean.
        For example, if a LLM returns "1", this function will return True.
        Likewise if an LLM returns "The answer is: \n1\n", this function will
        also return True.

        If value errors are common try changing the true and false values to
        rare characters so that it is unlikely the response could contain the
        character unless that was the 'intention'
        (insofar as that makes epistemological sense to say for a non-agential program)
         of the LLM.

        Args:
            text (str): The string to be parsed into a boolean.

        Raises:
            ValueError: If the input string is not a valid boolean.

        Returns:
            bool: The boolean value of the input string.
        """

        input_string = re.sub(
            r"[^" + "".join(self.true_values + self.false_values) + "]", "", text
        )
        if input_string == "":
            raise ValueError(
                "The input string contains neither true nor false characters and"
                " is therefore not a valid boolean."
            )
        # if the string has both true and false values, raise a value error
        if any([true_value in input_string for true_value in self.true_values]) and any(
            [false_value in input_string for false_value in self.false_values]
        ):
            raise ValueError(
                "The input string contains both true and false characters and "
                "therefore is not a valid boolean."
            )
        return input_string in self.true_values
