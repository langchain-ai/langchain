"""BasePrompt schema definition."""
from abc import ABC, abstractmethod
from typing import Any, Dict, List

from pydantic import BaseModel, root_validator

from langchain.formatting import formatter

DEFAULT_FORMATTER_MAPPING = {
    "f-string": formatter.format,
}


def check_valid_template(
    template: str, template_format: str, input_variables: List[str]
) -> None:
    """Check that template string is valid."""
    if template_format not in DEFAULT_FORMATTER_MAPPING:
        valid_formats = list(DEFAULT_FORMATTER_MAPPING)
        raise ValueError(
            f"Invalid template format. Got `{template_format}`;"
            f" should be one of {valid_formats}"
        )
    dummy_inputs = {input_variable: "foo" for input_variable in input_variables}
    try:
        formatter_func = DEFAULT_FORMATTER_MAPPING[template_format]
        formatter_func(template, **dummy_inputs)
    except KeyError:
        raise ValueError("Invalid prompt schema.")


class BasePromptTemplate(BaseModel, ABC):
    """Base prompt should expose the format method, returning a prompt."""

    input_variables: List[str]
    """A list of the names of the variables the prompt template expects."""

    @root_validator()
    def validate_variable_names(cls, values: Dict) -> Dict:
        """Validate variable names do not restricted names."""
        if "stop" in values["input_variables"]:
            raise ValueError(
                "Cannot have an input variable named 'stop', as it is used internally,"
                " please rename."
            )
        return values

    @abstractmethod
    def format(self, **kwargs: Any) -> str:
        """Format the prompt with the inputs.

        Args:
            kwargs: Any arguments to be passed to the prompt template.

        Returns:
            A formatted string.

        Example:

        .. code-block:: python

            prompt.format(variable1="foo")
        """
