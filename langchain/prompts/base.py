"""BasePrompt schema definition."""
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import yaml
from pydantic import BaseModel, Extra, root_validator

from langchain.formatting import formatter


def jinja2_formatter(template: str, **kwargs: Any) -> str:
    """Format a template using jinja2."""
    try:
        from jinja2 import Template
    except ImportError:
        raise ValueError(
            "jinja2 not installed, which is needed to use the jinja2_formatter. "
            "Please install it with `pip install jinja2`."
        )

    return Template(template).render(**kwargs)


DEFAULT_FORMATTER_MAPPING: Dict[str, Callable] = {
    "f-string": formatter.format,
    "jinja2": jinja2_formatter,
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


class BaseOutputParser(ABC):
    """Class to parse the output of an LLM call."""

    @abstractmethod
    def parse(self, text: str) -> Union[str, List[str], Dict[str, str]]:
        """Parse the output of an LLM call."""


class ListOutputParser(ABC):
    """Class to parse the output of an LLM call to a list."""

    @abstractmethod
    def parse(self, text: str) -> List[str]:
        """Parse the output of an LLM call."""


class BasePromptTemplate(BaseModel, ABC):
    """Base prompt should expose the format method, returning a prompt."""

    input_variables: List[str]
    """A list of the names of the variables the prompt template expects."""
    output_parser: Optional[BaseOutputParser] = None
    """How to parse the output of calling an LLM on this formatted prompt."""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

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

    def _prompt_dict(self) -> Dict:
        """Return a dictionary of the prompt."""
        return self.dict()

    def save(self, file_path: Union[Path, str]) -> None:
        """Save the prompt.

        Args:
            file_path: Path to directory to save prompt to.

        Example:
        .. code-block:: python

            prompt.save(file_path="path/prompt.yaml")
        """
        # Convert file to Path object.
        if isinstance(file_path, str):
            save_path = Path(file_path)
        else:
            save_path = file_path

        directory_path = save_path.parent
        directory_path.mkdir(parents=True, exist_ok=True)

        # Fetch dictionary to save
        prompt_dict = self._prompt_dict()

        if save_path.suffix == ".json":
            with open(file_path, "w") as f:
                json.dump(prompt_dict, f, indent=4)
        elif save_path.suffix == ".yaml":
            with open(file_path, "w") as f:
                yaml.dump(prompt_dict, f, default_flow_style=False)
        else:
            raise ValueError(f"{save_path} must be json or yaml")
