"""BasePrompt schema definition."""
from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Union

import yaml
from pydantic import BaseModel, Extra, Field, root_validator

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
    except KeyError as e:
        raise ValueError(
            "Invalid prompt schema; check for mismatched or missing input parameters. "
            + str(e)
        )


class BaseOutputParser(BaseModel, ABC):
    """Class to parse the output of an LLM call."""

    @abstractmethod
    def parse(self, text: str) -> Union[str, List[str], Dict[str, str]]:
        """Parse the output of an LLM call."""

    @property
    def _type(self) -> str:
        """Return the type key."""
        raise NotImplementedError

    def dict(self, **kwargs: Any) -> Dict:
        """Return dictionary representation of output parser."""
        output_parser_dict = super().dict()
        output_parser_dict["_type"] = self._type
        return output_parser_dict


class ListOutputParser(BaseOutputParser):
    """Class to parse the output of an LLM call to a list."""

    @abstractmethod
    def parse(self, text: str) -> List[str]:
        """Parse the output of an LLM call."""


class CommaSeparatedListOutputParser(ListOutputParser):
    """Parse out comma separated lists."""

    def parse(self, text: str) -> List[str]:
        """Parse the output of an LLM call."""
        return text.strip().split(", ")


class RegexParser(BaseOutputParser, BaseModel):
    """Class to parse the output into a dictionary."""

    regex: str
    output_keys: List[str]
    default_output_key: Optional[str] = None

    @property
    def _type(self) -> str:
        """Return the type key."""
        return "regex_parser"

    def parse(self, text: str) -> Dict[str, str]:
        """Parse the output of an LLM call."""
        match = re.search(self.regex, text)
        if match:
            return {key: match.group(i + 1) for i, key in enumerate(self.output_keys)}
        else:
            if self.default_output_key is None:
                raise ValueError(f"Could not parse output: {text}")
            else:
                return {
                    key: text if key == self.default_output_key else ""
                    for key in self.output_keys
                }


class BasePromptTemplate(BaseModel, ABC):
    """Base prompt should expose the format method, returning a prompt."""

    input_variables: List[str]
    """A list of the names of the variables the prompt template expects."""
    output_parser: Optional[BaseOutputParser] = None
    """How to parse the output of calling an LLM on this formatted prompt."""
    partial_variables: Mapping[str, Union[str, Callable[[], str]]] = Field(
        default_factory=dict
    )

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @root_validator()
    def validate_variable_names(cls, values: Dict) -> Dict:
        """Validate variable names do not include restricted names."""
        if "stop" in values["input_variables"]:
            raise ValueError(
                "Cannot have an input variable named 'stop', as it is used internally,"
                " please rename."
            )
        if "stop" in values["partial_variables"]:
            raise ValueError(
                "Cannot have an partial variable named 'stop', as it is used "
                "internally, please rename."
            )

        overall = set(values["input_variables"]).intersection(
            values["partial_variables"]
        )
        if overall:
            raise ValueError(
                f"Found overlapping input and partial variables: {overall}"
            )
        return values

    def partial(self, **kwargs: Union[str, Callable[[], str]]) -> BasePromptTemplate:
        """Return a partial of the prompt template."""
        prompt_dict = self.__dict__.copy()
        prompt_dict["input_variables"] = list(
            set(self.input_variables).difference(kwargs)
        )
        prompt_dict["partial_variables"] = {**self.partial_variables, **kwargs}
        return type(self)(**prompt_dict)

    def _merge_partial_and_user_variables(self, **kwargs: Any) -> Dict[str, Any]:
        # Get partial params:
        partial_kwargs = {
            k: v if isinstance(v, str) else v()
            for k, v in self.partial_variables.items()
        }
        return {**partial_kwargs, **kwargs}

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

    @property
    @abstractmethod
    def _prompt_type(self) -> str:
        """Return the prompt type key."""

    def dict(self, **kwargs: Any) -> Dict:
        """Return dictionary representation of prompt."""
        prompt_dict = super().dict(**kwargs)
        prompt_dict["_type"] = self._prompt_type
        return prompt_dict

    def save(self, file_path: Union[Path, str]) -> None:
        """Save the prompt.

        Args:
            file_path: Path to directory to save prompt to.

        Example:
        .. code-block:: python

            prompt.save(file_path="path/prompt.yaml")
        """
        if self.partial_variables:
            raise ValueError("Cannot save prompt with partial variables.")
        # Convert file to Path object.
        if isinstance(file_path, str):
            save_path = Path(file_path)
        else:
            save_path = file_path

        directory_path = save_path.parent
        directory_path.mkdir(parents=True, exist_ok=True)

        # Fetch dictionary to save
        prompt_dict = self.dict()

        if save_path.suffix == ".json":
            with open(file_path, "w") as f:
                json.dump(prompt_dict, f, indent=4)
        elif save_path.suffix == ".yaml":
            with open(file_path, "w") as f:
                yaml.dump(prompt_dict, f, default_flow_style=False)
        else:
            raise ValueError(f"{save_path} must be json or yaml")
