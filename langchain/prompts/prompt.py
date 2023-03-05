"""Prompt schema definition."""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from pathlib import Path
from string import Formatter
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Extra, root_validator

from langchain.prompts.base import (
    DEFAULT_FORMATTER_MAPPING,
    BasePromptTemplate,
    PromptValue,
    StringPromptValue,
    check_valid_template,
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


class BaseStringPromptTemplate(BasePromptTemplate, ABC):
    """The prompt template."""

    template_format: str = "f-string"
    """The format of the prompt template. Options are: 'f-string', 'jinja2'."""
    validate_template: bool = True
    """Whether or not to try validating the template."""
    output_parser: Optional[BaseOutputParser] = None
    """How to parse the output of calling an LLM on this formatted prompt."""

    @abstractmethod
    def format(self, **kwargs: Any) -> str:
        """Format the prompt as a string with the inputs.

        Args:
            kwargs: Any arguments to be passed to the prompt template.

        Returns:
            A formatted string.

        Example:

        .. code-block:: python

            prompt.format(variable1="foo")
        """

    def format_prompt(self, **kwargs: Any) -> PromptValue:
        """Format the prompt with the inputs."""
        return StringPromptValue(text=self.format(**kwargs))

    @classmethod
    def from_examples(
        cls,
        examples: List[str],
        suffix: str,
        input_variables: List[str],
        example_separator: str = "\n\n",
        prefix: str = "",
    ) -> BaseStringPromptTemplate:
        """Take examples in list format with prefix and suffix to create a prompt.

        Intended be used as a way to dynamically create a prompt from examples.

        Args:
            examples: List of examples to use in the prompt.
            suffix: String to go after the list of examples. Should generally
                set up the user's input.
            input_variables: A list of variable names the final prompt template
                will expect.
            example_separator: The separator to use in between examples. Defaults
                to two new line characters.
            prefix: String that should go before any examples. Generally includes
                examples. Default to an empty string.

        Returns:
            The final prompt generated.
        """
        template = example_separator.join([prefix, *examples, suffix])
        return cls(input_variables=input_variables, template=template)

    @classmethod
    def from_file(
        cls, template_file: Union[str, Path], input_variables: List[str]
    ) -> BaseStringPromptTemplate:
        """Load a prompt from a file.

        Args:
            template_file: The path to the file containing the prompt template.
            input_variables: A list of variable names the final prompt template
                will expect.
        Returns:
            The prompt loaded from the file.
        """
        with open(str(template_file), "r") as f:
            template = f.read()
        return cls(input_variables=input_variables, template=template)

    @classmethod
    def from_template(cls, template: str) -> BaseStringPromptTemplate:
        """Load a prompt template from a template."""
        input_variables = {
            v for _, v, _, _ in Formatter().parse(template) if v is not None
        }
        return cls(input_variables=list(sorted(input_variables)), template=template)


class StringPromptTemplate(BaseStringPromptTemplate, BaseModel):
    """Schema to represent a prompt for an LLM.

    Example:
        .. code-block:: python

            from langchain import PromptTemplate
            prompt = PromptTemplate(input_variables=["foo"], template="Say {foo}")
    """

    template: str

    @property
    def _prompt_type(self) -> str:
        """Return the prompt type key."""
        return "prompt"

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

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
        kwargs = self._merge_partial_and_user_variables(**kwargs)
        return DEFAULT_FORMATTER_MAPPING[self.template_format](self.template, **kwargs)

    @root_validator()
    def template_is_valid(cls, values: Dict) -> Dict:
        """Check that template and input variables are consistent."""
        if values["validate_template"]:
            all_inputs = values["input_variables"] + list(values["partial_variables"])
            check_valid_template(
                values["template"], values["template_format"], all_inputs
            )
        return values


# For backwards compatibility.
PromptTemplate = StringPromptTemplate
Prompt = PromptTemplate
