from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Extra, root_validator

from langchain.prompts.base import (
    DEFAULT_FORMATTER_MAPPING,
    BasePromptTemplate,
    check_valid_template,
)
from langchain.prompts.example_selector.base import BaseExampleSelector
from langchain.prompts.prompt import PromptTemplate


class FewShotPromptTemplate(BasePromptTemplate, BaseModel):
    """Prompt template that contains few shot examples."""

    examples: Optional[List[dict]] = None
    example_prompt: PromptTemplate
    suffix: str
    input_variables: List[str]
    example_separator: str = "\n\n"
    prefix: str = ""
    template_format: str = "f-string"
    example_selector: Optional[BaseExampleSelector] = None

    @root_validator(pre=True)
    def check_examples_and_selector(cls, values: Dict) -> Dict:
        if values["examples"] and values["example_selector"]:
            raise ValueError(
                "Only one of 'examples' and 'example_selector' should be provided"
            )

        if values["examples"] is None and values["example_selector"] is None:
            raise ValueError(
                "One of 'examples' and 'example_selector' should be provided"
            )

        return values

    @root_validator()
    def template_is_valid(cls, values: Dict) -> Dict:
        """Check that prefix, suffix and input variables are consistent."""
        check_valid_template(
            values["prefix"] + values["suffix"],
            values["template_format"],
            values["input_variables"],
        )
        return values

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    def _get_examples(self, **kwargs: Any) -> List[dict]:
        if self.examples is not None:
            return self.examples
        elif self.example_selector is not None:
            return self.example_selector.select_examples(kwargs)
        else:
            raise ValueError

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
        # Get the examples to use.
        examples = self._get_examples(**kwargs)
        # Format the examples.
        example_strings = [
            self.example_prompt.format(**example) for example in examples
        ]
        # Create the overall template.
        pieces = [self.prefix, *example_strings, self.suffix]
        template = self.example_separator.join([piece for piece in pieces if piece])
        # Format the template with the input variables.
        return DEFAULT_FORMATTER_MAPPING[self.template_format](template, **kwargs)
