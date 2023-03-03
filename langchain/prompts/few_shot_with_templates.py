"""Prompt template that contains few shot examples."""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Extra, root_validator

from langchain.prompts.base import DEFAULT_FORMATTER_MAPPING, BasePromptTemplate
from langchain.prompts.example_selector.base import BaseExampleSelector
from langchain.prompts.prompt import PromptTemplate


class FewShotPromptWithTemplates(BasePromptTemplate, BaseModel):
    """Prompt template that contains few shot examples."""

    examples: Optional[List[dict]] = None
    """Examples to format into the prompt.
    Either this or example_selector should be provided."""

    example_selector: Optional[BaseExampleSelector] = None
    """ExampleSelector to choose the examples to format into the prompt.
    Either this or examples should be provided."""

    example_prompt: PromptTemplate
    """PromptTemplate used to format an individual example."""

    suffix: BasePromptTemplate
    """A PromptTemplate to put after the examples."""

    input_variables: List[str]
    """A list of the names of the variables the prompt template expects."""

    example_separator: str = "\n\n"
    """String separator used to join the prefix, the examples, and suffix."""

    prefix: Optional[BasePromptTemplate] = None
    """A PromptTemplate to put before the examples."""

    template_format: str = "f-string"
    """The format of the prompt template. Options are: 'f-string', 'jinja2'."""

    validate_template: bool = True
    """Whether or not to try validating the template."""

    @root_validator(pre=True)
    def check_examples_and_selector(cls, values: Dict) -> Dict:
        """Check that one and only one of examples/example_selector are provided."""
        examples = values.get("examples", None)
        example_selector = values.get("example_selector", None)
        if examples and example_selector:
            raise ValueError(
                "Only one of 'examples' and 'example_selector' should be provided"
            )

        if examples is None and example_selector is None:
            raise ValueError(
                "One of 'examples' and 'example_selector' should be provided"
            )

        return values

    @root_validator()
    def template_is_valid(cls, values: Dict) -> Dict:
        """Check that prefix, suffix and input variables are consistent."""
        if values["validate_template"]:
            input_variables = values["input_variables"]
            expected_input_variables = set(values["suffix"].input_variables)
            expected_input_variables |= set(values["partial_variables"])
            if values["prefix"] is not None:
                expected_input_variables |= set(values["prefix"].input_variables)
            missing_vars = expected_input_variables.difference(input_variables)
            if missing_vars:
                raise ValueError(
                    f"Got input_variables={input_variables}, but based on "
                    f"prefix/suffix expected {expected_input_variables}"
                )
        return values

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

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
        kwargs = self._merge_partial_and_user_variables(**kwargs)
        # Get the examples to use.
        examples = self._get_examples(**kwargs)
        # Format the examples.
        example_strings = [
            self.example_prompt.format(**example) for example in examples
        ]
        # Create the overall prefix.
        if self.prefix is None:
            prefix = ""
        else:
            prefix_kwargs = {
                k: v for k, v in kwargs.items() if k in self.prefix.input_variables
            }
            for k in prefix_kwargs.keys():
                kwargs.pop(k)
            prefix = self.prefix.format(**prefix_kwargs)

        # Create the overall suffix
        suffix_kwargs = {
            k: v for k, v in kwargs.items() if k in self.suffix.input_variables
        }
        for k in suffix_kwargs.keys():
            kwargs.pop(k)
        suffix = self.suffix.format(
            **suffix_kwargs,
        )

        pieces = [prefix, *example_strings, suffix]
        template = self.example_separator.join([piece for piece in pieces if piece])
        # Format the template with the input variables.
        return DEFAULT_FORMATTER_MAPPING[self.template_format](template, **kwargs)

    @property
    def _prompt_type(self) -> str:
        """Return the prompt type key."""
        return "few_shot_with_templates"

    def dict(self, **kwargs: Any) -> Dict:
        """Return a dictionary of the prompt."""
        if self.example_selector:
            raise ValueError("Saving an example selector is not currently supported")
        return super().dict(**kwargs)
