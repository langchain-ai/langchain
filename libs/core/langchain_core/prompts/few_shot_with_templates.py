"""Prompt template that contains few shot examples."""

from pathlib import Path
from typing import Any

from pydantic import ConfigDict, model_validator
from typing_extensions import Self

from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.prompts.string import (
    DEFAULT_FORMATTER_MAPPING,
    PromptTemplateFormat,
    StringPromptTemplate,
)


class FewShotPromptWithTemplates(StringPromptTemplate):
    """Prompt template that contains few shot examples."""

    examples: list[dict] | None = None
    """Examples to format into the prompt.
    Either this or example_selector should be provided."""

    example_selector: Any = None
    """ExampleSelector to choose the examples to format into the prompt.
    Either this or examples should be provided."""

    example_prompt: PromptTemplate
    """PromptTemplate used to format an individual example."""

    suffix: StringPromptTemplate
    """A PromptTemplate to put after the examples."""

    example_separator: str = "\n\n"
    """String separator used to join the prefix, the examples, and suffix."""

    prefix: StringPromptTemplate | None = None
    """A PromptTemplate to put before the examples."""

    template_format: PromptTemplateFormat = "f-string"
    """The format of the prompt template.
    Options are: 'f-string', 'jinja2', 'mustache'."""

    validate_template: bool = False
    """Whether or not to try validating the template."""

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        """Get the namespace of the LangChain object.

        Returns:
            `["langchain", "prompts", "few_shot_with_templates"]`
        """
        return ["langchain", "prompts", "few_shot_with_templates"]

    @model_validator(mode="before")
    @classmethod
    def check_examples_and_selector(cls, values: dict) -> Any:
        """Check that one and only one of examples/example_selector are provided."""
        examples = values.get("examples")
        example_selector = values.get("example_selector")
        if examples and example_selector:
            msg = "Only one of 'examples' and 'example_selector' should be provided"
            raise ValueError(msg)

        if examples is None and example_selector is None:
            msg = "One of 'examples' and 'example_selector' should be provided"
            raise ValueError(msg)

        return values

    @model_validator(mode="after")
    def template_is_valid(self) -> Self:
        """Check that prefix, suffix, and input variables are consistent."""
        if self.validate_template:
            input_variables = self.input_variables
            expected_input_variables = set(self.suffix.input_variables)
            expected_input_variables |= set(self.partial_variables)
            if self.prefix is not None:
                expected_input_variables |= set(self.prefix.input_variables)
            missing_vars = expected_input_variables.difference(input_variables)
            if missing_vars:
                msg = (
                    f"Got input_variables={input_variables}, but based on "
                    f"prefix/suffix expected {expected_input_variables}"
                )
                raise ValueError(msg)
        else:
            self.input_variables = sorted(
                set(self.suffix.input_variables)
                | set(self.prefix.input_variables if self.prefix else [])
                - set(self.partial_variables)
            )
        return self

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    def _get_examples(self, **kwargs: Any) -> list[dict]:
        if self.examples is not None:
            return self.examples
        if self.example_selector is not None:
            return self.example_selector.select_examples(kwargs)
        raise ValueError

    async def _aget_examples(self, **kwargs: Any) -> list[dict]:
        if self.examples is not None:
            return self.examples
        if self.example_selector is not None:
            return await self.example_selector.aselect_examples(kwargs)
        raise ValueError

    def format(self, **kwargs: Any) -> str:
        """Format the prompt with the inputs.

        Args:
            **kwargs: Any arguments to be passed to the prompt template.

        Returns:
            A formatted string.

        Example:
        ```python
        prompt.format(variable1="foo")
        ```
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
            for k in prefix_kwargs:
                kwargs.pop(k)
            prefix = self.prefix.format(**prefix_kwargs)

        # Create the overall suffix
        suffix_kwargs = {
            k: v for k, v in kwargs.items() if k in self.suffix.input_variables
        }
        for k in suffix_kwargs:
            kwargs.pop(k)
        suffix = self.suffix.format(
            **suffix_kwargs,
        )

        pieces = [prefix, *example_strings, suffix]
        template = self.example_separator.join([piece for piece in pieces if piece])
        # Format the template with the input variables.
        return DEFAULT_FORMATTER_MAPPING[self.template_format](template, **kwargs)

    async def aformat(self, **kwargs: Any) -> str:
        """Async format the prompt with the inputs.

        Args:
            **kwargs: Any arguments to be passed to the prompt template.

        Returns:
            A formatted string.
        """
        kwargs = self._merge_partial_and_user_variables(**kwargs)
        # Get the examples to use.
        examples = await self._aget_examples(**kwargs)
        # Format the examples.
        example_strings = [
            # We can use the sync method here as PromptTemplate doesn't block
            self.example_prompt.format(**example)
            for example in examples
        ]
        # Create the overall prefix.
        if self.prefix is None:
            prefix = ""
        else:
            prefix_kwargs = {
                k: v for k, v in kwargs.items() if k in self.prefix.input_variables
            }
            for k in prefix_kwargs:
                kwargs.pop(k)
            prefix = await self.prefix.aformat(**prefix_kwargs)

        # Create the overall suffix
        suffix_kwargs = {
            k: v for k, v in kwargs.items() if k in self.suffix.input_variables
        }
        for k in suffix_kwargs:
            kwargs.pop(k)
        suffix = await self.suffix.aformat(
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

    def save(self, file_path: Path | str) -> None:
        """Save the prompt to a file.

        Args:
            file_path: The path to save the prompt to.

        Raises:
            ValueError: If example_selector is provided.
        """
        if self.example_selector:
            msg = "Saving an example selector is not currently supported"
            raise ValueError(msg)
        return super().save(file_path)
