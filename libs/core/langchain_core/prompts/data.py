"""Image prompt template for a multimodal model."""

from typing import Any, cast

from pydantic import Field

from langchain_core.messages import DataContentBlock
from langchain_core.prompt_values import DataPromptValue, PromptValue
from langchain_core.prompts.base import BasePromptTemplate
from langchain_core.prompts.string import (
    DEFAULT_FORMATTER_MAPPING,
    PromptTemplateFormat,
)
from langchain_core.runnables import run_in_executor


class DataPromptTemplate(BasePromptTemplate[DataContentBlock]):
    """Prompt template for a multi-modal model."""

    template: dict = Field(default_factory=dict)
    """Template for the prompt."""
    template_format: PromptTemplateFormat = "f-string"
    """The format of the prompt template.
    Options are: 'f-string', 'mustache', 'jinja2'."""

    def __init__(self, **kwargs: Any) -> None:
        """Create a prompt template for multi-modal data."""
        if "input_variables" not in kwargs:
            kwargs["input_variables"] = []

        overlap = set(kwargs["input_variables"]) & {
            "source",
            "source_type",
            "mime_type",
            "metadata",
        }
        if overlap:
            msg = (
                "input_variables for the template cannot contain"
                " any of 'source', 'source_type', 'mime_type', or 'metadata'."
                f" Found: {overlap}"
            )
            raise ValueError(msg)
        super().__init__(**kwargs)

    @property
    def _prompt_type(self) -> str:
        """Return the prompt type key."""
        return "data-prompt"

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "prompts", "data"]

    def format_prompt(self, **kwargs: Any) -> PromptValue:
        """Format the prompt with the inputs.

        Args:
            kwargs: Any arguments to be passed to the prompt template.

        Returns:
            A formatted string.
        """
        return DataPromptValue(content_block=self.format(**kwargs))

    async def aformat_prompt(self, **kwargs: Any) -> PromptValue:
        """Async format the prompt with the inputs.

        Args:
            kwargs: Any arguments to be passed to the prompt template.

        Returns:
            A formatted string.
        """
        return DataPromptValue(content_block=await self.aformat(**kwargs))

    def format(
        self,
        **kwargs: Any,
    ) -> DataContentBlock:
        """Format the prompt with the inputs.

        Args:
            kwargs: Any arguments to be passed to the prompt template.

        Returns:
            A formatted string.

        Raises:
            ValueError: If the url is not provided.
            ValueError: If the url is not a string.

        Example:

            .. code-block:: python

                prompt.format(variable1="foo")
        """
        formatted = {}
        for k, v in self.template.items():
            if isinstance(v, str):
                formatted[k] = DEFAULT_FORMATTER_MAPPING[self.template_format](
                    v, **kwargs
                )
            else:
                formatted[k] = v

        block = {}
        for k in ["type", "source_type", "source", "mime_type", "metadata"]:
            value = kwargs.get(k) or formatted.get(k)
            if value:
                block[k] = value

        for required_field in ["source", "source_type"]:
            if required_field not in block:
                msg = f"Missing required field: {required_field}"
                raise ValueError(msg)

        return cast("DataContentBlock", block)

    async def aformat(self, **kwargs: Any) -> DataContentBlock:
        """Async format the prompt with the inputs.

        Args:
            kwargs: Any arguments to be passed to the prompt template.

        Returns:
            A formatted string.

        Raises:
            ValueError: If the path or url is not a string.
        """
        return await run_in_executor(None, self.format, **kwargs)

    def pretty_repr(self, html: bool = False) -> str:
        """Return a pretty representation of the prompt.

        Args:
            html: Whether to return an html formatted string.

        Returns:
            A pretty representation of the prompt.
        """
        raise NotImplementedError
