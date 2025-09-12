"""Image prompt template for a multimodal model."""

from typing import Any

from pydantic import Field

from langchain_core.prompt_values import ImagePromptValue, ImageURL, PromptValue
from langchain_core.prompts.base import BasePromptTemplate
from langchain_core.prompts.string import (
    DEFAULT_FORMATTER_MAPPING,
    PromptTemplateFormat,
)
from langchain_core.runnables import run_in_executor


class ImagePromptTemplate(BasePromptTemplate[ImageURL]):
    """Image prompt template for a multimodal model."""

    template: dict = Field(default_factory=dict)
    """Template for the prompt."""
    template_format: PromptTemplateFormat = "f-string"
    """The format of the prompt template.
    Options are: 'f-string', 'mustache', 'jinja2'."""

    def __init__(self, **kwargs: Any) -> None:
        """Create an image prompt template.

        Raises:
            ValueError: If the input variables contain ``'url'``, ``'path'``, or
                ``'detail'``.
        """
        if "input_variables" not in kwargs:
            kwargs["input_variables"] = []

        overlap = set(kwargs["input_variables"]) & {"url", "path", "detail"}
        if overlap:
            msg = (
                "input_variables for the image template cannot contain"
                " any of 'url', 'path', or 'detail'."
                f" Found: {overlap}"
            )
            raise ValueError(msg)
        super().__init__(**kwargs)

    @property
    def _prompt_type(self) -> str:
        """Return the prompt type key."""
        return "image-prompt"

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        """Get the namespace of the langchain object.

        Returns:
            ``["langchain", "prompts", "image"]``
        """
        return ["langchain", "prompts", "image"]

    def format_prompt(self, **kwargs: Any) -> PromptValue:
        """Format the prompt with the inputs.

        Args:
            kwargs: Any arguments to be passed to the prompt template.

        Returns:
            A formatted string.
        """
        return ImagePromptValue(image_url=self.format(**kwargs))

    async def aformat_prompt(self, **kwargs: Any) -> PromptValue:
        """Async format the prompt with the inputs.

        Args:
            kwargs: Any arguments to be passed to the prompt template.

        Returns:
            A formatted string.
        """
        return ImagePromptValue(image_url=await self.aformat(**kwargs))

    def format(
        self,
        **kwargs: Any,
    ) -> ImageURL:
        """Format the prompt with the inputs.

        Args:
            kwargs: Any arguments to be passed to the prompt template.

        Returns:
            A formatted string.

        Raises:
            ValueError: If the url is not provided.
            ValueError: If the url is not a string.
            ValueError: If ``'path'`` is provided in the template or kwargs.

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
        url = kwargs.get("url") or formatted.get("url")
        if kwargs.get("path") or formatted.get("path"):
            msg = (
                "Loading images from 'path' has been removed as of 0.3.15 for security "
                "reasons. Please specify images by 'url'."
            )
            raise ValueError(msg)
        detail = kwargs.get("detail") or formatted.get("detail")
        if not url:
            msg = "Must provide url."
            raise ValueError(msg)
        if not isinstance(url, str):
            msg = "url must be a string."
            raise ValueError(msg)  # noqa: TRY004
        output: ImageURL = {"url": url}
        if detail:
            # Don't check literal values here: let the API check them
            output["detail"] = detail
        return output

    async def aformat(self, **kwargs: Any) -> ImageURL:
        """Async format the prompt with the inputs.

        Args:
            kwargs: Any arguments to be passed to the prompt template.

        Returns:
            A formatted string.
        """
        return await run_in_executor(None, self.format, **kwargs)

    def pretty_repr(
        self,
        html: bool = False,  # noqa: FBT001,FBT002
    ) -> str:
        """Return a pretty representation of the prompt.

        Args:
            html: Whether to return an html formatted string.

        Returns:
            A pretty representation of the prompt.
        """
        raise NotImplementedError
