from typing import Any, List

from langchain_core.prompt_values import ImagePromptValue, ImageURL, PromptValue
from langchain_core.prompts.base import BasePromptTemplate
from langchain_core.pydantic_v1 import Field
from langchain_core.runnables import run_in_executor


class ImagePromptTemplate(BasePromptTemplate[ImageURL]):
    """Image prompt template for a multimodal model."""

    template: dict = Field(default_factory=dict)
    """Template for the prompt."""

    def __init__(self, **kwargs: Any) -> None:
        if "input_variables" not in kwargs:
            kwargs["input_variables"] = []

        overlap = set(kwargs["input_variables"]) & set(("url", "path", "detail"))
        if overlap:
            raise ValueError(
                "input_variables for the image template cannot contain"
                " any of 'url', 'path', or 'detail'."
                f" Found: {overlap}"
            )
        super().__init__(**kwargs)

    @property
    def _prompt_type(self) -> str:
        """Return the prompt type key."""
        return "image-prompt"

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "prompts", "image"]

    def format_prompt(self, **kwargs: Any) -> PromptValue:
        return ImagePromptValue(image_url=self.format(**kwargs))

    async def aformat_prompt(self, **kwargs: Any) -> PromptValue:
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


        Example:

            .. code-block:: python

                prompt.format(variable1="foo")
        """
        formatted = {}
        for k, v in self.template.items():
            if isinstance(v, str):
                formatted[k] = v.format(**kwargs)
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
        elif not isinstance(url, str):
            msg = "url must be a string."
            raise ValueError(msg)
        else:
            output: ImageURL = {"url": url}
            if detail:
                # Don't check literal values here: let the API check them
                output["detail"] = detail  # type: ignore[typeddict-item]
        return output

    async def aformat(self, **kwargs: Any) -> ImageURL:
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
        raise NotImplementedError()
