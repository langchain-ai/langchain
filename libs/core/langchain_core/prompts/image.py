from typing import Any, Union

from langchain_core.prompt_values import ImagePromptValue, ImageURL, PromptValue
from langchain_core.prompts.base import BasePromptTemplate
from langchain_core.pydantic_v1 import Field
from langchain_core.utils.image import image_to_data_url


class ImagePromptTemplate(BasePromptTemplate[ImageURL]):
    """An image prompt template for a language model."""

    variable_name: Union[str, None] = None
    """Name of variable to use as messages."""
    template: dict = Field(default_factory=dict)
    """"""

    def __init__(self, **kwargs: Any) -> None:
        if "variable_name" in kwargs:
            # protected var names for formatting
            if kwargs["variable_name"] in ("url", "path", "detail"):
                raise ValueError("")
            if "input_variables" not in kwargs:
                kwargs["input_variables"] = (
                    [kwargs["variable_name"]] if kwargs["variable_name"] else []
                )

        super().__init__(**kwargs)

    @property
    def _prompt_type(self) -> str:
        """Return the prompt type key."""
        return "image-prompt"

    def format_prompt(self, **kwargs: Any) -> PromptValue:
        """Create Chat Messages."""
        return ImagePromptValue(image_url=self.format(**kwargs))

    def format(
        self,
        **kwargs: Any,
    ) -> ImageURL:
        """Format the prompt with the inputs.

        Args:
            kwargs: Any arguments to be passed to the prompt template.

        Returns:
            A formatted string.

        Example:

            .. code-block:: python

                prompt.format(variable1="foo")
        """
        var = kwargs.get(self.variable_name, {}) if self.variable_name else {}
        if isinstance(var, str):
            var = {"url": var}
        var = {**self.template, **var}
        url = kwargs.get("url") or var.get("url")
        path = kwargs.get("path") or var.get("path")
        detail = kwargs.get("detail") or var.get("detail")

        output: ImageURL = {"url": url or image_to_data_url(path)}
        if detail:
            output["detail"] = detail
        return output
