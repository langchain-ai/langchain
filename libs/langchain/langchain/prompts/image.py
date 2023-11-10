from typing import Any, Union

from langchain.prompts.base import ImagePromptValue
from langchain.pydantic_v1 import Field
from langchain.schema import BasePromptTemplate, PromptValue
from langchain.types.image import ImageURL
from langchain.utils.image import image_to_data_url


class ImagePromptTemplate(BasePromptTemplate):
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
        return ImagePromptValue(image=self.format(**kwargs))

    def format(
        self,
        url: Union[str, None] = None,
        path: Union[str, None] = None,
        detail: Union[str, None] = None,
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
        url = url or var.get("url")
        path = path or var.get("path")
        output = {"url": url or image_to_data_url(path)}
        detail = detail or var.get("detail")
        if detail:
            output["detail"] = detail
        return output
