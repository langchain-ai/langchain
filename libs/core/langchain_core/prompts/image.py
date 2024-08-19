from typing import Any, List, Union

from langchain_core.prompt_values import ImagePromptValue, ImageURL, PromptValue
from langchain_core.prompts.base import BasePromptTemplate
from langchain_core.prompts.string import get_template_variables
from langchain_core.pydantic_v1 import Field
from langchain_core.runnables import run_in_executor
from langchain_core.utils import image as image_utils


class ImagePromptTemplate(BasePromptTemplate[ImageURL]):
    """Image prompt template for a multimodal model."""

    template: dict = Field(default_factory=dict)
    """Template for the prompt."""

    def __init__(
        self,
        template: Union[dict, str],
        *,
        template_format: str = "f-string",
        **kwargs: Any,
    ) -> None:
        if isinstance(template, dict) and "image_url" in template:
            image_url = template["image_url"]
        else:
            image_url = template
        input_variables = kwargs.pop("input_variables", [])
        if isinstance(image_url, str):
            vars = get_template_variables(image_url, template_format)
            if vars:
                if len(vars) > 1:
                    raise ValueError(
                        "Only one format variable allowed per image"
                        f" template.\nGot: {vars}"
                        f"\nFrom: {template}"
                    )
                input_variables = [vars[0]]
            image_url = {"url": image_url}
        elif isinstance(image_url, dict):
            image_url = dict(image_url)
            for key in ["url", "path", "detail"]:
                if key in image_url:
                    input_variables.extend(
                        get_template_variables(image_url[key], template_format)
                    )
        overlap = set(input_variables) & set(("url", "path", "detail"))
        if overlap:
            raise ValueError(
                "input_variables for the image template cannot contain"
                " any of 'url', 'path', or 'detail'."
                f" Found: {overlap}"
            )
        super().__init__(
            template=image_url, input_variables=list(set(input_variables)), **kwargs
        )  # type: ignore[call-arg]

    @property
    def _prompt_type(self) -> str:
        """Return the prompt type key."""
        return "image-prompt"

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
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
            ValueError: If the url or path is not provided.
            ValueError: If the path or url is not a string.

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
        path = kwargs.get("path") or formatted.get("path")
        detail = kwargs.get("detail") or formatted.get("detail")
        if not url and not path:
            raise ValueError("Must provide either url or path.")
        if not url:
            if not isinstance(path, str):
                raise ValueError("path must be a string.")
            url = image_utils.image_to_data_url(path)
        if not isinstance(url, str):
            raise ValueError("url must be a string.")
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
            ValueError: If the url or path is not provided.
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
        raise NotImplementedError()
