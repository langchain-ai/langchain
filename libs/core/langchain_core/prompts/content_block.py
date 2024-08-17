from typing import Dict, Any, List

from langchain_core.prompt_values import PromptValue
from langchain_core.prompts.base import BasePromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.prompts.image import ImagePromptTemplate
from langchain_core.pydantic_v1 import Field


class ContentBlockPromptTemplate(BasePromptTemplate[Dict[str, Any]]):
    """Template for a single content block."""

    template: dict = Field(default_factory=dict)
    """Template for the block."""

    def __init__(self, template: dict, *, template_format: str = "f-string", **kwargs: Any) -> None:
        input_variables = []
        if "image_url" in template:
            template["image_url"] = ImagePromptTemplate(template["image_url"], template_format=template_format)
            input_variables += template["image_url"].input_variables
        if "text" in template:
            template["text"] = PromptTemplate.from_template(template["text"], template_format=template_format)
            input_variables += template["text"].input_variables
        super().__init__(template=template, input_variables=input_variables, **kwargs)

    @property
    def _prompt_type(self) -> str:
        """Return the prompt type key."""
        return "content-block-prompt"

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "prompts", "chat"]

    def format(self, **kwargs: Any ) -> Dict[str, Any]:
        """Format the prompt with the inputs.

        Args:
            kwargs: Any arguments to be passed to the prompt template.

        Returns:
            A formatted content block as a dict.
        """
        formatted = {}
        for k, v in self.template.items():
            if isinstance(v, BasePromptTemplate):
                formatted[k] = v.format(**kwargs)
            else:
                formatted[k] = v
        return formatted

    def format_prompt(self, **kwargs: Any) -> PromptValue:
        """Format the prompt with the inputs.

        Args:
            kwargs: Any arguments to be passed to the prompt template.

        Returns:
            A formatted string.
        """
        raise NotImplementedError()

    def pretty_repr(self, html: bool = False) -> str:
        """Return a pretty representation of the prompt.

        Args:
            html: Whether to return an html formatted string.

        Returns:
            A pretty representation of the prompt.
        """
        raise NotImplementedError()
