from __future__ import annotations

import logging
from typing import Optional, Type

from langchain_core.callbacks import CallbackManagerForToolRun
from pydantic import BaseModel, Field

from langchain_community.tools.edenai.edenai_base_tool import EdenaiTool

logger = logging.getLogger(__name__)


class TextModerationInput(BaseModel):
    query: str = Field(description="Text to moderate")


class EdenAiTextModerationTool(EdenaiTool):
    """Tool that queries the Eden AI Explicit text detection.

    for api reference check edenai documentation:
    https://docs.edenai.co/reference/image_explicit_content_create.

    To use, you should have
    the environment variable ``EDENAI_API_KEY`` set with your API token.
    You can find your token here: https://app.edenai.run/admin/account/settings

    """

    name: str = "edenai_explicit_content_detection_text"
    description: str = (
        "A wrapper around edenai Services explicit content detection for text. "
        """Useful for when you have to scan text for offensive, 
        sexually explicit or suggestive content,
        it checks also if there is any content of self-harm,
        violence, racist or hate speech."""
        """the structure of the output is : 
        'the type of the explicit content : the likelihood of it being explicit'
        the likelihood is a number 
        between 1 and 5, 1 being the lowest and 5 the highest.
        something is explicit if the likelihood is equal or higher than 3.
        for example : 
        nsfw_likelihood: 1
        this is not explicit.
        for example : 
        nsfw_likelihood: 3
        this is explicit.
        """
        "Input should be a string."
    )
    args_schema: Type[BaseModel] = TextModerationInput

    language: str

    feature: str = "text"
    subfeature: str = "moderation"

    def _parse_response(self, response: list) -> str:
        formatted_result = []
        for result in response:
            if "nsfw_likelihood" in result.keys():
                formatted_result.append(
                    "nsfw_likelihood: " + str(result["nsfw_likelihood"])
                )

            for label, likelihood in zip(result["label"], result["likelihood"]):
                formatted_result.append(f'"{label}": {str(likelihood)}')

        return "\n".join(formatted_result)

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        query_params = {"text": query, "language": self.language}
        return self._call_eden_ai(query_params)
