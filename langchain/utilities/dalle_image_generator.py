"""Utility that calls OpenAI's Dall-E Image Generator."""
from typing import Any, Dict, Optional

from langchain.pydantic_v1 import BaseModel, Extra, root_validator
from langchain.utils import get_from_dict_or_env


class DallEAPIWrapper(BaseModel):
    """Wrapper for OpenAI's DALL-E Image Generator.

    Docs for using:
    1. pip install openai
    2. save your OPENAI_API_KEY in an environment variable

    """

    client: Any  #: :meta private:
    openai_api_key: Optional[str] = None
    """number of images to generate"""
    n: int = 1
    """size of image to generate"""
    size: str = "1024x1024"

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    def _dalle_image_url(self, prompt: str) -> str:
        params = {"prompt": prompt, "n": self.n, "size": self.size}
        response = self.client.create(**params)
        return response["data"][0]["url"]

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        openai_api_key = get_from_dict_or_env(
            values, "openai_api_key", "OPENAI_API_KEY"
        )
        try:
            import openai

            openai.api_key = openai_api_key
            values["client"] = openai.Image
        except ImportError:
            raise ValueError(
                "Could not import openai python package. "
                "Please it install it with `pip install openai`."
            )
        return values

    def run(self, query: str) -> str:
        """Run query through OpenAI and parse result."""
        image_url = self._dalle_image_url(query)

        if image_url is None or image_url == "":
            # We don't want to return the assumption alone if answer is empty
            return "No image was generated"
        else:
            return image_url
