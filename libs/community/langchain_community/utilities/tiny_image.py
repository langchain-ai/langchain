"""Util that calls TinyImage."""
from typing import Dict, Optional

import aiohttp
import requests
from langchain_core.pydantic_v1 import BaseModel, SecretStr, root_validator
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env


class TinyImageAPIWrapper(BaseModel):
    """Wrapper for TinyImage api.

    To use, the environment variables ``TINY_IMAGE_API_KEY``, or pass `api_key` as
    api key to the constructor.

    Example:
        .. code-block:: python

            from langchain_community.utilities.tiny_image import TinyImageAPIWrapper
            cli = TinyImageAPIWrapper(
                api_key="xxx",
            )
            cli.run('https://tinypng.com/images/smart-resizing/new-aspect-ratio.jpg')
    """

    api_key: Optional[SecretStr] = None
    request_url: str = "https://api.tinify.com/shrink"
    """TinyImage account api key."""

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key in environment."""
        api_key = convert_to_secret_str(
            get_from_dict_or_env(values, "api_key", "TINY_IMAGE_API_KEY")
        )

        values["api_key"] = api_key
        return values

    def run(self, source: str) -> str:
        """Run image tiny with source input.

        Args:
            source: The url of the image.
        """  # noqa: E501
        session = requests.Session()
        session.auth = ("api", self.api_key.get_secret_value())
        r = session.post(self.request_url, json={"source": {"url": source}})
        if 400 <= r.status_code < 500:
            return "TokenError: Could not login to TinyImage, please check your credentials."  # noqa: E501
        return r.json()["output"]["url"]

    async def arun(self, source: str) -> str:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.request_url,
                json={"source": {"url": source}},
                auth=aiohttp.BasicAuth("api", self.api_key.get_secret_value()),
            ) as response:
                if 400 <= response.status < 500:
                    return "TokenError: Could not login to TinyImage, please check your credentials."  # noqa: E501
                response_json = await response.json(content_type=response.content_type)
                return response_json["output"]["url"]
