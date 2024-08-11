"""Utility that calls OpenAI's Dall-E Image Generator."""

import logging
import os
from typing import Any, Dict, Mapping, Optional, Tuple, Union

from langchain_core.pydantic_v1 import BaseModel, Field, root_validator
from langchain_core.utils import (
    get_from_dict_or_env,
    get_pydantic_field_names,
)

from langchain_community.utils.openai import is_openai_v1

logger = logging.getLogger(__name__)


class DallEAPIWrapper(BaseModel):
    """Wrapper for OpenAI's DALL-E Image Generator.

    https://platform.openai.com/docs/guides/images/generations?context=node

    Usage instructions:

    1. `pip install openai`
    2. save your OPENAI_API_KEY in an environment variable
    """

    client: Any  #: :meta private:
    async_client: Any = Field(default=None, exclude=True)  #: :meta private:
    model_name: str = Field(default="dall-e-2", alias="model")
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    openai_api_key: Optional[str] = Field(default=None, alias="api_key")
    """Automatically inferred from env var `OPENAI_API_KEY` if not provided."""
    openai_api_base: Optional[str] = Field(default=None, alias="base_url")
    """Base URL path for API requests, leave blank if not using a proxy or service 
        emulator."""
    openai_organization: Optional[str] = Field(default=None, alias="organization")
    """Automatically inferred from env var `OPENAI_ORG_ID` if not provided."""
    # to support explicit proxy for OpenAI
    openai_proxy: Optional[str] = None
    request_timeout: Union[float, Tuple[float, float], Any, None] = Field(
        default=None, alias="timeout"
    )
    n: int = 1
    """Number of images to generate"""
    size: str = "1024x1024"
    """Size of image to generate"""
    separator: str = "\n"
    """Separator to use when multiple URLs are returned."""
    quality: Optional[str] = "standard"
    """Quality of the image that will be generated"""
    max_retries: int = 2
    """Maximum number of retries to make when generating."""
    default_headers: Union[Mapping[str, str], None] = None
    default_query: Union[Mapping[str, object], None] = None
    # Configure a custom httpx client. See the
    # [httpx documentation](https://www.python-httpx.org/api/#client) for more details.
    http_client: Union[Any, None] = None
    """Optional httpx.Client."""

    class Config:
        extra = "forbid"

    @root_validator(pre=True)
    def build_extra(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Build extra kwargs from additional params that were passed in."""
        all_required_field_names = get_pydantic_field_names(cls)
        extra = values.get("model_kwargs", {})
        for field_name in list(values):
            if field_name in extra:
                raise ValueError(f"Found {field_name} supplied twice.")
            if field_name not in all_required_field_names:
                logger.warning(
                    f"""WARNING! {field_name} is not default parameter.
                    {field_name} was transferred to model_kwargs.
                    Please confirm that {field_name} is what you intended."""
                )
                extra[field_name] = values.pop(field_name)

        invalid_model_kwargs = all_required_field_names.intersection(extra.keys())
        if invalid_model_kwargs:
            raise ValueError(
                f"Parameters {invalid_model_kwargs} should be specified explicitly. "
                f"Instead they were passed in as part of `model_kwargs` parameter."
            )

        values["model_kwargs"] = extra
        return values

    @root_validator(pre=False, skip_on_failure=True)
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        values["openai_api_key"] = get_from_dict_or_env(
            values, "openai_api_key", "OPENAI_API_KEY"
        )
        # Check OPENAI_ORGANIZATION for backwards compatibility.
        values["openai_organization"] = (
            values["openai_organization"]
            or os.getenv("OPENAI_ORG_ID")
            or os.getenv("OPENAI_ORGANIZATION")
            or None
        )
        values["openai_api_base"] = values["openai_api_base"] or os.getenv(
            "OPENAI_API_BASE"
        )
        values["openai_proxy"] = get_from_dict_or_env(
            values,
            "openai_proxy",
            "OPENAI_PROXY",
            default="",
        )

        try:
            import openai

        except ImportError:
            raise ImportError(
                "Could not import openai python package. "
                "Please install it with `pip install openai`."
            )

        if is_openai_v1():
            client_params = {
                "api_key": values["openai_api_key"],
                "organization": values["openai_organization"],
                "base_url": values["openai_api_base"],
                "timeout": values["request_timeout"],
                "max_retries": values["max_retries"],
                "default_headers": values["default_headers"],
                "default_query": values["default_query"],
                "http_client": values["http_client"],
            }

            if not values.get("client"):
                values["client"] = openai.OpenAI(**client_params).images
            if not values.get("async_client"):
                values["async_client"] = openai.AsyncOpenAI(**client_params).images
        elif not values.get("client"):
            values["client"] = openai.Image
        else:
            pass
        return values

    def run(self, query: str) -> str:
        """Run query through OpenAI and parse result."""

        if is_openai_v1():
            response = self.client.generate(
                prompt=query,
                n=self.n,
                size=self.size,
                model=self.model_name,
                quality=self.quality,
            )
            image_urls = self.separator.join([item.url for item in response.data])
        else:
            response = self.client.create(
                prompt=query, n=self.n, size=self.size, model=self.model_name
            )
            image_urls = self.separator.join([item["url"] for item in response["data"]])

        return image_urls if image_urls else "No image was generated"
