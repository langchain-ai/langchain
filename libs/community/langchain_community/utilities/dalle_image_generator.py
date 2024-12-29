"""Utility that calls OpenAI's Dall-E Image Generator."""

import logging
from typing import Any, Dict, Mapping, Optional, Tuple, Union

from langchain_core.utils import (
    from_env,
    get_pydantic_field_names,
    secret_from_env,
)
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator
from typing_extensions import Self

from langchain_community.utils.openai import is_openai_v1

logger = logging.getLogger(__name__)


class DallEAPIWrapper(BaseModel):
    """Wrapper for OpenAI's DALL-E Image Generator.

    https://platform.openai.com/docs/guides/images/generations?context=node

    Usage instructions:

    1. `pip install openai`
    2. save your OPENAI_API_KEY in an environment variable
    """

    client: Any = None  #: :meta private:
    async_client: Any = Field(default=None, exclude=True)  #: :meta private:
    model_name: str = Field(default="dall-e-2", alias="model")
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    openai_api_key: Optional[SecretStr] = Field(
        alias="api_key",
        default_factory=secret_from_env(
            "OPENAI_API_KEY",
            default=None,
        ),
    )
    """Automatically inferred from env var `OPENAI_API_KEY` if not provided."""
    openai_api_base: Optional[str] = Field(
        alias="base_url", default_factory=from_env("OPENAI_API_BASE", default=None)
    )
    """Base URL path for API requests, leave blank if not using a proxy or service 
        emulator."""
    openai_organization: Optional[str] = Field(
        alias="organization",
        default_factory=from_env(
            ["OPENAI_ORG_ID", "OPENAI_ORGANIZATION"], default=None
        ),
    )
    """Automatically inferred from env var `OPENAI_ORG_ID` if not provided."""
    # to support explicit proxy for OpenAI
    openai_proxy: str = Field(default_factory=from_env("OPENAI_PROXY", default=""))
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

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    @model_validator(mode="before")
    @classmethod
    def build_extra(cls, values: Dict[str, Any]) -> Any:
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

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """Validate that api key and python package exists in environment."""
        try:
            import openai

        except ImportError:
            raise ImportError(
                "Could not import openai python package. "
                "Please install it with `pip install openai`."
            )

        if is_openai_v1():
            client_params = {
                "api_key": self.openai_api_key.get_secret_value()
                if self.openai_api_key
                else None,
                "organization": self.openai_organization,
                "base_url": self.openai_api_base,
                "timeout": self.request_timeout,
                "max_retries": self.max_retries,
                "default_headers": self.default_headers,
                "default_query": self.default_query,
                "http_client": self.http_client,
            }

            if not self.client:
                self.client = openai.OpenAI(**client_params).images  # type: ignore[arg-type, arg-type, arg-type, arg-type, arg-type, arg-type, arg-type, arg-type]
            if not self.async_client:
                self.async_client = openai.AsyncOpenAI(**client_params).images  # type: ignore[arg-type, arg-type, arg-type, arg-type, arg-type, arg-type, arg-type, arg-type]
        elif not self.client:
            self.client = openai.Image  # type: ignore[attr-defined]
        else:
            pass
        return self

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
