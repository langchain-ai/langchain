import os
from typing import Optional, Dict

from mixedbread_ai.client import MixedbreadAI, AsyncMixedbreadAI
from mixedbread_ai.core import RequestOptions

from langchain_core.pydantic_v1 import BaseModel, Extra, root_validator, Field


class MixedBreadAIClient(BaseModel):
    _client: MixedbreadAI = Field(exclude=True)
    _aclient: AsyncMixedbreadAI = Field(exclude=True)

    mxbai_api_key: Optional[str] = Field(default=None, alias="api_key")
    """Mixedbread AI API key. Must be specified directly or via environment variable 
        MIXEDBREAD_API_KEY."""
    mxbai_api_base: Optional[str] = Field(default=None, alias="base_url")
    timeout: Optional[float] = Field(default=None)
    """Timeout in seconds for the Mixedbread AI API request."""
    max_retries: Optional[int] = Field(default=None)
    """Maximum number of retries to make when generating."""
    _request_options: Optional[RequestOptions] = Field(default=None, exclude=True)

    class Config:
        """Configuration for this pydantic object."""
        arbitrary_types_allowed = True

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key exists in environment."""
        mxbai_api_key = values.get("mxbai_api_key") or os.getenv(
            "MXBAI_API_KEY", None
        )
        mxbai_api_base = values.get("mxbai_api_base")
        timeout = values.get("timeout")
        max_retries = values.get("max_retries")

        values["_client"] = MixedbreadAI(
            base_url=mxbai_api_base,
            api_key=mxbai_api_key,
            timeout=timeout,
        )
        values["_aclient"] = AsyncMixedbreadAI(
            base_url=mxbai_api_base,
            api_key=mxbai_api_key,
            timeout=timeout,
        )

        values["_request_options"] = RequestOptions(
            max_retries=max_retries
        ) if max_retries is not None else None

        return values
