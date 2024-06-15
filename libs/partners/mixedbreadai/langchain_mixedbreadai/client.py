import os
from typing import Dict, Optional

from langchain_core.pydantic_v1 import BaseModel, Field, root_validator
from mixedbread_ai.client import AsyncMixedbreadAI, MixedbreadAI
from mixedbread_ai.core import RequestOptions


class MixedBreadAIClient(BaseModel):
    _client: MixedbreadAI = Field(default=None, exclude=True)
    _aclient: AsyncMixedbreadAI = Field(default=None, exclude=True)

    api_key: Optional[str] = Field(default=None)
    """Mixedbread AI API key. Must be specified directly or via environment variable 
        MXBAI_API_KEY."""
    base_url: Optional[str] = Field(default=None)
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
        api_key = values.get("api_key") or os.getenv("MXBAI_API_KEY", None)
        base_url = values.get("api_base")
        timeout = values.get("timeout")
        max_retries = values.get("max_retries")

        values["_client"] = MixedbreadAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
        )
        values["_aclient"] = AsyncMixedbreadAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
        )

        values["_request_options"] = (
            RequestOptions(max_retries=max_retries) if max_retries is not None else None
        )

        return values
