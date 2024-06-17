import os
from typing import Dict, Optional

from langchain_core.pydantic_v1 import BaseModel, Field, root_validator
from mixedbread_ai.client import AsyncMixedbreadAI, MixedbreadAI  # type: ignore
from mixedbread_ai.core import RequestOptions  # type: ignore


class MixedBreadAIClient(BaseModel):
    _client: MixedbreadAI = Field(default=None, exclude=True)
    _aclient: AsyncMixedbreadAI = Field(default=None, exclude=True)
    _request_options: Optional[RequestOptions] = Field(default=None, exclude=True)

    api_key: str = Field(
        alias="mxbai_api_key",
        default_factory=lambda: os.environ.get("MXBAI_API_KEY", None),
        description="mixedbread ai API key. Must be specified directly or "
        "via environment variable 'MXBAI_API_KEY'",
    )
    base_url: Optional[str] = Field(default=None)
    timeout: Optional[float] = Field(
        default=None, description="Timeout for the mixedbread ai API"
    )
    max_retries: Optional[int] = Field(
        default=3, description="Max retries for the mixedbread ai API"
    )

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key exists in environment."""
        api_key = values.get("api_key")
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
