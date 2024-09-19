"""Util that calls SceneXplain.

In order to set this up, you need API key for the SceneXplain API.
You can obtain a key by following the steps below.
- Sign up for a free account at https://scenex.jina.ai/.
- Navigate to the API Access page (https://scenex.jina.ai/api) and create a new API key.
"""

from typing import Any, Dict

import requests
from langchain_core.utils import from_env, get_from_dict_or_env
from pydantic import BaseModel, Field, model_validator


class SceneXplainAPIWrapper(BaseModel):
    """Wrapper for SceneXplain API.

    In order to set this up, you need API key for the SceneXplain API.
    You can obtain a key by following the steps below.
    - Sign up for a free account at https://scenex.jina.ai/.
    - Navigate to the API Access page (https://scenex.jina.ai/api)
      and create a new API key.
    """

    scenex_api_key: str = Field(..., default_factory=from_env("SCENEX_API_KEY"))
    scenex_api_url: str = "https://api.scenex.jina.ai/v1/describe"

    def _describe_image(self, image: str) -> str:
        headers = {
            "x-api-key": f"token {self.scenex_api_key}",
            "content-type": "application/json",
        }
        payload = {
            "data": [
                {
                    "image": image,
                    "algorithm": "Ember",
                    "languages": ["en"],
                }
            ]
        }
        response = requests.post(self.scenex_api_url, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json().get("result", [])
        img = result[0] if result else {}

        return img.get("text", "")

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: Dict) -> Any:
        """Validate that api key exists in environment."""
        scenex_api_key = get_from_dict_or_env(
            values, "scenex_api_key", "SCENEX_API_KEY"
        )
        values["scenex_api_key"] = scenex_api_key

        return values

    def run(self, image: str) -> str:
        """Run SceneXplain image explainer."""
        description = self._describe_image(image)
        if not description:
            return "No description found."

        return description
