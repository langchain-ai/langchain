"""Util that calls Golden."""

import json
from typing import Dict, Optional

import requests
from langchain_core.pydantic_v1 import BaseModel, Extra, root_validator
from langchain_core.utils import get_from_dict_or_env

GOLDEN_BASE_URL = "https://golden.com"
GOLDEN_TIMEOUT = 5000


class GoldenQueryAPIWrapper(BaseModel):
    """Wrapper for Golden.

    Docs for using:

    1. Go to https://golden.com and sign up for an account
    2. Get your API Key from https://golden.com/settings/api
    3. Save your API Key into GOLDEN_API_KEY env variable

    """

    golden_api_key: Optional[str] = None

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @root_validator(pre=True)
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        golden_api_key = get_from_dict_or_env(
            values, "golden_api_key", "GOLDEN_API_KEY"
        )
        values["golden_api_key"] = golden_api_key

        return values

    def run(self, query: str) -> str:
        """Run query through Golden Query API and return the JSON raw result."""

        headers = {"apikey": self.golden_api_key or ""}

        response = requests.post(
            f"{GOLDEN_BASE_URL}/api/v2/public/queries/",
            json={"prompt": query},
            headers=headers,
            timeout=GOLDEN_TIMEOUT,
        )
        if response.status_code != 201:
            return response.text

        content = json.loads(response.content)
        query_id = content["id"]

        response = requests.get(
            (
                f"{GOLDEN_BASE_URL}/api/v2/public/queries/{query_id}/results/"
                "?pageSize=10"
            ),
            headers=headers,
            timeout=GOLDEN_TIMEOUT,
        )
        return response.text
