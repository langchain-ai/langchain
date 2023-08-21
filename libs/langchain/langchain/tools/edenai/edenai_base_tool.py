from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import requests
from pydantic import root_validator

from langchain.tools.base import BaseTool
from langchain.utils import get_from_dict_or_env

logger = logging.getLogger(__name__)


class EdenaiTool(BaseTool):

    """
    the base tool for all the EdenAI Tools .
    you should have
    the environment variable ``EDENAI_API_KEY`` set with your API token.
    You can find your token here: https://app.edenai.run/admin/account/settings
    """

    feature: str
    subfeature: str
    edenai_api_key: Optional[str] = None

    providers: list[str]
    """provider to use for the API call."""

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key exists in environment."""
        values["edenai_api_key"] = get_from_dict_or_env(
            values, "edenai_api_key", "EDENAI_API_KEY"
        )
        return values

    def _call_eden_ai(self, query_params: Dict[str, Any]) -> requests.Response:
        """
        Make an API call to the EdenAI service with the specified query parameters.

        Args:
            query_params (dict): The parameters to include in the API call.

        Returns:
            requests.Response: The response from the EdenAI API call.

        """

        # faire l'API call

        headers = {"Authorization": f"Bearer {self.edenai_api_key}"}

        url = f"https://api.edenai.run/v2/{self.feature}/{self.subfeature}"

        payload = {
            "providers": str(self.providers),
            "response_as_dict": False,
            "attributes_as_list": True,
            "show_original_response": False,
        }

        payload.update(query_params)

        response = requests.post(url, json=payload, headers=headers)

        if response.status_code >= 500:
            raise Exception(f"EdenAI Server: Error {response.status_code}")
        elif response.status_code >= 400:
            raise ValueError(f"EdenAI received an invalid payload: {response.text}")
        elif response.status_code != 200:
            raise Exception(
                f"EdenAI returned an unexpected response with status "
                f"{response.status_code}: {response.text}"
            )

        key = self.providers if payload.get("response_as_dict") else 0

        provider_response = response.json()[key]
        if provider_response["status"] == "fail":
            raise ValueError(provider_response["error"]["message"])

        return response

    def _get_edenai(self, url: str) -> requests.Response:
        headers = {
            "accept": "application/json",
            "authorization": f"Bearer {self.edenai_api_key}",
        }

        response = requests.get(url, headers=headers)

        if response.status_code >= 500:
            raise Exception(f"EdenAI Server: Error {response.status_code}")
        elif response.status_code >= 400:
            raise ValueError(f"EdenAI received an invalid payload: {response.text}")
        elif response.status_code != 200:
            raise Exception(
                f"EdenAI returned an unexpected response with status "
                f"{response.status_code}: {response.text}"
            )

        return response
