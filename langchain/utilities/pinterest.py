"""Util that calls Pinterest REST API"""
from typing import Dict

from pydantic import BaseModel, root_validator

from langchain.utils import get_from_dict_or_env


class PinteresetAPIWrapper(BaseModel):
    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        pinterest_access_token = get_from_dict_or_env(
            values, "pinterest_access_token", "PINTEREST_ACCESS_TOKEN"
        )
        values["pinterest_access_token"] = pinterest_access_token

        try:
            from pinterest.client import PinterestSDKClient

        except ImportError:
            raise ImportError(
                "pinterest-python-sdk is not installed. "
                "Please install it with `pip install pinterest-api-sdk`"
            )

        client = PinterestSDKClient.create_client_with_token(pinterest_access_token)
        values["client"] = client

        return values
