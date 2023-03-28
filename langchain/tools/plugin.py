from __future__ import annotations


from typing import Optional

import requests
from pydantic import BaseModel
import json

import requests
import yaml

from langchain.tools.base import BaseTool


class ApiConfig(BaseModel):
    type: str
    url: str
    has_user_authentication: Optional[bool] = False


class AIPlugin(BaseModel):
    schema_version: str
    name_for_model: str
    name_for_human: str
    description_for_model: str
    description_for_human: str
    auth: Optional[dict] = None
    api: ApiConfig
    logo_url: Optional[str]
    contact_email: Optional[str]
    legal_info_url: Optional[str]
    
def marshal_spec(txt: str) -> dict:
    """Convert the yaml or json serialized spec to a dict."""
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        return yaml.safe_load(txt)


class AIPluginTool(BaseTool):
    plugin: AIPlugin
    api_spec: str

    @classmethod
    def from_plugin_url(cls, url: str) -> AIPluginTool:
        open_api_spec_str = requests.get(url).text
        open_api_spec = marshal_spec(open_api_spec_str)
        cls.plugin = AIPlugin(**requests.get(url).json())
        description = (
            f"Call this tool to get the OpenAPI spec (and usage guide) "
            f"for interacting with the {cls.plugin.name_for_human} API. "
            f"You should only call this ONCE! What is the "
            f"{cls.plugin.name_for_human} API useful for? "
        ) + cls.plugin.description_for_human

        api_spec = (
            f"Usage Guide: {cls.plugin.description_for_model}\n\n"
            f"OpenAPI Spec: {requests.get(cls.plugin.api.url).json()}"
        )

        return cls(
            name=cls.plugin.name_for_model,
            description=description,
            plugin=cls.plugin,
            api_spec=api_spec,
        )

    def _run(self, tool_input: str) -> str:
        """Use the tool."""
        return self.api_spec

    async def _arun(self, tool_input: str) -> str:
        """Use the tool asynchronously."""
        return self.api_spec
