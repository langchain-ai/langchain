from __future__ import annotations

import json
from typing import Optional

import requests
import yaml
from pydantic import BaseModel

from langchain.tools.base import BaseTool


class ApiConfig(BaseModel):
    type: str
    url: str
    has_user_authentication: Optional[bool] = False


class AIPlugin(BaseModel):
    """AI Plugin Definition."""

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

    @classmethod
    def from_url(cls, url: str) -> AIPlugin:
        """Instantiate AIPlugin from a URL."""
        response = requests.get(url).json()
        return cls(**response)


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
        plugin = AIPlugin.from_url(url)
        description = (
            f"Call this tool to get the OpenAPI spec (and usage guide) "
            f"for interacting with the {plugin.name_for_human} API. "
            f"You should only call this ONCE! What is the "
            f"{plugin.name_for_human} API useful for? "
        ) + plugin.description_for_human
        open_api_spec_str = requests.get(plugin.api.url).text
        open_api_spec = marshal_spec(open_api_spec_str)
        api_spec = (
            f"Usage Guide: {plugin.description_for_model}\n\n"
            f"OpenAPI Spec: {open_api_spec}"
        )

        return cls(
            name=plugin.name_for_model,
            description=description,
            plugin=plugin,
            api_spec=api_spec,
        )

    def _run(self, tool_input: str) -> str:
        """Use the tool."""
        return self.api_spec

    async def _arun(self, tool_input: str) -> str:
        """Use the tool asynchronously."""
        return self.api_spec
