from __future__ import annotations

import json

import requests
import yaml

from langchain.tools.base import BaseTool


def marshal_spec(txt: str) -> dict:
    """Convert the yaml or json serialized spec to a dict."""
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        return yaml.safe_load(txt)


class AIPluginTool(BaseTool):
    api_spec: str

    @classmethod
    def from_plugin_url(cls, url: str) -> AIPluginTool:
        response = requests.get(url).json()
        description = (
            f"Call this tool to get the OpenAPI spec (and usage guide) "
            f"for interacting with the {response['name_for_human']} API. "
            f"You should only call this ONCE! What is the "
            f"{response['name_for_human']} API useful for? "
        ) + response["description_for_human"]
        url = response["api"]["url"]
        open_api_spec_str = requests.get(url).text
        open_api_spec = marshal_spec(open_api_spec_str)
        api_spec = (
            f"Usage Guide: {response['description_for_model']}\n\n"
            f"OpenAPI Spec: {open_api_spec}"
        )
        return cls(
            name=response["name_for_model"], description=description, api_spec=api_spec
        )

    def _run(self, tool_input: str) -> str:
        """Use the tool."""
        return self.api_spec

    async def _arun(self, tool_input: str) -> str:
        """Use the tool asynchronously."""
        return self.api_spec
