from __future__ import annotations

import json
from urllib.parse import urlparse
from typing import List, Dict, Tuple, Optional

import requests
import yaml
from pydantic import BaseModel

from langchain.tools.base import BaseTool
from langchain.tools.openapi.utils.api_models import APIOperation, OpenAPISpec
from langchain.tools.requests.tool import BaseRequestsTool
from langchain.requests import TextRequestsWrapper


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


class AIPluginTool(BaseRequestsTool, BaseTool):
    plugin: AIPlugin
    api_spec: str
    base_url: str
    operations: Dict[Tuple[str, str], APIOperation]

    @classmethod
    def from_plugin_url(cls, url: str) -> AIPluginTool:
        plugin = AIPlugin.from_url(url)

        # Strip the URL's ending path to get the base URL to send queries against
        parsed_url = urlparse(url)
        base_url = f"{parsed_url.scheme}://{parsed_url.hostname}"
        base_url = (
            base_url
            + f"{':' + str(parsed_url.port) if parsed_url.port != '80' else ''}"
        )
        base_url = base_url + f"{parsed_url.path.strip('./well-known/ai-plugin.json')}"

        description = (
            f"""Call this tool to get the OpenAPI spec (and usage guide) """
            f"""for interacting with the {plugin.name_for_human} API. """
            f"""Input should be either "usage_guide" or json string """
            f"""with three keys: "path", "method", and if the method """
            f"""is a POST request, include "data". """
            f"""The "data" value should be a dictionary of key-value pairs """
            f"""you want to POST to the url. """
            f"""Please check the usage guide for POST parameters. """
            f"""You should only call "usage_guide" ONCE! """
            f"""This plugin is useful for {plugin.description_for_human}"""
        )
        open_api_spec_str = requests.get(plugin.api.url).text
        open_api_spec = marshal_spec(open_api_spec_str)
        openapi_schema = OpenAPISpec.from_spec_dict(open_api_spec)

        spec_description = cls.generate_api_spec(openapi_schema)
        api_spec = (
            f"Usage Guide: {plugin.description_for_model}\n\n"
            f"OpenAPI Spec: {spec_description}"
        )

        operations = {}
        # TODO: Add support for other HTTP methods
        if openapi_schema.paths:
            for path, info in openapi_schema.paths.items():
                if info.get:
                    operations[(path, "get")] = APIOperation.from_openapi_spec(
                        openapi_schema, path, "get"
                    )
                if info.post:
                    operations[(path, "post")] = APIOperation.from_openapi_spec(
                        openapi_schema, path, "post"
                    )

        print("Description: ", description)
        return cls(
            name=plugin.name_for_model,
            description=description,
            plugin=plugin,
            api_spec=api_spec,
            base_url=base_url,
            operations=operations,
            requests_wrapper=TextRequestsWrapper(),
        )

    @classmethod
    def generate_api_spec(cls, openapi_schema: OpenAPISpec) -> str:
        """Generate a token-optimized API spec for using the tool."""
        operations: List[APIOperation] = []
        if openapi_schema.paths:
            for path, info in openapi_schema.paths.items():
                if info.get:
                    operations.append(
                        APIOperation.from_openapi_spec(openapi_schema, path, "get")
                    )
                if info.post:
                    operations.append(
                        APIOperation.from_openapi_spec(openapi_schema, path, "post")
                    )

        base_str = ""
        for op in operations:
            op_str = f"{op.path}: ({op.method}) {op.description}\n"
            # TODO: add support and tests for query_params
            if op.query_params:
                op_str += f"""Query Params: {op.query_params}\n"""
            if op.method == "post" and op.request_body:
                request_params = []
                for param in op.request_body.properties:
                    param_desc = f"{param.type}"
                    if param.required:
                        param_desc += " required"
                    request_params.append(f"{param.name} ({param_desc})")
                op_str += f"""Body Params: {', '.join(request_params)}\n"""
            base_str += op_str + "\n\n"
        return base_str

    def _run(self, tool_input: str) -> str:
        """Use the tool."""
        if tool_input == "usage_guide":
            return self.api_spec
        try:
            request_info = json.loads(tool_input)
            method = request_info["method"].lower()
            path = (
                request_info["path"]
                if request_info["path"].startswith("/")
                else "/" + request_info["path"]
            )
            if method == "post":
                return self.requests_wrapper.post(
                    self.base_url + path, data=request_info["data"]
                )
            elif method == "get":
                return self.requests_wrapper.get(self.base_url + path)
            raise ValueError("Model method must be either 'get' or 'post'")
        except json.JSONDecodeError:
            raise ValueError("Model input could not be parsed as valid json string")

    async def _arun(self, tool_input: str) -> str:
        """Use the tool asynchronously."""
        return self._run(tool_input)
