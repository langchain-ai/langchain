"""Requests toolkit."""
from __future__ import annotations

from typing import Any, List

from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import Tool

from langchain_community.agent_toolkits.base import BaseToolkit
from langchain_community.agent_toolkits.json.base import create_json_agent
from langchain_community.agent_toolkits.json.toolkit import JsonToolkit
from langchain_community.agent_toolkits.openapi.prompt import DESCRIPTION
from langchain_community.tools import BaseTool
from langchain_community.tools.json.tool import JsonSpec
from langchain_community.tools.requests.tool import (
    RequestsDeleteTool,
    RequestsGetTool,
    RequestsPatchTool,
    RequestsPostTool,
    RequestsPutTool,
)
from langchain_community.utilities.requests import TextRequestsWrapper


class RequestsToolkit(BaseToolkit):
    """Toolkit for making REST requests.

    *Security Note*: This toolkit contains tools to make GET, POST, PATCH, PUT,
        and DELETE requests to an API.

        Exercise care in who is allowed to use this toolkit. If exposing
        to end users, consider that users will be able to make arbitrary
        requests on behalf of the server hosting the code. For example,
        users could ask the server to make a request to a private API
        that is only accessible from the server.

        Control access to who can submit issue requests using this toolkit and
        what network access it has.

        See https://python.langchain.com/docs/security for more information.
    """

    requests_wrapper: TextRequestsWrapper

    def get_tools(self) -> List[BaseTool]:
        """Return a list of tools."""
        return [
            RequestsGetTool(requests_wrapper=self.requests_wrapper),
            RequestsPostTool(requests_wrapper=self.requests_wrapper),
            RequestsPatchTool(requests_wrapper=self.requests_wrapper),
            RequestsPutTool(requests_wrapper=self.requests_wrapper),
            RequestsDeleteTool(requests_wrapper=self.requests_wrapper),
        ]


class OpenAPIToolkit(BaseToolkit):
    """Toolkit for interacting with an OpenAPI API.

    *Security Note*: This toolkit contains tools that can read and modify
        the state of a service; e.g., by creating, deleting, or updating,
        reading underlying data.

        For example, this toolkit can be used to delete data exposed via
        an OpenAPI compliant API.
    """

    json_agent: Any
    requests_wrapper: TextRequestsWrapper

    def get_tools(self) -> List[BaseTool]:
        """Get the tools in the toolkit."""
        json_agent_tool = Tool(
            name="json_explorer",
            func=self.json_agent.run,
            description=DESCRIPTION,
        )
        request_toolkit = RequestsToolkit(requests_wrapper=self.requests_wrapper)
        return [*request_toolkit.get_tools(), json_agent_tool]

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        json_spec: JsonSpec,
        requests_wrapper: TextRequestsWrapper,
        **kwargs: Any,
    ) -> OpenAPIToolkit:
        """Create json agent from llm, then initialize."""
        json_agent = create_json_agent(llm, JsonToolkit(spec=json_spec), **kwargs)
        return cls(json_agent=json_agent, requests_wrapper=requests_wrapper)
