"""Requests toolkit."""

from __future__ import annotations

from typing import Any, List

from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import BaseToolkit, Tool

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

    Setup:
        Install ``langchain-community``.

        .. code-block:: bash

            pip install -U langchain-community

    Key init args:
        requests_wrapper: langchain_community.utilities.requests.GenericRequestsWrapper
            wrapper for executing requests.
        allow_dangerous_requests: bool
            Defaults to False. Must "opt-in" to using dangerous requests by setting to True.

    Instantiate:
        .. code-block:: python

            from langchain_community.agent_toolkits.openapi.toolkit import RequestsToolkit
            from langchain_community.utilities.requests import TextRequestsWrapper

            toolkit = RequestsToolkit(
                requests_wrapper=TextRequestsWrapper(headers={}),
                allow_dangerous_requests=ALLOW_DANGEROUS_REQUEST,
            )

    Tools:
        .. code-block:: python

            tools = toolkit.get_tools()
            tools

        .. code-block:: none

            [RequestsGetTool(requests_wrapper=TextRequestsWrapper(headers={}, aiosession=None, auth=None, response_content_type='text', verify=True), allow_dangerous_requests=True),
            RequestsPostTool(requests_wrapper=TextRequestsWrapper(headers={}, aiosession=None, auth=None, response_content_type='text', verify=True), allow_dangerous_requests=True),
            RequestsPatchTool(requests_wrapper=TextRequestsWrapper(headers={}, aiosession=None, auth=None, response_content_type='text', verify=True), allow_dangerous_requests=True),
            RequestsPutTool(requests_wrapper=TextRequestsWrapper(headers={}, aiosession=None, auth=None, response_content_type='text', verify=True), allow_dangerous_requests=True),
            RequestsDeleteTool(requests_wrapper=TextRequestsWrapper(headers={}, aiosession=None, auth=None, response_content_type='text', verify=True), allow_dangerous_requests=True)]

    Use within an agent:
        .. code-block:: python

            from langchain_openai import ChatOpenAI
            from langgraph.prebuilt import create_react_agent


            api_spec = \"\"\"
            openapi: 3.0.0
            info:
              title: JSONPlaceholder API
              version: 1.0.0
            servers:
              - url: https://jsonplaceholder.typicode.com
            paths:
              /posts:
                get:
                  summary: Get posts
                  parameters: &id001
                    - name: _limit
                      in: query
                      required: false
                      schema:
                        type: integer
                      example: 2
                      description: Limit the number of results
            \"\"\"

            system_message = \"\"\"
            You have access to an API to help answer user queries.
            Here is documentation on the API:
            {api_spec}
            \"\"\".format(api_spec=api_spec)

            llm = ChatOpenAI(model="gpt-4o-mini")
            agent_executor = create_react_agent(llm, tools, state_modifier=system_message)

            example_query = "Fetch the top two posts. What are their titles?"

            events = agent_executor.stream(
                {"messages": [("user", example_query)]},
                stream_mode="values",
            )
            for event in events:
                event["messages"][-1].pretty_print()

        .. code-block:: none

             ================================[1m Human Message [0m=================================

            Fetch the top two posts. What are their titles?
            ==================================[1m Ai Message [0m==================================
            Tool Calls:
            requests_get (call_RV2SOyzCnV5h2sm4WPgG8fND)
            Call ID: call_RV2SOyzCnV5h2sm4WPgG8fND
            Args:
                url: https://jsonplaceholder.typicode.com/posts?_limit=2
            =================================[1m Tool Message [0m=================================
            Name: requests_get

            [
            {
                "userId": 1,
                "id": 1,
                "title": "sunt aut facere repellat provident occaecati excepturi optio reprehenderit",
                "body": "quia et suscipit..."
            },
            {
                "userId": 1,
                "id": 2,
                "title": "qui est esse",
                "body": "est rerum tempore vitae..."
            }
            ]
            ==================================[1m Ai Message [0m==================================

            The titles of the top two posts are:
            1. "sunt aut facere repellat provident occaecati excepturi optio reprehenderit"
            2. "qui est esse"
    """  # noqa: E501

    requests_wrapper: TextRequestsWrapper
    """The requests wrapper."""
    allow_dangerous_requests: bool = False
    """Allow dangerous requests. See documentation for details."""

    def get_tools(self) -> List[BaseTool]:
        """Return a list of tools."""
        return [
            RequestsGetTool(
                requests_wrapper=self.requests_wrapper,
                allow_dangerous_requests=self.allow_dangerous_requests,
            ),
            RequestsPostTool(
                requests_wrapper=self.requests_wrapper,
                allow_dangerous_requests=self.allow_dangerous_requests,
            ),
            RequestsPatchTool(
                requests_wrapper=self.requests_wrapper,
                allow_dangerous_requests=self.allow_dangerous_requests,
            ),
            RequestsPutTool(
                requests_wrapper=self.requests_wrapper,
                allow_dangerous_requests=self.allow_dangerous_requests,
            ),
            RequestsDeleteTool(
                requests_wrapper=self.requests_wrapper,
                allow_dangerous_requests=self.allow_dangerous_requests,
            ),
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
    """The JSON agent."""
    requests_wrapper: TextRequestsWrapper
    """The requests wrapper."""
    allow_dangerous_requests: bool = False
    """Allow dangerous requests. See documentation for details."""

    def get_tools(self) -> List[BaseTool]:
        """Get the tools in the toolkit."""
        json_agent_tool = Tool(
            name="json_explorer",
            func=self.json_agent.run,
            description=DESCRIPTION,
        )
        request_toolkit = RequestsToolkit(
            requests_wrapper=self.requests_wrapper,
            allow_dangerous_requests=self.allow_dangerous_requests,
        )
        return [*request_toolkit.get_tools(), json_agent_tool]

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        json_spec: JsonSpec,
        requests_wrapper: TextRequestsWrapper,
        allow_dangerous_requests: bool = False,
        **kwargs: Any,
    ) -> OpenAPIToolkit:
        """Create json agent from llm, then initialize."""
        json_agent = create_json_agent(llm, JsonToolkit(spec=json_spec), **kwargs)
        return cls(
            json_agent=json_agent,
            requests_wrapper=requests_wrapper,
            allow_dangerous_requests=allow_dangerous_requests,
        )
