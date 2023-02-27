"""Toolkit for interacting with a OpenAPI api."""
from __future__ import annotations

from typing import Any, List

from langchain.agents.agent import AgentExecutor
from langchain.agents.agent_toolkits.json.base import create_json_agent
from langchain.agents.tools import Tool
from langchain.llms.base import BaseLLM
from langchain.requests import RequestsWrapper
from langchain.tools import BaseTool
from langchain.tools.base import BaseToolkit
from langchain.tools.json.tool import JsonSpec
from langchain.tools.json.toolkit import JsonToolkit
from langchain.tools.openapi.prompt import DESCRIPTION
from langchain.tools.requests.toolkit import RequestsToolkit


class OpenAPIToolkit(BaseToolkit):
    """Toolkit for interacting with a OpenAPI api."""

    json_agent: AgentExecutor
    requests_wrapper: RequestsWrapper

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
        llm: BaseLLM,
        json_spec: JsonSpec,
        requests_wrapper: RequestsWrapper,
        **kwargs: Any,
    ) -> OpenAPIToolkit:
        """Create json agent from llm, then initialize."""
        json_agent = create_json_agent(llm, JsonToolkit(spec=json_spec), **kwargs)
        return cls(json_agent=json_agent, requests_wrapper=requests_wrapper)
