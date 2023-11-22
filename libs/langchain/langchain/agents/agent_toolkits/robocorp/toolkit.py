"""Robocorp Action Server toolkit."""
from __future__ import annotations

from typing import List, Optional, Type
import requests
import json

from langchain.agents.agent_toolkits.base import BaseToolkit
from langchain.tools import BaseTool
from langchain.chains.llm import LLMChain
from langchain.tools.requests.tool import BaseRequestsTool
from langchain.agents.agent_toolkits.robocorp.prompts import (
    API_CONTROLLER_PROMPT,
    REQUESTS_GET_TOOL_DESCRIPTION,
    REQUESTS_POST_TOOL_DESCRIPTION,
    REQUESTS_RESPONSE_PROMPT,
    TOOLKIT_TOOL_DESCRIPTION,
)
from langchain.agents.agent_toolkits.robocorp.spec import (
    reduce_openapi_spec,
    get_required_param_descriptions,
)
from langchain.pydantic_v1 import BaseModel, Field
from langchain.schema.language_model import BaseLanguageModel
from langchain.utilities.requests import RequestsWrapper
from langchain.prompts import PromptTemplate
from langchain.agents.mrkl.base import ZeroShotAgent
from langchain.agents.agent import AgentExecutor
from langchain.agents.tools import Tool

MAX_RESPONSE_LENGTH = 5000


class ToolInputSchema(BaseModel):
    question: str = Field(...)


class RequestsGetToolWithParsing(BaseRequestsTool, BaseTool):
    """Requests GET tool with LLM-instructed extraction of truncated responses."""

    name: str = "requests_get"
    """Tool name."""
    description = REQUESTS_GET_TOOL_DESCRIPTION
    """Tool description."""
    response_length: Optional[int] = MAX_RESPONSE_LENGTH
    """Maximum length of the response to be returned."""
    llm_chain: LLMChain
    """LLMChain used to extract the response."""

    def _run(self, text: str) -> str:
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise e
        data_params = data.get("params")
        response = self.requests_wrapper.get(data["url"], params=data_params)
        response = response[: self.response_length]
        return self.llm_chain.predict(
            response=response, instructions=data["output_instructions"]
        ).strip()

    async def _arun(self, text: str) -> str:
        raise NotImplementedError()


class RequestsPostToolWithParsing(BaseRequestsTool, BaseTool):
    """Requests POST tool with LLM-instructed extraction of truncated responses."""

    name: str = "requests_post"
    """Tool name."""
    description = REQUESTS_POST_TOOL_DESCRIPTION
    """Tool description."""
    response_length: Optional[int] = MAX_RESPONSE_LENGTH
    """Maximum length of the response to be returned."""
    llm_chain: LLMChain
    """LLMChain used to extract the response."""

    def _run(self, text: str) -> str:
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise e
        response = self.requests_wrapper.post(data["url"], data["data"])
        response = response[: self.response_length]
        return self.llm_chain.predict(
            response=response, instructions=data["output_instructions"]
        ).strip()

    async def _arun(self, text: str) -> str:
        raise NotImplementedError()


class RobocorpToolkit(BaseToolkit):
    """Toolkit exposing Robocorp Action Server provided actions."""

    url: str = Field(exclude=True)
    llm: BaseLanguageModel = Field(exclude=True)

    class Config:
        arbitrary_types_allowed = True

    def get_tools(self) -> List[BaseTool]:
        # Fetch and format the API spec
        response = requests.get(f"{self.url}/openapi.json")
        json_data = response.json()
        api_spec = reduce_openapi_spec(json_data)

        # Prepare request tools
        llm_chain = LLMChain(llm=self.llm, prompt=REQUESTS_RESPONSE_PROMPT)

        requests_wrapper = RequestsWrapper(headers={})

        tools: List[BaseTool] = [
            RequestsGetToolWithParsing(
                requests_wrapper=requests_wrapper, llm_chain=llm_chain
            ),
            RequestsPostToolWithParsing(
                requests_wrapper=requests_wrapper, llm_chain=llm_chain
            ),
        ]

        tool_names = ", ".join([tool.name for tool in tools])
        tool_descriptions = "\n".join(
            [f"{tool.name}: {tool.description}" for tool in tools]
        )

        toolkit: List[BaseTool] = []

        prompt_variables = {
            "api_url": self.url,
            "tool_names": tool_names,
            "tool_descriptions": tool_descriptions,
        }

        # Prepare the toolkit
        for name, _, docs in api_spec.endpoints:
            tool_name = f"robocorp_action_server_{docs['operationId']}"
            tool_description = TOOLKIT_TOOL_DESCRIPTION.format(
                name=docs["summary"],
                description=docs["description"],
                required_params=get_required_param_descriptions(docs),
            )

            prompt_variables["api_docs"] = f"{name}: \n{json.dumps(docs, indent=4)}"
            prompt = PromptTemplate(
                template=API_CONTROLLER_PROMPT,
                input_variables=["input", "agent_scratchpad"],
                partial_variables=prompt_variables,
            )

            agent = ZeroShotAgent(
                llm_chain=LLMChain(llm=self.llm, prompt=prompt),
                allowed_tools=[tool.name for tool in tools],
            )

            executor = AgentExecutor.from_agent_and_tools(
                agent=agent, tools=tools, verbose=True
            )

            toolkit.append(
                Tool(
                    name=tool_name,
                    func=executor.run,
                    description=tool_description,
                    args_schema=ToolInputSchema,
                )
            )

        return toolkit
