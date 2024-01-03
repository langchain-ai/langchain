"""Robocorp Action Server toolkit."""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin

import requests
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.callbacks.manager import CallbackManager
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_core.tools import BaseTool, Tool
from langchain_core.tracers.context import _tracing_v2_is_enabled
from langsmith import Client

from langchain_robocorp._common import (
    get_required_param_descriptions,
    reduce_openapi_spec,
)
from langchain_robocorp._prompts import (
    API_CONTROLLER_PROMPT,
    TOOLKIT_TOOL_DESCRIPTION,
)

MAX_RESPONSE_LENGTH = 5000
LLM_TRACE_HEADER = "X-action-trace"


class RunDetailsCallbackHandler(BaseCallbackHandler):
    def __init__(self, run_details: dict) -> None:
        self.run_details = run_details

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        **kwargs: Any,
    ) -> None:
        if "parent_run_id" in kwargs:
            self.run_details["run_id"] = kwargs["parent_run_id"]
        else:
            if "run_id" in self.run_details:
                self.run_details.pop("run_id")


class ToolInputSchema(BaseModel):
    question: str = Field(...)


class ActionServerRequestTool(BaseTool):
    """Requests POST tool with LLM-instructed extraction of truncated responses."""

    name: str = "action_server_request"
    """Tool name."""
    description: str = "Useful to make requests to Action Server API"
    """Tool description."""
    response_length: Optional[int] = MAX_RESPONSE_LENGTH
    """Maximum length of the response to be returned."""
    run_details: dict
    """Request API key"""
    api_key: str
    """Action Server API key"""
    report_trace: bool
    """Should requests to Action Server include Langsmith trace, if available"""

    def _run(
        self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            json_text = query[query.find("{") : query.rfind("}") + 1]
            data = json.loads(json_text)

        except json.JSONDecodeError as e:
            raise e

        try:
            if self.report_trace and "run_id" in self.run_details:
                client = Client()
                run = client.read_run(self.run_details["run_id"])
                if run.url:
                    headers[LLM_TRACE_HEADER] = run.url
        except Exception:
            pass

        response = requests.post(
            data["url"], headers=headers, data=json.dumps(data["data"])
        )
        output = response.text[: self.response_length]

        return output

    async def _arun(self, text: str) -> str:
        raise NotImplementedError()


class ActionServerToolkit(BaseModel):
    """Toolkit exposing Robocorp Action Server provided actions as individual tools."""

    url: str = Field(exclude=True)
    """Action Server URL"""
    llm: BaseChatModel
    """Chat model to be used for the Toolkit agent"""
    api_key: str = Field(exclude=True, default="")
    """Action Server request API key"""
    report_trace: bool = Field(exclude=True, default=False)
    """Enable reporting Langsmith trace to Action Server runs"""

    class Config:
        arbitrary_types_allowed = True

    def get_tools(self, **kwargs: Any) -> List[BaseTool]:
        """Get the tools in the toolkit."""

        # Fetch and format the API spec
        try:
            spec_url = urljoin(self.url, "openapi.json")
            response = requests.get(spec_url)
            json_spec = response.json()
            api_spec = reduce_openapi_spec(self.url, json_spec)
        except Exception:
            raise ValueError(
                f"Failed to fetch OpenAPI schema from Action Server - {self.url}"
            )

        # Prepare request tools
        run_details: dict = {}

        request_tool = ActionServerRequestTool(
            run_details=run_details,
            report_trace=self.report_trace,
            api_key=self.api_key,
        )

        # Prepare callback manager
        callback_manager = kwargs.get("callback_manager", CallbackManager([]))
        callbacks: List[BaseCallbackHandler] = []

        if _tracing_v2_is_enabled():
            callbacks.append(RunDetailsCallbackHandler(run_details))

        for callback in callbacks:
            callback_manager.add_handler(callback)

        # Prepare the toolkit
        toolkit: List[BaseTool] = []

        prompt_variables = {
            "api_url": self.url,
        }

        for name, _, docs in api_spec.endpoints:
            if not name.startswith("/api/actions"):
                continue

            tool_name = f"robocorp_action_server_{docs['operationId']}"
            tool_description = TOOLKIT_TOOL_DESCRIPTION.format(
                name=docs["summary"],
                description=docs["description"],
                required_params=get_required_param_descriptions(docs),
            )

            prompt_variables["api_docs"] = f"{name}: \n{json.dumps(docs, indent=4)}"

            prompt = PromptTemplate(
                template=API_CONTROLLER_PROMPT,
                input_variables=["input"],
                partial_variables=prompt_variables,
            )

            chain: Runnable = (
                {"input": RunnablePassthrough()}
                | prompt
                | self.llm
                | StrOutputParser()
                | request_tool
            )

            toolkit.append(
                Tool(
                    name=tool_name,
                    func=chain.invoke,
                    description=tool_description,
                    args_schema=ToolInputSchema,
                    callback_manager=callback_manager,
                )
            )

        return toolkit
