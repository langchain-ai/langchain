"""Agent that interacts with OpenAPI APIs via a hierarchical planning approach."""

import json
import re
from functools import partial
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, cast

import yaml
from langchain_core.callbacks import BaseCallbackManager
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import BasePromptTemplate, PromptTemplate
from langchain_core.tools import BaseTool, Tool
from pydantic import Field

from langchain_community.agent_toolkits.openapi.planner_prompt import (
    API_CONTROLLER_PROMPT,
    API_CONTROLLER_TOOL_DESCRIPTION,
    API_CONTROLLER_TOOL_NAME,
    API_ORCHESTRATOR_PROMPT,
    API_PLANNER_PROMPT,
    API_PLANNER_TOOL_DESCRIPTION,
    API_PLANNER_TOOL_NAME,
    PARSING_DELETE_PROMPT,
    PARSING_GET_PROMPT,
    PARSING_PATCH_PROMPT,
    PARSING_POST_PROMPT,
    PARSING_PUT_PROMPT,
    REQUESTS_DELETE_TOOL_DESCRIPTION,
    REQUESTS_GET_TOOL_DESCRIPTION,
    REQUESTS_PATCH_TOOL_DESCRIPTION,
    REQUESTS_POST_TOOL_DESCRIPTION,
    REQUESTS_PUT_TOOL_DESCRIPTION,
)
from langchain_community.agent_toolkits.openapi.spec import ReducedOpenAPISpec
from langchain_community.llms import OpenAI
from langchain_community.tools.requests.tool import BaseRequestsTool
from langchain_community.utilities.requests import RequestsWrapper

#
# Requests tools with LLM-instructed extraction of truncated responses.
#
# Of course, truncating so bluntly may lose a lot of valuable
# information in the response.
# However, the goal for now is to have only a single inference step.
MAX_RESPONSE_LENGTH = 5000
"""Maximum length of the response to be returned."""

Operation = Literal["GET", "POST", "PUT", "DELETE", "PATCH"]


def _get_default_llm_chain(prompt: BasePromptTemplate) -> Any:
    from langchain.chains.llm import LLMChain

    return LLMChain(
        llm=OpenAI(),
        prompt=prompt,
    )


def _get_default_llm_chain_factory(
    prompt: BasePromptTemplate,
) -> Callable[[], Any]:
    """Returns a default LLMChain factory."""
    return partial(_get_default_llm_chain, prompt)


class RequestsGetToolWithParsing(BaseRequestsTool, BaseTool):  # type: ignore[override]
    """Requests GET tool with LLM-instructed extraction of truncated responses."""

    name: str = "requests_get"
    """Tool name."""
    description: str = REQUESTS_GET_TOOL_DESCRIPTION
    """Tool description."""
    response_length: int = MAX_RESPONSE_LENGTH
    """Maximum length of the response to be returned."""
    llm_chain: Any = Field(
        default_factory=_get_default_llm_chain_factory(PARSING_GET_PROMPT)
    )
    """LLMChain used to extract the response."""

    def _run(self, text: str) -> str:
        from langchain.output_parsers.json import parse_json_markdown

        try:
            data = parse_json_markdown(text)
        except json.JSONDecodeError as e:
            raise e
        data_params = data.get("params")
        response: str = cast(
            str, self.requests_wrapper.get(data["url"], params=data_params)
        )
        response = response[: self.response_length]
        return self.llm_chain.predict(
            response=response, instructions=data["output_instructions"]
        ).strip()

    async def _arun(self, text: str) -> str:
        raise NotImplementedError()


class RequestsPostToolWithParsing(BaseRequestsTool, BaseTool):  # type: ignore[override]
    """Requests POST tool with LLM-instructed extraction of truncated responses."""

    name: str = "requests_post"
    """Tool name."""
    description: str = REQUESTS_POST_TOOL_DESCRIPTION
    """Tool description."""
    response_length: int = MAX_RESPONSE_LENGTH
    """Maximum length of the response to be returned."""
    llm_chain: Any = Field(
        default_factory=_get_default_llm_chain_factory(PARSING_POST_PROMPT)
    )
    """LLMChain used to extract the response."""

    def _run(self, text: str) -> str:
        from langchain.output_parsers.json import parse_json_markdown

        try:
            data = parse_json_markdown(text)
        except json.JSONDecodeError as e:
            raise e
        response: str = cast(str, self.requests_wrapper.post(data["url"], data["data"]))
        response = response[: self.response_length]
        return self.llm_chain.predict(
            response=response, instructions=data["output_instructions"]
        ).strip()

    async def _arun(self, text: str) -> str:
        raise NotImplementedError()


class RequestsPatchToolWithParsing(BaseRequestsTool, BaseTool):  # type: ignore[override]
    """Requests PATCH tool with LLM-instructed extraction of truncated responses."""

    name: str = "requests_patch"
    """Tool name."""
    description: str = REQUESTS_PATCH_TOOL_DESCRIPTION
    """Tool description."""
    response_length: int = MAX_RESPONSE_LENGTH
    """Maximum length of the response to be returned."""
    llm_chain: Any = Field(
        default_factory=_get_default_llm_chain_factory(PARSING_PATCH_PROMPT)
    )
    """LLMChain used to extract the response."""

    def _run(self, text: str) -> str:
        from langchain.output_parsers.json import parse_json_markdown

        try:
            data = parse_json_markdown(text)
        except json.JSONDecodeError as e:
            raise e
        response: str = cast(
            str, self.requests_wrapper.patch(data["url"], data["data"])
        )
        response = response[: self.response_length]
        return self.llm_chain.predict(
            response=response, instructions=data["output_instructions"]
        ).strip()

    async def _arun(self, text: str) -> str:
        raise NotImplementedError()


class RequestsPutToolWithParsing(BaseRequestsTool, BaseTool):  # type: ignore[override]
    """Requests PUT tool with LLM-instructed extraction of truncated responses."""

    name: str = "requests_put"
    """Tool name."""
    description: str = REQUESTS_PUT_TOOL_DESCRIPTION
    """Tool description."""
    response_length: int = MAX_RESPONSE_LENGTH
    """Maximum length of the response to be returned."""
    llm_chain: Any = Field(
        default_factory=_get_default_llm_chain_factory(PARSING_PUT_PROMPT)
    )
    """LLMChain used to extract the response."""

    def _run(self, text: str) -> str:
        from langchain.output_parsers.json import parse_json_markdown

        try:
            data = parse_json_markdown(text)
        except json.JSONDecodeError as e:
            raise e
        response: str = cast(str, self.requests_wrapper.put(data["url"], data["data"]))
        response = response[: self.response_length]
        return self.llm_chain.predict(
            response=response, instructions=data["output_instructions"]
        ).strip()

    async def _arun(self, text: str) -> str:
        raise NotImplementedError()


class RequestsDeleteToolWithParsing(BaseRequestsTool, BaseTool):  # type: ignore[override]
    """Tool that sends a DELETE request and parses the response."""

    name: str = "requests_delete"
    """The name of the tool."""
    description: str = REQUESTS_DELETE_TOOL_DESCRIPTION
    """The description of the tool."""

    response_length: Optional[int] = MAX_RESPONSE_LENGTH
    """The maximum length of the response."""
    llm_chain: Any = Field(
        default_factory=_get_default_llm_chain_factory(PARSING_DELETE_PROMPT)
    )
    """The LLM chain used to parse the response."""

    def _run(self, text: str) -> str:
        from langchain.output_parsers.json import parse_json_markdown

        try:
            data = parse_json_markdown(text)
        except json.JSONDecodeError as e:
            raise e
        response: str = cast(str, self.requests_wrapper.delete(data["url"]))
        response = response[: self.response_length]
        return self.llm_chain.predict(
            response=response, instructions=data["output_instructions"]
        ).strip()

    async def _arun(self, text: str) -> str:
        raise NotImplementedError()


#
# Orchestrator, planner, controller.
#
def _create_api_planner_tool(
    api_spec: ReducedOpenAPISpec, llm: BaseLanguageModel
) -> Tool:
    from langchain.chains.llm import LLMChain

    endpoint_descriptions = [
        f"{name} {description}" for name, description, _ in api_spec.endpoints
    ]
    prompt = PromptTemplate(
        template=API_PLANNER_PROMPT,
        input_variables=["query"],
        partial_variables={"endpoints": "- " + "- ".join(endpoint_descriptions)},
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    tool = Tool(
        name=API_PLANNER_TOOL_NAME,
        description=API_PLANNER_TOOL_DESCRIPTION,
        func=chain.run,
    )
    return tool


def _create_api_controller_agent(
    api_url: str,
    api_docs: str,
    requests_wrapper: RequestsWrapper,
    llm: BaseLanguageModel,
    allow_dangerous_requests: bool,
    allowed_operations: Sequence[Operation],
) -> Any:
    from langchain.agents.agent import AgentExecutor
    from langchain.agents.mrkl.base import ZeroShotAgent
    from langchain.chains.llm import LLMChain

    tools: List[BaseTool] = []
    if "GET" in allowed_operations:
        get_llm_chain = LLMChain(llm=llm, prompt=PARSING_GET_PROMPT)
        tools.append(
            RequestsGetToolWithParsing(  # type: ignore[call-arg]
                requests_wrapper=requests_wrapper,
                llm_chain=get_llm_chain,
                allow_dangerous_requests=allow_dangerous_requests,
            )
        )
    if "POST" in allowed_operations:
        post_llm_chain = LLMChain(llm=llm, prompt=PARSING_POST_PROMPT)
        tools.append(
            RequestsPostToolWithParsing(  # type: ignore[call-arg]
                requests_wrapper=requests_wrapper,
                llm_chain=post_llm_chain,
                allow_dangerous_requests=allow_dangerous_requests,
            )
        )
    if "PUT" in allowed_operations:
        put_llm_chain = LLMChain(llm=llm, prompt=PARSING_PUT_PROMPT)
        tools.append(
            RequestsPutToolWithParsing(  # type: ignore[call-arg]
                requests_wrapper=requests_wrapper,
                llm_chain=put_llm_chain,
                allow_dangerous_requests=allow_dangerous_requests,
            )
        )
    if "DELETE" in allowed_operations:
        delete_llm_chain = LLMChain(llm=llm, prompt=PARSING_DELETE_PROMPT)
        tools.append(
            RequestsDeleteToolWithParsing(  # type: ignore[call-arg]
                requests_wrapper=requests_wrapper,
                llm_chain=delete_llm_chain,
                allow_dangerous_requests=allow_dangerous_requests,
            )
        )
    if "PATCH" in allowed_operations:
        patch_llm_chain = LLMChain(llm=llm, prompt=PARSING_PATCH_PROMPT)
        tools.append(
            RequestsPatchToolWithParsing(  # type: ignore[call-arg]
                requests_wrapper=requests_wrapper,
                llm_chain=patch_llm_chain,
                allow_dangerous_requests=allow_dangerous_requests,
            )
        )
    if not tools:
        raise ValueError("Tools not found")
    prompt = PromptTemplate(
        template=API_CONTROLLER_PROMPT,
        input_variables=["input", "agent_scratchpad"],
        partial_variables={
            "api_url": api_url,
            "api_docs": api_docs,
            "tool_names": ", ".join([tool.name for tool in tools]),
            "tool_descriptions": "\n".join(
                [f"{tool.name}: {tool.description}" for tool in tools]
            ),
        },
    )
    agent = ZeroShotAgent(
        llm_chain=LLMChain(llm=llm, prompt=prompt),
        allowed_tools=[tool.name for tool in tools],
    )
    return AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)


def _create_api_controller_tool(
    api_spec: ReducedOpenAPISpec,
    requests_wrapper: RequestsWrapper,
    llm: BaseLanguageModel,
    allow_dangerous_requests: bool,
    allowed_operations: Sequence[Operation],
) -> Tool:
    """Expose controller as a tool.

    The tool is invoked with a plan from the planner, and dynamically
    creates a controller agent with relevant documentation only to
    constrain the context.
    """

    base_url = api_spec.servers[0]["url"]  # TODO: do better.

    def _create_and_run_api_controller_agent(plan_str: str) -> str:
        pattern = r"\b(GET|POST|PATCH|DELETE|PUT)\s+(/\S+)*"
        matches = re.findall(pattern, plan_str)
        endpoint_names = [
            "{method} {route}".format(method=method, route=route.split("?")[0])
            for method, route in matches
        ]
        docs_str = ""
        for endpoint_name in endpoint_names:
            found_match = False
            for name, _, docs in api_spec.endpoints:
                regex_name = re.compile(re.sub("\\{.*?\\}", ".*", name))
                if regex_name.match(endpoint_name):
                    found_match = True
                    docs_str += f"== Docs for {endpoint_name} == \n{yaml.dump(docs)}\n"
            if not found_match:
                raise ValueError(f"{endpoint_name} endpoint does not exist.")

        agent = _create_api_controller_agent(
            base_url,
            docs_str,
            requests_wrapper,
            llm,
            allow_dangerous_requests,
            allowed_operations,
        )
        return agent.run(plan_str)

    return Tool(
        name=API_CONTROLLER_TOOL_NAME,
        func=_create_and_run_api_controller_agent,
        description=API_CONTROLLER_TOOL_DESCRIPTION,
    )


def create_openapi_agent(
    api_spec: ReducedOpenAPISpec,
    requests_wrapper: RequestsWrapper,
    llm: BaseLanguageModel,
    shared_memory: Optional[Any] = None,
    callback_manager: Optional[BaseCallbackManager] = None,
    verbose: bool = True,
    agent_executor_kwargs: Optional[Dict[str, Any]] = None,
    allow_dangerous_requests: bool = False,
    allowed_operations: Sequence[Operation] = ("GET", "POST"),
    **kwargs: Any,
) -> Any:
    """Construct an OpenAI API planner and controller for a given spec.

    Inject credentials via requests_wrapper.

    We use a top-level "orchestrator" agent to invoke the planner and controller,
    rather than a top-level planner
    that invokes a controller with its plan. This is to keep the planner simple.

    You need to set allow_dangerous_requests to True to use Agent with BaseRequestsTool.
    Requests can be dangerous and can lead to security vulnerabilities.
    For example, users can ask a server to make a request to an internal
    server. It's recommended to use requests through a proxy server
    and avoid accepting inputs from untrusted sources without proper sandboxing.
    Please see: https://python.langchain.com/docs/security
    for further security information.

    Args:
        api_spec: The OpenAPI spec.
        requests_wrapper: The requests wrapper.
        llm: The language model.
        shared_memory: Optional. The shared memory. Default is None.
        callback_manager: Optional. The callback manager. Default is None.
        verbose: Optional. Whether to print verbose output. Default is True.
        agent_executor_kwargs: Optional. Additional keyword arguments
            for the agent executor.
        allow_dangerous_requests: Optional. Whether to allow dangerous requests.
            Default is False.
        allowed_operations: Optional. The allowed operations.
            Default is ("GET", "POST").
        kwargs: Additional arguments.

    Returns:
        The agent executor.
    """
    from langchain.agents.agent import AgentExecutor
    from langchain.agents.mrkl.base import ZeroShotAgent
    from langchain.chains.llm import LLMChain

    tools = [
        _create_api_planner_tool(api_spec, llm),
        _create_api_controller_tool(
            api_spec,
            requests_wrapper,
            llm,
            allow_dangerous_requests,
            allowed_operations,
        ),
    ]
    prompt = PromptTemplate(
        template=API_ORCHESTRATOR_PROMPT,
        input_variables=["input", "agent_scratchpad"],
        partial_variables={
            "tool_names": ", ".join([tool.name for tool in tools]),
            "tool_descriptions": "\n".join(
                [f"{tool.name}: {tool.description}" for tool in tools]
            ),
        },
    )
    agent = ZeroShotAgent(
        llm_chain=LLMChain(llm=llm, prompt=prompt, memory=shared_memory),
        allowed_tools=[tool.name for tool in tools],
        **kwargs,
    )
    return AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        callback_manager=callback_manager,
        verbose=verbose,
        **(agent_executor_kwargs or {}),
    )
