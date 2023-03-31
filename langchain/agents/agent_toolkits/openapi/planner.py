"""Agent that interacts with OpenAPI spec'd APIs via a hierarchical planning approach,
inspired by recent works on LLMs for robotics application.

We're iterating on approaches for solving multi-step queries against massive specs, so code's kept here self-contained for now.

Observations on robustness across GPTs:
- gpt-4 works very well with this approach; the advantage of the approach is separating computation so as not to eat too many tokens
  (i.e. not stuffing a large and largely irrelevant spec into context)
- gpt-3 completion models aren't as robust, but this approach allows for replanning at plan level + retrying at policy/step level, and
  there's tons of room for improvement.
- gpt-3 chat models haven't been tested in depth.
"""
from langchain.agents.agent_toolkits.openapi.spec import (
    reduce_openapi_spec,
    ReducedOpenAPISpec,
)
from langchain.agents.agent import AgentExecutor
from langchain.agents.mrkl.base import ZeroShotAgent
from langchain.agents.tools import Tool
from langchain.chains.llm import LLMChain
from langchain.llms.openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.requests import RequestsWrapper
from langchain.schema import BaseLanguageModel
from langchain.tools.base import BaseTool
from langchain.tools.requests.tool import BaseRequestsTool

import json
import re
import yaml
from typing import Optional


#
# Prompts.
#

# Planning.
# TODO: remove API-specific few shots.
API_PLANNER_PROMPT = """You are a planner that plans a sequence of API calls to assist with user queries against an API.  You have access to GET and POST endpoints, documented below.
Some user queries can be resolved in a single API call, but some will require several API calls.
You will generate a plan of API calls and explain what they are accomplishing step by step.
The plan will be passed to an API controller that can format them into web requests and return the responses.

----

Endpoints:

{endpoints}

----

Suppose we're interested in Spotify's API:

Good examples:

User query: Find me songs by Norah Jones.
Plan:
1. GET /search to get the artist ID for Norah Jones.
2. GET /artists/{{id}}/top-tracks to get the artist's top-tracks.

User query: Give me some jazz recommendations.
Plan:
1. GET /recommendations/available-genre-seeds to find valid recommendation seeds.
2. GET /recommendations to find recommendations.

Bad examples:

User query: Add the song Central Park West to my queue
Plan:
1. POST /me/player/queue to add the track to the user's queue
Explanation: This plan did not look up the track id for the song. It also did not look up information about the user.

----

User query: {query}
Plan:"""
API_PLANNER_TOOL_NAME = "api_planner"
API_PLANNER_TOOL_DESCRIPTION = f"Can be used to generate the right API calls to assist with a user query, like {API_PLANNER_TOOL_NAME}(query). Should always be called before trying to calling the API controller."

# Execution.
API_CONTROLLER_PROMPT = """You are an agent that gets a sequence of API calls and given their documentation, should execute them and return the final response.
If you cannot complete them and run into issues, you should explain the issue. When interacting with API objects, you should extract ids for inputs to other API calls but ids and names for outputs returned to the User.


Here is documentation on the API:
Base url: {api_url}
Endpoints:
{api_docs}


Here are tools to execute requests against the API: {tool_descriptions}


Starting below, you should follow this format:

Plan: the plan of API calls to execute
Thought: you should always think about what to do
Action: the action to take, should be one of the tools [{tool_names}]
Action Input: the input to the action
Observation: the output of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I am finished executing the plan (or, I cannot finish executing the plan without knowing some other information.)
Final Answer: the final output from executing the plan or missing information I'd need to re-plan correctly.


You can use US as a country code.

Begin!

Plan: {input}
Thought:
{agent_scratchpad}
"""
API_CONTROLLER_TOOL_NAME = "api_controller"
API_CONTROLLER_TOOL_DESCRIPTION = (
    f"Can be used to execute a plan of API calls, like {API_CONTROLLER_TOOL_NAME}(plan)."
)

# Orchestrate planning + execution.
# The goal is to have an agent at the top-level (e.g. so it can recover from errors and re-plan) while
# keeping planning (and specifically the planning prompt) simple.
API_ORCHESTRATOR_PROMPT = """You are an agent that assists with user queries against API, things like querying information or creating resources.
Some user queries can be resolved in a single API call though some require several API call.
You should always plan your API calls first, and then execute the plan second.
You should never return information without executing the api_controller tool.


Here are the tools to plan and execute API requests: {tool_descriptions}


Starting below, you should follow this format:

User query: the query a User wants help with related to the API
Thought: you should always think about what to do
Action: the action to take, should be one of the tools [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I am finished executing a plan and have the information the user asked for or the data the used asked to create
Final Answer: the final output from executing the plan


Example:
User query: can you add some trendy stuff to my shopping cart.
Thought: I should plan API calls first.
Action: api_planner
Action Input: I need to find the right API calls to add trendy items to the users shopping cart
Observation: 1) GET /items/trending to get trending item ids
2) GET /user to get user
3) POST /cart to post the trending items to the user's cart
Thought: I'm ready to execute the API calls.
Action: api_controller
Action Input: 1) GET /items/trending to get trending item ids
2) GET /user to get user
3) POST /cart to post the trending items to the user's cart
...

Begin!

User query: {input}
Thought: I should generate a plan to help with this query and then copy that plan exactly to the controller.
{agent_scratchpad}"""


#
# Requests tools with LLM-instructed extraction of truncated responses.
#
# Of course, truncating so bluntly may lose a lot of valuable information in the response.
# However, the goal for now is to have only a single inference step.
MAX_RESPONSE_LENGTH = 5000


class RequestsGetToolWithParsing(BaseRequestsTool, BaseTool):

    name = "requests_get"
    description = """Use this to GET content from a website.
Input to the tool should be a json string with 2 keys: "url" and "output_instructions".
The value of "url" should be a string. The value of "output_instructions" should be instructions on what information to extract from the response, for example the id(s) for a resource(s) that the GET request fetches.
"""
    response_length: Optional[int] = MAX_RESPONSE_LENGTH
    llm_chain = LLMChain(
        llm=OpenAI(),
        prompt=PromptTemplate(
            template="""Here is an API response:\n\n{response}\n\n====
Your task is to extract some information according to these instructions: {instructions}
When working with API objects, you should usually use ids over names.
If the response indicates an error, you should instead output a summary of the error.

Output:""",
            input_variables=["response", "instructions"],
        ),
    )

    def _run(self, text: str) -> str:
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise e
        response = self.requests_wrapper.get(data["url"])
        response = response[:self.response_length]
        return self.llm_chain.predict(
            response=response, instructions=data["output_instructions"]
        ).strip()

    async def _arun(self, text: str) -> str:
        raise NotImplementedError()


class RequestsPostToolWithParsing(BaseRequestsTool, BaseTool):

    name = "requests_post"
    description = """Use this when you want to POST to a website.
Input to the tool should be a json string with 3 keys: "url", "data", and "output_instructions".
The value of "url" should be a string.
The value of "data" should be a dictionary of key-value pairs you want to POST to the url.
The value of "summary_instructions" should be instructions on what information to extract from the response, for example the id(s) for a resource(s) that the POST request creates.
Always use double quotes for strings in the json string."""

    response_length: Optional[int] = MAX_RESPONSE_LENGTH
    llm_chain = LLMChain(
        llm=OpenAI(),
        prompt=PromptTemplate(
            template="""Here is an API response:\n\n{response}\n\n====
Your task is to extract some information according to these instructions: {instructions}
When working with API objects, you should usually use ids over names. Do not return any ids or names that are not in the response.
If the response indicates an error, you should instead output a summary of the error.

Output:""",
            input_variables=["response", "instructions"],
        ),
    )

    def _run(self, text: str) -> str:
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise e
        response = self.requests_wrapper.post(data["url"], data["data"])
        response = response[:self.response_length]
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
) -> AgentExecutor:
    tools = [
        RequestsGetToolWithParsing(requests_wrapper=requests_wrapper),
        RequestsPostToolWithParsing(requests_wrapper=requests_wrapper),
    ]
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
    api_spec: ReducedOpenAPISpec, requests_wrapper: RequestsWrapper, llm: BaseLanguageModel
) -> Tool:
    """Expose controller as a tool. The tool is invoked with a plan from the planner, and dynamically
    creates a controller agent with relevant documentation only to constrain the context.
    """

    base_url = api_spec.servers[0]["url"]  # TODO: do better.

    def _create_and_run_api_controller_agent(plan_str: str) -> str:
        pattern = r"\b(GET|POST)\s+(/\S+)*"
        matches = re.findall(pattern, plan_str)
        endpoint_names = [f"{method} {route}" for method, route in matches]
        endpoint_docs_by_name = {name: docs for name, _, docs in api_spec.endpoints}
        docs_str = ""
        for endpoint_name in endpoint_names:
            docs = endpoint_docs_by_name.get(endpoint_name)
            if not docs:
                raise ValueError(f"{endpoint_name} endpoint does not exist.")
            docs_str += f"== Docs for {endpoint_name} == \n{yaml.dump(docs)}\n"

        agent = _create_api_controller_agent(base_url, docs_str, requests_wrapper, llm)
        return agent.run(plan_str)

    return Tool(
        name=API_CONTROLLER_TOOL_NAME,
        func=_create_and_run_api_controller_agent,
        description=API_CONTROLLER_TOOL_DESCRIPTION,
    )


def create_openapi_agent(
    api_spec: ReducedOpenAPISpec, requests_wrapper: RequestsWrapper, llm: BaseLanguageModel
) -> AgentExecutor:
    """Instantiate API planner and controller for a given spec. Inject credentials via requests_wrapper.

    We use a top-level "orchestrator" agent to invoke the planner and controller, rather than a top-level planner
    that invokes a controller with its plan. This is to keep the planner simple.
    """
    tools = [
        _create_api_planner_tool(api_spec, llm),
        _create_api_controller_tool(api_spec, requests_wrapper, llm),
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
        llm_chain=LLMChain(
            llm=llm,
            prompt=prompt),
        allowed_tools=[tool.name for tool in tools],
    )
    return AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)
