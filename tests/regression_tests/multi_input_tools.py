"""Test the BaseOutputParser class and its sub-classes."""

from collections import defaultdict
import json
from copy import deepcopy
from pathlib import Path
from typing import Iterable, List, Tuple
from pydantic import ValidationError

import pytest

from langchain.agents import initialize_agent
from langchain.agents.agent_toolkits.json.toolkit import JsonToolkit
from langchain.agents.agent_toolkits.nla.toolkit import NLAToolkit
from langchain.agents.agent_toolkits.openapi.toolkit import RequestsToolkit
from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.llms.openai import OpenAI
from langchain.memory.buffer import ConversationBufferMemory
from langchain.requests import TextRequestsWrapper
from langchain.schema import BaseLanguageModel
from langchain.tools.base import BaseTool
from langchain.tools.json.tool import JsonSpec


def _get_requests_tools_and_questions(**kwargs) -> List[Tuple[BaseTool, List[str]]]:
    requests_wrapper = TextRequestsWrapper()
    requests_toolkit = RequestsToolkit(requests_wrapper=requests_wrapper)
    tools = requests_toolkit.get_tools()
    tools_dict = {tool.name: tool for tool in tools}
    method_to_questions = {
        # "get": ["Get the header of google.com"],
        "post": ["Post data {'key': 'value'} to google.com"],
        "patch": ["Patch data {'key': 'value'} to google.com"],
        "put": ["Put data {'key': 'value'} to google.com"],
        "delete": ["Delete data with ID 1234abc from google.com"],
    }
    results = []
    for method, qs in method_to_questions.items():
        results.append((tools_dict[f"requests_{method}"], qs))
    return results


def _get_json_tools_and_questions(**kwargs) -> List[Tuple[BaseTool, List[str]]]:
    spec = JsonSpec.from_file(
        Path("tests/unit_tests/tools/openapi/test_specs/apis-guru/apispec.json")
    )
    json_toolkit = JsonToolkit(spec=spec)
    list_keys, get_value = json_toolkit.get_tools()
    return [
        (list_keys, "What keys are in the JSON spec?"),
        (get_value, "What's in the info.description?"),
    ]


def _get_nla_tools_nad_questions(
    *,
    llm: BaseLanguageModel,
) -> List[Tuple[BaseTool, List[str]]]:
    speak_toolkit = NLAToolkit.from_llm_and_url(
        llm, "https://api.speak.com/openapi.yaml"
    )
    # TODO: make more pointed questions
    speak_tools_and_questions = [
        (tool, ["Could you help me learn something new in Spanish?"])
        for tool in speak_toolkit.get_tools()
    ]
    klarna_toolkit = NLAToolkit.from_llm_and_url(
        llm, "https://www.klarna.com/us/shopping/public/openai/v0/api-docs/"
    )
    klarna_tools_and_questions = [
        (tool, ["I want to buy some cheap shoes"])
        for tool in klarna_toolkit.get_tools()
    ]
    return speak_tools_and_questions + klarna_tools_and_questions


def generate_tuples() -> (
    List[Tuple[BaseTool, List[str], BaseLanguageModel, AgentType, bool]]
):
    """Grid test."""
    llms = [
        # ChatOpenAI(),
        OpenAI(),
    ]
    generators = [
        # _get_nla_tools_nad_questions,
        # _get_json_tools_and_questions,
        _get_requests_tools_and_questions,
    ]
    # These types don't really support arbitrary single tools...
    # excluded_types = (AgentType.SELF_ASK_WITH_SEARCH, AgentType.REACT_DOCSTORE)

    # agent_types = [
    #     agent_type for agent_type in AgentType if agent_type not in excluded_types
    # ]
    agent_types = [
        # AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    ]
    results = []
    for llm in llms:
        for agent_type in agent_types:
            for generator in generators:
                tools_and_queries = generator(llm=llm)
                for tool, queries in tools_and_queries:
                    for append_args_to_description in [True, False]:
                        results.append(
                            (tool, queries, llm, agent_type, append_args_to_description)
                        )
    return results


_AGGREGATE_AXES = ["tool", "llm", "agent_type", "append_args_to_description"]
_FAILURE_COUNT = {k: defaultdict(int) for k in _AGGREGATE_AXES}


@pytest.mark.parametrize(
    "tool, queries, llm, agent_type, append_args_to_description", generate_tuples()
)
def test_run_tool(
    tool: BaseTool,
    queries: List[str],
    llm: BaseLanguageModel,
    agent_type: AgentType,
    append_args_to_description: bool,
) -> None:
    global _FAILURE_COUNT
    tool = deepcopy(tool)
    if append_args_to_description:
        # Double up brackets to permit injection into the prompt template
        args = json.dumps(tool.args).replace("{", "{{").replace("}", "}}")
        tool.description += args

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    agent = initialize_agent(
        llm=llm,
        tools=[tool],
        agent=agent_type,
        memory=memory,
        verbose=True,
    )
    results = []
    for query in queries:
        try:
            result = agent(query)
            results.append(result)
        except Exception as e:
            results.append(e)

    type_errors = [r for r in results if isinstance(r, TypeError)]
    if type_errors:
        print(f"{str(llm)}: {tool.name} failed with: {type_errors}")
        _FAILURE_COUNT["tool"][tool.name] += 1
        _FAILURE_COUNT["llm"][str(llm)] += 1
        _FAILURE_COUNT["agent_type"][str(agent_type)] += 1
        _FAILURE_COUNT["append_args_to_description"][
            str(append_args_to_description)
        ] += 1

    assert not type_errors, type_errors
    validation_errors = [r for r in results if isinstance(r, ValidationError)]
    assert not validation_errors, validation_errors
    import pdb

    pdb.set_trace()
