from __future__ import annotations

import json
from typing import List, Sequence

from langchain.automaton.chat_agent import ChatAgent, SimpleAutomaton
from langchain.automaton.prompt_generator import AdapterBasedGenerator
from langchain.automaton.typedefs import (
    AgentFinish,
    FunctionCallRequest,
    FunctionCallResponse,
)
from langchain.chat_models.openai import ChatOpenAI
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain.schema import AIMessage, BaseMessage, FunctionMessage, Generation
from langchain.schema.language_model import (
    BaseLanguageModel,
    LanguageModelInput,
    LanguageModelOutput,
)
from langchain.schema.output_parser import BaseGenerationOutputParser
from langchain.schema.runnable import Runnable
from langchain.tools import BaseTool, format_tool_to_openai_function
from langchain.automaton.runnables import create_llm_program


class OpenAIFunctionsParser(BaseGenerationOutputParser):
    def parse_result(self, result: List[Generation]):
        if len(result) != 1:
            raise AssertionError(f"Expected exactly one result")
        first_result = result[0]

        message = first_result.message

        if not isinstance(message, AIMessage) or not message.additional_kwargs:
            return AgentFinish(result=message)

        parser = JsonOutputFunctionsParser(strict=False, args_only=False)
        try:
            function_request = parser.parse_result(result)
        except Exception as e:
            raise RuntimeError(f"Error parsing result: {result} {repr(e)}") from e

        return FunctionCallRequest(
            name=function_request["name"],
            named_arguments=function_request["arguments"],
        )


def create_openai_agent(llm: ChatOpenAI, tools: Sequence[BaseTool]) -> ChatAgent:
    """Create an agent that uses OpenAI's API."""
    openai_funcs = [format_tool_to_openai_function(tool_) for tool_ in tools]
    adapter = create_openai_functions_adapter()
    return ChatAgent(
        llm.bind(functions=openai_funcs),
        prompt_generator=adapter.to_messages,
        tools=tools,
        parser=OpenAIFunctionsParser(),
    )


def create_openai_agent_2(
    llm: BaseLanguageModel[LanguageModelInput, LanguageModelOutput]
    | Runnable[LanguageModelInput, LanguageModelOutput],
    tools: Sequence[BaseTool],
) -> SimpleAutomaton:
    """Create an agent that uses OpenAI's API."""
    openai_funcs = [format_tool_to_openai_function(tool_) for tool_ in tools]
    prompt_generator = AdapterBasedGenerator(
        msg_adapters={
            FunctionCallResponse: lambda message: FunctionMessage(
                name=message.name, content=json.dumps(message.result)
            ),
            # No need to translate function call requests
        },
    )
    llm_program = create_llm_program(
        llm.bind(functions=openai_funcs),
        prompt_generator=prompt_generator,
        tools=tools,
        parser=OpenAIFunctionsParser(),
    )
    return SimpleAutomaton(
        states={
            "program": llm_program,
        },
        router=lambda state, message: "program",
    )
