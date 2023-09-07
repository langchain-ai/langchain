from __future__ import annotations

import json
from typing import List, Sequence

from langchain.automaton.chat_agent import ChatAgent
from langchain.automaton.typedefs import (
    AgentFinish,
    FunctionCall,
    FunctionResult,
    MessageLike,
)
from langchain.chat_models.openai import ChatOpenAI
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain.schema import AIMessage, BaseMessage, FunctionMessage, Generation
from langchain.schema.output_parser import BaseGenerationOutputParser
from langchain.tools import BaseTool, format_tool_to_openai_function


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

        return FunctionCall(
            name=function_request["name"],
            named_arguments=function_request["arguments"],
        )


def prompt_generator(input_messages: Sequence[MessageLike]) -> List[BaseMessage]:
    """Generate a prompt from a log of message like objects."""
    messages = []
    for message in input_messages:
        if isinstance(message, BaseMessage):
            messages.append(message)
        elif isinstance(message, FunctionResult):
            messages.append(
                FunctionMessage(name=message.name, content=json.dumps(message.result))
            )
        else:
            pass
    return messages


def create_openai_agent(llm: ChatOpenAI, tools: Sequence[BaseTool]) -> ChatAgent:
    """Create an agent that uses OpenAI's API."""
    openai_funcs = [format_tool_to_openai_function(tool_) for tool_ in tools]
    return ChatAgent(
        llm.bind(funcions=openai_funcs),
        prompt_generator=prompt_generator,
        tools=tools,
        parser=OpenAIFunctionsParser(),
    )
