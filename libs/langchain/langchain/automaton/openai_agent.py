"""Specialized open ai functions based agent."""
from __future__ import annotations

import json
from typing import Sequence, List, Iterator

from langchain.automaton.runnables import create_llm_program
from langchain.automaton.typedefs import (
    AgentFinish,
    FunctionCall,
    FunctionResult,
    MessageLike,
)
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain.schema import Generation
from langchain.schema.language_model import BaseLanguageModel
from langchain.schema.messages import BaseMessage, FunctionMessage, AIMessage
from langchain.schema.output_parser import BaseGenerationOutputParser
from langchain.tools import BaseTool


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


class OpenAIAgent:
    def __init__(
        self,
        llm: BaseLanguageModel,
        tools: Sequence[BaseTool],
    ) -> None:
        """Initialize the chat automaton."""
        self.llm_program = create_llm_program(
            llm,
            prompt_generator=prompt_generator,
            tools=tools,
            parser=OpenAIFunctionsParser(),
        )

    def invoke(
        self,
        messages: Sequence[MessageLike],
        *,
        max_iterations: int = 100,
    ) -> Iterator[MessageLike]:
        """Run the agent."""
        all_messages = list(messages)
        for _ in range(max_iterations):
            if all_messages and isinstance(all_messages[-1], AgentFinish):
                break

            new_messages = self.llm_program.invoke(all_messages)
            yield from new_messages
            all_messages.extend(new_messages)

