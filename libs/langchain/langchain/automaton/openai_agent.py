from __future__ import annotations

from typing import Sequence, List

from langchain.automaton.prompt_generators import MessageLogPromptValue
from langchain.automaton.runnables import create_llm_program
from langchain.automaton.typedefs import (
    MessageLog,
    AgentFinish,
    FunctionCall,
)
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain.schema import Generation
from langchain.schema.language_model import BaseLanguageModel
from langchain.schema.messages import AIMessage
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
            arguments=function_request["arguments"],
        )


class OpenAIAgent:
    def __init__(
        self,
        llm: BaseLanguageModel,
        tools: Sequence[BaseTool],
        *,
        max_iterations: int = 10,
    ) -> None:
        """Initialize the chat automaton."""
        self.llm_program = create_llm_program(
            llm,
            prompt_generator=MessageLogPromptValue.from_message_log,
            tools=tools,
            parser=OpenAIFunctionsParser(),
        )
        self.max_iterations = max_iterations

    def run(self, message_log: MessageLog) -> None:
        """Run the agent."""
        if not message_log:
            raise AssertionError(f"Expected at least one message in message_log")

        for _ in range(self.max_iterations):
            last_message = message_log[-1]

            if isinstance(last_message, AgentFinish):
                break

            messages = self.llm_program.invoke(message_log)
            message_log.add_messages(messages)
