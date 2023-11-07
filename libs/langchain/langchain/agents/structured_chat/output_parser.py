from __future__ import annotations

import json
import logging
import re
from typing import Optional, Union

from langchain.agents.agent import AgentOutputParser
from langchain.agents.structured_chat.prompt import FORMAT_INSTRUCTIONS
from langchain.output_parsers import OutputFixingParser
from langchain.pydantic_v1 import Field
from langchain.schema import AgentAction, AgentFinish, OutputParserException
from langchain.schema.language_model import BaseLanguageModel

logger = logging.getLogger(__name__)


class StructuredChatOutputParser(AgentOutputParser):
    """Output parser for the structured chat agent."""

    pattern = re.compile(r"```(?:json\s+)?(\W.*?)```", re.DOTALL)

    def get_format_instructions(self) -> str:
        return FORMAT_INSTRUCTIONS

    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        try:
            action_match = self.pattern.search(text)
            if action_match is not None:
                response = json.loads(action_match.group(1).strip(), strict=False)
                if isinstance(response, list):
                    # gpt turbo frequently ignores the directive to emit a single action
                    logger.warning("Got multiple action responses: %s", response)
                    response = response[0]
                if response["action"] == "Final Answer":
                    return AgentFinish({"output": response["action_input"]}, text)
                else:
                    return AgentAction(
                        response["action"], response.get("action_input", {}), text
                    )
            else:
                return AgentFinish({"output": text}, text)
        except Exception as e:
            raise OutputParserException(f"Could not parse LLM output: {text}") from e

    @property
    def _type(self) -> str:
        return "structured_chat"


class StructuredChatOutputParserWithRetries(AgentOutputParser):
    """Output parser with retries for the structured chat agent."""

    base_parser: AgentOutputParser = Field(default_factory=StructuredChatOutputParser)
    """The base parser to use."""
    output_fixing_parser: Optional[OutputFixingParser] = None
    """The output fixing parser to use."""

    def get_format_instructions(self) -> str:
        return FORMAT_INSTRUCTIONS

    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        try:
            if self.output_fixing_parser is not None:
                parsed_obj: Union[
                    AgentAction, AgentFinish
                ] = self.output_fixing_parser.parse(text)
            else:
                parsed_obj = self.base_parser.parse(text)
            return parsed_obj
        except Exception as e:
            raise OutputParserException(f"Could not parse LLM output: {text}") from e

    @classmethod
    def from_llm(
        cls,
        llm: Optional[BaseLanguageModel] = None,
        base_parser: Optional[StructuredChatOutputParser] = None,
    ) -> StructuredChatOutputParserWithRetries:
        if llm is not None:
            base_parser = base_parser or StructuredChatOutputParser()
            output_fixing_parser = OutputFixingParser.from_llm(
                llm=llm, parser=base_parser
            )
            return cls(output_fixing_parser=output_fixing_parser)
        elif base_parser is not None:
            return cls(base_parser=base_parser)
        else:
            return cls()

    @property
    def _type(self) -> str:
        return "structured_chat_with_retries"
