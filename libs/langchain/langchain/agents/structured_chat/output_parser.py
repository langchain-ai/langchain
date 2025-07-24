from __future__ import annotations

import json
import logging
import re
from re import Pattern
from typing import Optional, Union

from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseLanguageModel
from pydantic import Field

from langchain.agents.agent import AgentOutputParser
from langchain.agents.structured_chat.prompt import FORMAT_INSTRUCTIONS
from langchain.output_parsers import OutputFixingParser

logger = logging.getLogger(__name__)


class StructuredChatOutputParser(AgentOutputParser):
    """Output parser for the structured chat agent."""

    format_instructions: str = FORMAT_INSTRUCTIONS
    """Default formatting instructions"""

    pattern: Pattern = re.compile(r"```(?:json\s+)?(\W.*?)```", re.DOTALL)
    """Regex pattern to parse the output."""

    def get_format_instructions(self) -> str:
        """Returns formatting instructions for the given output parser."""
        return self.format_instructions

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
                return AgentAction(
                    response["action"],
                    response.get("action_input", {}),
                    text,
                )
            return AgentFinish({"output": text}, text)
        except Exception as e:
            msg = f"Could not parse LLM output: {text}"
            raise OutputParserException(msg) from e

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
                return self.output_fixing_parser.parse(text)
            return self.base_parser.parse(text)
        except Exception as e:
            msg = f"Could not parse LLM output: {text}"
            raise OutputParserException(msg) from e

    @classmethod
    def from_llm(
        cls,
        llm: Optional[BaseLanguageModel] = None,
        base_parser: Optional[StructuredChatOutputParser] = None,
    ) -> StructuredChatOutputParserWithRetries:
        if llm is not None:
            base_parser = base_parser or StructuredChatOutputParser()
            output_fixing_parser: OutputFixingParser = OutputFixingParser.from_llm(
                llm=llm,
                parser=base_parser,
            )
            return cls(output_fixing_parser=output_fixing_parser)
        if base_parser is not None:
            return cls(base_parser=base_parser)
        return cls()

    @property
    def _type(self) -> str:
        return "structured_chat_with_retries"
