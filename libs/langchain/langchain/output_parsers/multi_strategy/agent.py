"""Multi strategy parser that implements AgentOutputParser."""
from typing import Sequence, Union

from langchain.agents.agent import AgentOutputParser
from langchain.agents.conversational_chat.prompt import FORMAT_INSTRUCTIONS
from langchain.output_parsers.multi_strategy import strategies
from langchain.output_parsers.multi_strategy.base import (
    MultiStrategyParser,
    ParseStrategy,
)
from langchain.schema import (
    AgentAction,
    AgentFinish,
)

U = Union[AgentAction, AgentFinish]
TReactAgentOutput = U


class ConvMultiStrategyParser(MultiStrategyParser[U, dict], AgentOutputParser):
    """Multi strategy parser that implements AgentOutputParser."""

    def get_format_instructions(self) -> str:
        return FORMAT_INSTRUCTIONS

    def __init__(
        self, strategies: Sequence[ParseStrategy[dict]], **kwargs: dict
    ) -> None:
        super().__init__(strategies=strategies, **kwargs)

    def final_parse(self, text: str, parsed: dict) -> U:
        action, action_input = parsed["action"], parsed["action_input"]
        if action == "Final Answer":
            return AgentFinish({"output": action_input}, text)
        else:
            return AgentAction(action, action_input, text)


default_parser = ConvMultiStrategyParser(strategies.json_react_strategies)
