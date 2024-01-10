import asyncio
from abc import abstractmethod
from typing import List, Union

from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import (
    BaseMessage,
)
from langchain_core.outputs import ChatGeneration, Generation

from langchain.agents.agent import AgentOutputParser


class BaseFunctionsAgentOutputParser(AgentOutputParser):
    """Parses a message into agent action/finish.

    Is meant to be used with OpenAI models, as it relies on the specific
    function_call parameter from OpenAI to convey what tools to use.

    If a function_call parameter is passed, then that is used to get
    the tool and tool input.

    If one is not passed, then the AIMessage is assumed to be the final output.
    """

    @staticmethod
    @abstractmethod
    def _parse_ai_message(message: BaseMessage) -> Union[AgentAction, AgentFinish]:
        """Parse an AI message."""

    def parse_result(
        self, result: List[Generation], *, partial: bool = False
    ) -> Union[AgentAction, AgentFinish]:
        if not isinstance(result[0], ChatGeneration):
            raise ValueError("This output parser only works on ChatGeneration output")
        message = result[0].message
        return self._parse_ai_message(message)

    async def aparse_result(
        self, result: List[Generation], *, partial: bool = False
    ) -> Union[AgentAction, AgentFinish]:
        return await asyncio.get_running_loop().run_in_executor(
            None, self.parse_result, result
        )

    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        raise ValueError("Can only parse messages")

    @property
    @abstractmethod
    def _type(self) -> str:
        """Returns type of the AgentOutputParser"""
