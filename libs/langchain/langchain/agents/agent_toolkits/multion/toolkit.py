"""MultiOn agent."""
from __future__ import annotations

from typing import TYPE_CHECKING, List

from pydantic import Field

from langchain.agents.agent_toolkits.base import BaseToolkit
from langchain.tools import BaseTool

from langchain.agents.agent import AgentExecutor, BaseSingleActionAgent
from langchain.agents.agent_toolkits.python.prompt import PREFIX
from langchain.agents.mrkl.base import ZeroShotAgent
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain.agents.types import AgentType
from langchain.base_language import BaseLanguageModel
from langchain.callbacks.base import BaseCallbackManager
from langchain.chains.llm import LLMChain
from langchain.schema import SystemMessage

from langchain.tools.multion.create_session import MultionCreateSession
from langchain.tools.multion.update_session import MultionUpdateSession





class MultionToolkit(BaseToolkit):
    """Toolkit for interacting with the Browser Agent"""

    class Config:
        """Pydantic config."""

        arbitrary_types_allowed = True

    def get_tools(self) -> List[BaseTool]:
        """Get the tools in the toolkit."""
        return [
           MultionCreateSession(),
           MultionUpdateSession()
        ]


