"""Parsing utils to go from string to AgentAction or Agent Finish.

AgentAction means that an action should be taken.
This contains the name of the tool to use, the input to pass to that tool,
and a `log` variable (which contains a log of the agent's thinking).

AgentFinish means that a response should be given.
This contains a `return_values` dictionary. This usually contains a
single `output` key, but can be extended to contain more.
This also contains a `log` variable (which contains a log of the agent's thinking).
"""

from langchain_classic.agents.output_parsers.json import JSONAgentOutputParser
from langchain_classic.agents.output_parsers.openai_functions import (
    OpenAIFunctionsAgentOutputParser,
)
from langchain_classic.agents.output_parsers.react_json_single_input import (
    ReActJsonSingleInputOutputParser,
)
from langchain_classic.agents.output_parsers.react_single_input import (
    ReActSingleInputOutputParser,
)
from langchain_classic.agents.output_parsers.self_ask import SelfAskOutputParser
from langchain_classic.agents.output_parsers.tools import ToolsAgentOutputParser
from langchain_classic.agents.output_parsers.xml import XMLAgentOutputParser

__all__ = [
    "JSONAgentOutputParser",
    "OpenAIFunctionsAgentOutputParser",
    "ReActJsonSingleInputOutputParser",
    "ReActSingleInputOutputParser",
    "SelfAskOutputParser",
    "ToolsAgentOutputParser",
    "XMLAgentOutputParser",
]
