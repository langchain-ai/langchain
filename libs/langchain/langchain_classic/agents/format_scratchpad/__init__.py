"""Logic for formatting intermediate steps into an agent scratchpad.

Intermediate steps refers to the list of (AgentAction, observation) tuples
that result from previous iterations of the agent.
Depending on the prompting strategy you are using, you may want to format these
differently before passing them into the LLM.
"""

from langchain_classic.agents.format_scratchpad.log import format_log_to_str
from langchain_classic.agents.format_scratchpad.log_to_messages import (
    format_log_to_messages,
)
from langchain_classic.agents.format_scratchpad.openai_functions import (
    format_to_openai_function_messages,
    format_to_openai_functions,
)
from langchain_classic.agents.format_scratchpad.tools import format_to_tool_messages
from langchain_classic.agents.format_scratchpad.xml import format_xml

__all__ = [
    "format_log_to_messages",
    "format_log_to_str",
    "format_to_openai_function_messages",
    "format_to_openai_functions",
    "format_to_tool_messages",
    "format_xml",
]
