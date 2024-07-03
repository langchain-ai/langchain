"""Parsing utils to go from string to AgentAction or Agent Finish.

AgentAction means that an action should be taken.
This contains the name of the tool to use, the input to pass to that tool,
and a `log` variable (which contains a log of the agent's thinking).

AgentFinish means that a response should be given.
This contains a `return_values` dictionary. This usually contains a
single `output` key, but can be extended to contain more.
This also contains a `log` variable (which contains a log of the agent's thinking).
"""

from langchain_glm.agents.output_parsers.zhipuai_all_tools import (
    ZhipuAiALLToolsAgentOutputParser,
)

__all__ = ["ZhipuAiALLToolsAgentOutputParser"]
