from typing import Dict, List, Tuple, Union

from langchain.agents import AgentExecutor
from langchain.schema import AgentAction, AgentFinish, OutputParserException
from langchain.tools import BaseTool


class RetryAgentExecutor(AgentExecutor):
    """Agent executor that retries on output parser exceptions."""

    def _take_next_step(
        self,
        name_to_tool_map: Dict[str, BaseTool],
        color_mapping: Dict[str, str],
        inputs: Dict[str, str],
        intermediate_steps: List[Tuple[AgentAction, str]],
    ) -> Union[AgentFinish, List[Tuple[AgentAction, str]]]:
        # call super method
        try:
            return super()._take_next_step(
                name_to_tool_map, color_mapping, inputs, intermediate_steps
            )
        except OutputParserException as e:
            agent_action = AgentAction("_Exception", "", str(e))
            self.callback_manager.on_agent_action(
                agent_action, verbose=self.verbose, color="red"
            )
            return [(agent_action, "Invalid or incomplete response. Please try again.")]
