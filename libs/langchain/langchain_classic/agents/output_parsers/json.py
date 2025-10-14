from __future__ import annotations

import logging

from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.exceptions import OutputParserException
from langchain_core.utils.json import parse_json_markdown
from typing_extensions import override

from langchain_classic.agents.agent import AgentOutputParser

logger = logging.getLogger(__name__)


class JSONAgentOutputParser(AgentOutputParser):
    """Parses tool invocations and final answers in JSON format.

    Expects output to be in one of two formats.

    If the output signals that an action should be taken,
    should be in the below format. This will result in an AgentAction
    being returned.

    ```
    {"action": "search", "action_input": "2+2"}
    ```

    If the output signals that a final answer should be given,
    should be in the below format. This will result in an AgentFinish
    being returned.

    ```
    {"action": "Final Answer", "action_input": "4"}
    ```
    """

    @override
    def parse(self, text: str) -> AgentAction | AgentFinish:
        try:
            response = parse_json_markdown(text)
            if isinstance(response, list):  # type: ignore[unreachable]
                # gpt turbo frequently ignores the directive to emit a single action
                logger.warning("Got multiple action responses: %s", response)  # type: ignore[unreachable]
                response = response[0]
            if response["action"] == "Final Answer":
                return AgentFinish({"output": response["action_input"]}, text)
            action_input = response.get("action_input", {})
            if action_input is None:
                action_input = {}
            return AgentAction(response["action"], action_input, text)
        except Exception as e:
            msg = f"Could not parse LLM output: {text}"
            raise OutputParserException(msg) from e

    @property
    def _type(self) -> str:
        return "json-agent"
