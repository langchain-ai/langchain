import json
import re
from re import Pattern

from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.exceptions import OutputParserException
from typing_extensions import override

from langchain_classic.agents.agent import AgentOutputParser
from langchain_classic.agents.chat.prompt import FORMAT_INSTRUCTIONS

FINAL_ANSWER_ACTION = "Final Answer:"


class ReActJsonSingleInputOutputParser(AgentOutputParser):
    """Parses ReAct-style LLM calls that have a single tool input in json format.

    Expects output to be in one of two formats.

    If the output signals that an action should be taken,
    should be in the below format. This will result in an AgentAction
    being returned.

    ```
    Thought: agent thought here
    Action:
    ```
    {
        "action": "search",
        "action_input": "what is the temperature in SF"
    }
    ```
    ```

    If the output signals that a final answer should be given,
    should be in the below format. This will result in an AgentFinish
    being returned.

    ```
    Thought: agent thought here
    Final Answer: The temperature is 100 degrees
    ```

    """

    pattern: Pattern = re.compile(r"^.*?`{3}(?:json)?\n?(.*?)`{3}.*?$", re.DOTALL)
    """Regex pattern to parse the output."""

    @override
    def get_format_instructions(self) -> str:
        return FORMAT_INSTRUCTIONS

    @override
    def parse(self, text: str) -> AgentAction | AgentFinish:
        includes_answer = FINAL_ANSWER_ACTION in text
        try:
            found = self.pattern.search(text)
            if not found:
                # Fast fail to parse Final Answer.
                msg = "action not found"
                raise ValueError(msg)
            action = found.group(1)
            response = json.loads(action.strip())
            includes_action = "action" in response
            if includes_answer and includes_action:
                msg = (
                    "Parsing LLM output produced a final answer "
                    f"and a parse-able action: {text}"
                )
                raise OutputParserException(msg)
            return AgentAction(
                response["action"],
                response.get("action_input", {}),
                text,
            )

        except Exception as e:
            if not includes_answer:
                msg = f"Could not parse LLM output: {text}"
                raise OutputParserException(msg) from e
            output = text.split(FINAL_ANSWER_ACTION)[-1].strip()
            return AgentFinish({"output": output}, text)

    @property
    def _type(self) -> str:
        return "react-json-single-input"
