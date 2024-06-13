from typing import Optional, Tuple, Union

from langchain.agents import AgentOutputParser
from langchain_core.agents import AgentAction, AgentFinish


def extract_action_details(text: str) -> Tuple[Optional[str], Optional[str]]:
    # Split the text into lines and strip whitespace
    lines = [line.strip() for line in text.strip().split("\n")]

    # Initialize variables to hold the extracted values
    action = None
    action_input = None

    # Iterate through the lines to find and extract the desired information
    for line in lines:
        if line.startswith("Action:"):
            action = line.split(":", 1)[1].strip()
        elif line.startswith("Action Input:"):
            action_input = line.split(":", 1)[1].strip()

    return action, action_input


class FakeOutputParser(AgentOutputParser):
    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        print("FakeOutputParser", text)
        action, input = extract_action_details(text)

        if action:
            log = f"\nInvoking: `{action}` with `{input}"

            return AgentAction(tool=action, tool_input=(input or ""), log=log)
        elif "Final Answer" in text:
            return AgentFinish({"output": text}, text)

        return AgentAction(
            "Intermediate Answer", "after_colon", "Final Answer: This should end"
        )

    @property
    def _type(self) -> str:
        return "self_ask"
