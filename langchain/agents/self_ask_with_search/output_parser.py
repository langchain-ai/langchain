from typing import Union

from langchain.agents.agent import AgentOutputParser
from langchain.schema import AgentAction, AgentFinish, OutputParserException


class SelfAskOutputParser(AgentOutputParser):
    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        followup = "Follow up:"
        last_line = text.split("\n")[-1]

        if followup not in last_line:
            finish_string = "So the final answer is: "
            if finish_string not in last_line:
                raise OutputParserException(f"Could not parse output: {text}")
            return AgentFinish({"output": last_line[len(finish_string) :]}, text)

        after_colon = text.split(":")[-1]

        if " " == after_colon[0]:
            after_colon = after_colon[1:]
        return AgentAction("Intermediate Answer", after_colon, text)
