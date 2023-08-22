import re
from typing import Sequence, Union

from langchain.agents.agent import AgentOutputParser
from langchain.schema import AgentAction, AgentFinish, OutputParserException


class SelfAskOutputParser(AgentOutputParser):
    """Output parser for the self-ask agent."""

    followups: Sequence[str] = ("Follow up:", "Followup:")
    finish_string: str = "So the final answer is:"

    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        followup_pattern = "|".join(map(lambda followup: f"{followup}", self.followups))
        followup = re.findall(rf"({followup_pattern})(?ms:(.*))", text)
        end = re.findall(rf"({self.finish_string})(?ms:(.*))$", text)

        # if follow up
        if len(followup) > 0 and followup[-1][0] in self.followups:
            return AgentAction("Intermediate Answer", followup[-1][1].strip(), text)
        # if final output
        if len(end) > 0 and end[-1][0] == self.finish_string:
            return AgentFinish({"output": end[-1][1].strip()}, text)

        raise OutputParserException(f"Could not parse output: {text}")

    @property
    def _type(self) -> str:
        return "self_ask"
