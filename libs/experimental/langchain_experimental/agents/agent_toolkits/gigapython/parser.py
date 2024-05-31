import re
from typing import Union

from langchain.agents import AgentOutputParser
from langchain_core.agents import AgentAction, AgentFinish


class CodeOutputParser(AgentOutputParser):
    def get_format_instructions(self) -> str:
        return ""

    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        regex = r"```python(.+?)```"
        matches = re.findall(regex, text, re.DOTALL)
        if matches:
            code = "\n".join(matches)
            return AgentAction("python", code, text + "\n")

        else:
            return AgentFinish({"output": text}, text + "\n")

    @property
    def _type(self) -> str:
        return "code-input"
