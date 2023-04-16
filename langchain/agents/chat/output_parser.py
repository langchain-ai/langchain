import json
from typing import Union

from langchain.agents.agent import AgentOutputParser
from langchain.schema import AgentAction, AgentFinish

FINAL_ANSWER_ACTION = "Final Answer:"


class ChatOutputParser(AgentOutputParser):
    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        if FINAL_ANSWER_ACTION in text:
            return AgentFinish(
                {"output": text.split(FINAL_ANSWER_ACTION)[-1].strip()}, text
            )
        try:
            _, action, _ = text.split("```")
            response = json.loads(action.strip())
            return AgentAction(response["action"], response["action_input"], text)

        except Exception:
            raise ValueError(f"Could not parse LLM output: {text}")
