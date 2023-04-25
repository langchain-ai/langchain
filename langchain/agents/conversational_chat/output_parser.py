from __future__ import annotations

import json
from typing import Any, Union

from langchain.agents import AgentOutputParser
from langchain.agents.conversational_chat.prompt import FORMAT_INSTRUCTIONS
from langchain.schema import AgentAction, AgentFinish


class ConvoOutputParser(AgentOutputParser):
    def get_format_instructions(self) -> str:
        return FORMAT_INSTRUCTIONS

    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        class MessageJSONExtractor(json.JSONDecoder):
            """
            Custom JSON extractor that will extract the first JSON object in a string.
            This is to handle the LLM returning more than just the JSON object.
            """

            def decode(self, s: str, _w: Any = None) -> Any:
                for idx, char in enumerate(s):
                    if char == "{":
                        end_pos = self.raw_decode(s, idx)
                        break
                else:
                    raise json.JSONDecodeError("No JSON object found", s, 0)

                return end_pos[0]

        try:
            response = json.loads(text, cls=MessageJSONExtractor)
            # only accept this result if it has the required fields
            if all(field in response for field in ["action", "action_input"]):
                if response["action"] == "Final Answer":
                    return AgentFinish({"output": response["action_input"]}, text)
                else:
                    return AgentAction(
                        response["action"], response["action_input"], text
                    )
        except json.JSONDecodeError:
            pass

        # if the LLM did not provide a valid response object,
        # we will return the text as is
        return AgentFinish({"output": text}, text)
