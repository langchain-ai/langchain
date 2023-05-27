import re
from typing import Union

from langchain.agents.agent import AgentOutputParser
from langchain.agents.chat.prompt import FORMAT_INSTRUCTIONS
from langchain.output_parsers.json import parse_json_markdown
from langchain.schema import AgentAction, AgentFinish, OutputParserException

FINAL_ANSWER_ACTION = "Final Answer:"


class ChatOutputParser(AgentOutputParser):
    def get_format_instructions(self) -> str:
        return FORMAT_INSTRUCTIONS

    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        if FINAL_ANSWER_ACTION in text:
            return AgentFinish(
                {"output": text.split(FINAL_ANSWER_ACTION)[-1].strip()}, text
            )
        try:
            text = text.strip().replace("```json", "```")
            markdown_section = re.search("```(.*?)```", text, re.DOTALL)
            if markdown_section:
                extracted_text = markdown_section.group(
                    1
                ).strip()  # remove leading/trailing whitespaces
            else:
                extracted_text = ""
            response = parse_json_markdown(extracted_text)
            return AgentAction(response["action"], response["action_input"], text)

        except Exception:
            raise OutputParserException(f"Could not parse LLM output: {text}")

    @property
    def _type(self) -> str:
        return "chat"
