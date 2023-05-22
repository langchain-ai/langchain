from __future__ import annotations

import json
from typing import Union, List
import unicodedata

from langchain.agents import AgentOutputParser
from langchain.agents.conversational_chat.prompt import FORMAT_INSTRUCTIONS
from langchain.schema import AgentAction, AgentFinish, OutputParserException


class ConvoOutputParser(AgentOutputParser):
    # Override base class method to provide specific format instructions
    def get_format_instructions(self) -> str:
        return FORMAT_INSTRUCTIONS

    # The main parsing method that takes in the raw text and outputs structured data
    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        try:
            # Basic cleaning of the input
            cleaned_output = text.strip()

            # Removal of control characters from the input text
            cleaned_output = ''.join(ch for ch in cleaned_output if unicodedata.category(ch)[0]!="C")

            '''
            The input text may contain one or more responses from the ChatGPT API.
            Each response is enclosed within a ```json and ``` pair.
            The following lines of code split the text into sections, each section containing one response.
            '''
            sections = cleaned_output.split("```json")

            '''
            Next, we remove any trailing "```" from each section.
            This gives us a list of cleaned_output strings, each of which should be a valid JSON string.
            '''
            cleaned_sections: List[str] = [section.rstrip("```") for section in sections if section.rstrip("```")]

            '''
            Now we attempt to parse each cleaned_output string as JSON.
            If successful, we append the parsed JSON object to the responses list.
            If not, we simply ignore that string and move to the next one.
            '''
            responses = []
            for output in cleaned_sections:
                try:
                    response = json.loads(output)
                    responses.append(response)
                except json.JSONDecodeError:
                    pass  # Continue to the next section

            '''
            Once we have the list of responses, we process each response based on its action value.
            If the action is "Final Answer", we return an AgentFinish object.
            Otherwise, we return an AgentAction object.
            '''
            agent_action = None
            for response in responses:
                action, action_input = response["action"], response["action_input"]
                if action == "Final Answer":
                    return AgentFinish({"output": action_input}, text)
                else:
                    agent_action = AgentAction(action, action_input, text)

            if agent_action is not None:
                return agent_action
            else:
                raise OutputParserException("No suitable action found in LLM output")

            # If no suitable action is found, raise an exception
            raise OutputParserException("No suitable action found in LLM output")

        except Exception as e:
            '''
            If any exception is raised during the above process,
            we catch it here and raise an OutputParserException instead.
            This is to ensure that any issues are clearly flagged as related to the parsing process.
            '''
            raise OutputParserException(f"Could not parse LLM output: {text}") from e

    # This property method is required by the base class and simply returns the type of this parser
    @property
    def _type(self) -> str:
        return "conversational_chat"

