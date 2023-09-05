"""ReAct specific prompt processing"""
import re
from typing import Any, List

from langchain.prompts.base import PromptValue, StringPromptValue
from langchain.prompts.prompt import PromptTemplate
from langchain.schema import AIMessage, BaseMessage, HumanMessage


class ReActStringPromptValue(StringPromptValue):
    """PromptValue to be returned from ReActPromptTemplate"""

    def to_messages(self) -> List[BaseMessage]:
        # Matches the first word
        prefix_regex = re.compile(r"^.+?\b")
        thought = ""
        messages: List[BaseMessage] = []
        for line in self.text.split("\n"):
            # Process only non-empty lines
            if line:
                prefix = prefix_regex.match(line)[0]
                match prefix:
                    case "Question":
                        messages.append(HumanMessage(content=line))
                    case "Thought":
                        thought = line
                    case "Action":
                        messages.append(AIMessage(content=f"{thought}\n{line}"))
                    case "Observation":
                        messages.append(HumanMessage(content=line))

        return messages


class ReActPromptTemplate(PromptTemplate):
    """ReAct specific PromptTemplate for correct prompt formatting"""

    def format_prompt(self, **kwargs: Any) -> PromptValue:
        """Create Chat Messages."""
        return ReActStringPromptValue(text=self.format(**kwargs))
