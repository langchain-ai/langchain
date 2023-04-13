# flake8: noqa
import json
from langchain.prompts.chat import (
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.agents.schema import AgentScratchPadChatPromptTemplate
from langchain.prompts.base import BasePromptTemplate
from langchain.schema import AgentAction, AgentFinish
from langchain.tools.base import BaseTool
from typing import Sequence, Optional, List, Union
from langchain.agents.agent import AgentOutputParser

PREFIX = """Answer the following questions as best you can. You have access to the following tools:"""
FORMAT_INSTRUCTIONS = """The way you use the tools is by specifying a json blob.
Specifically, this json should have a `action` key (with the name of the tool to use) and a `action_input` key (with the input to the tool going here).

The only values that should be in the "action" field are: {tool_names}

The $JSON_BLOB should only contain a SINGLE action, do NOT return a list of multiple actions. Here is an example of a valid $JSON_BLOB:

```
{{{{
  "action": $TOOL_NAME,
  "action_input": $INPUT
}}}}
```

ALWAYS use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action:
```
$JSON_BLOB
```
Observation: the result of the action
... (this Thought/Action/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question"""
SUFFIX = """Begin! Reminder to always use the exact characters `Final Answer` when responding."""


def create_prompt(
    tools: Sequence[BaseTool],
    prefix: str = PREFIX,
    suffix: str = SUFFIX,
    format_instructions: str = FORMAT_INSTRUCTIONS,
    input_variables: Optional[List[str]] = None,
) -> BasePromptTemplate:
    tool_strings = "\n".join([f"{tool.name}: {tool.description}" for tool in tools])
    tool_names = ", ".join([tool.name for tool in tools])
    format_instructions = format_instructions.format(tool_names=tool_names)
    template = "\n\n".join([prefix, tool_strings, format_instructions, suffix])
    messages = [
        SystemMessagePromptTemplate.from_template(template),
        HumanMessagePromptTemplate.from_template("{input}\n\n{agent_scratchpad}"),
    ]
    if input_variables is None:
        input_variables = ["input", "intermediate_steps"]
    return AgentScratchPadChatPromptTemplate(
        input_variables=input_variables, messages=messages
    )


class ChatOutputParser(AgentOutputParser):
    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        if "Final Answer:" in text:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": text.split("Final Answer:")[-1].strip()},
                log=text,
            )
        try:
            _, action, _ = text.split("```")
            response = json.loads(action.strip())
            agent_action = AgentAction(
                tool=response["action"], tool_input=response["action_input"], log=text
            )
            return agent_action

        except Exception:
            raise ValueError(f"Could not parse LLM output: {text}")
