from collections.abc import Callable
from typing import Any, Dict, List, Sequence, Tuple, Union

from langchain_core.agents import AgentAction, AgentActionMessageLog, AgentFinish
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages.ai import AIMessage
from langchain_core.messages.base import BaseMessage
from langchain_core.messages.system import SystemMessage
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts.base import BasePromptTemplate
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_core.tools import BaseTool

from langchain_cohere.cohere_agent import (
    JSON_TO_PYTHON_TYPES,
    _remove_signature_from_description,
)
from langchain_cohere.react_multi_hop.parsing import (
    parse_actions,
    parse_answer_with_prefixes,
)
from langchain_cohere.react_multi_hop.prompt import (
    multi_hop_prompt_partial,
    render_structured_preamble,
)

"""
    Cohere multi-hop tool allows you to use multiple tools in parallel or 
    sequentially to complete your task. The agent uses a multi hop prompt by Cohere. 
    This prompt is experimental and subject to change.
"""


def create_cohere_react_agent(
    llm: BaseLanguageModel,
    tools: Sequence[BaseTool],
    prompt: ChatPromptTemplate,
) -> Runnable:
    agent = (
        RunnablePassthrough.assign(
            # Handled in the multi_hop_prompt
            agent_scratchpad=lambda _: [],
        )
        | multi_hop_prompt(tools=tools, prompt=prompt)
        | llm.bind(stop=["\nObservation:"], raw_prompting=True)
        | CohereToolsReactAgentOutputParser()
    )
    return agent


class CohereToolsReactAgentOutputParser(
    BaseOutputParser[Union[List[AgentAction], AgentFinish]]
):
    """Parses a message into agent actions/finish."""

    def parse(self, text: str) -> Union[List[AgentAction], AgentFinish]:
        # Parse the structured output of the final answer.
        if "Answer: " in text:
            prefix_map = {
                "answer": "Answer:",
                "grounded_answer": "Grounded answer:",
                "relevant_docs": "Relevant Documents:",
                "cited_docs": "Cited Documents:",
            }
            parsed_answer = parse_answer_with_prefixes(text, prefix_map)
            return AgentFinish({"output": parsed_answer["answer"]}, text)
        elif any([x in text for x in ["Plan: ", "Reflection: ", "Action: "]]):
            completion, plan, actions = parse_actions(text)
            agent_actions: List[AgentAction] = []
            for i, action in enumerate(actions):
                agent_action = AgentActionMessageLog(
                    tool=action["tool_name"],
                    tool_input=action["parameters"],
                    log=f"\n{action}\n" if i > 0 else f"\n{plan}\n{action}\n",
                    message_log=[AIMessage(content=completion)],
                )
                agent_actions.append(agent_action)

            return agent_actions
        else:
            raise ValueError(
                "\nCould not parse generation as it did not contain Plan, Reflection,"
                + f"Action, or Answer. Input: {text}\n\n"
            )


def multi_hop_prompt(
    tools: Sequence[BaseTool], prompt: ChatPromptTemplate
) -> Callable[[Dict], BasePromptTemplate]:
    """Returns a function which produces a BasePromptTemplate suitable for multi-hop."""

    def inner(x: Dict) -> BasePromptTemplate:
        return multi_hop_prompt_partial.partial(
            structured_preamble=render_structured_preamble(
                preamble=x.get("preamble", None)
            ),
            tools="\n\n".join([render_tool_description(t) for t in tools]),
            user_prompt=render_messages(prompt.invoke(x).to_messages()),
            steps=render_intermediate_steps(x["intermediate_steps"]),
            history=render_messages(x.get("chat_history", [])),
        )

    return inner


def get_type(type_: str, is_optional: bool) -> str:
    python_type = JSON_TO_PYTHON_TYPES.get(type_, type_)
    if is_optional:
        return f"Optional[{python_type}]"
    else:
        return python_type


def get_tool_signature(tool: BaseTool) -> str:
    """Get the tool signature."""
    args = []
    for parameter_name, parameter_definition in tool.args.items():
        type_ = get_type(
            parameter_definition.get("type"), "default" in parameter_definition
        )
        args.append(f"{parameter_name}: {type_}")
    signature = ", ".join(args)
    return f"def {tool.name}({signature}) -> List[Dict]:"


def get_tool_args(tool: BaseTool) -> str:
    """Get the tool args."""
    if not tool.args:
        return ""
    indent = " "

    prompt_content = f"\n\n{indent * 4}Args:\n{indent * 8}"

    rendered_args = []
    for parameter_name, parameter_definition in tool.args.items():
        type_ = get_type(
            parameter_definition.get("type"), "default" in parameter_definition
        )
        description = parameter_definition.get("description", "")
        rendered_args.append(f"{parameter_name} ({type_}): {description}")

    prompt_content += f"\n{indent * 8}".join(rendered_args)
    return prompt_content


def render_messages(messages: Sequence[BaseMessage]) -> str:
    """Render chat history."""
    return "".join(
        [
            f"<|START_OF_TURN_TOKEN|>{get_role(message)}{message.content}<|END_OF_TURN_TOKEN|>"
            for message in messages
        ]
    )


def get_role(message: BaseMessage) -> str:
    if isinstance(message, AIMessage):
        return "<|CHATBOT_TOKEN|>"
    elif isinstance(message, SystemMessage):
        return "<|SYSTEM_TOKEN|>"
    else:
        return "<|USER_TOKEN|>"


def render_tool_description(tool: BaseTool) -> str:
    """Render the tool name and description.

    For example:
    ```python
        def calculator_calc(expression: str) -> List[Dict]:
            \"\"\"This is a powerful multi-purpose calculator.
            It is capable of a wide array of math calculation and a range of features.

            Args:
                expression (str): The expression for the calculator to evaluate.

            \"\"\"
            pass
        ```
    """

    template = """```python
{tool_signature}
    \"\"\"{tool_description}{tool_args}
    \"\"\"
    pass
```"""
    return template.format(
        tool_signature=get_tool_signature(tool),
        tool_description=_remove_signature_from_description(
            tool.name, tool.description
        ),
        tool_args=get_tool_args(tool),
    )


def render_observation(
    observation: List[Dict[str, str]], index: int
) -> Tuple[BaseMessage, int]:
    if not isinstance(observation, list) or any(
        not isinstance(item, dict) for item in observation
    ):
        raise ValueError("observation is not a list of dictionaries")

    rendered_documents = []
    document_prompt = """Document: {index}
{fields}"""
    for doc in observation:
        rendered_documents.append(
            document_prompt.format(
                index=index,
                fields="\n".join(
                    f"{"URL" if k.lower() == "url" else k.title()}: {v}"
                    for k, v in doc.items()
                ),
            )
        )
        index += 1
    prompt_content = f"<results>\n{"\n\n".join(rendered_documents)}\n</results>"
    return SystemMessage(content=prompt_content), index


def render_intermediate_steps(
    intermediate_steps: List[Tuple[AgentAction, Any]],
) -> str:
    """Construct the scratchpad that lets the agent continue its thought process."""
    prompt_content = ""
    if any(
        not isinstance(action, AgentActionMessageLog)
        for action, _ in intermediate_steps
    ):
        raise ValueError("action is not AgentActionMessageLog")

    i = 0
    for action, observation in intermediate_steps:
        prompt_content += render_messages(action.messages)
        observation_message, i = render_observation(observation, i)
        prompt_content += render_messages([observation_message])
    # Always add an 'open' chatbot turn.
    prompt_content += "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>"

    return prompt_content
