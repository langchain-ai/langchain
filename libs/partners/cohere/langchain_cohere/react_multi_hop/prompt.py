from datetime import datetime
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Type, Union

from langchain_core.agents import AgentAction, AgentActionMessageLog
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.prompts import (
    BasePromptTemplate,
    ChatPromptTemplate,
    PromptTemplate,
)
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.tools import BaseTool

from langchain_cohere.cohere_agent import (
    JSON_TO_PYTHON_TYPES,
    _remove_signature_from_description,
)
from langchain_cohere.react_multi_hop.default_prompt_constants import (
    default_basic_rules,
    default_multi_hop_instruction,
    default_safety_rules,
    default_style_guide,
    default_system_prefix,
    default_task_context,
)
from langchain_cohere.utils import render_messages


def render_structured_preamble(
    preamble: Optional[str] = None,
) -> str:
    if preamble is None:
        default_preamble = """## Task And Context
{task_and_context}

## Style Guide
{style_guide}"""
        preamble = default_preamble.format(
            task_and_context=default_task_context.format(
                now=datetime.now().strftime("%A, %B %d, %Y %H:%M:%S")
            ),
            style_guide=default_style_guide,
        )

    structured_preamble_template = """{system_prefix}# Safety Preamble
{safety_rules}

# System Preamble
## Basic Rules
{basic_rules}

# User Preamble
{preamble}"""
    return structured_preamble_template.format(
        system_prefix=default_system_prefix,
        safety_rules=default_safety_rules,
        basic_rules=default_basic_rules,
        preamble=preamble,
    )


multi_hop_prompt_partial = PromptTemplate.from_template(
    """{structured_preamble}

## Available Tools
Here is a list of tools that you have available to you:

{tools}<|END_OF_TURN_TOKEN|>{history}{user_prompt}<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>{multi_hop_instruction}<|END_OF_TURN_TOKEN|>{steps}"""
).partial(
    multi_hop_instruction=default_multi_hop_instruction,
)


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


def render_observations(
    observations: Union[List[Mapping[str, str]], List[str], Mapping[str, str], str],
    index: int,
) -> Tuple[BaseMessage, int]:
    """Renders the 'output' part of an Agent's intermediate step into prompt content."""
    if (
        not isinstance(observations, list)
        and not isinstance(observations, str)
        and not isinstance(observations, Mapping)
    ):
        raise ValueError("observation must be a list, a Mapping, or a string")

    rendered_documents = []
    document_prompt = """Document: {index}
{fields}"""

    if isinstance(observations, str):
        # strings are turned into a key/value pair and a key of 'output' is added.
        observations = [{"output": observations}]  # type: ignore

    if isinstance(observations, Mapping):
        observations = [observations]

    if isinstance(observations, list):
        for doc in observations:
            if isinstance(doc, str):
                # strings are turned into a key/value pair and a key of 'output'
                # is added.
                doc = {"output": doc}

            if not isinstance(doc, Mapping):
                raise ValueError(
                    "all observation list items must be a Mapping or a string"
                )

            fields: List[str] = []
            for k, v in doc.items():
                if k.lower() == "url":
                    # 'url' is a special field which is always upper case.
                    k = "URL"
                else:
                    # keys are otherwise transformed into title case.
                    k = k.title()
                fields.append(f"{k}: {v}")

            rendered_documents.append(
                document_prompt.format(
                    index=index,
                    fields="\n".join(fields),
                )
            )
            index += 1

    prompt_content = "<results>\n" + "\n\n".join(rendered_documents) + "\n</results>"
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
        observation_message, i = render_observations(observation, i)
        prompt_content += render_messages([observation_message])
    # Always add an 'open' chatbot turn.
    prompt_content += "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>"

    return prompt_content


def multi_hop_prompt(
    tools: List[BaseTool], prompt: ChatPromptTemplate
) -> Callable[[Dict], BasePromptTemplate]:
    """Returns a function which produces a BasePromptTemplate suitable for multi-hop."""

    # the directly_answer tool is used internally by the model, but never produces an
    # AgentAction, so we just need to add it to the prompt.
    tools.insert(0, _create_directly_answer_tool())

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


def _create_directly_answer_tool() -> BaseTool:
    """
    directly_answer is a special tool that's always presented to the model as an
    available tool. The model only ever invokes this whilst answering and no AgentAction
    is produced, so it only needs to be added to the prompt.
    """

    class DirectlyAnswerTool(BaseTool):
        class InputSchema(BaseModel):
            pass

        name = "directly_answer"
        description = "Calls a standard (un-augmented) AI chatbot to generate a response given the conversation history"  # noqa: E501
        args_schema: Type[InputSchema] = InputSchema

        def _run(self, *args: Any, **kwargs: Any) -> Any:
            raise NotImplementedError()

    return DirectlyAnswerTool()
