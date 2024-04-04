from __future__ import annotations

from datetime import datetime
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from langchain_core.agents import AgentAction, AgentActionMessageLog
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_core.prompts import (
    BasePromptTemplate,
    ChatPromptTemplate,
    PromptTemplate,
)
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.tools import BaseTool

from langchain_cohere.react_multi_hop.default_prompt_constants import (
    _SpecialToken,
    default_basic_rules,
    default_multi_hop_instruction,
    default_safety_rules,
    default_style_guide,
    default_task_context,
)
from langchain_cohere.utils import (
    JSON_TO_PYTHON_TYPES,
    _remove_signature_from_tool_description,
)

multi_hop_prompt_partial = PromptTemplate.from_template(
    """{structured_preamble}

## Available Tools
Here is a list of tools that you have available to you:

{tools}{end_turn}{history}{user_prompt}{start_turn}{system_role}{multi_hop_instruction}{end_turn}{steps}"""
).partial(
    start_turn=_SpecialToken.start_turn.value,
    end_turn=_SpecialToken.end_turn.value,
    system_role=_SpecialToken.role_system.value,
    multi_hop_instruction=default_multi_hop_instruction,
)


def render_structured_preamble(
    preamble: Optional[str] = None,
) -> str:
    """Renders the structured preamble part of the prompt content."""
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

    structured_preamble_template = """{prompt_start}# Safety Preamble
{safety_rules}

# System Preamble
## Basic Rules
{basic_rules}

# User Preamble
{preamble}"""
    return structured_preamble_template.format(
        prompt_start=f"{_SpecialToken.bos.value}{_SpecialToken.start_turn.value}{_SpecialToken.role_system.value}",
        safety_rules=default_safety_rules,
        basic_rules=default_basic_rules,
        preamble=preamble,
    )


def render_tool(tool: BaseTool) -> str:
    """Renders a tool into prompt content"""

    template = """```python
{tool_signature}
    \"\"\"{tool_description}{tool_args}
    \"\"\"
    pass
```"""
    return template.format(
        tool_signature=render_tool_signature(tool),
        tool_description=_remove_signature_from_tool_description(
            tool.name, tool.description
        ),
        tool_args=render_tool_args(tool),
    )


def render_observations(
    observations: Union[List[Mapping[str, str]], List[str], Mapping[str, str], str],
    index: int,
) -> Tuple[BaseMessage, int]:
    """Renders the 'output' part of an Agent's intermediate step into prompt content."""
    documents = convert_to_documents(observations)

    rendered_documents: List[str] = []
    document_prompt = """Document: {index}
{fields}"""
    for doc in documents:
        # Render document fields into Key: value strings.
        fields: List[str] = []
        for k, v in doc.items():
            if k.lower() == "url":
                # 'url' is a special key which is always upper case.
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


def convert_to_documents(
    observations: Any,
) -> List[Mapping]:
    """Converts observations into a 'document' dict"""
    documents: List[Mapping] = []
    if isinstance(observations, str):
        # strings are turned into a key/value pair and a key of 'output' is added.
        observations = [{"output": observations}]
    elif isinstance(observations, Mapping):
        # single mappings are transformed into a list to simplify the rest of the code.
        observations = [observations]
    elif not isinstance(observations, Sequence):
        # all other types are turned into a key/value pair within a list
        observations = [{"output": observations}]

    for doc in observations:
        if not isinstance(doc, Mapping):
            # types that aren't Mapping are turned into a key/value pair.
            doc = {"output": doc}
        documents.append(doc)

    return documents


def render_intermediate_steps(
    intermediate_steps: List[Tuple[AgentAction, Any]],
) -> str:
    """Renders an agent's intermediate steps into prompt content."""
    prompt_content = ""
    if any(
        not isinstance(action, AgentActionMessageLog)
        for action, _ in intermediate_steps
    ):
        raise ValueError("all AgentAction steps must implement AgentActionMessageLog")

    i = 0
    for action, observation in intermediate_steps:
        prompt_content += render_messages(action.messages)
        observation_message, i = render_observations(observation, i)
        prompt_content += render_messages([observation_message])
    # Always add an 'open' chatbot turn because prompts for the current turn always end
    # with an open turn.
    prompt_content += (
        f"{_SpecialToken.start_turn.value}{_SpecialToken.role_chatbot.value}"
    )

    return prompt_content


def multi_hop_prompt(
    tools: Sequence[BaseTool], prompt: ChatPromptTemplate
) -> Callable[[Dict], BasePromptTemplate]:
    """The returned function produces a BasePromptTemplate suitable for multi-hop."""

    # the directly_answer tool is used internally by the model, but never produces an
    # AgentAction, so we only need to add it to the prompt.
    tools = list(tools)
    tools.insert(0, create_directly_answer_tool())

    def inner(x: Dict) -> BasePromptTemplate:
        return multi_hop_prompt_partial.partial(
            structured_preamble=render_structured_preamble(
                preamble=x.get("preamble", None)
            ),
            tools="\n\n".join([render_tool(t) for t in tools]),
            user_prompt=render_messages(prompt.invoke(x).to_messages()),
            steps=render_intermediate_steps(x["intermediate_steps"]),
            history=render_messages(x.get("chat_history", [])),
        )

    return inner


def render_type(type_: str, is_optional: bool) -> str:
    """
    Renders a tool's type into prompt content. Types should be Python types, but JSON
    schema is allowed and converted.
    """
    python_type = JSON_TO_PYTHON_TYPES.get(type_, type_)
    if is_optional:
        return f"Optional[{python_type}]"
    else:
        return python_type


def render_tool_signature(tool: BaseTool) -> str:
    """Renders the signature of a tool into prompt content."""
    args = []
    for parameter_name, parameter_definition in tool.args.items():
        type_ = render_type(
            parameter_definition.get("type"), "default" in parameter_definition
        )
        args.append(f"{parameter_name}: {type_}")
    signature = ", ".join(args)
    return f"def {tool.name}({signature}) -> List[Dict]:"


def render_tool_args(tool: BaseTool) -> str:
    """Renders the 'Args' section of a tool's prompt content."""
    if not tool.args:
        return ""
    indent = " "

    prompt_content = f"\n\n{indent * 4}Args:\n{indent * 8}"

    rendered_args = []
    for parameter_name, parameter_definition in tool.args.items():
        type_ = render_type(
            parameter_definition.get("type"), "default" in parameter_definition
        )
        description = parameter_definition.get("description", "")
        rendered_args.append(f"{parameter_name} ({type_}): {description}")

    prompt_content += f"\n{indent * 8}".join(rendered_args)
    return prompt_content


def create_directly_answer_tool() -> BaseTool:
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
        args_schema = InputSchema

        @property
        def args(self) -> dict:
            return {}

        def _run(self, *args: Any, **kwargs: Any) -> Any:
            raise NotImplementedError()

    return DirectlyAnswerTool()


def render_role(message: BaseMessage) -> str:
    """Renders the role of a message into prompt content."""
    if isinstance(message, AIMessage):
        return _SpecialToken.role_chatbot.value
    elif isinstance(message, SystemMessage):
        return _SpecialToken.role_system.value
    else:
        return _SpecialToken.role_user.value


def render_messages(messages: Sequence[BaseMessage]) -> str:
    """Renders one or more BaseMessage implementations into prompt content."""
    return "".join(
        [
            f"{_SpecialToken.start_turn.value}{render_role(message)}{message.content}{_SpecialToken.end_turn.value}"
            for message in messages
        ]
    )
