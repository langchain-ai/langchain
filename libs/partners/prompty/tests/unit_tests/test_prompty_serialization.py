import json
import os

from langchain_classic.agents.format_scratchpad import (
    format_to_openai_function_messages,
)
from langchain_classic.tools import tool
from langchain_core.language_models import FakeListLLM
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.utils.function_calling import convert_to_openai_function
from pydantic import BaseModel, Field

import langchain_prompty

from .fake_callback_handler import FakeCallbackHandler
from .fake_chat_model import FakeEchoPromptChatModel
from .fake_output_parser import FakeOutputParser

prompty_folder_relative = "./prompts/"
# Get the directory of the current script
current_script_dir = os.path.dirname(__file__)

# Combine the current script directory with the relative path
prompty_folder = os.path.abspath(
    os.path.join(current_script_dir, prompty_folder_relative)
)


def test_prompty_basic_chain() -> None:
    prompt = langchain_prompty.create_chat_prompt(f"{prompty_folder}/chat.prompty")
    model = FakeEchoPromptChatModel()
    chain = prompt | model

    parsed_prompts = chain.invoke(
        {
            "firstName": "fakeFirstName",
            "lastName": "fakeLastName",
            "input": "fakeQuestion",
        }
    )

    if isinstance(parsed_prompts.content, str):
        msgs = json.loads(str(parsed_prompts.content))
    else:
        msgs = parsed_prompts.content

    assert len(msgs) == 2
    # Test for system and user entries
    system_message = msgs[0]
    user_message = msgs[1]

    # Check the types of the messages
    assert system_message["type"] == "system", (
        "The first message should be of type 'system'."
    )
    assert user_message["type"] == "human", (
        "The second message should be of type 'human'."
    )

    # Test for existence of fakeFirstName and fakeLastName in the system message
    assert "fakeFirstName" in system_message["content"], (
        "The string 'fakeFirstName' should be in the system message content."
    )
    assert "fakeLastName" in system_message["content"], (
        "The string 'fakeLastName' should be in the system message content."
    )

    # Test for existence of fakeQuestion in the user message
    assert "fakeQuestion" in user_message["content"], (
        "The string 'fakeQuestion' should be in the user message content."
    )


def test_prompty_used_in_agent() -> None:
    prompt = langchain_prompty.create_chat_prompt(f"{prompty_folder}/chat.prompty")
    tool_name = "search"
    responses = [
        f"FooBarBaz\nAction: {tool_name}\nAction Input: fakeSearch",
        "Oh well\nFinal Answer: fakefinalresponse",
    ]
    callbackHandler = FakeCallbackHandler()
    llm = FakeListLLM(responses=responses, callbacks=[callbackHandler])

    @tool
    def search(query: str) -> str:
        """Look up things."""
        return "FakeSearchResponse"

    tools = [search]
    llm_with_tools = llm.bind(functions=[convert_to_openai_function(t) for t in tools])

    agent = (
        {  # type: ignore[var-annotated]
            "firstName": lambda x: x["firstName"],
            "lastName": lambda x: x["lastName"],
            "input": lambda x: x["input"],
            "chat_history": lambda x: x["chat_history"],
            "agent_scratchpad": lambda x: (
                format_to_openai_function_messages(x["intermediate_steps"])
                if "intermediate_steps" in x
                else []
            ),
        }
        | prompt
        | llm_with_tools
        | FakeOutputParser()
    )

    from langchain_classic.agents import AgentExecutor

    class AgentInput(BaseModel):
        input: str
        chat_history: list[tuple[str, str]] = Field(
            ...,
            json_schema_extra={
                "widget": {"type": "chat", "input": "input", "output": "output"}
            },
        )

    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True).with_types(
        input_type=AgentInput  # type: ignore[arg-type]
    )

    agent_executor.invoke(
        {
            "firstName": "fakeFirstName",
            "lastName": "fakeLastName",
            "input": "fakeQuestion",
            "chat_history": [
                AIMessage(content="chat_history_1_ai"),
                HumanMessage(content="chat_history_1_human"),
            ],
        }
    )
    input_prompt = callbackHandler.input_prompts[0]

    # Test for existence of fakeFirstName and fakeLastName in the system message
    assert "fakeFirstName" in input_prompt
    assert "fakeLastName" in input_prompt
    assert "chat_history_1_ai" in input_prompt
    assert "chat_history_1_human" in input_prompt
    assert "fakeQuestion" in input_prompt
    assert "fakeSearch" in input_prompt


def test_all_prompty_can_run() -> None:
    exclusions = ["embedding.prompty", "groundedness.prompty"]

    prompty_files = [
        f
        for f in os.listdir(prompty_folder)
        if os.path.isfile(os.path.join(prompty_folder, f))
        and f.endswith(".prompty")
        and f not in exclusions
    ]

    for file in prompty_files:
        file_path = os.path.join(prompty_folder, file)

        prompt = langchain_prompty.create_chat_prompt(file_path)
        model = FakeEchoPromptChatModel()
        chain = prompt | model
        chain.invoke({})
