from typing import Any

from langchain.chains.openai_functions import (
    create_openai_fn_chain,
)
from langchain.chat_models.ernie_bot import ErnieBotChat
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.schema import ChatGeneration, LLMResult
from langchain.schema.messages import (
    AIMessage,
    BaseMessage,
    FunctionMessage,
    HumanMessage,
)

_FUNCTIONS: Any = [
    {
        "name": "format_person_info",
        "description": (
            "Output formatter. Should always be used to format your response to the"
            " user."
        ),
        "parameters": {
            "title": "Person",
            "description": "Identifying information about a person.",
            "type": "object",
            "properties": {
                "name": {
                    "title": "Name",
                    "description": "The person's name",
                    "type": "string",
                },
                "age": {
                    "title": "Age",
                    "description": "The person's age",
                    "type": "integer",
                },
                "fav_food": {
                    "title": "Fav Food",
                    "description": "The person's favorite food",
                    "type": "string",
                },
            },
            "required": ["name", "age"],
        },
    },
    {
        "name": "get_current_temperature",
        "description": ("Used to get the location's temperature."),
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "city name",
                },
                "unit": {
                    "type": "string",
                    "enum": ["centigrade", "Fahrenheit"],
                },
            },
            "required": ["location", "unit"],
        },
        "responses": {
            "type": "object",
            "properties": {
                "temperature": {
                    "type": "integer",
                    "description": "city temperature",
                },
                "unit": {
                    "type": "string",
                    "enum": ["centigrade", "Fahrenheit"],
                },
            },
        },
    },
]


def test_chat_ernie_bot() -> None:
    chat = ErnieBotChat()
    message = HumanMessage(content="Hello")
    response = chat([message])
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)


def test_multiple_history() -> None:
    """Tests multiple history works."""
    chat = ErnieBotChat()

    response = chat(
        messages=[
            HumanMessage(content="Hello."),
            AIMessage(content="Hello!"),
            HumanMessage(content="How are you doing?"),
        ]
    )
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)


def test_functions_call_thoughts() -> None:
    chat = ErnieBotChat()

    prompt_tmpl = "Use the given functions to answer following question: {input}"
    prompt_msgs = [
        HumanMessagePromptTemplate.from_template(prompt_tmpl),
    ]
    prompt = ChatPromptTemplate(messages=prompt_msgs)

    chain = create_openai_fn_chain(
        _FUNCTIONS,
        chat,
        prompt,
        output_parser=None,
    )

    message = HumanMessage(content="What's the temperature in Shanghai today?")
    response = chain.generate([{"input": message}])
    assert isinstance(response.generations[0][0], ChatGeneration)
    assert isinstance(response.generations[0][0].message, AIMessage)
    assert "function_call" in response.generations[0][0].message.additional_kwargs


def test_functions_call() -> None:
    chat = ErnieBotChat()

    prompt = ChatPromptTemplate(
        messages=[
            HumanMessage(content="What's the temperature in Shanghai today?"),
            AIMessage(
                content="",
                additional_kwargs={
                    "function_call": {
                        "name": "get_current_temperature",
                        "thoughts": "i will use get_current_temperature "
                        "to resolve the questions",
                        "arguments": '{"location":"Shanghai","unit":"centigrade"}',
                    }
                },
            ),
            FunctionMessage(
                name="get_current_weather",
                content='{"temperature": "25", \
                                "unit": "摄氏度", "description": "晴朗"}',
            ),
        ]
    )
    llm_chain = create_openai_fn_chain(
        _FUNCTIONS,
        chat,
        prompt,
        output_parser=None,
    )
    resp = llm_chain.generate([{}])
    assert isinstance(resp, LLMResult)
