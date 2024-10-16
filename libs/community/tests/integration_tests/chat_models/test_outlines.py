# flake8: noqa
"""Test ChatOutlines wrapper."""

from typing import Generator
import re
import platform
import pytest

from langchain_community.chat_models.outlines import ChatOutlines
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGenerationChunk
from pydantic import BaseModel

from tests.unit_tests.callbacks.fake_callback_handler import FakeCallbackHandler


MODEL = "microsoft/Phi-3-mini-4k-instruct"
LLAMACPP_MODEL = "TheBloke/Llama-2-7B-Chat-GGUF/llama-2-7b-chat.Q4_K_M.gguf"

BACKENDS = ["transformers", "llamacpp"]
if platform.system() != "Darwin":
    BACKENDS.append("vllm")
if platform.system() == "Darwin":
    BACKENDS.append("mlxlm")


@pytest.fixture(params=BACKENDS)
def chat_model(request):
    if request.param == "llamacpp":
        return ChatOutlines(model=LLAMACPP_MODEL, backend=request.param, max_tokens=10)
    else:
        return ChatOutlines(model=MODEL, backend=request.param, max_tokens=10)


def test_chat_outlines_inference(chat_model: ChatOutlines) -> None:
    """Test valid ChatOutlines inference."""
    messages = [HumanMessage(content="Say foo:")]
    output = chat_model.invoke(messages)
    assert isinstance(output, AIMessage)
    assert len(output.content) > 1


def test_chat_outlines_streaming(chat_model: ChatOutlines) -> None:
    """Test streaming tokens from ChatOutlines."""
    messages = [HumanMessage(content="Q: How do you say 'hello' in Spanish? A:'")]
    generator = chat_model.stream(messages, stop=["'"])
    stream_results_string = ""
    assert isinstance(generator, Generator)

    for chunk in generator:
        assert isinstance(chunk, ChatGenerationChunk)
        stream_results_string += chunk.message.content
    assert len(stream_results_string.strip()) > 1


def test_chat_outlines_streaming_callback(chat_model: ChatOutlines) -> None:
    """Test that streaming correctly invokes on_llm_new_token callback."""
    MAX_TOKENS = 5
    OFF_BY_ONE = 1  # There may be an off by one error in the upstream code!

    callback_handler = FakeCallbackHandler()
    chat_model.callbacks = [callback_handler]
    chat_model.verbose = True
    messages = [HumanMessage(content="Q: Can you count to 10? A:'1, ")]
    chat_model.invoke(messages)
    assert callback_handler.llm_streams <= MAX_TOKENS + OFF_BY_ONE


def test_chat_outlines_model_kwargs(chat_model: ChatOutlines) -> None:
    chat_model.model_kwargs = {"n_gqa": None}
    assert chat_model.model_kwargs == {"n_gqa": None}


def test_chat_outlines_regex(chat_model: ChatOutlines) -> None:
    """Test regex for generating a valid IP address"""
    ip_regex = r"((25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(25[0-5]|2[0-4]\d|[01]?\d\d?)"
    chat_model.regex = ip_regex
    assert chat_model.regex == ip_regex

    messages = [
        HumanMessage(content="Q: What is the IP address of Google's DNS server? A: ")
    ]
    output = chat_model.invoke(messages)

    assert isinstance(output, AIMessage)
    assert re.match(
        ip_regex, output.content
    ), f"Generated output '{output.content}' is not a valid IP address"


def test_chat_outlines_type_constraints(chat_model: ChatOutlines) -> None:
    """Test type constraints for generating an integer"""
    chat_model.type_constraints = int
    messages = [
        HumanMessage(
            content="Q: What is the answer to life, the universe, and everything? A: "
        )
    ]
    output = chat_model.invoke(messages)
    assert isinstance(int(output.content), int)


def test_chat_outlines_json(chat_model: ChatOutlines) -> None:
    """Test json for generating a valid JSON object"""

    class Person(BaseModel):
        name: str

    chat_model.json_schema = Person
    messages = [HumanMessage(content="Q: Who is the author of LangChain?  A: ")]
    output = chat_model.invoke(messages)
    person = Person.model_validate_json(output.content)
    assert isinstance(person, Person)


def test_chat_outlines_json_schema(chat_model: ChatOutlines) -> None:
    """Test json schema for generating a valid JSON object"""

    class Food(BaseModel):
        ingredients: list[str]
        calories: int

    chat_model.json_schema = Food.model_json_schema()
    messages = [
        HumanMessage(
            content="Q: What is the nutritional information for a Big Mac? A: "
        )
    ]
    output = chat_model.invoke(messages)
    food = Food.model_validate_json(output.content)
    assert isinstance(food, Food)


def test_chat_outlines_grammar(chat_model: ChatOutlines) -> None:
    """Test grammar for generating a valid arithmetic expression"""
    chat_model.grammar = """
        ?start: expression
        ?expression: term (("+" | "-") term)*
        ?term: factor (("*" | "/") factor)*
        ?factor: NUMBER | "-" factor | "(" expression ")"
        %import common.NUMBER
        %import common.WS
        %ignore WS
    """

    messages = [HumanMessage(content="Here is a complex arithmetic expression: ")]
    output = chat_model.invoke(messages)

    # Validate the output is a non-empty string
    assert (
        isinstance(output.content, str) and output.content.strip()
    ), "Output should be a non-empty string"

    # Use a simple regex to check if the output contains basic arithmetic operations and numbers
    assert re.search(
        r"[\d\+\-\*/\(\)]+", output.content
    ), f"Generated output '{output.content}' does not appear to be a valid arithmetic expression"


def test_chat_outlines_with_structured_output(chat_model) -> None:
    """Test that ChatOutlines can generate structured outputs"""
    pass  # TODO: Implement this test
