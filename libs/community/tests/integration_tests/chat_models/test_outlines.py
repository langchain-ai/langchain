# flake8: noqa
"""Test ChatOutlines wrapper."""

from typing import Generator
import re
import platform
import pytest

from langchain_community.chat_models.outlines import ChatOutlines
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from langchain_core.messages import BaseMessageChunk
from pydantic import BaseModel

from tests.unit_tests.callbacks.fake_callback_handler import FakeCallbackHandler


MODEL = "microsoft/Phi-3-mini-4k-instruct"
LLAMACPP_MODEL = "bartowski/qwen2.5-7b-ins-v3-GGUF/qwen2.5-7b-ins-v3-Q4_K_M.gguf"

BACKENDS = ["transformers", "llamacpp"]
if platform.system() != "Darwin":
    BACKENDS.append("vllm")
if platform.system() == "Darwin":
    BACKENDS.append("mlxlm")


@pytest.fixture(params=BACKENDS)
def chat_model(request: pytest.FixtureRequest) -> ChatOutlines:
    if request.param == "llamacpp":
        return ChatOutlines(model=LLAMACPP_MODEL, backend=request.param)
    else:
        return ChatOutlines(model=MODEL, backend=request.param)


def test_chat_outlines_inference(chat_model: ChatOutlines) -> None:
    """Test valid ChatOutlines inference."""
    messages = [HumanMessage(content="Say foo:")]
    output = chat_model.invoke(messages)
    assert isinstance(output, AIMessage)
    assert len(output.content) > 1


def test_chat_outlines_streaming(chat_model: ChatOutlines) -> None:
    """Test streaming tokens from ChatOutlines."""
    messages = [HumanMessage(content="How do you say 'hello' in Spanish?")]
    generator = chat_model.stream(messages)
    stream_results_string = ""
    assert isinstance(generator, Generator)

    for chunk in generator:
        assert isinstance(chunk, BaseMessageChunk)
        if isinstance(chunk.content, str):
            stream_results_string += chunk.content
        else:
            raise ValueError(
                f"Invalid content type, only str is supported, "
                f"got {type(chunk.content)}"
            )
    assert len(stream_results_string.strip()) > 1


def test_chat_outlines_streaming_callback(chat_model: ChatOutlines) -> None:
    """Test that streaming correctly invokes on_llm_new_token callback."""
    MIN_CHUNKS = 5
    callback_handler = FakeCallbackHandler()
    chat_model.callbacks = [callback_handler]
    chat_model.verbose = True
    messages = [HumanMessage(content="Can you count to 10?")]
    chat_model.invoke(messages)
    assert callback_handler.llm_streams >= MIN_CHUNKS


def test_chat_outlines_regex(chat_model: ChatOutlines) -> None:
    """Test regex for generating a valid IP address"""
    ip_regex = r"((25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(25[0-5]|2[0-4]\d|[01]?\d\d?)"
    chat_model.regex = ip_regex
    assert chat_model.regex == ip_regex

    messages = [HumanMessage(content="What is the IP address of Google's DNS server?")]
    output = chat_model.invoke(messages)

    assert isinstance(output, AIMessage)
    assert re.match(
        ip_regex, str(output.content)
    ), f"Generated output '{output.content}' is not a valid IP address"


def test_chat_outlines_type_constraints(chat_model: ChatOutlines) -> None:
    """Test type constraints for generating an integer"""
    chat_model.type_constraints = int
    messages = [
        HumanMessage(
            content="What is the answer to life, the universe, and everything?"
        )
    ]
    output = chat_model.invoke(messages)
    assert isinstance(int(str(output.content)), int)


def test_chat_outlines_json(chat_model: ChatOutlines) -> None:
    """Test json for generating a valid JSON object"""

    class Person(BaseModel):
        name: str

    chat_model.json_schema = Person
    messages = [HumanMessage(content="Who are the main contributors to LangChain?")]
    output = chat_model.invoke(messages)
    person = Person.model_validate_json(str(output.content))
    assert isinstance(person, Person)


def test_chat_outlines_grammar(chat_model: ChatOutlines) -> None:
    """Test grammar for generating a valid arithmetic expression"""
    if chat_model.backend == "mlxlm":
        pytest.skip("MLX grammars not yet supported.")

    chat_model.grammar = """
        ?start: expression
        ?expression: term (("+" | "-") term)*
        ?term: factor (("*" | "/") factor)*
        ?factor: NUMBER | "-" factor | "(" expression ")"
        %import common.NUMBER
        %import common.WS
        %ignore WS
    """

    messages = [HumanMessage(content="Give me a complex arithmetic expression:")]
    output = chat_model.invoke(messages)

    # Validate the output is a non-empty string
    assert (
        isinstance(output.content, str) and output.content.strip()
    ), "Output should be a non-empty string"

    # Use a simple regex to check if the output contains basic arithmetic operations and numbers
    assert re.search(
        r"[\d\+\-\*/\(\)]+", output.content
    ), f"Generated output '{output.content}' does not appear to be a valid arithmetic expression"


def test_chat_outlines_with_structured_output(chat_model: ChatOutlines) -> None:
    """Test that ChatOutlines can generate structured outputs"""

    class AnswerWithJustification(BaseModel):
        """An answer to the user question along with justification for the answer."""

        answer: str
        justification: str

    structured_chat_model = chat_model.with_structured_output(AnswerWithJustification)

    result = structured_chat_model.invoke(
        "What weighs more, a pound of bricks or a pound of feathers?"
    )

    assert isinstance(result, AnswerWithJustification)
    assert isinstance(result.answer, str)
    assert isinstance(result.justification, str)
    assert len(result.answer) > 0
    assert len(result.justification) > 0

    structured_chat_model_with_raw = chat_model.with_structured_output(
        AnswerWithJustification, include_raw=True
    )

    result_with_raw = structured_chat_model_with_raw.invoke(
        "What weighs more, a pound of bricks or a pound of feathers?"
    )

    assert isinstance(result_with_raw, dict)
    assert "raw" in result_with_raw
    assert "parsed" in result_with_raw
    assert "parsing_error" in result_with_raw
    assert isinstance(result_with_raw["raw"], BaseMessage)
    assert isinstance(result_with_raw["parsed"], AnswerWithJustification)
    assert result_with_raw["parsing_error"] is None
