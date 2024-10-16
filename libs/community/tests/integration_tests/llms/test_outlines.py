# flake8: noqa
"""Test Outlines wrapper."""

from typing import Generator
import re
import platform
import pytest

from langchain_community.llms.outlines import Outlines
from langchain_core.outputs import GenerationChunk
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
def llm(request):
    if request.param == "llamacpp":
        return Outlines(model=LLAMACPP_MODEL, backend=request.param, max_tokens=10)
    else:
        return Outlines(model=MODEL, backend=request.param, max_tokens=10)


def test_outlines_inference(llm: Outlines) -> None:
    """Test valid outlines inference."""
    output = llm.invoke("Say foo:")
    assert isinstance(output, str)
    assert len(output) > 1


def test_outlines_streaming(llm: Outlines) -> None:
    """Test streaming tokens from Outlines."""
    generator = llm.stream("Q: How do you say 'hello' in Spanish? A:'", stop=["'"])
    stream_results_string = ""
    assert isinstance(generator, Generator)

    for chunk in generator:
        print(chunk)
        assert isinstance(chunk, GenerationChunk)
        stream_results_string += chunk.text
    print(stream_results_string)
    assert len(stream_results_string.strip()) > 1


def test_outlines_streaming_callback(llm: Outlines) -> None:
    """Test that streaming correctly invokes on_llm_new_token callback."""
    MAX_TOKENS = 5
    OFF_BY_ONE = 1  # There may be an off by one error in the upstream code!

    callback_handler = FakeCallbackHandler()
    llm.callbacks = [callback_handler]
    llm.verbose = True
    llm.invoke("Q: Can you count to 10? A:'1, ")
    assert callback_handler.llm_streams <= MAX_TOKENS + OFF_BY_ONE


def test_outlines_model_kwargs(llm: Outlines) -> None:
    llm.model_kwargs = {"n_gqa": None}
    assert llm.model_kwargs == {"n_gqa": None}


def test_outlines_regex(llm: Outlines) -> None:
    """Test regex for generating a valid IP address"""
    ip_regex = r"((25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(25[0-5]|2[0-4]\d|[01]?\d\d?)"
    llm.regex = ip_regex
    assert llm.regex == ip_regex

    output = llm.invoke("Q: What is the IP address of googles dns server? A: ")

    assert isinstance(output, str)

    assert re.match(
        ip_regex, output
    ), f"Generated output '{output}' is not a valid IP address"


def test_outlines_type_constraints(llm: Outlines) -> None:
    """Test type constraints for generating an integer"""
    llm.type_constraints = int
    output = llm.invoke(
        "Q: What is the answer to life, the universe, and everything? A: "
    )
    assert isinstance(output, int)


def test_outlines_json(llm: Outlines) -> None:
    """Test json for generating a valid JSON object"""

    class Person(BaseModel):
        name: str

    llm.json_schema = Person
    output = llm.invoke("Q: Who is the author of LangChain?  A: ")
    person = Person.model_validate_json(output)
    assert isinstance(person, Person)


def test_outlines_json_schema(llm: Outlines) -> None:
    """Test json schema for generating a valid JSON object"""

    class Food(BaseModel):
        ingredients: list[str]
        calories: int

    llm.json_schema = Food.model_json_schema()
    output = llm.invoke("Q: What is the nutritional information for a Big Mac? A: ")
    food = Food.model_validate_json(output)
    assert isinstance(food, Food)


def test_outlines_grammar(llm: Outlines) -> None:
    """Test grammar for generating a valid arithmetic expression"""
    llm.grammar = """
        ?start: expression
        ?expression: term (("+" | "-") term)*
        ?term: factor (("*" | "/") factor)*
        ?factor: NUMBER | "-" factor | "(" expression ")"
        %import common.NUMBER
        %import common.WS
        %ignore WS
    """

    output = llm.invoke("Here is a complex arithmetic expression: ")

    # Validate the output is a non-empty string
    assert (
        isinstance(output, str) and output.strip()
    ), "Output should be a non-empty string"

    # Use a simple regex to check if the output contains basic arithmetic operations and numbers
    assert re.search(
        r"[\d\+\-\*/\(\)]+", output
    ), f"Generated output '{output}' does not appear to be a valid arithmetic expression"


def test_outlines_with_structured_output(llm) -> None:
    """Test that outlines can generate structured outputs"""
    pass  # TODO: Implement this test
