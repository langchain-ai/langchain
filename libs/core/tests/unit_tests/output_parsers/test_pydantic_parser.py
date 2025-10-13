"""Test PydanticOutputParser."""

import sys
from enum import Enum
from typing import Literal

import pydantic
import pytest
from pydantic import BaseModel, Field
from pydantic.v1 import BaseModel as V1BaseModel

from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import ParrotFakeChatModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.output_parsers.json import JsonOutputParser
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.utils.pydantic import PydanticBaseModel, TBaseModel


class ForecastV2(pydantic.BaseModel):
    temperature: int
    f_or_c: Literal["F", "C"]
    forecast: str


_FORECAST_MODELS: list[type[BaseModel | V1BaseModel]] = [ForecastV2]
_FORECAST_MODELS_TYPES = type[ForecastV2]

if sys.version_info < (3, 14):

    class ForecastV1(V1BaseModel):
        temperature: int
        f_or_c: Literal["F", "C"]
        forecast: str

    _FORECAST_MODELS_TYPES |= type[ForecastV1]
    _FORECAST_MODELS.append(ForecastV1)


@pytest.mark.parametrize("pydantic_object", _FORECAST_MODELS)
def test_pydantic_parser_chaining(
    pydantic_object: _FORECAST_MODELS_TYPES,
) -> None:
    prompt = PromptTemplate(
        template="""{{
        "temperature": 20,
        "f_or_c": "C",
        "forecast": "Sunny"
    }}""",
        input_variables=[],
    )

    model = ParrotFakeChatModel()

    parser = PydanticOutputParser[PydanticBaseModel](pydantic_object=pydantic_object)
    chain = prompt | model | parser

    res = chain.invoke({})
    assert isinstance(res, pydantic_object)
    assert res.f_or_c == "C"
    assert res.temperature == 20
    assert res.forecast == "Sunny"


@pytest.mark.parametrize("pydantic_object", _FORECAST_MODELS)
def test_pydantic_parser_validation(pydantic_object: TBaseModel) -> None:
    bad_prompt = PromptTemplate(
        template="""{{
        "temperature": "oof",
        "f_or_c": 1,
        "forecast": "Sunny"
    }}""",
        input_variables=[],
    )

    model = ParrotFakeChatModel()

    parser: PydanticOutputParser[PydanticBaseModel] = PydanticOutputParser(
        pydantic_object=pydantic_object
    )
    chain = bad_prompt | model | parser
    with pytest.raises(OutputParserException):
        chain.invoke({})


# JSON output parser tests
@pytest.mark.parametrize("pydantic_object", _FORECAST_MODELS)
def test_json_parser_chaining(
    pydantic_object: TBaseModel,
) -> None:
    prompt = PromptTemplate(
        template="""{{
        "temperature": 20,
        "f_or_c": "C",
        "forecast": "Sunny"
    }}""",
        input_variables=[],
    )

    model = ParrotFakeChatModel()

    parser = JsonOutputParser(pydantic_object=pydantic_object)
    chain = prompt | model | parser

    res = chain.invoke({})
    assert res["f_or_c"] == "C"
    assert res["temperature"] == 20
    assert res["forecast"] == "Sunny"


class Actions(Enum):
    SEARCH = "Search"
    CREATE = "Create"
    UPDATE = "Update"
    DELETE = "Delete"


class TestModel(BaseModel):
    action: Actions = Field(description="Action to be performed")
    action_input: str = Field(description="Input to be used in the action")
    additional_fields: str | None = Field(description="Additional fields", default=None)
    for_new_lines: str = Field(description="To be used to test newlines")


# Prevent pytest from trying to run tests on TestModel
TestModel.__test__ = False  # type: ignore[attr-defined]


DEF_RESULT = """{
    "action": "Update",
    "action_input": "The PydanticOutputParser class is powerful",
    "additional_fields": null,
    "for_new_lines": "not_escape_newline:\n escape_newline: \\n"
}"""

# action 'update' with a lowercase 'u' to test schema validation failure.
DEF_RESULT_FAIL = """{
    "action": "update",
    "action_input": "The PydanticOutputParser class is powerful",
    "additional_fields": null
}"""

DEF_EXPECTED_RESULT = TestModel(
    action=Actions.UPDATE,
    action_input="The PydanticOutputParser class is powerful",
    additional_fields=None,
    for_new_lines="not_escape_newline:\n escape_newline: \n",
)


def test_pydantic_output_parser() -> None:
    """Test PydanticOutputParser."""
    pydantic_parser: PydanticOutputParser = PydanticOutputParser(
        pydantic_object=TestModel
    )

    result = pydantic_parser.parse(DEF_RESULT)
    assert result == DEF_EXPECTED_RESULT
    assert pydantic_parser.OutputType is TestModel


def test_pydantic_output_parser_fail() -> None:
    """Test PydanticOutputParser where completion result fails schema validation."""
    pydantic_parser: PydanticOutputParser = PydanticOutputParser(
        pydantic_object=TestModel
    )

    with pytest.raises(
        OutputParserException, match="Failed to parse TestModel from completion"
    ):
        pydantic_parser.parse(DEF_RESULT_FAIL)


def test_pydantic_output_parser_type_inference() -> None:
    """Test pydantic output parser type inference."""

    class SampleModel(BaseModel):
        foo: int
        bar: str

    # Ignoring mypy error that appears in python 3.8, but not 3.11.
    # This seems to be functionally correct, so we'll ignore the error.
    pydantic_parser = PydanticOutputParser[SampleModel](pydantic_object=SampleModel)
    schema = pydantic_parser.get_output_schema().model_json_schema()

    assert schema == {
        "properties": {
            "bar": {"title": "Bar", "type": "string"},
            "foo": {"title": "Foo", "type": "integer"},
        },
        "required": ["foo", "bar"],
        "title": "SampleModel",
        "type": "object",
    }


def test_format_instructions_preserves_language() -> None:
    """Test format instructions does not attempt to encode into ascii."""
    description = (
        "你好, こんにちは, नमस्ते, Bonjour, Hola, "
        "Olá, 안녕하세요, Jambo, Merhaba, Γειά σου"  # noqa: RUF001
    )

    class Foo(BaseModel):
        hello: str = Field(
            description=(
                "你好, こんにちは, नमस्ते, Bonjour, Hola, "
                "Olá, 안녕하세요, Jambo, Merhaba, Γειά σου"  # noqa: RUF001
            )
        )

    parser = PydanticOutputParser[Foo](pydantic_object=Foo)
    assert description in parser.get_format_instructions()
