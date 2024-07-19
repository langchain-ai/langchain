from typing import Literal

import pydantic  # pydantic: ignore
import pytest

from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import ParrotFakeChatModel
from langchain_core.output_parsers.json import JsonOutputParser
from langchain_core.output_parsers.pydantic import PydanticOutputParser, TBaseModel
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.utils.pydantic import PYDANTIC_MAJOR_VERSION

V1BaseModel = pydantic.BaseModel
if PYDANTIC_MAJOR_VERSION == 2:
    from pydantic.v1 import BaseModel  # pydantic: ignore

    V1BaseModel = BaseModel  # type: ignore


class ForecastV2(pydantic.BaseModel):
    temperature: int
    f_or_c: Literal["F", "C"]
    forecast: str


class ForecastV1(V1BaseModel):
    temperature: int
    f_or_c: Literal["F", "C"]
    forecast: str


@pytest.mark.parametrize("pydantic_object", [ForecastV2, ForecastV1])
def test_pydantic_parser_chaining(
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

    parser = PydanticOutputParser(pydantic_object=pydantic_object)  # type: ignore
    chain = prompt | model | parser

    res = chain.invoke({})
    assert type(res) is pydantic_object
    assert res.f_or_c == "C"
    assert res.temperature == 20
    assert res.forecast == "Sunny"


@pytest.mark.parametrize("pydantic_object", [ForecastV2, ForecastV1])
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

    parser = PydanticOutputParser(pydantic_object=pydantic_object)  # type: ignore
    chain = bad_prompt | model | parser
    with pytest.raises(OutputParserException):
        chain.invoke({})


# JSON output parser tests
@pytest.mark.parametrize("pydantic_object", [ForecastV2, ForecastV1])
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

    parser = JsonOutputParser(pydantic_object=pydantic_object)  # type: ignore
    chain = prompt | model | parser

    res = chain.invoke({})
    assert res["f_or_c"] == "C"
    assert res["temperature"] == 20
    assert res["forecast"] == "Sunny"
