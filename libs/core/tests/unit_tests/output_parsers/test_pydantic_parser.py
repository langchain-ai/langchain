from typing import Literal, Type, Union

import pytest

from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.pydantic_v2 import BaseModel as V2BaseModel
from tests.unit_tests.fake.chat_model import ParrotFakeChatModel


class ForecastV2(V2BaseModel):
    temperature: int
    f_or_c: Literal["F", "C"]
    forecast: str


class Forecast(BaseModel):
    temperature: int
    f_or_c: Literal["F", "C"]
    forecast: str


TBaseModel = Union[Type[BaseModel], Type[V2BaseModel]]


@pytest.mark.parametrize("pydantic_object", [ForecastV2, Forecast])
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
    assert type(res) == pydantic_object
    assert res.f_or_c == "C"
    assert res.temperature == 20
    assert res.forecast == "Sunny"


@pytest.mark.parametrize("pydantic_object", [ForecastV2, Forecast])
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
