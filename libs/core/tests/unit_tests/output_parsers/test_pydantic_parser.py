from typing import Literal

from langchain_core.output_parsers.pydantic_v2 import PydanticOutputParser
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.pydantic_v2 import BaseModel, Field
from tests.unit_tests.fake.chat_model import ParrotFakeChatModel


class Forecast(BaseModel):
    temperature: int = Field(
        description="Temperature in degrees. Can be Fareinheit or Celsius."
    )
    f_or_c: Literal["F", "C"] = Field(description="F for Fareinheit, C for Celsius")
    forecast: str = Field(
        description="The forecast for the day.", examples=["Sunny", "Rainy", "Cloudy"]
    )


def test_pydantic_v2_parser_chaining():
    prompt = PromptTemplate(
        template="""{{
        "temperature": 20,
        "f_or_c": "C",
        "forecast": "Sunny"
    }}""",
        input_variables=[],
    )

    model = ParrotFakeChatModel()

    parser = PydanticOutputParser(pydantic_v2_object=Forecast)
    chain = prompt | model | parser

    res = chain.invoke({})
    assert type(res) == Forecast
    assert res.f_or_c == "C"
    assert res.temperature == 20
    assert res.forecast == "Sunny"
