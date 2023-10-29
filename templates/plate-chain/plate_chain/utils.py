import json

from pydantic import BaseModel, Field, conint


class LLMPlateResponse(BaseModel):
    row_start: conint(ge=0) = Field(
        ..., description="The starting row of the plate (0-indexed)"
    )
    row_end: conint(ge=0) = Field(
        ..., description="The ending row of the plate (0-indexed)"
    )
    col_start: conint(ge=0) = Field(
        ..., description="The starting column of the plate (0-indexed)"
    )
    col_end: conint(ge=0) = Field(
        ..., description="The ending column of the plate (0-indexed)"
    )
    contents: str


def parse_llm_output(result: str):
    """
    Based on the prompt we expect the result to be a string that looks like:

    '[{"row_start": 12, "row_end": 19, "col_start": 1, \
    "col_end": 12, "contents": "Entity ID"}]'

    We'll load that JSON and turn it into a Pydantic model
    """
    return [LLMPlateResponse(**plate_r) for plate_r in json.loads(result)]
