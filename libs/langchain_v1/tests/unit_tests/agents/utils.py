import json
from pathlib import Path
from typing import Type

from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel


class BaseSchema(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        from_attributes=True,
    )


def load_spec(spec_name: str, as_model: Type[BaseModel]) -> list[BaseModel]:
    with (Path(__file__).parent / "specifications" / f"{spec_name}.json").open(
        "r", encoding="utf-8"
    ) as f:
        data = json.load(f)
        return [as_model(**item) for item in data]
