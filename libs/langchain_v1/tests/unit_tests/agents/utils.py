import json
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel


class BaseSchema(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        from_attributes=True,
    )


_T = TypeVar("_T", bound=BaseModel)


def load_spec(spec_name: str, as_model: type[_T]) -> list[_T]:
    with (Path(__file__).parent / "specifications" / f"{spec_name}.json").open(
        "r", encoding="utf-8"
    ) as f:
        data = json.load(f)
        return [as_model(**item) for item in data]
