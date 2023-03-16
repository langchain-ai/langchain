from typing import Any, Type
import json

from pydantic import BaseModel
from langchain.output_parsers.base import BaseOutputParser
from langchain.output_parsers.format_instructions import PYDANTIC_FORMAT_INSTRUCTIONS


class PydanticOutputParser(BaseOutputParser):

    pydantic_object: Type[BaseModel]
    def parse(self, text: str) -> Any:
        json_object = json.loads(text.strip())
        return self.pydantic_object.parse_obj(json_object)

    def get_format_instructions(self) -> str:
        return PYDANTIC_FORMAT_INSTRUCTIONS.format(schema=str(self.pydantic_object.schema()))