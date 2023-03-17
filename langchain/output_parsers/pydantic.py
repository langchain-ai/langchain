import json
import re

from pydantic import BaseModel
from langchain.output_parsers.base import BaseOutputParser
from langchain.output_parsers.format_instructions import PYDANTIC_FORMAT_INSTRUCTIONS


class PydanticOutputParser(BaseOutputParser):

    pydantic_object: BaseModel

    def parse(self, text: str) -> BaseModel:
        # Greedy search for 1st json candidate.
        json_str = re.search('\{.*\}', text.strip())
        json_object = json.loads(json_str)
        return self.pydantic_object.parse_obj(json_object)

    def get_format_instructions(self) -> str:
        schema = self.pydantic_object.schema()

        # Remove extraneous fields.
        reduced_schema = {
            prop: {
                'description': data['description'],
                'type': data['type']
            }
            for prop, data in schema['properties'].items()
        }
        # Ensure json in context is well-formed with double quotes.
        schema = json.dumps(reduced_schema)

        return PYDANTIC_FORMAT_INSTRUCTIONS.format(schema=schema)
