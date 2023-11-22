# flake8: noqa
"""Tools for interacting with a Cube Semantic Layer."""
import json
from datetime import date
from typing import Optional

from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.output_parsers import PydanticOutputParser
from langchain.pydantic_v1 import BaseModel, Field
from langchain.schema import OutputParserException
from langchain.tools.base import BaseTool
from langchain.utilities.cube import CubeAPIWrapper, Query

LOAD_CUBE_INPUT_FORMAT_INSTRUCTIONS = """The input should be formatted as a JSON instance that conforms to the JSON schema below.
As an example, for the schema {{"properties": {{"foo": {{"title": "Foo", "description": "a list of strings", "type": "array", "items": {{"type": "string"}}}}}}, "required": ["foo"]}}
the object {{"foo": ["bar", "baz"]}} is a well-formatted instance of the schema. The object {{"properties": {{"foo": ["bar", "baz"]}}}} is not well-formatted.
Here is the input schema:
```
{schema}
```"""


class BaseCubeTool(BaseModel):
    """Base tool for interacting with a Cube Semantic Layer."""

    cube: CubeAPIWrapper = Field(exclude=True)

    class Config(BaseTool.Config):
        pass


def _get_format_instructions() -> str:
    schema = Query.schema()

    # Remove extraneous fields.
    reduced_schema = schema
    if "title" in reduced_schema:
        del reduced_schema["title"]
    if "type" in reduced_schema:
        del reduced_schema["type"]
    # Ensure json in context is well-formed with double quotes.
    schema_str = json.dumps(reduced_schema)

    format_instructions = LOAD_CUBE_INPUT_FORMAT_INSTRUCTIONS.format(schema=schema_str)

    # Escape curly braces.
    return format_instructions.replace("{", "{{").replace("}", "}}")


class LoadCubeTool(BaseCubeTool, BaseTool):
    """Tool for getting the data for a query."""

    name: str = "load_cube"
    description: str = (
        "Input to this tool is a detailed and correct Cube query, it format is JSON. "
        "Output is a result from the Cube, it format is JSON."
        f"This current date is {date.today().isoformat()}."
        "If the query is not correct, an error message will be returned."
        "If an error is returned, rewrite the query, check the query, and try again.\n"
        f"{_get_format_instructions()}"
    )

    def _run(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Get the data for a query."""

        try:
            parser: PydanticOutputParser = PydanticOutputParser(pydantic_object=Query)
            data = self.cube.load(parser.parse(query))
            return json.dumps(data["data"])
        except (OutputParserException, ValueError) as e:
            return f"Error: {e}"


class MetaInformationCubeTool(BaseCubeTool, BaseTool):
    """Tool for getting meta-information about a Cube Semantic Layer."""

    name: str = "meta_information_cube"
    description: str = (
        "Input to this tool is a comma-separated list of models, "
        "output is a Markdown table of the meta-information for those models."
        "Be sure that the models actually exist by calling list_models_cube first!"
        'Example Input: "model1, model2, model3"'
    )

    def _run(
        self,
        model_names: str,
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Get the meta-information of the Cube Semantic Layer."""
        return self.cube.get_model_meta_information(model_names.split(", "))


class ListCubeTool(BaseCubeTool, BaseTool):
    """Tool for getting models names and descriptions."""

    name: str = "list_models_cube"
    description: str = (
        "Input is an empty string, output is a Markdown table of models in the Cube."
    )

    def _run(
        self,
        tool_input: str = "",
        *,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Get the names, descriptions of the models."""
        return self.cube.get_usable_models()
